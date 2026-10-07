"""
Detection systems compared in the adaptive-PUEA experiments.

Every system sees, per slot, only fusion-center-observable information:
the per-SU RSS / noise-estimate vector, the static SU positions and the
known PU location. Labels, attacker parameters, channel ground truth and
oracle observations are never passed in (see ``run_stream``).

Per-slot evidence
-----------------
* ``xgb_flag``   : XGBoost (trained on the simulated training stream) says PUEA.
* ``prof_flag``  : feature vector is outside the (static or adaptive) PU profile.
* ``phys_ok``    : spatial RSS pattern is consistent with the PU location.

Decision rules (flag = "PUEA", i.e. SUs do NOT vacate the channel)
------------------------------------------------------------------
A  Baseline-XGBoost               flag = xgb
   XGB+StaticProfile              flag = xgb AND prof_0
B  Naive-Adaptive                 flag = xgb AND prof_t          update if not flag
   Anchored-Update-Only           flag = xgb AND prof_t          update if not flag AND phys_ok
C  XGB+Physical                   flag = (xgb AND prof_0) OR NOT phys_ok
D  Physically-Anchored-Adaptive   flag = (xgb AND prof_t) OR NOT phys_ok   update if not flag
   Physical-Only                  flag = NOT phys_ok
   Spatial-XGBoost                flag = XGBoost on the raw 40-SU RSS vector

Sequential extension (src/sequential_spatial.py; ``seq_alarm`` = the
drift-invariant GLR-CUSUM on the location score is above its threshold):

E  Sequential-Anchored            as D; no profile update while seq_alarm
F  Sequential-Anchored+Rollback   as E; when seq_alarm switches on, the profile
                                  is restored to its last checkpoint before the
                                  estimated change point
G  Sequential-Anchored-Full       as F, but during an alarm the profile keeps
                                  adapting with updates weighted by each slot's
                                  posterior of coming from the PU (two-source
                                  mixture; frozen until the first fit), and a
                                  slot is also flagged if its LLR (fitted
                                  attacker location vs PU) exceeds the
                                  Neyman-Pearson threshold for the per-slot
                                  false-alarm budget (attr_flag)
   Physical-Only+Sequential       flag = NOT phys_ok OR (seq_alarm AND attr_flag)

"xgb AND prof" means an alarm needs both the static classifier and the PU
profile to object; this is what lets an *adaptive* profile suppress false
alarms caused by legitimate drift the static classifier never saw, and it
is also exactly what a poisoner exploits. "PU-like" := NOT (xgb AND prof).

"Anchored-Update-Only" isolates the anchor's role as an *update gate*: its
decisions ignore physics, so any difference from Naive-Adaptive comes only
from what was allowed into the profile.

An unverifiable slot (too few SUs above ``min_snr_db``) never raises a
physical alarm and never passes the physical gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_profile import AdaptivePUProfile, update_allowed
from fc_features import fc_features
from sequential_spatial import (
    SequentialSpatialMonitor,
    candidate_grid,
    fit_mixture,
    np_llr_threshold,
    slot_posterior,
    slot_scores,
)

LARGE_NEG = -1e6


@dataclass(frozen=True)
class SystemSpec:
    name: str
    ablation: str  # A/B/C/D or "ref"
    detector: str = "xgb"  # "xgb" | "spatial_xgb" | "none"
    uses_profile: bool = False
    adapt: bool = False
    physical_in_decision: bool = False
    require_physical_for_update: bool = False
    sequential: bool = False  # run the sequential location monitor; freeze updates during an alarm
    rollback: bool = False  # restore the pre-change-point profile when an alarm starts
    attribution: bool = False  # per-slot mixture attribution during an alarm (in the decision)
    description: str = ""


SYSTEMS = [
    SystemSpec("Baseline-XGBoost", "A", description="Static XGBoost on FC features."),
    SystemSpec("Naive-Adaptive", "B", uses_profile=True, adapt=True,
               description="XGBoost + adaptive profile; every PU-like slot updates the profile."),
    SystemSpec("XGB+Physical", "C", uses_profile=True, physical_in_decision=True,
               description="XGBoost + frozen profile + physical check in the decision; no adaptation."),
    SystemSpec("Physically-Anchored-Adaptive", "D", uses_profile=True, adapt=True,
               physical_in_decision=True, require_physical_for_update=True,
               description="Proposed: PU-like AND physically consistent -> update; inconsistent -> PUEA."),
    SystemSpec("XGB+StaticProfile", "ref", uses_profile=True,
               description="XGBoost + frozen profile (isolates the effect of adaptation in B)."),
    SystemSpec("Anchored-Update-Only", "ref", uses_profile=True, adapt=True,
               require_physical_for_update=True,
               description="Like B but updates gated by physics; physics NOT used for decisions."),
    SystemSpec("Physical-Only", "ref", detector="none", physical_in_decision=True,
               description="Only the physical check (tests whether physics does all the work)."),
    SystemSpec("Spatial-XGBoost", "ref", detector="spatial_xgb",
               description="XGBoost on the raw per-SU RSS vector (can a classifier learn location itself?)."),
    SystemSpec("Sequential-Anchored", "E", uses_profile=True, adapt=True, physical_in_decision=True,
               require_physical_for_update=True, sequential=True,
               description="D + sequential drift-invariant location test; profile frozen while it alarms."),
    SystemSpec("Sequential-Anchored+Rollback", "F", uses_profile=True, adapt=True, physical_in_decision=True,
               require_physical_for_update=True, sequential=True, rollback=True,
               description="E + profile restored to its checkpoint before the estimated change point."),
    SystemSpec("Sequential-Anchored-Full", "G", uses_profile=True, adapt=True, physical_in_decision=True,
               require_physical_for_update=True, sequential=True, rollback=True, attribution=True,
               description="Proposed: F + per-slot attribution to the fitted attacker location during an alarm."),
    SystemSpec("Physical-Only+Sequential", "ref", detector="none", physical_in_decision=True,
               sequential=True, attribution=True,
               description="Physical check + sequential test + attribution, no classifier/profile "
                           "(does the adaptive profile add anything?)."),
]
SYSTEMS_BY_NAME = {s.name: s for s in SYSTEMS}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SlotEvidence:
    """Detector-independent evidence for a batch of slots (computed once per stream)."""

    def __init__(self, obs: dict, models: dict):
        rss, noise = obs["rss_dbm"], obs["noise_est_dbm"]
        rss2 = np.atleast_2d(rss)
        self.z = np.atleast_2d(fc_features(rss, noise))
        self.p_xgb = models["xgb"].predict_puea(self.z)
        self.p_spatial = models["spatial_xgb"].predict_puea(rss2)
        self.phys = models["physical"].compute(rss, noise)
        self.seq = None
        if models.get("seq_cal") is not None:
            pm = models["phys_model"]
            u0, J0, s2, s, valid = slot_scores(rss2, np.atleast_2d(noise), models["su_xy"], pm.pu_xy, pm.alpha,
                                               pm.min_snr_db, pm.min_distance, pm.area_size, pm.grid_step)
            self.seq = {"u0": u0, "J0": J0, "sigma2": s2, "s": s, "valid": valid}

    def __len__(self):
        return self.z.shape[0]


class DetectionSystem:
    """Stateful per-slot detector for one :class:`SystemSpec`."""

    def __init__(self, spec: SystemSpec, profile: AdaptivePUProfile | None, xgb_threshold: float,
                 phys_threshold: float, seq_context: dict | None = None):
        self.spec = spec
        self.profile = profile.copy() if (profile is not None and spec.uses_profile) else None
        self.thr = xgb_threshold
        self.phys_thr = phys_threshold
        self.monitor = None
        if spec.sequential:
            if seq_context is None:
                raise ValueError(f"{spec.name} needs the sequential calibration")
            self.ctx = seq_context
            self.monitor = SequentialSpatialMonitor(seq_context["cal"], len(seq_context["su_xy"]))
            self.snapshots = []  # (slot, profile copy)
            self.prev_alarm = False
            self.fit = None
            self.fit_slot = -10 ** 9

    # ---------------------------------------------------------- sequential
    def _snapshot(self, t):
        sc = self.ctx["cfg"]
        if self.profile is None or t % int(sc["snapshot_every"]):
            return
        self.snapshots.append((t, self.profile.copy()))
        horizon = self.monitor.cal.window + int(sc["snapshot_every"])
        while self.snapshots and self.snapshots[0][0] < t - 2 * horizon:
            self.snapshots.pop(0)

    def _rollback(self, change_slot) -> int:
        """Restore the latest checkpoint taken strictly before the change point. Returns its slot or -1."""
        best = None
        for slot, prof in self.snapshots:
            if slot < change_slot:
                best = (slot, prof)
        if best is None:
            return -1
        self.profile = best[1].copy()
        return int(best[0])

    def _sequential(self, ev: SlotEvidence, i: int, t: int, verifiable: bool) -> dict:
        q = ev.seq
        out = self.monitor.step(t, q["u0"][i], q["J0"][i], float(q["sigma2"][i]), q["s"][i], q["valid"][i],
                                verifiable)
        out.update({"rollback_to_slot": -1, "attr_posterior": np.nan, "attr_x": np.nan, "attr_y": np.nan,
                    "attr_rho": np.nan, "attr_flag": False})
        alarm = out["seq_alarm"]
        if out["seq_episode_start"] and self.spec.rollback:
            out["rollback_to_slot"] = self._rollback(out["seq_change_slot"])
        if not alarm:
            self.fit = None
        if alarm and self.spec.attribution and verifiable:
            sc = self.ctx["cfg"]
            if self.fit is None or t - self.fit_slot >= int(sc["refit_every"]):
                s_w, v_w = self.monitor.window_since(out["seq_change_slot"])
                if len(s_w) >= int(sc["min_slots_for_fit"]):
                    pm = self.ctx["phys_model"]
                    self.fit = fit_mixture(s_w, v_w, self.monitor.sigma2, pm.pu_xy, self.ctx["su_xy"], pm.alpha,
                                           pm.min_distance, self.ctx["candidates"])
                    self.fit_slot = t
            if self.fit is not None:
                pm = self.ctx["phys_model"]
                post, llr = slot_posterior(q["s"][i], q["valid"][i], self.monitor.sigma2, self.fit,
                                           pm.pu_xy, self.ctx["su_xy"], pm.alpha, pm.min_distance)
                out["attr_posterior"] = post
                out["attr_flag"] = llr > np_llr_threshold(self.fit["lam_a"], self.ctx["attr_pfa"])
                out["attr_x"], out["attr_y"] = float(self.fit["x_a"][0]), float(self.fit["x_a"][1])
                out["attr_rho"] = self.fit["rho"]
        self.prev_alarm = alarm
        return out

    def step(self, ev: SlotEvidence, i: int, t: int | None = None) -> dict:
        s = self.spec
        t = i if t is None else t
        z = ev.z[i]
        p_xgb = float(ev.p_xgb[i])
        verifiable = bool(ev.phys["verifiable"][i])
        phys_ok = bool(ev.phys["physically_consistent"][i])
        resid = float(ev.phys["physical_residual"][i])

        s_xgb = float(_logit(p_xgb) - _logit(self.thr))
        xgb_flag = p_xgb >= self.thr
        if s.detector == "spatial_xgb":
            p_sp = float(ev.p_spatial[i])
            classifier_flag = p_sp >= self.thr
            cls_score = float(_logit(p_sp) - _logit(self.thr))
        elif s.detector == "none":
            classifier_flag, cls_score = False, LARGE_NEG
        else:
            classifier_flag, cls_score = xgb_flag, s_xgb

        d2 = prof_flag = None
        if self.profile is not None:
            d2 = float(self.profile.score(z))
            prof_flag = d2 > self.profile.threshold
            classifier_flag = classifier_flag and prof_flag
            cls_score = min(cls_score, float(np.log(max(d2, 1e-12) / self.profile.threshold)))

        seq = None
        if self.monitor is not None:
            seq = self._sequential(ev, i, t, verifiable)
        attr_flag = bool(seq is not None and seq["seq_alarm"] and seq["attr_flag"])

        pu_like = not classifier_flag
        phys_flag = s.physical_in_decision and verifiable and not phys_ok
        flag = classifier_flag or phys_flag or attr_flag
        score = cls_score
        if s.physical_in_decision:
            s_phys = float(np.log((resid + 1.0) / (self.phys_thr + 1.0))) if verifiable else LARGE_NEG
            score = max(score, s_phys)
        if seq is not None and s.attribution:
            p_att = seq["attr_posterior"] if (seq["seq_alarm"] and np.isfinite(seq["attr_posterior"])) else 0.0
            score = max(score, float(_logit(p_att)) if p_att > 0 else LARGE_NEG)  # ranking only (ROC)

        accepted, weight = False, 0.0
        if s.adapt and self.profile is not None:
            weight = 1.0
            if seq is not None and seq["seq_alarm"]:
                # E/F freeze during an alarm; G soft-updates with the PU posterior once a fit exists.
                post = seq["attr_posterior"]
                weight = (1.0 - post) if (s.attribution and np.isfinite(post)) else 0.0
            accepted = weight > 0 and update_allowed(pu_like, phys_ok if verifiable else None,
                                                     s.require_physical_for_update) and not attr_flag
            if accepted:
                self.profile.update(z, tracked={
                    "pu_power_hat_dbm": float(ev.phys["pu_power_hat_dbm"][i]),
                    "physical_residual": resid,
                }, weight=weight)
            else:
                weight = 0.0

        out = {
            "p_puea_xgb": p_xgb,
            "classifier_confidence_pu": 1.0 - p_xgb,
            "xgb_flag": bool(xgb_flag),
            "profile_d2": d2 if d2 is not None else np.nan,
            "profile_threshold": self.profile.threshold if self.profile is not None else np.nan,
            "profile_flag": bool(prof_flag) if prof_flag is not None else False,
            "pu_like": pu_like,
            "physical_residual": resid,
            "physical_consistency_score": float(ev.phys["physical_consistency_score"][i]),
            "physically_consistent": phys_ok,
            "verifiable": verifiable,
            "pu_power_hat_dbm": float(ev.phys["pu_power_hat_dbm"][i]),
            "source_x_hat": float(ev.phys["source_x_hat"][i]),
            "source_y_hat": float(ev.phys["source_y_hat"][i]),
            "attack_detected": bool(flag),
            "score": float(score),
            "accepted_for_update": bool(accepted),
            "update_weight": float(weight),
            "rejected_for_update": bool(s.adapt and pu_like and not accepted),
            "seq_stat": seq["seq_stat"] if seq else np.nan,
            "seq_alarm": bool(seq["seq_alarm"]) if seq else False,
            "seq_change_slot": seq["seq_change_slot"] if seq else -1,
            "seq_raw_alarm": bool(seq["seq_raw_alarm"]) if seq else False,
            "rollback_to_slot": seq["rollback_to_slot"] if seq else -1,
            "attr_posterior": seq["attr_posterior"] if seq else np.nan,
            "attr_x": seq["attr_x"] if seq else np.nan,
            "attr_y": seq["attr_y"] if seq else np.nan,
            "attr_rho": seq["attr_rho"] if seq else np.nan,
            "attr_flag": bool(attr_flag),
        }
        if self.monitor is not None:
            self._snapshot(t)
        return out


class XGBWrapper:
    """Thin wrapper so fitted XGBoost models expose P(PUEA) quickly for single rows."""

    def __init__(self, model):
        self.model = model
        self.booster = model.get_booster()

    def predict_puea(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(self.booster.inplace_predict(X), dtype=float).reshape(-1)
