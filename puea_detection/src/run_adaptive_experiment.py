"""
Step 2 of the adaptive-PUEA experiments: calibrate, then run every detection
system on every scenario and write per-slot traces.

Per regime and seed:

1. Load simulation data written by run_simulation.py (simulation-settings hash checked).
2. Train XGBoost (FC features) and Spatial-XGBoost (raw per-SU RSS) on
   ``xgb_train`` only. Hyperparameters are those of the repository baseline.
3. Calibrate on the PU-only ``calibration`` stream only:
   - physical model (alpha, sigma, residual threshold) - physical_consistency.py
   - initial PU profile (mean, covariance, Mahalanobis threshold) - adaptive_profile.py
   - sequential location test (sigma reference, AR(1) phi, whitening, threshold) - sequential_spatial.py
   Saved to reports/adaptive/calibration/<regime>_seed_<k>.json.
4. For every scenario and every system in adaptive_systems.SYSTEMS, process
   the stream slot by slot (causally). Closed-loop scenarios (poisoning) are
   simulated live against each system, with identical randomness across
   systems.
5. Write traces to reports/adaptive/traces/<regime>/seed_<k>/<scenario>.csv.gz.

Finally runs the anchor-resolution sweep (reports/adaptive/anchor_resolution.csv).

Information available to a detector at slot t: the SU RSS and noise-estimate
vectors of slots 0..t, the SU positions, the known PU location, the trained
classifiers and the calibration outputs. Evaluation-only columns (labels,
oracle profile, contamination) are computed by this script *outside* the
detector and are never fed back into it.

Usage (from the repository root):
    python puea_detection/src/run_adaptive_experiment.py [--seeds 0] [--regimes default] [--skip-sweep]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from adaptive_profile import calibrate_profile
from adaptive_systems import SYSTEMS, DetectionSystem, SlotEvidence, XGBWrapper
from fc_features import FEATURE_NAMES, fc_features
from physical_consistency import PhysicalConsistencyChecker, calibrate_physical_model
from sequential_spatial import calibrate_sequential, candidate_grid
from simulation import (
    CRNSimulator,
    ScenarioStream,
    StreamData,
    simulation_hash,
    experiment_paths,
    load_config,
    matched_power_dbm,
    regime_config,
    regime_names,
)
from simulation.scenarios import AttackPolicy, AttackSlot


# ---------------------------------------------------------------------------
# Training / calibration
# ---------------------------------------------------------------------------

def train_xgb(X, y, cfg):
    x = cfg["xgboost"]
    model = XGBClassifier(
        n_estimators=x["n_estimators"], max_depth=x["max_depth"], learning_rate=x["learning_rate"],
        subsample=x["subsample"], colsample_bytree=x["colsample_bytree"], n_jobs=-1,
        random_state=x["random_state"], eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def build_models(cfg, regime, seed, sim, train: StreamData, cal: StreamData, paths) -> dict:
    tr = train.observable()
    y = train.labels
    xgb = train_xgb(fc_features(tr["rss_dbm"], tr["noise_est_dbm"]), y, cfg)
    spatial = train_xgb(tr["rss_dbm"], y, cfg)
    xgb.save_model(os.path.join(paths["sim_models"], f"xgb_fc_{regime}_seed{seed}.json"))
    spatial.save_model(os.path.join(paths["sim_models"], f"xgb_spatial_{regime}_seed{seed}.json"))

    co = cal.observable()
    phys_model = calibrate_physical_model(co["rss_dbm"], co["noise_est_dbm"], sim.geometry.su_xy, cfg)
    profile = calibrate_profile(fc_features(co["rss_dbm"], co["noise_est_dbm"]), cfg, FEATURE_NAMES)
    seq_cal = calibrate_sequential(co["rss_dbm"], co["noise_est_dbm"], sim.geometry.su_xy, phys_model, cfg)
    sc = cfg["sequential"]
    candidates = candidate_grid(phys_model.pu_xy, phys_model.area_size, float(sc["coarse_step"]),
                                float(sc["local_radius"]), float(sc["local_step"]), float(sc["min_offset"]))

    cal_dir = os.path.join(paths["adaptive_reports"], "calibration")
    os.makedirs(cal_dir, exist_ok=True)
    with open(os.path.join(cal_dir, f"{regime}_seed_{seed}.json"), "w") as fh:
        json.dump({
            "regime": regime,
            "seed": seed,
            "xgb_decision_threshold": cfg["xgboost"]["decision_threshold"],
            "physical_model": {k: v for k, v in phys_model.to_dict().items()
                               if k != "calibration_residuals_sorted"},
            "n_physical_calibration_residuals": len(phys_model.calibration_residuals_sorted),
            "profile": profile.to_dict(),
            "sequential": seq_cal.to_dict(),
        }, fh, indent=2)

    return {
        "xgb": XGBWrapper(xgb),
        "spatial_xgb": XGBWrapper(spatial),
        "physical": PhysicalConsistencyChecker(phys_model, sim.geometry.su_xy),
        "phys_model": phys_model,
        "profile": profile,
        "xgb_threshold": float(cfg["xgboost"]["decision_threshold"]),
        "seq_cal": seq_cal,
        "su_xy": sim.geometry.su_xy,
        "seq_context": {"cal": seq_cal, "cfg": sc, "phys_model": phys_model, "su_xy": sim.geometry.su_xy,
                        "candidates": candidates,
                        "attr_pfa": float(cfg["physical_consistency"]["calibration_false_alarm"])},
    }


# ---------------------------------------------------------------------------
# Running systems
# ---------------------------------------------------------------------------

SCALAR_KEEP = ["slot", "label", "transmitter", "tx_x", "tx_y", "tx_power_dbm", "pu_power_dbm",
               "shadow_sigma_db", "noise_floor_dbm", "attack_progress", "attack_phase"]


def _new_system(spec, models):
    return DetectionSystem(spec, models["profile"], models["xgb_threshold"], models["phys_model"].threshold,
                           models["seq_context"])


def _evaluation_columns(rows, labels, oracle_feats, profile_means, accepted, lam, cfg, initial_profile):
    """
    Evaluation-only columns: adaptation error, profile drift, contamination.
    Contamination follows rollbacks: when a system restores the checkpoint
    taken after slot k, the contamination returns to its value after slot k.
    """
    rollback = rows["rollback_to_slot"].to_numpy(int) if "rollback_to_slot" in rows else np.full(len(labels), -1)
    weight = rows["update_weight"].to_numpy(float) if "update_weight" in rows else accepted.astype(float)
    w = int(cfg["evaluation"]["oracle_window"])
    csum = np.cumsum(np.vstack([np.zeros(oracle_feats.shape[1]), oracle_feats]), axis=0)
    idx = np.arange(len(oracle_feats))
    lo = np.maximum(idx + 1 - w, 0)
    oracle_mean = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)[:, None]

    inv0 = initial_profile._inv0
    if profile_means is not None:
        diff = profile_means - oracle_mean
        rows["adaptation_error"] = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv0, diff))
        d0 = profile_means - initial_profile.initial_mean
        rows["profile_drift"] = np.sqrt(np.einsum("ij,jk,ik->i", d0, inv0, d0))
        contamination = np.zeros(len(labels))
        a = 0.0
        for t in range(len(labels)):
            if rollback[t] >= 0:
                a = contamination[rollback[t]]
            if accepted[t]:
                lw = lam * weight[t]
                a = (1 - lw) * a + lw * labels[t]
            contamination[t] = a
        rows["profile_contamination"] = contamination
        for j, n in enumerate(FEATURE_NAMES):
            rows[f"profile_mu_{n}"] = profile_means[:, j]
    else:
        rows["adaptation_error"] = np.nan
        rows["profile_drift"] = np.nan
        rows["profile_contamination"] = np.nan
    for j, n in enumerate(FEATURE_NAMES):
        rows[f"oracle_mu_{n}"] = oracle_mean[:, j]
    return rows


def run_open_loop(stream: StreamData, models, cfg) -> pd.DataFrame:
    ev = SlotEvidence(stream.observable(), models)  # detector input: observable fields only
    o = stream.oracle()
    oracle_feats = fc_features(o["rss_dbm"], o["noise_est_dbm"])
    labels = stream.labels
    frames = []
    for spec in SYSTEMS:
        system = _new_system(spec, models)
        out, means = [], []
        for t in range(len(ev)):
            out.append(system.step(ev, t))
            if system.profile is not None:
                means.append(system.profile.mean.copy())
        df = pd.DataFrame(out)
        df = pd.concat([stream.scalars[SCALAR_KEEP].reset_index(drop=True), df], axis=1)
        df.insert(0, "system", spec.name)
        df = _evaluation_columns(df, labels, oracle_feats, np.array(means) if means else None,
                                 df["accepted_for_update"].to_numpy(), models["profile"].ewma_lambda,
                                 cfg, models["profile"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_closed_loop(sim, name, scfg, models, cfg, realized_dir) -> pd.DataFrame:
    n = cfg["streams"]["n_slots"]
    frames = []
    for spec in SYSTEMS:
        stream = ScenarioStream(sim, name, scfg, n)  # same stream name -> identical randomness
        system = _new_system(spec, models)
        rows, obs_all, out, means = [], [], [], []
        for t in range(n):
            row, obs = stream.slot(t)
            ev = SlotEvidence({"rss_dbm": obs["rss_dbm"], "noise_est_dbm": obs["noise_est_dbm"]}, models)
            res = system.step(ev, 0, t)
            stream.feedback(res["attack_detected"])  # attacker observes whether SUs vacated
            rows.append(row)
            obs_all.append(obs)
            out.append(res)
            if system.profile is not None:
                means.append(system.profile.mean.copy())
        realized = StreamData.from_slots(name, rows, obs_all)
        realized.save(os.path.join(realized_dir, f"{name}__{spec.name}.csv.gz"))
        o = realized.oracle()
        df = pd.concat([realized.scalars[SCALAR_KEEP], pd.DataFrame(out)], axis=1)
        df.insert(0, "system", spec.name)
        df = _evaluation_columns(df, realized.labels, fc_features(o["rss_dbm"], o["noise_est_dbm"]),
                                 np.array(means) if means else None, df["accepted_for_update"].to_numpy(),
                                 models["profile"].ewma_lambda, cfg, models["profile"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Anchor-resolution sweep
# ---------------------------------------------------------------------------

class _FixedPointAttacker(AttackPolicy):
    """Power-matched attacker at a fixed point, active every slot (sweep only)."""

    def __init__(self, sim, xy, rnd):
        super().__init__(sim, {}, rnd)
        self.xy = np.asarray(xy, float)

    def slot_params(self, t, sched):
        p = matched_power_dbm(self.xy, sched.pu_power_dbm[t], self.geometry, sched.alpha[t])
        return AttackSlot(True, self.xy, p, 1.0, "attack")


def anchor_resolution_sweep(cfg, seeds) -> pd.DataFrame:
    sw = cfg["resolution_sweep"]
    records = []
    for sigma in sw["shadow_sigmas_db"]:
        for n_sus in sw["n_sus"]:
            c = copy.deepcopy(cfg)
            c["channel"]["shadow_sigma_db"] = sigma
            c["geometry"]["n_sus"] = n_sus
            for seed in seeds:
                sim = CRNSimulator(c, seed)
                cal = ScenarioStream(sim, "calibration", {}, c["streams"]["n_calibration_slots"],
                                     "calibration").generate().observable()
                model = calibrate_physical_model(cal["rss_dbm"], cal["noise_est_dbm"], sim.geometry.su_xy, c)
                checker = PhysicalConsistencyChecker(model, sim.geometry.su_xy)
                pu = ScenarioStream(sim, "sweep_pu", {}, sw["n_slots_per_point"], "sweep/pu").generate()
                pu_ok = checker.compute(**pu.observable())["physically_consistent"]
                rng = np.random.default_rng(seed)
                area = c["geometry"]["area_size"]
                for d in sw["distances"]:
                    rejected, total = 0, 0
                    for k in range(sw["n_directions"]):
                        for _ in range(100):
                            ang = rng.uniform(0, 2 * np.pi)
                            xy = sim.geometry.pu_xy + d * np.array([np.cos(ang), np.sin(ang)])
                            if np.all((xy >= 0) & (xy <= area)):
                                break
                        st = ScenarioStream(sim, "sweep", {}, sw["n_slots_per_point"], f"sweep/{d}/{k}")
                        st.policy = _FixedPointAttacker(sim, xy, st.rnd)
                        out = checker.compute(**st.generate().observable())
                        rejected += int((~out["physically_consistent"]).sum())
                        total += len(out["physically_consistent"])
                    records.append({
                        "shadow_sigma_db": sigma, "n_sus": n_sus, "seed": seed, "distance": d,
                        "attacker_rejection_rate": rejected / total,
                        "pu_false_rejection_rate": float(1 - pu_ok.mean()),
                        "threshold": model.threshold,
                    })
        print(f"[sweep] sigma={sigma} done")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_seed(cfg, regime, seed, paths):
    t0 = time.time()
    sim = CRNSimulator(cfg, seed)
    sdir = os.path.join(paths["sim_data"], regime, f"seed_{seed}")
    train = StreamData.load("xgb_train", os.path.join(sdir, "xgb_train.csv.gz"))
    cal = StreamData.load("calibration", os.path.join(sdir, "calibration.csv.gz"))
    models = build_models(cfg, regime, seed, sim, train, cal, paths)
    pm = models["phys_model"]
    print(f"[{regime} seed {seed}] alpha_hat={pm.alpha:.3f} sigma_hat={pm.sigma_db:.2f} dB "
          f"phys_threshold={pm.threshold:.2f} (chi2 ref {pm.theoretical_threshold:.2f}) "
          f"profile_threshold={models['profile'].threshold:.2f}")

    tdir = os.path.join(paths["traces"], regime, f"seed_{seed}")
    rdir = os.path.join(sdir, "realized")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    for name, scfg in cfg["scenarios"].items():
        path = os.path.join(sdir, f"test_{name}.csv.gz")
        if os.path.exists(path):
            trace = run_open_loop(StreamData.load(name, path), models, cfg)
        else:
            trace = run_closed_loop(sim, name, scfg, models, cfg, rdir)
        trace.insert(0, "scenario", name)
        trace.insert(0, "seed", seed)
        trace.insert(0, "regime", regime)
        trace.to_csv(os.path.join(tdir, f"{name}.csv.gz"), index=False, compression="gzip")
    print(f"[{regime} seed {seed}] traces written in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = experiment_paths()
    seeds = args.seeds if args.seeds else cfg["seeds"]
    regimes = args.regimes if args.regimes else regime_names(cfg)

    manifest_path = os.path.join(paths["sim_data"], "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("No simulation data. Run: python puea_detection/src/run_simulation.py")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    if manifest.get("simulation_hash") != simulation_hash(cfg):
        raise SystemExit("Simulation data was generated with different simulation settings. "
                         "Re-run run_simulation.py.")

    for regime in regimes:
        rcfg = regime_config(cfg, regime)
        for seed in seeds:
            if str(seed) not in manifest["regimes"].get(regime, {}):
                raise SystemExit(f"{regime}/seed {seed} missing from simulation data. Re-run run_simulation.py.")
            run_seed(rcfg, regime, seed, paths)

    if not args.skip_sweep:
        t0 = time.time()
        sweep = anchor_resolution_sweep(cfg, seeds)
        sweep.to_csv(os.path.join(paths["adaptive_reports"], "anchor_resolution.csv"), index=False)
        print(f"[sweep] anchor resolution written in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
