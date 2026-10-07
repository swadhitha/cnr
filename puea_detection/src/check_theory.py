"""
Kill-criterion 1: does the analytic location-information model predict the
measured single-slot anchor resolution?

For every (shadow sigma, n_SUs, seed, distance) cell of the anchor-resolution
sweep (reports/adaptive/anchor_resolution.csv), rebuild the same geometry and
attacker positions (same RNG sequence as run_adaptive_experiment.py), compute

* lam_exact = exact GLRT noncentrality (P profiled out),
* lam_fisher = delta^T J delta (linearised),

and predict the rejection rate P(ncx2(2, lam) > calibrated threshold). Writes
reports/adaptive/theory_check.csv and prints the agreement.

Part 2 (``--sequential``): does the first-order delay formula predict how
long the sequential test takes to alarm? A power-matched attacker at a
fixed distance d from the PU (direction toward the default attacker point)
transmits in a fraction rho of slots after a PU-only warm-up; the monitor is
calibrated exactly as in the experiments. Measured first-alarm delay is
compared with the model-based first-passage prediction
(predicted_delay_model, geometry + rho + phi + h only) and with the
first-order drift formula (predicted_delay). Writes
reports/adaptive/theory_check_sequential.csv.

Usage: python puea_detection/src/check_theory.py [--sequential]
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import pandas as pd

from sequential_spatial import (
    SequentialSpatialMonitor,
    calibrate_sequential,
    fisher_information,
    location_gradients,
    noncentrality,
    predicted_delay,
    predicted_delay_model,
    predicted_rejection,
    score_noncentrality,
    slot_scores,
)
from physical_consistency import calibrate_physical_model
from simulation import CRNSimulator, ScenarioStream, experiment_paths, load_config, matched_power_dbm, regime_config
from simulation.scenarios import AttackPolicy, AttackSlot, point_toward

SEQ_DISTANCES = [2.0, 3.0, 5.0, 8.0]
SEQ_RATES = [0.1, 0.3]
SEQ_WARMUP = 200
SEQ_HORIZON = 3000


class _IntermittentFixedAttacker(AttackPolicy):
    """Power-matched attacker at a fixed point, active with probability ``rate`` after ``onset``."""

    def __init__(self, sim, xy, rate, onset, rnd):
        super().__init__(sim, {}, rnd)
        self.xy, self.rate, self.onset = np.asarray(xy, float), rate, onset

    def slot_params(self, t, sched):
        if not self.is_active(t, self.onset, self.rate):
            return AttackSlot(active=False)
        p = matched_power_dbm(self.xy, sched.pu_power_dbm[t], self.geometry, sched.alpha[t])
        return AttackSlot(True, self.xy, p, 1.0, "attack")


def sequential_check(cfg0, paths):
    rows = []
    for regime in ["default", "weak_anchor", "correlated_shadowing"]:
        cfg = regime_config(cfg0, regime)
        for seed in cfg["seeds"]:
            sim = CRNSimulator(cfg, seed)
            geo = sim.geometry
            cal = ScenarioStream(sim, "calibration", {}, cfg["streams"]["n_calibration_slots"],
                                 "calibration").generate().observable()
            pm = calibrate_physical_model(cal["rss_dbm"], cal["noise_est_dbm"], geo.su_xy, cfg)
            sc = calibrate_sequential(cal["rss_dbm"], cal["noise_est_dbm"], geo.su_xy, pm, cfg)
            for d in SEQ_DISTANCES:
                xy = point_toward(geo.pu_xy, cfg["attacker"]["basic_xy"], d)
                sig = np.sqrt(sc.sigma2_ref)
                lam = noncentrality(xy, geo.pu_xy, geo.su_xy, pm.alpha, sig, geo.min_distance)
                mu2 = score_noncentrality(xy, geo.pu_xy, geo.su_xy, pm.alpha, sig, geo.min_distance)
                for rho in SEQ_RATES:
                    st = ScenarioStream(sim, "seqcheck", {}, SEQ_WARMUP + SEQ_HORIZON, f"seqcheck/{d}/{rho}")
                    st.policy = _IntermittentFixedAttacker(sim, xy, rho, SEQ_WARMUP, st.rnd)
                    obs = st.generate().observable()
                    u0, J0, s2, s, valid = slot_scores(obs["rss_dbm"], obs["noise_est_dbm"], geo.su_xy, pm.pu_xy,
                                                       pm.alpha, pm.min_snr_db, pm.min_distance, pm.area_size,
                                                       pm.grid_step)
                    mon = SequentialSpatialMonitor(sc, geo.n_sus)
                    ver = valid.sum(axis=1) >= pm.min_valid_sus
                    first, pre_alarm = None, 0
                    for t in range(len(u0)):
                        a = mon.step(t, u0[t], J0[t], s2[t], s[t], valid[t], ver[t])["seq_alarm"]
                        if a and t < SEQ_WARMUP:
                            pre_alarm += 1
                        if a and t >= SEQ_WARMUP:
                            first = t - SEQ_WARMUP
                            break
                    rows.append({
                        "regime": regime, "seed": seed, "distance": d, "rho": rho, "lam_exact": lam,
                        "score_noncentrality": mu2, "phi": sc.phi, "threshold": sc.threshold,
                        "predicted_delay_first_order": predicted_delay(mu2, rho, sc.threshold, sc.phi),
                        "predicted_delay": predicted_delay_model(mu2, rho, sc.threshold, sc.window, sc.phi,
                                                                 horizon=SEQ_HORIZON),
                        "measured_delay": float(first) if first is not None else np.nan,
                        "censored": first is None, "warmup_alarm_slots": pre_alarm,
                    })
        print(f"[seq-check] {regime} done")
    df = pd.DataFrame(rows)
    out = os.path.join(paths["adaptive_reports"], "theory_check_sequential.csv")
    df.to_csv(out, index=False)
    print(f"cells {len(df)}, censored (no alarm within {SEQ_HORIZON}) {int(df['censored'].sum())}, "
          f"warm-up false-alarm slots {int(df['warmup_alarm_slots'].sum())}")
    for col in ["predicted_delay", "predicted_delay_first_order"]:
        ok = df[~df["censored"] & np.isfinite(df[col])]
        ratio = ok["measured_delay"] / ok[col].clip(lower=1)
        print(f"measured/{col}: median {ratio.median():.2f}, IQR {ratio.quantile(.25):.2f}-"
              f"{ratio.quantile(.75):.2f}; corr(log) "
              f"{np.corrcoef(np.log(ok['measured_delay'] + 1), np.log(ok[col] + 1))[0, 1]:.3f}")
    # Seed-level medians: a single realisation's first passage is very noisy, so compare
    # the median over seeds with the model's median.
    g = df.groupby(["regime", "distance", "rho"])
    tab = pd.DataFrame({"pred_model": g["predicted_delay"].median(),
                        "pred_first_order": g["predicted_delay_first_order"].median(),
                        "measured_median": g["measured_delay"].median(),
                        "censored": g["censored"].sum()}).round(0)
    print(tab.to_string())
    print(f"written {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequential", action="store_true", help="run the sequential-delay check instead")
    args = ap.parse_args()
    if args.sequential:
        sequential_check(load_config(), experiment_paths())
        return
    cfg = load_config()
    paths = experiment_paths()
    meas = pd.read_csv(os.path.join(paths["adaptive_reports"], "anchor_resolution.csv"))
    sw = cfg["resolution_sweep"]
    alpha = cfg["channel"]["path_loss_exponent"]
    rows = []
    for (sigma, n_sus, seed), grp in meas.groupby(["shadow_sigma_db", "n_sus", "seed"]):
        c = copy.deepcopy(cfg)
        c["channel"]["shadow_sigma_db"] = sigma
        c["geometry"]["n_sus"] = int(n_sus)
        sim = CRNSimulator(c, int(seed))
        geo = sim.geometry
        g = location_gradients(geo.pu_xy, geo.su_xy, alpha, geo.min_distance)
        J = fisher_information(g, np.ones(geo.n_sus), sigma)
        thr = float(grp["threshold"].iloc[0])
        rng = np.random.default_rng(int(seed))  # same sequence as anchor_resolution_sweep
        area = c["geometry"]["area_size"]
        for d in sw["distances"]:
            lam_e, lam_f = [], []
            for _ in range(sw["n_directions"]):
                for _ in range(100):
                    ang = rng.uniform(0, 2 * np.pi)
                    xy = geo.pu_xy + d * np.array([np.cos(ang), np.sin(ang)])
                    if np.all((xy >= 0) & (xy <= area)):
                        break
                lam_e.append(noncentrality(xy, geo.pu_xy, geo.su_xy, alpha, sigma, geo.min_distance))
                delta = xy - geo.pu_xy
                lam_f.append(float(delta @ J @ delta))
            m = grp[grp["distance"] == d]["attacker_rejection_rate"].iloc[0]
            rows.append({
                "shadow_sigma_db": sigma, "n_sus": int(n_sus), "seed": int(seed), "distance": d,
                "measured": m,
                "predicted_exact": float(np.mean(predicted_rejection(lam_e, thr))),
                "predicted_fisher": float(np.mean(predicted_rejection(lam_f, thr))),
                "lam_exact_mean": float(np.mean(lam_e)), "lam_fisher_mean": float(np.mean(lam_f)),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(paths["adaptive_reports"], "theory_check.csv")
    df.to_csv(out, index=False)

    for col in ["predicted_exact", "predicted_fisher"]:
        err = df[col] - df["measured"]
        print(f"{col:17s}: MAE {err.abs().mean():.3f}  max |err| {err.abs().max():.3f}  "
              f"bias {err.mean():+.3f}  corr {np.corrcoef(df[col], df['measured'])[0, 1]:.3f}")
    summ = df.groupby(["shadow_sigma_db", "n_sus", "distance"])[["measured", "predicted_exact",
                                                               "predicted_fisher"]].mean().round(3)
    print(summ.to_string())
    print(f"written {out}")


if __name__ == "__main__":
    main()
