"""
Step 3 of the adaptive-PUEA experiments: metrics, report and figures.

Reads the per-slot traces written by run_adaptive_experiment.py and computes,
for every (regime, seed, scenario, system):

Standard metrics (all slots of the stream):
    accuracy, precision, recall, f1, fpr, fnr, roc_auc
    (precision/recall/f1/fnr/roc_auc are NaN for PU-only scenarios)

Research-specific metrics:
    adaptation_error_mean / _final
        Mahalanobis distance (initial-profile metric) between the system's
        profile mean and the true legitimate PU feature mean, estimated from
        evaluation-only oracle PU observations (trailing window). "_final" =
        mean over the last ``rolling_window`` slots.
    poisoning_rate
        Fraction of attacker slots accepted for a profile update.
    profile_contamination_max / _final
        Fraction of the profile mean's EWMA weight contributed by attacker
        slots (exact for the EWMA mean).
    drift_false_alarm_rate
        FPR on legitimate slots from the start of the drift ramp onward.
    attack_detection_rate
        Recall on attacker slots (= recall).
    detection_delay
        Slots from the first attacker slot to the first flagged attacker slot.
    recovery_slots / recovered
        Settling time: slots from drift onset to the last slot at which the
        rolling FPR (legit slots, window ``rolling_window``) exceeds
        ``recovery_fpr_tolerance``; 0 if it never exceeds it. If it is still
        above at the end, recovery_slots = NaN and recovered = 0. In the
        summary, ``recovered`` is the fraction of seeds that recovered and
        recovery_slots is averaged over recovered seeds only.
    attack_phase_recall / poison_phase_recall / max_poison_progress
        Poisoning scenarios only.
    contamination_at_attack_start
        Profile contamination at the first slot of the attack phase
        (poisoning scenarios; accounts for rollbacks).

Sequential-extension metrics (systems with the sequential monitor only):
    seq_alarm_delay
        Slots from the first attacker slot to the first slot in alarm.
    seq_alarm_before_attack
        Fraction of slots in alarm before the first attacker slot (all slots
        for PU-only scenarios) - the sequential test's false-alarm burden.
    n_rollbacks
        Number of rollbacks performed.

Outputs:
    reports/adaptive/metrics_by_seed.csv
    reports/adaptive/metrics_summary.csv      (mean and std over seeds)
    reports/adaptive/adaptive_report.md
    reports/figures/adaptive_*.png            (via adaptive_plots.py)

Usage (from the repository root):
    python puea_detection/src/evaluate_adaptive.py [--no-figures]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adaptive_systems import SYSTEMS
from simulation import drift_window, experiment_paths, load_config, regime_config

ABLATION_ORDER = [s.name for s in SYSTEMS]


def load_traces(paths) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(paths["traces"], "*", "seed_*", "*.csv.gz")))
    if not files:
        raise SystemExit("No traces found. Run run_adaptive_experiment.py first.")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def _rate(num, den):
    return float(num) / float(den) if den else np.nan


def recovery(legit: pd.DataFrame, onset: int, window: int, tol: float):
    """
    Settling time after drift onset: slots from ``onset`` to the last slot at
    which the rolling FPR (legitimate slots, ``window``) exceeds ``tol``.
    Returns (slots, recovered): (0, True) if it never exceeds ``tol``;
    (NaN, False) if it is still above ``tol`` at the end of the stream.
    """
    after = legit[legit["slot"] >= onset]
    if len(after) < window:
        return np.nan, False
    roll = after["attack_detected"].astype(float).rolling(window, min_periods=window).mean().to_numpy()
    slots = after["slot"].to_numpy()
    above = np.where(roll > tol)[0]
    if above.size == 0:
        return 0.0, True
    if roll[-1] > tol:
        return np.nan, False
    return float(slots[above[-1]] - onset), True


def metrics_for(d: pd.DataFrame, scfg: dict, cfg: dict) -> dict:
    y = d["label"].to_numpy(int)
    p = d["attack_detected"].to_numpy(bool)
    tp, fp = int(((y == 1) & p).sum()), int(((y == 0) & p).sum())
    fn, tn = int(((y == 1) & ~p).sum()), int(((y == 0) & ~p).sum())
    precision = _rate(tp, tp + fp) if y.sum() else np.nan
    recall = _rate(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if y.sum() and (precision + recall) > 0 else np.nan
    m = {
        "n_slots": len(d), "n_attack": int(y.sum()),
        "accuracy": _rate(tp + tn, len(d)), "precision": precision, "recall": recall, "f1": f1,
        "fpr": _rate(fp, fp + tn), "fnr": _rate(fn, tp + fn),
        "roc_auc": roc_auc_score(y, d["score"]) if 0 < y.sum() < len(y) else np.nan,
        "attack_detection_rate": recall,
    }

    has_profile = d["adaptation_error"].notna().any()
    ev = cfg["evaluation"]
    w = int(ev["rolling_window"])
    m["adaptation_error_mean"] = d["adaptation_error"].mean() if has_profile else np.nan
    m["adaptation_error_final"] = d["adaptation_error"].tail(w).mean() if has_profile else np.nan
    m["profile_drift_final"] = d["profile_drift"].iloc[-1] if has_profile else np.nan
    m["profile_contamination_max"] = d["profile_contamination"].max() if has_profile else np.nan
    m["profile_contamination_final"] = d["profile_contamination"].iloc[-1] if has_profile else np.nan
    adaptive = d["accepted_for_update"].any() or d["rejected_for_update"].any()
    m["poisoning_rate"] = d.loc[y == 1, "accepted_for_update"].mean() if (adaptive and y.sum()) else np.nan
    m["n_profile_updates"] = int(d["accepted_for_update"].sum())

    if y.sum():
        att = d[y == 1]
        onset = int(att["slot"].iloc[0])
        hit = att[att["attack_detected"]]
        m["detection_delay"] = float(hit["slot"].iloc[0] - onset) if len(hit) else np.nan
    else:
        m["detection_delay"] = np.nan

    dw = drift_window(scfg)
    legit = d[y == 0]
    if dw:
        m["drift_false_alarm_rate"] = legit.loc[legit["slot"] >= dw[0], "attack_detected"].mean()
        rec, ok = recovery(legit, dw[0], w, float(ev["recovery_fpr_tolerance"]))
        m["recovery_slots"], m["recovered"] = rec, float(ok)
    else:
        m["drift_false_alarm_rate"], m["recovery_slots"], m["recovered"] = np.nan, np.nan, np.nan

    if (scfg.get("attack") or {}).get("type") == "poisoning":
        ph = d["attack_phase"]
        m["poison_phase_recall"] = d.loc[(y == 1) & (ph == "poison"), "attack_detected"].mean()
        m["attack_phase_recall"] = d.loc[(y == 1) & (ph == "attack"), "attack_detected"].mean()
        m["max_poison_progress"] = d.loc[(y == 1) & (ph == "poison"), "attack_progress"].max()
        att_phase = d[ph == "attack"]
        m["contamination_at_attack_start"] = (att_phase["profile_contamination"].iloc[0]
                                              if has_profile and len(att_phase) else np.nan)

    has_seq = "seq_stat" in d and d["seq_stat"].notna().any()
    if has_seq:
        alarm = d["seq_alarm"].astype(bool).to_numpy()
        slots = d["slot"].to_numpy()
        onset = int(d.loc[y == 1, "slot"].iloc[0]) if y.sum() else None
        pre = slots < onset if onset is not None else np.ones(len(d), bool)
        m["seq_alarm_before_attack"] = float(alarm[pre].mean()) if pre.any() else np.nan
        hit = slots[alarm & ~pre] if onset is not None else np.array([])
        m["seq_alarm_delay"] = float(hit[0] - onset) if hit.size else np.nan
        m["n_rollbacks"] = int((d["rollback_to_slot"] >= 0).sum())
    else:
        m["seq_alarm_before_attack"] = m["seq_alarm_delay"] = m["n_rollbacks"] = np.nan
    return m


def compute_metrics(traces: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for (regime, seed, scen, system), d in traces.groupby(["regime", "seed", "scenario", "system"], sort=False):
        rcfg = regime_config(cfg, regime)
        m = metrics_for(d.sort_values("slot"), rcfg["scenarios"][scen], rcfg)
        rows.append({"regime": regime, "seed": seed, "scenario": scen, "system": system, **m})
    return pd.DataFrame(rows)


def summarise(by_seed: pd.DataFrame) -> pd.DataFrame:
    num = by_seed.drop(columns=["seed"]).select_dtypes(include=[np.number, bool]).columns
    g = by_seed.groupby(["regime", "scenario", "system"], sort=False)[list(num)]
    mean, std = g.mean(), g.std()
    std.columns = [f"{c}_std" for c in std.columns]
    out = pd.concat([mean, std], axis=1).reset_index()
    out["n_seeds"] = g.size().to_numpy()
    return out


# ---------------------------------------------------------------------------
# Markdown report (tables only - interpretation is left to the reader)
# ---------------------------------------------------------------------------

def _fmt(mean, std, digits=3):
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if not pd.isna(std) else f"{mean:.{digits}f}"


def _table(summary, regime, scenarios, metric, digits=3):
    s = summary[summary["regime"] == regime]
    header = "| System | " + " | ".join(scenarios) + " |"
    lines = [header, "|" + "---|" * (len(scenarios) + 1)]
    for system in ABLATION_ORDER:
        cells = []
        for scen in scenarios:
            r = s[(s["system"] == system) & (s["scenario"] == scen)]
            cells.append(_fmt(r[metric].iloc[0], r[f"{metric}_std"].iloc[0], digits) if len(r) else "-")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, cfg: dict, path: str, sweep: pd.DataFrame | None):
    scen_all = [k for k in cfg["scenarios"]]
    legit = [k for k, v in cfg["scenarios"].items() if v["group"] == "legitimate"]
    attack = [k for k, v in cfg["scenarios"].items() if v["group"] == "attack"]
    poison = [k for k, v in cfg["scenarios"].items() if (v.get("attack") or {}).get("type") == "poisoning"]
    n_seeds = int(summary["n_seeds"].max())

    out = [
        "# Physically Anchored Adaptive PUEA Detection — Experiment Report",
        "",
        "_Auto-generated by `src/evaluate_adaptive.py` from the traces in `reports/adaptive/traces/`. "
        f"Values are mean ± std over {n_seeds} seeds. Simulated data only — see docs/adaptive_puea.md "
        "for the threat model, assumptions and limitations._",
        "",
        "## Systems",
        "",
        "| System | Ablation | Description |",
        "|---|---|---|",
    ]
    for s in SYSTEMS:
        out.append(f"| {s.name} | {s.ablation} | {s.description} |")
    out += ["", "## Scenarios", ""]
    for k, v in cfg["scenarios"].items():
        out.append(f"- **{k}** ({v['group']}): {v['description']}")

    for regime in summary["regime"].unique():
        out += ["", f"## Regime: `{regime}`", ""]
        ov = cfg.get("regimes", {}).get(regime, {})
        out += [f"Overrides: `{ov}`", ""]
        blocks = [
            ("False-positive rate (legitimate slots)", scen_all, "fpr"),
            ("F1 (attack scenarios)", attack, "f1"),
            ("Recall / attack detection rate", attack, "recall"),
            ("ROC-AUC (composite score; attack scenarios)", attack, "roc_auc"),
            ("False alarms after drift onset (legitimate drift scenarios)", legit, "drift_false_alarm_rate"),
            ("Adaptation recovery / settling time (slots; mean over seeds that recovered)", legit, "recovery_slots"),
            ("Fraction of seeds whose false-alarm rate recovered", legit, "recovered"),
            ("Adaptation error, final (Mahalanobis units)", scen_all, "adaptation_error_final"),
            ("Poisoning rate (attacker slots accepted for update)", attack, "poisoning_rate"),
            ("Profile contamination, max", attack, "profile_contamination_max"),
            ("Detection delay (slots)", attack, "detection_delay"),
            ("Recall in attack phase after poisoning", poison, "attack_phase_recall"),
            ("Max poisoning progress reached (0 = PU mimic, 1 = target)", poison, "max_poison_progress"),
            ("Profile contamination at the start of the attack phase (after any rollback)", poison,
             "contamination_at_attack_start"),
            ("Sequential alarm delay (slots from first attacker slot; sequential systems)", attack,
             "seq_alarm_delay"),
            ("Fraction of slots in sequential alarm before any attack (false-alarm burden)", scen_all,
             "seq_alarm_before_attack"),
        ]
        for title, scens, metric in blocks:
            digits = 1 if metric in ("recovery_slots", "detection_delay", "seq_alarm_delay") else 3
            out += [f"### {title}", "", _table(summary, regime, scens, metric, digits), ""]

    if sweep is not None and not sweep.empty:
        out += ["## Anchor resolution (physical check alone)", "",
                "Fraction of slots from a power-matched attacker at distance d from the PU that the "
                "physical check rejects (mean over seeds and directions). Rightmost column: PU slots "
                "falsely rejected.", ""]
        piv = sweep.pivot_table(index=["shadow_sigma_db", "n_sus"], columns="distance",
                                values="attacker_rejection_rate", aggfunc="mean")
        fr = sweep.groupby(["shadow_sigma_db", "n_sus"])["pu_false_rejection_rate"].mean()
        cols = list(piv.columns)
        out.append("| σ (dB) | SUs | " + " | ".join(f"d={c:g}" for c in cols) + " | PU false rej. |")
        out.append("|" + "---|" * (len(cols) + 3))
        for (sig, n), row in piv.iterrows():
            out.append(f"| {sig:g} | {n} | " + " | ".join(f"{v:.2f}" for v in row) + f" | {fr[(sig, n)]:.3f} |")

    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = experiment_paths()
    traces = load_traces(paths)
    by_seed = compute_metrics(traces, cfg)
    summary = summarise(by_seed)
    rdir = paths["adaptive_reports"]
    by_seed.to_csv(os.path.join(rdir, "metrics_by_seed.csv"), index=False)
    summary.to_csv(os.path.join(rdir, "metrics_summary.csv"), index=False)
    sweep_path = os.path.join(rdir, "anchor_resolution.csv")
    sweep = pd.read_csv(sweep_path) if os.path.exists(sweep_path) else None
    write_report(summary, cfg, os.path.join(rdir, "adaptive_report.md"), sweep)
    print(f"[evaluate_adaptive] metrics + report written to {rdir}")

    if not args.no_figures:
        import adaptive_plots

        adaptive_plots.make_all(traces, summary, sweep, cfg, paths)
        print(f"[evaluate_adaptive] figures written to {paths['figures']}")


if __name__ == "__main__":
    main()
