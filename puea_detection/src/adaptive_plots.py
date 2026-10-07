"""
Figures for the adaptive-PUEA experiments (called by evaluate_adaptive.py).

All figures are written to reports/figures/ with an ``adaptive_`` prefix so
they never overwrite the original pipeline's figures. Time-series figures
show seed 0 of the 'default' regime (one realisation); comparison figures
aggregate over all seeds.

Colours follow the entity (system), in a fixed order, with line styles as a
secondary encoding. Time-series panels plot at most four systems.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

SYSTEM_STYLE = {
    "Baseline-XGBoost": ("#2a78d6", "-"),
    "Naive-Adaptive": ("#eb6834", "--"),
    "XGB+Physical": ("#1baf7a", "-."),
    "Physically-Anchored-Adaptive": ("#eda100", "-"),
    "XGB+StaticProfile": ("#e87ba4", ":"),
    "Anchored-Update-Only": ("#008300", "--"),
    "Physical-Only": ("#4a3aa7", "-."),
    "Spatial-XGBoost": ("#e34948", ":"),
}
# Sequential-extension systems are variants of D / Physical-Only: they share the
# parent's hue (no 9th categorical hue) and differ by line style + legend.
VARIANT_STYLE = {
    "Sequential-Anchored": ("#eda100", ":"),
    "Sequential-Anchored+Rollback": ("#eda100", "-."),
    "Sequential-Anchored-Full": ("#eda100", "--"),
    "Physical-Only+Sequential": ("#4a3aa7", "--"),
}
ALL_STYLE = {**SYSTEM_STYLE, **VARIANT_STYLE}
CORE = ["Baseline-XGBoost", "Naive-Adaptive", "XGB+Physical", "Physically-Anchored-Adaptive"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PU_COLOR, ATT_COLOR = "#2a78d6", "#eb6834"


def _style():
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def _save(fig, paths, name):
    out = os.path.join(paths["figures"], name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _one(traces, scenario, regime="default", seed=0):
    d = traces[(traces["regime"] == regime) & (traces["seed"] == seed) & (traces["scenario"] == scenario)]
    return d.sort_values(["system", "slot"])


def _plot_systems(ax, d, col, systems, window=None, legit_only=False, attack_only=False):
    for s in systems:
        x = d[d["system"] == s]
        if legit_only:
            x = x[x["label"] == 0]
        if attack_only:
            x = x[x["label"] == 1]
        y = x[col].astype(float)
        if window:
            y = y.rolling(window, min_periods=max(5, window // 4)).mean()
        c, ls = ALL_STYLE[s]
        ax.plot(x["slot"], y, color=c, ls=ls, label=s)


# ---------------------------------------------------------------------------

def fig_geometry(traces, cfg, paths):
    geo_path = os.path.join(paths["sim_data"], "default", "seed_0", "geometry.csv")
    if not os.path.exists(geo_path):
        return
    geo = pd.read_csv(geo_path)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(geo["x"], geo["y"], s=36, marker="^", color=MUTED, label="Secondary users (SUs)", zorder=3)
    pu = cfg["geometry"]["pu_xy"]
    ax.scatter(*pu, s=160, marker="*", color=PU_COLOR, edgecolor="white", lw=1.5, zorder=5, label="Legitimate PU")
    marks = [("basic_puea", "o", "Basic / power-matching attacker"),
             ("stealthy_puea", "s", "Stealthy attacker"),
             ("gradual_puea", "D", "Gradual attacker (stages)")]
    for scen, m, lab in marks:
        d = _one(traces, scen)
        d = d[(d["system"] == "Baseline-XGBoost") & (d["label"] == 1)]
        pts = d[["tx_x", "tx_y"]].drop_duplicates()
        ax.scatter(pts["tx_x"], pts["tx_y"], s=60, marker=m, facecolor="white", edgecolor=ATT_COLOR,
                   lw=2, zorder=4, label=lab)
    for scen, ls, lab in [("poisoning", "-", "Poisoning path (Naive-Adaptive run)"),
                          ("near_pu_poisoning", ":", "Near-PU poisoning path (Naive-Adaptive run)")]:
        d = _one(traces, scen)
        d = d[(d["system"] == "Naive-Adaptive") & (d["label"] == 1)]
        if len(d):
            ax.plot(d["tx_x"], d["tx_y"], ls=ls, color=ATT_COLOR, lw=1.5, alpha=0.9, label=lab)
    area = cfg["geometry"]["area_size"]
    ax.set_xlim(-2, area + 2)
    ax.set_ylim(-2, area + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Network geometry (default regime, seed 0)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    _save(fig, paths, "adaptive_geometry.png")


def fig_physical_residual(traces, paths):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, scen in zip(axes, ["gradual_puea", "near_pu_poisoning"]):
        d = _one(traces, scen)
        d = d[d["system"] == "Physically-Anchored-Adaptive"]
        if d.empty:
            continue
        for lab, c, name in [(0, PU_COLOR, "Legitimate PU slot"), (1, ATT_COLOR, "Attacker slot")]:
            x = d[d["label"] == lab]
            ax.scatter(x["slot"], x["physical_residual"] + 1, s=8, color=c, alpha=0.6, label=name, lw=0)
        cal = _calibration_threshold(paths)
        if cal is not None:
            ax.axhline(cal + 1, color=INK, lw=1, ls="--", label="Calibrated threshold")
        ax.set_yscale("log")
        ax.set_title(f"Physical residual (GLRT + 1) — {scen}")
        ax.set_xlabel("Slot")
    axes[0].set_ylabel("Residual + 1 (log scale)")
    axes[0].legend(loc="upper left", fontsize=8)
    _save(fig, paths, "adaptive_physical_residual.png")


def _calibration_threshold(paths, regime="default", seed=0):
    import json

    p = os.path.join(paths["adaptive_reports"], "calibration", f"{regime}_seed_{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)["physical_model"]["threshold"]


def fig_xgb_confidence(traces, paths):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, scen in zip(axes, ["gradual_puea", "power_drift"]):
        d = _one(traces, scen)
        d = d[d["system"] == "Baseline-XGBoost"]
        for lab, c, name in [(0, PU_COLOR, "Legitimate PU slot"), (1, ATT_COLOR, "Attacker slot")]:
            x = d[d["label"] == lab]
            if len(x):
                ax.scatter(x["slot"], x["classifier_confidence_pu"], s=8, color=c, alpha=0.6, lw=0, label=name)
        ax.axhline(0.5, color=INK, lw=1, ls="--", label="Decision threshold")
        ax.set_title(f"XGBoost P(PU) — {scen}")
        ax.set_xlabel("Slot")
    axes[0].set_ylabel("XGBoost confidence that slot is PU")
    axes[0].legend(loc="lower left", fontsize=8)
    _save(fig, paths, "adaptive_xgb_confidence.png")


def fig_profile_drift(traces, paths):
    systems = ["Naive-Adaptive", "Physically-Anchored-Adaptive", "XGB+StaticProfile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, scen in zip(axes, ["combined_drift", "poisoning"]):
        d = _one(traces, scen)
        _plot_systems(ax, d, "adaptation_error", systems)
        ax.set_title(f"Profile error vs true PU distribution — {scen}")
        ax.set_xlabel("Slot")
        ax.set_ylabel("Adaptation error (Mahalanobis)")
    axes[0].legend(fontsize=8)
    _save(fig, paths, "adaptive_profile_drift.png")


def fig_poisoning(traces, paths):
    systems = ["Naive-Adaptive", "Anchored-Update-Only", "Physically-Anchored-Adaptive", "XGB+StaticProfile"]
    scens = ["poisoning", "near_pu_poisoning"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    for j, scen in enumerate(scens):
        d = _one(traces, scen)
        for s in systems:
            x = d[(d["system"] == s) & (d["attack_phase"] == "poison") & (d["label"] == 1)]
            c, ls = SYSTEM_STYLE[s]
            axes[0, j].plot(x["slot"], x["attack_progress"], color=c, ls=ls, label=s)
        axes[0, j].set_title(f"Attacker poisoning progress — {scen}")
        axes[0, j].set_ylabel("Progress (0 = PU mimic, 1 = target)")
        _plot_systems(axes[1, j], d, "profile_contamination", systems[:3])
        axes[1, j].set_title("Profile contamination (attacker share of EWMA mean)")
        axes[1, j].set_xlabel("Slot")
        axes[1, j].set_ylabel("Contamination")
    axes[0, 0].legend(fontsize=8)
    _save(fig, paths, "adaptive_poisoning_progression.png")


def fig_attack_detection(traces, cfg, paths):
    w = int(cfg["evaluation"]["rolling_window"]) // 3
    scens = ["gradual_puea", "poisoning", "near_pu_poisoning"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, scen in zip(axes, scens):
        d = _one(traces, scen)
        _plot_systems(ax, d, "attack_detected", CORE, window=w, attack_only=True)
        ax.set_title(f"Rolling recall ({w} attacker slots) — {scen}")
        ax.set_xlabel("Slot")
    axes[0].set_ylabel("Fraction of attacker slots detected")
    axes[0].legend(fontsize=8)
    _save(fig, paths, "adaptive_attack_detection_over_time.png")


def fig_drift_adaptation(traces, cfg, paths):
    w = int(cfg["evaluation"]["rolling_window"])
    scens = ["power_drift", "shadowing_drift", "combined_drift"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, scen in zip(axes, scens):
        d = _one(traces, scen)
        _plot_systems(ax, d, "attack_detected", CORE, window=w, legit_only=True)
        sc = cfg["scenarios"][scen]["drift"]
        ax.axvspan(min(v["start"] for v in sc.values()), max(v["end"] for v in sc.values()),
                   color=GRID, alpha=0.5, lw=0, label="Drift ramp")
        ax.set_title(f"Rolling false-alarm rate — {scen}")
        ax.set_xlabel("Slot")
    axes[0].set_ylabel(f"FPR (rolling {w} legit slots)")
    axes[0].legend(fontsize=8)
    _save(fig, paths, "adaptive_drift_adaptation.png")


def fig_comparison(summary, cfg, paths):
    legit = [k for k, v in cfg["scenarios"].items() if v["group"] == "legitimate"]
    attack = [k for k, v in cfg["scenarios"].items() if v["group"] == "attack"]
    systems = [sy for sy in ALL_STYLE if sy in set(summary["system"])]
    regimes = list(summary["regime"].unique())
    fig, axes = plt.subplots(len(regimes), 2, figsize=(15, 5.2 * len(regimes)),
                             gridspec_kw={"width_ratios": [len(legit), len(attack)]}, squeeze=False)
    for i, regime in enumerate(regimes):
        s = summary[summary["regime"] == regime]
        for j, (scens, metric, title, cmap) in enumerate([
            (legit, "fpr", "False-positive rate (lower is better)", "Blues"),
            (attack, "f1", "F1 (higher is better)", "Blues"),
        ]):
            mat = np.array([[s[(s["system"] == sy) & (s["scenario"] == sc)][metric].mean() for sc in scens]
                            for sy in systems])
            ax = axes[i, j]
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.grid(False)
            for (r, c), v in np.ndenumerate(mat):
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 else INK)
            ax.set_xticks(range(len(scens)))
            ax.set_xticklabels(scens, rotation=30, ha="right")
            ax.set_yticks(range(len(systems)))
            ax.set_yticklabels(systems if j == 0 else [])
            ax.set_title(f"{title} — regime: {regime}")
    fig.tight_layout()
    _save(fig, paths, "adaptive_system_comparison.png")


def fig_sequential(traces, cfg, paths):
    """Near-PU poisoning, seed 0, every regime: contamination (profile systems) and rolling recall."""
    w = int(cfg["evaluation"]["rolling_window"]) // 3
    regimes = list(traces["regime"].unique())
    prof = ["Physically-Anchored-Adaptive", "Sequential-Anchored+Rollback", "Sequential-Anchored-Full"]
    dec = ["Physically-Anchored-Adaptive", "Sequential-Anchored-Full", "Physical-Only", "Physical-Only+Sequential"]
    fig, axes = plt.subplots(len(regimes), 2, figsize=(13, 3.4 * len(regimes)), squeeze=False)
    for i, regime in enumerate(regimes):
        d = _one(traces, "near_pu_poisoning", regime=regime)
        if d.empty:
            continue
        _plot_systems(axes[i, 0], d, "profile_contamination", prof)
        axes[i, 0].set_title(f"Profile contamination — near_pu_poisoning ({regime})")
        axes[i, 0].set_ylabel("Attacker share of profile mean")
        _plot_systems(axes[i, 1], d, "attack_detected", dec, window=w, attack_only=True)
        axes[i, 1].set_title(f"Rolling recall ({w} attacker slots) — {regime}")
        axes[i, 1].set_ylabel("Attacker slots detected")
        for ax in axes[i]:
            ax.axvline(cfg["scenarios"]["near_pu_poisoning"]["attack"]["attack_start"], color=MUTED, lw=1, ls=":")
            ax.set_xlabel("Slot")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, paths, "adaptive_sequential.png")


def fig_confusion(traces, cfg, paths):
    attack = [k for k, v in cfg["scenarios"].items() if v["group"] == "attack"]
    regimes = list(traces["regime"].unique())
    fig, axes = plt.subplots(len(regimes), 4, figsize=(13, 3.2 * len(regimes)), squeeze=False)
    for i, regime in enumerate(regimes):
        d = traces[(traces["regime"] == regime) & traces["scenario"].isin(attack)]
        for j, s in enumerate(CORE):
            x = d[d["system"] == s]
            cm = confusion_matrix(x["label"], x["attack_detected"].astype(int), labels=[0, 1])
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues")
            ax.grid(False)
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else INK)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pred PU", "Pred PUEA"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["True PU", "True PUEA"] if j == 0 else [])
            ax.set_title(f"{s}\n({regime}, attack scenarios pooled)", fontsize=8)
    fig.tight_layout()
    _save(fig, paths, "adaptive_confusion_matrices.png")


def fig_roc(traces, cfg, paths):
    attack = [k for k, v in cfg["scenarios"].items() if v["group"] == "attack"]
    regimes = list(traces["regime"].unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        d = traces[(traces["regime"] == regime) & traces["scenario"].isin(attack)]
        for s in SYSTEM_STYLE:
            x = d[d["system"] == s]
            if x.empty:
                continue
            fpr, tpr, _ = roc_curve(x["label"], x["score"])
            c, ls = SYSTEM_STYLE[s]
            ax.plot(fpr, tpr, color=c, ls=ls, lw=1.8, label=s)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_title(f"ROC, attack scenarios pooled — {regime}")
        ax.legend(fontsize=7, loc="lower right")
    _save(fig, paths, "adaptive_roc_curves.png")


def fig_anchor_resolution(sweep, paths):
    if sweep is None or sweep.empty:
        return
    sigmas = sorted(sweep["shadow_sigma_db"].unique())
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.6 * len(sigmas), 3.6), sharey=True)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    styles = [":", "--", "-"]
    for ax, sig in zip(np.atleast_1d(axes), sigmas):
        d = sweep[sweep["shadow_sigma_db"] == sig]
        for k, (n, g) in enumerate(d.groupby("n_sus")):
            m = g.groupby("distance")["attacker_rejection_rate"].agg(["mean", "std"])
            ax.errorbar(m.index, m["mean"], yerr=m["std"], color=colors[k % 3], ls=styles[k % 3],
                        marker="o", ms=4, capsize=2, label=f"{n} SUs")
        ax.set_xscale("log")
        ax.set_title(f"Shadowing σ = {sig:g} dB")
        ax.set_xlabel("Attacker distance from PU (m, log)")
    np.atleast_1d(axes)[0].set_ylabel("Attacker slots rejected by physical check")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _save(fig, paths, "adaptive_anchor_resolution.png")


def make_all(traces, summary, sweep, cfg, paths):
    _style()
    fig_geometry(traces, cfg, paths)
    fig_physical_residual(traces, paths)
    fig_xgb_confidence(traces, paths)
    fig_profile_drift(traces, paths)
    fig_poisoning(traces, paths)
    fig_attack_detection(traces, cfg, paths)
    fig_drift_adaptation(traces, cfg, paths)
    fig_comparison(summary, cfg, paths)
    fig_sequential(traces, cfg, paths)
    fig_confusion(traces, cfg, paths)
    fig_roc(traces, cfg, paths)
    fig_anchor_resolution(sweep, paths)
