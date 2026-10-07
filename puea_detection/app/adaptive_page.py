"""
New-method pages of the Streamlit demo (called from app/demo_app.py).

* render_replay()  - "Attack replay": old vs new method on one simulated run.
* render_results() - "New method results": averages over 5 runs, limits, theory check.

Both pages replay the per-slot traces written by src/run_adaptive_experiment.py
and the summaries written by src/evaluate_adaptive.py / src/check_theory.py.
Nothing is retrained here.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from evaluate_adaptive import metrics_for  # noqa: E402
from simulation import load_config, regime_config  # noqa: E402

REPORTS = os.path.join(ROOT, "reports", "adaptive")
TRACES = os.path.join(REPORTS, "traces")
SIM = os.path.join(ROOT, "data", "simulation")
CAL = os.path.join(REPORTS, "calibration")

OLD = "Physically-Anchored-Adaptive"
NEW_VERSIONS = {
    "Freeze + undo (best against poisoning)": "Sequential-Anchored+Rollback",
    "Soft update (best when the real user's signal changes)": "Sequential-Anchored-Full",
}
RIVAL = "Spatial-XGBoost"

ATTACKS = {
    "near_pu_poisoning": "Poisoning from right next to the real user",
    "slow_near_pu_poisoning": "Slow poisoning from next to the real user",
    "poisoning": "Poisoning from far away",
    "stealthy_puea": "Stealthy attacker while the real user's power changes",
    "power_matching_puea": "Attacker copying the real user's power",
    "power_drift": "No attack: the real user's power slowly changes",
}
SENSING = {
    "weak_anchor": "Weak (20 sensors, many obstacles)",
    "default": "Strong (40 sensors)",
    "correlated_shadowing": "Strong, obstacles change slowly",
}
OLD_C, NEW_C, RIVAL_C = "#9b9a96", "#2a78d6", "#4a3aa7"
ATT_C, INK, MUTED = "#eb6834", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------- loading
@st.cache_data(show_spinner="Loading run…")
def load_trace(regime, seed, scenario):
    return pd.read_csv(os.path.join(TRACES, regime, f"seed_{seed}", f"{scenario}.csv.gz"))


@st.cache_data
def load_geometry(regime, seed):
    return pd.read_csv(os.path.join(SIM, regime, f"seed_{seed}", "geometry.csv"))


@st.cache_data
def load_calibration(regime, seed):
    with open(os.path.join(CAL, f"{regime}_seed_{seed}.json")) as fh:
        return json.load(fh)


@st.cache_data
def load_summary():
    p = os.path.join(REPORTS, "metrics_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def available():
    out = {}
    for d in sorted(glob.glob(os.path.join(TRACES, "*", "seed_*"))):
        out.setdefault(d.split(os.sep)[-2], []).append(int(d.split("_")[-1]))
    return out


def run_metrics(trace, system, regime, scenario):
    cfg = regime_config(load_config(), regime)
    d = trace[trace["system"] == system].sort_values("slot")
    m = metrics_for(d, cfg["scenarios"][scenario], cfg)
    poison = (cfg["scenarios"][scenario].get("attack") or {}).get("type") == "poisoning"
    return {
        "caught": m["attack_phase_recall"] if poison else m["recall"],
        "poisoned": m.get("contamination_at_attack_start") if poison else m["profile_contamination_max"],
        "false_alarms": m["fpr"],
    }


def pct(x):
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ---------------------------------------------------------------- charts
def evidence_chart(d, threshold, pu_xy, area, geo):
    """Evidence vs alarm level with alarm shading (left) and a zoomed map with the estimated attacker (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(d["slot"], d["seq_stat"], color=INK, lw=1.2, label="Evidence of a second transmitter")
    if threshold:
        ax.axhline(threshold, color=ATT_C, ls="--", lw=1.2, label="Alarm level")
    alarm = d["seq_alarm"].astype(bool).to_numpy()
    sl = d["slot"].to_numpy()
    if alarm.any():
        edges = np.flatnonzero(np.diff(np.r_[0, alarm.astype(int), 0]))
        for k, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            ax.axvspan(sl[a], sl[b - 1], color=ATT_C, alpha=0.12, lw=0, label="Alarm on" if k == 0 else None)
    att = d[d["label"] == 1]
    if len(att):
        ax.scatter(att["slot"], np.full(len(att), 0.5), marker="|", s=30, color=ATT_C, alpha=0.5,
                   label="Attacker transmits")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3)
    ax.set_xlabel("Time (sensing slot)")
    ax.set_yticks([])
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)

    ax = axes[1]
    ax.scatter(geo["x"], geo["y"], marker="^", s=30, color=MUTED, label="Sensors")
    ax.scatter(*pu_xy, marker="*", s=280, color=NEW_C, edgecolor="white", lw=1.5, label="Real user", zorder=5)
    if len(att):
        ax.scatter(att["tx_x"], att["tx_y"], s=14, color=ATT_C, alpha=0.4, lw=0, label="Attacker (true)")
    fit = d[d["attr_x"].notna()] if "attr_x" in d else d.iloc[:0]
    if len(fit):
        ax.scatter(fit["attr_x"].iloc[-1], fit["attr_y"].iloc[-1], marker="X", s=150, color=INK,
                   label="Attacker (estimated)", zorder=6)
    far = len(att) and np.hypot(att["tx_x"] - pu_xy[0], att["tx_y"] - pu_xy[1]).max() > 25
    cx, cy, half = (area / 2, area / 2, area / 2) if far else (pu_xy[0], pu_xy[1], 20.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Map" if far else "Map (zoomed)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def comparison_panels(rows, measures):
    """One small bar panel per measure; rows = [(label, color, {measure: value})]."""
    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 2.4))
    axes = np.atleast_1d(axes)
    for j, (ax, (key, title, better)) in enumerate(zip(axes, measures)):
        vals = [r[2].get(key, np.nan) for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, [0 if pd.isna(v) else v for v in vals], color=[r[1] for r in rows], height=0.6)
        for yi, v in zip(y, vals):
            ax.text((0 if pd.isna(v) else v) + 0.02, yi, "n/a (no memory)" if pd.isna(v) else pct(v),
                    va="center", fontsize=10, color=INK if pd.notna(v) else MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if j == 0 else [])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_xticks([])
        ax.set_title(f"{title}\n({better})", fontsize=10)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- pages
def render_replay():
    st.title("New method vs old: replay an attack")
    st.markdown(
        "- Sensors listen to a real transmitter. An attacker tries to pass as it.\n"
        "- **Old method:** checks each moment on its own, so it misses attackers standing close by.\n"
        "- **New method:** adds up tiny location clues over time, raises an alarm and undoes the damage."
    )
    avail = available()
    if not avail:
        st.error("No experiment results found. Run `src/run_simulation.py`, then `src/run_adaptive_experiment.py`.")
        return

    # Optional deep links: ?attack=near_pu_poisoning&sensing=weak_anchor&run=5
    qp = st.query_params
    c = st.columns([2, 1.4, 0.6])
    keys = list(ATTACKS)
    scenario = c[0].selectbox("Attack", keys, format_func=ATTACKS.get, key="rp_attack",
                              index=keys.index(qp["attack"]) if qp.get("attack") in keys else 0)
    regimes = [r for r in SENSING if r in avail]
    regime = c[1].selectbox("Sensing conditions", regimes, format_func=SENSING.get, key="rp_regime",
                            index=regimes.index(qp["sensing"]) if qp.get("sensing") in regimes else 0)
    runs = avail[regime]
    run_q = int(qp["run"]) - 1 if qp.get("run", "").isdigit() else None
    seed = c[2].selectbox("Run", runs, format_func=lambda s: f"#{s + 1}", key="rp_seed",
                          index=runs.index(run_q) if run_q in runs else 0)
    with st.expander("Advanced: choose the new method's version"):
        new_label = st.radio("Version", list(NEW_VERSIONS), key="rp_version", label_visibility="collapsed")
    new = NEW_VERSIONS[new_label]

    trace = load_trace(regime, seed, scenario)
    is_attack = bool(trace["label"].any())
    m_old, m_new = run_metrics(trace, OLD, regime, scenario), run_metrics(trace, new, regime, scenario)

    cols = st.columns(2)
    for col, name, m in [(cols[0], "Old method", m_old), (cols[1], "New method", m_new)]:
        with col.container(border=True):
            st.markdown(f"#### {name}")
            k = st.columns(3 if is_attack else 1)
            i = 0
            if is_attack:
                k[0].metric("Attacks caught", pct(m["caught"]), help="Share of the attacker's transmissions flagged")
                k[1].metric("Detector poisoned", pct(m["poisoned"]),
                            help="How much of the detector's memory the attacker managed to corrupt")
                i = 2
            k[i].metric("False alarms", f"{m['false_alarms']:.1%}", help="Real-user signals wrongly flagged as attacks")

    if is_attack:
        gain = m_new["caught"] - m_old["caught"]
        if gain > 0.05:
            st.success(f"New method caught **{pct(m_new['caught'])}** of the attack vs "
                       f"**{pct(m_old['caught'])}**.")
        elif gain < -0.05:
            st.error(f"Here the new method does worse ({pct(m_new['caught'])} vs {pct(m_old['caught'])}) — "
                     "see *New method results*, Limits.")
        else:
            st.info(f"Both catch about the same ({pct(m_new['caught'])} vs {pct(m_old['caught'])}).")
    else:
        st.success(f"No attack, and no extra false alarms ({m_new['false_alarms']:.1%} vs "
                   f"{m_old['false_alarms']:.1%}): normal changes don't fool the new method.")

    st.markdown("#### How the new method spots it")
    d = trace[trace["system"] == new].sort_values("slot")
    cfg = load_config()
    show(evidence_chart(d, load_calibration(regime, seed).get("sequential", {}).get("threshold"),
                        cfg["geometry"]["pu_xy"], cfg["geometry"]["area_size"], load_geometry(regime, seed)))
    alarm = d[d["seq_alarm"].astype(bool)]["slot"]
    att = d[d["label"] == 1]["slot"]
    if len(alarm) and len(att):
        st.caption(f"Alarm raised **{int(alarm.iloc[0] - att.iloc[0])} slots** after the attacker started.")
    elif not len(alarm):
        st.caption("No alarm raised in this run.")

    s = load_summary()
    if s is not None and is_attack:
        col = "attack_phase_recall" if "poison" in scenario else "recall"
        pick = lambda sy: s[(s.regime == regime) & (s.scenario == scenario) & (s.system == sy)]  # noqa: E731
        a, b = pick(OLD), pick(new)
        if len(a) and len(b):
            st.caption(f"Average over 5 runs. Attacks caught: old {pct(a[col].iloc[0])}, "
                       f"new {pct(b[col].iloc[0])}.")


def render_results():
    st.title("New method: results")
    s = load_summary()
    if s is None:
        st.error("Run `src/evaluate_adaptive.py` first.")
        return
    new = NEW_VERSIONS["Freeze + undo (best against poisoning)"]
    soft = NEW_VERSIONS["Soft update (best when the real user's signal changes)"]

    def val(regime, scen, sys_, col):
        x = s[(s.regime == regime) & (s.scenario == scen) & (s.system == sys_)]
        return float(x[col].iloc[0]) if len(x) and pd.notna(x[col].iloc[0]) else np.nan

    st.markdown("#### Hardest case: weak sensing, attacker right next to the real user")
    caught = [val("weak_anchor", "near_pu_poisoning", sy, "attack_phase_recall") for sy in (OLD, new)]
    poisoned = [val("weak_anchor", "near_pu_poisoning", sy, "contamination_at_attack_start") for sy in (OLD, new)]
    fa = [val("weak_anchor", "power_drift", sy, "fpr") for sy in (OLD, new)]
    c = st.columns(3)
    for col, label, newv, oldv in [(c[0], "Attacks caught", pct(caught[1]), pct(caught[0])),
                                   (c[1], "Detector poisoned", pct(poisoned[1]), pct(poisoned[0])),
                                   (c[2], "False alarms (normal changes)", f"{fa[1]:.1%}", f"{fa[0]:.1%}")]:
        col.metric(label, newv)
        col.caption(f"Old method: {oldv}")
    st.caption("Average of 5 simulated runs.")

    rows = []
    for label, color, sy in [("Old method", OLD_C, OLD), ("New method", NEW_C, new),
                             ("Rival (Spatial-XGBoost)", RIVAL_C, RIVAL)]:
        rows.append((label, color, {
            "caught": val("weak_anchor", "near_pu_poisoning", sy, "attack_phase_recall"),
            "poisoned": val("weak_anchor", "near_pu_poisoning", sy, "contamination_at_attack_start"),
            "fa": val("weak_anchor", "power_drift", sy, "fpr"),
        }))
    show(comparison_panels(rows, [("caught", "Attacks caught", "higher is better"),
                                  ("poisoned", "Detector poisoned", "lower is better"),
                                  ("fa", "False alarms when power changes", "lower is better")]))

    c = st.columns(2)
    with c[0].container(border=True):
        st.markdown("**What works**\n"
                    "- Catches near-by attackers the old method misses\n"
                    "- Undoes poisoning once it raises an alarm\n"
                    "- No extra false alarms from normal signal changes\n"
                    "- Detection speed is predictable from the sensor layout")
    with c[1].container(border=True):
        st.markdown("**Limits**\n"
                    "- Simulation only (one attacker, fixed real user)\n"
                    "- The rival matches it on near-by attacks, but false-alarms when signals change\n"
                    "- 'Freeze' version struggles if the attacker stays while the real user's power changes\n"
                    "- The gain comes from the new location test, not the adaptive profile")

    tc, ts = os.path.join(REPORTS, "theory_check.csv"), os.path.join(REPORTS, "theory_check_sequential.csv")
    if os.path.exists(tc) and os.path.exists(ts):
        with st.expander("Does the theory match the simulation?"):
            a, b = pd.read_csv(tc), pd.read_csv(ts)
            b = b[~b["censored"] & np.isfinite(b["predicted_delay"])]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            axes[0].scatter(a["predicted_exact"], a["measured"], s=14, color=NEW_C, alpha=0.6, lw=0)
            axes[0].plot([0, 1], [0, 1], color=MUTED, ls=":")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Measured")
            err = (a["predicted_exact"] - a["measured"]).abs().mean()
            axes[0].set_title(f"Detection in one slot\n(average error {err:.0%})")
            hi = max(b["predicted_delay"].max(), b["measured_delay"].max()) * 1.2
            axes[1].scatter(b["predicted_delay"] + 1, b["measured_delay"] + 1, s=14, color=ATT_C, alpha=0.6, lw=0)
            axes[1].plot([1, hi], [1, hi], color=MUTED, ls=":")
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Predicted time to alarm (slots)")
            axes[1].set_ylabel("Measured")
            ratio = (b["measured_delay"] / b["predicted_delay"].clip(lower=1)).median()
            axes[1].set_title(f"Time to alarm\n(typical measured / predicted = {ratio:.1f})")
            for ax in axes:
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            fig.tight_layout()
            show(fig)
            st.caption("Points near the dotted line = the maths predicts what the simulation does.")

    with st.expander("All conditions (table)"):
        names = {OLD: "Old method", new: "New method (freeze + undo)", soft: "New method (soft update)",
                 RIVAL: "Rival (Spatial-XGBoost)"}
        out = []
        for regime in [r for r in SENSING if r in set(s.regime)]:
            for sy, label in names.items():
                out.append({
                    "Sensing": SENSING[regime], "Method": label,
                    "Near-by attack caught": pct(val(regime, "near_pu_poisoning", sy, "attack_phase_recall")),
                    "Detector poisoned": pct(val(regime, "near_pu_poisoning", sy, "contamination_at_attack_start")),
                    "Stealthy attack (F1)": f"{val(regime, 'stealthy_puea', sy, 'f1'):.2f}",
                    "False alarms (power change)": f"{val(regime, 'power_drift', sy, 'fpr'):.1%}",
                })
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
