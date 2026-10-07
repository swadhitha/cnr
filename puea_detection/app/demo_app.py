"""
Streamlit demo for the PUEA detection project.

Run from the repository root:
    streamlit run puea_detection/app/demo_app.py

Pages (sidebar):
    Home                 - what the project is, in three cards
    Check a signal       - classify one signal from the test set, with the top reasons
    Compare models       - which classifier is best on the test set
    What the model uses  - which signal features matter most
    Attack replay        - the new method vs the old one on a simulated attack (adaptive_page.py)
    New method results   - headline numbers for the new method, plus limits (adaptive_page.py)
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in (SRC_DIR, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import FEATURE_COLUMNS, LABEL_COLUMN, get_project_paths  # noqa: E402

# Plain-language names shown in the UI (column names stay unchanged underneath).
PLAIN = {
    "RSS_dBm": "Signal strength",
    "SNR_dB": "Signal-to-noise",
    "Transmission_Power_dBm": "Transmit power",
    "X_Coordinate_km": "Position (x)",
    "Y_Coordinate_km": "Position (y)",
    "Channel_Occupancy_Time_s": "Time on channel",
    "Frequency_Hz": "Frequency",
    "RSS_Deviation": "Signal strength vs usual",
    "SNR_Deviation": "Signal-to-noise vs usual",
    "Distance_Mismatch_km": "Location mismatch",
    "SINR_dB": "Signal-to-interference",
}
UNITS = {
    "RSS_dBm": "dBm", "SNR_dB": "dB", "Transmission_Power_dBm": "dBm", "X_Coordinate_km": "km",
    "Y_Coordinate_km": "km", "Channel_Occupancy_Time_s": "s", "Frequency_Hz": "Hz", "RSS_Deviation": "dB",
    "SNR_Deviation": "dB", "Distance_Mismatch_km": "km", "SINR_dB": "dB",
}
FEATURE_BOUNDS = {
    "RSS_dBm": (-130.0, -39.0, -104.0),
    "SNR_dB": (-20.0, 60.0, -4.0),
    "Transmission_Power_dBm": (25.0, 59.0, 40.6),
    "X_Coordinate_km": (0.0, 100.0, 50.0),
    "Y_Coordinate_km": (0.0, 100.0, 41.2),
    "Channel_Occupancy_Time_s": (5.0, 500.0, 255.0),
    "Frequency_Hz": (4.698e8, 5.302e8, 5.0e8),
    "RSS_Deviation": (0.0, 28.5, 3.6),
    "SNR_Deviation": (0.0, 30.0, 4.0),
    "Distance_Mismatch_km": (0.0, 54.0, 5.8),
    "SINR_dB": (-20.0, 58.0, -7.3),
}
MODEL_SPECS = [
    ("KNN", "knn.pkl", "scaled"),
    ("SVM", "svm.pkl", "scaled"),
    ("Neural network", "ann.pkl", "scaled"),
    ("Random Forest", "random_forest.pkl", "raw"),
    ("XGBoost", "xgboost.pkl", "raw"),
    ("Ensemble", "voting_ensemble.pkl", "raw"),
]
METRIC_NAME = {"ANN": "Neural network", "Ensemble (Stacking)": "Ensemble"}
MAIN_MODEL = "XGBoost"  # best single model on the test set

ATTACK_C, GENUINE_C, MUTED, INK = "#eb6834", "#2a78d6", "#9b9a96", "#0b0b0b"

PAGES = {
    "Home": "What this project is",
    "Check a signal": "Is this signal real or an attack?",
    "Compare models": "Which detector works best?",
    "What the model uses": "Which signal features matter?",
    "Attack replay": "New method vs old, on a simulated attack",
    "New method results": "The numbers behind the new method",
}


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """Load the scaler and every classifier present on disk (some large ones are git-ignored)."""
    import joblib

    paths = get_project_paths()
    scaler = joblib.load(os.path.join(paths["models"], "scaler.pkl"))
    models, missing = {}, []
    for name, filename, mode in MODEL_SPECS:
        path = os.path.join(paths["models"], filename)
        if os.path.exists(path):
            models[name] = {"model": joblib.load(path), "mode": mode}
        else:
            missing.append(name)
    return scaler, models, paths, missing


@st.cache_data(show_spinner=False)
def load_test_dataframe():
    return pd.read_csv(os.path.join(get_project_paths()["data"], "puea_dataset_test.csv"))


@st.cache_data(show_spinner=False)
def load_csv(name):
    path = os.path.join(get_project_paths()["reports"], name)
    return pd.read_csv(path) if os.path.exists(path) else None


def predict_all(models, scaler, X_raw: pd.DataFrame) -> pd.DataFrame:
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_COLUMNS)
    rows = []
    for name, bundle in models.items():
        X = X_scaled if bundle["mode"] == "scaled" else X_raw
        p = float(bundle["model"].predict_proba(X)[0, 1])
        rows.append({"Model": name, "Verdict": "Attack" if p >= 0.5 else "Genuine", "Attack chance": p})
    return pd.DataFrame(rows)


def shap_contributions(model, X_raw: pd.DataFrame) -> pd.Series:
    """Per-feature push towards 'attack' (+) or 'genuine' (-) for one sample (XGBoost SHAP values)."""
    import shap

    sv = shap.TreeExplainer(model).shap_values(X_raw, check_additivity=False)
    values = np.asarray(sv[-1] if isinstance(sv, list) else sv)
    values = values[0, :, -1] if values.ndim == 3 else values.reshape(-1)
    return pd.Series(values, index=FEATURE_COLUMNS)


def bar_chart(labels, values, colors, xlabel, fmt="{:.0%}", xlim=None, height=None, ticks=True):
    """Simple horizontal bar chart with values printed at the bar ends."""
    fig, ax = plt.subplots(figsize=(7, height or 0.45 * len(labels) + 0.6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    lim = xlim or (0, max(values) * 1.18 if max(values) > 0 else 1)
    ax.set_xlim(*lim)
    for yi, v in zip(y, values):
        ax.text(v + (lim[1] - lim[0]) * 0.01, yi, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_xlabel(xlabel)
    if not ticks:
        ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ---------------------------------------------------------------- pages
def page_home():
    st.title("Detecting fake primary users in cognitive radio")
    st.markdown("**The threat:** an attacker pretends to be the licensed user (the *primary user*) so others "
                "leave the channel. This is a *Primary User Emulation Attack* (PUEA).")
    c = st.columns(3)
    with c[0].container(border=True, height=250):
        st.markdown("#### 1. Classify signals")
        st.markdown("Machine-learning models label each signal **genuine** or **attack**.")
        st.metric("Best accuracy (XGBoost)", "92.2%")
    with c[1].container(border=True, height=250):
        st.markdown("#### 2. Explain decisions")
        st.markdown("Shows **why** a signal was flagged.")
        st.metric("Most telling feature", "Location mismatch")
    with c[2].container(border=True, height=250):
        st.markdown("#### 3. New method")
        st.markdown("Catches attackers **standing right next to** the real user, who fool normal checks.")
        st.metric("Nearby attackers caught", "63%")
        st.caption("Old method: 12%")
    st.caption("Use the sidebar to open each part. Parts 1 and 2 use the original dataset; part 3 uses a "
               "simulated radio network.")


MODEL_ROLE = {
    "XGBoost": "Many small decision trees built one after another, each fixing the last one's mistakes",
    "Random Forest": "Many decision trees built independently; they vote",
    "Ensemble": "Combines Random Forest and XGBoost",
    "Neural network": "Layers of learned weights",
    "SVM": "Draws the best dividing boundary between genuine and attack",
    "KNN": "Copies the answer of the most similar training signals",
}


def fmt_value(f, v):
    """Human-readable feature value (frequency in MHz)."""
    if f == "Frequency_Hz":
        return f"{v / 1e6:.1f} MHz"
    return f"{v:,.1f} {UNITS[f]}"


def pick_sample(test_df, label=None):
    pool = test_df.index if label is None else test_df.index[test_df[LABEL_COLUMN] == label]
    st.session_state["sample_idx"] = int(np.random.choice(pool))
    st.session_state.pop("manual", None)


def why_chart(contrib, X):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:5]
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    ax.barh(range(len(top)), top.values, color=[ATTACK_C if v > 0 else GENUINE_C for v in top.values],
            height=0.6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{PLAIN[f]} = {fmt_value(f, X[f].iloc[0])}" for f in top.index], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    lim = np.abs(top.values).max() * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-lim * 0.55, lim * 0.55])
    ax.set_xticklabels(["points to GENUINE", "points to ATTACK"], fontsize=8)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig, top


def page_check_signal(scaler, models):
    st.title("Is this signal real or an attack?")
    test_df = load_test_dataframe()

    c = st.columns([1, 1, 1, 2])
    if "sample_idx" not in st.session_state and st.query_params.get("sample", "").isdigit():
        st.session_state["sample_idx"] = int(st.query_params["sample"])  # deep link, e.g. ?sample=123
    if c[0].button("Random signal", type="primary", use_container_width=True) \
            or "sample_idx" not in st.session_state:
        pick_sample(test_df)
    if c[1].button("Show an attack", use_container_width=True):
        pick_sample(test_df, 1)
    if c[2].button("Show a genuine signal", use_container_width=True):
        pick_sample(test_df, 0)
    idx = st.session_state["sample_idx"]
    row = test_df.iloc[idx]
    X = pd.DataFrame([{f: float(row[f]) for f in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    truth = int(row[LABEL_COLUMN])
    c[3].caption(f"Signal #{idx} from the test set (never seen during training).")

    with st.expander("Or enter the signal values yourself"):
        cols = st.columns(3)
        vals = {}
        for i, f in enumerate(FEATURE_COLUMNS):
            lo, hi, _ = FEATURE_BOUNDS[f]
            cur = float(np.clip(X[f].iloc[0], lo, hi))
            if f == "Frequency_Hz":  # entered in MHz, stored in Hz
                vals[f] = 1e6 * cols[i % 3].number_input("Frequency (MHz)", lo / 1e6, hi / 1e6, cur / 1e6,
                                                          step=0.1, format="%.1f", key="in_freq_mhz")
            else:
                vals[f] = cols[i % 3].number_input(f"{PLAIN[f]} ({UNITS[f]})", float(lo), float(hi), cur,
                                                   key=f"in_{f}")
        if st.button("Check these values"):
            st.session_state["manual"] = vals
    if "manual" in st.session_state:
        X = pd.DataFrame([st.session_state["manual"]], columns=FEATURE_COLUMNS)
        truth = None

    res = predict_all(models, scaler, X)
    main = res[res["Model"] == MAIN_MODEL].iloc[0]
    p = float(main["Attack chance"])
    is_attack = p >= 0.5

    with st.container(border=True):
        v = st.columns([2, 1, 1])
        if is_attack:
            v[0].markdown("## :red[Attack]\n**Fake primary user detected**")
        else:
            v[0].markdown("## :blue[Genuine]\n**Real primary user**")
        v[1].metric("Confidence", f"{max(p, 1 - p):.0%}")
        if truth is not None:
            v[2].metric("True answer", "Attack" if truth else "Genuine")
            v[2].markdown(":green[Model is correct]" if is_attack == bool(truth) else ":red[Model is wrong]")
        votes = int((res["Verdict"] == main["Verdict"]).sum())
        st.caption(f"Decided by {MAIN_MODEL}. {votes} of {len(res)} models agree.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Why?")
        try:
            fig, top = why_chart(shap_contributions(models[MAIN_MODEL]["model"], X), X)
            show(fig)
            st.caption(f"Biggest reason: **{PLAIN[top.index[0]]}** "
                       f"({'towards attack' if top.iloc[0] > 0 else 'towards genuine'}).")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Explanation unavailable ({exc}).")
    with right:
        st.markdown("#### How each model voted")
        t = res.assign(**{"Attack chance": res["Attack chance"].map("{:.0%}".format)})
        t = t.set_index("Model").reindex([m for m in MODEL_ROLE if m in set(t["Model"])]).reset_index()
        st.dataframe(t, hide_index=True, use_container_width=True)
        with st.expander("What does each model do?"):
            for m in t["Model"]:
                st.markdown(f"**{m}**: {MODEL_ROLE[m]}")


def page_compare():
    st.title("Which detector works best?")
    m = load_csv("metrics_summary.csv")
    if m is None:
        st.warning("Results not found. Run `src/evaluate.py`.")
        return
    m = m.assign(Model=m["Model"].replace(METRIC_NAME)).sort_values("Accuracy", ascending=False)
    best = m.iloc[0]
    c = st.columns(3)
    c[0].metric("Best model", best["Model"])
    c[1].metric("Correct decisions", f"{best['Accuracy']:.1%}")
    c[2].metric("False alarms", f"{best['False_Alarm_Rate']:.1%}", help="Genuine signals wrongly flagged as attacks")
    st.markdown("#### Accuracy on 20,000 unseen signals")
    colors = [GENUINE_C if n == best["Model"] else MUTED for n in m["Model"]]
    show(bar_chart(m["Model"].tolist(), m["Accuracy"].tolist(), colors, "Accuracy", "{:.1%}", xlim=(0, 1.08),
                   ticks=False, height=2.6))
    st.caption("All six are within 3.5 points of each other. XGBoost and the ensemble are effectively tied "
               "(0.2 points apart); XGBoost raises fewer false alarms (3.7% vs 4.9%).")
    with st.expander("Full table"):
        t = m.rename(columns={"False_Alarm_Rate": "False alarms", "ROC_AUC": "ROC-AUC",
                              "Recall": "Attacks caught", "Precision": "Alarms that were real"})
        for col in t.columns[1:]:
            t[col] = t[col].map("{:.1%}".format) if col != "ROC-AUC" else t[col].map("{:.3f}".format)
        st.dataframe(t, hide_index=True, use_container_width=True)
        roc = os.path.join(get_project_paths()["figures"], "roc_curves_comparison.png")
        if os.path.exists(roc):
            st.columns([3, 2])[0].image(roc, caption="ROC curves: the closer a curve hugs the top-left corner, "
                                        "the better the model (ANN = neural network)", use_container_width=True)


def page_features():
    st.title("Which signal features matter?")
    imp = load_csv("shap_feature_importance.csv")
    if imp is None:
        st.warning("Run `src/explain.py` first.")
        return
    s = imp.set_index(imp.columns[0])["XGBoost"].sort_values(ascending=False)
    share = s / s.sum()
    top = share.iloc[:6]
    st.metric("Most important feature", PLAIN[top.index[0]])
    st.caption(f"{top.iloc[0]:.0%} of the model's attention")
    colors = [GENUINE_C] + [MUTED] * (len(top) - 1)
    show(bar_chart([PLAIN[f] for f in top.index], top.tolist(), colors, "Share of the model's attention",
                   ticks=False))
    st.caption("Takeaway: an attacker can copy the real user's power, but not its **location**, so the "
               "location mismatch gives it away.")
    with st.expander("Detailed explanation plots"):
        fig_dir = get_project_paths()["figures"]
        cols = st.columns(2, gap="large")
        for col, (f, title, how) in zip(cols, [
            ("shap_summary_xgboost.png", "How each feature pushes decisions",
             "One dot per signal. Right = pushed towards attack, left = towards genuine. "
             "Red = high value, blue = low."),
            ("shap_dependence_xgboost_Distance_Mismatch_km.png", "Effect of location mismatch",
             "Below about 7 km the model leans genuine; above about 13 km it is strongly sure it is an attack."),
        ]):
            p = os.path.join(fig_dir, f)
            if os.path.exists(p):
                col.markdown(f"**{title}**")
                col.image(p, use_container_width=True)
                col.caption(how)


# ---------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="PUEA Detection", layout="wide",
                       initial_sidebar_state="expanded")
    with st.sidebar:
        st.markdown("## PUEA Detection")
        # Optional deep link, e.g. ?page=Attack%20replay (used for screenshots and demos)
        qp = st.query_params.get("page")
        start = list(PAGES).index(qp) if qp in PAGES else 0
        page = st.radio("Page", list(PAGES), index=start, label_visibility="collapsed")
        st.caption(PAGES[page])

    if page == "Home":
        page_home()
    elif page in ("Attack replay", "New method results"):
        import adaptive_page

        if page == "Attack replay":
            adaptive_page.render_replay()
        else:
            adaptive_page.render_results()
    else:
        try:
            scaler, models, _, missing = load_artifacts()
        except FileNotFoundError as exc:
            st.error(f"Model file missing: {exc}")
            st.stop()
        if page == "Check a signal":
            page_check_signal(scaler, models)
        elif page == "Compare models":
            page_compare()
            if missing:
                st.caption("Not loaded on this machine: " + ", ".join(missing) + " (large files, not in git).")
        else:
            page_features()


if __name__ == "__main__":
    main()
