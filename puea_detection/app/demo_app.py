"""
Stage 6 — Streamlit Demo App for PUEA Detection.

Run from the project root:
    streamlit run app/demo_app.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Allow importing project modules from src/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    LABEL_COLUMN,
    get_project_paths,
)

# Slider bounds derived from training-set min/max (rounded for UI)
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
    ("ANN", "ann.pkl", "scaled"),
    ("Random Forest", "random_forest.pkl", "raw"),
    ("XGBoost", "xgboost.pkl", "raw"),
    ("Ensemble (Stacking)", "voting_ensemble.pkl", "raw"),
]


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    """Load scaler + all six classifiers once per session."""
    import joblib

    paths = get_project_paths()
    models_dir = paths["models"]
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    models = {}
    for name, filename, mode in MODEL_SPECS:
        models[name] = {
            "model": joblib.load(os.path.join(models_dir, filename)),
            "mode": mode,
        }
    return scaler, models, paths


@st.cache_data(show_spinner=False)
def load_test_dataframe():
    paths = get_project_paths()
    test_path = os.path.join(paths["data"], "puea_dataset_test.csv")
    return pd.read_csv(test_path)


@st.cache_data(show_spinner=False)
def load_metrics_table():
    paths = get_project_paths()
    csv_path = os.path.join(paths["reports"], "metrics_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def features_from_widgets(defaults: dict | None = None) -> pd.DataFrame:
    """Render feature inputs in a 2-column grid and return a 1-row DataFrame."""
    defaults = defaults or {}
    # Sync widget state when a test sample is loaded
    for feat in FEATURE_COLUMNS:
        if feat in defaults:
            st.session_state[f"feat_{feat}"] = float(defaults[feat])

    values = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURE_COLUMNS):
        lo, hi, med = FEATURE_BOUNDS[feat]
        key = f"feat_{feat}"
        if key not in st.session_state:
            st.session_state[key] = float(med)
        # Clamp any out-of-range session value
        st.session_state[key] = float(np.clip(st.session_state[key], lo, hi))
        step = (hi - lo) / 200.0
        if feat == "Frequency_Hz":
            step = 1e4
        with cols[i % 2]:
            values[feat] = st.number_input(
                f"{feat}",
                min_value=float(lo),
                max_value=float(hi),
                step=float(step),
                help=FEATURE_DESCRIPTIONS.get(feat, feat),
                format="%.4f" if feat != "Frequency_Hz" else "%.1f",
                key=key,
            )
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def predict_all(models, scaler, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Run every model on one sample; return name / pred / proba table."""
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_COLUMNS)
    rows = []
    for name, bundle in models.items():
        X = X_scaled if bundle["mode"] == "scaled" else X_raw
        model = bundle["model"]
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        rows.append(
            {
                "Model": name,
                "Prediction": "PUEA" if pred == 1 else "Benign",
                "PUEA Probability": proba,
            }
        )
    return pd.DataFrame(rows)


def shap_waterfall_figure(model, X_raw: pd.DataFrame, title: str):
    """Single-sample TreeExplainer waterfall (XGBoost / RF)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_raw, check_additivity=False)
    if isinstance(sv, list):
        values = np.asarray(sv[-1])[0]
    else:
        values = np.asarray(sv)
        if values.ndim == 3:
            values = values[0, :, -1]
        else:
            values = values[0]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base_arr = np.asarray(base).ravel()
        base = float(base_arr[-1])
    else:
        base = float(base)

    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=X_raw.values[0],
        feature_names=FEATURE_COLUMNS,
    )
    fig = plt.figure()
    shap.plots.waterfall(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title(title)
    plt.tight_layout()
    return fig


def page_predict(scaler, models):
    st.subheader("Live PUEA Prediction")
    st.caption(
        "Enter spectrum / location features or load a held-out test sample. "
        "Baselines use scaled features; tree models and the ensemble use raw features."
    )

    test_df = load_test_dataframe()
    mode = st.radio(
        "Input source",
        ["Manual entry", "Random test sample", "Test sample by index"],
        horizontal=True,
    )

    defaults = None
    true_label = None
    sample_idx = None

    if mode == "Random test sample":
        if st.button("Draw random sample", type="secondary"):
            st.session_state["sample_idx"] = int(
                np.random.randint(0, len(test_df))
            )
        sample_idx = st.session_state.get("sample_idx", 0)
        row = test_df.iloc[sample_idx]
        defaults = {c: float(row[c]) for c in FEATURE_COLUMNS}
        true_label = int(row[LABEL_COLUMN])
        st.info(f"Loaded test index **{sample_idx}** (true label = {true_label} → {'PUEA' if true_label else 'Benign'})")
    elif mode == "Test sample by index":
        sample_idx = st.number_input(
            "Test row index",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            step=1,
        )
        row = test_df.iloc[int(sample_idx)]
        defaults = {c: float(row[c]) for c in FEATURE_COLUMNS}
        true_label = int(row[LABEL_COLUMN])
        st.info(f"True label = {true_label} → {'PUEA' if true_label else 'Benign'}")

    X_raw = features_from_widgets(defaults)

    primary = st.selectbox(
        "Primary model for decision banner",
        [name for name, _, _ in MODEL_SPECS],
        index=5,  # Ensemble default
    )

    if st.button("Run prediction", type="primary"):
        results = predict_all(models, scaler, X_raw)
        st.session_state["last_results"] = results
        st.session_state["last_X"] = X_raw
        st.session_state["last_true"] = true_label
        st.session_state["last_primary"] = primary

    if "last_results" not in st.session_state:
        st.warning("Set features, then click **Run prediction**.")
        return

    results = st.session_state["last_results"]
    X_raw = st.session_state["last_X"]
    true_label = st.session_state.get("last_true")
    primary = st.session_state.get("last_primary", primary)

    primary_row = results.loc[results["Model"] == primary].iloc[0]
    pred_label = primary_row["Prediction"]
    proba = float(primary_row["PUEA Probability"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary model", primary)
    c2.metric("Decision", pred_label)
    c3.metric("PUEA probability", f"{proba:.3f}")

    if true_label is not None:
        match = (pred_label == "PUEA") == bool(true_label)
        if match:
            st.success("Primary decision matches the true test label.")
        else:
            st.error("Primary decision differs from the true test label.")

    st.markdown("#### All-model votes on this sample")
    chart_df = results.set_index("Model")[["PUEA Probability"]]
    st.bar_chart(chart_df, height=280)
    st.dataframe(
        results.style.format({"PUEA Probability": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Local SHAP explanation (XGBoost)")
    st.caption(
        "TreeExplainer on the strongest single model — also the dominant stacking base "
        "(meta-coefficient ≫ RF)."
    )
    try:
        fig = shap_waterfall_figure(
            models["XGBoost"]["model"],
            X_raw,
            "Local SHAP Waterfall — XGBoost",
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render SHAP waterfall: {exc}")


def page_leaderboard():
    st.subheader("Test-set Model Leaderboard")
    st.caption("Metrics from Stage 4 — evaluated once on the untouched 20,000-row test set.")
    metrics = load_metrics_table()
    if metrics is None:
        st.warning("metrics_summary.csv not found. Run `src/evaluate.py` first.")
        return

    display = metrics.copy()
    for col in display.columns:
        if col != "Model":
            display[col] = display[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    paths = get_project_paths()
    roc = os.path.join(paths["figures"], "roc_curves_comparison.png")
    bars = os.path.join(paths["figures"], "accuracy_f1_comparison.png")
    c1, c2 = st.columns(2)
    if os.path.exists(roc):
        c1.image(roc, caption="ROC curves (all models)", use_container_width=True)
    if os.path.exists(bars):
        c2.image(bars, caption="Accuracy & F1 comparison", use_container_width=True)


def page_explainability():
    st.subheader("Global Explainability Gallery")
    st.caption("SHAP artifacts from Stage 5 (`src/explain.py`).")
    paths = get_project_paths()
    figures = paths["figures"]

    report = os.path.join(paths["reports"], "explainability_report.md")
    if os.path.exists(report):
        with open(report, encoding="utf-8") as f:
            st.markdown(f.read())

    st.markdown("#### Summary beeswarm plots")
    for label, fname in [
        ("XGBoost", "shap_summary_xgboost.png"),
        ("Random Forest", "shap_summary_random_forest.png"),
        ("Ensemble (Stacking)", "shap_summary_ensemble_stacking.png"),
    ]:
        path = os.path.join(figures, fname)
        if os.path.exists(path):
            st.image(path, caption=f"SHAP summary — {label}", use_container_width=True)

    st.markdown("#### Mean |SHAP| bar plots")
    cols = st.columns(3)
    for col, (label, fname) in zip(
        cols,
        [
            ("XGBoost", "shap_bar_xgboost.png"),
            ("Random Forest", "shap_bar_random_forest.png"),
            ("Ensemble", "shap_bar_ensemble_stacking.png"),
        ],
    ):
        path = os.path.join(figures, fname)
        if os.path.exists(path):
            col.image(path, caption=label, use_container_width=True)


def main():
    st.set_page_config(
        page_title="PUEA Detection Demo",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("PUEA Detection — Explainable Ensemble Demo")
    st.markdown(
        "Cognitive radio **Primary User Emulation Attack** detector using an "
        "explainable stacking ensemble (Random Forest + XGBoost → LogisticRegression)."
    )

    try:
        scaler, models, paths = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"Missing model artifact: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Page",
            ["Predict", "Leaderboard", "Explainability"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Proposed model**")
        st.write("StackingClassifier (RF + XGBoost)")
        st.markdown("**Best single model (test acc)**")
        st.write("XGBoost — 92.16%")
        st.markdown("**Ensemble (test)**")
        st.write("92.04% acc · 0.9704 AUC")
        st.caption(f"Models dir: `{paths['models']}`")

    if page == "Predict":
        page_predict(scaler, models)
    elif page == "Leaderboard":
        page_leaderboard()
    else:
        page_explainability()


if __name__ == "__main__":
    main()
