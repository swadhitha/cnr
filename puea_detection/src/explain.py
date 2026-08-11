"""
Stage 5 — SHAP Explainability for PUEA Detection Models.
Produces global and local explanations for Random Forest, XGBoost,
and the stacking ensemble; writes figures and a feature-importance report.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    get_project_paths,
    load_object,
)
from preprocessing import load_and_preprocess_data

# Subsample sizes keep TreeExplainer / KernelExplainer tractable on 20k test rows
RF_EXPLAIN_N = 400          # RF TreeExplainer is O(trees×samples); keep modest
XGB_EXPLAIN_N = 1000        # XGBoost TreeExplainer is comparatively fast
ENSEMBLE_BACKGROUND_N = 40
ENSEMBLE_EXPLAIN_N = 60
KERNEL_NSAMPLES = 50
RANDOM_STATE = 42


def _safe_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _positive_class_explanation(explanation):
    """
    Normalize SHAP Explanation / array / list output to positive-class values.
    Returns (values[n, f], base_values_or_None, data_or_None).
    """
    # Older API: list [class0, class1]
    if isinstance(explanation, list):
        return np.asarray(explanation[-1]), None, None

    # Raw ndarray from explainer.shap_values(...)
    if isinstance(explanation, np.ndarray):
        values = explanation
        if values.ndim == 3:
            values = values[:, :, -1]
        return values, None, None

    # shap.Explanation
    values = np.asarray(explanation.values)
    base = getattr(explanation, "base_values", None)
    data = getattr(explanation, "data", None)

    if values.ndim == 3:
        values = values[:, :, -1]
        if base is not None:
            base = np.asarray(base)
            if base.ndim == 2:
                base = base[:, -1]
            elif base.ndim == 1 and base.shape[0] == 2:
                base = base[-1]
    return values, base, data


def _to_explanation(values, base_values, data, feature_names):
    """Build a shap.Explanation suitable for beeswarm / bar / waterfall plots."""
    if base_values is None:
        base_values = np.zeros(len(values))
    else:
        base_values = np.asarray(base_values)
        if base_values.ndim > 1:
            base_values = base_values.reshape(len(values), -1)[:, -1]
        if base_values.shape == ():
            base_values = np.full(len(values), float(base_values))
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=np.asarray(data) if data is not None else None,
        feature_names=list(feature_names),
    )


def subsample_frame(X, n, random_state=RANDOM_STATE):
    """Stratified-friendly random subsample of a DataFrame (or Series-aligned X)."""
    n = min(n, len(X))
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx].reset_index(drop=True), idx


def explain_tree_model(name, model, X_sample, figures_dir, y_sample=None, approximate=False):
    """
    TreeExplainer global + local plots for RF / XGBoost.
    Returns mean |SHAP| Series ranked descending.
    """
    print(f"\n--- TreeExplainer: {name} (n={len(X_sample)}, approximate={approximate}) ---")
    slug = _safe_slug(name)

    explainer = shap.TreeExplainer(model)
    # Prefer shap_values path: supports approximate=True for large RF forests
    raw_values = explainer.shap_values(
        X_sample, approximate=approximate, check_additivity=False
    )
    values, _, _ = _positive_class_explanation(raw_values)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base_arr = np.asarray(base)
        base = base_arr.ravel()[-1] if base_arr.size > 1 else float(base_arr.ravel()[0])
    explanation = _to_explanation(values, base, X_sample.values, FEATURE_COLUMNS)

    # Beeswarm (summary)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title(f"SHAP Summary — {name}")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"shap_summary_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved {out}")

    # Global mean |SHAP| bar
    plt.figure()
    shap.plots.bar(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"shap_bar_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved {out}")

    # Dependence plot for top feature
    mean_abs = np.abs(explanation.values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feat = FEATURE_COLUMNS[top_idx]
    plt.figure()
    shap.plots.scatter(explanation[:, top_feat], color=explanation, show=False)
    plt.title(f"SHAP Dependence — {name}: {top_feat}")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"shap_dependence_{slug}_{top_feat}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved {out}")

    # Local waterfall: one predicted PUEA and one predicted Benign (if labels available)
    _save_local_waterfalls(name, slug, model, explanation, X_sample, y_sample, figures_dir)

    importance = pd.Series(mean_abs, index=FEATURE_COLUMNS, name=name).sort_values(ascending=False)
    return importance


def _save_local_waterfalls(name, slug, model, explanation, X_sample, y_sample, figures_dir):
    """Save waterfall plots for representative PUEA / Benign instances."""
    proba = model.predict_proba(X_sample)[:, 1]
    pred = (proba >= 0.5).astype(int)

    picks = {}
    # Prefer correctly classified examples when labels are provided
    if y_sample is not None:
        y_arr = np.asarray(y_sample)
        for label, key in [(1, "puea"), (0, "benign")]:
            mask = (y_arr == label) & (pred == label)
            if mask.any():
                # Most confident correct prediction
                candidates = np.where(mask)[0]
                scores = proba[candidates] if label == 1 else (1 - proba[candidates])
                picks[key] = int(candidates[np.argmax(scores)])
    if "puea" not in picks and (pred == 1).any():
        picks["puea"] = int(np.argmax(proba))
    if "benign" not in picks and (pred == 0).any():
        picks["benign"] = int(np.argmin(proba))

    for key, idx in picks.items():
        plt.figure()
        shap.plots.waterfall(explanation[idx], max_display=len(FEATURE_COLUMNS), show=False)
        plt.title(f"Local SHAP Waterfall — {name} ({key}, idx={idx})")
        plt.tight_layout()
        out = os.path.join(figures_dir, f"shap_waterfall_{slug}_{key}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explain] Saved {out}")


def explain_ensemble(model, X_background, X_explain, figures_dir, y_explain=None):
    """
    KernelExplainer on StackingClassifier predict_proba (positive class).
    Also records meta-learner coefficients for RF vs XGBoost base probabilities.
    """
    print(
        f"\n--- KernelExplainer: Ensemble "
        f"(background={len(X_background)}, explain={len(X_explain)}) ---"
    )
    slug = "ensemble_stacking"

    def predict_puea_proba(X):
        X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        return model.predict_proba(X_df)[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_puea_proba, X_background.values)
        shap_values = explainer.shap_values(X_explain.values, nsamples=KERNEL_NSAMPLES, silent=True)

    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base_arr = np.asarray(base)
        base = float(base_arr.ravel()[-1])
    explanation = _to_explanation(values, base, X_explain.values, FEATURE_COLUMNS)

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title("SHAP Summary — Ensemble (Stacking)")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"shap_summary_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved {out}")

    plt.figure()
    shap.plots.bar(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title("SHAP Feature Importance — Ensemble (Stacking)")
    plt.tight_layout()
    out = os.path.join(figures_dir, f"shap_bar_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved {out}")

    _save_local_waterfalls(
        "Ensemble (Stacking)", slug, model, explanation, X_explain, y_explain, figures_dir
    )

    mean_abs = np.abs(explanation.values).mean(axis=0)
    importance = pd.Series(
        mean_abs, index=FEATURE_COLUMNS, name="Ensemble (Stacking)"
    ).sort_values(ascending=False)

    meta_info = _meta_learner_summary(model)
    return importance, meta_info


def _meta_learner_summary(stacking_model):
    """Extract LogisticRegression meta-learner weights over base estimator probabilities."""
    final = stacking_model.final_estimator_
    names = list(stacking_model.named_estimators_.keys())
    # Stacking with predict_proba passes 2 columns per estimator (or 1 if decision_function)
    coef = np.asarray(final.coef_).ravel()
    intercept = float(final.intercept_.ravel()[0])

    # Map coefficients: sklearn StackingClassifier with predict_proba uses class-1 probs
    # when stack_method='auto' for classifiers that support predict_proba → 1 col each.
    rows = []
    if len(coef) == len(names):
        for name, w in zip(names, coef):
            rows.append({"base_estimator": name, "meta_coefficient": float(w)})
    else:
        # Fallback: report raw coefficients
        for i, w in enumerate(coef):
            rows.append({"base_estimator": f"meta_feature_{i}", "meta_coefficient": float(w)})

    return {"intercept": intercept, "coefficients": rows}


def write_explainability_report(importance_df, meta_info, reports_dir):
    """Markdown report summarizing SHAP rankings and stacking meta-weights."""
    out = os.path.join(reports_dir, "explainability_report.md")
    lines = [
        "# PUEA Detection — Explainability Report (SHAP)",
        "",
        "Global feature attributions computed with SHAP on held-out test subsamples.",
        "",
        f"- **Random Forest:** `TreeExplainer` (approximate=True, n={RF_EXPLAIN_N}).",
        f"- **XGBoost:** `TreeExplainer` (n={XGB_EXPLAIN_N}).",
        "- **Ensemble (Stacking):** `KernelExplainer` on `predict_proba` "
        f"(background={ENSEMBLE_BACKGROUND_N}, explain={ENSEMBLE_EXPLAIN_N}, "
        f"nsamples={KERNEL_NSAMPLES}).",
        "",
        "## Mean |SHAP| Feature Rankings",
        "",
    ]

    # Ranked table per model
    for col in importance_df.columns:
        ranked = importance_df[col].sort_values(ascending=False)
        lines.append(f"### {col}")
        lines.append("")
        lines.append("| Rank | Feature | Description | Mean |SHAP| |")
        lines.append("|---:|---|---|---:|")
        for i, (feat, val) in enumerate(ranked.items(), start=1):
            desc = FEATURE_DESCRIPTIONS.get(feat, "")
            lines.append(f"| {i} | `{feat}` | {desc} | {val:.6f} |")
        lines.append("")

    # Consensus: average rank across models
    ranks = importance_df.rank(ascending=False)
    mean_rank = ranks.mean(axis=1).sort_values()
    lines.extend(
        [
            "## Consensus Ranking (mean rank across explained models)",
            "",
            "| Rank | Feature | Mean Rank |",
            "|---:|---|---:|",
        ]
    )
    for i, (feat, r) in enumerate(mean_rank.items(), start=1):
        lines.append(f"| {i} | `{feat}` | {r:.2f} |")
    lines.append("")

    if meta_info is not None:
        lines.extend(
            [
                "## Stacking Meta-Learner Weights",
                "",
                "LogisticRegression coefficients on stacked base-model outputs "
                "(positive → pushes toward PUEA class).",
                "",
                f"- Intercept: `{meta_info['intercept']:.6f}`",
                "",
                "| Base Estimator | Meta Coefficient |",
                "|---|---:|",
            ]
        )
        for row in meta_info["coefficients"]:
            lines.append(
                f"| `{row['base_estimator']}` | {row['meta_coefficient']:.6f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
            "Saved under `reports/figures/`:",
            "",
            "- `shap_summary_*.png` — beeswarm global attributions",
            "- `shap_bar_*.png` — mean |SHAP| importance",
            "- `shap_dependence_*.png` — dependence for top RF/XGB feature",
            "- `shap_waterfall_*_puea.png` / `*_benign.png` — local explanations",
            "",
        ]
    )

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[explain] Saved {out}")


def main():
    paths = get_project_paths()
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    print("Loading data and tree/ensemble models...")
    (
        X_train_raw,
        X_test_raw,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    ) = load_and_preprocess_data(save_scaler=False)

    rf = load_object("random_forest.pkl")
    xgb = load_object("xgboost.pkl")
    ensemble = load_object("voting_ensemble.pkl")

    X_rf, rf_idx = subsample_frame(X_test_raw, RF_EXPLAIN_N, random_state=RANDOM_STATE)
    y_rf = y_test.iloc[rf_idx].reset_index(drop=True)
    X_xgb, xgb_idx = subsample_frame(X_test_raw, XGB_EXPLAIN_N, random_state=RANDOM_STATE)
    y_xgb = y_test.iloc[xgb_idx].reset_index(drop=True)

    X_bg, _ = subsample_frame(X_test_raw, ENSEMBLE_BACKGROUND_N, random_state=RANDOM_STATE)
    X_ens, ens_idx = subsample_frame(
        X_test_raw, ENSEMBLE_EXPLAIN_N, random_state=RANDOM_STATE + 1
    )
    y_ens = y_test.iloc[ens_idx].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Stage 5 — SHAP Explainability")
    print("=" * 60)

    imp_rf = explain_tree_model(
        "Random Forest", rf, X_rf, figures_dir, y_rf, approximate=True
    )
    imp_xgb = explain_tree_model(
        "XGBoost", xgb, X_xgb, figures_dir, y_xgb, approximate=False
    )
    imp_ens, meta_info = explain_ensemble(ensemble, X_bg, X_ens, figures_dir, y_ens)

    importance_df = pd.concat([imp_rf, imp_xgb, imp_ens], axis=1)
    # Align to FEATURE_COLUMNS order for CSV, keep values
    importance_df = importance_df.reindex(FEATURE_COLUMNS)

    csv_path = os.path.join(reports_dir, "shap_feature_importance.csv")
    importance_df.to_csv(csv_path, float_format="%.8f")
    print(f"\n[explain] Saved {csv_path}")

    meta_csv = os.path.join(reports_dir, "stacking_meta_coefficients.csv")
    pd.DataFrame(meta_info["coefficients"]).to_csv(meta_csv, index=False, float_format="%.8f")
    print(f"[explain] Saved {meta_csv}")

    write_explainability_report(importance_df, meta_info, reports_dir)

    print("\n" + "=" * 60)
    print("Stage 5 Explainability Complete!")
    print(f"Reports : {reports_dir}")
    print(f"Figures : {figures_dir}")
    print("=" * 60)
    print("\nTop-5 features by mean |SHAP| (XGBoost):")
    print(imp_xgb.head(5).to_string())
    print("\nTop-5 features by mean |SHAP| (Ensemble):")
    print(imp_ens.head(5).to_string())
    print("\nStacking meta-learner coefficients:")
    for row in meta_info["coefficients"]:
        print(f"  {row['base_estimator']}: {row['meta_coefficient']:.6f}")


if __name__ == "__main__":
    main()
