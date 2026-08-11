"""
Stage 4 — Unified Test-Set Evaluation for all PUEA Detection Models.
Evaluates KNN, SVM, ANN, Random Forest, XGBoost, and the final ensemble once
on the untouched test set; writes comparison tables and figure artifacts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from utils import load_object, get_project_paths
from preprocessing import load_and_preprocess_data

# Display order for tables / plots
MODEL_SPECS = [
    # (display_name, artifact_filename, feature_mode)
    ("KNN", "knn.pkl", "scaled"),
    ("SVM", "svm.pkl", "scaled"),
    ("ANN", "ann.pkl", "scaled"),
    ("Random Forest", "random_forest.pkl", "raw"),
    ("XGBoost", "xgboost.pkl", "raw"),
    ("Ensemble (Stacking)", "voting_ensemble.pkl", "raw"),
]


def false_alarm_rate(y_true, y_pred):
    """FAR = FP / (FP + TN). Returns 0.0 if denominator is zero."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def evaluate_model(name, model, X_test, y_test):
    """Compute classification metrics and ROC curve data for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "False_Alarm_Rate": false_alarm_rate(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics, fpr, tpr, cm


def plot_roc_curves(roc_data, figures_dir):
    """Overlay ROC curves for all models on a single figure."""
    plt.figure(figsize=(9, 7))
    for name, fpr, tpr, auc in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — PUEA Detection Models")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_dir, "roc_curves_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved {out}")


def plot_confusion_matrices(cm_data, figures_dir):
    """Save one confusion-matrix heatmap per model."""
    for name, cm in cm_data:
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign (0)", "PUEA (1)"],
            yticklabels=["Benign (0)", "PUEA (1)"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        out = os.path.join(figures_dir, f"confusion_matrix_{safe}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[evaluate] Saved {out}")


def plot_accuracy_f1_bars(df, figures_dir):
    """Grouped bar chart comparing Accuracy and F1 across models."""
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, df["Accuracy"], width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, df["F1"], width, label="F1-Score", color="#55A868")

    ax.set_ylabel("Score")
    ax.set_title("Accuracy & F1 Comparison — PUEA Detection Models")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylim(0.80, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    out = os.path.join(figures_dir, "accuracy_f1_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Saved {out}")


def write_comparison_report(df, reports_dir):
    """Write markdown comparison table to reports/comparison_report.md."""
    out = os.path.join(reports_dir, "comparison_report.md")
    lines = [
        "# PUEA Detection — Model Comparison Report",
        "",
        "All metrics computed **once** on the untouched test set (20,000 samples).",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | False Alarm Rate | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['Model']} "
            f"| {row['Accuracy']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['F1']:.4f} "
            f"| {row['False_Alarm_Rate']:.4f} "
            f"| {row['ROC_AUC']:.4f} |"
        )

    best_acc = df.loc[df["Accuracy"].idxmax()]
    best_auc = df.loc[df["ROC_AUC"].idxmax()]
    best_f1 = df.loc[df["F1"].idxmax()]
    lowest_far = df.loc[df["False_Alarm_Rate"].idxmin()]

    lines.extend(
        [
            "",
            "## Highlights",
            "",
            f"- **Highest Accuracy:** {best_acc['Model']} ({best_acc['Accuracy']:.4f})",
            f"- **Highest F1:** {best_f1['Model']} ({best_f1['F1']:.4f})",
            f"- **Highest ROC-AUC:** {best_auc['Model']} ({best_auc['ROC_AUC']:.4f})",
            f"- **Lowest False Alarm Rate:** {lowest_far['Model']} ({lowest_far['False_Alarm_Rate']:.4f})",
            "",
            "## Notes",
            "",
            "- KNN, SVM, and ANN were evaluated on **scaled** features.",
            "- Random Forest, XGBoost, and the Ensemble were evaluated on **raw** features.",
            "- The Ensemble is a StackingClassifier (RF + XGBoost → LogisticRegression),",
            "  selected via 5-fold CV accuracy on the training set only.",
            "",
        ]
    )

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[evaluate] Saved {out}")


def main():
    paths = get_project_paths()
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    print("Loading preprocessed test data...")
    (
        X_train_raw,
        X_test_raw,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    ) = load_and_preprocess_data(save_scaler=False)

    rows = []
    roc_data = []
    cm_data = []

    print("\n" + "=" * 60)
    print("Evaluating all 6 models on untouched test set")
    print("=" * 60)

    for name, filename, mode in MODEL_SPECS:
        print(f"\n--- {name} ({filename}, {mode} features) ---")
        model = load_object(filename)
        X_test = X_test_scaled if mode == "scaled" else X_test_raw

        metrics, fpr, tpr, cm = evaluate_model(name, model, X_test, y_test)
        rows.append(metrics)
        roc_data.append((name, fpr, tpr, metrics["ROC_AUC"]))
        cm_data.append((name, cm))

        print(
            f"  Acc={metrics['Accuracy']:.4f}  "
            f"Prec={metrics['Precision']:.4f}  "
            f"Rec={metrics['Recall']:.4f}  "
            f"F1={metrics['F1']:.4f}  "
            f"FAR={metrics['False_Alarm_Rate']:.4f}  "
            f"AUC={metrics['ROC_AUC']:.4f}"
        )

    df = pd.DataFrame(rows)

    # CSV summary
    csv_path = os.path.join(reports_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[evaluate] Saved {csv_path}")

    # Markdown report
    write_comparison_report(df, reports_dir)

    # Figures
    print("\nGenerating figures...")
    plot_roc_curves(roc_data, figures_dir)
    plot_confusion_matrices(cm_data, figures_dir)
    plot_accuracy_f1_bars(df, figures_dir)

    print("\n" + "=" * 60)
    print("Stage 4 Evaluation Complete!")
    print(f"Reports : {reports_dir}")
    print(f"Figures : {figures_dir}")
    print("=" * 60)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
