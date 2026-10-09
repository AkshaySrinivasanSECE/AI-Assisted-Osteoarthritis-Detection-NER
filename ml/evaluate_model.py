from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from model_utils import METRICS, TARGET


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "full_indicators_summary.csv"
RESULTS_DIR = ROOT_DIR / "docs" / "results"


def save_model_comparison(results_df, output_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(results_df))
    width = 0.18
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    for index, metric in enumerate(METRICS):
        ax.bar(
            [position + index * width for position in x],
            results_df[f"{metric} Mean"].tolist(),
            yerr=results_df[f"{metric} Std"].tolist(),
            capsize=3,
            width=width,
            label=f"{metric} (mean ± SD)",
            color=colors[index],
            alpha=0.9,
        )

    ax.set_title("Stratified 5-Fold Model Comparison (Real Held-Out Subjects)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([position + 1.5 * width for position in x])
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_distribution(training_df, output_path):
    counts = training_df.groupby([TARGET, "record_source"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=[0, 1], fill_value=0)
    counts = counts.reindex(columns=["real", "synthetic"], fill_value=0)
    counts.index = ["Healthy", "Knee OA"]

    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Training Records by Class and Provenance")
    ax.set_ylabel("Rows")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Record source")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, output_path, title_suffix):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=["Healthy", "Knee OA"],
    ).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Selected Model Confusion Matrix ({title_suffix})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return matrix


def save_feature_importance(importance_df, output_path, title_suffix):
    ordered = importance_df.sort_values("Importance Mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        ordered["Feature"],
        ordered["Importance Mean"],
        xerr=ordered["Importance Std"],
        color="#1d4ed8",
        alpha=0.9,
        capsize=4,
    )
    ax.axvline(0, color="#374151", linewidth=0.8)
    ax.set_title(f"Held-Out Permutation Importance ({title_suffix})")
    ax.set_xlabel("Mean decrease in held-out F1 when shuffled (mean ± SD)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_probability_calibration(y_true, baseline_probabilities, calibrated_probabilities, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect calibration")
    for label, probabilities, color in (
        ("Uncalibrated", baseline_probabilities, "#2563eb"),
        ("Sigmoid calibration candidate", calibrated_probabilities, "#dc2626"),
    ):
        observed, predicted = calibration_curve(
            y_true,
            probabilities,
            n_bins=5,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", linewidth=2, color=color, label=label)
    ax.set_title("Probability Calibration on Unified Active-Data OOF Rows")
    ax.set_xlabel("Mean predicted OA probability")
    ax.set_ylabel("Observed OA fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_all_plots(
    model_comparison,
    training_df,
    y_true,
    selected_predictions,
    selected_importance,
    baseline_probabilities,
    calibrated_probabilities,
    selected_variant,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_model_comparison(model_comparison, RESULTS_DIR / "model_comparison.png")
    save_class_distribution(training_df, RESULTS_DIR / "class_distribution.png")
    matrix = save_confusion_matrix(
        y_true,
        selected_predictions,
        RESULTS_DIR / "confusion_matrix.png",
        f"{selected_variant}; unified active-data OOF rows",
    )
    save_feature_importance(
        selected_importance,
        RESULTS_DIR / "feature_importance.png",
        f"{selected_variant}; unified active-data OOF rows",
    )
    save_probability_calibration(
        y_true,
        baseline_probabilities,
        calibrated_probabilities,
        RESULTS_DIR / "probability_calibration.png",
    )
    return matrix


def main():
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    oof = pd.read_csv(RESULTS_DIR / "selected_model_oof_predictions.csv")
    importance = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    training_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    selected_variant = oof["Selected Variant"].iloc[0]
    save_all_plots(
        comparison,
        training_df,
        oof["Actual Target"],
        oof["Selected Prediction"],
        importance,
        oof["Uncalibrated OA Probability"],
        oof["Calibrated OA Probability"],
        selected_variant,
    )
    print(f"Regenerated evaluation plots in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
