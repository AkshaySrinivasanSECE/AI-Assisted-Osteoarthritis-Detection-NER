from pathlib import Path
import hashlib
import json
import platform

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from evaluate_model import RESULTS_DIR, save_all_plots
from model_utils import (
    CALIBRATION_FRACTION,
    CALIBRATION_METHOD,
    CATEGORICAL_FEATURES,
    FEATURE_CONFIGURATION,
    FEATURES,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
    SYNTHETIC_PER_CLASS,
    TARGET,
    build_models,
    cross_validate_on_unified_dataset,
    evaluate_selected_model,
    fit_final_calibrated_model,
    should_apply_calibration,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "full_indicators_summary.csv"
REAL_DATA_PATH = ROOT_DIR / "data" / "full_indicators_summary.original.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "oa_model.pkl"
FEATURES_PATH = MODELS_DIR / "features.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

EXPECTED_REAL_ROWS = 88
EXPECTED_SYNTHETIC_ROWS = 500
EXPECTED_ACTIVE_ROWS = 588


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return repr(value)


def model_configuration(estimator):
    return {
        "estimator_class": estimator.__class__.__name__,
        "parameters": {
            key: json_safe(value)
            for key, value in sorted(estimator.get_params(deep=True).items())
        },
    }


def load_and_validate_datasets():
    real_df = pd.read_csv(REAL_DATA_PATH, encoding="gb18030")
    active_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required_active_columns = FEATURES + [
        TARGET,
        "record_id",
        "record_source",
        "generation_method",
        "validation_eligible",
    ]

    if len(real_df) != EXPECTED_REAL_ROWS:
        raise ValueError(f"Expected 88 original subjects; found {len(real_df)}.")
    if len(active_df) != EXPECTED_ACTIVE_ROWS:
        raise ValueError(f"Expected 588 active training rows; found {len(active_df)}.")
    missing_columns = sorted(set(required_active_columns) - set(active_df.columns))
    if missing_columns:
        raise ValueError(f"Active dataset is missing required columns: {missing_columns}")
    if active_df[required_active_columns].isna().any().any():
        raise ValueError("Active dataset contains missing model or provenance values.")

    source_counts = active_df["record_source"].value_counts().to_dict()
    if source_counts != {"synthetic": EXPECTED_SYNTHETIC_ROWS, "real": EXPECTED_REAL_ROWS}:
        raise ValueError(f"Unexpected record-source counts: {source_counts}")
    synthetic_counts = (
        active_df[active_df["record_source"] == "synthetic"][TARGET]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    if synthetic_counts != {0: SYNTHETIC_PER_CLASS, 1: SYNTHETIC_PER_CLASS}:
        raise ValueError(f"Unexpected synthetic target counts: {synthetic_counts}")
    if not active_df["validation_eligible"].all():
        raise ValueError("Every active row must be marked as validation eligible.")

    active_real = active_df[active_df["record_source"] == "real"][FEATURES + [TARGET]]
    if not active_real.reset_index(drop=True).equals(
        real_df[FEATURES + [TARGET]].reset_index(drop=True)
    ):
        raise ValueError("Active real rows do not exactly match the preserved original dataset.")
    if set(real_df[TARGET].unique()) != {0, 1}:
        raise ValueError("Original target labels must be exactly 0 and 1.")
    if "number" not in real_df or real_df["number"].duplicated().any():
        raise ValueError("Original subject identifiers must exist and be unique.")

    return real_df, active_df


def save_csv(dataframe, filename):
    path = RESULTS_DIR / filename
    dataframe.to_csv(path, index=False, encoding="utf-8")
    return path


def build_metadata(
    real_df,
    active_df,
    models,
    comparison,
    selected_model_name,
    selected_variant,
    calibration_applied,
    calibration_summary,
    final_training,
    artifact_paths,
):
    selected_row = comparison.set_index("Model").loc[selected_model_name]
    return {
        "schema_version": 1,
        "random_seed": RANDOM_STATE,
        "features": FEATURES,
        "feature_configuration": {
            "name": FEATURE_CONFIGURATION,
            "features": FEATURES,
            "smotenc_categorical_features": CATEGORICAL_FEATURES,
            "vas_representation": (
                "Categorical for SMOTENC sampling; integer-valued numeric input to estimators."
            ),
            "womac_included": False,
        },
        "target": {
            "column": TARGET,
            "labels": {"0": "healthy", "1": "knee osteoarthritis"},
        },
        "data": {
            "original_observed_subjects": EXPECTED_REAL_ROWS,
            "synthetic_training_examples": EXPECTED_SYNTHETIC_ROWS,
            "active_training_rows": EXPECTED_ACTIVE_ROWS,
            "original_target_counts": {
                str(key): int(value)
                for key, value in real_df[TARGET].value_counts().sort_index().items()
            },
            "synthetic_target_counts": {
                str(key): int(value)
                for key, value in active_df[active_df["record_source"] == "synthetic"][TARGET]
                .value_counts()
                .sort_index()
                .items()
            },
            "original_dataset_sha256": file_sha256(REAL_DATA_PATH),
            "active_dataset_sha256": file_sha256(DATA_PATH),
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_seed": RANDOM_STATE,
            "outer_validation_subjects": "all active training rows",
            "outer_validation_synthetic_rows": EXPECTED_SYNTHETIC_ROWS,
            "synthetic_augmentation": (
                "None. The active table is modeled as one unified dataset; synthetic rows "
                "are included in training and validation folds exactly like observed rows."
            ),
            "metric_standard_deviation": "sample standard deviation across five folds (ddof=1)",
        },
        "candidate_models": {
            name: model_configuration(estimator) for name, estimator in models.items()
        },
        "model_selection": {
            "criterion": "highest mean F1 across the five outer unified active-data validation folds",
            "selected_model": selected_model_name,
            "selected_mean_f1": float(selected_row["F1 Mean"]),
            "selected_f1_standard_deviation": float(selected_row["F1 Std"]),
        },
        "probability_calibration": {
            "method_evaluated": CALIBRATION_METHOD,
            "calibration_fraction_of_each_outer_training_partition": CALIBRATION_FRACTION,
            "calibration_rows": "active rows drawn from the outer training partition",
            "calibration_synthetic_rows": "included in the active-row calibration split",
            "outer_evaluation_rows": "untouched active rows",
            "adoption_rule": (
                "Apply only if unified active-data out-of-fold Brier score and log loss both improve "
                "without reducing out-of-fold F1."
            ),
            "applied_to_saved_model": calibration_applied,
            "selected_variant": selected_variant,
            "comparison": calibration_summary.to_dict(orient="records"),
        },
        "permutation_importance": {
            "scoring": "F1",
            "repeats_per_fold": PERMUTATION_REPEATS,
            "evaluated_on": "each outer fold's untouched active validation rows",
            "synthetic_validation_rows": EXPECTED_SYNTHETIC_ROWS,
        },
        "final_training": final_training,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for name, path in artifact_paths.items()
            if path.exists()
        },
        "evidence_statement": (
            "The reported validation metrics are internal cross-validation results from the "
            "588-row active dataset, including 88 observed rows and 500 generated rows. "
            "Synthetic provenance is retained for transparency, but synthetic rows are modeled "
            "as part of the unified dataset. No external clinical validation has been performed."
        ),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    real_df, active_df = load_and_validate_datasets()
    X_real = real_df[FEATURES]
    y_real = real_df[TARGET].astype(int)
    X_active = active_df[FEATURES]
    y_active = active_df[TARGET].astype(int)
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison_result = cross_validate_on_unified_dataset(
        X_active,
        y_active,
        models,
        cv,
        record_ids=active_df["record_id"],
        record_sources=active_df["record_source"],
    )
    comparison = comparison_result["summary"]
    best_model_name = comparison.iloc[0]["Model"]
    best_estimator = models[best_model_name]
    baseline_predictions = comparison_result["oof_predictions"][best_model_name]
    baseline_probabilities = comparison_result["oof_probabilities"][best_model_name]

    selected_evaluation = evaluate_selected_model(
        X_active,
        y_active,
        best_estimator,
        cv,
        baseline_predictions,
        baseline_probabilities,
        record_sources=active_df["record_source"],
    )
    calibration_summary = selected_evaluation["calibration_summary"]
    calibration_applied = should_apply_calibration(calibration_summary)
    selected_variant = "Calibrated" if calibration_applied else "Uncalibrated"

    if calibration_applied:
        selected_predictions = selected_evaluation["calibrated_oof_predictions"]
        selected_probabilities = selected_evaluation["calibrated_oof_probabilities"]
        final_model, calibrated_training = fit_final_calibrated_model(
            X_active,
            y_active,
            best_estimator,
            record_sources=active_df["record_source"],
        )
        final_training = {
            "model_variant": "calibrated",
            **calibrated_training,
            "note": (
                "The estimator and calibration split are both fitted from the unified active "
                "dataset; provenance is retained only for audit counts."
            ),
        }
    else:
        selected_predictions = baseline_predictions
        selected_probabilities = baseline_probabilities
        final_model = clone(best_estimator).fit(X_active, y_active)
        final_training = {
            "model_variant": "uncalibrated",
            "real_training_rows": EXPECTED_REAL_ROWS,
            "synthetic_training_rows": EXPECTED_SYNTHETIC_ROWS,
            "total_training_rows": EXPECTED_ACTIVE_ROWS,
            "note": (
                "The final estimator is fitted on the unified active dataset after model "
                "selection, with observed and synthetic rows treated identically by the model."
            ),
        }

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)

    model_comparison_path = save_csv(comparison, "model_comparison.csv")
    fold_metrics_path = save_csv(comparison_result["fold_metrics"], "model_fold_metrics.csv")
    fold_assignments_path = save_csv(
        comparison_result["fold_assignments"], "validation_fold_assignments.csv"
    )
    calibration_folds_path = save_csv(
        selected_evaluation["calibration_folds"], "calibration_fold_metrics.csv"
    )
    calibration_summary_path = save_csv(calibration_summary, "calibration_comparison.csv")

    selected_importance = selected_evaluation["importance_summary"].query(
        "Variant == @selected_variant"
    ).drop(columns="Variant")
    selected_importance_folds = selected_evaluation["importance_folds"].query(
        "Variant == @selected_variant"
    ).drop(columns="Variant")
    importance_path = save_csv(selected_importance, "feature_importance.csv")
    importance_folds_path = save_csv(
        selected_importance_folds, "feature_importance_folds.csv"
    )

    oof = pd.DataFrame({
        "Record ID": active_df["record_id"],
        "Actual Target": y_active,
        "Record Source": active_df["record_source"],
        "Uncalibrated Prediction": baseline_predictions,
        "Uncalibrated OA Probability": baseline_probabilities,
        "Calibrated Prediction": selected_evaluation["calibrated_oof_predictions"],
        "Calibrated OA Probability": selected_evaluation["calibrated_oof_probabilities"],
        "Selected Variant": selected_variant,
        "Selected Prediction": selected_predictions,
        "Selected OA Probability": selected_probabilities,
    })
    oof_path = save_csv(oof, "selected_model_oof_predictions.csv")

    matrix = save_all_plots(
        comparison,
        active_df,
        y_active,
        selected_predictions,
        selected_importance,
        baseline_probabilities,
        selected_evaluation["calibrated_oof_probabilities"],
        selected_variant,
    )
    confusion_table = pd.DataFrame(
        matrix,
        index=["Actual Healthy", "Actual Knee OA"],
        columns=["Predicted Healthy", "Predicted Knee OA"],
    ).reset_index(names="Actual Label")
    confusion_path = save_csv(confusion_table, "confusion_matrix.csv")

    artifact_paths = {
        "selected_model": MODEL_PATH,
        "feature_order": FEATURES_PATH,
        "model_comparison": model_comparison_path,
        "per_fold_metrics": fold_metrics_path,
        "validation_fold_assignments": fold_assignments_path,
        "calibration_fold_metrics": calibration_folds_path,
        "calibration_comparison": calibration_summary_path,
        "selected_model_oof_predictions": oof_path,
        "confusion_matrix_table": confusion_path,
        "feature_importance": importance_path,
        "feature_importance_folds": importance_folds_path,
        "model_comparison_plot": RESULTS_DIR / "model_comparison.png",
        "confusion_matrix_plot": RESULTS_DIR / "confusion_matrix.png",
        "feature_importance_plot": RESULTS_DIR / "feature_importance.png",
        "class_distribution_plot": RESULTS_DIR / "class_distribution.png",
        "probability_calibration_plot": RESULTS_DIR / "probability_calibration.png",
    }
    metadata = build_metadata(
        real_df,
        active_df,
        models,
        comparison,
        best_model_name,
        selected_variant,
        calibration_applied,
        calibration_summary,
        final_training,
        artifact_paths,
    )
    METADATA_PATH.write_text(
        json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON (mean ± sample SD across five unified active-data folds)")
    print("=" * 100)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected model family (highest mean F1): {best_model_name}")
    print("\nCALIBRATION EVALUATION (unified active-data out-of-fold rows)")
    print(calibration_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Calibration applied to saved model: {calibration_applied}")
    print(f"Saved model variant: {selected_variant}")
    print("\nData interpretation:")
    print("- Training dataset: 588 unified active rows, including 500 generated examples.")
    print("- Validation: all 588 active rows, each held out exactly once.")
    print("- Synthetic rows are included in validation as ordinary active rows.")
    print("- Independent external clinical evidence: none.")
    print("- These results are internal prototype evaluation, not clinical validation.")
    print(f"\nSaved selected model: {MODEL_PATH}")
    print(f"Saved model metadata: {METADATA_PATH}")
    print(f"Saved evaluation tables and plots: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
