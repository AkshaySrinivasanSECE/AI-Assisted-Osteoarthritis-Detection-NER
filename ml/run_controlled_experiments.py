"""Run leakage-safe, non-production feature representation experiments.

This module intentionally does not modify the saved production model, feature
order, active synthetic dataset, API, or frontend. It evaluates configurations
using only the preserved observed dataset and generates augmentation separately
inside each outer training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform

import imblearn
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from model_utils import (
    METRICS,
    RANDOM_STATE,
    SYNTHETIC_PER_CLASS,
    TARGET,
    build_models,
    classification_metrics,
    positive_class_probability,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "full_indicators_summary.original.csv"
OUTPUT_DIR = ROOT_DIR / "docs" / "results" / "controlled_experiments"
JSON_PATH = OUTPUT_DIR / "controlled_experiment_results.json"
REPORT_PATH = OUTPUT_DIR / "controlled_experiment_report.md"
SUMMARY_PATH = OUTPUT_DIR / "summary_metrics.csv"
FOLDS_PATH = OUTPUT_DIR / "fold_metrics.csv"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrices.csv"
ASSIGNMENTS_PATH = OUTPUT_DIR / "fold_assignments.csv"

EXPECTED_ROWS = 88
EXPECTED_TARGET_COUNTS = {0: 45, 1: 43}
BASELINE_FEATURES = ("age", "gender", "BMI", "VAS score")
WOMAC_FEATURES = (*BASELINE_FEATURES, "WOMAC score")


@dataclass(frozen=True)
class ExperimentConfiguration:
    key: str
    label: str
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]


CONFIGURATIONS = (
    ExperimentConfiguration(
        key="vas_categorical_baseline",
        label="4-feature baseline; VAS categorical in SMOTENC",
        features=BASELINE_FEATURES,
        categorical_features=("gender", "VAS score"),
    ),
    ExperimentConfiguration(
        key="vas_numeric_baseline",
        label="4-feature baseline; VAS numeric in SMOTENC",
        features=BASELINE_FEATURES,
        categorical_features=("gender",),
    ),
    ExperimentConfiguration(
        key="womac_candidate",
        label="5-feature WOMAC candidate; VAS categorical in SMOTENC",
        features=WOMAC_FEATURES,
        categorical_features=("gender", "VAS score"),
    ),
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def load_observed_data() -> pd.DataFrame:
    observed = pd.read_csv(DATA_PATH, encoding="gb18030")
    required = {"number", TARGET, *WOMAC_FEATURES}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"Observed dataset is missing required columns: {missing}")
    if len(observed) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} observed subjects; found {len(observed)}.")
    counts = observed[TARGET].value_counts().sort_index().to_dict()
    if counts != EXPECTED_TARGET_COUNTS:
        raise ValueError(f"Unexpected observed target counts: {counts}")
    if observed[list(WOMAC_FEATURES) + [TARGET, "number"]].isna().any().any():
        raise ValueError("Observed experiment fields contain missing values.")
    if observed["number"].duplicated().any():
        raise ValueError("Observed subject identifiers must be unique.")
    return observed


def build_outer_folds(X: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return [(train.copy(), validation.copy()) for train, validation in splitter.split(X, y)]


def augment_training_partition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    configuration: ExperimentConfiguration,
) -> tuple[pd.DataFrame, pd.Series]:
    features = list(configuration.features)
    X_train = X_train[features].reset_index(drop=True)
    y_train = y_train.astype(int).reset_index(drop=True)
    class_counts = y_train.value_counts().sort_index()
    if set(class_counts.index) != {0, 1} or int(class_counts.min()) < 2:
        raise ValueError(f"Invalid outer training class counts: {class_counts.to_dict()}")

    sampler = SMOTENC(
        categorical_features=[
            features.index(feature) for feature in configuration.categorical_features
        ],
        sampling_strategy={
            int(label): int(count + SYNTHETIC_PER_CLASS)
            for label, count in class_counts.items()
        },
        random_state=RANDOM_STATE,
        k_neighbors=min(5, int(class_counts.min()) - 1),
    )
    augmented_X, augmented_y = sampler.fit_resample(X_train, y_train)
    augmented_X = pd.DataFrame(augmented_X, columns=features)
    augmented_y = pd.Series(augmented_y, name=TARGET).astype(int)

    # Preserve the same discrete/precision postprocessing used by production.
    for feature in ("age", "gender", "VAS score", "WOMAC score"):
        if feature in augmented_X:
            augmented_X[feature] = augmented_X[feature].round().astype(int)
    augmented_X["BMI"] = augmented_X["BMI"].round(2)

    generated_rows = len(augmented_X) - len(X_train)
    if generated_rows != SYNTHETIC_PER_CLASS * 2:
        raise RuntimeError(
            f"Expected {SYNTHETIC_PER_CLASS * 2} fold-local synthetic rows; "
            f"generated {generated_rows}."
        )
    return augmented_X, augmented_y


def evaluate_configuration(
    observed: pd.DataFrame,
    configuration: ExperimentConfiguration,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], list[dict]]:
    features = list(configuration.features)
    X = observed[features].reset_index(drop=True)
    y = observed[TARGET].astype(int).reset_index(drop=True)
    subject_ids = observed["number"].astype(int).reset_index(drop=True)
    models = build_models()
    fold_rows: list[dict] = []
    oof_rows: list[dict] = []

    for model_name, estimator in models.items():
        predictions = np.full(len(y), -1, dtype=int)
        probabilities = np.full(len(y), np.nan, dtype=float)

        for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
            augmented_X, augmented_y = augment_training_partition(
                X.iloc[train_indices], y.iloc[train_indices], configuration
            )
            fitted = clone(estimator).fit(augmented_X, augmented_y)
            fold_predictions = fitted.predict(X.iloc[validation_indices]).astype(int)
            fold_probabilities = positive_class_probability(
                fitted, X.iloc[validation_indices]
            )
            predictions[validation_indices] = fold_predictions
            probabilities[validation_indices] = fold_probabilities
            fold_rows.append({
                "Configuration": configuration.key,
                "Configuration Label": configuration.label,
                "Model": model_name,
                "Fold": fold_number,
                "Real Training Rows": int(len(train_indices)),
                "Synthetic Training Rows": SYNTHETIC_PER_CLASS * 2,
                "Validation Real Rows": int(len(validation_indices)),
                "Validation Synthetic Rows": 0,
                **classification_metrics(y.iloc[validation_indices], fold_predictions),
            })

        if (predictions < 0).any() or np.isnan(probabilities).any():
            raise RuntimeError(
                f"Incomplete OOF results for {configuration.key}/{model_name}."
            )
        for row_index in range(len(y)):
            fold_number = next(
                index
                for index, (_, validation_indices) in enumerate(folds, start=1)
                if row_index in validation_indices
            )
            oof_rows.append({
                "Configuration": configuration.key,
                "Model": model_name,
                "Fold": fold_number,
                "Subject ID": int(subject_ids.iloc[row_index]),
                "Record Source": "real",
                "Actual Target": int(y.iloc[row_index]),
                "Predicted Target": int(predictions[row_index]),
                "OA Probability": float(probabilities[row_index]),
            })
    return fold_rows, oof_rows


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (configuration, label, model), group in fold_metrics.groupby(
        ["Configuration", "Configuration Label", "Model"], sort=False
    ):
        row = {
            "Configuration": configuration,
            "Configuration Label": label,
            "Model": model,
        }
        for metric in METRICS:
            row[f"{metric} Mean"] = float(group[metric].mean())
            row[f"{metric} Std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def build_confusion_tables(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (configuration, model), group in oof.groupby(
        ["Configuration", "Model"], sort=False
    ):
        matrix = confusion_matrix(
            group["Actual Target"], group["Predicted Target"], labels=[0, 1]
        )
        rows.append({
            "Configuration": configuration,
            "Model": model,
            "True Negative": int(matrix[0, 0]),
            "False Positive": int(matrix[0, 1]),
            "False Negative": int(matrix[1, 0]),
            "True Positive": int(matrix[1, 1]),
            "OOF Subjects": int(matrix.sum()),
        })
    return pd.DataFrame(rows)


def select_by_mean_f1(summary: pd.DataFrame, keys: tuple[str, ...]) -> dict:
    candidates = summary[summary["Configuration"].isin(keys)].copy()
    selected = candidates.sort_values(
        "F1 Mean", ascending=False, kind="stable"
    ).iloc[0]
    return {
        "configuration": selected["Configuration"],
        "model": selected["Model"],
        "mean_f1": float(selected["F1 Mean"]),
        "f1_sample_standard_deviation": float(selected["F1 Std"]),
    }


def fold_assignments(observed: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for fold_number, (_, validation_indices) in enumerate(folds, start=1):
        for index in validation_indices:
            rows.append({
                "Fold": fold_number,
                "Subject ID": int(observed.iloc[index]["number"]),
                "Target": int(observed.iloc[index][TARGET]),
                "Record Source": "real",
                "Validation Eligible": True,
            })
    return pd.DataFrame(rows)


def markdown_table(dataframe: pd.DataFrame, decimals: int = 6) -> str:
    headers = [str(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_human_report(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    selections: dict,
    dataset_hash: str,
) -> None:
    compact_summary = summary.drop(columns=["Configuration Label"])
    sections = [
        "# Controlled OA feature experiments",
        "",
        "## Scope and data",
        "",
        f"- Preserved observed dataset: `data/{DATA_PATH.name}` (SHA-256 `{dataset_hash}`).",
        "- Observed subjects: 88 (45 healthy, 43 OA).",
        "- Target: `group` (`0` healthy, `1` OA).",
        "- No production model, feature artifact, API, frontend, or active synthetic dataset was changed.",
        "- No K-L grade, KOA period, JPR, EMG, or other excluded feature was used.",
        "",
        "## Exact methodology",
        "",
        "All configurations use the same shuffled `StratifiedKFold(n_splits=5, "
        "shuffle=True, random_state=42)` assignments and the same five production candidate "
        "models with unchanged hyperparameters. In every fold, SMOTENC is fitted only to that "
        "fold's original observed training partition and adds 250 examples per target class "
        "(500 total). Validation contains only untouched original subjects and zero synthetic "
        "rows. Metrics are computed separately per fold; reported standard deviations are "
        "sample standard deviations across the five folds (`ddof=1`). Selection is strictly "
        "the highest mean fold F1.",
        "",
        "In this codebase, categorical representation refers to SMOTENC's categorical feature "
        "mask. The estimator pipelines do not one-hot encode gender or VAS; they receive the "
        "postprocessed numeric values exactly as in the current pipeline. VAS is rounded back "
        "to its integer score after augmentation in both VAS variants, preserving existing "
        "postprocessing. WOMAC is numeric in SMOTENC and rounded to its observed integer-score "
        "representation after augmentation.",
        "",
        "## Configurations",
        "",
        "- `vas_categorical_baseline`: age, gender, BMI, VAS score; SMOTENC categorical fields: gender and VAS score.",
        "- `vas_numeric_baseline`: age, gender, BMI, VAS score; SMOTENC categorical field: gender only.",
        "- `womac_candidate`: age, gender, BMI, VAS score, WOMAC score; SMOTENC categorical fields: gender and VAS score.",
        "",
        "## Summary metrics",
        "",
        markdown_table(compact_summary),
        "",
        "## Selection under the existing mean-F1 rule",
        "",
        f"- VAS comparison: `{selections['vas_representation']['configuration']}` with "
        f"`{selections['vas_representation']['model']}` (mean F1 "
        f"{selections['vas_representation']['mean_f1']:.6f}).",
        f"- Baseline vs WOMAC: `{selections['womac']['configuration']}` with "
        f"`{selections['womac']['model']}` (mean F1 "
        f"{selections['womac']['mean_f1']:.6f}).",
        "",
        "## OOF confusion matrices",
        "",
        markdown_table(confusion),
        "",
        "## Per-fold metrics",
        "",
        markdown_table(fold_metrics.drop(columns=["Configuration Label"])),
        "",
        "## Reproduction command",
        "",
        "```powershell",
        "python ml\\run_controlled_experiments.py",
        "```",
        "",
        "The complete subject-level OOF predictions and probabilities are stored in "
        "`oof_predictions.csv`; the shared real-only validation assignments are stored in "
        "`fold_assignments.csv`; and the full machine-readable record is stored in "
        "`controlled_experiment_results.json`.",
    ]
    REPORT_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    observed = load_observed_data()
    y = observed[TARGET].astype(int).reset_index(drop=True)
    folds = build_outer_folds(observed[list(BASELINE_FEATURES)], y)
    all_fold_rows = []
    all_oof_rows = []
    for configuration in CONFIGURATIONS:
        fold_rows, oof_rows = evaluate_configuration(observed, configuration, folds)
        all_fold_rows.extend(fold_rows)
        all_oof_rows.extend(oof_rows)

    fold_metrics = pd.DataFrame(all_fold_rows)
    oof = pd.DataFrame(all_oof_rows)
    summary = summarize_metrics(fold_metrics)
    confusion = build_confusion_tables(oof)
    assignments = fold_assignments(observed, folds)

    if len(assignments) != EXPECTED_ROWS or assignments["Subject ID"].nunique() != EXPECTED_ROWS:
        raise RuntimeError("Each observed subject must appear in validation exactly once.")
    if (fold_metrics["Validation Synthetic Rows"] != 0).any():
        raise RuntimeError("Synthetic rows entered validation.")
    if set(oof["Record Source"]) != {"real"}:
        raise RuntimeError("Non-observed records entered OOF predictions.")

    selections = {
        "vas_representation": select_by_mean_f1(
            summary, ("vas_categorical_baseline", "vas_numeric_baseline")
        ),
        "womac": select_by_mean_f1(
            summary, ("vas_categorical_baseline", "womac_candidate")
        ),
    }
    dataset_hash = file_sha256(DATA_PATH)
    configuration_records = [
        {
            "key": configuration.key,
            "label": configuration.label,
            "features": list(configuration.features),
            "smotenc_categorical_features": list(configuration.categorical_features),
        }
        for configuration in CONFIGURATIONS
    ]
    machine_record = {
        "schema_version": 1,
        "experiment_only": True,
        "production_model_modified": False,
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT_DIR)).replace("\\", "/"),
            "sha256": dataset_hash,
            "observed_subjects": EXPECTED_ROWS,
            "target_counts": {str(key): value for key, value in EXPECTED_TARGET_COUNTS.items()},
        },
        "target": TARGET,
        "configurations": configuration_records,
        "validation_methodology": {
            "splitter": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_seed": RANDOM_STATE,
            "fold_assignments_shared_by_all_configurations": True,
            "validation_rows": "original observed subjects only",
            "validation_synthetic_rows": 0,
            "augmentation": "SMOTENC fitted separately on each outer training partition",
            "synthetic_examples_per_fold": SYNTHETIC_PER_CLASS * 2,
            "synthetic_examples_per_target_per_fold": SYNTHETIC_PER_CLASS,
            "metric_standard_deviation": "sample standard deviation across five folds (ddof=1)",
            "selection_rule": "highest mean F1 across the five folds",
        },
        "candidate_models": list(build_models().keys()),
        "summary_metrics": summary.to_dict(orient="records"),
        "per_fold_metrics": fold_metrics.to_dict(orient="records"),
        "oof_confusion_matrices": confusion.to_dict(orient="records"),
        "oof_predictions": oof.to_dict(orient="records"),
        "fold_assignments": assignments.to_dict(orient="records"),
        "selections": selections,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
        },
        "reproduction_command": "python ml\\run_controlled_experiments.py",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    fold_metrics.to_csv(FOLDS_PATH, index=False, encoding="utf-8")
    oof.to_csv(OOF_PATH, index=False, encoding="utf-8")
    confusion.to_csv(CONFUSION_PATH, index=False, encoding="utf-8")
    assignments.to_csv(ASSIGNMENTS_PATH, index=False, encoding="utf-8")
    JSON_PATH.write_text(
        json.dumps(json_safe(machine_record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_human_report(summary, fold_metrics, confusion, selections, dataset_hash)

    print("CONTROLLED EXPERIMENT SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nSELECTIONS")
    print(json.dumps(selections, indent=2, sort_keys=True))
    print(f"\nWrote experiment artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
