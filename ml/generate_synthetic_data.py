from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from model_utils import FEATURES, RANDOM_STATE, SYNTHETIC_PER_CLASS, TARGET


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "full_indicators_summary.csv"
ORIGINAL_DATA_PATH = DATA_DIR / "full_indicators_summary.original.csv"
QUALITY_REPORT_PATH = DATA_DIR / "synthetic_data_quality_report.json"

EXPECTED_ORIGINAL_ROWS = 88
EXPECTED_SYNTHETIC_ROWS = SYNTHETIC_PER_CLASS * 2
EXPECTED_TOTAL_ROWS = EXPECTED_ORIGINAL_ROWS + EXPECTED_SYNTHETIC_ROWS
CANDIDATE_SYNTHETIC_PER_CLASS = 1000

ALLOWED_TARGETS = {0, 1}
ALLOWED_GENDERS = {0, 1}
FEATURE_RANGES = {
    "age": {"minimum": 18, "maximum": 100, "integer": True},
    "BMI": {"minimum_exclusive": 0, "maximum": 80, "integer": False},
    "VAS score": {"minimum": 0, "maximum": 10, "integer": True},
}
PROVENANCE_COLUMNS = [
    "record_id",
    "record_source",
    "generation_method",
    "validation_eligible",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_numeric_range(data, column, rules, dataset_name):
    values = pd.to_numeric(data[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{dataset_name}: {column} contains non-numeric or non-finite values.")

    if "minimum" in rules and (values < rules["minimum"]).any():
        raise ValueError(f"{dataset_name}: {column} contains values below {rules['minimum']}.")
    if "minimum_exclusive" in rules and (values <= rules["minimum_exclusive"]).any():
        raise ValueError(
            f"{dataset_name}: {column} must be greater than {rules['minimum_exclusive']}."
        )
    if (values > rules["maximum"]).any():
        raise ValueError(f"{dataset_name}: {column} contains values above {rules['maximum']}.")
    if rules["integer"] and not np.allclose(values, np.round(values)):
        raise ValueError(f"{dataset_name}: {column} must contain integer values.")


def validate_model_columns(data, dataset_name):
    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{dataset_name}: missing required columns: {missing_columns}")
    if data[required_columns].isna().any().any():
        missing_counts = data[required_columns].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"{dataset_name}: missing model values: {missing_counts}")

    for column, rules in FEATURE_RANGES.items():
        validate_numeric_range(data, column, rules, dataset_name)

    gender_values = set(pd.to_numeric(data["gender"], errors="coerce").dropna().astype(int))
    if gender_values - ALLOWED_GENDERS:
        raise ValueError(f"{dataset_name}: invalid gender values: {sorted(gender_values)}")
    if not np.allclose(data["gender"], np.round(data["gender"])):
        raise ValueError(f"{dataset_name}: gender must contain integer category values.")

    target_values = set(pd.to_numeric(data[TARGET], errors="coerce").dropna().astype(int))
    if target_values != ALLOWED_TARGETS:
        raise ValueError(
            f"{dataset_name}: expected target labels {sorted(ALLOWED_TARGETS)}; "
            f"found {sorted(target_values)}."
        )
    if not np.allclose(data[TARGET], np.round(data[TARGET])):
        raise ValueError(f"{dataset_name}: target labels must be integers.")


def load_original_data():
    if not ORIGINAL_DATA_PATH.exists():
        raise FileNotFoundError(
            "The preserved 88-record original dataset is required and will not be recreated "
            f"from the active training file: {ORIGINAL_DATA_PATH}"
        )

    original = pd.read_csv(ORIGINAL_DATA_PATH, encoding="gb18030")
    if len(original) != EXPECTED_ORIGINAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ORIGINAL_ROWS} original records; found {len(original)}."
        )
    if "number" not in original.columns:
        raise ValueError("Original dataset is missing the subject identifier column: number")
    if original["number"].isna().any() or original["number"].duplicated().any():
        raise ValueError("Original subject identifiers must be complete and unique.")

    validate_model_columns(original, "Original dataset")
    return original


def postprocess_synthetic_candidates(sampled_X, sampled_y, original_rows):
    synthetic = pd.DataFrame(sampled_X, columns=FEATURES).iloc[original_rows:].copy()
    synthetic[TARGET] = pd.Series(sampled_y, name=TARGET).iloc[original_rows:].to_numpy()

    synthetic["age"] = synthetic["age"].round().astype(int)
    synthetic["gender"] = synthetic["gender"].round().astype(int)
    synthetic["VAS score"] = synthetic["VAS score"].round().astype(int)
    synthetic["BMI"] = synthetic["BMI"].round(2)
    synthetic[TARGET] = synthetic[TARGET].astype(int)
    return synthetic


def select_unique_synthetic_rows(candidates, original):
    candidate_rows_generated = len(candidates)
    original_feature_keys = set(map(tuple, original[FEATURES].to_numpy()))
    candidate_feature_keys = candidates[FEATURES].apply(tuple, axis=1)
    real_overlap_mask = candidate_feature_keys.isin(original_feature_keys)
    real_overlap_count = int(real_overlap_mask.sum())
    candidates = candidates.loc[~real_overlap_mask].copy()

    candidates = candidates.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    duplicate_candidate_count = int(candidates.duplicated(subset=FEATURES, keep="first").sum())
    candidates = candidates.drop_duplicates(subset=FEATURES, keep="first")

    available_counts = candidates[TARGET].value_counts().sort_index().to_dict()
    for target_label in sorted(ALLOWED_TARGETS):
        available = int(available_counts.get(target_label, 0))
        if available < SYNTHETIC_PER_CLASS:
            raise ValueError(
                f"Only {available} unique non-overlapping synthetic candidates were available "
                f"for target {target_label}; {SYNTHETIC_PER_CLASS} are required."
            )

    synthetic = (
        candidates.groupby(TARGET, group_keys=False, sort=True)
        .head(SYNTHETIC_PER_CLASS)
        .sort_values([TARGET] + FEATURES, kind="stable")
        .reset_index(drop=True)
    )
    synthetic["record_id"] = [
        f"synthetic_{index:04d}" for index in range(1, len(synthetic) + 1)
    ]
    synthetic["record_source"] = "synthetic"
    synthetic["generation_method"] = "SMOTENC"
    synthetic["validation_eligible"] = False

    filtering_summary = {
        "candidate_rows_generated": int(candidate_rows_generated),
        "candidate_rows_matching_real_feature_vectors_removed": real_overlap_count,
        "duplicate_candidate_feature_vectors_removed": duplicate_candidate_count,
        "unique_candidates_available_by_target": {
            str(key): int(value) for key, value in available_counts.items()
        },
    }
    return synthetic, filtering_summary


def add_real_provenance(original):
    real = original[FEATURES + [TARGET]].copy()
    real["record_id"] = original["number"].map(lambda value: f"real_{int(value):04d}")
    real["record_source"] = "real"
    real["generation_method"] = "observed"
    real["validation_eligible"] = True
    return real


def validate_active_dataset(active, original):
    validate_model_columns(active, "Active training dataset")

    required_columns = FEATURES + [TARGET] + PROVENANCE_COLUMNS
    missing_columns = sorted(set(required_columns) - set(active.columns))
    if missing_columns:
        raise ValueError(f"Active training dataset: missing provenance columns: {missing_columns}")
    if active[required_columns].isna().any().any():
        raise ValueError("Active training dataset contains missing model or provenance values.")
    if len(active) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} active rows; found {len(active)}.")
    if active["record_id"].duplicated().any():
        raise ValueError("Active training dataset contains duplicate record identifiers.")

    source_counts = active["record_source"].value_counts().to_dict()
    if source_counts != {"synthetic": EXPECTED_SYNTHETIC_ROWS, "real": EXPECTED_ORIGINAL_ROWS}:
        raise ValueError(f"Unexpected record-source counts: {source_counts}")
    if set(active["record_source"].unique()) != {"real", "synthetic"}:
        raise ValueError("record_source may contain only 'real' and 'synthetic'.")

    real = active[active["record_source"] == "real"]
    synthetic = active[active["record_source"] == "synthetic"]
    synthetic_counts = synthetic[TARGET].value_counts().sort_index().to_dict()
    if synthetic_counts != {0: SYNTHETIC_PER_CLASS, 1: SYNTHETIC_PER_CLASS}:
        raise ValueError(f"Unexpected synthetic target counts: {synthetic_counts}")
    if synthetic.duplicated(subset=FEATURES, keep=False).any():
        raise ValueError("Synthetic data contains duplicate feature rows.")
    if synthetic.duplicated(subset=FEATURES + [TARGET], keep=False).any():
        raise ValueError("Synthetic data contains duplicate feature/target rows.")

    real_feature_keys = set(map(tuple, real[FEATURES].to_numpy()))
    synthetic_feature_keys = set(map(tuple, synthetic[FEATURES].to_numpy()))
    if real_feature_keys & synthetic_feature_keys:
        raise ValueError("Synthetic data contains feature rows copied from real records.")

    if not real["validation_eligible"].eq(True).all():
        raise ValueError("Every real record must be marked validation_eligible=True.")
    if not synthetic["validation_eligible"].eq(False).all():
        raise ValueError("Synthetic records must never be marked as validation subjects.")
    if not real["generation_method"].eq("observed").all():
        raise ValueError("Real records must have generation_method='observed'.")
    if not synthetic["generation_method"].eq("SMOTENC").all():
        raise ValueError("Synthetic records must have generation_method='SMOTENC'.")

    expected_real = original[FEATURES + [TARGET]].reset_index(drop=True)
    active_real = real[FEATURES + [TARGET]].reset_index(drop=True)
    if not active_real.equals(expected_real):
        raise ValueError("Real rows in the active dataset do not match the preserved original data.")


def numeric_summary(series):
    values = pd.to_numeric(series)
    return {
        "count": int(values.count()),
        "mean": round(float(values.mean()), 6),
        "standard_deviation": round(float(values.std(ddof=1)), 6),
        "minimum": round(float(values.min()), 6),
        "p25": round(float(values.quantile(0.25)), 6),
        "median": round(float(values.median()), 6),
        "p75": round(float(values.quantile(0.75)), 6),
        "maximum": round(float(values.max()), 6),
    }


def categorical_summary(series):
    counts = series.value_counts().sort_index()
    total = len(series)
    return {
        str(value): {
            "count": int(count),
            "proportion": round(float(count / total), 6),
        }
        for value, count in counts.items()
    }


def distribution_comparison(real, synthetic):
    comparison = {}
    for feature in FEATURES:
        real_numeric = numeric_summary(real[feature])
        synthetic_numeric = numeric_summary(synthetic[feature])
        entry = {
            "real": real_numeric,
            "synthetic": synthetic_numeric,
            "synthetic_minus_real_mean": round(
                synthetic_numeric["mean"] - real_numeric["mean"], 6
            ),
        }
        if feature in {"gender", "VAS score"}:
            entry["real_category_distribution"] = categorical_summary(real[feature])
            entry["synthetic_category_distribution"] = categorical_summary(synthetic[feature])
        comparison[feature] = entry
    return comparison


def build_quality_report(active, original, filtering_summary, original_sha256):
    real = active[active["record_source"] == "real"]
    synthetic = active[active["record_source"] == "synthetic"]
    quality_warnings = []
    for feature in ("gender", "VAS score"):
        real_categories = set(real[feature].unique())
        synthetic_categories = set(synthetic[feature].unique())
        missing_synthetic_categories = sorted(real_categories - synthetic_categories)
        if missing_synthetic_categories:
            quality_warnings.append(
                {
                    "code": "synthetic_category_coverage_gap",
                    "feature": feature,
                    "categories_present_in_real_but_missing_from_synthetic": [
                        int(value) for value in missing_synthetic_categories
                    ],
                    "interpretation": (
                        "The synthetic data does not cover every category observed in the "
                        "88 real subjects; it must not be treated as additional clinical evidence."
                    ),
                }
            )

    return {
        "schema_version": 1,
        "quality_status": "passed_with_warnings" if quality_warnings else "passed",
        "quality_warnings": quality_warnings,
        "dataset_identity": {
            "observed_subjects": EXPECTED_ORIGINAL_ROWS,
            "synthetic_training_examples": EXPECTED_SYNTHETIC_ROWS,
            "total_active_training_rows": EXPECTED_TOTAL_ROWS,
            "validation_subject_policy": (
                "Only the 88 rows marked record_source=real and validation_eligible=true "
                "may be treated as independent validation subjects."
            ),
            "claim_guardrail": (
                "The 500 synthetic rows are generated training examples, not additional "
                "observed patients or independent validation subjects."
            ),
        },
        "files": {
            "original_dataset": ORIGINAL_DATA_PATH.name,
            "active_training_dataset": DATA_PATH.name,
            "quality_report": QUALITY_REPORT_PATH.name,
            "original_dataset_sha256": original_sha256,
        },
        "generation": {
            "method": "SMOTENC",
            "random_seed": RANDOM_STATE,
            "categorical_features": ["gender", "VAS score"],
            "k_neighbors": min(5, int(original[TARGET].value_counts().min()) - 1),
            "candidate_synthetic_rows_requested_per_target": CANDIDATE_SYNTHETIC_PER_CLASS,
            "selected_synthetic_rows_per_target": SYNTHETIC_PER_CLASS,
            "postprocessing": [
                "age rounded to integer",
                "BMI rounded to two decimal places",
                "VAS score rounded to integer",
                "candidate feature rows matching real feature rows removed",
                "duplicate synthetic feature rows removed before deterministic selection",
            ],
            **filtering_summary,
        },
        "counts": {
            "by_record_source": {
                str(key): int(value)
                for key, value in active["record_source"].value_counts().items()
            },
            "real_by_target": {
                str(key): int(value) for key, value in real[TARGET].value_counts().sort_index().items()
            },
            "synthetic_by_target": {
                str(key): int(value)
                for key, value in synthetic[TARGET].value_counts().sort_index().items()
            },
            "active_by_target": {
                str(key): int(value)
                for key, value in active[TARGET].value_counts().sort_index().items()
            },
        },
        "quality_checks": {
            "original_row_count_valid": len(original) == EXPECTED_ORIGINAL_ROWS,
            "active_row_count_valid": len(active) == EXPECTED_TOTAL_ROWS,
            "record_source_counts_valid": active["record_source"].value_counts().to_dict()
            == {"synthetic": EXPECTED_SYNTHETIC_ROWS, "real": EXPECTED_ORIGINAL_ROWS},
            "synthetic_target_counts_valid": synthetic[TARGET]
            .value_counts()
            .sort_index()
            .to_dict()
            == {0: SYNTHETIC_PER_CLASS, 1: SYNTHETIC_PER_CLASS},
            "model_columns_have_no_missing_values": not active[FEATURES + [TARGET]].isna().any().any(),
            "provenance_fields_have_no_missing_values": not active[PROVENANCE_COLUMNS].isna().any().any(),
            "record_ids_unique": not active["record_id"].duplicated().any(),
            "feature_ranges_valid": True,
            "target_labels_valid": set(active[TARGET].unique()) == ALLOWED_TARGETS,
            "gender_categories_valid": set(active["gender"].unique()).issubset(ALLOWED_GENDERS),
            "synthetic_feature_rows_unique": not synthetic.duplicated(subset=FEATURES).any(),
            "synthetic_rows_do_not_copy_real_feature_rows": not bool(
                set(map(tuple, real[FEATURES].to_numpy()))
                & set(map(tuple, synthetic[FEATURES].to_numpy()))
            ),
            "synthetic_rows_excluded_from_validation": not synthetic["validation_eligible"].any(),
            "real_rows_eligible_for_validation": bool(real["validation_eligible"].all()),
            "real_rows_match_original": real[FEATURES + [TARGET]]
            .reset_index(drop=True)
            .equals(original[FEATURES + [TARGET]].reset_index(drop=True)),
            "original_dataset_hash_preserved": file_sha256(ORIGINAL_DATA_PATH) == original_sha256,
        },
        "feature_ranges": FEATURE_RANGES,
        "distribution_comparison": distribution_comparison(real, synthetic),
        "distribution_comparison_by_target": {
            str(target_label): distribution_comparison(
                real[real[TARGET] == target_label],
                synthetic[synthetic[TARGET] == target_label],
            )
            for target_label in sorted(ALLOWED_TARGETS)
        },
    }


def write_outputs(active, report, original_sha256):
    data_temp_path = DATA_PATH.with_suffix(".csv.tmp")
    report_temp_path = QUALITY_REPORT_PATH.with_suffix(".json.tmp")
    try:
        active.to_csv(data_temp_path, index=False, encoding="utf-8-sig")
        report["files"]["active_training_dataset_sha256"] = file_sha256(data_temp_path)
        report_temp_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if file_sha256(ORIGINAL_DATA_PATH) != original_sha256:
            raise RuntimeError("Original dataset changed during generation; outputs were not replaced.")

        data_temp_path.replace(DATA_PATH)
        report_temp_path.replace(QUALITY_REPORT_PATH)
    finally:
        data_temp_path.unlink(missing_ok=True)
        report_temp_path.unlink(missing_ok=True)


def generate_dataset():
    original = load_original_data()
    original_sha256 = file_sha256(ORIGINAL_DATA_PATH)
    X = original[FEATURES]
    y = original[TARGET].astype(int)
    counts = y.value_counts().sort_index()
    sampling_strategy = {
        int(target_label): int(count + CANDIDATE_SYNTHETIC_PER_CLASS)
        for target_label, count in counts.items()
    }

    sampler = SMOTENC(
        categorical_features=[FEATURES.index("gender"), FEATURES.index("VAS score")],
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=min(5, int(counts.min()) - 1),
    )
    sampled_X, sampled_y = sampler.fit_resample(X, y)
    candidates = postprocess_synthetic_candidates(sampled_X, sampled_y, len(original))

    for feature in ("age", "BMI"):
        candidates[feature] = candidates[feature].clip(
            lower=original[feature].min(),
            upper=original[feature].max(),
        )
    validate_model_columns(candidates, "Synthetic candidate pool")

    synthetic, filtering_summary = select_unique_synthetic_rows(candidates, original)
    real = add_real_provenance(original)
    active = pd.concat([real, synthetic], ignore_index=True)
    active = active[FEATURES + [TARGET] + PROVENANCE_COLUMNS]

    validate_active_dataset(active, original)
    report = build_quality_report(active, original, filtering_summary, original_sha256)
    if not all(report["quality_checks"].values()):
        failed_checks = [
            name for name, passed in report["quality_checks"].items() if not passed
        ]
        raise ValueError(f"Synthetic-data quality checks failed: {failed_checks}")

    write_outputs(active, report, original_sha256)

    print(f"Preserved original dataset: {ORIGINAL_DATA_PATH}")
    print(f"Original dataset SHA-256: {original_sha256}")
    print(f"Updated active dataset: {DATA_PATH}")
    print(f"Synthetic-data quality report: {QUALITY_REPORT_PATH}")
    print(f"Observed subjects: {EXPECTED_ORIGINAL_ROWS}")
    print(f"Synthetic training examples: {EXPECTED_SYNTHETIC_ROWS}")
    print(f"Total active training rows: {EXPECTED_TOTAL_ROWS}")
    print("Synthetic examples by target:")
    print(synthetic[TARGET].value_counts().sort_index().to_string())
    print("Duplicate synthetic feature rows: 0")
    print("Synthetic rows eligible for validation: 0")


if __name__ == "__main__":
    generate_dataset()
