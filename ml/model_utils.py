import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # scikit-learn 1.4 and 1.5 compatibility
    FrozenEstimator = None


FEATURES = ["age", "gender", "BMI", "VAS score"]
CATEGORICAL_FEATURES = ["gender", "VAS score"]
FEATURE_CONFIGURATION = "vas_categorical_baseline"
TARGET = "group"
SYNTHETIC_PER_CLASS = 250
RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.20
CALIBRATION_METHOD = "sigmoid"
PERMUTATION_REPEATS = 30
METRICS = ("Accuracy", "Precision", "Recall", "F1")


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def positive_class_probability(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Model does not contain the OA target class 1: {classes}")
    return probabilities[:, classes.index(1)]


def summarize_fold_metrics(fold_metrics_df):
    rows = []
    for model_name, model_rows in fold_metrics_df.groupby("Model", sort=False):
        summary = {"Model": model_name}
        for metric in METRICS:
            summary[f"{metric} Mean"] = model_rows[metric].mean()
            summary[f"{metric} Std"] = model_rows[metric].std(ddof=1)
        rows.append(summary)
    return (
        pd.DataFrame(rows)
        .sort_values("F1 Mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _source_counts(record_sources, indices):
    selected = pd.Series(record_sources).iloc[list(indices)]
    counts = selected.value_counts().to_dict()
    return {
        "real": int(counts.get("real", 0)),
        "synthetic": int(counts.get("synthetic", 0)),
    }


def cross_validate_on_unified_dataset(
    X,
    y,
    models,
    cv,
    record_ids=None,
    record_sources=None,
):
    """Compare models using the complete active table in every fold.

    Synthetic rows are already present in ``X`` and are treated exactly like
    observed rows. Their source is retained only in audit columns.
    """
    X = X[FEATURES].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
    if record_ids is None:
        record_ids = pd.Series(range(1, len(y) + 1))
    else:
        record_ids = pd.Series(record_ids).reset_index(drop=True)
    if record_sources is None:
        record_sources = pd.Series(["real"] * len(y))
    else:
        record_sources = pd.Series(record_sources).reset_index(drop=True)
    if len(record_ids) != len(y) or len(record_sources) != len(y):
        raise ValueError("Record IDs and sources must have one value per active row.")

    fold_metric_rows = []
    fold_assignment_rows = []
    out_of_fold_predictions = {
        name: np.full(len(y), -1, dtype=int) for name in models
    }
    out_of_fold_probabilities = {
        name: np.full(len(y), np.nan, dtype=float) for name in models
    }

    for fold_number, (train_indices, validation_indices) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]
        train_sources = _source_counts(record_sources, train_indices)
        validation_sources = _source_counts(record_sources, validation_indices)

        for validation_index in validation_indices:
            fold_assignment_rows.append({
                "Fold": fold_number,
                "Record ID": str(record_ids.iloc[validation_index]),
                "Target": int(y.iloc[validation_index]),
                "Record Source": str(record_sources.iloc[validation_index]),
            })

        for name, estimator in models.items():
            fold_model = clone(estimator)
            fold_model.fit(X_train, y_train)
            predictions = fold_model.predict(X_validation).astype(int)
            probabilities = positive_class_probability(fold_model, X_validation)
            out_of_fold_predictions[name][validation_indices] = predictions
            out_of_fold_probabilities[name][validation_indices] = probabilities
            fold_metric_rows.append({
                "Model": name,
                "Fold": fold_number,
                "Training Rows": int(len(train_indices)),
                "Real Training Rows": train_sources["real"],
                "Synthetic Training Rows": train_sources["synthetic"],
                "Validation Rows": int(len(validation_indices)),
                "Validation Real Rows": validation_sources["real"],
                "Validation Synthetic Rows": validation_sources["synthetic"],
                **classification_metrics(y_validation, predictions),
            })

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    if any((predictions < 0).any() for predictions in out_of_fold_predictions.values()):
        raise RuntimeError("Some active rows did not receive an out-of-fold prediction.")

    return {
        "summary": summarize_fold_metrics(fold_metrics_df),
        "fold_metrics": fold_metrics_df,
        "fold_assignments": pd.DataFrame(fold_assignment_rows),
        "oof_predictions": out_of_fold_predictions,
        "oof_probabilities": out_of_fold_probabilities,
    }


def cross_validate_with_fold_augmentation(X, y, models, cv, subject_ids=None):
    """Backward-compatible alias for the unified active-data cross-validation."""
    return cross_validate_on_unified_dataset(
        X,
        y,
        models,
        cv,
        record_ids=subject_ids,
    )


def build_prefit_calibrator(prefit_model):
    if FrozenEstimator is not None:
        estimator = FrozenEstimator(prefit_model)
        return CalibratedClassifierCV(estimator, method=CALIBRATION_METHOD)
    return CalibratedClassifierCV(
        prefit_model,
        method=CALIBRATION_METHOD,
        cv="prefit",
    )


def probability_metrics(y_true, probabilities):
    return {
        "Brier Score": brier_score_loss(y_true, probabilities),
        "Log Loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }


def evaluate_selected_model(
    X,
    y,
    estimator,
    cv,
    baseline_oof_predictions,
    baseline_oof_probabilities,
    record_sources=None,
):
    """Evaluate calibration and permutation importance on unified active folds."""
    X = X[FEATURES].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
    if record_sources is None:
        record_sources = pd.Series(["real"] * len(y))
    else:
        record_sources = pd.Series(record_sources).reset_index(drop=True)
    calibrated_oof_predictions = np.full(len(y), -1, dtype=int)
    calibrated_oof_probabilities = np.full(len(y), np.nan, dtype=float)
    calibration_fold_rows = []
    importance_fold_rows = []
    importance_values = {"Uncalibrated": {feature: [] for feature in FEATURES}}
    importance_values["Calibrated"] = {feature: [] for feature in FEATURES}
    f1_scorer = make_scorer(f1_score, zero_division=0)

    for fold_number, (outer_train_indices, validation_indices) in enumerate(
        cv.split(X, y), start=1
    ):
        X_outer_train = X.iloc[outer_train_indices].reset_index(drop=True)
        y_outer_train = y.iloc[outer_train_indices].reset_index(drop=True)
        outer_record_sources = record_sources.iloc[outer_train_indices].reset_index(drop=True)
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]
        outer_validation_sources = _source_counts(record_sources, validation_indices)

        baseline_model = clone(estimator).fit(X_outer_train, y_outer_train)

        calibration_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=CALIBRATION_FRACTION,
            random_state=RANDOM_STATE + fold_number,
        )
        base_indices, calibration_indices = next(
            calibration_split.split(X_outer_train, y_outer_train)
        )
        X_base = X_outer_train.iloc[base_indices].reset_index(drop=True)
        y_base = y_outer_train.iloc[base_indices].reset_index(drop=True)
        X_calibration = X_outer_train.iloc[calibration_indices]
        y_calibration = y_outer_train.iloc[calibration_indices]
        prefit_model = clone(estimator).fit(X_base, y_base)
        calibrated_model = build_prefit_calibrator(prefit_model)
        calibrated_model.fit(X_calibration, y_calibration)

        baseline_probabilities = baseline_oof_probabilities[validation_indices]
        baseline_predictions = baseline_oof_predictions[validation_indices]
        calibrated_probabilities = positive_class_probability(calibrated_model, X_validation)
        calibrated_predictions = calibrated_model.predict(X_validation).astype(int)
        calibrated_oof_probabilities[validation_indices] = calibrated_probabilities
        calibrated_oof_predictions[validation_indices] = calibrated_predictions

        for variant, predictions, probabilities in (
            ("Uncalibrated", baseline_predictions, baseline_probabilities),
            ("Calibrated", calibrated_predictions, calibrated_probabilities),
        ):
            calibration_sources = _source_counts(outer_record_sources, calibration_indices)
            base_sources = _source_counts(outer_record_sources, base_indices)
            outer_train_sources = _source_counts(record_sources, outer_train_indices)
            calibration_fold_rows.append({
                "Variant": variant,
                "Fold": fold_number,
                "Outer Validation Real Rows": outer_validation_sources["real"],
                "Outer Validation Synthetic Rows": outer_validation_sources["synthetic"],
                "Calibration Real Rows": (
                    0 if variant == "Uncalibrated" else calibration_sources["real"]
                ),
                "Calibration Synthetic Rows": (
                    0 if variant == "Uncalibrated" else calibration_sources["synthetic"]
                ),
                "Base Real Training Rows": (
                    outer_train_sources["real"]
                    if variant == "Uncalibrated"
                    else base_sources["real"]
                ),
                "Base Synthetic Training Rows": (
                    outer_train_sources["synthetic"]
                    if variant == "Uncalibrated"
                    else base_sources["synthetic"]
                ),
                **classification_metrics(y_validation, predictions),
                **probability_metrics(y_validation, probabilities),
            })

        for variant, fitted_model in (
            ("Uncalibrated", baseline_model),
            ("Calibrated", calibrated_model),
        ):
            importance = permutation_importance(
                fitted_model,
                X_validation,
                y_validation,
                scoring=f1_scorer,
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_STATE + fold_number,
            )
            for feature_index, feature in enumerate(FEATURES):
                values = importance.importances[feature_index]
                importance_values[variant][feature].extend(float(value) for value in values)
                importance_fold_rows.append({
                    "Variant": variant,
                    "Fold": fold_number,
                    "Feature": feature,
                    "Importance Mean": float(values.mean()),
                    "Importance Std": float(values.std(ddof=1)),
                    "Held-Out Real Rows": outer_validation_sources["real"],
                    "Held-Out Synthetic Rows": _source_counts(
                        record_sources, validation_indices
                    )["synthetic"],
                    "Repeats": PERMUTATION_REPEATS,
                })

    calibration_folds = pd.DataFrame(calibration_fold_rows)
    calibration_summary_rows = []
    for variant, rows in calibration_folds.groupby("Variant", sort=False):
        summary = {"Variant": variant}
        predictions = (
            baseline_oof_predictions
            if variant == "Uncalibrated"
            else calibrated_oof_predictions
        )
        probabilities = (
            baseline_oof_probabilities
            if variant == "Uncalibrated"
            else calibrated_oof_probabilities
        )
        summary.update({
            f"{metric} Mean": rows[metric].mean()
            for metric in (*METRICS, "Brier Score", "Log Loss")
        })
        summary.update({
            f"{metric} Std": rows[metric].std(ddof=1)
            for metric in (*METRICS, "Brier Score", "Log Loss")
        })
        summary.update({
            "OOF Accuracy": accuracy_score(y, predictions),
            "OOF Precision": precision_score(y, predictions, zero_division=0),
            "OOF Recall": recall_score(y, predictions, zero_division=0),
            "OOF F1": f1_score(y, predictions, zero_division=0),
            "OOF Brier Score": brier_score_loss(y, probabilities),
            "OOF Log Loss": log_loss(y, probabilities, labels=[0, 1]),
        })
        calibration_summary_rows.append(summary)

    importance_summary_rows = []
    for variant, feature_values in importance_values.items():
        for feature, values in feature_values.items():
            importance_summary_rows.append({
                "Variant": variant,
                "Feature": feature,
                "Importance Mean": float(np.mean(values)),
                "Importance Std": float(np.std(values, ddof=1)),
                "Held-Out Folds": 5,
                "Total Permutations": len(values),
            })

    return {
        "calibration_folds": calibration_folds,
        "calibration_summary": pd.DataFrame(calibration_summary_rows),
        "calibrated_oof_predictions": calibrated_oof_predictions,
        "calibrated_oof_probabilities": calibrated_oof_probabilities,
        "importance_folds": pd.DataFrame(importance_fold_rows),
        "importance_summary": pd.DataFrame(importance_summary_rows),
    }


def should_apply_calibration(calibration_summary):
    baseline = calibration_summary.set_index("Variant").loc["Uncalibrated"]
    calibrated = calibration_summary.set_index("Variant").loc["Calibrated"]
    return bool(
        calibrated["OOF Brier Score"] < baseline["OOF Brier Score"]
        and calibrated["OOF Log Loss"] < baseline["OOF Log Loss"]
        and calibrated["OOF F1"] >= baseline["OOF F1"]
    )


def fit_final_calibrated_model(X, y, estimator, record_sources=None):
    X = X[FEATURES].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
    if record_sources is None:
        record_sources = pd.Series(["real"] * len(y))
    else:
        record_sources = pd.Series(record_sources).reset_index(drop=True)
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
    )
    base_indices, calibration_indices = next(split.split(X, y))
    X_base = X.iloc[base_indices].reset_index(drop=True)
    y_base = y.iloc[base_indices].reset_index(drop=True)
    X_calibration = X.iloc[calibration_indices]
    y_calibration = y.iloc[calibration_indices]
    prefit_model = clone(estimator).fit(X_base, y_base)
    calibrated_model = build_prefit_calibrator(prefit_model)
    calibrated_model.fit(X_calibration, y_calibration)
    base_sources = _source_counts(record_sources, base_indices)
    calibration_sources = _source_counts(record_sources, calibration_indices)
    metadata = {
        "base_training_rows": int(len(base_indices)),
        "calibration_rows": int(len(calibration_indices)),
        "base_real_training_rows": base_sources["real"],
        "base_synthetic_training_rows": base_sources["synthetic"],
        "real_calibration_rows": calibration_sources["real"],
        "synthetic_calibration_rows": calibration_sources["synthetic"],
    }
    return calibrated_model, metadata
