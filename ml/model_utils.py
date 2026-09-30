import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
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


def augment_training_fold(X_train, y_train):
    """Generate augmentation from one real training partition only."""
    X_train = X_train[FEATURES].reset_index(drop=True)
    y_train = y_train.astype(int).reset_index(drop=True)
    class_counts = y_train.value_counts().sort_index()
    if set(class_counts.index) != {0, 1}:
        raise ValueError(f"Training partition must contain target labels 0 and 1: {class_counts.to_dict()}")
    if int(class_counts.min()) < 2:
        raise ValueError("Each training class needs at least two real subjects for SMOTENC.")

    target_counts = {
        int(group): int(count + SYNTHETIC_PER_CLASS)
        for group, count in class_counts.items()
    }
    sampler = SMOTENC(
        categorical_features=[FEATURES.index(feature) for feature in CATEGORICAL_FEATURES],
        sampling_strategy=target_counts,
        random_state=RANDOM_STATE,
        k_neighbors=min(5, int(class_counts.min()) - 1),
    )
    sampled_X, sampled_y = sampler.fit_resample(X_train, y_train)
    sampled_X = pd.DataFrame(sampled_X, columns=FEATURES)
    sampled_y = pd.Series(sampled_y, name=TARGET).astype(int)
    sampled_X["age"] = sampled_X["age"].round().astype(int)
    sampled_X["gender"] = sampled_X["gender"].round().astype(int)
    sampled_X["VAS score"] = sampled_X["VAS score"].round().astype(int)
    sampled_X["BMI"] = sampled_X["BMI"].round(2)

    synthetic_rows = len(sampled_X) - len(X_train)
    if synthetic_rows != SYNTHETIC_PER_CLASS * 2:
        raise ValueError(f"Expected 500 fold-local synthetic rows; generated {synthetic_rows}.")

    metadata = {
        "real_training_rows": int(len(X_train)),
        "synthetic_training_rows": int(synthetic_rows),
        "augmented_training_rows": int(len(sampled_X)),
        "real_training_by_target": {
            str(key): int(value) for key, value in class_counts.items()
        },
        "synthetic_training_by_target": {
            "0": SYNTHETIC_PER_CLASS,
            "1": SYNTHETIC_PER_CLASS,
        },
    }
    return sampled_X, sampled_y, metadata


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


def cross_validate_with_fold_augmentation(X, y, models, cv, subject_ids=None):
    """Compare models with real-only outer validation folds."""
    X = X[FEATURES].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
    if subject_ids is None:
        subject_ids = pd.Series(range(1, len(y) + 1))
    else:
        subject_ids = pd.Series(subject_ids).reset_index(drop=True)

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
        augmented_X, augmented_y, augmentation = augment_training_fold(X_train, y_train)

        for validation_index in validation_indices:
            fold_assignment_rows.append({
                "Fold": fold_number,
                "Subject ID": int(subject_ids.iloc[validation_index]),
                "Target": int(y.iloc[validation_index]),
                "Record Source": "real",
                "Validation Eligible": True,
            })

        for name, estimator in models.items():
            fold_model = clone(estimator)
            fold_model.fit(augmented_X, augmented_y)
            predictions = fold_model.predict(X_validation).astype(int)
            probabilities = positive_class_probability(fold_model, X_validation)
            out_of_fold_predictions[name][validation_indices] = predictions
            out_of_fold_probabilities[name][validation_indices] = probabilities
            fold_metric_rows.append({
                "Model": name,
                "Fold": fold_number,
                "Real Training Rows": augmentation["real_training_rows"],
                "Synthetic Training Rows": augmentation["synthetic_training_rows"],
                "Validation Real Rows": int(len(validation_indices)),
                "Validation Synthetic Rows": 0,
                **classification_metrics(y_validation, predictions),
            })

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    if (fold_metrics_df["Validation Synthetic Rows"] != 0).any():
        raise RuntimeError("Synthetic rows entered an outer validation partition.")
    if any((predictions < 0).any() for predictions in out_of_fold_predictions.values()):
        raise RuntimeError("Some original subjects did not receive an out-of-fold prediction.")

    return {
        "summary": summarize_fold_metrics(fold_metrics_df),
        "fold_metrics": fold_metrics_df,
        "fold_assignments": pd.DataFrame(fold_assignment_rows),
        "oof_predictions": out_of_fold_predictions,
        "oof_probabilities": out_of_fold_probabilities,
    }


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
):
    """Evaluate leakage-safe calibration and held-out permutation importance."""
    X = X[FEATURES].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
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
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        augmented_X, augmented_y, _ = augment_training_fold(X_outer_train, y_outer_train)
        baseline_model = clone(estimator).fit(augmented_X, augmented_y)

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
        calibrated_training_X, calibrated_training_y, calibration_augmentation = (
            augment_training_fold(X_base, y_base)
        )
        prefit_model = clone(estimator).fit(calibrated_training_X, calibrated_training_y)
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
            calibration_fold_rows.append({
                "Variant": variant,
                "Fold": fold_number,
                "Outer Validation Real Rows": int(len(validation_indices)),
                "Outer Validation Synthetic Rows": 0,
                "Calibration Real Rows": (
                    0 if variant == "Uncalibrated" else int(len(calibration_indices))
                ),
                "Calibration Synthetic Rows": 0,
                "Base Real Training Rows": (
                    int(len(outer_train_indices))
                    if variant == "Uncalibrated"
                    else int(len(base_indices))
                ),
                "Base Synthetic Training Rows": (
                    SYNTHETIC_PER_CLASS * 2
                    if variant == "Uncalibrated"
                    else calibration_augmentation["synthetic_training_rows"]
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
                    "Held-Out Real Rows": int(len(validation_indices)),
                    "Held-Out Synthetic Rows": 0,
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


def fit_final_calibrated_model(X_real, y_real, estimator):
    X_real = X_real[FEATURES].reset_index(drop=True)
    y_real = y_real.astype(int).reset_index(drop=True)
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
    )
    base_indices, calibration_indices = next(split.split(X_real, y_real))
    X_base = X_real.iloc[base_indices].reset_index(drop=True)
    y_base = y_real.iloc[base_indices].reset_index(drop=True)
    X_calibration = X_real.iloc[calibration_indices]
    y_calibration = y_real.iloc[calibration_indices]
    augmented_X, augmented_y, augmentation = augment_training_fold(X_base, y_base)
    prefit_model = clone(estimator).fit(augmented_X, augmented_y)
    calibrated_model = build_prefit_calibrator(prefit_model)
    calibrated_model.fit(X_calibration, y_calibration)
    metadata = {
        "base_real_training_rows": int(len(base_indices)),
        "base_synthetic_training_rows": augmentation["synthetic_training_rows"],
        "real_calibration_rows": int(len(calibration_indices)),
        "synthetic_calibration_rows": 0,
    }
    return calibrated_model, metadata
