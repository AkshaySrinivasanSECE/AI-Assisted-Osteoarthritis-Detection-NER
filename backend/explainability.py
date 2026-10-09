import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_BASED_DISCLAIMER = (
    "This is a model-based explanation of how the fitted Logistic Regression combined "
    "the supplied inputs. It describes learned associations in this model and does not "
    "establish that any feature causes or prevents knee osteoarthritis."
)


def _logit(probability):
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return math.log(probability / (1 - probability))


def contribution_direction(value):
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "neutral"
    return "toward_oa" if value > 0 else "toward_healthy"


class LogisticRegressionExplainer:
    """Explain a StandardScaler -> binary LogisticRegression pipeline."""

    def __init__(self, model, features):
        self.features = list(features)
        self._validate_model(model)
        self.model = model
        self.scaler = model.named_steps["scaler"]
        self.classifier = model.named_steps["model"]
        self.coefficients = self.classifier.coef_[0].astype(float)
        self.intercept = float(self.classifier.intercept_[0])
        self.global_feature_importance = self._build_global_importance()

    def _validate_model(self, model):
        if not isinstance(model, Pipeline):
            raise TypeError("Coefficient explanations require a scikit-learn Pipeline.")
        if "scaler" not in model.named_steps or "model" not in model.named_steps:
            raise TypeError("Expected pipeline steps named 'scaler' and 'model'.")
        if not isinstance(model.named_steps["scaler"], StandardScaler):
            raise TypeError("Expected the selected model to use StandardScaler.")
        if not isinstance(model.named_steps["model"], LogisticRegression):
            raise TypeError("Coefficient explanations are implemented for Logistic Regression.")
        classifier = model.named_steps["model"]
        if list(classifier.classes_) != [0, 1]:
            raise ValueError(f"Expected model classes [0, 1]; found {list(classifier.classes_)}.")
        if classifier.coef_.shape != (1, len(self.features)):
            raise ValueError(
                "The Logistic Regression coefficient count does not match the saved feature list."
            )
        scaler = model.named_steps["scaler"]
        if len(scaler.mean_) != len(self.features) or len(scaler.scale_) != len(self.features):
            raise ValueError("The scaler statistics do not match the saved feature list.")

    def _build_global_importance(self):
        absolute_coefficients = np.abs(self.coefficients)
        total = float(absolute_coefficients.sum())
        relative = (
            np.zeros_like(absolute_coefficients)
            if math.isclose(total, 0.0, abs_tol=1e-15)
            else absolute_coefficients / total * 100
        )
        rows = []
        for index, feature in enumerate(self.features):
            coefficient = float(self.coefficients[index])
            rows.append({
                "feature": feature,
                "standardized_coefficient": round(coefficient, 6),
                "direction_when_feature_increases": contribution_direction(coefficient),
                "relative_importance": round(float(relative[index]), 2),
            })
        return sorted(rows, key=lambda row: row["relative_importance"], reverse=True)

    def explain(self, patient):
        if not isinstance(patient, pd.DataFrame) or len(patient) != 1:
            raise ValueError("A per-prediction explanation requires a one-row pandas DataFrame.")
        missing_features = sorted(set(self.features) - set(patient.columns))
        if missing_features:
            raise ValueError(f"Patient data is missing model features: {missing_features}")

        ordered_patient = patient[self.features]
        values = ordered_patient.iloc[0].astype(float).to_numpy()
        standardized_values = self.scaler.transform(ordered_patient)[0]
        contributions = standardized_values * self.coefficients
        absolute_total = float(np.abs(contributions).sum())
        relative_contributions = (
            np.zeros_like(contributions)
            if math.isclose(absolute_total, 0.0, abs_tol=1e-15)
            else np.abs(contributions) / absolute_total * 100
        )

        local_rows = []
        for index, feature in enumerate(self.features):
            contribution = float(contributions[index])
            local_rows.append({
                "feature": feature,
                "input_value": float(values[index]),
                "contribution_direction": contribution_direction(contribution),
                "relative_contribution": round(float(relative_contributions[index]), 2),
                "log_odds_contribution": round(contribution, 6),
            })
        local_rows.sort(key=lambda row: row["relative_contribution"], reverse=True)

        reconstructed_log_odds = self.intercept + float(contributions.sum())
        model_log_odds = float(self.model.decision_function(ordered_patient)[0])
        if not math.isclose(reconstructed_log_odds, model_log_odds, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError("Feature contributions do not reconstruct the model decision score.")

        return {
            "explanation_type": "model-based Logistic Regression explanation",
            "target_explained": "class 1: knee osteoarthritis risk pattern",
            "global_importance_method": (
                "Absolute Logistic Regression coefficients after StandardScaler, normalized "
                "to 100%. Larger values mean greater influence on this fitted model overall."
            ),
            "local_contribution_method": (
                "For this prediction, each contribution is the standardized input value "
                "multiplied by its Logistic Regression coefficient. Relative contribution is "
                "that term's absolute share of all four feature terms; the intercept is excluded."
            ),
            "direction_definition": {
                "toward_oa": "Raises the model's OA log-odds relative to its learned baseline.",
                "toward_healthy": "Lowers the model's OA log-odds relative to its learned baseline.",
                "neutral": "Has approximately zero contribution for this input.",
            },
            "baseline_log_odds": round(self.intercept, 6),
            "prediction_log_odds": round(model_log_odds, 6),
            "global_feature_importance": [dict(row) for row in self.global_feature_importance],
            "local_feature_contributions": local_rows,
            "disclaimer": MODEL_BASED_DISCLAIMER,
        }


class ModelAblationExplainer:
    """Provide model-agnostic probability explanations for non-linear estimators."""

    def __init__(self, model, features, reference_data):
        self.model = model
        self.features = list(features)
        self.reference_data = reference_data[self.features].copy()
        if self.reference_data.empty:
            raise ValueError("A non-empty reference dataset is required for explanations.")
        self.baseline = self.reference_data.median(numeric_only=True)
        self.global_feature_importance = self._build_global_importance()

    def _oa_probability(self, data):
        probabilities = self.model.predict_proba(data)
        classes = list(self.model.classes_)
        if 1 not in classes:
            raise ValueError(f"Model does not contain the OA target class 1: {classes}.")
        return float(probabilities[0, classes.index(1)])

    def _ablation_log_odds(self, data, feature):
        ablated = data.copy()
        ablated[feature] = self.baseline[feature]
        return _logit(self._oa_probability(ablated))

    def _build_global_importance(self):
        baseline_row = pd.DataFrame([self.baseline], columns=self.features)
        effects = []
        for feature in self.features:
            low_row = baseline_row.copy()
            high_row = baseline_row.copy()
            low_row[feature] = self.reference_data[feature].quantile(0.25)
            high_row[feature] = self.reference_data[feature].quantile(0.75)
            effect = _logit(self._oa_probability(high_row)) - _logit(
                self._oa_probability(low_row)
            )
            effects.append(float(effect))
        total = sum(abs(effect) for effect in effects)
        rows = []
        for feature, effect in zip(self.features, effects):
            relative = 0.0 if math.isclose(total, 0.0) else abs(effect) / total * 100
            rows.append({
                "feature": feature,
                "standardized_coefficient": round(effect, 6),
                "direction_when_feature_increases": contribution_direction(effect),
                "relative_importance": round(relative, 2),
            })
        return sorted(rows, key=lambda row: row["relative_importance"], reverse=True)

    def explain(self, patient):
        if not isinstance(patient, pd.DataFrame) or len(patient) != 1:
            raise ValueError("A per-prediction explanation requires a one-row pandas DataFrame.")
        ordered_patient = patient[self.features]
        prediction_log_odds = _logit(self._oa_probability(ordered_patient))
        baseline_row = pd.DataFrame([self.baseline], columns=self.features)
        baseline_log_odds = _logit(self._oa_probability(baseline_row))
        contributions = []
        for feature in self.features:
            contribution = prediction_log_odds - self._ablation_log_odds(ordered_patient, feature)
            contributions.append((feature, float(contribution)))
        total = sum(abs(contribution) for _, contribution in contributions)
        local_rows = []
        for feature, contribution in contributions:
            relative = 0.0 if math.isclose(total, 0.0) else abs(contribution) / total * 100
            local_rows.append({
                "feature": feature,
                "input_value": float(ordered_patient.iloc[0][feature]),
                "contribution_direction": contribution_direction(contribution),
                "relative_contribution": round(relative, 2),
                "log_odds_contribution": round(contribution, 6),
            })
        local_rows.sort(key=lambda row: row["relative_contribution"], reverse=True)
        return {
            "explanation_type": "model-based feature-ablation explanation",
            "target_explained": "class 1: knee osteoarthritis risk pattern",
            "global_importance_method": (
                "Interquartile feature-ablation effects from the selected fitted model, "
                "normalized to 100%."
            ),
            "local_contribution_method": (
                "Each local contribution is the change in model OA log-odds when that input "
                "is replaced by its active-dataset median."
            ),
            "direction_definition": {
                "toward_oa": "Raises the model's OA log-odds.",
                "toward_healthy": "Lowers the model's OA log-odds.",
                "neutral": "Has approximately zero ablation effect.",
            },
            "baseline_log_odds": round(baseline_log_odds, 6),
            "prediction_log_odds": round(prediction_log_odds, 6),
            "global_feature_importance": [dict(row) for row in self.global_feature_importance],
            "local_feature_contributions": local_rows,
            "disclaimer": (
                "This is a model-based feature-ablation explanation. It describes learned "
                "associations and does not establish that any feature causes or prevents OA."
            ),
        }


def build_explainer(model, features, reference_data):
    if isinstance(model, Pipeline) and isinstance(
        model.named_steps.get("model"), LogisticRegression
    ):
        return LogisticRegressionExplainer(model, features)
    return ModelAblationExplainer(model, features, reference_data)
