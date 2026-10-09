import json
import math
import unittest
from pathlib import Path

import joblib
import pandas as pd

from backend.explainability import build_explainer, contribution_direction
from backend.main import PatientData, predict


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "oa_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "features.pkl"


class ExplanationCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_PATH)
        cls.features = joblib.load(FEATURES_PATH)
        active = pd.read_csv(ROOT_DIR / "data" / "full_indicators_summary.csv")
        cls.explainer = build_explainer(cls.model, cls.features, active)
        cls.patient = pd.DataFrame([{
            "age": 60,
            "gender": 0,
            "BMI": 26.71,
            "VAS score": 5,
        }])
        cls.explanation = cls.explainer.explain(cls.patient)

    def test_local_contributions_reconstruct_model_log_odds(self):
        if not hasattr(self.explainer, "coefficients"):
            self.skipTest("The unified active-data model is not coefficient-based.")
        local_sum = sum(
            row["log_odds_contribution"]
            for row in self.explanation["local_feature_contributions"]
        )
        reconstructed = self.explanation["baseline_log_odds"] + local_sum
        model_log_odds = float(self.model.decision_function(self.patient)[0])
        self.assertAlmostEqual(reconstructed, model_log_odds, places=5)

    def test_local_contributions_equal_coefficient_times_standardized_value(self):
        if not hasattr(self.explainer, "coefficients"):
            self.skipTest("The unified active-data model is not coefficient-based.")
        standardized = self.explainer.scaler.transform(self.patient)[0]
        expected = standardized * self.explainer.coefficients
        rows = {
            row["feature"]: row for row in self.explanation["local_feature_contributions"]
        }
        for index, feature in enumerate(self.features):
            self.assertAlmostEqual(
                rows[feature]["log_odds_contribution"],
                float(expected[index]),
                places=5,
            )

    def test_relative_local_contributions_sum_to_100_percent(self):
        relative_total = sum(
            row["relative_contribution"]
            for row in self.explanation["local_feature_contributions"]
        )
        self.assertAlmostEqual(relative_total, 100.0, places=1)

    def test_global_importance_covers_all_features_and_sums_to_100_percent(self):
        global_rows = self.explanation["global_feature_importance"]
        self.assertEqual({row["feature"] for row in global_rows}, set(self.features))
        self.assertAlmostEqual(
            sum(row["relative_importance"] for row in global_rows),
            100.0,
            places=1,
        )

    def test_global_importance_uses_normalized_absolute_coefficients(self):
        if not hasattr(self.explainer, "coefficients"):
            self.skipTest("The unified active-data model is not coefficient-based.")
        expected = abs(self.explainer.coefficients)
        expected = expected / expected.sum() * 100
        rows = {
            row["feature"]: row for row in self.explanation["global_feature_importance"]
        }
        for index, feature in enumerate(self.features):
            self.assertAlmostEqual(
                rows[feature]["relative_importance"],
                float(expected[index]),
                places=2,
            )

    def test_direction_labels_match_contribution_sign(self):
        for row in self.explanation["local_feature_contributions"]:
            self.assertEqual(
                row["contribution_direction"],
                contribution_direction(row["log_odds_contribution"]),
            )

    def test_feature_input_values_are_preserved(self):
        rows = {
            row["feature"]: row for row in self.explanation["local_feature_contributions"]
        }
        for feature in self.features:
            self.assertEqual(rows[feature]["input_value"], float(self.patient.iloc[0][feature]))

    def test_mean_feature_vector_has_zero_local_contributions(self):
        baseline = (
            self.explainer.scaler.mean_
            if hasattr(self.explainer, "scaler")
            else self.explainer.baseline
        )
        mean_patient = pd.DataFrame([dict(zip(self.features, baseline))])
        explanation = self.explainer.explain(mean_patient)
        for row in explanation["local_feature_contributions"]:
            self.assertEqual(row["contribution_direction"], "neutral")
            self.assertEqual(row["relative_contribution"], 0.0)
            self.assertTrue(math.isclose(row["log_odds_contribution"], 0.0, abs_tol=1e-10))

    def test_explanation_is_explicitly_model_based_and_non_causal(self):
        self.assertIn("model-based", self.explanation["explanation_type"].lower())
        self.assertIn("does not establish", self.explanation["disclaimer"].lower())
        self.assertIn("causes", self.explanation["disclaimer"].lower())


class PredictionEndpointExplanationTests(unittest.TestCase):
    def test_prediction_endpoint_returns_explanation_contract(self):
        response = predict(PatientData(age=60, gender=0, BMI=26.71, VAS_score=5))
        self.assertIn("explanation", response)
        explanation = response["explanation"]
        self.assertEqual(len(explanation["global_feature_importance"]), 4)
        self.assertEqual(len(explanation["local_feature_contributions"]), 4)
        required_local_fields = {
            "feature",
            "input_value",
            "contribution_direction",
            "relative_contribution",
            "log_odds_contribution",
        }
        for row in explanation["local_feature_contributions"]:
            self.assertTrue(required_local_fields.issubset(row))
        json.dumps(response)

    def test_explanation_does_not_change_prediction(self):
        payload = PatientData(age=60, gender=0, BMI=26.71, VAS_score=5)
        response = predict(payload)
        patient = pd.DataFrame([{
            "age": payload.age,
            "gender": payload.gender,
            "BMI": payload.BMI,
            "VAS score": payload.VAS_score,
        }])
        model = joblib.load(MODEL_PATH)
        expected_prediction = int(model.predict(patient)[0])
        expected_probability = round(float(model.predict_proba(patient)[0, 1]) * 100, 2)
        self.assertEqual(response["prediction"], expected_prediction)
        self.assertEqual(response["oa_probability"], expected_probability)


if __name__ == "__main__":
    unittest.main()
