import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import evaluate_model
import train_model
from model_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_CONFIGURATION,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    build_models,
    cross_validate_with_fold_augmentation,
)


EXPECTED_MODELS = {
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "SVM",
}


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temp_directory.name)
        cls.temp_data = cls.temp_root / "data"
        cls.temp_models = cls.temp_root / "models"
        cls.temp_results = cls.temp_root / "docs" / "results"
        cls.temp_data.mkdir(parents=True)

        shutil.copy2(
            PROJECT_ROOT / "data" / "full_indicators_summary.csv",
            cls.temp_data / "full_indicators_summary.csv",
        )
        shutil.copy2(
            PROJECT_ROOT / "data" / "full_indicators_summary.original.csv",
            cls.temp_data / "full_indicators_summary.original.csv",
        )

        cls.patch_stack = ExitStack()
        patches = [
            patch.object(train_model, "ROOT_DIR", cls.temp_root),
            patch.object(train_model, "DATA_PATH", cls.temp_data / "full_indicators_summary.csv"),
            patch.object(
                train_model,
                "REAL_DATA_PATH",
                cls.temp_data / "full_indicators_summary.original.csv",
            ),
            patch.object(train_model, "MODELS_DIR", cls.temp_models),
            patch.object(train_model, "MODEL_PATH", cls.temp_models / "oa_model.pkl"),
            patch.object(train_model, "FEATURES_PATH", cls.temp_models / "features.pkl"),
            patch.object(
                train_model,
                "METADATA_PATH",
                cls.temp_models / "model_metadata.json",
            ),
            patch.object(train_model, "RESULTS_DIR", cls.temp_results),
            patch.object(evaluate_model, "RESULTS_DIR", cls.temp_results),
        ]
        for active_patch in patches:
            cls.patch_stack.enter_context(active_patch)

        with contextlib.redirect_stdout(io.StringIO()):
            train_model.main()

        cls.comparison = pd.read_csv(cls.temp_results / "model_comparison.csv")
        cls.fold_metrics = pd.read_csv(cls.temp_results / "model_fold_metrics.csv")
        cls.assignments = pd.read_csv(
            cls.temp_results / "validation_fold_assignments.csv"
        )
        cls.calibration_folds = pd.read_csv(
            cls.temp_results / "calibration_fold_metrics.csv"
        )
        cls.metadata = json.loads(
            (cls.temp_models / "model_metadata.json").read_text(encoding="utf-8")
        )
        cls.controlled_summary = pd.read_csv(
            PROJECT_ROOT
            / "docs"
            / "results"
            / "controlled_experiments"
            / "summary_metrics.csv"
        )

    @classmethod
    def tearDownClass(cls):
        cls.patch_stack.close()
        cls.temp_directory.cleanup()

    def test_complete_training_command_creates_required_artifacts(self):
        required_paths = [
            self.temp_models / "oa_model.pkl",
            self.temp_models / "features.pkl",
            self.temp_models / "model_metadata.json",
            self.temp_results / "model_comparison.csv",
            self.temp_results / "model_fold_metrics.csv",
            self.temp_results / "confusion_matrix.csv",
            self.temp_results / "feature_importance.csv",
        ]
        for path in required_paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_all_candidate_models_train_in_every_fold(self):
        self.assertEqual(set(self.comparison["Model"]), EXPECTED_MODELS)
        self.assertEqual(set(self.fold_metrics["Model"]), EXPECTED_MODELS)
        per_model_folds = self.fold_metrics.groupby("Model")["Fold"].nunique()
        self.assertTrue((per_model_folds == 5).all())
        self.assertEqual(len(self.fold_metrics), 25)

    def test_stratified_cross_validation_covers_each_real_subject_once(self):
        self.assertEqual(set(self.assignments["Fold"]), {1, 2, 3, 4, 5})
        self.assertEqual(len(self.assignments), 88)
        self.assertEqual(self.assignments["Subject ID"].nunique(), 88)
        self.assertEqual(set(self.assignments["Target"]), {0, 1})
        fold_targets = self.assignments.groupby("Fold")["Target"].nunique()
        self.assertTrue((fold_targets == 2).all())

    def test_synthetic_rows_never_enter_validation_or_calibration(self):
        self.assertTrue((self.fold_metrics["Validation Synthetic Rows"] == 0).all())
        self.assertEqual(set(self.assignments["Record Source"]), {"real"})
        self.assertTrue(self.assignments["Validation Eligible"].all())
        self.assertTrue(
            (self.calibration_folds["Outer Validation Synthetic Rows"] == 0).all()
        )
        self.assertTrue(
            (self.calibration_folds["Calibration Synthetic Rows"] == 0).all()
        )

    def test_model_selection_uses_highest_mean_f1(self):
        expected = self.comparison.sort_values(
            "F1 Mean", ascending=False, kind="stable"
        ).iloc[0]
        selection = self.metadata["model_selection"]
        self.assertEqual(selection["selected_model"], expected["Model"])
        self.assertAlmostEqual(selection["selected_mean_f1"], expected["F1 Mean"])
        self.assertIn("highest mean F1", selection["criterion"])

    def test_production_configuration_matches_controlled_experiment_selection(self):
        expected = self.controlled_summary.sort_values(
            "F1 Mean", ascending=False, kind="stable"
        ).iloc[0]
        configuration = self.metadata["feature_configuration"]
        self.assertEqual(configuration["name"], FEATURE_CONFIGURATION)
        self.assertEqual(configuration["name"], expected["Configuration"])
        self.assertEqual(configuration["features"], FEATURES)
        self.assertEqual(configuration["smotenc_categorical_features"], CATEGORICAL_FEATURES)
        self.assertEqual(configuration["vas_representation"], (
            "Categorical for SMOTENC sampling; integer-valued numeric input to estimators."
        ))
        self.assertFalse(configuration["womac_included"])
        self.assertEqual(self.metadata["model_selection"]["selected_model"], expected["Model"])
        self.assertAlmostEqual(
            self.metadata["model_selection"]["selected_mean_f1"], expected["F1 Mean"]
        )

    def test_saved_model_and_feature_order_load_and_predict(self):
        model = joblib.load(self.temp_models / "oa_model.pkl")
        features = joblib.load(self.temp_models / "features.pkl")
        real = pd.read_csv(
            self.temp_data / "full_indicators_summary.original.csv",
            encoding="gb18030",
        )
        sample = real[features].iloc[[0]]
        prediction = model.predict(sample)
        probabilities = model.predict_proba(sample)
        self.assertEqual(features, FEATURES)
        self.assertEqual(list(model.feature_names_in_), FEATURES)
        self.assertEqual(prediction.shape, (1,))
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_fixed_seed_produces_deterministic_predictions(self):
        real = pd.read_csv(
            self.temp_data / "full_indicators_summary.original.csv",
            encoding="gb18030",
        )
        X = real[FEATURES]
        y = real[TARGET]
        estimator = {"Logistic Regression": build_models()["Logistic Regression"]}

        def run_cross_validation():
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            return cross_validate_with_fold_augmentation(
                X,
                y,
                estimator,
                cv,
                subject_ids=real["number"],
            )

        first = run_cross_validation()
        second = run_cross_validation()
        np.testing.assert_array_equal(
            first["oof_predictions"]["Logistic Regression"],
            second["oof_predictions"]["Logistic Regression"],
        )
        np.testing.assert_allclose(
            first["oof_probabilities"]["Logistic Regression"],
            second["oof_probabilities"]["Logistic Regression"],
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
