import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import generate_synthetic_data as generator
from model_utils import FEATURES, TARGET


ORIGINAL_PATH = ROOT_DIR / "data" / "full_indicators_summary.original.csv"
ACTIVE_PATH = ROOT_DIR / "data" / "full_indicators_summary.csv"
REPORT_PATH = ROOT_DIR / "data" / "synthetic_data_quality_report.json"


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_csv(ORIGINAL_PATH, encoding="gb18030")
        cls.active = pd.read_csv(ACTIVE_PATH, encoding="utf-8-sig")
        cls.real = cls.active[cls.active["record_source"] == "real"]
        cls.synthetic = cls.active[cls.active["record_source"] == "synthetic"]

    def test_dataset_shapes(self):
        self.assertEqual(self.original.shape, (88, 28))
        self.assertEqual(self.active.shape, (588, 9))
        self.assertEqual(self.real.shape[0], 88)
        self.assertEqual(self.synthetic.shape[0], 500)

    def test_target_values_are_binary(self):
        self.assertEqual(set(self.original[TARGET].unique()), {0, 1})
        self.assertEqual(set(self.active[TARGET].unique()), {0, 1})

    def test_synthetic_counts_are_balanced_and_exact(self):
        self.assertEqual(
            self.synthetic[TARGET].value_counts().sort_index().to_dict(),
            {0: 250, 1: 250},
        )

    def test_model_and_provenance_fields_have_no_missing_values(self):
        required = FEATURES + [
            TARGET,
            "record_id",
            "record_source",
            "generation_method",
            "validation_eligible",
        ]
        self.assertFalse(self.active[required].isna().any().any())
        self.assertFalse(self.original[FEATURES + [TARGET]].isna().any().any())

    def test_model_values_are_within_accepted_ranges(self):
        generator.validate_model_columns(self.original, "Original test dataset")
        generator.validate_model_columns(self.active, "Active test dataset")
        self.assertTrue(self.active["age"].between(18, 100).all())
        self.assertTrue(self.active["BMI"].gt(0).all())
        self.assertTrue(self.active["BMI"].le(80).all())
        self.assertTrue(self.active["VAS score"].between(0, 10).all())
        self.assertEqual(set(self.active["gender"].unique()), {0, 1})

    def test_provenance_labels_and_validation_eligibility(self):
        self.assertEqual(set(self.active["record_source"]), {"real", "synthetic"})
        self.assertEqual(set(self.real["generation_method"]), {"observed"})
        self.assertEqual(set(self.synthetic["generation_method"]), {"SMOTENC"})
        self.assertTrue(self.real["validation_eligible"].all())
        self.assertTrue(self.synthetic["validation_eligible"].all())
        self.assertTrue(self.real["record_id"].str.startswith("real_").all())
        self.assertTrue(self.synthetic["record_id"].str.startswith("synthetic_").all())

    def test_active_real_rows_match_preserved_original_records(self):
        expected = self.original[FEATURES + [TARGET]].reset_index(drop=True)
        observed = self.real[FEATURES + [TARGET]].reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)

    def test_synthetic_feature_vectors_are_unique_and_do_not_copy_real_rows(self):
        self.assertFalse(self.synthetic.duplicated(subset=FEATURES).any())
        real_keys = set(map(tuple, self.real[FEATURES].to_numpy()))
        synthetic_keys = set(map(tuple, self.synthetic[FEATURES].to_numpy()))
        self.assertTrue(real_keys.isdisjoint(synthetic_keys))

    def test_invalid_missing_range_category_and_target_values_are_rejected(self):
        invalid_cases = []

        missing = self.original.copy()
        missing.loc[0, "BMI"] = np.nan
        invalid_cases.append(missing)

        invalid_age = self.original.copy()
        invalid_age.loc[0, "age"] = 101
        invalid_cases.append(invalid_age)

        invalid_gender = self.original.copy()
        invalid_gender.loc[0, "gender"] = 2
        invalid_cases.append(invalid_gender)

        invalid_vas = self.original.copy()
        invalid_vas.loc[0, "VAS score"] = 11
        invalid_cases.append(invalid_vas)

        invalid_target = self.original.copy()
        invalid_target.loc[0, TARGET] = 3
        invalid_cases.append(invalid_target)

        for invalid in invalid_cases:
            with self.subTest(columns=invalid.columns.tolist()), self.assertRaises(ValueError):
                generator.validate_model_columns(invalid, "Invalid test dataset")

    def test_quality_report_matches_dataset_identity(self):
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        identity = report["dataset_identity"]
        self.assertEqual(identity["observed_subjects"], 88)
        self.assertEqual(identity["synthetic_training_examples"], 500)
        self.assertEqual(identity["total_active_training_rows"], 588)
        self.assertEqual(report["counts"]["synthetic_by_target"], {"0": 250, "1": 250})
        self.assertIn("not additional observed patients", identity["claim_guardrail"])


if __name__ == "__main__":
    unittest.main()
