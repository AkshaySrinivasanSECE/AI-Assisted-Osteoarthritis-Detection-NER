import json
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


class DemonstrationDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
        cls.readme_lower = cls.readme.lower()

    def test_readme_contains_required_demonstration_sections(self):
        required_headings = [
            "## Project objective",
            "## Architecture",
            "## Technologies",
            "## Dataset",
            "## Synthetic data methodology",
            "## ML algorithms",
            "## Validation methodology",
            "## Model selection",
            "## API",
            "## Frontend",
            "## Installation",
            "## Execution commands",
            "## Demonstration workflow",
            "## Limitations",
            "## Ethical and medical disclaimer",
        ]
        for heading in required_headings:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.readme)

    def test_readme_documents_every_demo_api_endpoint(self):
        for endpoint in ("/health", "/model-info", "/evaluation", "/predict"):
            with self.subTest(endpoint=endpoint):
                self.assertIn(endpoint, self.readme)

    def test_stale_copied_results_and_placeholders_are_removed(self):
        stale_fragments = [
            "screenshots for demonstration can be added",
            '"healthy_probability": 24.75',
            '"oa_probability": 75.25',
            "| logistic regression | 0.716",
            "explainable ai\n- healthcare-worker dashboard",
        ]
        for fragment in stale_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.readme_lower)

    def test_documented_dataset_counts_match_metadata(self):
        metadata = json.loads(
            (ROOT_DIR / "models" / "model_metadata.json").read_text(encoding="utf-8")
        )
        data = metadata["data"]
        expected_statements = [
            f"**{data['original_observed_subjects']} original observed subjects**",
            f"**{data['synthetic_training_examples']} synthetic training examples**",
            f"**{data['active_training_rows']} active training rows**",
        ]
        for statement in expected_statements:
            with self.subTest(statement=statement):
                self.assertIn(statement, self.readme)

    def test_documented_selected_model_matches_comparison_artifact(self):
        metadata = json.loads(
            (ROOT_DIR / "models" / "model_metadata.json").read_text(encoding="utf-8")
        )
        comparison = pd.read_csv(ROOT_DIR / "docs" / "results" / "model_comparison.csv")
        selected = comparison.sort_values("F1 Mean", ascending=False, kind="stable").iloc[0][
            "Model"
        ]
        self.assertEqual(metadata["model_selection"]["selected_model"], selected)
        self.assertIn(
            f"identifies {selected} as the selected family",
            self.readme,
        )


if __name__ == "__main__":
    unittest.main()
