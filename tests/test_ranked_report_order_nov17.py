"""Ranked evidence should be deterministic without re-running training."""
import json
import unittest

from scripts.report_model_comparison import report
from scripts.validate_saved_metrics import METRICS


class ReportOrderingTests(unittest.TestCase):
    def test_ranking_agrees_with_saved_failure_class_f1(self):
        snapshot = json.loads(METRICS.read_text(encoding="utf-8"))
        result = report()
        expected = sorted(
            snapshot["results"],
            key=lambda name: (-snapshot["results"][name]["f1"], name),
        )
        self.assertEqual([item["model"] for item in result["ranking"]], expected)
        self.assertEqual(result["selected_model"], snapshot["selected_model"])
        self.assertEqual(result["sample_size"], snapshot["test_size"])
        self.assertIn("same test split", result["limitation"])

    def test_no_model_training_or_data_mutation_is_needed(self):
        self.assertEqual(report(), report())


if __name__ == "__main__":
    unittest.main()
