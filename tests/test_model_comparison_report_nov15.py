"""The comparison report must match saved evidence and disclose test reuse."""
import json
import unittest

from scripts.report_model_comparison import report
from scripts.validate_saved_metrics import METRICS


class ModelComparisonReportTests(unittest.TestCase):
    def test_ranking_is_sorted_and_matches_saved_f1(self):
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        result = report()
        expected = sorted(
            data["results"],
            key=lambda model: (-data["results"][model]["f1"], model),
        )
        self.assertEqual([row["model"] for row in result["ranking"]], expected)
        self.assertEqual(result["selected_model"], data["selected_model"])
        self.assertEqual(result["sample_size"], data["test_size"])
        self.assertEqual(result["failure_prevalence"], data["failure_rate"])
        self.assertIn("same test split", result["limitation"])
        self.assertIn("No independent final-test", result["limitation"])

    def test_report_is_stable_across_repeated_reads(self):
        self.assertEqual(report(), report())


if __name__ == "__main__":
    unittest.main()
