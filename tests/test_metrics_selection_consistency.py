"""Static verification of saved benchmark metrics; no model retraining."""
import json
import math
import unittest
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "models" / "metrics.json"


class SavedModelSelectionTests(unittest.TestCase):
    def test_metrics_are_finite_and_selection_matches_reported_f1(self):
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        results = data["results"]
        self.assertGreater(len(results), 1)
        self.assertGreater(data["test_size"], 0)
        for name, item in results.items():
            with self.subTest(model=name):
                for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    self.assertTrue(math.isfinite(item[metric]))
                    self.assertGreaterEqual(item[metric], 0)
                    self.assertLessEqual(item[metric], 1)
        selected = max(results, key=lambda name: results[name]["f1"])
        self.assertEqual(data["selected_model"], selected)

    def test_target_and_failure_labels_are_not_features(self):
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        forbidden = {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
        self.assertTrue(forbidden.isdisjoint(data["features"]))
        self.assertEqual(data["target"], "Machine failure")


if __name__ == "__main__":
    unittest.main()
