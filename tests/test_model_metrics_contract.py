"""Read-only audit of the checked-in model comparison."""
import json
import math
import unittest
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "models" / "metrics.json"

class MetricsContractTest(unittest.TestCase):
    def test_saved_metrics_are_finite_probabilities(self):
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        self.assertIn(data["selected_model"], data["results"])
        self.assertGreater(data["test_size"], 0)
        for model, values in data["results"].items():
            with self.subTest(model=model):
                for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    value = values[metric]
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)

    def test_selected_model_has_highest_reported_failure_f1(self):
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        ranked = max(data["results"], key=lambda name: data["results"][name]["f1"])
        self.assertEqual(data["selected_model"], ranked)

if __name__ == "__main__":
    unittest.main()
