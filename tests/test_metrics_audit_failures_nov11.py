"""Negative cases for the local metrics audit; no training or model loading."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_saved_metrics import METRICS, validate


class MetricsAuditFailureTests(unittest.TestCase):
    def write_snapshot(self, directory, data):
        path = Path(directory) / "metrics.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_rejects_corrupted_f1_value(self):
        original = json.loads(METRICS.read_text(encoding="utf-8"))
        original["results"]["Random Forest"]["f1"] = 0.99
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                validate(self.write_snapshot(directory, original))

    def test_rejects_unknown_selected_model(self):
        original = json.loads(METRICS.read_text(encoding="utf-8"))
        original["selected_model"] = "Nonexistent model"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                validate(self.write_snapshot(directory, original))


if __name__ == "__main__":
    unittest.main()
