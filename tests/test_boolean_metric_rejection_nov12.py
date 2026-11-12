"""JSON boolean values must not be treated as numeric performance metrics."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_saved_metrics import METRICS, validate


class BooleanMetricValidationTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(METRICS.read_text(encoding="utf-8"))

    def validate_snapshot(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return validate(path)

    def test_boolean_accuracy_is_rejected(self):
        self.original["results"]["Random Forest"]["accuracy"] = True
        with self.assertRaises(AssertionError):
            self.validate_snapshot(self.original)

    def test_boolean_test_size_is_rejected(self):
        self.original["test_size"] = True
        with self.assertRaises(AssertionError):
            self.validate_snapshot(self.original)


if __name__ == "__main__":
    unittest.main()
