"""Guard against invalid prevalence metadata in offline model comparisons."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.validate_saved_metrics import METRICS, validate


class FailurePrevalenceTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(METRICS.read_text(encoding="utf-8"))

    def write_and_validate(self, prevalence):
        with tempfile.TemporaryDirectory() as directory:
            data = dict(self.original)
            data["failure_rate"] = prevalence
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return validate(path)

    def test_existing_fractional_prevalence_passes(self):
        self.assertEqual(self.write_and_validate(0.0339), self.original["selected_model"])

    def test_non_numeric_and_out_of_range_prevalence_fails(self):
        for value in (-0.01, 1.01, True, "3.39%", math.nan, math.inf):
            with self.subTest(value=str(value)), self.assertRaises(AssertionError):
                self.write_and_validate(value)


if __name__ == "__main__":
    unittest.main()
