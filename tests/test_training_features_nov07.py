"""Static training-contract tests; no model retraining or dataset modification."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def assignment(tree, name):
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in statement.targets):
                return ast.literal_eval(statement.value)
    raise AssertionError(f"Missing {name} assignment")


class TrainingFeaturesTest(unittest.TestCase):
    def test_predictors_do_not_include_identifiers_or_failure_labels(self):
        text = (ROOT / "train_model.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        features = assignment(tree, "FEATURES")
        target = assignment(tree, "TARGET")
        self.assertEqual(len(features), 6)
        self.assertEqual(len(set(features)), 6)
        self.assertEqual(target, "Machine failure")
        self.assertTrue(set(features).isdisjoint({
            "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", target,
        }))
        self.assertIn("X = df[FEATURES]", text)
        self.assertIn("y = df[TARGET]", text)


if __name__ == "__main__":
    unittest.main()
