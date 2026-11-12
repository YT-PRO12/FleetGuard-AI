"""Audit stored benchmark metadata, without retraining or changing the model."""
import json
import math
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "models" / "metrics.json"


def validate(path=METRICS):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data["results"]
    assert results and data["selected_model"] in results
    assert type(data["test_size"]) is int and data["test_size"] > 0
    assert len(data["features"]) == len(set(data["features"]))
    assert data["target"] not in data["features"]

    for name, item in results.items():
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            value = item[metric]
            assert type(value) in (int, float) and math.isfinite(value)
            assert 0 <= value <= 1, (name, metric, value)
        p, r = item["precision"], item["recall"]
        expected_f1 = 2 * p * r / (p + r) if p + r else 0
        assert math.isclose(item["f1"], expected_f1, abs_tol=1e-8), name

    winner = max(results, key=lambda name: results[name]["f1"])
    assert data["selected_model"] == winner, "Saved winner does not match F1 selection"
    return data["selected_model"]


if __name__ == "__main__":
    print(f"PASS: saved metrics consistent; selected model: {validate()}")
