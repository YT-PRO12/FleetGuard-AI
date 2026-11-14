"""Render the saved synthetic benchmark comparison without retraining a model.

CAUTION: The recorded winner was chosen on the test split, so its reported
result is not an unbiased untouched-final-test estimate.
"""
import json
from pathlib import Path

from validate_saved_metrics import METRICS, validate


def report(path=METRICS):
    validate(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ranking = sorted(
        (
            {"model": name, "failure_f1": metrics["f1"],
             "precision": metrics["precision"], "recall": metrics["recall"]}
            for name, metrics in data["results"].items()
        ),
        key=lambda item: (-item["failure_f1"], item["model"]),
    )
    return {
        "selected_model": data["selected_model"],
        "sample_size": data["test_size"],
        "failure_prevalence": data["failure_rate"],
        "ranking": ranking,
        "limitation": (
            "Model selection used the same test split as reporting. "
            "No independent final-test or real-world fleet performance is claimed."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
