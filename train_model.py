from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "assets" / "plots"
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
NUMERIC = FEATURES[1:]
CATEGORICAL = ["Type"]


def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def main():
    df = pd.read_csv(DATA)
    print("Shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicates:", int(df.duplicated().sum()))
    print("Failure rate:", round(df[TARGET].mean() * 100, 2), "%")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=42
        ),
    }

    results = {}
    trained = {}
    for name, classifier in models.items():
        pipe = build_pipeline(classifier)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        trained[name] = pipe

    # Choose the model with the best failure-class F1 score.
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = trained[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    metadata = {
        "selected_model": best_name,
        "selection_rule": "Highest test-set F1 score for the failure class (simple student version)",
        "features": FEATURES,
        "target": TARGET,
        "failure_rate": float(y.mean()),
        "test_size": int(len(y_test)),
        "results": results,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))

    # 1. Class distribution
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(7, 4.5))
    plt.bar(["No Failure", "Failure"], counts.values)
    plt.title("Machine Failure Class Distribution")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_class_distribution.png", dpi=220)
    plt.close()

    # 2. Failure rate by product type
    by_type = df.groupby("Type")[TARGET].mean().mul(100)
    plt.figure(figsize=(7, 4.5))
    plt.bar(by_type.index, by_type.values)
    plt.title("Failure Rate by Product Type")
    plt.xlabel("Product Type")
    plt.ylabel("Failure Rate (%)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_failure_by_type.png", dpi=220)
    plt.close()

    # 3. RPM vs torque
    plt.figure(figsize=(7, 4.8))
    normal = df[df[TARGET] == 0]
    failed = df[df[TARGET] == 1]
    plt.scatter(normal["Rotational speed [rpm]"], normal["Torque [Nm]"], s=10, alpha=.25, label="No Failure")
    plt.scatter(failed["Rotational speed [rpm]"], failed["Torque [Nm]"], s=18, alpha=.75, label="Failure")
    plt.title("Rotational Speed vs Torque")
    plt.xlabel("Rotational Speed (rpm)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_rpm_torque_failure.png", dpi=220)
    plt.close()

    # 4. Tool wear by class
    plt.figure(figsize=(7, 4.5))
    plt.boxplot([
        normal["Tool wear [min]"],
        failed["Tool wear [min]"],
    ], tick_labels=["No Failure", "Failure"])
    plt.title("Tool Wear and Machine Failure")
    plt.ylabel("Tool Wear (min)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_tool_wear_failure.png", dpi=220)
    plt.close()

    # 5. Model F1 comparison
    names = list(results)
    f1_values = [results[n]["f1"] for n in names]
    plt.figure(figsize=(8, 4.8))
    plt.bar(names, f1_values)
    plt.title("Model Comparison - Failure Class F1 Score")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_model_comparison.png", dpi=220)
    plt.close()

    # 6. Best-model confusion matrix
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Failure", "Failure"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{best_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_confusion_matrix.png", dpi=220)
    plt.close()

    # 7. Random Forest feature importance (if selected; otherwise train one just for interpretation)
    rf_pipe = trained["Random Forest"]
    prep = rf_pipe.named_steps["preprocessor"]
    feature_names = list(NUMERIC) + list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    importances = rf_pipe.named_steps["classifier"].feature_importances_
    imp = pd.Series(importances, index=feature_names).sort_values().tail(8)
    plt.figure(figsize=(8, 4.8))
    plt.barh(imp.index, imp.values)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_feature_importance.png", dpi=220)
    plt.close()

    print("\nModel results:")
    for name, m in results.items():
        print(f"{name:20s} Accuracy={m['accuracy']:.3f} Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f}")
    print("\nSelected model:", best_name)
    print("Saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
