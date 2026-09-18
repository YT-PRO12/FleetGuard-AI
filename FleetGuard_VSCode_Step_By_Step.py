# %% [markdown]
# FLEETGUARD AI — STEP-BY-

# HOW TO USE IN VS CODE
# 1. Open this file in VS Code.
# 2. Install the Python extension + Jupyter extension.
# 3. VS Code will show "Run Cell" above every # %% block.
# 4. Run blocks ONE BY ONE from top to bottom.
# 5. After each block, take your PPT screenshot from the output/graph.
#
# IMPORTANT:
# Keep the AI4I CSV inside:
#     data/ai4i2020.csv
#
# This version is intentionally simple enough to explain in a B.Tech viva.

# %%
# SECTION 1 — IMPORT LIBRARIES
# Screenshot idea: show this small import block only.

import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("Libraries imported successfully.")

# %%
# SECTION 2 — LOAD THE DATASET
# Screenshot idea: code + dataset shape + first 5 rows.

DATA_PATH = Path("data/ai4i2020.csv")

if not DATA_PATH.exists():
    # Helpful fallback if the script is run from another folder.
    possible = list(Path(".").rglob("ai4i2020.csv"))
    if not possible:
        raise FileNotFoundError(
            "Could not find ai4i2020.csv. Put it inside data/ai4i2020.csv"
        )
    DATA_PATH = possible[0]

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# %%
# SECTION 3 — BASIC DATA UNDERSTANDING
# Screenshot idea: this output is very useful for viva.

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

failure_count = int(df["Machine failure"].sum())
failure_rate = df["Machine failure"].mean() * 100

print("Machine failures:", failure_count)
print(f"Failure rate: {failure_rate:.2f}%")

print("\nData types:")
print(df.dtypes)

# %%
# SECTION 4 — CLASS DISTRIBUTION GRAPH
# PPT screenshot: MACHINE FAILURE CLASS DISTRIBUTION

class_counts = df["Machine failure"].value_counts().sort_index()

plt.figure(figsize=(7, 4.5))
sns.barplot(
    x=["No Failure", "Failure"],
    y=[class_counts.get(0, 0), class_counts.get(1, 0)]
)
plt.title("Machine Failure Class Distribution")
plt.ylabel("Number of Records")
plt.xlabel("Class")
plt.tight_layout()
plt.show()

# %%
# SECTION 5 — FAILURE RATE BY PRODUCT TYPE
# PPT screenshot: FAILURE RATE BY PRODUCT TYPE

failure_by_type = (
    df.groupby("Type")["Machine failure"]
    .mean()
    .mul(100)
    .reset_index(name="Failure Rate (%)")
)

print(failure_by_type)

plt.figure(figsize=(7, 4.5))
sns.barplot(data=failure_by_type, x="Type", y="Failure Rate (%)")
plt.title("Failure Rate by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Failure Rate (%)")
plt.tight_layout()
plt.show()

# %%
# SECTION 6 — RPM VS TORQUE
# PPT screenshot: one of the strongest EDA graphs.

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Rotational speed [rpm]",
    y="Torque [Nm]",
    hue="Machine failure",
    alpha=0.65
)
plt.title("Rotational Speed vs Torque")
plt.xlabel("Rotational Speed (rpm)")
plt.ylabel("Torque (Nm)")
plt.tight_layout()
plt.show()

# %%
# SECTION 7 — TOOL WEAR VS MACHINE FAILURE
# PPT screenshot: simple and easy to explain.

plt.figure(figsize=(7.5, 5))
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Tool wear [min]"
)
plt.xticks([0, 1], ["No Failure", "Failure"])
plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Status")
plt.ylabel("Tool Wear (minutes)")
plt.tight_layout()
plt.show()

# %%
# SECTION 8 — SELECT GENUINE INPUT FEATURES
# Screenshot idea: show this block in viva to explain leakage prevention.

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Input features used:")
for feature in FEATURES:
    print(" -", feature)

print("\nTarget:", TARGET)

print("\nNot used as predictors:")
print("UDI, Product ID, TWF, HDF, PWF, OSF, RNF")

# %%
# SECTION 9 — TRAIN / TEST SPLIT
# Screenshot idea: explain why stratify=y is used.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Train failure rate: {y_train.mean() * 100:.2f}%")
print(f"Test failure rate:  {y_test.mean() * 100:.2f}%")

# %%
# SECTION 10 — PREPROCESSING PIPELINE
# Screenshot idea: strong code screenshot for PPT.

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

print("Preprocessing created:")
print("- StandardScaler for numerical features")
print("- OneHotEncoder for Product Type")

# %%
# SECTION 11 — DEFINE FOUR CLASSICAL ML MODELS
# Screenshot idea: best code screenshot to prove model comparison.

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    ),
    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42,
    ),
}

print("Models ready:")
for model_name in models:
    print(" -", model_name)

# %%
# SECTION 12 — TRAIN AND COMPARE ALL MODELS
# PPT screenshot: console output + comparison table.

results = []
trained_pipelines = {}

for name, classifier in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    results.append(metrics)
    trained_pipelines[name] = pipeline

results_df = pd.DataFrame(results).sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))

# %%
# SECTION 13 — MODEL COMPARISON GRAPH
# PPT screenshot: very important results graph.

comparison = results_df.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(10, 5.5))
sns.barplot(
    data=comparison,
    x="Model",
    y="Score",
    hue="Metric",
)
plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=12)
plt.tight_layout()
plt.show()

# %%
# SECTION 14 — SELECT RANDOM FOREST
# We deliberately keep Random Forest as the final student-version model
# because it is strong, explainable, and easy to defend in viva.

best_model_name = "Random Forest"
best_model = trained_pipelines[best_model_name]

rf_pred = best_model.predict(X_test)
rf_prob = best_model.predict_proba(X_test)[:, 1]

rf_metrics = {
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(y_test, rf_pred, zero_division=0),
    "Recall": recall_score(y_test, rf_pred, zero_division=0),
    "F1 Score": f1_score(y_test, rf_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, rf_prob),
}

print("Selected Model:", best_model_name)
for metric, value in rf_metrics.items():
    print(f"{metric}: {value:.4f}")

# %%
# SECTION 15 — CONFUSION MATRIX
# PPT screenshot: final model evaluation.

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"],
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# %%
# SECTION 16 — RANDOM FOREST FEATURE IMPORTANCE
# PPT screenshot: excellent for explaining model behavior.

fitted_preprocessor = best_model.named_steps["preprocessor"]
rf_classifier = best_model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names
]

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf_classifier.feature_importances_,
    }
).sort_values("Importance", ascending=False)

print(importance_df.round(4).to_string(index=False))

plt.figure(figsize=(9, 5.5))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %%
# SECTION 17 — SAVE THE TRAINED MODEL
# Screenshot is optional; useful to explain deployment.

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

joblib.dump(best_model, MODEL_FILE)

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        {
            "selected_model": best_model_name,
            **{k: float(v) for k, v in rf_metrics.items()},
        },
        f,
        indent=2,
    )

print("Model saved to:", MODEL_FILE)
print("Metrics saved to:", METRICS_FILE)

# %%
# SECTION 18 — TEST ONE REAL PREDICTION
# PPT screenshot: show input + prediction output.

sample_machine = pd.DataFrame(
    [{
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 180,
    }]
)

prediction = int(best_model.predict(sample_machine)[0])
probability = float(best_model.predict_proba(sample_machine)[0, 1])

print("Sample Machine Input:")
print(sample_machine.to_string(index=False))

print("\nPrediction:", "Machine Failure" if prediction == 1 else "No Failure")
print(f"Failure Probability: {probability * 100:.2f}%")

# %%
# SECTION 19 — SIMPLE RISK LEVEL FOR PRESENTATION
# This is UI logic, not a new ML target.

if probability < 0.25:
    risk = "LOW"
elif probability < 0.50:
    risk = "MEDIUM"
elif probability < 0.75:
    risk = "HIGH"
else:
    risk = "CRITICAL"

print("Failure Probability:", f"{probability * 100:.2f}%")
print("Maintenance Risk Level:", risk)

# %% [markdown]
# END
#
# BEST SCREENSHOTS FOR YOUR PPT
#
# 1. Section 2  — Dataset loaded + first rows
# 2. Section 6  — RPM vs Torque graph
# 3. Section 10 — Preprocessing code
# 4. Section 11 — Four ML models code
# 5. Section 13 — Model comparison graph
# 6. Section 15 — Confusion matrix
# 7. Section 16 — Feature importance
# 8. Section 18 — Real prediction output
#
# You do NOT need to show every block in the presentation.
