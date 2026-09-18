# FleetGuard AI — Student Version

A deliberately simplified **Data Science + Machine Learning** predictive-maintenance project using the AI4I 2020 dataset.

## Why this version exists
The original engineering-heavy version included threshold optimization, extensive reports, deployment infrastructure, CI tests, and a large multi-page UI. This version keeps only concepts that are easy to explain in an internship viva.

## Problem
Predict whether a machine is likely to fail from six operating features:
- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

`UDI`, `Product ID`, and the failure-mode labels `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are not used as predictors.

## ML workflow
1. Load dataset
2. Inspect missing values and duplicates
3. Select legitimate input features
4. Stratified 80/20 train-test split
5. Scale numeric columns and one-hot encode Product Type
6. Train Logistic Regression, Decision Tree, Random Forest, and SVM
7. Compare Accuracy, Precision, Recall, F1 and ROC-AUC
8. Select the model with the strongest failure-class F1 score
9. Save the complete preprocessing + model pipeline with Joblib
10. Use the saved model in a simple Streamlit app

## Class imbalance
Only a small percentage of records are failures, so every classifier uses `class_weight="balanced"`. This gives the minority failure class more importance during training.

## Run locally
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Files to understand before your viva
- `train_model.py` — complete ML workflow
- `app.py` — simple 3-page application
- `models/metrics.json` — real model results
- `assets/plots/` — PPT-ready figures

## Dataset limitation
AI4I 2020 is a synthetic industrial predictive-maintenance benchmark. It is not real vehicle telemetry. FleetGuard AI is the project brand; the underlying dataset represents industrial machine operating conditions.
