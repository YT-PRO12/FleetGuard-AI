# FleetGuard AI — Viva Guide

## 30-second explanation
FleetGuard AI is a binary-classification project using the AI4I 2020 predictive-maintenance dataset. I use six operating features to predict machine failure. I compare Logistic Regression, Decision Tree, Random Forest and SVM. Because failures are rare, I use class-weight balancing and look at precision, recall and F1 instead of accuracy alone. The selected trained pipeline is saved with Joblib and used by a Streamlit prediction page.

## Questions you should be able to answer

### What is the target?
`Machine failure`: 0 means no failure and 1 means failure.

### Why did you drop UDI and Product ID?
They are identifiers, not useful operating measurements.

### Why did you drop TWF, HDF, PWF, OSF and RNF from inputs?
They directly describe failure modes. Using them to predict the main failure label would leak target information.

### Why stratified train-test split?
The failure class is rare, so stratification keeps approximately the same failure proportion in training and test data.

### Why StandardScaler?
Logistic Regression and SVM are sensitive to feature scale. Scaling puts numeric inputs on comparable scales.

### Why OneHotEncoder?
Product Type is categorical (L/M/H), so it must be converted to numeric columns without implying an artificial order.

### Why class_weight='balanced'?
Failures are a minority. Balanced class weights increase the importance of failure examples during training.

### Why not use accuracy only?
A model predicting almost everything as no-failure can still have high accuracy. Recall and F1 show how well actual failures are detected.

### What is precision?
Of all cases predicted as failures, how many were actually failures.

### What is recall?
Of all real failures, how many the model detected.

### What is F1-score?
The harmonic mean of precision and recall; useful when both matter.

### Why Random Forest?
In this simplified workflow it gives the strongest failure-class F1 score while remaining easy to explain. It combines many decision trees and averages their decisions.

### How does the app work?
The user enters six values. Streamlit creates one row, the saved preprocessing+model pipeline transforms it, `predict_proba` returns failure probability, and `predict` returns 0 or 1.

### Is it a vehicle dataset?
No. AI4I 2020 is synthetic industrial machine data. FleetGuard AI is the project branding for predictive maintenance.
