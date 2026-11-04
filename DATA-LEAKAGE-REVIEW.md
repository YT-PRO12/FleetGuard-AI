# Predictive-maintenance feature leakage review

The AI4I benchmark includes identifiers and machine failure-mode outputs that should not be used as input features for predicting the same target.

The intended predictors are the six operating features in [README](README.md): product type, two temperatures, rotational speed, torque, and tool wear.

## Audit before retraining

- Confirm `UDI`, `Product ID`, and the failure-mode labels `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from predictors.
- Fit encoders, imputers, and scalers on training data only.
- Inspect class prevalence separately in each split.
- Measure minority failure-class precision, recall and F1, not accuracy alone.
- Choose models on validation evidence; retain an untouched final test set for unbiased reporting.
- Verify the saved preprocessing pipeline is the same one used at inference.

AI4I is synthetic industrial data, not live fleet telemetry. Models must not be presented as real-world safety predictions without domain-specific validation.
