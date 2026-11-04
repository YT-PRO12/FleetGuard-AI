# FleetGuard model evidence and limitations

This learning project evaluates machine-failure classifiers on the **AI4I 2020 synthetic industrial** benchmark. It does not use live fleet telemetry.

## Stored metrics

The reference values come from [models/metrics.json](models/metrics.json). The saved comparison records accuracy, precision, recall, failure-class F1, and ROC AUC. Failure is the minority outcome; high overall accuracy can coexist with poor precision for failure alerts.

The recorded selection procedure chooses the model with the largest **failure-class F1 on the test split**. Because the same test split was used to select the winner, the winner's performance is **not an unbiased untouched final-test estimate**. Treat it as a demonstration comparison. A more rigorous process would select using training/validation or cross-validation and reserve an untouched final test set.

## Usage boundaries

- No claim of real-world predictive maintenance or safety certification.
- No inference that the model generalizes to vehicles, turbines, factories, or new operating domains without external validation.
- Thresholds, class balance, prevalence, and monitoring costs would need domain-specific review before deployment.
- Preprocessing must be fit only on training records and reused unchanged at inference.

See the [project overview](README.md) and the checked-in model comparison before citing any metric.
