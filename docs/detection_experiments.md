# Detection experiments

The detection runner compares four methods under a frozen evaluation protocol:

| Method | Training data | Headline input |
| --- | --- | --- |
| `normal_range_rule` | Normal training rows define the 1st–99th percentile operating range | Physical signal; the rule is feature-independent |
| `isolation_forest` | Normal training rows only | Saved default feature table |
| `xgboost` | Labelled training rows | Saved default feature table |
| `lstm_autoencoder` | Normal training sequences only | Physical signal |

TensorFlow is a project dependency because the LSTM autoencoder is part of the
planned comparison. If an environment cannot install TensorFlow, the adapter
records the model as unavailable rather than silently replacing it with another
model.

For every fold, preprocessing is fitted only on training rows. No AHU name is
used as a predictor. Isolation Forest thresholds are fitted from normal
training scores, XGBoost class weighting is calculated from training labels,
and LSTM reconstruction thresholds are fitted from normal training sequences.
For LSTM evaluation, the first `sequence_length - 1` rows of each test AHU
stream have no complete within-test history and are reported as unscored rather
than assigned an artificial prediction. Sequence windows also break whenever
the elapsed time exceeds the configured one-hour cadence; a window can never
bridge an unobserved timestamp. Past-only rolling features use the same gap
boundary.

## Headline run

The former raw-feature detection runner has been removed. All headline pooled
detection uses the quality-controlled saved-feature runner described in
`docs/quality_detection.md`. This prevents an experiment from silently
rebuilding a different feature table inside the model command.

The operational-context feature set is a separate sensitivity comparison in
the results notebook. It is not silently mixed into the headline run.
