# Detection experiments

The detection runner compares four methods under a frozen evaluation protocol:

| Method | Training data | Feature variants |
| --- | --- | --- |
| `normal_range_rule` | Normal training rows define the 1st–99th percentile operating range | Physical signal; the rule is feature-independent |
| `isolation_forest` | Normal training rows only | Physical signal and operational context |
| `xgboost` | Labelled training rows | Physical signal and operational context |
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

## Run

First create the canonical table and frozen assignments, then run:

```bash
run_detection_experiments \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/detection \
  --config configs/detection.yaml \
  --building-config configs/multibuilding_ahu.yaml
```

The output contains fold metrics, grouped summaries, a manifest and a readable
Markdown table. The saved Phase 7 baseline results remain the simple-model
comparison; they are not overwritten by this command.
