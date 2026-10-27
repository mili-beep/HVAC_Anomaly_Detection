# Detection experiments

The detection runner compares four methods under a frozen evaluation protocol:

| Method | Training data | Feature variants |
| --- | --- | --- |
| `statistical_envelope` | Normal training rows define the 1st–99th percentile envelope | Physical signal; the rule is feature-independent |
| `isolation_forest` | Normal training rows only | Physical signal and operational context |
| `xgboost` | Labelled training rows | Physical signal and operational context |
| `lstm_autoencoder` | Normal training sequences only | Physical signal |

The statistical, Isolation Forest and XGBoost tracks run without a deep-learning
framework. The LSTM adapter requires TensorFlow and is recorded as unavailable
rather than being silently replaced by another model when TensorFlow is absent.

For every fold, preprocessing is fitted only on training rows. No AHU name is
used as a predictor. Isolation Forest thresholds are fitted from normal
training scores, XGBoost class weighting is calculated from training labels,
and LSTM reconstruction thresholds are fitted from normal training sequences.

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
