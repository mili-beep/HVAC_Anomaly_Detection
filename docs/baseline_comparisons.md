# Baseline comparisons

The repository evaluates five simple binary detectors before introducing
XGBoost. All five consume the same frozen evaluation assignments and report
fold-level metrics.

| Baseline | What it tells us |
| --- | --- |
| `majority_class` | The minimum benchmark obtained by always predicting the most common training class. |
| `normal_envelope_rule` | Whether a transparent range rule can detect values outside the normal training envelope. |
| `missingness_only` | Whether sensor availability alone is acting as a shortcut. |
| `temperature_only` | How much useful signal exists in thermal measurements without controls or pumps. |
| `actuator_pump_only` | How much useful signal exists in fan, valve and pump measurements. |

The published-label distribution baseline was omitted because it is equivalent
to the majority-class predictor for this binary target. The normal-envelope
quantiles are fitted using normal rows in the training fold only. Logistic
baselines fit medians, scaling and coefficients on training rows only.

Run the comparison with:

```bash
evaluate_detection_baselines \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/baselines \
  --config configs/baselines.yaml
```
