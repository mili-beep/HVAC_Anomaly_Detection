# Baseline comparisons

The repository evaluates three simple binary detectors before introducing
XGBoost. All three consume the same frozen evaluation assignments and report
fold-level metrics.

| Baseline | What it tells us |
| --- | --- |
| `majority_class` | The minimum benchmark obtained by always predicting the most common training class. |
| `normal_envelope_rule` | Whether a transparent range rule can detect values outside the normal training envelope. |
| `actuator_pump_only` | How much useful signal exists in fan, valve and pump measurements. |

The normal-envelope rule uses the complete configured physical sensor set. The
published-label distribution, missingness-only and temperature-only diagnostics
were excluded from the headline table: the first duplicates the majority
class, the second is a data-quality diagnostic rather than a physical detector,
and the third is redundant with the envelope comparison. Missingness remains
visible in the raw audit and operational-context sensitivity analysis. The
normal-envelope quantiles are fitted using normal rows in the training fold
only. The actuator/pump logistic baseline fits medians, scaling and coefficients
on training rows only.

Run the comparison with:

```bash
evaluate_detection_baselines \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/baselines \
  --config configs/baselines.yaml
```
