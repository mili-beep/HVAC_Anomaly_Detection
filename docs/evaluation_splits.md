# Evaluation split implementation

The repository creates evaluation assignments before model training. The
assignments are saved in `data/interim/evaluation_splits/evaluation_splits.csv`
and are not regenerated inside a model run.

## Designs

| Design | Question answered | Leakage protection |
| --- | --- | --- |
| `ahu_held_out` | Can a model work on an unseen AHU in a known building? | All rows for an AHU stay in one partition. |
| `chronological` | Does performance survive a later time period? | Each AHU is split into past training, middle validation and future test rows. |
| `building_held_out` | Does a model transfer to an unseen building? | The held-out building is absent from training and validation. |
| `nested_grouped` | What is a robust estimate after model selection? | Outer AHU groups stay untouched while inner grouped validation selects settings. |

The statistical baseline and XGBoost must use the same assignment rows for a
paired comparison. The split command does not select features, fit imputers,
or train models; those operations belong inside each training fold later.

## Command

```bash
create_evaluation_splits \
  --input-file data/interim/canonical/canonical_ahu.csv \
  --output-dir data/interim/evaluation_splits \
  --config configs/evaluation_splits.yaml
```

The output manifest records the input, configuration, designs, fold counts and
row/group support. The Markdown summary is intended for quick inspection; the
CSV is the machine-readable source used by later experiments.
