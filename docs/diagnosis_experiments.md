# Diagnosis experiments

Phase 9 separates fault diagnosis from binary detection. Only fault-labelled
rows are used for the diagnosis task; `Normal condition` rows are handled by
the detection experiment and are not treated as a diagnosis class.

The runner evaluates:

- exact fault-type XGBoost classification using the dataset's original labels;
- per-class and per-building metrics;
- confusion matrices;
- unsupported test classes, reported separately rather than counted as ordinary
  model errors;
- a minimum-support gate that excludes fault labels with fewer than 30 rows;
- a fixed 0.70 confidence threshold and abstention/coverage metrics.

Diagnosis uses the chronological, nested-grouped and building-held-out split
designs. The pooled AHU-held-out assignment is retained in the split artefact
but disabled for current modelling; the nested-grouped design provides the
stronger repeated unseen-AHU assessment.

The default headline feature variant is `physical_signal`. The confidence value
is the maximum XGBoost class score, not a calibrated probability. The report
must therefore describe it as model confidence unless a later calibration study
is completed. The project does not derive or evaluate broader fault groups:
the dataset contains seven original fault labels, and exact-type performance is
the more direct and transparent diagnosis result.

The minimum-support gate is applied only to diagnosis. Rare labels remain in the
canonical data and binary detection inputs for traceability; the manifest records
which labels and rows were excluded from diagnosis.

Run it after canonical ingestion and frozen split creation:

```bash
run_diagnosis_experiments \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/diagnosis \
  --config configs/diagnosis.yaml \
  --building-config configs/multibuilding_ahu.yaml
```

The output includes a summary table, per-class metrics, per-building metrics,
confusion matrices, row-level predictions with confidence and abstention flags,
a manifest and a Markdown summary. If a fault class is absent from a training
fold, its test rows are counted as `unsupported_rows` and are not disguised as
ordinary poor classification performance.
