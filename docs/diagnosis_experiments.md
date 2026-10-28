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
- a fixed 0.70 model-score threshold and abstention/coverage metrics.

Diagnosis uses the chronological, nested-grouped and building-held-out split
designs. The pooled AHU-held-out assignment is retained in the split artefact
but disabled for current modelling; the nested-grouped design provides the
stronger repeated unseen-AHU assessment.

The default headline feature variant is the saved default engineered feature
set. The `type_model_score` value is the maximum XGBoost class score, not a
calibrated probability. It must therefore be displayed and described as a
model score, not as a probability or calibrated confidence. The project does
not derive or evaluate broader fault groups:
the dataset contains seven original fault labels, and exact-type performance is
the more direct and transparent diagnosis result.

The minimum-support gate is applied only to diagnosis. Rare labels remain in the
canonical data and binary detection inputs for traceability; the manifest records
which labels and rows were excluded from diagnosis.

Run it after quality-controlled cleaning, saved Phase 5 feature generation and
quality-controlled split creation. Diagnosis must consume the same saved feature
table as detection; it must not rebuild a smaller feature set internally:

```bash
run_diagnosis_experiments \
  --data-file data/interim/cleaned/cleaned_ahu.csv \
  --feature-file data/interim/features/multibuilding_features.csv \
  --feature-manifest data/interim/features/feature_manifest.json \
  --splits-file data/interim/quality_splits/exclude_ambiguous_keys/evaluation_splits.csv \
  --output-dir results/diagnosis \
  --config configs/diagnosis.yaml \
  --building-config configs/multibuilding_ahu.yaml \
  --duplicate-policy exclude_ambiguous_keys
```

The input table contains one row per unambiguous building/AHU/timestamp key.
Rows excluded by the duplicate policy remain documented by the quality-control
audit and are not silently reintroduced for diagnosis.

The output includes a summary table, per-class metrics, per-building metrics,
confusion matrices, row-level predictions with model scores and abstention flags,
a manifest and a Markdown summary. If a fault class is absent from a training
fold, its test rows are counted as `unsupported_rows` and are not disguised as
ordinary poor classification performance.

## Diagnosis feature sensitivity

Notebook 4 also includes one deliberately narrow sensitivity run. It repeats
the same diagnosis protocol with the direct fan, valve, exchanger and pump
signals removed:

```bash
run_diagnosis_experiments \
  --data-file data/interim/cleaned/cleaned_ahu.csv \
  --feature-file data/interim/features/multibuilding_features.csv \
  --feature-manifest data/interim/features/feature_manifest.json \
  --splits-file data/interim/quality_splits/exclude_ambiguous_keys/evaluation_splits.csv \
  --output-dir results/diagnosis_symptom_only \
  --config configs/diagnosis_symptom_only.yaml \
  --building-config configs/multibuilding_ahu.yaml \
  --duplicate-policy exclude_ambiguous_keys \
  --variant-config configs/feature_variants.yaml
```

This is a robustness check, not a replacement headline model. A large drop in
pump-fault performance indicates that direct pump/status channels explain much
of the practical model's diagnosis result.
