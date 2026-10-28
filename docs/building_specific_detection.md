# Building-specific detection

Step 9 evaluates whether a model trained within one building is stronger than
the pooled model. Auditorium, office and hospital are never mixed within a
building-specific fold. The saved `building_splits.csv` contains an AHU-held-
out and a chronological design for each building.

Run:

    run_building_detection \
      --data-file data/interim/cleaned/cleaned_ahu.csv \
      --feature-file data/interim/features/multibuilding_features.csv \
      --feature-manifest data/interim/features/feature_manifest.json \
      --splits-file data/interim/building_splits/building_splits.csv \
      --output-dir results/building_specific \
      --config configs/detection.yaml

The runner uses the default physical-signal feature list. For every
building/design it trains the normal-range rule, Isolation Forest, XGBoost and
LSTM (when TensorFlow is available) on the training partition and scores the
validation partition. The selection metric is declared in
`configs/detection.yaml` and is currently F1; balanced accuracy is used only
as a tie-break. Only validation results are used to choose a model. The
selected model is then refitted on training plus validation rows and scored
once on the held-out test partition.

Outputs:

- `building_detection_validation_metrics.csv`: validation metrics for every
  candidate model;
- `building_detection_selected_models.csv`: the selected model and validation
  score for each building/design;
- `building_detection_test_metrics.csv`: the selected model's final test
  result;
- `building_detection_manifest.json`: input paths, feature count, selection
  rule and model status;
- `building_detection_summary.md`: a concise staff/reporting summary.

The test file must not be used to choose a model. It is reserved for the final
building-specific estimate after model selection.
