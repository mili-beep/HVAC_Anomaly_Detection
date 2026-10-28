# Configuration

`auditorium.yaml` is the single frozen policy for the auditorium source,
feature roles, cleaning rules, feature engineering and normal-operation
pattern discovery. Keeping these related declarations together prevents the
same sensor list from drifting across phase-specific files.

`multibuilding_ahu.yaml` remains separate because it describes source-to-
canonical mappings for all three buildings rather than the auditorium policy.
Future building policies should follow the same one-file-per-building pattern.

`evaluation_splits.yaml` contains only evaluation-design settings. It is used
by `create_evaluation_splits` to freeze the four leakage-safe split designs;
it does not contain model hyperparameters.

`baselines.yaml` defines the three headline transparent detection comparisons
and their feature groups. It is kept separate from model configurations
because these are diagnostic benchmarks, not the final supervised model.

`detection.yaml` freezes the model settings and detection thresholds used by
the normal operating-range rule, Isolation Forest, XGBoost and optional LSTM
tracks.

`diagnosis.yaml` freezes the fault-group mapping, XGBoost diagnosis settings,
and confidence threshold used by the Phase 9 flat and hierarchical diagnosis
comparisons.
