# Configuration

`auditorium.yaml` is the single frozen policy for the auditorium source,
feature roles, cleaning rules, feature engineering and normal-operation
pattern discovery. Keeping these related declarations together prevents the
same sensor list from drifting across phase-specific files.

`multibuilding_ahu.yaml` remains separate because it describes source-to-
canonical mappings for all three buildings rather than the auditorium policy.
Future building policies should follow the same one-file-per-building pattern.

`evaluation_splits.yaml` contains only evaluation-design settings. It is used
by `create_evaluation_splits` to freeze the four assignment designs; the
pooled `ahu_held_out` assignment is retained for traceability but disabled for
current model runs. The active pooled designs are chronological,
building-held-out and nested-grouped. It does not contain model hyperparameters.

`baselines.yaml` defines the three headline transparent detection comparisons
and their feature groups. It is kept separate from model configurations
because these are diagnostic benchmarks, not the final supervised model.

`detection.yaml` freezes the model settings and detection thresholds used by
the normal operating-range rule, Isolation Forest, XGBoost and optional LSTM
tracks.

`diagnosis.yaml` freezes the XGBoost diagnosis settings and confidence
threshold used by the Phase 9 exact fault-type comparison. It also records the
minimum total support required for a fault label to enter diagnosis modelling.

`duplicate_policies.yaml` freezes the two pre-declared hospital duplicate-key
policies used by the quality-controlled experiments. It deliberately defines
neither first-row selection nor aggregation because the audit found conflicting
observations for some keys.

`clean_multibuilding_ahu` applies those deterministic policies to the canonical
table. It does not fit an imputer; model runners must fit imputation within
each training fold.

`verify_multibuilding_cleaning` performs the second audit and fails if row
identity, labels, timestamps, duplicate counts or quality flags do not match
the declared policy.

`feature_engineering.yaml` contains only shared feature-generation settings.
The canonical sensor list remains in `multibuilding_ahu.yaml`. The
`engineer_multibuilding_features` command creates one feature table plus a
manifest that declares the default physical-signal and secondary
operational-context column lists.

`evaluation_splits.yaml` is shared by the frozen pooled splits and the new
building-specific splits. `create_building_splits` creates independent
auditorium, hospital and office assignments. `create_quality_splits` creates
pooled assignments for a cleaned duplicate policy and keeps duplicate keys
together in chronological folds.
