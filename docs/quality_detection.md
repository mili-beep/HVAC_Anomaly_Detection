# Quality-controlled pooled detection

Step 8 reruns the four detection methods on the cleaned master table using the
default physical-signal feature columns:

- normal-range rule;
- Isolation Forest;
- XGBoost;
- LSTM autoencoder.

The run is performed once for each frozen duplicate policy. The main policy
excludes ambiguous duplicate keys; the sensitivity policy retains those rows,
flags them, and treats their timestamps as temporal boundaries. Pooled quality
splits are generated separately for each policy, with duplicate keys kept in
one chronological partition. Imputation and scaling are fitted inside each
training fold only.

The output reports fold-level F1, balanced accuracy, precision, recall,
false-positive rate and LSTM coverage. A low LSTM coverage is reported rather
than hidden when a test stream lacks a complete contiguous sequence.

Example run for the main policy:

    run_quality_detection \
      --data-file data/interim/cleaned/cleaned_ahu.csv \
      --feature-file data/interim/features/multibuilding_features.csv \
      --feature-manifest data/interim/features/feature_manifest.json \
      --splits-file data/interim/quality_splits/exclude_ambiguous_keys/evaluation_splits.csv \
      --output-dir results/quality_controlled/exclude_ambiguous_keys \
      --duplicate-policy exclude_ambiguous_keys

The retained-duplicate run uses the corresponding retained feature table,
split file and output directory. Existing pooled and transfer results are not
overwritten. The complete sensitivity command is:

    create_quality_splits \
      --input-file data/interim/cleaned/cleaned_ahu_retain_flagged_keys.csv \
      --output-dir data/interim/quality_splits/retain_flagged_keys \
      --config configs/evaluation_splits.yaml

    run_quality_detection \
      --data-file data/interim/cleaned/cleaned_ahu_retain_flagged_keys.csv \
      --feature-file data/interim/features/retain_flagged_keys/multibuilding_features.csv \
      --feature-manifest data/interim/features/retain_flagged_keys/feature_manifest.json \
      --splits-file data/interim/quality_splits/retain_flagged_keys/evaluation_splits.csv \
      --output-dir results/quality_controlled/retain_flagged_keys \
      --duplicate-policy retain_flagged_keys

The comparison is interpreted as a data-quality sensitivity check, not as a
choice made after seeing which model score is higher. In the completed run,
XGBoost remained the strongest pooled detector under both policies. Its mean
F1 was 0.743 when ambiguous duplicate keys were excluded and 0.737 when they
were retained. The small change supports the conclusion that the duplicate
policy is not driving the headline result. These are pooled results; the
building-specific models are a separate next step.

