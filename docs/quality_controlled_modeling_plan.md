# Quality-controlled and building-specific modelling plan

This plan follows the frozen multibuilding baseline. Existing pooled and
cross-building results are preserved and are not overwritten.

## Step 1 — Freeze existing work

Preserve the canonical table, audit, split assignments, pooled baselines,
detection, diagnosis, transfer results, notebooks and source code. Store new
outputs under `results/quality_controlled/` and `results/building_specific/`.

## Step 2 — Freeze duplicate policies

Use `configs/duplicate_policies.yaml`:

- `exclude_ambiguous_keys`: exclude every row for a duplicated
  `building + ahu + timestamp` key and retain the affected rows in the audit.
- `retain_flagged_keys`: retain every row and preserve the duplicate flag.

Neither policy selects the first row or averages conflicting rows.

## Step 3 — Build one cleaned multibuilding master table

Apply deterministic placeholder, non-finite, timestamp and duplicate handling to
all buildings. Preserve structural missingness and source-row traceability. Do
not perform global imputation.

Run it with:

```bash
clean_multibuilding_ahu \
  --input-file data/interim/canonical/canonical_ahu.csv \
  --output-dir data/interim/cleaned \
  --building-config configs/multibuilding_ahu.yaml \
  --duplicate-policy-config configs/duplicate_policies.yaml
```

## Step 4 — Verify cleaning

Re-audit row counts, duplicate counts, labels, timestamps and quality flags.
The main table must contain no duplicate AHU/timestamp keys, and the sensitivity
table must retain the expected hospital duplicate observations.

Run it with:

```bash
verify_multibuilding_cleaning \
  --input-file data/interim/canonical/canonical_ahu.csv \
  --output-dir data/interim/cleaned \
  --building-config configs/multibuilding_ahu.yaml \
  --duplicate-policy-config configs/duplicate_policies.yaml
```

## Step 5 — Share feature generation

Use `engineer_multibuilding_features` for all buildings. It writes one feature
table and a manifest containing two declared column variants. Headline models
select the default physical-signal predictors: raw sensors, physical
relationships and past-only temporal features. Operational-context predictors
add missingness/availability, operating-mode and cyclical time indicators;
they are shown only in Notebook 2 as a secondary XGBoost comparison. No
imputation is performed here.

Run it with:

    engineer_multibuilding_features \
      --input-file data/interim/cleaned/cleaned_ahu.csv \
      --output-dir data/interim/features \
      --building-config configs/multibuilding_ahu.yaml \
      --feature-config configs/feature_engineering.yaml

## Step 6 — Make sequences gap-aware

LSTM and past-only temporal features must break at missing timestamps. They must
not treat non-consecutive rows as consecutive observations or fabricate rows.
The shared feature settings use the nominal one-hour cadence as the temporal
contiguity limit. The LSTM uses the same one-hour limit from
`configs/detection.yaml`. A complete history is required within one contiguous
segment; the first rows after a gap are therefore unscored or have missing
past-only features rather than being filled from across the gap.

## Step 7 — Create building-specific splits

Generate AHU-held-out and chronological train/validation/test assignments
independently for auditorium, office and hospital. No building-specific split
may contain rows from another building.

Run it with:

    create_building_splits \
      --input-file data/interim/cleaned/cleaned_ahu.csv \
      --output-dir data/interim/building_splits \
      --config configs/evaluation_splits.yaml

## Step 8 — Re-run pooled quality-controlled models

Run the normal-range rule, Isolation Forest, XGBoost and LSTM using the default
features under both duplicate policies. Record metrics and LSTM coverage.
Generate a matched quality split table for each policy with
`create_quality_splits`, then run `run_quality_detection`. Existing results are
not overwritten; outputs belong under `results/quality_controlled/`.

## Step 9 — Run independent building models

Train and evaluate separate model sets for auditorium, office and hospital.
Fit imputers and scalers within each building and fold. Select the best model
using validation F1, with balanced accuracy as a tie-break, then refit that selected model on training plus
validation rows and evaluate it once on the held-out test data. The validation
comparison for every model and the selected-model test result are saved under
`results/building_specific/`.

Run it with:

    run_building_detection \
      --data-file data/interim/cleaned/cleaned_ahu.csv \
      --feature-file data/interim/features/multibuilding_features.csv \
      --feature-manifest data/interim/features/feature_manifest.json \
      --splits-file data/interim/building_splits/building_splits.csv \
      --output-dir results/building_specific \
      --config configs/detection.yaml

## Step 10 — Re-run diagnosis

Run pooled fault-type diagnosis on the main quality-controlled table. Report
rare and unsupported classes separately. Add building-specific diagnosis only
where class support is sufficient.

## Step 11 — Update notebooks

Notebook 2 presents pooled, building-specific and duplicate-sensitivity results.
Notebook 3 presents final diagnosis results. Notebook 4 presents only final
operational outputs.

## Step 12 — Validate and clean up

Test partition isolation, training-only imputation, gap-aware sequences,
duplicate-policy counts, feature exclusions and reproducibility. Remove only
obsolete files that are no longer referenced after the new evidence is verified.

## Acceptance criteria

- Both duplicate policies are reproducible and documented.
- The main cleaned table has unique AHU/timestamp keys.
- No imputation or scaling uses validation or test data.
- LSTM sequences do not cross time gaps.
- Each building has independently evaluated models.
- Existing transfer results remain unchanged.
