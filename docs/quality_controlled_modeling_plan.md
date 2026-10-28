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

Use one feature implementation for all buildings. Headline models use default
sensor and relationship features. Operational-context features are shown only
in Notebook 2 as a secondary XGBoost comparison.

## Step 6 — Make sequences gap-aware

LSTM and past-only temporal features must break at missing timestamps. They must
not treat non-consecutive rows as consecutive observations or fabricate rows.

## Step 7 — Create building-specific splits

Generate AHU-held-out and chronological train/validation/test assignments
independently for auditorium, office and hospital. No building-specific split
may contain rows from another building.

## Step 8 — Re-run pooled quality-controlled models

Run the normal-range rule, Isolation Forest, XGBoost and LSTM using the default
features under both duplicate policies. Record metrics and LSTM coverage.

## Step 9 — Run independent building models

Train and evaluate separate model sets for auditorium, office and hospital.
Fit imputers and scalers within each building and fold. Select the best model
using validation data only, then evaluate it once on the test data.

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
