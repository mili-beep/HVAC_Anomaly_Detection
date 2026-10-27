# Auditorium Phase 2 data-quality policy

This policy is frozen before model training. It applies only to the
`auditorium_scientific_data.csv` source file.

## Scope

The auditorium file contains 113,375 rows from 13 AHUs covering 2022. The
Phase 1 audit found no invalid timestamps, duplicate AHU/timestamp keys, exact
duplicate rows, literal placeholders, or non-finite values. It did find 505
missing hourly observations and mode-dependent missingness in heating and
cooling fields.

## Label mapping

The source label is preserved in `label_raw`. The binary detection target is:

```text
Normal condition -> normal
Every other source label -> fault
```

The original fault label remains available for later multiclass diagnosis. A
binary target is not used to overwrite or discard the original diagnosis label.

## Timestamp and ordering

`Time` is parsed as a timezone-naive local timestamp. Prepared rows are sorted
by AHU and timestamp. The original row number is retained as `source_row_id`.
The source data is not modified.

Missing hourly observations are not synthesised. `hours_since_previous_observation`
and `is_time_gap` are recorded. Temporal features must not cross a gap.

## Missing values

Missing sensor values are retained as `NaN`. They are not replaced with zero
and are not forward-filled. A model pipeline may fit a median imputer on each
training fold only.

Two pre-declared feature variants will be compared later:

1. **Physical-signal headline variant:** training-fold median imputation with
   no missingness indicators as model features.
2. **Operational-context sensitivity variant:** the same numeric features plus
   feature-observed/missing indicators.

The choice between variants will not be made after inspecting results.

## Feature exclusions

AHU identity, source row number, timestamp, labels, and label-derived run
information are metadata or targets, not predictive features. They may be used
for grouping, splitting, traceability, and reporting.

## Quality failures

The preparation command fails if an auditorium file contains an unparseable
timestamp or duplicate AHU/timestamp key. This prevents silent changes to the
headline dataset.
