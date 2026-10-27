# Auditorium Phase 4 cleaning and imputation policy

Phase 4 consumes the sorted Phase 2 auditorium table. It produces one cleaned
master table, explicit operating-mode and availability fields, and a second
audit. It does not create model-ready feature variants; those belong to Phase
5. It does not change the raw file or use labels to choose any value.

## Missingness decision

The auditorium contains mutually exclusive heating and cooling fields. A row
is classified as heating-active when at least one configured heating field is
observed, and cooling-active when at least one configured cooling field is
observed. Inactive subsystem fields remain missing. They are not filled with a
median, zero, or a value copied from another mode.

An active sensor missing run is eligible for imputation only when it is no more
than two consecutive hourly rows, remains within one AHU and operating mode,
does not cross a recorded time gap, and is bounded by observed values on both
sides. The value used is a median fitted on the training fold only. Long or
unresolved active gaps remain missing and are flagged.

The real auditorium snapshot has no eligible active sensor gaps: its missing
heating and cooling values are structural mode availability. Therefore the
Phase 4 command correctly reports zero imputed rows and zero imputed cells.

## Output boundary

The cleaned master table retains the raw sensor columns, identifiers, targets,
operating-mode fields and quality flags needed by later phases. Phase 4 does
not write separate physical-signal or operational-context CSVs. Phase 5 reads
this master table and writes the two declared model-ready predictor variants.

## Leakage controls and audit

`TrainingFoldMedianImputer` must be fitted on a training fold and then applied
to a validation or test fold. The CLI never fits an imputer on the complete
dataset. Every placeholder conversion and imputation is recorded per sensor;
the second audit reports changed, excluded, retained-with-flags and imputed
rows and cells.
