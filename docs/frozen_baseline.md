# Frozen baseline checkpoint

This checkpoint records the existing multibuilding results before the
quality-controlled and building-specific experiments. Existing files must not
be overwritten by the new experiments.

The frozen baseline includes:

- `data/interim/canonical/canonical_ahu.csv` and its manifest;
- `data/interim/evaluation_splits/`;
- `results/audit/`;
- `results/baselines/`;
- `results/detection/`;
- `results/diagnosis/`;
- the current notebooks and source code.

The existing transfer results remain reference evidence. New quality-controlled
outputs belong under `results/quality_controlled/` and new building-specific
outputs belong under `results/building_specific/`.

The current working tree already contains an intentional user edit to
`notebooks/01_auditorium_pipeline_walkthrough.ipynb`; it is not part of this
checkpoint and must not be overwritten or staged by automation.

## Frozen policy decision

Duplicate AHU/timestamp keys are treated as ambiguous observations. The main
quality-controlled policy excludes every row belonging to an ambiguous key and
retains those rows in the audit. The sensitivity policy retains all rows and
preserves the duplicate flag. Neither policy selects the first row or averages
conflicting observations.
