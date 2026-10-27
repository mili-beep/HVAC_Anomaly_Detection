# Auditorium Phase 5 feature-engineering policy

Phase 5 consumes the Phase 4 cleaned table. It creates reproducible predictors,
a feature-lineage table, and a shortcut audit. Feature engineering is
label-independent; labels are used only by the separate shortcut audit.

## Declared feature variants

The **physical-signal** variant contains:

- the 15 raw numeric sensor signals;
- setpoint, supply/return, temperature-span, actuator and pump relationships;
- past-only three-row and 24-row rolling means;
- past-only rates of change.

The **operational-context** variant adds operating-mode indicators, sensor
availability indicators, missingness indicators and selected quality flags. It
is a pre-declared sensitivity comparison, not a feature set chosen after
seeing model scores.

Identifiers, timestamps, labels and targets are retained only as keys for
joining and evaluation. They are never predictor columns.

## Temporal leakage controls

Rows are sorted by AHU and timestamp before temporal features are calculated.
Rolling means use `shift(1)` before the rolling window, so the current row and
future rows cannot contribute. Rates of change are calculated only within one
AHU and only when the elapsed time is positive and no more than two hours.

The feature generator rejects duplicate AHU/timestamp keys and unparseable
timestamps. Imputation is not performed here; Phase 4's training-fold-only
imputer must be applied inside each later train/validation split.

## Shortcut audit

The audit reports normal/fault missingness rates, missingness-rate gaps,
value-based ROC-AUC and missingness-based ROC-AUC for every declared predictor.
Flags identify features requiring review; they do not remove features or alter
the model input. In the auditorium data, heating/cooling availability is
expected to be strongly associated with operating mode, so those flags provide
an explicit warning that the operational-context variant may encode subsystem
availability rather than fault physics.
