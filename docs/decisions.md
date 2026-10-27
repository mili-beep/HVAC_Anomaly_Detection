# Initial decisions

## 2026-10-27 — Separate repository

The multibuilding AHU work is kept in `/Users/mili/Desktop/IRPN` rather than
being mixed with the completed RBC-ASHRAE1312 project. The older repository is
used only as a source of reusable engineering ideas.

## 2026-10-27 — Code-first implementation

Ingestion, audit, feature engineering, splitting, modelling, evaluation, and
artefact generation will be implemented in tested Python code. Notebooks will
be added later for presentation and interpretation, and will read saved
artefacts instead of becoming a second implementation.

## 2026-10-27 — No modelling before the audit gate

The initial dataset concerns—placeholders, duplicate hospital timestamps,
mode-dependent missingness, class imbalance, and possible AHU-specific label
shortcuts—must be measured and assigned an explicit policy before headline
models are trained.
