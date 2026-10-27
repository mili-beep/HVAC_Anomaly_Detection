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

## 2026-10-27 — Phase 2 auditorium policy implemented

The auditorium policy is frozen in
`configs/auditorium.yaml` and documented in
`docs/auditorium_phase2_policy.md`. The source label mapping is explicitly:

```text
Normal condition -> normal
Every other label -> fault
```

The preparation command sorts rows by AHU and timestamp, preserves the raw
label and source row number, retains missing values as `NaN`, records temporal
gaps and per-feature missingness flags, and fails on invalid timestamps or
duplicate AHU/timestamp keys. No imputation or modelling is performed in this
phase.

The policy also defines disjoint `sensor_feature_columns`, `quality_columns`,
and `metadata_columns`. Feature selection must use these declarations rather
than an inferred exclusion list. The physical-signal variant uses only sensor
columns; the operational-context sensitivity variant adds the declared
missingness indicators; metadata and targets are never model inputs.

## 2026-10-27 — Phase 1 audit completed

The reproducible audit command is:

```bash
audit_multibuilding_ahu \
  --data-root /Users/mili/Downloads/27147678 \
  --output-dir results/audit
```

The generated manifest and summaries are kept in `results/audit/`. The audit
confirmed that the auditorium file is chronologically ordered and has no
duplicate AHU/timestamp keys; the hospital file has 184 duplicate keys (56
with conflicting labels) and literal dot placeholders; and the office and
hospital files are not fully ordered in their source row order. Missingness,
class support, calendar coverage, label associations, run lengths, and
numeric screening counts are recorded in the manifest. These findings remain
observations only; cleaning and inclusion decisions are deferred to Phase 2.

## 2026-10-27 — Phase 3 canonical ingestion

The three source schemas are mapped through
`configs/multibuilding_ahu.yaml` into the stable schema documented in
`docs/canonical_schema.md`. Ingestion preserves raw sensor values, source row
identifiers, labels and source text; it does not impute, interpolate,
deduplicate, relabel or remove rows. Hospital duplicate keys and literal
placeholders are retained with explicit quality flags for the Phase 4 policy.
The canonical adapter uses building-specific source mappings, so later
cleaning and modelling code does not need separate parsing implementations.

## 2026-10-27 — Phase 4 auditorium cleaning

Auditorium heating and cooling missingness is treated as structural operating
mode availability. Inactive subsystem fields remain missing. Only a bounded,
active-subsystem gap can be imputed, and its median must be fitted on the
training fold. The physical-signal variant excludes quality indicators; the
operational-context variant includes pre-declared mode and availability
indicators. The auditorium snapshot contains no eligible active gaps, so the
second audit records zero imputed rows and zero imputed cells.

## 2026-10-27 — Phase 5 auditorium feature engineering

Feature generation is label-independent and produces declared physical-signal
and operational-context variants. Relationships, past-only rolling means and
past-only rates are calculated after AHU/timestamp sorting. A shortcut audit
reports missingness and univariate label associations without changing the
feature set. Mode-dependent heating/cooling availability is explicitly flagged
for review rather than being silently treated as fault evidence.

## 2026-10-27 — Auditorium pattern discovery

Normal-operation pattern discovery is a descriptive artefact, separate from
feature engineering and model training. Profiles are calculated only from rows
whose raw label is `Normal condition`, grouped by AHU, operating mode, hour and
month. The output includes sensor summaries, normal mode transitions and PCA
visualisations. Fault rows are excluded from these profiles, and labels are not
used as model inputs. PCA uses deterministic per-mode sampling and median
filling only to create a two-dimensional visualisation; it must not be treated
as the Phase 4 training-fold imputation policy.
