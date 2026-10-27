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
