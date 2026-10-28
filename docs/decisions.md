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

## 2026-10-28 — Frozen quality-controlled modelling checkpoint

The existing pooled and transfer results are preserved as a baseline. New
quality-controlled outputs will use the two policies in
`configs/duplicate_policies.yaml`. Hospital duplicate AHU/timestamp keys are
ambiguous observations: the main policy excludes every row for an ambiguous
key, while the sensitivity policy retains all rows. Neither policy selects the
first row or averages conflicting observations. Imputation remains training-fold
only, and existing transfer outputs are not overwritten.

## 2026-10-28 — Cleaning second audit completed

The second audit re-read the canonical input and both saved policy outputs. All
checks passed. The main `exclude_ambiguous_keys` output contains 354,587 rows,
has zero duplicate AHU/timestamp keys, and excludes all 368 rows belonging to
184 ambiguous keys. The `retain_flagged_keys` sensitivity output contains all
354,955 input rows and retains the 184 duplicate keys (368 rows) with explicit
quality flags. Labels and source-row identities are preserved under both
policies. The deterministic cleaning step changed 8,208 rows containing
placeholder or other invalid sensor cells; 8,162 of those remain in the main
output after duplicate-key exclusion. No imputation was performed: both
policies report zero imputed rows and zero imputed cells. Detailed audit
artefacts are written to `data/interim/cleaned/cleaning_second_audit.*` and
`cleaning_verification_manifest.json`.

## 2026-10-27 — Phase 3 canonical ingestion

The three source schemas are mapped through
`configs/multibuilding_ahu.yaml` into the stable schema documented in
`docs/canonical_schema.md`. Ingestion preserves raw sensor values, source row
identifiers, labels and source text; it does not impute, interpolate,
deduplicate, relabel or remove rows. Hospital duplicate keys and literal
placeholders are retained with explicit quality flags for the Phase 4 policy.
The canonical adapter uses building-specific source mappings, so later
cleaning and modelling code does not need separate parsing implementations.

## 2026-10-27 — Frozen evaluation split designs

The project saves four evaluation assignments rather than using a random row
split: AHU-held-out within-building, chronological, building-held-out transfer,
and nested grouped validation. The `create_evaluation_splits` command writes
one row-assignment table, a manifest and a human-readable summary. The pooled
AHU-held-out assignment is retained but disabled; active pooled experiments use
the other three assignments so the statistical baseline and XGBoost are
compared on identical folds.

## 2026-10-28 — Disable pooled AHU-held-out modelling

The pooled `ahu_held_out` assignment remains saved for reproducibility and
possible supplementary analysis, but is disabled from current model runners,
headline comparisons and result notebooks. The active pooled designs are
chronological, building-held-out and nested-grouped. The separate
`building_ahu_held_out` experiment remains active because it evaluates an
unseen AHU within one specified building rather than the pooled split.

## 2026-10-27 — Detection model comparison matrix

The detection comparison uses the normal operating-range rule, Isolation Forest,
XGBoost and an LSTM autoencoder. The normal operating-range rule and LSTM use the
physical-signal track; Isolation Forest and XGBoost are tested with both the
physical-signal and operational-context variants. Every method consumes the
same frozen chronological, building-held-out and nested grouped assignments.
The pooled AHU-held-out assignment is retained but disabled. The LSTM is an optional TensorFlow adapter and must be reported as
unavailable when its dependency is absent, never replaced silently.

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

## 2026-10-28 — Shared multibuilding feature generation

Feature generation is now shared across auditorium, hospital and office. The
quality-controlled main table produced 354,587 rows with 88 default
physical-signal predictors and 135 operational-context predictors. The default
variant contains raw sensors, physical relationships and past-only rolling and
rate features. The secondary context variant adds missingness and availability
indicators, derived operating modes (including an explicit unknown mode when
neither subsystem has an observed signal), and cyclical time context. One
feature table and a manifest are used rather than duplicating the large table;
the manifest declares both column lists. No imputation was performed, and
labels were used only by the separate shortcut audit.

## 2026-10-28 — Gap-aware temporal features and sequences

Past-only rolling means and rates now reset at timestamp gaps larger than the
nominal one-hour cadence. The LSTM sequence builder uses the same boundary and
constructs windows only within contiguous building/AHU segments. It does not
fabricate missing rows or carry history across an outage. The gap rule is
declared in `configs/feature_engineering.yaml` and `configs/detection.yaml`,
and is covered by unit tests.

## 2026-10-27 — Auditorium pattern discovery

Normal-operation pattern discovery is a descriptive artefact, separate from
feature engineering and model training. Profiles are calculated only from rows
whose raw label is `Normal condition`, grouped by AHU, operating mode, hour and
month. The output includes sensor summaries, normal mode transitions and PCA
visualisations. Fault rows are excluded from these profiles, and labels are not
used as model inputs. PCA uses deterministic per-mode sampling and median
filling only to create a two-dimensional visualisation; it must not be treated
as the Phase 4 training-fold imputation policy.

## 2026-10-28 — Building-specific split assignments

Building-specific modelling is evaluated independently for auditorium, office,
and hospital. Each building receives its own AHU-held-out and chronological
assignments; no assignment mixes buildings. These assignments are written to
`data/interim/building_splits/` and do not replace the frozen pooled or
building-transfer assignments.

## 2026-10-28 — Quality-controlled pooled detection

The four detection methods (normal-range rule, Isolation Forest, XGBoost and
the LSTM autoencoder) were rerun on the cleaned master table under both frozen
duplicate policies. Headline inputs use the default physical-signal feature
variant and fold-local preprocessing. The main policy excludes ambiguous
duplicate keys; the sensitivity policy retains and flags them. Outputs are
stored separately under `results/quality_controlled/` so earlier experiments
remain reproducible and unchanged. LSTM coverage is reported explicitly when
gaps prevent a complete sequence; rows without a valid sequence are not
silently treated as normal.

## 2026-10-28 — Building-specific model selection

Independent building models use the default physical-signal features and the
saved building-specific AHU-held-out and chronological assignments. F1 is the
pre-declared validation selection metric. Every candidate model is assessed on
validation rows, the best candidate is selected separately for each
building/design, and only that selected model is refit with training plus
validation rows before one final test evaluation. Test scores are never used
to choose the model.

## 2026-10-29 — Focused XGBoost feature sensitivity

The practical pooled detection headline continues to use the default BMS
signals. A separate, deliberately small sensitivity experiment compares that
feature set with `symptom_only`, which excludes direct fan, valve, exchanger and
pump-state channels and their derived histories/interactions. It runs XGBoost
only, on the same duplicate-excluded rows and the same active pooled designs:
chronological, building-held-out and nested-grouped. This is a feature ablation,
not a fourth baseline and not a replacement for the practical model.

The resolved feature lists and source-count guard are frozen in
`configs/feature_variants.yaml`; the generated run manifest records the exact
columns used. The first complete comparison produced 88 default predictors and
35 symptoms-only predictors. Symptoms-only performance was lower on the
chronological and nested-grouped designs but higher on the building-held-out
design, so the practical model remains the headline while the sensitivity is
reported as evidence about shortcut dependence.

The same checkpoint adds a small zero-shot directional transfer table for
auditorium-to-office, office-to-auditorium and auditorium-plus-office-to-hospital
using the chronological fold. It is presented separately from the formal
building-held-out and matched pooled-versus-individual comparisons.

Diagnosis support is now reported from the saved assignments as a complete
train/test matrix for every active design, fold, building and eligible fault
type. A class with no training rows is explicitly marked unsupported rather than
being interpreted as an ordinary model error.
