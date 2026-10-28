# Multibuilding AHU Fault-Detection Project Plan

## Purpose

This plan defines a controlled rebuild of the project around the multibuilding AHU dataset in `/Users/mili/Downloads/27147678/`.

The existing SFSA repository is used as a source of reusable engineering ideas, not as a template to copy blindly. Components will be retained only when they fit the new dataset and research question.

The new project will investigate supervised AHU fault detection and diagnosis using real operational data from auditorium, hospital and office buildings. It will explicitly test whether performance survives unseen-AHU and unseen-building evaluation.

The project will proceed one phase at a time. Each phase has a deliverable and an acceptance gate. We do not build later phases on top of an unresolved earlier phase.

## Proposed research scope

### Primary question

Can supervised machine-learning models identify abnormal AHU operation and distinguish fault categories from real operational BMS data when evaluated on AHUs and buildings that were not used for training?

### Secondary questions

- How much performance comes from physical sensor relationships versus operating-mode and missingness information?
- Does a model trained on one building transfer to another building?
- Which fault classes are sufficiently represented to support credible diagnosis?
- How should uncertain or unsupported diagnoses be communicated to facilities staff?

### Recommended dataset role

- **Auditorium:** primary multiclass diagnosis dataset because all seven published classes are present and the timestamp structure is comparatively clean.
- **Office:** primary operational/generalisation dataset because it is the largest file and contains many AHUs.
- **Hospital:** data-quality and transfer case study. Its duplicated timestamps, placeholder values, severe AHU imbalance and rare classes must be handled explicitly before it can be used as a headline result.

This recommendation is provisional. Phase 1 may change it if the reproducible audit finds additional problems.

## What we will not assume from the old project

The following RBC-specific assumptions will not be carried into the new project unless evidence justifies them:

- paired normal/fault baseline files;
- one-minute sampling;
- complete 1,440-row fault sessions;
- fault onset at the first labelled row;
- filename-derived fault intensity;
- same-day baseline comparisons;
- weather-file pairing;
- controlled fault-injection interpretation;
- intensity prediction;
- RBC-specific fault taxonomy;
- RBC-specific feature names and units.

## Phase 0 — Project boundary, preservation and provenance

### Objectives

1. Preserve the finished RBC project unchanged.
2. Decide whether the new work is a separate repository, branch or clearly separated project tree.
3. Record the alternative dataset provenance, licence and intended citation.
4. Freeze the initial research questions before looking at model results.

### Tasks

- Do not overwrite the current RBC data, configurations or final results.
- Create a separate result root, for example `results/ahu_runs/`.
- Record the source DOI, download date, filenames and file checksums.
- Record that the source article describes real operational BMS data and expert-labelled operating conditions.
- Record the known initial concerns:
  - hospital placeholder values;
  - duplicated hospital AHU/timestamp keys;
  - mode-dependent missingness;
  - rare hospital classes;
  - AHU-specific label imbalance;
  - office leap-year/date-range interpretation.
- Decide whether the report will present this as a replacement project or as a new project informed by the original work.

### Deliverables

- Project decision record.
- Dataset provenance record.
- Raw-data checksum manifest.
- Separate result directory.

### Acceptance gate

No modelling starts until the raw files are preserved, checksummed and clearly separated from the original RBC project.

## Phase 1 — Reproducible raw-data audit

### Objective

Turn the exploratory audit into a repeatable command that can be run by the examiner.

### Audit checks

For each building file, report:

- file size and checksum;
- row count and column count;
- column names and inferred types;
- AHU count and AHU row counts;
- timestamp parse success;
- timestamp range;
- chronological ordering within each AHU;
- duplicate AHU/timestamp keys;
- exact duplicate rows;
- gaps and missing calendar dates;
- missing values by feature;
- literal placeholders such as `.`;
- non-finite values;
- unique-value counts and constant fields;
- numeric minima, maxima and suspicious ranges;
- label counts;
- label counts by AHU, month and hour;
- labels associated with missingness or placeholders;
- label run lengths;
- expected versus observed calendar coverage.

### Deliverables

- `audit_multibuilding_ahu` command or equivalent script.
- JSON manifest containing all audit results.
- Human-readable Markdown/CSV audit summary.
- Tests for the timestamp, duplicate and placeholder checks.

### Acceptance gate

The audit must produce the same results on two independent runs. Any unresolved issue must have an explicit decision: repair, exclude, retain with a flag, or use only as a limitation case study.

## Phase 2 — Data-quality and label policy

### Objective

Define what each problematic value means before implementing cleaning.

### Decisions required

#### Missingness categories

Classify missingness as:

1. structural/inactive subsystem;
2. short logging gap;
3. long logging outage;
4. sensor failure or fault symptom;
5. unresolved/invalid value.

#### Hospital placeholder values

Convert `.` to a formal missing value. Retain a separate missingness flag. Do not silently impute it.

The report must state whether this is treated as:

- a legitimate sensor-failure signal;
- a data-quality shortcut excluded from the physical-signal model;
- or both, through two declared feature variants.

#### Duplicate hospital timestamps

Define and document a deterministic rule for conflicting duplicate keys. Possible outcomes are:

- remove duplicates after selecting a documented authoritative record;
- aggregate only where aggregation is physically meaningful;
- retain a conflict flag and exclude the affected rows from headline evaluation.

#### Rare classes

Predefine the minimum support required for a headline class. A class below that threshold must be reported as insufficient, not silently removed after seeing results.

### Deliverables

- Data-quality policy document.
- Label dictionary and cross-building label map.
- Rare-class policy.
- Duplicate-resolution policy.
- Decision log for every excluded or transformed field.

### Acceptance gate

The policy is frozen before any model is trained. We do not change it because a later model result looks good or bad.

## Phase 3 — Canonical data model and ingestion adapter

### Objective

Create a clean, reusable representation for all three buildings.

### Implementation

Add a new adapter rather than modifying the RBC parser in place, for example:

```text
src/ahu_fdd/multibuilding.py
```

The canonical table should contain:

- `building`;
- `ahu`;
- `timestamp`;
- canonical sensor names;
- original source column names where traceability is required;
- `label`;
- raw-quality flags;
- missingness indicators;
- source-file row identifier.

### Requirements

- Preserve raw values before cleaning.
- Parse auditorium/office timestamps and hospital `DATE` plus `Time` consistently.
- Sort by building, AHU and timestamp after ingestion.
- Never include AHU name as a predictive feature.
- Keep source row identifiers so every model result can be traced back to the raw file.

### Deliverables

- New adapter.
- Building configuration file.
- Canonical schema documentation.
- Unit tests for all three source schemas.

### Acceptance gate

The adapter must load all three files, reproduce the audited row counts, preserve labels and expose every transformation in the manifest.

## Phase 4 — Cleaning and imputation

### Objective

Implement only the cleaning decisions approved in Phase 2. Phase 4 produces a
single cleaned master table; model-ready feature variants are generated only
in Phase 5.

### Cleaning outputs

- one cleaned master table containing numeric sensors, operating-mode fields and
  quality flags;
- a per-sensor audit of placeholders, invalid values, missingness and eligible
  gaps;
- a manifest and human-readable second-audit summary.

The physical-signal and operational-context model variants are created in
Phase 5, after cleaning decisions have been applied.

### Rules

- Fit imputers on training data only.
- Never use labels to select an imputation value.
- Do not interpolate across AHUs.
- Do not interpolate across long gaps.
- Do not impute inactive heating/cooling measurements as if they were active.
- Record every imputed row and column.

### Acceptance gate

The cleaned data must pass a second audit. The second audit must show exactly how many rows were changed, excluded, imputed or retained with flags.

## Normal-operation pattern discovery — supporting activity before modelling

### Objective

Describe normal AHU behaviour before selecting or evaluating models. This is a
separate descriptive activity, not a detector and not a replacement for the
Phase 5 feature-engineering work.

### Analysis

- use only rows labelled `Normal condition`;
- summarise sensor distributions by AHU, operating mode, hour and month;
- record normal-mode transitions within each AHU stream;
- create deterministic, mode-specific PCA visualisations of normal operation;
- exclude labels from all profile and PCA inputs;
- keep any PCA median filling local to the visualisation matrix and separate
  from training-fold imputation.

### Deliverables

- the multibuilding pattern-discovery command and its saved artefacts;
- normal-operation profile tables and transition summaries;
- PCA coordinates, explained-variance tables and figures;
- input-checksummed manifest and human-readable summary;
- tests proving fault rows are excluded and duplicate AHU/timestamp keys fail.

### Acceptance gate

The outputs must be reproducible from the cleaned auditorium table, clearly
identify the normal rows used, and state that the profiles are descriptive
artefacts rather than evidence of detector performance.

## Phase 5 — Feature engineering and shortcut audit

### Objective

Build features that are useful for AHU diagnosis while measuring shortcut dependence.

### Feature groups

1. Raw temperature and control measurements.
2. Setpoint residuals.
3. Supply/return temperature relationships.
4. Valve, fan and pump relationships.
5. Operating-mode indicators.
6. Missingness and data-quality flags.
7. Past-only rolling means.
8. Past-only rates of change.
9. Cross-sensor consistency checks.

### Required leakage checks

- Verify rolling windows are calculated after timestamp sorting.
- Verify rolling windows are past-only.
- Verify test rows are not used to fit imputers or scalers.
- Compare missingness-only performance.
- Compare AHU-identity-only performance as a diagnostic, never as a production feature.
- Compare temperature-only, actuator-only and full-feature models.
- Document whether any feature is part of the authors’ labelling criteria.

### Deliverables

- Feature dictionary.
- Feature lineage table.
- Shortcut audit table.
- Feature-generation tests.

### Acceptance gate

No feature is promoted to the headline model without a documented physical or operational justification.

## Evaluation split implementation

### Objective

Ensure the reported scores measure generalisation rather than memorisation.

### Required evaluation designs

#### Design A: AHU-held-out within-building evaluation

Train on some AHUs and test on unseen AHUs in the same building.

The pooled assignment is retained for traceability but is currently disabled
from headline modelling. The separate building-specific version remains active.

#### Design B: Chronological evaluation

Train on earlier observations and test on later observations.

#### Design C: Building-held-out transfer

Train on one or more buildings and test on a building not used for training or tuning.

#### Design D: Nested grouped validation

Use inner folds for model selection and an untouched outer fold for the final estimate.

### Rules

- No random row split as a headline result.
- No AHU name as a feature.
- Hyperparameters for cross-building tests must be selected using source buildings only.
- Report fold-level results, not only pooled rows.
- Report macro F1, balanced accuracy, per-class recall, binary detection F1 and false-positive rate.
- Report support for every class.

### Acceptance gate

The split implementation must pass synthetic tests proving that the same AHU cannot appear in both train and test for a grouped split.

## Baseline comparisons

### Objective

Establish whether machine learning adds value beyond simple alternatives.

### Baselines

1. Majority-class predictor.
2. Simple normal operating-range rule.
3. Actuator/pump-only classifier.

The published-label distribution baseline is not included as a separate run:
for this binary task it duplicates the information in the majority-class
predictor and adds no operationally distinct comparison. Missingness-only and
temperature-only diagnostics are retained conceptually in the audit and
feature-variant discussion, but are not headline detection baselines.

These baselines are essential because a high ML score is not meaningful if a
transparent physical rule achieves the same result. Missingness remains a
quality-control signal rather than a headline detector.

### Acceptance gate

The baseline table must be complete before selecting a headline model.

## Detection experiments

### Objective

Test normal-versus-fault detection under the frozen feature and split policies.

### Candidate models

- transparent rule/statistical baseline;
- logistic or linear baseline where appropriate;
- XGBoost as the main supervised model;
- one tree-based comparison model if it adds information;
- Isolation Forest only as a clearly labelled unsupervised comparison, not as the assumed best model.

### Required comparisons

- operational versus physical-signal feature variants;
- auditorium versus office versus hospital;
- AHU-held-out versus chronological versus building-held-out results;
- model results versus simple baselines.

### Acceptance gate

The binary model is not accepted as “best” until its advantage over the strongest simple baseline is shown under the same split.

## Phase 9 — Diagnosis experiments

### Objective

Evaluate exact fault-type classification separately from binary detection.

### Tasks

- flat multiclass classification;
- model scores and abstention for unsupported classes;
- per-class and per-building metrics;
- rare-class sensitivity analysis;
- confusion-matrix interpretation.

The implemented headline runner uses XGBoost with physical-signal features and
the dataset's original fault labels. Normal-versus-fault detection remains a
separate task. Fault classes missing from a training fold are recorded as
unsupported and excluded from ordinary classification metrics rather than being
reported as misleading poor scores.

### Rules

- Do not claim a class is diagnosed reliably when it has insufficient support.
- Do not combine absent classes with normal without documenting the consequence.
- Report “not evaluated” separately from a genuine poor prediction score.

### Acceptance gate

Detection and diagnosis results must appear as separate tables. A strong binary score must never be used as evidence of strong fault-type diagnosis.

## Phase 10 — Generalisation and robustness experiments

### Objective

Test whether the model is useful beyond the building and data pattern it saw during training.

### Experiments

- auditorium to office;
- office to auditorium;
- auditorium/office to hospital;
- pooled training to unseen building;
- physical-signal versus operational feature transfer;
- shortcut ablation;
- sensitivity to duplicate and missing-value policy.

### Acceptance gate

The report must include at least one strong transfer result and one failure case. Poor transfer is a finding, not a reason to hide the experiment.

## Phase 11 — Model scores and operational outputs

### Objective

Translate model probabilities into safe staff-facing outputs.

### Outputs

- anomaly status;
- predicted fault type;
- model score with explicit non-calibration wording;
- uncertain/abstain state;
- supporting sensor evidence;
- event start and end based on consecutive detector-alert rows;
- priority score using documented severity and criticality tables.

### Limitations to state

- hourly labels do not establish the physical fault onset;
- event duration is detected-alert duration, not verified physical duration;
- no fault intensity is available;
- asset criticality must be supplied by the client or treated as a documented assumption.

### Acceptance gate

Every example alert must be traceable to a saved row-level detector prediction, any matching diagnosis prediction, the model-score threshold, event grouping and priority-policy inputs. Normal detector rows must not become priority events, and unmatched alerts must remain explicitly uncertain.

## Phase 12 — Notebooks and examiner workflow

### Objective

Make the analysis understandable without duplicating the implementation.

### Notebooks

1. `01_multibuilding_pipeline_walkthrough.ipynb`
2. `02_detection_results.ipynb`
3. `03_building_detection_results.ipynb`
4. `04_diagnosis_results.ipynb`
5. `05_operational_outputs.ipynb`

The notebooks should call the package code. They should not contain a second hidden implementation.

### Requirements

- no absolute user-specific paths;
- no dependence on an unavailable external module;
- clear run order;
- saved summary tables;
- fixed random seeds;
- short smoke mode and full mode;
- links to the exact configuration files used.

### Acceptance gate

An examiner must be able to run the audit and at least one small experiment from a clean environment using the README instructions.

## Phase 13 — Documentation and report integration

### Required documentation changes

- replace RBC dataset background with the multibuilding AHU source;
- describe real operational data and expert annotation;
- explain building-specific schemas;
- explain missingness and duplicate handling;
- justify AHU-held-out and building-held-out evaluation;
- distinguish binary detection from diagnosis;
- discuss class imbalance and rare classes;
- report shortcut audits;
- explain why fault intensity is no longer claimed;
- update ethical, legal and licensing considerations;
- document reproducibility and data provenance.

### Claims that must not be made

- universal production accuracy;
- physical fault onset detection;
- reliable diagnosis for unsupported classes;
- generalisation to every building;
- fault-intensity prediction from this dataset.

## Phase 14 — Independent final audit

Before submission, perform a clean audit of:

- source data and checksums;
- preprocessing decisions;
- duplicate and missingness policy;
- train/test separation;
- feature lineage;
- hyperparameter selection;
- class support;
- result reproducibility;
- notebook execution;
- report claims against actual outputs;
- removal of absolute paths and personal data;
- repository size and licensing.

The audit should be performed from a fresh environment or clean checkout where possible.

## Final acceptance criteria

The new project is ready only when:

1. The raw data audit is reproducible.
2. Every cleaning decision is documented.
3. The canonical schema is tested.
4. Imputation and missingness handling are frozen before modelling.
5. AHU and building holdouts are leakage-safe.
6. Simple baselines are included.
7. Binary detection and multiclass diagnosis are reported separately.
8. Rare and absent classes are explicitly disclosed.
9. Cross-building transfer results are included.
10. The notebooks run from portable paths.
11. Results can be traced to configurations and manifests.
12. The report claims no more than the evidence supports.

## Working rule

At the end of every phase, we review the deliverable and resolve issues before starting the next phase. If a phase changes the research question, label policy or evaluation design, the plan is updated and the change is recorded before implementation continues.
