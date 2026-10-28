# Multibuilding AHU Fault Detection and Diagnosis

This is a new, standalone research repository for analysing the real
multibuilding AHU dataset from the Scientific Data publication associated with
Figshare record 27147678. It is separate from the completed RBC-ASHRAE1312
project.

The project will investigate whether machine-learning methods can detect
abnormal AHU operation, diagnose fault categories, and generalise across AHUs
and buildings. The initial building roles are provisional until the raw-data
audit is complete:

- auditorium: primary multiclass diagnosis study;
- office: primary unseen-building/generalisation study;
- hospital: data-quality and transfer case study.

## Development approach

The Python package will be the source of truth for ingestion, data-quality
decisions, feature engineering, splitting, modelling, evaluation, and saved
artefacts. Notebooks will be added later as a readable reporting layer; they
will load saved outputs rather than duplicate the implementation.

No model training is part of the initial repository setup. The first gate is a
reproducible audit of the raw files and a frozen data-quality policy.

## Setup

```bash
cd /Users/mili/Desktop/IRPN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The source CSV files are not committed to Git. Download or place them locally
as described in [data/raw/README.md](data/raw/README.md).

## Run the Phase 1 audit

After installing the package, run the audit against the local source directory:

```bash
audit_multibuilding_ahu \
  --data-root /Users/mili/Downloads/27147678 \
  --output-dir results/audit
```

The command writes `audit_manifest.json`, `audit_summary.md`,
`audit_columns.csv`, `audit_missingness.csv`, and `audit_label_counts.csv`.

## Build the Phase 3 canonical table

```bash
ingest_multibuilding_ahu \
  --data-root /Users/mili/Downloads/27147678 \
  --output-dir data/interim/canonical \
  --config configs/multibuilding_ahu.yaml
```

This applies the same explicit source-to-canonical mapping to auditorium,
hospital and office data. It preserves raw sensor values, labels and source
row identifiers, and records missingness, placeholders and duplicate keys. It
does not impute, deduplicate, relabel or remove rows. See
[docs/canonical_schema.md](docs/canonical_schema.md).

## Build shared multibuilding features under the Step 5 policy

After the quality-controlled cleaning command has completed, generate one
feature table for all three buildings:

```bash
engineer_multibuilding_features \
  --input-file data/interim/cleaned/cleaned_ahu.csv \
  --output-dir data/interim/features \
  --building-config configs/multibuilding_ahu.yaml \
  --feature-config configs/feature_engineering.yaml
```

The manifest declares the default physical-signal predictors and the
secondary operational-context comparison columns. It also records feature
lineage and a shortcut audit. No imputation is performed here; later model
folds fit imputers on training rows only. See
[docs/multibuilding_feature_engineering.md](docs/multibuilding_feature_engineering.md).

## Create building-specific splits

```bash
create_building_splits \
  --input-file data/interim/cleaned/cleaned_ahu.csv \
  --output-dir data/interim/building_splits \
  --config configs/evaluation_splits.yaml
```

This creates independent AHU-held-out and chronological assignments for each
building without changing the frozen pooled assignments.

## Run quality-controlled pooled detection

Create matched pooled assignments for each duplicate policy, keeping duplicate
timestamp keys together in a chronological partition:

```bash
create_quality_splits \
  --input-file data/interim/cleaned/cleaned_ahu.csv \
  --output-dir data/interim/quality_splits/exclude_ambiguous_keys \
  --config configs/evaluation_splits.yaml
```

Run the four detection methods on the main cleaned table:

```bash
run_quality_detection \
  --data-file data/interim/cleaned/cleaned_ahu.csv \
  --feature-file data/interim/features/multibuilding_features.csv \
  --feature-manifest data/interim/features/feature_manifest.json \
  --splits-file data/interim/quality_splits/exclude_ambiguous_keys/evaluation_splits.csv \
  --output-dir results/quality_controlled/exclude_ambiguous_keys \
  --duplicate-policy exclude_ambiguous_keys
```

Repeat the same commands with the retained cleaned table, retained feature
directory and `retain_flagged_keys` output directory for the duplicate-policy
sensitivity comparison. See
[docs/quality_detection.md](docs/quality_detection.md) for the complete
command set and interpretation of coverage.

## Run independent building models

After the building-specific split assignments exist, train the four detection
methods independently within each building:

```bash
run_building_detection \
  --data-file data/interim/cleaned/cleaned_ahu.csv \
  --feature-file data/interim/features/multibuilding_features.csv \
  --feature-manifest data/interim/features/feature_manifest.json \
  --splits-file data/interim/building_splits/building_splits.csv \
  --output-dir results/building_specific \
  --config configs/detection.yaml
```

The model with the best validation F1 is selected separately for each
building/design. It is then refitted on that building's training and validation
rows and evaluated once on its test rows. See
[docs/building_specific_detection.md](docs/building_specific_detection.md).

## Create evaluation splits

Before any model is trained, create the frozen evaluation assignments:

```bash
create_evaluation_splits \
  --input-file data/interim/canonical/canonical_ahu.csv \
  --output-dir data/interim/evaluation_splits \
  --config configs/evaluation_splits.yaml
```

This writes one assignment table, manifest and readable summary. The table
contains four named assignments: AHU-held-out, chronological, building-held-out
transfer, and nested grouped validation. The AHU-held-out assignment is retained
for traceability but is currently disabled for pooled headline modelling. Later
pooled models use the chronological, building-held-out and nested grouped
assignments rather than creating new random row splits.

## Evaluate the three headline detection baselines

The baseline comparison uses the frozen assignments and evaluates three
complementary references: a majority-class lower bound, a transparent normal
operating-range rule, and actuator/pump-only evidence. Missingness remains visible in
the audit and operational-context sensitivity results rather than being used as
a headline detector.

```bash
evaluate_detection_baselines \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/baselines \
  --config configs/baselines.yaml
```

The output contains fold metrics, a summary table, a manifest and a readable
Markdown comparison. Imputation is fitted inside each training fold; no test
labels or test values are used to choose thresholds.

## Showcase baseline and model results

After the baseline and detection commands have completed, open
`notebooks/02_detection_results.ipynb`. It loads the saved metrics and creates
report-ready tables and Matplotlib figures for the three headline baselines,
all four detection models, the quality-controlled duplicate-policy comparison,
the physical-signal headline variant, false-positive rates, and LSTM scoring
coverage. Building-specific results are presented separately in
`notebooks/03_building_detection_results.ipynb`. Neither notebook retrains
models; they are reproducible presentation layers over the saved results.

## Run diagnosis experiments

Diagnosis is evaluated separately from binary detection. The default Phase 9
run uses physical-signal features to classify the original fault-type labels
directly:

```bash
run_diagnosis_experiments \
  --data-file data/interim/cleaned/cleaned_ahu.csv \
  --feature-file data/interim/features/multibuilding_features.csv \
  --feature-manifest data/interim/features/feature_manifest.json \
  --splits-file data/interim/quality_splits/exclude_ambiguous_keys/evaluation_splits.csv \
  --output-dir results/diagnosis \
  --config configs/diagnosis.yaml \
  --building-config configs/multibuilding_ahu.yaml \
  --duplicate-policy exclude_ambiguous_keys
```

The diagnosis run uses the same cleaned rows, saved Phase 5 feature table and
duplicate-excluded split policy as the headline detection run. It does not
rebuild a separate in-memory feature set.

The outputs include class and building metrics, confusion matrices, unsupported
class counts, and row-level model-score/abstention fields. A fault class absent
from a training fold is reported as unsupported rather than being silently
treated as a normal classification error. Labels with fewer than the configured
minimum support are excluded from diagnosis only and recorded in the manifest.
See
[docs/diagnosis_experiments.md](docs/diagnosis_experiments.md).

Notebook 4 also reports one symptom-only diagnosis sensitivity run. It uses the
same rows and folds but excludes direct pump, fan, valve and exchanger channels.
The practical BMS-signal run remains the headline; the sensitivity run tests
whether exact diagnosis depends on direct equipment-state information.

## Run secondary analyses

These commands generate supporting evidence for the notebooks and report. They
use the same frozen data, feature manifests and split assignments; they do not
replace the headline pooled or building-specific runs:

- `run_feature_sensitivity` compares the declared XGBoost feature variants.
- `report_feature_dependence` audits each feature's held-out univariate AUC.
- `compare_pooled_building_models` compares individual and pooled models on
  matched building test rows.
- `run_directional_transfer` reports the three pre-declared zero-shot
  source-to-target building transfers.
- `report_diagnosis_support` reports train/test support and unsupported classes
  for the diagnosis folds.

Each command writes its own CSV/JSON evidence directory. See the corresponding
module docstrings and `docs/` pages for required input paths.
The commands intentionally remain separate so related analyses can be rerun
without retraining the main models.

## Showcase operational evidence

After the quality-controlled detection and diagnosis runs, create the
operational events from the detector's saved row-level predictions:

```bash
score_ahu_priorities \
  --predictions results/quality_controlled/exclude_ambiguous_keys/quality_detection_predictions.csv \
  --diagnosis-predictions results/diagnosis/diagnosis_predictions.csv \
  --model xgboost \
  --design chronological \
  --policy configs/operational_priority.yaml \
  --output-dir results/operational
```

The command ignores normal detector rows, groups consecutive detected-alert
rows into events, and attaches a diagnosis only when a matching row exists.
Open `notebooks/05_operational_outputs.ipynb` afterwards for concise evidence
cards for facilities staff. It is an historical evidence presentation, not a
live dashboard.

The five notebook roles are:

1. `01_multibuilding_pipeline_walkthrough.ipynb` — audit, cleaning, features and normal patterns across all buildings;
2. `02_detection_results.ipynb` — pooled baselines, models and duplicate-policy sensitivity;
3. `03_building_detection_results.ipynb` — independent auditorium, hospital and office models;
4. `04_diagnosis_results.ipynb` — exact fault-type diagnosis;
5. `05_operational_outputs.ipynb` — staff-facing evidence cards and deployment limitations.

## Showcase diagnosis results

After the diagnosis run has completed, open
`notebooks/04_diagnosis_results.ipynb`. It reads the saved Phase 9 outputs and
creates report-ready tables and Matplotlib figures for split-level performance,
fault-type metrics, unsupported classes, model-score coverage, and a grouped-AHU
confusion matrix. It does not retrain the model.

## Repository status

Phase 0, the Phase 1 raw-data audit, the Phase 2 data-quality policy, the Phase
3 canonical ingestion adapter, the Phase 4 quality-controlled cleaning audit,
the shared Phase 5 feature generator, multibuilding normal-operation pattern
discovery, and the evaluation split implementation are complete.
The three headline detection baselines and the first detection experiment
runner are now implemented; all model training uses the saved split
assignments. See
[docs/project_plan.md](docs/project_plan.md) for the staged implementation plan.
