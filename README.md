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

## Prepare auditorium data under the Phase 2 policy

```bash
prepare_auditorium_ahu \
  --data-root /Users/mili/Downloads/27147678 \
  --output-dir data/processed/auditorium \
  --policy configs/auditorium.yaml
```

This creates a sorted, traceable derived table and a preparation manifest. It
does not impute values or train a model. The manifest records explicit sensor,
quality, and metadata column groups for later feature selection.

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

## Clean auditorium data under the Phase 4 policy

First prepare the Phase 2 table, then run:

```bash
clean_auditorium_ahu \
  --input-file data/processed/auditorium/auditorium_prepared.csv \
  --output-dir data/processed/auditorium/phase4 \
  --config configs/auditorium.yaml
```

This writes one cleaned master table and a second-audit manifest/summary. It
does not create model variants or fit an imputer on the full dataset;
imputation is training-fold-only.

## Engineer auditorium features under the Phase 5 policy

```bash
engineer_auditorium_features \
  --input-file data/processed/auditorium/phase4/auditorium_cleaned.csv \
  --output-dir data/processed/auditorium/phase5 \
  --config configs/auditorium.yaml
```

This writes the two model-ready physical and operational variant tables, a
feature-lineage table, and a shortcut-audit table. See
[docs/auditorium_phase5_policy.md](docs/auditorium_phase5_policy.md).

## Discover normal-operation patterns

Pattern discovery is a descriptive step for understanding normal operation. It
uses only rows labelled `Normal condition`; fault labels are excluded from the
profiles and are not predictors. It produces AHU/mode summaries, hourly and
monthly profiles, mode transitions, and PCA visualisations:

```bash
discover_auditorium_patterns \
  --input-file data/processed/auditorium/phase4/auditorium_cleaned.csv \
  --output-dir data/processed/auditorium/patterns \
  --config configs/auditorium.yaml
```

PCA median-fills values only for the visualisation matrix; this is not the
training imputation policy. See
[docs/auditorium_pattern_discovery.md](docs/auditorium_pattern_discovery.md).

## Demonstrate the complete auditorium workflow

After the audit, preparation, cleaning, feature-engineering and pattern
discovery commands have completed, open
`notebooks/01_auditorium_pipeline_walkthrough.ipynb`. It reads the saved
artefacts and creates report-ready Matplotlib figures for label support,
missingness, operating modes, feature families, shortcut flags, hourly and
monthly profiles, and normal-operation PCA. The notebook is a presentation
layer; the tested Python modules remain the source of truth.

## Create evaluation splits

Before any model is trained, create the frozen evaluation assignments:

```bash
create_evaluation_splits \
  --input-file data/interim/canonical/canonical_ahu.csv \
  --output-dir data/interim/evaluation_splits \
  --config configs/evaluation_splits.yaml
```

This writes one assignment table, manifest and readable summary. The table
contains four named designs: AHU-held-out, chronological, building-held-out
transfer, and nested grouped validation. Later models must use these saved
assignments rather than creating new random row splits.

## Evaluate the three headline detection baselines

The baseline comparison uses the frozen assignments and evaluates three
complementary references: a majority-class lower bound, a transparent normal
normal operating-range rule, and actuator/pump-only evidence. Missingness remains visible in
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
all four detection models, the physical-signal headline variant, one XGBoost
physical-versus-operational comparison, the three principal split designs,
false-positive rates, and LSTM scoring coverage. It does not retrain models; it
is a reproducible presentation layer over the saved results.

## Run detection experiments

After creating the canonical table and frozen split assignments, run the core
detection comparison:

```bash
run_detection_experiments \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/detection \
  --config configs/detection.yaml \
  --building-config configs/multibuilding_ahu.yaml
```

This evaluates the normal operating-range rule, Isolation Forest, XGBoost and LSTM
autoencoder. TensorFlow is included in the project requirements. If a platform
cannot install it, the run records the LSTM as unavailable rather than
substituting another model. See [docs/detection_experiments.md](docs/detection_experiments.md).

## Run diagnosis experiments

Diagnosis is evaluated separately from binary detection. The default Phase 9
run uses physical-signal features to classify the original fault-type labels
directly:

```bash
run_diagnosis_experiments \
  --data-file data/interim/canonical/canonical_ahu.csv \
  --splits-file data/interim/evaluation_splits/evaluation_splits.csv \
  --output-dir results/diagnosis \
  --config configs/diagnosis.yaml \
  --building-config configs/multibuilding_ahu.yaml
```

The outputs include class and building metrics, confusion matrices, unsupported
class counts, and row-level confidence/abstention fields. A fault class absent
from a training fold is reported as unsupported rather than being silently
treated as a normal classification error. See
[docs/diagnosis_experiments.md](docs/diagnosis_experiments.md).

## Repository status

Phase 0, the Phase 1 raw-data audit, the Phase 2 auditorium preparation policy,
the Phase 3 canonical ingestion adapter, the Phase 4 auditorium cleaning
policy, the Phase 5 auditorium feature policy, auditorium normal-operation
pattern discovery, and the evaluation split implementation are complete.
The three headline detection baselines and the first detection experiment
runner are now implemented; all model training uses the saved split
assignments. See
[docs/project_plan.md](docs/project_plan.md) for the staged implementation plan.
