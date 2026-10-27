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
  --policy configs/auditorium_phase2.yaml
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
  --config configs/auditorium_phase4.yaml
```

This writes the cleaned table, physical-signal and operational-context feature
tables, and a second-audit manifest/summary. It does not fit an imputer on the
full dataset; imputation is training-fold-only.

## Repository status

Phase 0, the Phase 1 raw-data audit, the Phase 2 auditorium preparation policy,
the Phase 3 canonical ingestion adapter and the Phase 4 auditorium cleaning
policy are complete. The next gate is
cleaning/imputation policy for all buildings; no modelling should begin until
those choices are frozen. See
[docs/project_plan.md](docs/project_plan.md) for the staged implementation plan.
