# Auditorium pattern discovery

Pattern discovery is the project's normal-operation analysis step. It answers
descriptive questions such as:

- which operating modes and AHUs are represented during normal operation;
- how sensor values vary by hour and month;
- how often normal rows transition between heating, cooling and other modes;
- whether normal operating states form visible clusters in a PCA projection.

It is not a fault detector and it does not estimate model performance. The
profiles use only rows with the configured `normal_label` (`Normal condition`).
Fault-labelled rows are counted in the manifest as excluded from profiles, but
they do not influence the profile statistics or PCA fit.

## Running it

Run Phase 4 first, then execute:

```bash
discover_auditorium_patterns \
  --input-file data/processed/auditorium/phase4/auditorium_cleaned.csv \
  --output-dir data/processed/auditorium/patterns \
  --config configs/auditorium.yaml
```

The same operation can be run without an editable package installation:

```bash
.venv/bin/python -m ahu_fdd.patterns \
  --input-file data/processed/auditorium/phase4/auditorium_cleaned.csv \
  --output-dir data/processed/auditorium/patterns \
  --config configs/auditorium.yaml
```

## Outputs

- `pattern_manifest.json`: input checksum, policy, counts and output paths;
- `pattern_summary.md`: short human-readable overview;
- `mode_summary.csv`: normal rows and AHU counts by operating mode;
- `ahu_summary.csv`: normal coverage by AHU and mode;
- `sensor_profile.csv`: overall normal sensor distributions by mode;
- `hourly_profile.csv`: normal sensor distributions by mode and hour;
- `monthly_profile.csv`: normal sensor distributions by mode and month;
- `transition_summary.csv`: observed normal-mode transitions within AHU streams;
- `pca_coordinates.csv`: normal-row PCA coordinates by mode and AHU;
- `pca_variance.csv`: explained variance for each mode-specific PCA;
- `figures/normal_hourly_temperature_profiles.png` and mode-specific PCA plots.

PCA is a visual summary, not a prediction model. For each mode, only sensors
with sufficient observed variation are used. Values missing in that mode are
filled with that mode's median solely to make the visualisation matrix
complete; this does not modify the cleaned data and must not be reused as a
training imputer. Large modes are sampled deterministically using the configured
random seed.

## Interpretation limits

Normal profiles describe the observed dataset rather than universal AHU rules.
They can reveal operating regimes, seasonal differences and unusual normal
clusters, but they cannot establish causation or prove that a cluster is a
fault. The later split and evaluation phase must define how these patterns are
used without leaking held-out information into a trained detector.
