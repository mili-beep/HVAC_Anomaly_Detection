# Canonical multibuilding AHU schema

Phase 3 uses one explicit union schema for the auditorium, hospital and office
files. Source-specific names are mapped in
`configs/multibuilding_ahu.yaml`; the raw files are never edited.

## Core columns

Every row contains `building`, `ahu`, `timestamp`, `label`, `source_file`, and
`source_row_id`. The source AHU name, source label and original timestamp text
are retained for row-level traceability. Source SHA-256 values are recorded in
the manifest. `ahu` is an identifier for grouping and splitting, not a
predictive feature.

The canonical sensor columns are:

```text
setpoint_temperature
return_temperature
supply_air_temperature
supply_fan
valve_position
heating_exchanger
heating_supply_temperature_1
heating_supply_temperature_2
heating_pump_total
heating_pump_1, heating_pump_2, heating_pump_3
cooling_supply_temperature_1, cooling_supply_temperature_2
cooling_pump_total
cooling_pump_1, cooling_pump_2, cooling_pump_3, cooling_pump_4
```

When a building does not provide a sensor, its canonical column is present but
missing for that building. This preserves a stable schema without inventing a
measurement.

## Raw-quality fields

For every canonical sensor, the adapter records `is_missing__...` and
`has_placeholder__...`. Aggregate counts (`missing_sensor_count` and
`placeholder_sensor_count`) and flags are also included. Duplicate
AHU/timestamp rows are retained and marked with `duplicate_ahu_timestamp`; the
Phase 2 policy decides later how they are handled for modelling.

The adapter does not impute, interpolate, relabel, deduplicate, or remove
rows. Literal placeholders remain visible in the canonical sensor column and
are separately flagged for Phase 4 cleaning.

## Reproducibility

`ingest_multibuilding_ahu` writes `canonical_ahu.csv` and
`canonical_ahu_manifest.json`. The manifest records source filenames,
checksums, row counts, mappings, timestamp parsing, labels, missingness,
placeholders and duplicate counts.
