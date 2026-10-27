"""Canonical ingestion adapter for the three-building AHU dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ahu_fdd.audit import PLACEHOLDER_TOKENS, sha256_file

COMMON_SOURCE_COLUMNS = {"AHU name", "labeling"}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("buildings"), dict):
        raise ValueError("The ingestion configuration must define buildings")
    canonical = config.get("canonical_sensor_columns")
    if not isinstance(canonical, list) or not canonical or len(canonical) != len(set(canonical)):
        raise ValueError("canonical_sensor_columns must be a unique non-empty list")
    for building, spec in config["buildings"].items():
        _validate_building_config(building, spec, canonical)
    return config


def _validate_building_config(
    building: str, spec: Any, canonical: list[str]
) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"Configuration for {building} must be a mapping")
    timestamp = spec.get("timestamp", {})
    timestamp_columns = timestamp.get("columns") if isinstance(timestamp, dict) else None
    if not isinstance(timestamp_columns, list) or not timestamp_columns:
        raise ValueError(f"{building} must define timestamp columns")
    mappings = spec.get("sensor_columns")
    if not isinstance(mappings, dict) or set(mappings) != set(canonical):
        raise ValueError(f"{building} sensor mapping must cover the canonical schema exactly")
    sources = [value for value in mappings.values() if value is not None]
    if len(sources) != len(set(sources)):
        raise ValueError(f"{building} sensor mapping contains duplicate source columns")


def _parse_timestamp(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    timestamp = spec["timestamp"]
    columns = timestamp["columns"]
    if not set(columns).issubset(frame.columns):
        missing = sorted(set(columns) - set(frame.columns))
        raise ValueError(f"Missing timestamp columns: {missing}")
    if len(columns) == 1:
        raw = frame[columns[0]].astype("string")
    else:
        raw = frame[columns].astype("string").fillna("").agg(" ".join, axis=1)
    return pd.to_datetime(raw, format=timestamp.get("format"), errors="coerce")


def _placeholder_mask(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.isin(PLACEHOLDER_TOKENS) & series.notna()


def _source_timestamp_text(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    columns = spec["timestamp"]["columns"]
    if len(columns) == 1:
        return frame[columns[0]].astype("string")
    return frame[columns].astype("string").fillna("").agg(" ".join, axis=1)


def canonicalize_frame(
    frame: pd.DataFrame,
    building: str,
    spec: dict[str, Any],
    canonical_sensor_columns: list[str],
    source_file: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map one raw frame without cleaning or imputing its sensor values."""
    missing_common = COMMON_SOURCE_COLUMNS - set(frame.columns)
    if missing_common:
        raise ValueError(f"{building} is missing required columns: {sorted(missing_common)}")
    mappings = spec["sensor_columns"]
    required_sources = {source for source in mappings.values() if source is not None}
    missing_sensors = required_sources - set(frame.columns)
    if missing_sensors:
        raise ValueError(f"{building} is missing configured sensors: {sorted(missing_sensors)}")

    timestamp = _parse_timestamp(frame, spec)
    if timestamp.isna().any():
        raise ValueError(f"{building} contains unparseable timestamps")

    source_row_id = pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    output = pd.DataFrame(index=frame.index)
    output["building"] = building
    output["ahu"] = frame["AHU name"].astype("string")
    output["timestamp"] = timestamp
    output["label"] = frame["labeling"].astype("string")
    output["source_file"] = source_file
    output["source_row_id"] = source_row_id
    output["source_ahu_name"] = frame["AHU name"].astype("string")
    output["source_label"] = frame["labeling"].astype("string")
    output["source_timestamp_text"] = _source_timestamp_text(frame, spec)

    for canonical in canonical_sensor_columns:
        source = mappings[canonical]
        values = frame[source] if source is not None else pd.Series(pd.NA, index=frame.index)
        output[canonical] = values
        output[f"is_missing__{canonical}"] = values.isna()
        output[f"has_placeholder__{canonical}"] = _placeholder_mask(values)

    output["missing_sensor_count"] = output[
        [f"is_missing__{column}" for column in canonical_sensor_columns]
    ].sum(axis=1)
    output["has_missing_sensor"] = output["missing_sensor_count"].gt(0)
    output["placeholder_sensor_count"] = output[
        [f"has_placeholder__{column}" for column in canonical_sensor_columns]
    ].sum(axis=1)
    output["has_placeholder"] = output["placeholder_sensor_count"].gt(0)

    duplicate_keys = output.duplicated(["ahu", "timestamp"], keep=False)
    output["duplicate_ahu_timestamp"] = duplicate_keys
    output = output.sort_values(["ahu", "timestamp", "source_row_id"], kind="stable")
    output = output.reset_index(drop=True)
    metadata = {
        "building": building,
        "row_count": int(len(output)),
        "source_columns": list(frame.columns),
        "source_column_map": mappings,
        "timestamp_columns": spec["timestamp"]["columns"],
        "timestamp_format": spec["timestamp"].get("format"),
        "timestamp_parse_success_count": int(len(output)),
        "timestamp_parse_failure_count": 0,
        "duplicate_ahu_timestamp_rows": int(duplicate_keys.sum()),
        "placeholder_counts": {
            column: int(output[f"has_placeholder__{column}"].sum())
            for column in canonical_sensor_columns
            if output[f"has_placeholder__{column}"].any()
        },
        "missing_counts": {
            column: int(output[f"is_missing__{column}"].sum())
            for column in canonical_sensor_columns
            if output[f"is_missing__{column}"].any()
        },
        "label_counts": output["label"].value_counts(dropna=False).to_dict(),
    }
    return output, metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool, str, int)) or value is None:
        return value
    return str(value)


def ingest_multibuilding(
    data_root: Path, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    data_root = data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = list(config["canonical_sensor_columns"])
    frames: list[pd.DataFrame] = []
    source_manifests: list[dict[str, Any]] = []

    for building, spec in config["buildings"].items():
        source_path = data_root / str(spec["source_file"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found for {building}: {source_path}")
        raw = pd.read_csv(source_path, low_memory=False)
        expected_rows = spec.get("expected_row_count")
        if expected_rows is not None and len(raw) != int(expected_rows):
            raise ValueError(
                f"{building} row count changed: expected {expected_rows}, found {len(raw)}"
            )
        prepared, metadata = canonicalize_frame(
            raw, building, spec, canonical, source_path.name
        )
        frames.append(prepared)
        source_manifests.append(
            {
                "source_file": source_path.name,
                "source_path": str(source_path),
                "sha256": sha256_file(source_path),
                "source_row_count": int(len(raw)),
                **metadata,
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["building", "ahu", "timestamp", "source_row_id"], kind="stable"
    ).reset_index(drop=True)
    output_csv = output_dir / "canonical_ahu.csv"
    combined.to_csv(output_csv, index=False)
    manifest = _json_safe(
        {
            "schema_version": int(config["schema_version"]),
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_root": str(data_root),
            "canonical_sensor_columns": canonical,
            "metadata_columns": [
                "building", "ahu", "timestamp", "label", "source_file",
                "source_row_id", "source_ahu_name", "source_label",
                "source_timestamp_text",
            ],
            "quality_columns": [
                *(f"is_missing__{column}" for column in canonical),
                *(f"has_placeholder__{column}" for column in canonical),
                "missing_sensor_count", "has_missing_sensor", "placeholder_sensor_count",
                "has_placeholder", "duplicate_ahu_timestamp",
            ],
            "source_files": source_manifests,
            "combined_row_count": int(len(combined)),
            "combined_ahu_count": int(combined["ahu"].nunique()),
            "combined_building_counts": combined["building"].value_counts().to_dict(),
            "output_file": str(output_csv),
            "config_file": str(config_path.expanduser().resolve()),
        }
    )
    manifest_path = output_dir / "canonical_ahu_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest the multibuilding AHU files into a canonical table"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/canonical"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multibuilding_ahu.yaml")
    )
    args = parser.parse_args(argv)
    manifest = ingest_multibuilding(args.data_root, args.output_dir, args.config)
    print(
        f"Ingested {manifest['combined_row_count']:,} rows from "
        f"{len(manifest['source_files'])} buildings into {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
