"""Phase 2 preparation policy for the auditorium AHU data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ahu_fdd.audit import sha256_file

REQUIRED_COLUMNS = {
    "AHU name",
    "Time",
    "labeling",
}
def load_policy(path: Path) -> dict[str, Any]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("building") != "auditorium":
        raise ValueError("The Phase 2 policy must describe the auditorium building")
    _validate_feature_policy(policy)
    return policy


def _validate_feature_policy(policy: dict[str, Any]) -> None:
    """Validate the explicit feature-role contract before any preparation."""
    roles = ("sensor_feature_columns", "metadata_columns", "quality_columns")
    for role in roles:
        columns = policy.get(role)
        if not isinstance(columns, list) or not columns or not all(
            isinstance(column, str) and column for column in columns
        ):
            raise ValueError(f"Policy field {role!r} must be a non-empty list of names")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Policy field {role!r} contains duplicate names")
    sensor = set(policy["sensor_feature_columns"])
    metadata = set(policy["metadata_columns"])
    quality = set(policy["quality_columns"])
    if sensor & (metadata | quality):
        raise ValueError("Sensor, metadata, and quality column roles must be disjoint")
    if "AHU name" not in metadata or "Time" not in metadata or "labeling" not in metadata:
        raise ValueError("Metadata columns must include the raw identity and label fields")
    expected_indicators = {f"is_missing__{column}" for column in sensor}
    if not expected_indicators <= quality:
        missing = sorted(expected_indicators - quality)
        raise ValueError(f"Quality columns omit missingness indicators: {missing}")


def _sensor_columns(policy: dict[str, Any]) -> list[str]:
    return list(policy["sensor_feature_columns"])


def _quality_columns(policy: dict[str, Any]) -> list[str]:
    return list(policy["quality_columns"])


def _metadata_columns(policy: dict[str, Any]) -> list[str]:
    return list(policy["metadata_columns"])


def _missingness_columns(policy: dict[str, Any]) -> list[str]:
    sensor_columns = _sensor_columns(policy)
    return [f"is_missing__{column}" for column in sensor_columns]


def _feature_variant_columns(policy: dict[str, Any]) -> dict[str, list[str]]:
    """Resolve the declared physical and operational-context feature variants."""
    return {
        "physical_signal": _sensor_columns(policy),
        "operational_context": _sensor_columns(policy) + _missingness_columns(policy),
    }


def prepare_frame(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """Create the deterministic auditorium Phase 2 table from a raw frame."""
    _validate_feature_policy(policy)
    sensor_columns = _sensor_columns(policy)
    missingness_columns = _missingness_columns(policy)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required auditorium columns: {sorted(missing_columns)}")
    missing_sensors = set(sensor_columns) - set(frame.columns)
    if missing_sensors:
        raise ValueError(f"Missing configured auditorium sensors: {sorted(missing_sensors)}")

    prepared = frame.copy()
    prepared.insert(0, "source_row_id", np.arange(len(prepared), dtype=np.int64))
    prepared["timestamp"] = pd.to_datetime(prepared["Time"], errors="coerce")
    if prepared["timestamp"].isna().any():
        raise ValueError("Auditorium data contains unparseable timestamps")

    duplicate_keys = prepared.duplicated(["AHU name", "timestamp"], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "Auditorium data contains duplicate AHU/timestamp keys; "
            "resolve them before preparation"
        )

    prepared["building"] = "auditorium"
    prepared["ahu"] = prepared["AHU name"].astype("string")
    prepared["label_raw"] = prepared["labeling"].astype("string")
    normal_label = str(policy["normal_label"])
    prepared["target_binary"] = np.where(
        prepared["label_raw"].eq(normal_label),
        str(policy["binary_normal_label"]),
        str(policy["binary_fault_label"]),
    )

    prepared = prepared.sort_values(["ahu", "timestamp"], kind="stable").reset_index(drop=True)
    deltas = prepared.groupby("ahu", sort=False)["timestamp"].diff().dt.total_seconds() / 3600
    prepared["hours_since_previous_observation"] = deltas
    prepared["is_time_gap"] = deltas.notna() & ~np.isclose(deltas, 1.0)
    for column in sensor_columns:
        prepared[f"is_missing__{column}"] = prepared[column].isna()
    prepared["missing_sensor_count"] = prepared[sensor_columns].isna().sum(axis=1)
    prepared["has_missing_sensor"] = prepared["missing_sensor_count"].gt(0)
    actual_quality = set(_quality_columns(policy))
    missing_quality = actual_quality - set(prepared.columns)
    if missing_quality:
        raise ValueError(f"Configured quality columns were not produced: {sorted(missing_quality)}")
    if set(missingness_columns) - actual_quality:
        raise ValueError("Configured quality columns must include all sensor missingness flags")
    declared_columns = set(_sensor_columns(policy)) | set(_metadata_columns(policy)) | actual_quality
    undeclared_columns = set(prepared.columns) - declared_columns
    if undeclared_columns:
        raise ValueError(
            "Prepared data contains columns with no declared feature role: "
            f"{sorted(undeclared_columns)}"
        )
    missing_declared_columns = declared_columns - set(prepared.columns)
    if missing_declared_columns:
        raise ValueError(
            "Feature-role policy declares columns that are absent from prepared data: "
            f"{sorted(missing_declared_columns)}"
        )
    return prepared


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool, str, int)) or value is None:
        return value
    return str(value)


def prepare_auditorium(data_root: Path, output_dir: Path, policy_path: Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    data_root = data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = data_root / str(policy["source_file"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Auditorium source file not found: {source_path}")

    raw = pd.read_csv(source_path, low_memory=False)
    prepared = prepare_frame(raw, policy)
    output_csv = output_dir / "auditorium_prepared.csv"
    prepared.to_csv(output_csv, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "building": "auditorium",
            "policy": policy,
            "source_file": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_row_count": len(raw),
            "prepared_row_count": len(prepared),
            "ahu_count": prepared["ahu"].nunique(),
            "timestamp_minimum": prepared["timestamp"].min(),
            "timestamp_maximum": prepared["timestamp"].max(),
            "target_binary_counts": prepared["target_binary"].value_counts().to_dict(),
            "source_label_counts": prepared["label_raw"].value_counts().to_dict(),
            "time_gap_rows": int(prepared["is_time_gap"].sum()),
            "missing_sensor_rows": int(prepared["has_missing_sensor"].sum()),
            "sensor_feature_columns": _sensor_columns(policy),
            "quality_columns": _quality_columns(policy),
            "metadata_columns": _metadata_columns(policy),
            "feature_variants": _feature_variant_columns(policy),
            "output_file": str(output_csv),
            "policy_file": str(policy_path.resolve()),
        }
    )
    (output_dir / "auditorium_preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare auditorium AHU data under the frozen Phase 2 policy")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/auditorium"))
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/auditorium_phase2.yaml"),
    )
    args = parser.parse_args(argv)
    manifest = prepare_auditorium(args.data_root, args.output_dir, args.policy)
    print(f"Prepared {manifest['prepared_row_count']:,} auditorium rows in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
