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
METADATA_COLUMNS = {
    "building",
    "ahu",
    "source_row_id",
    "timestamp",
    "label_raw",
    "target_binary",
    "hours_since_previous_observation",
    "is_time_gap",
}


def load_policy(path: Path) -> dict[str, Any]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("building") != "auditorium":
        raise ValueError("The Phase 2 policy must describe the auditorium building")
    return policy


def _sensor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = METADATA_COLUMNS | {
        "AHU name",
        "Time",
        "labeling",
        "missing_sensor_count",
        "has_missing_sensor",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and not column.startswith("is_missing__")
    ]


def prepare_frame(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """Create the deterministic auditorium Phase 2 table from a raw frame."""
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required auditorium columns: {sorted(missing_columns)}")

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

    sensor_columns = [
        column
        for column in prepared.columns
        if column not in {"AHU name", "Time", "labeling", "timestamp", "building", "ahu", "source_row_id", "label_raw", "target_binary"}
    ]
    prepared = prepared.sort_values(["ahu", "timestamp"], kind="stable").reset_index(drop=True)
    deltas = prepared.groupby("ahu", sort=False)["timestamp"].diff().dt.total_seconds() / 3600
    prepared["hours_since_previous_observation"] = deltas
    prepared["is_time_gap"] = deltas.notna() & ~np.isclose(deltas, 1.0)
    for column in sensor_columns:
        prepared[f"is_missing__{column}"] = prepared[column].isna()
    prepared["missing_sensor_count"] = prepared[sensor_columns].isna().sum(axis=1)
    prepared["has_missing_sensor"] = prepared["missing_sensor_count"].gt(0)
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
            "feature_columns": _sensor_columns(prepared),
            "missingness_indicator_columns": [
                column for column in prepared.columns if column.startswith("is_missing__")
            ],
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
