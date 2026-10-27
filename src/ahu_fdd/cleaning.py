"""Phase 4 cleaning and training-fold imputation for prepared AHU data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ahu_fdd.audit import PLACEHOLDER_TOKENS, sha256_file


def _placeholder_mask(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.isin(PLACEHOLDER_TOKENS) & series.notna()


def load_cleaning_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sensor_feature_columns"), list):
        raise ValueError("The cleaning policy must define sensor_feature_columns")
    sensors = config["sensor_feature_columns"]
    if not sensors or len(sensors) != len(set(sensors)):
        raise ValueError("sensor_feature_columns must be a unique non-empty list")
    groups = config.get("sensor_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("The cleaning policy must define sensor_groups")
    grouped = [column for columns in groups.values() for column in columns]
    if set(grouped) != set(sensors) or len(grouped) != len(set(grouped)):
        raise ValueError("sensor_groups must partition sensor_feature_columns exactly")
    short_gap = config.get("short_gap_policy", {})
    if int(short_gap.get("maximum_rows", 0)) < 1:
        raise ValueError("short_gap_policy.maximum_rows must be positive")
    return config


def _sensor_group(config: dict[str, Any], sensor: str) -> str:
    for group, sensors in config["sensor_groups"].items():
        if sensor in sensors:
            return group
    raise ValueError(f"No sensor group configured for {sensor}")


def _subsystem_active(
    frame: pd.DataFrame, config: dict[str, Any], group: str
) -> pd.Series:
    sensors = config["sensor_groups"][group]
    if group == "common":
        return pd.Series(True, index=frame.index)
    return frame[sensors].notna().any(axis=1)


def _short_gap_mask(
    frame: pd.DataFrame,
    sensor: str,
    active: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """Find bounded missing runs without crossing AHUs, gaps, or modes."""
    maximum_rows = int(config["short_gap_policy"]["maximum_rows"])
    maximum_hours = float(config["short_gap_policy"]["maximum_hours"])
    missing = frame[sensor].isna() & active
    time_gap = frame.get("is_time_gap", pd.Series(False, index=frame.index)).fillna(False)
    eligible = pd.Series(False, index=frame.index)

    for _, group_frame in frame.groupby("ahu", sort=False):
        positions = list(group_frame.index)
        position = 0
        while position < len(positions):
            row = positions[position]
            if not bool(missing.loc[row]) or bool(time_gap.loc[row]):
                position += 1
                continue
            run = [row]
            next_position = position + 1
            while next_position < len(positions):
                previous = positions[next_position - 1]
                current = positions[next_position]
                delta = (frame.loc[current, "timestamp"] - frame.loc[previous, "timestamp"]).total_seconds() / 3600
                if (
                    bool(missing.loc[current])
                    and not bool(time_gap.loc[current])
                    and delta <= maximum_hours
                ):
                    run.append(current)
                    next_position += 1
                else:
                    break
            before_position = position - 1
            after_position = next_position
            bounded = before_position >= 0 and after_position < len(positions)
            if bounded and len(run) <= maximum_rows:
                before = positions[before_position]
                after = positions[after_position]
                before_delta = (frame.loc[run[0], "timestamp"] - frame.loc[before, "timestamp"]).total_seconds() / 3600
                after_delta = (frame.loc[after, "timestamp"] - frame.loc[run[-1], "timestamp"]).total_seconds() / 3600
                if (
                    not bool(frame.loc[before, sensor] is None or pd.isna(frame.loc[before, sensor]))
                    and not bool(frame.loc[after, sensor] is None or pd.isna(frame.loc[after, sensor]))
                    and bool(active.loc[before])
                    and bool(active.loc[after])
                    and before_delta <= maximum_hours
                    and after_delta <= maximum_hours
                ):
                    eligible.loc[run] = True
            position = next_position
    return eligible


def clean_frame(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clean placeholders and classify missingness without fitting an imputer."""
    sensors = list(config["sensor_feature_columns"])
    required = {"ahu", "timestamp", "source_row_id", "label_raw", "target_binary"} | set(sensors)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared frame is missing required columns: {sorted(missing)}")
    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    if cleaned["timestamp"].isna().any():
        raise ValueError("Phase 4 input contains unparseable timestamps")
    if cleaned.duplicated(["ahu", "timestamp"], keep=False).any():
        raise ValueError("Phase 4 auditorium input contains duplicate AHU/timestamp keys")
    cleaned = cleaned.sort_values(["ahu", "timestamp", "source_row_id"], kind="stable").reset_index(drop=True)

    placeholder_columns: list[str] = []
    invalid_numeric_columns: list[str] = []
    input_missing: dict[str, int] = {}
    placeholder_counts: dict[str, int] = {}
    initial_quality: dict[str, pd.Series] = {}
    for sensor in sensors:
        raw = cleaned[sensor].copy()
        placeholders = _placeholder_mask(raw)
        numeric = pd.to_numeric(raw.mask(placeholders), errors="coerce")
        invalid_numeric = numeric.isna() & raw.notna() & ~placeholders
        cleaned[sensor] = numeric
        initial_quality[f"placeholder_converted__{sensor}"] = placeholders
        initial_quality[f"invalid_numeric__{sensor}"] = invalid_numeric
        input_missing[sensor] = int(raw.isna().sum())
        placeholder_counts[sensor] = int(placeholders.sum())
        if placeholders.any():
            placeholder_columns.append(sensor)
        if invalid_numeric.any():
            invalid_numeric_columns.append(sensor)
    cleaned = pd.concat([cleaned, pd.DataFrame(initial_quality, index=cleaned.index)], axis=1)

    active_by_group = {
        group: _subsystem_active(cleaned, config, group)
        for group in config["sensor_groups"]
    }
    heating_active = active_by_group.get("heating", pd.Series(False, index=cleaned.index))
    cooling_active = active_by_group.get("cooling", pd.Series(False, index=cleaned.index))
    operating_mode_heating = heating_active & ~cooling_active
    operating_mode_cooling = cooling_active & ~heating_active
    operating_mode_mixed = heating_active & cooling_active
    operating_mode_unknown = ~heating_active & ~cooling_active
    operating_mode = np.select(
        [
            operating_mode_heating,
            operating_mode_cooling,
            operating_mode_mixed,
        ],
        ["heating", "cooling", "mixed"],
        default="unknown",
    )
    cleaned = pd.concat(
        [
            cleaned,
            pd.DataFrame(
                {
                    "heating_active": heating_active,
                    "cooling_active": cooling_active,
                    "operating_mode_heating": operating_mode_heating,
                    "operating_mode_cooling": operating_mode_cooling,
                    "operating_mode_mixed": operating_mode_mixed,
                    "operating_mode_unknown": operating_mode_unknown,
                    "operating_mode": operating_mode,
                },
                index=cleaned.index,
            ),
        ],
        axis=1,
    )

    audit_rows: list[dict[str, Any]] = []
    sensor_quality: dict[str, pd.Series] = {}
    for sensor in sensors:
        group = _sensor_group(config, sensor)
        active = active_by_group[group]
        missing_values = cleaned[sensor].isna()
        eligible = _short_gap_mask(cleaned, sensor, active, config)
        long_or_unresolved = missing_values & active & ~eligible
        sensor_quality[f"sensor_available__{sensor}"] = ~missing_values
        sensor_quality[f"structural_missing__{sensor}"] = missing_values & ~active
        sensor_quality[f"active_missing__{sensor}"] = missing_values & active
        sensor_quality[f"short_gap_eligible__{sensor}"] = eligible
        sensor_quality[f"long_or_unresolved_missing__{sensor}"] = long_or_unresolved
        sensor_quality[f"imputed__{sensor}"] = pd.Series(False, index=cleaned.index)
        audit_rows.append(
            {
                "sensor": sensor,
                "sensor_group": group,
                "input_missing_count": input_missing[sensor],
                "placeholder_count": placeholder_counts[sensor],
                "invalid_numeric_count": int(cleaned[f"invalid_numeric__{sensor}"].sum()),
                "structural_missing_count": int((missing_values & ~active).sum()),
                "active_missing_count": int((missing_values & active).sum()),
                "short_gap_eligible_count": int(eligible.sum()),
                "long_or_unresolved_missing_count": int(long_or_unresolved.sum()),
                "imputed_count": 0,
                "output_missing_count": int(missing_values.sum()),
            }
        )
    cleaned = pd.concat([cleaned, pd.DataFrame(sensor_quality, index=cleaned.index)], axis=1)

    missing_sensor_count = cleaned[
        [f"sensor_available__{sensor}" for sensor in sensors]
    ].eq(False).sum(axis=1)
    has_missing_sensor = missing_sensor_count.gt(0)
    structural_missing_sensor_count = cleaned[
        [f"structural_missing__{sensor}" for sensor in sensors]
    ].sum(axis=1)
    has_structural_missing = structural_missing_sensor_count.gt(0)
    short_gap_eligible_sensor_count = cleaned[
        [f"short_gap_eligible__{sensor}" for sensor in sensors]
    ].sum(axis=1)
    phase4_changed = cleaned[
        [f"placeholder_converted__{sensor}" for sensor in sensors]
        + [f"invalid_numeric__{sensor}" for sensor in sensors]
    ].any(axis=1)
    quality_flag = has_missing_sensor | has_structural_missing | short_gap_eligible_sensor_count.gt(0)
    if "is_time_gap" in cleaned:
        quality_flag = quality_flag | cleaned["is_time_gap"].astype(bool)
    replaced_aggregate_columns = [
        "missing_sensor_count",
        "has_missing_sensor",
        "structural_missing_sensor_count",
        "has_structural_missing",
        "short_gap_eligible_sensor_count",
        "imputed_sensor_count",
        "phase4_changed",
        "phase4_row_status",
        "phase4_quality_flag",
    ]
    cleaned = cleaned.drop(columns=replaced_aggregate_columns, errors="ignore")
    cleaned = pd.concat(
        [
            cleaned,
            pd.DataFrame(
                {
                    "missing_sensor_count": missing_sensor_count,
                    "has_missing_sensor": has_missing_sensor,
                    "structural_missing_sensor_count": structural_missing_sensor_count,
                    "has_structural_missing": has_structural_missing,
                    "short_gap_eligible_sensor_count": short_gap_eligible_sensor_count,
                    "imputed_sensor_count": 0,
                    "phase4_changed": phase4_changed,
                    "phase4_row_status": "retained",
                    "phase4_quality_flag": quality_flag,
                },
                index=cleaned.index,
            ),
        ],
        axis=1,
    )
    audit = pd.DataFrame(audit_rows)
    summary = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(cleaned)),
        "rows_changed": int(phase4_changed.sum()),
        "rows_excluded": 0,
        "rows_retained": int(len(cleaned)),
        "rows_retained_with_quality_flags": int(quality_flag.sum()),
        "rows_with_structural_missingness": int(has_structural_missing.sum()),
        "rows_with_short_gap_candidates": int(short_gap_eligible_sensor_count.gt(0).sum()),
        "rows_imputed": 0,
        "imputed_cells": 0,
        "placeholder_cells_converted": int(sum(placeholder_counts.values())),
        "invalid_numeric_cells_converted": int(
            sum(int(cleaned[f"invalid_numeric__{sensor}"].sum()) for sensor in sensors)
        ),
        "placeholder_columns": placeholder_columns,
        "invalid_numeric_columns": invalid_numeric_columns,
        "operating_mode_counts": cleaned["operating_mode"].value_counts().to_dict(),
    }
    return cleaned, audit, summary


@dataclass
class TrainingFoldMedianImputer:
    """Median imputer whose fit data must be supplied by the caller's train fold."""

    sensor_columns: list[str]
    medians: dict[str, float | None] | None = None

    def fit(self, training_frame: pd.DataFrame) -> "TrainingFoldMedianImputer":
        medians: dict[str, float] = {}
        for sensor in self.sensor_columns:
            values = pd.to_numeric(training_frame[sensor], errors="coerce")
            valid = values.dropna()
            medians[sensor] = None if valid.empty else float(valid.median())
        self.medians = medians
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.medians is None:
            raise RuntimeError("Fit the imputer on a training fold before transform")
        transformed = frame.copy()
        rows: list[dict[str, Any]] = []
        for sensor in self.sensor_columns:
            eligible = transformed[f"short_gap_eligible__{sensor}"].astype(bool)
            median = self.medians[sensor]
            to_impute = transformed[sensor].isna() & eligible & (median is not None)
            if median is not None:
                transformed.loc[to_impute, sensor] = median
            transformed[f"imputed__{sensor}"] = to_impute
            rows.append(
                {
                    "sensor": sensor,
                    "training_median": median,
                    "imputed_count": int(to_impute.sum()),
                }
            )
        transformed["imputed_sensor_count"] = transformed[
            [f"imputed__{sensor}" for sensor in self.sensor_columns]
        ].sum(axis=1)
        return transformed, pd.DataFrame(rows)


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


def clean_auditorium(
    input_file: Path, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    config = load_cleaning_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_file, low_memory=False)
    cleaned, audit, summary = clean_frame(frame, config)
    output_csv = output_dir / "auditorium_cleaned.csv"
    cleaned.to_csv(output_csv, index=False)
    audit_csv = output_dir / "phase4_audit.csv"
    audit.to_csv(audit_csv, index=False)
    manifest = _json_safe(
        {
            "schema_version": int(config["schema_version"]),
            "generated_at": datetime.now().astimezone().isoformat(),
            "building": config["building"],
            "input_file": str(input_file.resolve()),
            "input_sha256": sha256_file(input_file),
            "policy_file": str(config_path.resolve()),
            "policy": config,
            "summary": summary,
            "sensor_feature_columns": config["sensor_feature_columns"],
            "output_files": {
                "cleaned": str(output_csv),
                "audit_csv": str(audit_csv),
            },
            "imputation": {
                "performed_by_cli": False,
                "fit_scope": "training_fold_only",
                "method": "training_fold_median_for_bounded_active_short_gaps",
                "inactive_heating_cooling_values_imputed": False,
            },
        }
    )
    manifest_path = output_dir / "phase4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = output_dir / "phase4_audit.md"
    markdown.write_text(
        "# Auditorium Phase 4 second audit\n\n"
        f"- Input rows: {summary['input_rows']:,}\n"
        f"- Output rows: {summary['output_rows']:,}\n"
        f"- Rows changed: {summary['rows_changed']:,}\n"
        f"- Rows excluded: {summary['rows_excluded']:,}\n"
        f"- Rows retained with quality flags: {summary['rows_retained_with_quality_flags']:,}\n"
        f"- Rows imputed: {summary['rows_imputed']:,}\n"
        f"- Imputed cells: {summary['imputed_cells']:,}\n\n"
        "The CLI does not fit an imputer on the complete dataset. A median\n"
        "imputer must be fitted on each training fold and then applied only to\n"
        "eligible active-subsystem short gaps.\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean prepared auditorium data under the Phase 4 policy")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/auditorium/phase4"))
    parser.add_argument("--config", type=Path, default=Path("configs/auditorium.yaml"))
    args = parser.parse_args(argv)
    manifest = clean_auditorium(args.input_file, args.output_dir, args.config)
    print(f"Cleaned {manifest['summary']['output_rows']:,} auditorium rows into {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
