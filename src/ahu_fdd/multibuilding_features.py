"""Shared, label-independent feature generation for the AHU buildings."""

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
from sklearn.metrics import roc_auc_score

from ahu_fdd.audit import sha256_file
from ahu_fdd.multibuilding import load_config

KEY_COLUMNS = ["building", "ahu", "timestamp", "source_row_id", "label"]


def load_feature_settings(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The feature settings must be a mapping")
    windows = config.get("rolling_windows_rows")
    if not isinstance(windows, list) or not windows or any(int(w) < 1 for w in windows):
        raise ValueError("rolling_windows_rows must contain positive window sizes")
    if float(config.get("temporal_contiguity_max_gap_hours", 0)) <= 0:
        raise ValueError("temporal_contiguity_max_gap_hours must be positive")
    if float(config.get("rate_max_gap_hours", 0)) <= 0:
        raise ValueError("rate_max_gap_hours must be positive")
    context = config.get("operational_context", {})
    if not isinstance(context, dict):
        raise ValueError("operational_context must be a mapping")
    return config


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


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _add(
    values: dict[str, pd.Series],
    lineage: list[dict[str, Any]],
    variants: dict[str, list[str]],
    name: str,
    series: pd.Series,
    family: str,
    source_columns: list[str],
    formula: str,
    *,
    operational: bool = False,
    past_only: bool = False,
    quality_or_context: bool = False,
) -> None:
    if name in values:
        raise ValueError(f"Feature generated twice: {name}")
    values[name] = pd.to_numeric(series, errors="coerce").astype(float)
    variants["physical_signal" if not operational else "operational_context"].append(name)
    lineage.append(
        {
            "feature": name,
            "family": family,
            "source_columns": source_columns,
            "formula": formula,
            "past_only": past_only,
            "quality_or_context": quality_or_context,
        }
    )


def _past_rolling(
    frame: pd.DataFrame, column: str, window: int, max_gap_hours: float
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, indices in frame.groupby(["building", "ahu"], sort=False).groups.items():
        group = frame.loc[indices]
        values = _numeric(group, column)
        hours = pd.to_datetime(group["timestamp"]).diff().dt.total_seconds().div(3600)
        duplicate = group.get(
            "duplicate_ahu_timestamp", pd.Series(False, index=group.index)
        ).astype(bool)
        boundary = hours.gt(max_gap_hours) | hours.le(0) | duplicate | duplicate.shift(
            1, fill_value=False
        )
        breaks = np.flatnonzero(boundary.to_numpy()[1:]) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(group)]))
        output = np.full(len(group), np.nan, dtype=float)
        for start, end in zip(starts, ends):
            output[start:end] = (
                values.iloc[start:end].shift(1).rolling(window, min_periods=1).mean().to_numpy()
            )
        result.loc[indices] = output
    return result


def _past_rate(frame: pd.DataFrame, column: str, max_gap_hours: float) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, indices in frame.groupby(["building", "ahu"], sort=False).groups.items():
        group = frame.loc[indices]
        values = _numeric(group, column)
        timestamps = pd.to_datetime(group["timestamp"])
        duplicate = group.get(
            "duplicate_ahu_timestamp", pd.Series(False, index=group.index)
        ).astype(bool)
        previous = values.shift(1)
        hours = timestamps.diff().dt.total_seconds().div(3600)
        valid = (
            hours.gt(0)
            & hours.le(max_gap_hours)
            & values.notna()
            & previous.notna()
            & ~duplicate
            & ~duplicate.shift(1, fill_value=False)
        )
        result.loc[indices] = ((values - previous).div(hours).where(valid)).to_numpy()
    return result


def engineer_multibuilding_features(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    settings: dict[str, Any],
    *,
    allow_duplicate_keys: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Generate the same physical/context features for every building.

    Labels are carried as evaluation keys only. No label is read while creating
    a predictor; the separate shortcut audit is the only label-aware step.
    """
    required = set(KEY_COLUMNS) | set(sensor_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature input is missing columns: {sorted(missing)}")
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise ValueError("Feature input contains unparseable timestamps")
    key_columns = ["building", "ahu", "timestamp"]
    if ordered.duplicated(key_columns, keep=False).any() and not allow_duplicate_keys:
        raise ValueError("Feature input contains duplicate building/AHU/timestamp keys")
    ordered = ordered.sort_values(key_columns + ["source_row_id"], kind="stable").reset_index(drop=True)

    values: dict[str, pd.Series] = {}
    lineage: list[dict[str, Any]] = []
    variants = {"physical_signal": [], "operational_context": []}
    numeric = {column: _numeric(ordered, column) for column in sensor_columns}

    temporal_gap = float(settings["temporal_contiguity_max_gap_hours"])
    rate_gap = min(float(settings["rate_max_gap_hours"]), temporal_gap)
    for column in sensor_columns:
        _add(values, lineage, variants, f"sensor__{column}", numeric[column], "raw_sensor", [column], column)

    def sensor(column: str) -> pd.Series:
        return numeric[column]

    relationships = [
        (
            "relationship__supply_setpoint_residual",
            sensor("supply_air_temperature") - sensor("setpoint_temperature"),
            ["supply_air_temperature", "setpoint_temperature"],
            "supply_air_temperature - setpoint_temperature",
        ),
        (
            "relationship__return_setpoint_residual",
            sensor("return_temperature") - sensor("setpoint_temperature"),
            ["return_temperature", "setpoint_temperature"],
            "return_temperature - setpoint_temperature",
        ),
        (
            "relationship__supply_return_delta",
            sensor("supply_air_temperature") - sensor("return_temperature"),
            ["supply_air_temperature", "return_temperature"],
            "supply_air_temperature - return_temperature",
        ),
        (
            "relationship__supply_return_abs_delta",
            (sensor("supply_air_temperature") - sensor("return_temperature")).abs(),
            ["supply_air_temperature", "return_temperature"],
            "abs(supply_air_temperature - return_temperature)",
        ),
        (
            "relationship__heating_supply_air_delta",
            sensor("heating_supply_temperature_1") - sensor("supply_air_temperature"),
            ["heating_supply_temperature_1", "supply_air_temperature"],
            "heating_supply_temperature_1 - supply_air_temperature",
        ),
        (
            "relationship__cooling_supply_air_delta",
            sensor("supply_air_temperature") - sensor("cooling_supply_temperature_1"),
            ["supply_air_temperature", "cooling_supply_temperature_1"],
            "supply_air_temperature - cooling_supply_temperature_1",
        ),
        (
            "relationship__cooling_sensor_delta",
            sensor("cooling_supply_temperature_1") - sensor("cooling_supply_temperature_2"),
            ["cooling_supply_temperature_1", "cooling_supply_temperature_2"],
            "cooling_supply_temperature_1 - cooling_supply_temperature_2",
        ),
        (
            "relationship__fan_valve_interaction",
            sensor("supply_fan") * sensor("valve_position"),
            ["supply_fan", "valve_position"],
            "supply_fan * valve_position",
        ),
    ]
    for name, series, source_columns, formula in relationships:
        _add(values, lineage, variants, name, series, "relationship", source_columns, formula)

    heating_components = [sensor(f"heating_pump_{i}") for i in (1, 2, 3)]
    cooling_components = [sensor(f"cooling_pump_{i}") for i in (1, 2, 3, 4)]
    _add(
        values,
        lineage,
        variants,
        "relationship__heating_pump_total_residual",
        sensor("heating_pump_total") - sum(heating_components),
        "pump_relationship",
        ["heating_pump_total", "heating_pump_1", "heating_pump_2", "heating_pump_3"],
        "heating_pump_total - sum(heating_pump_1:3)",
    )
    _add(
        values,
        lineage,
        variants,
        "relationship__cooling_pump_total_residual",
        sensor("cooling_pump_total") - sum(cooling_components),
        "pump_relationship",
        ["cooling_pump_total", "cooling_pump_1", "cooling_pump_2", "cooling_pump_3", "cooling_pump_4"],
        "cooling_pump_total - sum(cooling_pump_1:4)",
    )
    _add(
        values,
        lineage,
        variants,
        "relationship__heating_pump_any_on",
        pd.concat(heating_components, axis=1).gt(0).any(axis=1),
        "pump_relationship",
        ["heating_pump_1", "heating_pump_2", "heating_pump_3"],
        "any(heating_pump_1:3 > 0)",
    )
    _add(
        values,
        lineage,
        variants,
        "relationship__cooling_pump_any_on",
        pd.concat(cooling_components, axis=1).gt(0).any(axis=1),
        "pump_relationship",
        ["cooling_pump_1", "cooling_pump_2", "cooling_pump_3", "cooling_pump_4"],
        "any(cooling_pump_1:4 > 0)",
    )

    for column in sensor_columns:
        for window in settings["rolling_windows_rows"]:
            window = int(window)
            _add(
                values,
                lineage,
                variants,
                f"temporal__past_mean_{window}__{column}",
                _past_rolling(ordered, column, window, temporal_gap),
                "past_rolling_mean",
                [column],
                f"mean(previous {window} rows of {column})",
                past_only=True,
            )
        _add(
            values,
            lineage,
            variants,
            f"temporal__past_rate__{column}",
            _past_rate(ordered, column, rate_gap),
            "past_rate_of_change",
            [column, "timestamp"],
            f"delta({column}) / elapsed_hours; max_gap={rate_gap}",
            past_only=True,
        )

    context = settings.get("operational_context", {})
    if context.get("include_missingness_indicators", True):
        for column in sensor_columns:
            _add(
                values,
                lineage,
                variants,
                f"quality__missing__{column}",
                numeric[column].isna(),
                "missingness",
                [column],
                f"is_missing({column})",
                operational=True,
                quality_or_context=True,
            )
    if context.get("include_sensor_availability", True):
        for column in sensor_columns:
            _add(
                values,
                lineage,
                variants,
                f"quality__available__{column}",
                numeric[column].notna(),
                "sensor_availability",
                [column],
                f"observed({column})",
                operational=True,
                quality_or_context=True,
            )
    if context.get("include_operating_mode", True):
        heating_signals = pd.concat(
            [sensor("heating_exchanger"), sensor("heating_pump_total"), *heating_components], axis=1
        )
        cooling_signals = pd.concat(
            [sensor("cooling_pump_total"), *cooling_components], axis=1
        )
        heating_observed = heating_signals.notna().any(axis=1)
        cooling_observed = cooling_signals.notna().any(axis=1)
        heating_active = heating_signals.gt(0).any(axis=1)
        cooling_active = cooling_signals.gt(0).any(axis=1)
        mode_unknown = ~heating_observed & ~cooling_observed
        mode_features = [
            ("mode__heating_active", heating_active, "any(heating actuator/pump > 0)"),
            ("mode__cooling_active", cooling_active, "any(cooling pump > 0)"),
            ("mode__mixed_active", heating_active & cooling_active, "heating_active and cooling_active"),
            ("mode__idle", ~heating_active & ~cooling_active & ~mode_unknown, "not active and mode signals observed"),
            ("mode__unknown", mode_unknown, "no heating or cooling mode signal observed"),
        ]
        for name, series, formula in mode_features:
            _add(
                values,
                lineage,
                variants,
                name,
                series,
                "operating_mode",
                ["heating_exchanger", "heating_pump_total", "cooling_pump_total"],
                formula,
                operational=True,
                quality_or_context=True,
            )
    if context.get("include_time_context", True):
        timestamp = ordered["timestamp"]
        hour = timestamp.dt.hour + timestamp.dt.minute / 60
        weekday = timestamp.dt.day_of_week
        for name, series, formula in [
            ("context__hour_sin", np.sin(2 * np.pi * hour / 24), "sin(2*pi*hour/24)"),
            ("context__hour_cos", np.cos(2 * np.pi * hour / 24), "cos(2*pi*hour/24)"),
            ("context__weekday_sin", np.sin(2 * np.pi * weekday / 7), "sin(2*pi*weekday/7)"),
            ("context__weekday_cos", np.cos(2 * np.pi * weekday / 7), "cos(2*pi*weekday/7)"),
        ]:
            _add(
                values,
                lineage,
                variants,
                name,
                series,
                "time_context",
                ["timestamp"],
                formula,
                operational=True,
                quality_or_context=True,
            )

    physical = variants["physical_signal"]
    variants["operational_context"] = physical + variants["operational_context"]
    feature_frame = pd.concat(
        [ordered[KEY_COLUMNS].reset_index(drop=True), pd.DataFrame(values)], axis=1
    )
    if len(set(variants["operational_context"])) != len(variants["operational_context"]):
        raise ValueError("Operational feature variant contains duplicate names")
    forbidden = {"building", "ahu", "timestamp", "source_row_id", "label", "target_binary"}
    if forbidden & set(variants["physical_signal"] + variants["operational_context"]):
        raise ValueError("Metadata or labels leaked into predictor columns")
    return feature_frame, pd.DataFrame(lineage), variants


def shortcut_audit(
    feature_frame: pd.DataFrame,
    lineage: pd.DataFrame,
    variants: dict[str, list[str]],
    normal_label: str,
) -> pd.DataFrame:
    """Report label associations without changing the declared features."""
    y = ~feature_frame["label"].astype("string").str.strip().eq(normal_label)
    lineage_map = lineage.set_index("feature").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for variant, features in variants.items():
        for feature in features:
            values = pd.to_numeric(feature_frame[feature], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            missing_rates = feature_frame.assign(_missing=~valid).groupby(y)["_missing"].mean()
            normal_missing = float(missing_rates.get(False, np.nan))
            fault_missing = float(missing_rates.get(True, np.nan))
            value_auc = np.nan
            if valid.sum() and y[valid].nunique() == 2 and values[valid].nunique() > 1:
                value_auc = float(roc_auc_score(y[valid], values[valid]))
            missing_auc = np.nan
            if y.nunique() == 2 and (~valid).nunique() > 1:
                missing_auc = float(roc_auc_score(y, (~valid).astype(int)))
            rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "family": lineage_map[feature]["family"],
                    "normal_missing_rate": normal_missing,
                    "fault_missing_rate": fault_missing,
                    "missing_rate_gap": abs(fault_missing - normal_missing),
                    "value_auc": value_auc,
                    "missingness_auc": missing_auc,
                }
            )
    return pd.DataFrame(rows)


def engineer_multibuilding(
    input_file: Path,
    output_dir: Path,
    building_config_path: Path,
    feature_config_path: Path,
    allow_duplicate_keys: bool = False,
) -> dict[str, Any]:
    building_config = load_config(building_config_path)
    settings = load_feature_settings(feature_config_path)
    sensors = list(building_config["canonical_sensor_columns"])
    frame = pd.read_csv(input_file, low_memory=False)
    features, lineage, variants = engineer_multibuilding_features(
        frame, sensors, settings, allow_duplicate_keys=allow_duplicate_keys
    )
    audit = shortcut_audit(features, lineage, variants, str(settings["normal_label"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "multibuilding_features.csv"
    features.to_csv(feature_path, index=False)
    lineage_path = output_dir / "feature_lineage.csv"
    lineage.to_csv(lineage_path, index=False)
    audit_path = output_dir / "shortcut_audit.csv"
    audit.to_csv(audit_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": int(settings["schema_version"]),
            "generated_at": datetime.now().astimezone().isoformat(),
            "input_file": str(input_file.expanduser().resolve()),
            "input_sha256": sha256_file(input_file),
            "building_config_file": str(building_config_path.expanduser().resolve()),
            "feature_config_file": str(feature_config_path.expanduser().resolve()),
            "row_count": int(len(features)),
            "building_row_counts": features["building"].value_counts().to_dict(),
            "feature_variants": variants,
            "feature_counts": {name: len(columns) for name, columns in variants.items()},
            "lineage_file": str(lineage_path),
            "shortcut_audit_file": str(audit_path),
            "output_file": str(feature_path),
            "label_used_only_for_audit": True,
            "allow_duplicate_keys": allow_duplicate_keys,
            "past_only_features_verified": bool(lineage.loc[lineage["past_only"], "past_only"].all()),
        }
    )
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Multibuilding feature-generation summary",
        "",
        "The physical-signal variant is the default modelling feature set. The operational-context variant is a secondary comparison only.",
        "",
        f"- Rows: {len(features):,}",
        f"- Physical-signal predictors: {len(variants['physical_signal']):,}",
        f"- Operational-context predictors: {len(variants['operational_context']):,}",
        f"- Lineage entries: {len(lineage):,}",
        "- Imputation: none; fitted inside later training folds.",
        "",
        "| Variant | Feature family | Count |",
        "| --- | --- | ---: |",
    ]
    lineage_counts = lineage.groupby(["family"], as_index=False).size()
    for row in lineage_counts.itertuples(index=False):
        lines.append(f"| shared | {row.family} | {row.size:,} |")
    (output_dir / "feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate shared multibuilding AHU features")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/features"))
    parser.add_argument("--building-config", type=Path, default=Path("configs/multibuilding_ahu.yaml"))
    parser.add_argument("--feature-config", type=Path, default=Path("configs/feature_engineering.yaml"))
    parser.add_argument(
        "--allow-duplicate-keys",
        action="store_true",
        help="Retain flagged duplicate keys for the declared sensitivity policy",
    )
    args = parser.parse_args(argv)
    manifest = engineer_multibuilding(
        args.input_file,
        args.output_dir,
        args.building_config,
        args.feature_config,
        args.allow_duplicate_keys,
    )
    print(
        f"Generated {manifest['feature_counts']['physical_signal']:,} default and "
        f"{manifest['feature_counts']['operational_context']:,} context predictors "
        f"for {manifest['row_count']:,} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

