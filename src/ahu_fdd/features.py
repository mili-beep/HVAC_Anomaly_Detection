"""Phase 5 feature engineering and shortcut audit for prepared AHU data."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from ahu_fdd.audit import sha256_file

KEY_COLUMNS = ["building", "ahu", "timestamp", "source_row_id", "label_raw", "target_binary"]


def load_feature_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sensor_feature_columns"), list):
        raise ValueError("The feature policy must define sensor_feature_columns")
    sensors = config["sensor_feature_columns"]
    if not sensors or len(sensors) != len(set(sensors)):
        raise ValueError("sensor_feature_columns must be a unique non-empty list")
    windows = config.get("rolling_windows_rows")
    if not isinstance(windows, list) or not windows or any(int(window) < 1 for window in windows):
        raise ValueError("rolling_windows_rows must contain positive window sizes")
    return config


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _add_feature(
    values: dict[str, pd.Series], lineage: list[dict[str, Any]], name: str,
    series: pd.Series, family: str, source_columns: list[str], formula: str,
    past_only: bool = False, quality: bool = False,
) -> None:
    if name in values:
        raise ValueError(f"Feature generated twice: {name}")
    values[name] = series
    lineage.append(
        {
            "feature": name,
            "family": family,
            "source_columns": source_columns,
            "formula": formula,
            "past_only": past_only,
            "quality_or_context": quality,
        }
    )


def _past_rolling(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, indices in frame.groupby("ahu", sort=False).groups.items():
        values = _numeric(frame.loc[indices], column)
        result.loc[indices] = values.shift(1).rolling(window, min_periods=1).mean().to_numpy()
    return result


def _past_rate(frame: pd.DataFrame, column: str, max_gap_hours: float) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, indices in frame.groupby("ahu", sort=False).groups.items():
        values = _numeric(frame.loc[indices], column)
        timestamps = pd.to_datetime(frame.loc[indices, "timestamp"])
        previous_values = values.shift(1)
        hours = timestamps.diff().dt.total_seconds().div(3600)
        valid = hours.gt(0) & hours.le(max_gap_hours) & values.notna() & previous_values.notna()
        result.loc[indices] = ((values - previous_values).div(hours).where(valid)).to_numpy()
    return result


def engineer_features(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Generate features after sorting, without using labels or future rows."""
    sensors = list(config["sensor_feature_columns"])
    required = set(KEY_COLUMNS) | {"timestamp", "ahu"} | set(sensors)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature input is missing columns: {sorted(missing)}")
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise ValueError("Feature input contains unparseable timestamps")
    if ordered.duplicated(["ahu", "timestamp"], keep=False).any():
        raise ValueError("Feature input contains duplicate AHU/timestamp keys")
    ordered = ordered.sort_values(["ahu", "timestamp", "source_row_id"], kind="stable").reset_index(drop=True)

    values: dict[str, pd.Series] = {}
    lineage: list[dict[str, Any]] = []
    raw_names: dict[str, str] = {}
    for sensor in sensors:
        name = f"sensor__{_slug(sensor)}"
        raw_names[sensor] = name
        _add_feature(values, lineage, name, _numeric(ordered, sensor), "raw_sensor", [sensor], sensor)

    def sensor(name: str) -> pd.Series:
        if name not in ordered:
            return pd.Series(np.nan, index=ordered.index, dtype="float64")
        return _numeric(ordered, name)

    setpoint = sensor("Set point temperature")
    supply = sensor("Supply air temperature")
    return_air = sensor("Return temperature")
    valve = sensor("Valve position")
    fan = sensor("Supply fan")
    heating_supply = sensor("Heating supply temperature")
    cooling_1 = sensor("Cooling supply temperature 1")
    cooling_2 = sensor("Cooling supply temperature 2")
    _add_feature(values, lineage, "relationship__supply_setpoint_residual", supply - setpoint, "relationship", ["Supply air temperature", "Set point temperature"], "supply_air_temperature - setpoint_temperature")
    _add_feature(values, lineage, "relationship__return_setpoint_residual", return_air - setpoint, "relationship", ["Return temperature", "Set point temperature"], "return_temperature - setpoint_temperature")
    _add_feature(values, lineage, "relationship__supply_return_delta", supply - return_air, "relationship", ["Supply air temperature", "Return temperature"], "supply_air_temperature - return_temperature")
    _add_feature(values, lineage, "relationship__supply_return_abs_delta", (supply - return_air).abs(), "relationship", ["Supply air temperature", "Return temperature"], "abs(supply_air_temperature - return_temperature)")
    _add_feature(values, lineage, "relationship__heating_supply_air_delta", heating_supply - supply, "relationship", ["Heating supply temperature", "Supply air temperature"], "heating_supply_temperature - supply_air_temperature")
    _add_feature(values, lineage, "relationship__cooling_supply_air_delta", supply - cooling_1, "relationship", ["Supply air temperature", "Cooling supply temperature 1"], "supply_air_temperature - cooling_supply_temperature_1")
    _add_feature(values, lineage, "relationship__cooling_sensor_delta", cooling_1 - cooling_2, "relationship", ["Cooling supply temperature 1", "Cooling supply temperature 2"], "cooling_supply_temperature_1 - cooling_supply_temperature_2")
    _add_feature(values, lineage, "relationship__fan_valve_interaction", fan * valve, "actuator_relationship", ["Supply fan", "Valve position"], "supply_fan * valve_position")

    heating_total = sensor("Total heating pump")
    heating_components = [sensor(f"Heating pump {i}") for i in (1, 2, 3)]
    cooling_total = sensor("Total cooling pump")
    cooling_components = [sensor(f"Cooling pump {i}") for i in (1, 2)]
    _add_feature(values, lineage, "relationship__heating_pump_total_residual", heating_total - sum(heating_components), "pump_relationship", ["Total heating pump", "Heating pump 1", "Heating pump 2", "Heating pump 3"], "total_heating_pump - sum(heating_pump_1:3)")
    _add_feature(values, lineage, "relationship__cooling_pump_total_residual", cooling_total - sum(cooling_components), "pump_relationship", ["Total cooling pump", "Cooling pump 1", "Cooling pump 2"], "total_cooling_pump - sum(cooling_pump_1:2)")
    heating_any = pd.concat(heating_components, axis=1).notna().any(axis=1) & pd.concat(heating_components, axis=1).fillna(0).gt(0).any(axis=1)
    cooling_any = pd.concat(cooling_components, axis=1).notna().any(axis=1) & pd.concat(cooling_components, axis=1).fillna(0).gt(0).any(axis=1)
    _add_feature(values, lineage, "relationship__heating_pump_any_on", heating_any.astype(float), "pump_relationship", ["Heating pump 1", "Heating pump 2", "Heating pump 3"], "any(heating_pump_1:3 > 0)")
    _add_feature(values, lineage, "relationship__cooling_pump_any_on", cooling_any.astype(float), "pump_relationship", ["Cooling pump 1", "Cooling pump 2"], "any(cooling_pump_1:2 > 0)")

    for sensor_name in sensors:
        slug = _slug(sensor_name)
        for window in config["rolling_windows_rows"]:
            name = f"temporal__past_mean_{int(window)}__{slug}"
            _add_feature(values, lineage, name, _past_rolling(ordered, sensor_name, int(window)), "past_rolling_mean", [sensor_name], f"mean(previous {int(window)} rows of {sensor_name})", past_only=True)
        name = f"temporal__past_rate__{slug}"
        _add_feature(values, lineage, name, _past_rate(ordered, sensor_name, float(config["rate_max_gap_hours"])), "past_rate_of_change", [sensor_name, "timestamp"], f"delta({sensor_name}) / elapsed_hours; max_gap={config['rate_max_gap_hours']}", past_only=True)

    temporal_features = [entry["feature"] for entry in lineage if entry["family"].startswith("past_")]
    physical_features = [
        entry["feature"]
        for entry in lineage
        if entry["family"] in {"raw_sensor", "relationship", "actuator_relationship", "pump_relationship"}
    ] + temporal_features
    operational_features = list(physical_features)
    quality_features: list[str] = []
    for column in config.get("operational_quality_columns", []):
        if column in ordered:
            name = f"quality__{_slug(column)}"
            _add_feature(values, lineage, name, ordered[column].astype(float) if ordered[column].dtype == bool else ordered[column], "operating_context", [column], column, quality=True)
            quality_features.append(name)
    if config.get("operational_sensor_availability", False):
        for sensor_name in sensors:
            source = f"sensor_available__{sensor_name}"
            if source in ordered:
                name = f"quality__available__{_slug(sensor_name)}"
                _add_feature(values, lineage, name, ordered[source].astype(float), "sensor_availability", [source], f"observed({sensor_name})", quality=True)
                quality_features.append(name)
    if config.get("operational_missingness_indicators", False):
        for sensor_name in sensors:
            source = f"is_missing__{sensor_name}"
            if source in ordered:
                name = f"quality__missing__{_slug(sensor_name)}"
                _add_feature(values, lineage, name, ordered[source].astype(float), "missingness", [source], f"is_missing({sensor_name})", quality=True)
                quality_features.append(name)
    operational_features += quality_features

    feature_frame = pd.concat([ordered[KEY_COLUMNS].reset_index(drop=True), pd.DataFrame(values)], axis=1)
    all_feature_names = physical_features + quality_features
    if len(all_feature_names) != len(set(all_feature_names)):
        raise ValueError("Feature variant contains duplicate names")
    forbidden = set(KEY_COLUMNS) | {"label", "labeling", "target_binary", "label_raw", "ahu", "timestamp"}
    if forbidden & set(all_feature_names):
        raise ValueError("Metadata or labels leaked into predictor columns")
    variants = {"physical_signal": physical_features, "operational_context": operational_features}
    return feature_frame, pd.DataFrame(lineage), variants


def shortcut_audit(
    feature_frame: pd.DataFrame, lineage: pd.DataFrame, variants: dict[str, list[str]], config: dict[str, Any]
) -> pd.DataFrame:
    """Measure feature/label associations for review; never changes predictors."""
    y = feature_frame["target_binary"].eq("fault").astype(int)
    rows: list[dict[str, Any]] = []
    lineage_map = lineage.set_index("feature").to_dict(orient="index")
    missing_threshold = float(config["shortcut_audit"]["missing_rate_gap_flag"])
    auc_threshold = float(config["shortcut_audit"]["auc_distance_flag"])
    for variant, features in variants.items():
        for feature in features:
            values = pd.to_numeric(feature_frame[feature], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            missing_rates = feature_frame.assign(_missing=~valid).groupby(y)['_missing'].mean()
            normal_missing = float(missing_rates.get(0, np.nan))
            fault_missing = float(missing_rates.get(1, np.nan))
            value_auc = np.nan
            if valid.sum() and y[valid].nunique() == 2 and values[valid].nunique() > 1:
                value_auc = float(roc_auc_score(y[valid], values[valid]))
            missing_auc = np.nan
            if y.nunique() == 2 and (~valid).nunique() > 1:
                missing_auc = float(roc_auc_score(y, (~valid).astype(int)))
            missing_gap = abs(fault_missing - normal_missing) if np.isfinite(normal_missing) else np.nan
            auc_distance = abs(value_auc - 0.5) if np.isfinite(value_auc) else np.nan
            missing_shortcut = bool(np.isfinite(missing_gap) and missing_gap >= missing_threshold)
            value_shortcut = bool(np.isfinite(auc_distance) and auc_distance >= auc_threshold)
            rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "family": lineage_map[feature]["family"],
                    "normal_missing_rate": normal_missing,
                    "fault_missing_rate": fault_missing,
                    "missing_rate_gap": missing_gap,
                    "value_auc": value_auc,
                    "missingness_auc": missing_auc,
                    "missingness_shortcut_flag": missing_shortcut,
                    "value_separation_flag": value_shortcut,
                    "review_required": missing_shortcut or value_shortcut,
                }
            )
    return pd.DataFrame(rows)


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


def engineer_auditorium(input_file: Path, output_dir: Path, config_path: Path) -> dict[str, Any]:
    config = load_feature_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_file, low_memory=False)
    features, lineage, variants = engineer_features(frame, config)
    audit = shortcut_audit(features, lineage, variants, config)
    feature_csv = output_dir / "auditorium_features.csv"
    lineage_csv = output_dir / "feature_lineage.csv"
    shortcut_csv = output_dir / "shortcut_audit.csv"
    features.to_csv(feature_csv, index=False)
    lineage.to_csv(lineage_csv, index=False)
    audit.to_csv(shortcut_csv, index=False)
    variant_paths: dict[str, str] = {}
    for variant, columns in variants.items():
        path = output_dir / f"auditorium_{variant}_features.csv"
        features[KEY_COLUMNS + columns].to_csv(path, index=False)
        variant_paths[variant] = str(path)
    manifest = _json_safe(
        {
            "schema_version": int(config["schema_version"]),
            "generated_at": datetime.now().astimezone().isoformat(),
            "building": config["building"],
            "input_file": str(input_file.resolve()),
            "input_sha256": sha256_file(input_file),
            "policy_file": str(config_path.resolve()),
            "input_rows": int(len(frame)),
            "output_rows": int(len(features)),
            "feature_count": int(len(set().union(*variants.values()))),
            "feature_variants": variants,
            "lineage_file": str(lineage_csv),
            "shortcut_audit_file": str(shortcut_csv),
            "review_required_feature_count": int(audit["review_required"].sum()),
            "past_only_features_verified": bool(lineage.loc[lineage["past_only"], "past_only"].all()),
            "output_files": {"features": str(feature_csv), **variant_paths},
        }
    )
    manifest_path = output_dir / "phase5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Engineer auditorium AHU features and audit shortcuts")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/auditorium/phase5"))
    parser.add_argument("--config", type=Path, default=Path("configs/auditorium.yaml"))
    args = parser.parse_args(argv)
    manifest = engineer_auditorium(args.input_file, args.output_dir, args.config)
    print(f"Generated {manifest['feature_count']} auditorium features into {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
