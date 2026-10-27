"""Normal-operation pattern discovery for prepared AHU data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ahu_fdd.audit import sha256_file


def load_pattern_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sensor_feature_columns"), list):
        raise ValueError("The pattern policy must define sensor_feature_columns")
    sensors = config["sensor_feature_columns"]
    if not sensors or len(sensors) != len(set(sensors)):
        raise ValueError("sensor_feature_columns must be a unique non-empty list")
    return config


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _mode_column(frame: pd.DataFrame) -> pd.Series:
    if "operating_mode" in frame:
        return frame["operating_mode"].astype("string").fillna("unknown")
    if {"heating_active", "cooling_active"} <= set(frame.columns):
        heating = frame["heating_active"].astype(bool)
        cooling = frame["cooling_active"].astype(bool)
        return pd.Series(
            np.select(
                [heating & ~cooling, cooling & ~heating, heating & cooling],
                ["heating", "cooling", "mixed"],
                default="unknown",
            ),
            index=frame.index,
            dtype="string",
        )
    return pd.Series("unknown", index=frame.index, dtype="string")


def _profile_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    sensors: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        for sensor in sensors:
            values = _numeric(group, sensor)
            observed = values.dropna()
            rows.append(
                {
                    **key_values,
                    "sensor": sensor,
                    "row_count": int(len(group)),
                    "observed_count": int(observed.size),
                    "missing_rate": float(values.isna().mean()),
                    "mean": float(observed.mean()) if not observed.empty else np.nan,
                    "median": float(observed.median()) if not observed.empty else np.nan,
                    "std": float(observed.std(ddof=0)) if not observed.empty else np.nan,
                    "minimum": float(observed.min()) if not observed.empty else np.nan,
                    "maximum": float(observed.max()) if not observed.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _pca_outputs(
    normal: pd.DataFrame,
    sensors: list[str],
    pca_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if not pca_config.get("enabled", True):
        return pd.DataFrame(), pd.DataFrame(), []
    coordinate_frames: list[pd.DataFrame] = []
    variance_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    max_rows = int(pca_config.get("max_rows_per_mode", 20000))
    random_state = int(pca_config.get("random_state", 42))
    for mode, mode_frame in normal.groupby("operating_mode", sort=True):
        available = [
            sensor for sensor in sensors
            if sensor in mode_frame
            and _numeric(mode_frame, sensor).notna().sum() > 1
            and _numeric(mode_frame, sensor).nunique(dropna=True) > 1
        ]
        if len(available) < 2:
            mode_rows.append({"mode": mode, "status": "skipped_less_than_two_variable_sensors", "row_count": len(mode_frame)})
            continue
        selected = mode_frame
        sampled = False
        if len(selected) > max_rows:
            selected = selected.sample(max_rows, random_state=random_state)
            selected = selected.sort_values(["ahu", "timestamp", "source_row_id"], kind="stable")
            sampled = True
        matrix = selected[available].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.fillna(matrix.median())
        scaled = StandardScaler().fit_transform(matrix)
        model = PCA(n_components=2, random_state=random_state)
        coordinates = model.fit_transform(scaled)
        coordinate_frames.append(
            pd.DataFrame(
                {
                    "building": selected["building"].to_numpy(),
                    "ahu": selected["ahu"].to_numpy(),
                    "timestamp": selected["timestamp"].to_numpy(),
                    "source_row_id": selected["source_row_id"].to_numpy(),
                    "operating_mode": mode,
                    "pc1": coordinates[:, 0],
                    "pc2": coordinates[:, 1],
                }
            )
        )
        variance_rows.extend(
            {
                "operating_mode": mode,
                "component": f"PC{i + 1}",
                "explained_variance_ratio": float(value),
                "cumulative_explained_variance_ratio": float(np.sum(model.explained_variance_ratio_[: i + 1])),
                "sensor_count": len(available),
                "row_count": len(selected),
                "sampled": sampled,
            }
            for i, value in enumerate(model.explained_variance_ratio_)
        )
        mode_rows.append(
            {
                "mode": mode,
                "status": "completed",
                "row_count": len(mode_frame),
                "rows_used": len(selected),
                "sampled": sampled,
                "sensors_used": ",".join(available),
            }
        )
    coordinates = pd.concat(coordinate_frames, ignore_index=True) if coordinate_frames else pd.DataFrame()
    variance = pd.DataFrame(variance_rows)
    return coordinates, variance, mode_rows


def discover_patterns(
    frame: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    sensors = list(config["sensor_feature_columns"])
    required = {"building", "ahu", "timestamp", "source_row_id", "label_raw"} | set(sensors)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pattern input is missing columns: {sorted(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Pattern input contains unparseable timestamps")
    if data.duplicated(["ahu", "timestamp"], keep=False).any():
        raise ValueError("Pattern input contains duplicate AHU/timestamp keys")
    data = data.sort_values(["ahu", "timestamp", "source_row_id"], kind="stable").reset_index(drop=True)
    data["operating_mode"] = _mode_column(data)
    data["hour"] = data["timestamp"].dt.hour
    data["month"] = data["timestamp"].dt.month
    normal_label = str(config["normal_label"])
    normal = data.loc[data["label_raw"].eq(normal_label)].copy()
    if normal.empty:
        raise ValueError(f"No rows with normal_label={normal_label!r}")

    mode_summary = (
        normal.groupby("operating_mode", dropna=False)
        .agg(row_count=("source_row_id", "size"), ahu_count=("ahu", "nunique"))
        .reset_index()
    )
    ahu_summary = (
        normal.groupby(["building", "ahu", "operating_mode"], dropna=False)
        .agg(row_count=("source_row_id", "size"), first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"))
        .reset_index()
    )
    hourly = _profile_table(normal, ["operating_mode", "hour"], sensors)
    monthly = _profile_table(normal, ["operating_mode", "month"], sensors)
    sensor_profile = _profile_table(normal, ["operating_mode"], sensors)

    transitions = normal[["ahu", "timestamp", "operating_mode"]].copy()
    transitions["previous_mode"] = transitions.groupby("ahu", sort=False)["operating_mode"].shift(1)
    transitions = transitions.loc[
        transitions["previous_mode"].notna()
        & transitions["operating_mode"].ne(transitions["previous_mode"])
    ]
    transition_summary = (
        transitions.groupby(["previous_mode", "operating_mode"], dropna=False)
        .size()
        .rename("transition_count")
        .reset_index()
    )
    pca_coordinates, pca_variance, pca_modes = _pca_outputs(normal, sensors, config.get("pca", {}))
    summary = {
        "input_rows": int(len(data)),
        "normal_rows": int(len(normal)),
        "non_normal_rows_excluded_from_profiles": int(len(data) - len(normal)),
        "ahu_count": int(normal["ahu"].nunique()),
        "mode_counts": normal["operating_mode"].value_counts().to_dict(),
        "normal_label": normal_label,
        "pca_modes": pca_modes,
    }
    return {
        "normal": normal,
        "mode_summary": mode_summary,
        "ahu_summary": ahu_summary,
        "sensor_profile": sensor_profile,
        "hourly_profile": hourly,
        "monthly_profile": monthly,
        "transition_summary": transition_summary,
        "pca_coordinates": pca_coordinates,
        "pca_variance": pca_variance,
        "summary": summary,
    }


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


def _write_figures(outputs: dict[str, Any], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    hourly = outputs["hourly_profile"]
    chosen = hourly.loc[
        hourly["sensor"].isin(["Supply air temperature", "Return temperature"])
    ]
    if not chosen.empty:
        figure, axis = plt.subplots(figsize=(9, 5))
        for (mode, sensor), group in chosen.groupby(["operating_mode", "sensor"], sort=True):
            axis.plot(group["hour"], group["mean"], marker="o", label=f"{mode}: {sensor}")
        axis.set(title="Normal hourly temperature profiles", xlabel="Hour of day", ylabel="Mean value")
        axis.legend(fontsize="small")
        axis.grid(alpha=0.25)
        path = figure_dir / "normal_hourly_temperature_profiles.png"
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(str(path))
    pca = outputs["pca_coordinates"]
    if not pca.empty:
        for mode, group in pca.groupby("operating_mode", sort=True):
            figure, axis = plt.subplots(figsize=(7, 5))
            for ahu, ahu_group in group.groupby("ahu", sort=True):
                axis.scatter(ahu_group["pc1"], ahu_group["pc2"], s=5, alpha=0.25, label=ahu)
            axis.set(title=f"Normal-operation PCA: {mode}", xlabel="PC1", ylabel="PC2")
            if group["ahu"].nunique() <= 15:
                axis.legend(fontsize="xx-small", ncol=2)
            axis.grid(alpha=0.2)
            path = figure_dir / f"normal_pca_{mode}.png"
            figure.tight_layout()
            figure.savefig(path, dpi=140)
            plt.close(figure)
            paths.append(str(path))
    return paths


def discover_auditorium(input_file: Path, output_dir: Path, config_path: Path) -> dict[str, Any]:
    config = load_pattern_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_file, low_memory=False)
    outputs = discover_patterns(frame, config)
    files: dict[str, str] = {}
    for name in ["mode_summary", "ahu_summary", "sensor_profile", "hourly_profile", "monthly_profile", "transition_summary", "pca_coordinates", "pca_variance"]:
        path = output_dir / f"{name}.csv"
        outputs[name].to_csv(path, index=False)
        files[name] = str(path)
    figure_paths = _write_figures(outputs, output_dir / "figures")
    summary = outputs["summary"]
    manifest = _json_safe(
        {
            "schema_version": int(config["schema_version"]),
            "generated_at": datetime.now().astimezone().isoformat(),
            "building": config["building"],
            "input_file": str(input_file.resolve()),
            "input_sha256": sha256_file(input_file),
            "policy_file": str(config_path.resolve()),
            "summary": summary,
            "profile_rows_are_normal_only": True,
            "labels_used_as_predictors": False,
            "outputs": {**files, "figures": figure_paths},
        }
    )
    manifest_path = output_dir / "pattern_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = output_dir / "pattern_summary.md"
    report.write_text(
        "# Auditorium normal-operation pattern discovery\n\n"
        f"- Input rows: {summary['input_rows']:,}\n"
        f"- Normal rows used for profiles: {summary['normal_rows']:,}\n"
        f"- Non-normal rows excluded from profiles: {summary['non_normal_rows_excluded_from_profiles']:,}\n"
        f"- AHUs represented: {summary['ahu_count']:,}\n"
        f"- Operating-mode counts: `{summary['mode_counts']}`\n\n"
        "Profiles and PCA are descriptive normal-operation artefacts. They do\n"
        "not constitute fault labels or a trained detector.\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover normal-operation patterns in prepared AHU data")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/auditorium/patterns"))
    parser.add_argument("--config", type=Path, default=Path("configs/auditorium.yaml"))
    args = parser.parse_args(argv)
    manifest = discover_auditorium(args.input_file, args.output_dir, args.config)
    print(f"Discovered patterns from {manifest['summary']['normal_rows']:,} normal rows into {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
