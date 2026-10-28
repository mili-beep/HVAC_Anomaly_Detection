"""Leakage-safe binary detection experiments for the multibuilding AHU data."""

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
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ahu_fdd.baselines import (
    _metric_row,
    _normal_range_prediction,
    _validate_assignments,
)

MODEL_NAMES = (
    "normal_range_rule",
    "isolation_forest",
    "xgboost",
    "lstm_autoencoder",
)
VARIANT_NAMES = ("physical_signal", "operational_context")


def load_detection_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The detection configuration must be a mapping")
    if str(config.get("normal_label", "")).strip() == "":
        raise ValueError("The detection configuration must define normal_label")
    for model in MODEL_NAMES:
        if model not in config.get("models", {}):
            raise ValueError(f"The detection configuration must define {model}")
    return config


def load_sensor_config(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    columns = config.get("canonical_sensor_columns") if isinstance(config, dict) else None
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise ValueError("The building configuration must define unique sensor columns")
    return list(columns)


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


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def build_detection_features(
    frame: pd.DataFrame, sensor_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create row-local physical and operational features without fitting anything."""
    required = {"building", "ahu", "timestamp", "label"} | set(sensor_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Detection input is missing columns: {sorted(missing)}")
    ordered = frame.copy().reset_index(drop=True)
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise ValueError("Detection input contains unparseable timestamps")
    ordered["row_id"] = np.arange(len(ordered), dtype=np.int64)
    values: dict[str, pd.Series] = {}
    physical: list[str] = []

    def add(name: str, series: pd.Series, target: list[str]) -> None:
        if name in values:
            raise ValueError(f"Detection feature generated twice: {name}")
        values[name] = series.astype(float)
        target.append(name)

    numeric = {column: pd.to_numeric(ordered[column], errors="coerce").replace([np.inf, -np.inf], np.nan) for column in sensor_columns}
    for column in sensor_columns:
        name = f"sensor__{column}"
        add(name, numeric[column], physical)

    def sensor(column: str) -> pd.Series:
        return numeric[column]

    relationships = [
        ("relationship__supply_setpoint_residual", sensor("supply_air_temperature") - sensor("setpoint_temperature")),
        ("relationship__return_setpoint_residual", sensor("return_temperature") - sensor("setpoint_temperature")),
        ("relationship__supply_return_delta", sensor("supply_air_temperature") - sensor("return_temperature")),
        ("relationship__heating_supply_air_delta", sensor("heating_supply_temperature_1") - sensor("supply_air_temperature")),
        ("relationship__cooling_supply_air_delta", sensor("supply_air_temperature") - sensor("cooling_supply_temperature_1")),
        ("relationship__fan_valve_interaction", sensor("supply_fan") * sensor("valve_position")),
        ("relationship__heating_pump_total_residual", sensor("heating_pump_total") - sensor("heating_pump_1") - sensor("heating_pump_2") - sensor("heating_pump_3")),
        ("relationship__cooling_pump_total_residual", sensor("cooling_pump_total") - sensor("cooling_pump_1") - sensor("cooling_pump_2") - sensor("cooling_pump_3") - sensor("cooling_pump_4")),
    ]
    for name, values_series in relationships:
        add(name, values_series, physical)

    operational = list(physical)
    missing_features: list[str] = []
    for column in sensor_columns:
        name = f"quality__missing__{column}"
        add(name, numeric[column].isna().astype(float), missing_features)
    operational += missing_features

    heating = ["heating_exchanger", "heating_pump_total", "heating_pump_1", "heating_pump_2", "heating_pump_3"]
    cooling = ["cooling_pump_total", "cooling_pump_1", "cooling_pump_2", "cooling_pump_3", "cooling_pump_4"]
    heating_active = pd.concat([numeric[column] for column in heating], axis=1).gt(0).any(axis=1)
    cooling_active = pd.concat([numeric[column] for column in cooling], axis=1).gt(0).any(axis=1)
    add("mode__heating_active", heating_active.astype(float), operational)
    add("mode__cooling_active", cooling_active.astype(float), operational)
    add("mode__idle", (~heating_active & ~cooling_active).astype(float), operational)
    hour = ordered["timestamp"].dt.hour.astype(float)
    add("context__hour_sin", np.sin(2 * np.pi * hour / 24), operational)
    add("context__hour_cos", np.cos(2 * np.pi * hour / 24), operational)

    output = pd.concat(
        [ordered[["row_id", "building", "ahu", "timestamp", "label"]], pd.DataFrame(values)],
        axis=1,
    )
    return output, {"physical_signal": physical, "operational_context": operational}


class FoldTransformer:
    """Fit imputation/scaling on training rows only and transform another fold."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.usable: list[str] = []
        self.pipeline: Any = None

    def fit(self, train: pd.DataFrame) -> "FoldTransformer":
        finite = train[self.columns].replace([np.inf, -np.inf], np.nan)
        self.usable = [
            column
            for column in self.columns
            if finite[column].notna().any() and finite[column].nunique(dropna=True) > 1
        ]
        if self.usable:
            self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
            self.pipeline.fit(finite[self.usable])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.usable or self.pipeline is None:
            return np.empty((len(frame), 0), dtype=float)
        values = frame[self.usable].replace([np.inf, -np.inf], np.nan)
        return np.asarray(self.pipeline.transform(values), dtype=float)


def _majority(y_train: pd.Series, count: int) -> np.ndarray:
    majority = int(y_train.value_counts().sort_index().idxmax())
    return np.full(count, majority, dtype=int)


def _isolation_forest(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y_train: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    normal = train_matrix[y_train.to_numpy() == 0]
    if len(normal) < 2 or train_matrix.shape[1] == 0:
        return _majority(y_train, len(test_matrix))
    model_config = config["models"]["isolation_forest"]
    configured_max_samples = model_config.get("max_samples", "auto")
    max_samples = (
        min(int(configured_max_samples), len(normal))
        if configured_max_samples != "auto"
        else "auto"
    )
    model = IsolationForest(
        n_estimators=int(model_config["n_estimators"]),
        max_samples=max_samples,
        random_state=int(model_config["random_state"]),
        n_jobs=int(model_config.get("n_jobs", 2)),
    )
    model.fit(normal)
    threshold = np.quantile(
        model.score_samples(normal), float(model_config["normal_score_quantile"])
    )
    return (model.score_samples(test_matrix) <= threshold).astype(int)


def _xgboost(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y_train: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    if train_matrix.shape[1] == 0 or y_train.nunique() < 2:
        return _majority(y_train, len(test_matrix))
    model_config = dict(config["models"]["xgboost"])
    normal = int((y_train == 0).sum())
    fault = int((y_train == 1).sum())
    model_config["scale_pos_weight"] = normal / fault if fault else 1.0
    model = XGBClassifier(**model_config)
    model.fit(train_matrix, y_train.to_numpy())
    return model.predict(test_matrix).astype(int)


def _lstm_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _sequence_windows(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    sequence_length: int,
    max_gap_hours: float,
) -> tuple[np.ndarray, list[int]]:
    """Create within-AHU windows without crossing a timestamp gap."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if max_gap_hours <= 0:
        raise ValueError("max_gap_hours must be positive")
    chunks: list[np.ndarray] = []
    rows: list[int] = []
    for _, indices in frame.groupby(["building", "ahu"], sort=False).groups.items():
        ordered = frame.loc[indices].sort_values(["timestamp", "row_id"], kind="stable")
        positions = frame.index.get_indexer(ordered.index)
        timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError("Sequence input contains unparseable timestamps")
        hours = timestamps.diff().dt.total_seconds().div(3600)
        breaks = np.flatnonzero(hours.to_numpy()[1:] > max_gap_hours) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(positions)]))
        for start, end in zip(starts, ends):
            segment = positions[start:end]
            for offset in range(sequence_length - 1, len(segment)):
                window_positions = segment[offset - sequence_length + 1 : offset + 1]
                chunks.append(matrix[window_positions])
                rows.append(int(segment[offset]))
    if not chunks:
        return np.empty((0, sequence_length, matrix.shape[1]), dtype=float), []
    return np.stack(chunks), rows


def _lstm_autoencoder(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    columns: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.Series]:
    """Train an LSTM autoencoder on normal training sequences only.

    The returned mask excludes the first sequence_length - 1 rows of each test
    AHU stream because they do not have a complete within-fold history.
    """
    if not _lstm_available():
        raise RuntimeError("lstm_autoencoder requires TensorFlow; install the optional deep extra")
    import tensorflow as tf

    transformer = FoldTransformer(columns).fit(train.loc[y_train.eq(0)])
    if not transformer.usable:
        return _majority(y_train, len(test)), pd.Series(True, index=test.index)
    sequence_length = int(config["models"]["lstm_autoencoder"]["sequence_length"])
    train_matrix = transformer.transform(train)
    test_matrix = transformer.transform(test)

    sequence_gap = float(config["models"]["lstm_autoencoder"].get("sequence_max_gap_hours", 1))
    train_sequences, _ = _sequence_windows(
        train, train_matrix, sequence_length, sequence_gap
    )
    test_sequences, test_rows = _sequence_windows(
        test, test_matrix, sequence_length, sequence_gap
    )
    if len(train_sequences) == 0 or len(test_sequences) == 0:
        return _majority(y_train, len(test)), pd.Series(True, index=test.index)
    maximum = int(config["models"]["lstm_autoencoder"].get("max_train_sequences", 20000))
    if len(train_sequences) > maximum:
        rng = np.random.default_rng(int(config["models"]["lstm_autoencoder"]["random_state"]))
        selected = rng.choice(len(train_sequences), size=maximum, replace=False)
        train_sequences = train_sequences[selected]
    hidden = int(config["models"]["lstm_autoencoder"]["hidden_units"])
    inputs = tf.keras.Input(shape=(sequence_length, train_sequences.shape[2]))
    encoded = tf.keras.layers.LSTM(hidden)(inputs)
    decoded = tf.keras.layers.RepeatVector(sequence_length)(encoded)
    decoded = tf.keras.layers.LSTM(hidden, return_sequences=True)(decoded)
    decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(train_sequences.shape[2]))(decoded)
    model = tf.keras.Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        train_sequences,
        train_sequences,
        epochs=int(config["models"]["lstm_autoencoder"]["epochs"]),
        batch_size=int(config["models"]["lstm_autoencoder"]["batch_size"]),
        verbose=0,
        shuffle=False,
    )
    train_errors = np.mean(np.square(model.predict(train_sequences, verbose=0) - train_sequences), axis=(1, 2))
    test_errors = np.mean(np.square(model.predict(test_sequences, verbose=0) - test_sequences), axis=(1, 2))
    threshold = np.quantile(
        train_errors,
        float(config["models"]["lstm_autoencoder"]["normal_error_quantile"]),
    )
    predictions = _majority(y_train, len(test))
    predictions[np.asarray(test_rows)] = (test_errors >= threshold).astype(int)
    scored = pd.Series(False, index=test.index)
    scored.iloc[test_rows] = True
    return predictions, scored


def _requested_models(config: dict[str, Any], requested: str | None) -> list[str]:
    names = list(MODEL_NAMES) if requested is None else [name.strip() for name in requested.split(",")]
    unknown = set(names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown detection models: {sorted(unknown)}")
    return names


def run_detection(
    data_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_path: Path,
    building_config_path: Path,
    requested_models: str | None = None,
) -> dict[str, Any]:
    config = load_detection_config(config_path)
    models = _requested_models(config, requested_models)
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    assignments = pd.read_csv(splits_file, low_memory=False)
    _validate_assignments(assignments, len(data))
    sensors = load_sensor_config(building_config_path)
    features, variants = build_detection_features(data, sensors)
    labels = data["label"].astype("string").str.strip()
    normal_label = str(config["normal_label"]).strip()
    if labels.isna().any():
        raise ValueError("Detection input contains missing labels")
    y = (~labels.eq(normal_label)).astype(int)
    metrics: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for model_name in models:
        if model_name == "lstm_autoencoder" and not _lstm_available():
            statuses.append({"model": model_name, "status": "unavailable", "reason": "TensorFlow is not installed"})
            continue
        model_variants = ["physical_signal"] if model_name in {"normal_range_rule", "lstm_autoencoder"} else list(VARIANT_NAMES)
        for variant in model_variants:
            for (design, fold), assignment in assignments.groupby(["design", "fold"], sort=True):
                train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
                test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
                train_features = features.iloc[train_ids]
                test_features = features.iloc[test_ids]
                y_train = y.iloc[train_ids]
                y_test = y.iloc[test_ids]
                columns = variants[variant]
                if model_name == "normal_range_rule":
                    predictions = _normal_range_prediction(
                        data.iloc[train_ids], data.iloc[test_ids], y_train, sensors, config["normal_range_rule"]
                    )
                    scored = pd.Series(True, index=test_ids)
                elif model_name == "isolation_forest":
                    transformer = FoldTransformer(columns).fit(train_features)
                    predictions = _isolation_forest(
                        transformer.transform(train_features), transformer.transform(test_features), y_train, config
                    )
                    scored = pd.Series(True, index=test_ids)
                elif model_name == "xgboost":
                    transformer = FoldTransformer(columns).fit(train_features)
                    predictions = _xgboost(
                        transformer.transform(train_features), transformer.transform(test_features), y_train, config
                    )
                    scored = pd.Series(True, index=test_ids)
                else:
                    predictions, scored = _lstm_autoencoder(
                        train_features, test_features, y_train, y_test, columns, config
                    )
                scored_ids = scored.loc[scored].index
                y_scored = y.loc[scored_ids]
                prediction_array = np.asarray(predictions)[scored.to_numpy()]
                row = _metric_row(design, fold, model_name, y_scored, prediction_array)
                row.update({"variant": variant, "scored_rows": int(len(scored_ids)), "unscored_rows": int(len(test_ids) - len(scored_ids))})
                row["model"] = row.pop("baseline")
                metrics.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "detection_metrics_by_fold.csv"
    fold_frame = pd.DataFrame(metrics)
    fold_frame.to_csv(fold_path, index=False)
    summary = (
        fold_frame.groupby(["design", "variant", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            scored_rows=("scored_rows", "sum"),
            mean_f1=("f1", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
        )
    )
    summary_path = output_dir / "detection_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_path.expanduser().resolve()),
            "building_config_file": str(building_config_path.expanduser().resolve()),
            "row_count": int(len(data)),
            "models_requested": models,
            "model_status": statuses,
            "feature_variants": {name: len(columns) for name, columns in variants.items()},
            "metric_file": str(fold_path),
            "summary_file": str(summary_path),
        }
    )
    manifest_path = output_dir / "detection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = output_dir / "detection_summary.md"
    lines = [
        "# Detection experiment summary",
        "",
        "All scores are averages of fold-level metrics. Preprocessing is fitted on training rows only.",
        "",
        "| Design | Variant | Model | Mean F1 | Mean balanced accuracy | Mean false-positive rate |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values(["design", "variant", "mean_f1"], ascending=[True, True, False]).itertuples(index=False):
        lines.append(
            f"| {row.design} | {row.variant} | {row.model} | {row.mean_f1:.3f} | "
            f"{row.mean_balanced_accuracy:.3f} | {row.mean_false_positive_rate:.3f} |"
        )
    if statuses:
        lines += ["", "## Not run", ""]
        lines.extend(f"- `{item['model']}`: {item['reason']}." for item in statuses)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "row_count": int(len(data)),
        "models_requested": models,
        "model_status": statuses,
        "fold_metrics": str(fold_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "markdown": str(markdown),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run leakage-safe AHU detection experiments")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/detection"))
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--building-config", type=Path, default=Path("configs/multibuilding_ahu.yaml"))
    parser.add_argument("--models", help="Comma-separated subset of models; default is all four")
    args = parser.parse_args(argv)
    result = run_detection(
        args.data_file,
        args.splits_file,
        args.output_dir,
        args.config,
        args.building_config,
        args.models,
    )
    print(f"Evaluated {len(result['models_requested'])} requested detection models on {result['row_count']:,} rows")
    for status in result["model_status"]:
        print(f"Skipped {status['model']}: {status['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
