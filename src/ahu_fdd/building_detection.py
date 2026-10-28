"""Leakage-safe detection experiments trained independently per building."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahu_fdd.baselines import (
    _metric_row,
    _normal_range_prediction,
    _validate_assignments,
)
from ahu_fdd.detection import (
    MODEL_NAMES,
    FoldTransformer,
    _isolation_forest,
    _lstm_autoencoder,
    _lstm_available,
    _xgboost,
    load_detection_config,
)

BUILDING_DESIGNS = ("building_ahu_held_out", "building_chronological")


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


def _load_inputs(
    data_file: Path, feature_file: Path, feature_manifest_file: Path, splits_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    features = pd.read_csv(feature_file, low_memory=False).reset_index(drop=True)
    manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
    physical = list(manifest["feature_variants"]["physical_signal"])
    key_columns = ["building", "ahu", "timestamp", "source_row_id", "label"]
    if len(data) != len(features):
        raise ValueError("Cleaned data and feature table have different row counts")
    missing = set(key_columns) | set(physical)
    if not missing.issubset(features.columns):
        raise ValueError(
            f"Feature table is missing columns: {sorted(missing - set(features.columns))}"
        )
    for column in key_columns:
        if column == "timestamp":
            left = pd.to_datetime(data[column], errors="coerce")
            right = pd.to_datetime(features[column], errors="coerce")
        else:
            left = data[column].astype("string")
            right = features[column].astype("string")
        if not left.equals(right):
            raise ValueError(f"Feature table key column does not match cleaned data: {column}")
    assignments = pd.read_csv(splits_file, low_memory=False)
    required = {"building_scope", "design", "fold", "row_id", "split"}
    missing_assignments = required - set(assignments.columns)
    if missing_assignments:
        raise ValueError(
            f"Building split assignments are missing columns: {sorted(missing_assignments)}"
        )
    _validate_assignments(assignments, len(data))
    if not set(assignments["design"].unique()).issubset(BUILDING_DESIGNS):
        raise ValueError("Building split assignments contain an unknown design")
    return data, features, assignments, physical


def _empty_metric(
    building: str,
    design: str,
    fold: str,
    model: str,
    stage: str,
    total_rows: int,
) -> dict[str, Any]:
    return {
        "building": building,
        "design": design,
        "fold": fold,
        "stage": stage,
        "model": model,
        "rows": 0,
        "normal_support": 0,
        "fault_support": 0,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "macro_f1": np.nan,
        "false_positive_rate": np.nan,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
        "scored_rows": 0,
        "unscored_rows": total_rows,
        "coverage": 0.0,
    }


def _metric(
    building: str,
    design: str,
    fold: str,
    model: str,
    stage: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    scored: pd.Series,
) -> dict[str, Any]:
    scored_ids = scored.loc[scored].index
    if not len(scored_ids):
        return _empty_metric(building, design, fold, model, stage, len(scored))
    y_scored = y_true.loc[scored_ids]
    prediction_array = np.asarray(predictions)[scored.to_numpy()]
    row = _metric_row(design, fold, model, y_scored, prediction_array)
    row["model"] = row.pop("baseline")
    row.update(
        {
            "building": building,
            "stage": stage,
            "scored_rows": int(len(scored_ids)),
            "unscored_rows": int(len(scored) - len(scored_ids)),
            "coverage": float(len(scored_ids) / len(scored)) if len(scored) else np.nan,
        }
    )
    return row


def _predict(
    model_name: str,
    raw_train: pd.DataFrame,
    raw_eval: pd.DataFrame,
    feature_train: pd.DataFrame,
    feature_eval: pd.DataFrame,
    y_train: pd.Series,
    y_eval: pd.Series,
    columns: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.Series]:
    if model_name == "normal_range_rule":
        predictions = _normal_range_prediction(
            raw_train,
            raw_eval,
            y_train,
            [column.removeprefix("sensor__") for column in columns if column.startswith("sensor__")],
            config["normal_range_rule"],
        )
        return predictions, pd.Series(True, index=raw_eval.index)
    if model_name == "isolation_forest":
        transformer = FoldTransformer(columns).fit(feature_train)
        predictions = _isolation_forest(
            transformer.transform(feature_train),
            transformer.transform(feature_eval),
            y_train,
            config,
        )
        return predictions, pd.Series(True, index=feature_eval.index)
    if model_name == "xgboost":
        transformer = FoldTransformer(columns).fit(feature_train)
        predictions = _xgboost(
            transformer.transform(feature_train),
            transformer.transform(feature_eval),
            y_train,
            config,
        )
        return predictions, pd.Series(True, index=feature_eval.index)
    sequence_train = feature_train.copy()
    sequence_eval = feature_eval.copy()
    if "row_id" not in sequence_train:
        sequence_train["row_id"] = sequence_train.index.to_numpy()
    if "row_id" not in sequence_eval:
        sequence_eval["row_id"] = sequence_eval.index.to_numpy()
    return _lstm_autoencoder(
        sequence_train,
        sequence_eval,
        y_train,
        y_eval,
        columns,
        config,
    )


def _requested_models(requested: str | None) -> list[str]:
    models = list(MODEL_NAMES) if requested is None else [item.strip() for item in requested.split(",")]
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown detection models: {sorted(unknown)}")
    if not models:
        raise ValueError("At least one detection model is required")
    return models


def run_building_detection(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_file: Path,
    requested_models: str | None = None,
) -> dict[str, Any]:
    config = load_detection_config(config_file)
    models = _requested_models(requested_models)
    data, features, assignments, physical = _load_inputs(
        data_file, feature_file, feature_manifest_file, splits_file
    )
    labels = data["label"].astype("string").str.strip()
    normal_label = str(config["normal_label"]).strip()
    if labels.isna().any():
        raise ValueError("Building-specific detection input contains missing labels")
    y = (~labels.eq(normal_label)).astype(int)
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    selection_metric = str(config.get("building_specific", {}).get("selection_metric", "f1"))
    if selection_metric not in {"f1", "balanced_accuracy", "macro_f1"}:
        raise ValueError("building_specific.selection_metric must be f1, balanced_accuracy or macro_f1")

    grouped = assignments.groupby(["building_scope", "design", "fold"], sort=True)
    for (building, design, fold), assignment in grouped:
        train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
        validation_ids = assignment.loc[assignment["split"].eq("validation"), "row_id"].astype(int)
        test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
        if not set(data.iloc[train_ids]["building"].unique()) <= {building}:
            raise ValueError("Building-specific training rows contain another building")
        if not set(data.iloc[validation_ids]["building"].unique()) <= {building}:
            raise ValueError("Building-specific validation rows contain another building")
        if not set(data.iloc[test_ids]["building"].unique()) <= {building}:
            raise ValueError("Building-specific test rows contain another building")
        raw_train, raw_validation, raw_test = data.iloc[train_ids], data.iloc[validation_ids], data.iloc[test_ids]
        feature_train = features.iloc[train_ids]
        feature_validation = features.iloc[validation_ids]
        y_train, y_validation, y_test = y.iloc[train_ids], y.iloc[validation_ids], y.iloc[test_ids]
        if "lstm_autoencoder" in models and not _lstm_available():
            statuses.append({"building": building, "design": design, "model": "lstm_autoencoder", "status": "unavailable", "reason": "TensorFlow is not installed"})
        candidates: list[dict[str, Any]] = []
        for model_name in models:
            if model_name == "lstm_autoencoder" and not _lstm_available():
                continue
            predictions, scored = _predict(
                model_name,
                raw_train,
                raw_validation,
                feature_train,
                feature_validation,
                y_train,
                y_validation,
                physical,
                config,
            )
            row = _metric(building, design, fold, model_name, "validation", y_validation, predictions, scored)
            validation_rows.append(row)
            candidates.append(row)
        eligible = [row for row in candidates if np.isfinite(row[selection_metric])]
        if not eligible:
            raise ValueError(f"No model produced a valid validation {selection_metric} for {building}/{design}")
        selected = max(eligible, key=lambda row: (row[selection_metric], row["balanced_accuracy"]))
        selected_rows.append(
            {
                "building": building,
                "design": design,
                "fold": fold,
                "selected_model": selected["model"],
                "selection_metric": selection_metric,
                "validation_score": selected[selection_metric],
                "validation_balanced_accuracy": selected["balanced_accuracy"],
                "validation_coverage": selected["coverage"],
            }
        )
        combined_ids = pd.Index(train_ids.tolist() + validation_ids.tolist())
        combined_raw = data.iloc[combined_ids]
        combined_features = features.iloc[combined_ids]
        combined_y = y.iloc[combined_ids]
        predictions, scored = _predict(
            selected["model"],
            combined_raw,
            raw_test,
            combined_features,
            features.iloc[test_ids],
            combined_y,
            y_test,
            physical,
            config,
        )
        test_rows.append(_metric(building, design, fold, selected["model"], "test", y_test, predictions, scored))

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_frame = pd.DataFrame(validation_rows)
    test_frame = pd.DataFrame(test_rows)
    selected_frame = pd.DataFrame(selected_rows)
    validation_path = output_dir / "building_detection_validation_metrics.csv"
    test_path = output_dir / "building_detection_test_metrics.csv"
    selected_path = output_dir / "building_detection_selected_models.csv"
    validation_frame.to_csv(validation_path, index=False)
    test_frame.to_csv(test_path, index=False)
    selected_frame.to_csv(selected_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "feature_file": str(feature_file.expanduser().resolve()),
            "feature_manifest_file": str(feature_manifest_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_file.expanduser().resolve()),
            "row_count": int(len(data)),
            "buildings": sorted(data["building"].astype(str).unique()),
            "designs": list(BUILDING_DESIGNS),
            "models_requested": models,
            "variant": "physical_signal",
            "physical_feature_count": len(physical),
            "selection_metric": selection_metric,
            "model_status": statuses,
            "output_files": {
                "validation": str(validation_path),
                "test": str(test_path),
                "selected_models": str(selected_path),
            },
        }
    )
    manifest_path = output_dir / "building_detection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Building-specific detection results",
        "",
        "Models were selected using validation F1, with balanced accuracy as a tie-break. The selected model was then retrained on training plus validation rows and evaluated once on the held-out test rows.",
        "",
        "## Selected models and test results",
        "",
        "| Building | Design | Selected model | Validation F1 | Test F1 | Test balanced accuracy | Test FPR | Test coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selection in selected_frame.itertuples(index=False):
        test = test_frame[(test_frame["building"] == selection.building) & (test_frame["design"] == selection.design)].iloc[0]
        lines.append(
            f"| {selection.building} | {selection.design} | {selection.selected_model} | "
            f"{selection.validation_score:.3f} | {test.f1:.3f} | {test.balanced_accuracy:.3f} | "
            f"{test.false_positive_rate:.3f} | {test.coverage:.3f} |"
        )
    lines.extend(["", "## Validation comparison", "", "See `building_detection_validation_metrics.csv` for every model.", ""])
    (output_dir / "building_detection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "row_count": int(len(data)),
        "validation_metrics": str(validation_path),
        "test_metrics": str(test_path),
        "selected_models": str(selected_path),
        "manifest": str(manifest_path),
        "model_status": statuses,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent detection models for each building")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--models", default=None, help="Comma-separated model names; defaults to all models")
    args = parser.parse_args(argv)
    result = run_building_detection(
        args.data_file,
        args.feature_file,
        args.feature_manifest,
        args.splits_file,
        args.output_dir,
        args.config,
        args.models,
    )
    print(f"Created building-specific detection results in {args.output_dir.resolve()}")
    if result["model_status"]:
        print(f"Unavailable models: {result['model_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
