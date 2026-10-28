"""Quality-controlled pooled detection experiments."""

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


def _load_features(
    data_file: Path, feature_file: Path, feature_manifest_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    features = pd.read_csv(feature_file, low_memory=False).reset_index(drop=True)
    manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
    physical = list(manifest["feature_variants"]["physical_signal"])
    key_columns = ["building", "ahu", "timestamp", "source_row_id", "label"]
    missing = set(key_columns) | set(physical)
    if len(data) != len(features):
        raise ValueError("Cleaned data and feature table have different row counts")
    if not missing.issubset(features.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(missing - set(features.columns))}")
    for column in key_columns:
        if column == "timestamp":
            left = pd.to_datetime(data[column], errors="coerce")
            right = pd.to_datetime(features[column], errors="coerce")
        else:
            left = data[column].astype("string")
            right = features[column].astype("string")
        if not left.equals(right):
            raise ValueError(f"Feature table key column does not match cleaned data: {column}")
    return data, features, physical


def run_quality_detection(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_file: Path,
    duplicate_policy: str,
    requested_models: list[str] | None = None,
) -> dict[str, Any]:
    config = load_detection_config(config_file)
    models = list(MODEL_NAMES) if requested_models is None else requested_models
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown detection models: {sorted(unknown)}")
    data, features, physical = _load_features(
        data_file, feature_file, feature_manifest_file
    )
    assignments = pd.read_csv(splits_file, low_memory=False)
    _validate_assignments(assignments, len(data))
    labels = data["label"].astype("string").str.strip()
    normal_label = str(config["normal_label"]).strip()
    if labels.isna().any():
        raise ValueError("Quality-controlled detection input contains missing labels")
    y = (~labels.eq(normal_label)).astype(int)

    metrics: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for model_name in models:
        if model_name == "lstm_autoencoder" and not _lstm_available():
            statuses.append(
                {
                    "model": model_name,
                    "status": "unavailable",
                    "reason": "TensorFlow is not installed",
                }
            )
            continue
        for (design, fold), assignment in assignments.groupby(["design", "fold"], sort=True):
            train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
            test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
            train_features = features.iloc[train_ids].copy()
            test_features = features.iloc[test_ids].copy()
            train_features["row_id"] = train_ids.to_numpy()
            test_features["row_id"] = test_ids.to_numpy()
            y_train = y.iloc[train_ids]
            y_test = y.iloc[test_ids]
            if model_name == "normal_range_rule":
                predictions = _normal_range_prediction(
                    train_features,
                    test_features,
                    y_train,
                    physical,
                    config["normal_range_rule"],
                )
                scored = pd.Series(True, index=test_features.index)
            elif model_name == "isolation_forest":
                transformer = FoldTransformer(physical).fit(train_features)
                predictions = _isolation_forest(
                    transformer.transform(train_features),
                    transformer.transform(test_features),
                    y_train,
                    config,
                )
                scored = pd.Series(True, index=test_features.index)
            elif model_name == "xgboost":
                transformer = FoldTransformer(physical).fit(train_features)
                predictions = _xgboost(
                    transformer.transform(train_features),
                    transformer.transform(test_features),
                    y_train,
                    config,
                )
                scored = pd.Series(True, index=test_features.index)
            else:
                predictions, scored = _lstm_autoencoder(
                    train_features,
                    test_features,
                    y_train,
                    y_test,
                    physical,
                    config,
                )
            scored_ids = scored.loc[scored].index
            y_scored = y.loc[scored_ids]
            prediction_array = np.asarray(predictions)[scored.to_numpy()]
            row = _metric_row(design, fold, model_name, y_scored, prediction_array)
            row.update(
                {
                    "duplicate_policy": duplicate_policy,
                    "variant": "physical_signal",
                    "scored_rows": int(len(scored_ids)),
                    "unscored_rows": int(len(test_ids) - len(scored_ids)),
                    "coverage": float(len(scored_ids) / len(test_ids)) if len(test_ids) else np.nan,
                }
            )
            row["model"] = row.pop("baseline")
            metrics.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(metrics)
    fold_path = output_dir / "quality_detection_metrics_by_fold.csv"
    fold_frame.to_csv(fold_path, index=False)
    summary = (
        fold_frame.groupby(["duplicate_policy", "design", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            scored_rows=("scored_rows", "sum"),
            unscored_rows=("unscored_rows", "sum"),
            mean_coverage=("coverage", "mean"),
            mean_f1=("f1", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
        )
    )
    summary_path = output_dir / "quality_detection_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "duplicate_policy": duplicate_policy,
            "data_file": str(data_file.expanduser().resolve()),
            "feature_file": str(feature_file.expanduser().resolve()),
            "feature_manifest_file": str(feature_manifest_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_file.expanduser().resolve()),
            "row_count": int(len(data)),
            "physical_feature_count": len(physical),
            "models_requested": models,
            "model_status": statuses,
            "output_files": {"by_fold": str(fold_path), "summary": str(summary_path)},
        }
    )
    manifest_path = output_dir / "quality_detection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Quality-controlled pooled detection summary",
        "",
        "All models use the declared physical-signal feature variant. Imputation and scaling are fitted within each training fold.",
        "",
        "| Policy | Design | Model | Mean F1 | Mean balanced accuracy | Mean FPR | Mean coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values(["design", "mean_f1"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| {row.duplicate_policy} | {row.design} | {row.model} | {row.mean_f1:.3f} | "
            f"{row.mean_balanced_accuracy:.3f} | {row.mean_false_positive_rate:.3f} | "
            f"{row.mean_coverage:.3f} |"
        )
    if statuses:
        lines.extend(["", "## Not run", ""])
        lines.extend(f"- {item['model']}: {item['reason']}." for item in statuses)
    (output_dir / "quality_detection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "row_count": int(len(data)),
        "models_requested": models,
        "model_status": statuses,
        "fold_metrics": str(fold_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quality-controlled pooled detection")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duplicate-policy", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--models", help="Comma-separated subset of models; default is all four")
    args = parser.parse_args(argv)
    models = None if args.models is None else [name.strip() for name in args.models.split(",")]
    result = run_quality_detection(
        args.data_file,
        args.feature_file,
        args.feature_manifest,
        args.splits_file,
        args.output_dir,
        args.config,
        args.duplicate_policy,
        models,
    )
    print(
        f"Evaluated {len(result['models_requested'])} models on "
        f"{result['row_count']:,} quality-controlled rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
