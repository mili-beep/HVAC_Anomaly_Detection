"""XGBoost feature-set sensitivity experiments for pooled AHU detection."""

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

from ahu_fdd.baselines import _metric_row, _validate_assignments
from ahu_fdd.detection import FoldTransformer, _xgboost, load_detection_config
from ahu_fdd.splits import MODELING_DESIGN_NAMES

VARIANT_NAMES = ("default_bms_signals", "symptom_only", "raw_sensor_only")
DIRECT_SENSOR_PREFIX = "sensor__"
TEMPORAL_PREFIX = "temporal__"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool, str, int)) or value is None:
        return value
    return str(value)


def load_variant_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Feature variant configuration must be a mapping")
    missing = set(VARIANT_NAMES) - set(config)
    if missing:
        raise ValueError(f"Feature variant configuration is missing: {sorted(missing)}")
    return config


def _base_name(feature: str) -> str | None:
    if feature.startswith(DIRECT_SENSOR_PREFIX):
        return feature.removeprefix(DIRECT_SENSOR_PREFIX)
    if feature.startswith(TEMPORAL_PREFIX):
        parts = feature.split("__", 2)
        return parts[2] if len(parts) == 3 else None
    return None


def resolve_feature_variants(
    feature_manifest: dict[str, Any], variant_config: dict[str, Any]
) -> dict[str, list[str]]:
    """Resolve and validate the frozen feature policies against generated columns."""
    source = list(feature_manifest["feature_variants"]["physical_signal"])
    resolved: dict[str, list[str]] = {}
    for variant in VARIANT_NAMES:
        settings = variant_config[variant]
        if settings.get("source_variant") != "physical_signal":
            raise ValueError(f"Unsupported source variant for {variant}")
        expected_count = settings.get("expected_source_feature_count")
        if expected_count is not None and int(expected_count) != len(source):
            raise ValueError(
                f"Frozen source feature count for {variant} is {expected_count}, "
                f"but the generated manifest contains {len(source)}"
            )
        excluded_bases = set(settings.get("excluded_sensor_bases", []))
        excluded_names = set(settings.get("excluded_feature_names", []))
        included_prefixes = tuple(settings.get("included_prefixes", []))
        columns = [
            feature
            for feature in source
            if feature not in excluded_names
            and _base_name(feature) not in excluded_bases
        ]
        if included_prefixes:
            columns = [feature for feature in columns if feature.startswith(included_prefixes)]
        if not columns:
            raise ValueError(f"Feature variant {variant} resolved to no columns")
        resolved[variant] = columns
    for variant in ("symptom_only", "raw_sensor_only"):
        if set(resolved[variant]) >= set(resolved["default_bms_signals"]):
            raise ValueError(f"{variant} must be a strict subset of default_bms_signals")
    return resolved


def _load_inputs(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    splits_file: Path,
    variant_config_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    features = pd.read_csv(feature_file, low_memory=False).reset_index(drop=True)
    feature_manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
    variant_config = load_variant_config(variant_config_file)
    variants = resolve_feature_variants(feature_manifest, variant_config)
    key_columns = ["building", "ahu", "timestamp", "source_row_id", "label"]
    required = set(key_columns) | set(variants["default_bms_signals"])
    if len(data) != len(features):
        raise ValueError("Cleaned data and feature table have different row counts")
    if not required.issubset(features.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(required - set(features.columns))}")
    for column in key_columns:
        left = pd.to_datetime(data[column], errors="coerce") if column == "timestamp" else data[column].astype("string")
        right = pd.to_datetime(features[column], errors="coerce") if column == "timestamp" else features[column].astype("string")
        if not left.equals(right):
            raise ValueError(f"Feature table key column does not match cleaned data: {column}")
    assignments = pd.read_csv(splits_file, low_memory=False)
    _validate_assignments(assignments, len(data))
    return data, features, assignments, variants


def run_feature_sensitivity(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_file: Path,
    variant_config_file: Path,
) -> dict[str, Any]:
    config = load_detection_config(config_file)
    variant_config = load_variant_config(variant_config_file)
    data, features, assignments, variants = _load_inputs(
        data_file, feature_file, feature_manifest_file, splits_file, variant_config_file
    )
    generated_manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
    variants = resolve_feature_variants(generated_manifest, variant_config)
    assignments = assignments.loc[assignments["design"].isin(MODELING_DESIGN_NAMES)].copy()
    labels = data["label"].astype("string").str.strip()
    normal_label = str(config["normal_label"]).strip()
    y = (~labels.eq(normal_label)).astype(int)

    rows: list[dict[str, Any]] = []
    for variant, columns in variants.items():
        for (design, fold), assignment in assignments.groupby(["design", "fold"], sort=True):
            train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
            test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
            train_frame = features.iloc[train_ids]
            test_frame = features.iloc[test_ids]
            transformer = FoldTransformer(columns).fit(train_frame)
            predictions = _xgboost(
                transformer.transform(train_frame),
                transformer.transform(test_frame),
                y.iloc[train_ids],
                config,
            )
            metric = _metric_row(design, fold, f"xgboost_{variant}", y.iloc[test_ids], predictions)
            metric.update(
                {
                    "variant": variant,
                    "feature_count": len(columns),
                    "usable_feature_count": len(transformer.usable),
                    "train_rows": len(train_ids),
                    "test_rows": len(test_ids),
                    "duplicate_policy": "exclude_ambiguous_keys",
                }
            )
            rows.append(metric)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(rows)
    fold_path = output_dir / "pooled_xgboost_feature_sensitivity_by_fold.csv"
    fold_frame.to_csv(fold_path, index=False)
    summary = (
        fold_frame.groupby(["variant", "design"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            feature_count=("feature_count", "first"),
            mean_f1=("f1", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
            test_rows=("test_rows", "sum"),
        )
    )
    summary_path = output_dir / "pooled_xgboost_feature_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "feature_file": str(feature_file.expanduser().resolve()),
            "feature_manifest_file": str(feature_manifest_file.expanduser().resolve()),
            "variant_config_file": str(variant_config_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_file.expanduser().resolve()),
            "duplicate_policy": "exclude_ambiguous_keys",
            "model": "xgboost",
            "variants": variants,
            "modeling_designs": list(MODELING_DESIGN_NAMES),
            "output_files": {"by_fold": str(fold_path), "summary": str(summary_path)},
        }
    )
    manifest_path = output_dir / "pooled_xgboost_feature_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Pooled XGBoost feature sensitivity",
        "",
        "The default BMS-signal model is the practical headline. Raw-sensor-only and symptoms-only are feature ablations, not simple baselines.",
        "",
        "| Variant | Design | Folds | Features | Mean F1 | Mean balanced accuracy | Mean FPR | Test rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values(["design", "variant"]).itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.design} | {row.folds} | {row.feature_count} | {row.mean_f1:.3f} | "
            f"{row.mean_balanced_accuracy:.3f} | {row.mean_false_positive_rate:.3f} | {row.test_rows:,} |"
        )
    (output_dir / "pooled_xgboost_feature_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"fold_metrics": str(fold_path), "summary": str(summary_path), "manifest": str(manifest_path)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pooled XGBoost feature sensitivity")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--variant-config", type=Path, default=Path("configs/feature_variants.yaml"))
    args = parser.parse_args(argv)
    run_feature_sensitivity(
        args.data_file,
        args.feature_file,
        args.feature_manifest,
        args.splits_file,
        args.output_dir,
        args.config,
        args.variant_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
