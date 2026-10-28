"""Feature robustness analyses for pooled AHU detection.

This module contains both declared feature-set sensitivity experiments and the
held-out single-feature dependence audit.  They share the same frozen inputs
and are exposed as separate commands for clear reporting.
"""

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
from sklearn.metrics import roc_auc_score

from ahu_fdd.metrics import binary_metric_row
from ahu_fdd.modeling import FoldTransformer, load_detection_config, xgboost
from ahu_fdd.splits import MODELING_DESIGN_NAMES
from ahu_fdd.utils import json_safe, numeric_frame, numeric_series, write_json
from ahu_fdd.validation import validate_assignments

VARIANT_NAMES = ("default_bms_signals", "symptom_only", "raw_sensor_only")
DIRECT_SENSOR_PREFIX = "sensor__"
TEMPORAL_PREFIX = "temporal__"


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
    validate_assignments(assignments, len(data))
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
            predictions = xgboost(
                transformer.transform(train_frame),
                transformer.transform(test_frame),
                y.iloc[train_ids],
                config,
            )
            metric = binary_metric_row(design, fold, f"xgboost_{variant}", y.iloc[test_ids], predictions)
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
    manifest = json_safe(
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
    write_json(manifest_path, manifest)
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


def absolute_roc_auc(y_true: pd.Series, values: pd.Series) -> float:
    """Return the direction-free AUC of one feature, or NaN if undefined."""
    finite = numeric_series(values)
    usable = finite.notna() & y_true.notna()
    labels = y_true.loc[usable].astype(int)
    scores = finite.loc[usable]
    if labels.nunique() < 2 or scores.nunique(dropna=True) < 2:
        return float("nan")
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def _load_dependence_inputs(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    feature_lineage_file: Path,
    splits_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    features = pd.read_csv(feature_file, low_memory=False).reset_index(drop=True)
    manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
    feature_names = list(manifest["feature_variants"]["physical_signal"])
    required = {"building", "ahu", "timestamp", "source_row_id", "label"} | set(feature_names)
    if len(data) != len(features):
        raise ValueError("Cleaned data and feature table have different row counts")
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    for column in ("building", "ahu", "timestamp", "source_row_id", "label"):
        left = data[column]
        right = features[column]
        if column == "timestamp":
            left = pd.to_datetime(left, errors="coerce")
            right = pd.to_datetime(right, errors="coerce")
        else:
            left = left.astype("string")
            right = right.astype("string")
        if not left.equals(right):
            raise ValueError(f"Feature table key column does not match cleaned data: {column}")
    assignments = pd.read_csv(splits_file, low_memory=False)
    validate_assignments(assignments, len(data))
    lineage = pd.read_csv(feature_lineage_file, low_memory=False)
    families = dict(zip(lineage["feature"], lineage["family"], strict=True))
    missing_families = set(feature_names) - set(families)
    if missing_families:
        raise ValueError(f"Feature lineage is missing: {sorted(missing_families)}")
    return data, features, assignments, feature_names, families


def run_feature_dependence(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    feature_lineage_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_file: Path,
) -> dict[str, str]:
    """Measure how much held-out separation a single feature provides."""
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not str(config.get("normal_label", "")).strip():
        raise ValueError("Detection configuration must define normal_label")
    data, features, assignments, feature_names, families = _load_dependence_inputs(
        data_file, feature_file, feature_manifest_file, feature_lineage_file, splits_file
    )
    assignments = assignments.loc[assignments["design"].isin(MODELING_DESIGN_NAMES)].copy()
    labels = (
        ~data["label"].astype("string").str.strip().eq(str(config["normal_label"]).strip())
    ).astype(int)
    numeric = numeric_frame(features, feature_names)

    rows: list[dict[str, Any]] = []
    for (design, fold), assignment in assignments.groupby(["design", "fold"], sort=True):
        train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
        test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
        y_train = labels.iloc[train_ids]
        y_test = labels.iloc[test_ids]
        for feature in feature_names:
            train_values = numeric.iloc[train_ids][feature]
            test_values = numeric.iloc[test_ids][feature]
            median = train_values.median()
            if pd.isna(median):
                status = "no_training_values"
                auc = float("nan")
                test_filled = test_values
            else:
                status = "ok"
                test_filled = test_values.fillna(median)
                auc = absolute_roc_auc(y_test, test_filled)
                if np.isnan(auc):
                    status = "single_test_class_or_constant"
            rows.append(
                {
                    "design": design,
                    "fold": fold,
                    "feature": feature,
                    "family": families[feature],
                    "absolute_auc": auc,
                    "status": status,
                    "train_rows": len(train_ids),
                    "test_rows": len(test_ids),
                    "train_missing_fraction": float(train_values.isna().mean()),
                    "test_missing_fraction": float(test_values.isna().mean()),
                    "training_median": median,
                    "normal_train_rows": int((y_train == 0).sum()),
                    "fault_train_rows": int((y_train == 1).sum()),
                    "normal_test_rows": int((y_test == 0).sum()),
                    "fault_test_rows": int((y_test == 1).sum()),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    by_fold = pd.DataFrame(rows)
    by_fold_path = output_dir / "univariate_feature_auc_by_fold.csv"
    by_fold.to_csv(by_fold_path, index=False)
    summary = (
        by_fold.groupby(["feature", "family"], as_index=False)
        .agg(
            folds=("design", "size"),
            valid_folds=("absolute_auc", "count"),
            mean_absolute_auc=("absolute_auc", "mean"),
            median_absolute_auc=("absolute_auc", "median"),
            max_absolute_auc=("absolute_auc", "max"),
        )
        .sort_values(["max_absolute_auc", "mean_absolute_auc"], ascending=False)
    )
    summary_path = output_dir / "univariate_feature_auc_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "feature_file": str(feature_file.expanduser().resolve()),
            "feature_manifest_file": str(feature_manifest_file.expanduser().resolve()),
            "feature_lineage_file": str(feature_lineage_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_file.expanduser().resolve()),
            "normal_label": str(config["normal_label"]).strip(),
            "feature_variant": "physical_signal",
            "modeling_designs": list(MODELING_DESIGN_NAMES),
            "imputation": "training-fold median only; no test values used to fit medians",
            "auc_definition": "max(ROC-AUC, 1 - ROC-AUC), so either direction is counted",
            "output_files": {"by_fold": str(by_fold_path), "summary": str(summary_path)},
        }
    )
    manifest_path = output_dir / "univariate_feature_auc_manifest.json"
    write_json(manifest_path, manifest)
    lines = [
        "# Single-feature held-out dependence audit",
        "",
        "Each feature was evaluated by itself on held-out rows. Missing values were filled with the training-fold median only.",
        "A direction-free AUC close to 0.5 means little independent separation; a value close to 1 means a strong one-feature shortcut.",
        "",
        "| Feature | Family | Valid folds | Mean absolute AUC | Max absolute AUC |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary.head(20).itertuples(index=False):
        mean_auc = "n/a" if pd.isna(row.mean_absolute_auc) else f"{row.mean_absolute_auc:.3f}"
        max_auc = "n/a" if pd.isna(row.max_absolute_auc) else f"{row.max_absolute_auc:.3f}"
        lines.append(
            f"| {row.feature} | {row.family} | {row.valid_folds}/{row.folds} | {mean_auc} | {max_auc} |"
        )
    (output_dir / "univariate_feature_auc_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"by_fold": str(by_fold_path), "summary": str(summary_path), "manifest": str(manifest_path)}


def dependence_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit one-feature held-out dependence")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--feature-lineage", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    args = parser.parse_args(argv)
    run_feature_dependence(
        args.data_file,
        args.feature_file,
        args.feature_manifest,
        args.feature_lineage,
        args.splits_file,
        args.output_dir,
        args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
