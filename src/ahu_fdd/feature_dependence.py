"""Held-out univariate feature-dependence audit for pooled AHU detection."""

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

from ahu_fdd.baselines import _validate_assignments
from ahu_fdd.splits import MODELING_DESIGN_NAMES


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


def absolute_roc_auc(y_true: pd.Series, values: pd.Series) -> float:
    """Return the direction-free AUC of one feature, or NaN if undefined."""
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = finite.notna() & y_true.notna()
    labels = y_true.loc[usable].astype(int)
    scores = finite.loc[usable]
    if labels.nunique() < 2 or scores.nunique(dropna=True) < 2:
        return float("nan")
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def _load_inputs(
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
    _validate_assignments(assignments, len(data))
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
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not str(config.get("normal_label", "")).strip():
        raise ValueError("Detection configuration must define normal_label")
    data, features, assignments, feature_names, families = _load_inputs(
        data_file, feature_file, feature_manifest_file, feature_lineage_file, splits_file
    )
    assignments = assignments.loc[assignments["design"].isin(MODELING_DESIGN_NAMES)].copy()
    labels = (~data["label"].astype("string").str.strip().eq(str(config["normal_label"]).strip())).astype(int)
    numeric = features[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

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
    manifest = _json_safe(
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
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        lines.append(f"| {row.feature} | {row.family} | {row.valid_folds}/{row.folds} | {mean_auc} | {max_auc} |")
    (output_dir / "univariate_feature_auc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"by_fold": str(by_fold_path), "summary": str(summary_path), "manifest": str(manifest_path)}


def main(argv: Iterable[str] | None = None) -> int:
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
