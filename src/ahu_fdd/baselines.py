"""Simple binary detection baselines evaluated on frozen split assignments."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ahu_fdd.metrics import binary_metric_row
from ahu_fdd.splits import (
    DISABLED_MODELING_DESIGN_NAMES,
    MODELING_DESIGN_NAMES,
)
from ahu_fdd.utils import json_safe, numeric_frame, write_json
from ahu_fdd.validation import validate_assignments

BASELINE_NAMES = (
    "majority_class",
    "normal_range_rule",
    "actuator_pump_only",
)


def load_baseline_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The baseline configuration must be a mapping")
    if str(config.get("normal_label", "")).strip() == "":
        raise ValueError("The baseline configuration must define normal_label")
    groups = config.get("feature_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("The baseline configuration must define feature_groups")
    rule = config.get("normal_range_rule", {})
    for key in ("lower_quantile", "upper_quantile"):
        value = float(rule.get(key, -1))
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between zero and one")
    if float(rule["lower_quantile"]) >= float(rule["upper_quantile"]):
        raise ValueError("lower_quantile must be below upper_quantile")
    return config


def _usable_columns(train: pd.DataFrame, columns: list[str]) -> list[str]:
    numeric = numeric_frame(train, columns)
    return [
        column
        for column in columns
        if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1
    ]


def _majority_prediction(y_train: pd.Series, count: int) -> np.ndarray:
    majority = int(y_train.value_counts().sort_index().idxmax())
    return np.full(count, majority, dtype=int)


def _fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
) -> np.ndarray:
    usable = _usable_columns(train, columns)
    if not usable:
        return _majority_prediction(y_train, len(test))
    train_matrix = numeric_frame(train, usable)
    test_matrix = numeric_frame(test, usable)
    usable = [column for column in train_matrix if train_matrix[column].nunique() > 1]
    if not usable or y_train.nunique() < 2:
        return _majority_prediction(y_train, len(test))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    model.fit(train_matrix[usable], y_train)
    return model.predict(test_matrix[usable]).astype(int)


def normal_range_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    normal = train.loc[y_train.eq(0)]
    usable = _usable_columns(normal, columns)
    if normal.empty or not usable:
        return _majority_prediction(y_train, len(test))
    lower_q = float(config["lower_quantile"])
    upper_q = float(config["upper_quantile"])
    normal_values = numeric_frame(normal, usable)
    lower = normal_values.quantile(lower_q)
    upper = normal_values.quantile(upper_q)
    test_values = numeric_frame(test, usable)
    outside = test_values.lt(lower, axis="columns") | test_values.gt(upper, axis="columns")
    return outside.any(axis=1).astype(int).to_numpy()


# Compatibility aliases for existing tests and downstream scripts. New code
# imports the explicit public names from ahu_fdd.metrics and ahu_fdd.validation.
_metric_row = binary_metric_row
_validate_assignments = validate_assignments
_normal_range_prediction = normal_range_prediction


def run_baselines(
    data_file: Path, splits_file: Path, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    config = load_baseline_config(config_path)
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    assignments = pd.read_csv(splits_file, low_memory=False)
    validate_assignments(assignments, len(data))
    modeling_assignments = assignments.loc[
        assignments["design"].isin(MODELING_DESIGN_NAMES)
    ].copy()
    if "label" not in data:
        raise ValueError("Baseline input must contain the canonical label column")
    normal_label = str(config["normal_label"]).strip()
    labels = data["label"].astype("string").str.strip()
    if labels.isna().any():
        raise ValueError("Baseline input contains missing labels")
    y = (~labels.eq(normal_label)).astype(int)
    groups = config["feature_groups"]
    all_sensors = sorted({column for columns in groups.values() for column in columns})
    missing_sensors = set(all_sensors) - set(data.columns)
    if missing_sensors:
        raise ValueError(f"Baseline input is missing configured sensors: {sorted(missing_sensors)}")

    metrics: list[dict[str, Any]] = []
    for (design, fold), assignment in modeling_assignments.groupby(
        ["design", "fold"], sort=True
    ):
        row_ids = assignment["row_id"].astype(int)
        train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
        test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
        train = data.iloc[train_ids]
        test = data.iloc[test_ids]
        y_train = y.iloc[train_ids]
        y_test = y.iloc[test_ids]
        predictions = {
            "majority_class": _majority_prediction(y_train, len(test)),
            "normal_range_rule": normal_range_prediction(
                train, test, y_train, all_sensors, config["normal_range_rule"]
            ),
            "actuator_pump_only": _fit_logistic(
                train, test, y_train, list(groups["actuator_pump"])
            ),
        }
        if len(row_ids) != len(assignment):
            raise ValueError("Unexpected assignment row count")
        for baseline, predicted in predictions.items():
            metrics.append(binary_metric_row(design, fold, baseline, y_test, predicted))

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "baseline_metrics_by_fold.csv"
    fold_frame = pd.DataFrame(metrics)
    fold_frame.to_csv(fold_path, index=False)
    summary = (
        fold_frame.groupby(["design", "baseline"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            rows=("rows", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
        )
    )
    summary_path = output_dir / "baseline_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_path.expanduser().resolve()),
            "row_count": int(len(data)),
            "normal_label": normal_label,
            "baseline_names": list(BASELINE_NAMES),
            "modeling_designs": list(MODELING_DESIGN_NAMES),
            "disabled_modeling_designs": list(DISABLED_MODELING_DESIGN_NAMES),
            "output_files": {"by_fold": str(fold_path), "summary": str(summary_path)},
        }
    )
    manifest_path = output_dir / "baseline_manifest.json"
    write_json(manifest_path, manifest)
    markdown = output_dir / "baseline_summary.md"
    display = summary.sort_values(["design", "mean_f1"], ascending=[True, False])
    markdown_lines = [
        "# Binary baseline comparison",
        "",
        "These are fold-level averages. The statistical and ML models must use the same saved assignments.",
        "",
        "| Design | Baseline | Mean F1 | Mean balanced accuracy | Mean false-positive rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in display.itertuples(index=False):
        markdown_lines.append(
            f"| {row.design} | {row.baseline} | {row.mean_f1:.3f} | "
            f"{row.mean_balanced_accuracy:.3f} | {row.mean_false_positive_rate:.3f} |"
        )
    markdown.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {
        "row_count": int(len(data)),
        "baseline_names": list(BASELINE_NAMES),
        "fold_metrics": str(fold_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "markdown": str(markdown),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate simple AHU detection baselines")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    args = parser.parse_args(argv)
    result = run_baselines(args.data_file, args.splits_file, args.output_dir, args.config)
    print(f"Evaluated {len(result['baseline_names'])} baselines on {result['row_count']:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
