"""Simple binary detection baselines evaluated on frozen split assignments."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_NAMES = (
    "majority_class",
    "normal_envelope_rule",
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
    rule = config.get("normal_envelope_rule", {})
    for key in ("lower_quantile", "upper_quantile"):
        value = float(rule.get(key, -1))
        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between zero and one")
    if float(rule["lower_quantile"]) >= float(rule["upper_quantile"]):
        raise ValueError("lower_quantile must be below upper_quantile")
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


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _usable_columns(train: pd.DataFrame, columns: list[str]) -> list[str]:
    numeric = _numeric(train, columns)
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
    train_matrix = _numeric(train, usable)
    test_matrix = _numeric(test, usable)
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


def _normal_envelope_prediction(
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
    lower_q = float(config["normal_envelope_rule"]["lower_quantile"])
    upper_q = float(config["normal_envelope_rule"]["upper_quantile"])
    normal_values = _numeric(normal, usable)
    lower = normal_values.quantile(lower_q)
    upper = normal_values.quantile(upper_q)
    test_values = _numeric(test, usable)
    outside = test_values.lt(lower, axis="columns") | test_values.gt(upper, axis="columns")
    return outside.any(axis=1).astype(int).to_numpy()


def _metric_row(
    design: str,
    fold: str,
    baseline: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    true = y_true.to_numpy(dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    true_negative = int(((true == 0) & (predicted == 0)).sum())
    false_positive = int(((true == 0) & (predicted == 1)).sum())
    false_negative = int(((true == 1) & (predicted == 0)).sum())
    true_positive = int(((true == 1) & (predicted == 1)).sum())
    return {
        "design": design,
        "fold": fold,
        "baseline": baseline,
        "rows": int(len(true)),
        "normal_support": int((true == 0).sum()),
        "fault_support": int((true == 1).sum()),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": (
            float(
                0.5
                * (
                    true_negative / (true_negative + false_positive)
                    + true_positive / (true_positive + false_negative)
                )
            )
            if (true_negative + false_positive) and (true_positive + false_negative)
            else np.nan
        ),
        "precision": float(
            precision_score(true, predicted, labels=[0, 1], zero_division=0)
        ),
        "recall": float(recall_score(true, predicted, labels=[0, 1], zero_division=0)),
        "f1": float(f1_score(true, predicted, labels=[0, 1], zero_division=0)),
        "macro_f1": float(
            f1_score(true, predicted, labels=[0, 1], average="macro", zero_division=0)
        ),
        "false_positive_rate": float(
            false_positive / (false_positive + true_negative)
            if true_negative + false_positive
            else 0.0
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def _validate_assignments(assignments: pd.DataFrame, row_count: int) -> None:
    required = {"row_id", "design", "fold", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")
    if assignments["row_id"].isna().any() or assignments["row_id"].duplicated().any():
        # Duplicate row IDs are valid across designs/folds, but not within one fold.
        duplicate_groups = assignments.duplicated(["design", "fold", "row_id"]).any()
        if duplicate_groups:
            raise ValueError("Split assignments duplicate a row within a design/fold")
    if assignments["row_id"].astype(int).lt(0).any() or assignments["row_id"].astype(int).ge(row_count).any():
        raise ValueError("Split assignments contain row IDs outside the input table")
    for (design, fold), group in assignments.groupby(["design", "fold"]):
        if set(group["split"]) != {"train", "validation", "test"}:
            raise ValueError(f"{design}/{fold} must contain train, validation and test rows")


def run_baselines(
    data_file: Path, splits_file: Path, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    config = load_baseline_config(config_path)
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    assignments = pd.read_csv(splits_file, low_memory=False)
    _validate_assignments(assignments, len(data))
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
    for (design, fold), assignment in assignments.groupby(["design", "fold"], sort=True):
        row_ids = assignment["row_id"].astype(int)
        train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
        test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
        train = data.iloc[train_ids]
        test = data.iloc[test_ids]
        y_train = y.iloc[train_ids]
        y_test = y.iloc[test_ids]
        predictions = {
            "majority_class": _majority_prediction(y_train, len(test)),
            "normal_envelope_rule": _normal_envelope_prediction(
                train, test, y_train, all_sensors, config
            ),
            "actuator_pump_only": _fit_logistic(
                train, test, y_train, list(groups["actuator_pump"])
            ),
        }
        if len(row_ids) != len(assignment):
            raise ValueError("Unexpected assignment row count")
        for baseline, predicted in predictions.items():
            metrics.append(_metric_row(design, fold, baseline, y_test, predicted))

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
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_path.expanduser().resolve()),
            "row_count": int(len(data)),
            "normal_label": normal_label,
            "baseline_names": list(BASELINE_NAMES),
            "output_files": {"by_fold": str(fold_path), "summary": str(summary_path)},
        }
    )
    manifest_path = output_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
