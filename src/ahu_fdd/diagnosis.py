"""Leakage-safe multiclass AHU fault diagnosis experiments."""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from xgboost import XGBClassifier

from ahu_fdd.baselines import _validate_assignments
from ahu_fdd.detection import (
    FoldTransformer,
    build_detection_features,
    load_sensor_config,
)

DESIGN_NAMES = ("ahu_held_out", "chronological", "building_held_out", "nested_grouped")
TASK_NAMES = ("flat_type", "hierarchical_group", "hierarchical_type")


def load_diagnosis_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The diagnosis configuration must be a mapping")
    if not str(config.get("normal_label", "")).strip():
        raise ValueError("The diagnosis configuration must define normal_label")
    threshold = float(config.get("confidence_threshold", 0))
    if not 0 < threshold < 1:
        raise ValueError("confidence_threshold must be between zero and one")
    groups = config.get("fault_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("The diagnosis configuration must define fault_groups")
    labels: list[str] = []
    for group, members in groups.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"Fault group {group!r} must contain labels")
        labels.extend(str(member) for member in members)
    if len(labels) != len(set(labels)):
        raise ValueError("A fault label cannot occur in more than one group")
    variants = config.get("feature_variants", ["physical_signal"])
    if not isinstance(variants, list) or not variants:
        raise ValueError("feature_variants must be a non-empty list")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("The diagnosis configuration must define model settings")
    config["confidence_threshold"] = threshold
    config["fault_groups"] = {str(key): [str(value) for value in values] for key, values in groups.items()}
    config["feature_variants"] = [str(value) for value in variants]
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


def _fit_multiclass(
    train_matrix: np.ndarray,
    train_labels: pd.Series,
    test_matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a weighted XGBoost classifier and return labels, confidence, probabilities."""
    classes = np.asarray(sorted(train_labels.astype(str).unique()), dtype=object)
    if not len(classes):
        return classes, np.empty(len(test_matrix), dtype=object), np.empty((len(test_matrix), 0))
    encoded = pd.Categorical(train_labels.astype(str), categories=classes).codes
    if len(classes) == 1 or train_matrix.shape[1] == 0:
        probabilities = np.ones((len(test_matrix), len(classes)), dtype=float)
        return (
            classes,
            np.full(len(test_matrix), classes[0], dtype=object),
            probabilities.max(axis=1),
        )
    counts = np.bincount(encoded, minlength=len(classes)).astype(float)
    weights = len(encoded) / (len(classes) * counts[encoded])
    model_config = dict(config["model"])
    model_config.update({"objective": "multi:softprob", "num_class": len(classes)})
    model = XGBClassifier(**model_config)
    model.fit(train_matrix, encoded, sample_weight=weights)
    probabilities = np.asarray(model.predict_proba(test_matrix), dtype=float)
    indices = probabilities.argmax(axis=1)
    return classes, classes[indices], probabilities.max(axis=1)


def _metric_values(true: pd.Series, predicted: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    if not len(true):
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(
            recall_score(
                true,
                predicted,
                labels=np.unique(true.astype(str)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(true, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true, predicted, labels=classes, average="weighted", zero_division=0)
        ),
    }


def _record_metrics(
    summary: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    building_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    *,
    design: str,
    fold: str,
    variant: str,
    task: str,
    true_labels: pd.Series,
    predicted: np.ndarray,
    confidence: np.ndarray,
    classes: np.ndarray,
    all_test_rows: int,
    unsupported_rows: int,
    threshold: float,
    buildings: pd.Series,
) -> None:
    known = true_labels.astype(str).isin(classes)
    true = true_labels.astype(str).loc[known]
    pred = np.asarray(predicted, dtype=object)[known.to_numpy()]
    conf = np.asarray(confidence, dtype=float)[known.to_numpy()]
    confident = conf >= threshold
    metrics = _metric_values(true, pred, classes)
    confident_metrics = _metric_values(true.loc[confident], pred[confident], classes)
    summary.append(
        {
            "design": design,
            "fold": fold,
            "variant": variant,
            "task": task,
            "evaluated_rows": int(len(true)),
            "unsupported_rows": int(unsupported_rows),
            "all_test_rows": int(all_test_rows),
            "class_count": int(len(classes)),
            "confidence_threshold": threshold,
            "coverage": float(confident.mean()) if len(confident) else np.nan,
            "abstention_rate": float((~confident).mean()) if len(confident) else np.nan,
            "confident_accuracy": confident_metrics["accuracy"],
            **metrics,
        }
    )
    if not len(true):
        return
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=classes, zero_division=0
    )
    for label, p_value, r_value, f_value, support_value in zip(
        classes, precision, recall, f1, support
    ):
        class_rows.append(
            {
                "design": design,
                "fold": fold,
                "variant": variant,
                "task": task,
                "class_label": str(label),
                "precision": float(p_value),
                "recall": float(r_value),
                "f1": float(f_value),
                "support": int(support_value),
            }
        )
    matrix = confusion_matrix(true, pred, labels=classes)
    for row_index, actual in enumerate(classes):
        for column_index, predicted_label in enumerate(classes):
            confusion_rows.append(
                {
                    "design": design,
                    "fold": fold,
                    "variant": variant,
                    "task": task,
                    "true_label": str(actual),
                    "predicted_label": str(predicted_label),
                    "count": int(matrix[row_index, column_index]),
                }
            )
    known_buildings = buildings.loc[known].reset_index(drop=True)
    true_reset = true.reset_index(drop=True)
    pred_reset = pd.Series(pred).reset_index(drop=True)
    for building, indices in known_buildings.groupby(known_buildings).groups.items():
        building_true = true_reset.iloc[list(indices)]
        building_pred = pred_reset.iloc[list(indices)].to_numpy()
        building_rows.append(
            {
                "design": design,
                "fold": fold,
                "variant": variant,
                "task": task,
                "building": str(building),
                "support": int(len(indices)),
                **_metric_values(building_true, building_pred, classes),
            }
        )


def _requested_designs(requested: str | None) -> list[str]:
    designs = list(DESIGN_NAMES) if requested is None else [item.strip() for item in requested.split(",")]
    unknown = set(designs) - set(DESIGN_NAMES)
    if unknown:
        raise ValueError(f"Unknown diagnosis designs: {sorted(unknown)}")
    return designs


def run_diagnosis(
    data_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_path: Path,
    building_config_path: Path,
    requested_designs: str | None = None,
) -> dict[str, Any]:
    config = load_diagnosis_config(config_path)
    designs = _requested_designs(requested_designs)
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    assignments = pd.read_csv(splits_file, low_memory=False)
    _validate_assignments(assignments, len(data))
    sensors = load_sensor_config(building_config_path)
    features, variants = build_detection_features(data, sensors)
    for variant in config["feature_variants"]:
        if variant not in variants:
            raise ValueError(f"Unknown feature variant in diagnosis config: {variant}")

    labels = data["label"].astype("string").str.strip()
    normal_label = str(config["normal_label"]).strip()
    if labels.isna().any():
        raise ValueError("Diagnosis input contains missing labels")
    label_to_group = {
        label: group
        for group, members in config["fault_groups"].items()
        for label in members
    }
    fault_mask = ~labels.eq(normal_label)
    unmapped = sorted(set(labels[fault_mask].astype(str)) - set(label_to_group))
    if unmapped:
        raise ValueError(f"Fault labels are not mapped to a diagnosis group: {unmapped}")
    groups = labels.map(label_to_group).astype("string")

    summary: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    building_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    threshold = float(config["confidence_threshold"])
    for variant in config["feature_variants"]:
        for (design, fold), assignment in assignments.groupby(["design", "fold"], sort=True):
            if design not in designs:
                continue
            train_ids = assignment.loc[assignment["split"].eq("train"), "row_id"].astype(int)
            test_ids = assignment.loc[assignment["split"].eq("test"), "row_id"].astype(int)
            train_fault_ids = train_ids[fault_mask.iloc[train_ids].to_numpy()]
            test_fault_ids = test_ids[fault_mask.iloc[test_ids].to_numpy()]
            train_all = features.iloc[train_ids]
            train_fault = features.iloc[train_fault_ids]
            test_fault = features.iloc[test_fault_ids]
            transformer = FoldTransformer(variants[variant]).fit(train_all)
            train_matrix = transformer.transform(train_fault)
            test_matrix = transformer.transform(test_fault)
            train_types = labels.iloc[train_fault_ids]
            test_types = labels.iloc[test_fault_ids]
            test_groups = groups.iloc[test_fault_ids]
            test_buildings = data.iloc[test_fault_ids]["building"].astype("string").reset_index(drop=True)

            type_classes, flat_pred, flat_conf = _fit_multiclass(
                train_matrix, train_types, test_matrix, config
            )
            _record_metrics(
                summary,
                class_rows,
                building_rows,
                confusion_rows,
                design=design,
                fold=fold,
                variant=variant,
                task="flat_type",
                true_labels=test_types.reset_index(drop=True),
                predicted=flat_pred,
                confidence=flat_conf,
                classes=type_classes,
                all_test_rows=len(test_fault_ids),
                unsupported_rows=int((~test_types.astype(str).isin(type_classes)).sum()),
                threshold=threshold,
                buildings=test_buildings,
            )
            for index, row_id in enumerate(test_fault_ids):
                prediction_rows.append(
                    {
                        "row_id": int(row_id),
                        "building": str(data.iloc[row_id]["building"]),
                        "ahu": str(data.iloc[row_id]["ahu"]),
                        "timestamp": str(data.iloc[row_id]["timestamp"]),
                        "design": design,
                        "fold": fold,
                        "variant": variant,
                        "task": "flat_type",
                        "true_group": str(test_groups.iloc[index]),
                        "true_type": str(test_types.iloc[index]),
                        "predicted_group": label_to_group.get(str(flat_pred[index]), "Uncertain"),
                        "predicted_type": str(flat_pred[index]),
                        "group_confidence": np.nan,
                        "type_confidence": float(flat_conf[index]),
                        "group_abstained": False,
                        "type_abstained": bool(flat_conf[index] < threshold),
                    }
                )

            group_train = groups.iloc[train_fault_ids]
            group_classes, group_pred, group_conf = _fit_multiclass(
                train_matrix, group_train, test_matrix, config
            )
            _record_metrics(
                summary,
                class_rows,
                building_rows,
                confusion_rows,
                design=design,
                fold=fold,
                variant=variant,
                task="hierarchical_group",
                true_labels=test_groups.reset_index(drop=True),
                predicted=group_pred,
                confidence=group_conf,
                classes=group_classes,
                all_test_rows=len(test_fault_ids),
                unsupported_rows=int((~test_groups.astype(str).isin(group_classes)).sum()),
                threshold=threshold,
                buildings=test_buildings,
            )

            type_pred = np.full(len(test_fault_ids), "Uncertain", dtype=object)
            type_conf = np.zeros(len(test_fault_ids), dtype=float)
            for group in group_classes:
                train_group_mask = group_train.astype(str).eq(str(group)).to_numpy()
                test_group_mask = np.asarray(group_pred, dtype=object) == group
                _, predicted_types, confidence = _fit_multiclass(
                    train_matrix[train_group_mask],
                    train_types.iloc[train_group_mask],
                    test_matrix[test_group_mask],
                    config,
                )
                type_pred[test_group_mask] = predicted_types
                type_conf[test_group_mask] = group_conf[test_group_mask] * confidence
            _record_metrics(
                summary,
                class_rows,
                building_rows,
                confusion_rows,
                design=design,
                fold=fold,
                variant=variant,
                task="hierarchical_type",
                true_labels=test_types.reset_index(drop=True),
                predicted=type_pred,
                confidence=type_conf,
                classes=type_classes,
                all_test_rows=len(test_fault_ids),
                unsupported_rows=int((~test_types.astype(str).isin(type_classes)).sum()),
                threshold=threshold,
                buildings=test_buildings,
            )
            for index, row_id in enumerate(test_fault_ids):
                group_is_confident = group_conf[index] >= threshold
                type_is_confident = group_is_confident and type_conf[index] >= threshold
                prediction_rows.append(
                    {
                        "row_id": int(row_id),
                        "building": str(data.iloc[row_id]["building"]),
                        "ahu": str(data.iloc[row_id]["ahu"]),
                        "timestamp": str(data.iloc[row_id]["timestamp"]),
                        "design": design,
                        "fold": fold,
                        "variant": variant,
                        "task": "hierarchical",
                        "true_group": str(test_groups.iloc[index]),
                        "true_type": str(test_types.iloc[index]),
                        "predicted_group": str(group_pred[index]),
                        "predicted_type": str(type_pred[index]),
                        "group_confidence": float(group_conf[index]),
                        "type_confidence": float(type_conf[index]),
                        "group_abstained": not group_is_confident,
                        "type_abstained": not type_is_confident,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary)
    summary_path = output_dir / "diagnosis_metrics_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    class_path = output_dir / "diagnosis_metrics_by_class.csv"
    pd.DataFrame(class_rows).to_csv(class_path, index=False)
    building_path = output_dir / "diagnosis_metrics_by_building.csv"
    pd.DataFrame(building_rows).to_csv(building_path, index=False)
    confusion_path = output_dir / "diagnosis_confusion_matrix.csv"
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)
    predictions_path = output_dir / "diagnosis_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_file": str(data_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_path.expanduser().resolve()),
            "building_config_file": str(building_config_path.expanduser().resolve()),
            "row_count": int(len(data)),
            "fault_row_count": int(fault_mask.sum()),
            "normal_label": normal_label,
            "fault_groups": config["fault_groups"],
            "confidence_threshold": threshold,
            "designs": designs,
            "feature_variants": config["feature_variants"],
            "output_files": {
                "summary": str(summary_path),
                "by_class": str(class_path),
                "by_building": str(building_path),
                "confusion_matrix": str(confusion_path),
                "predictions": str(predictions_path),
            },
        }
    )
    manifest_path = output_dir / "diagnosis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "diagnosis_summary.md"
    display_frame = summary_frame.groupby(["design", "variant", "task"], as_index=False).agg(
        folds=("fold", "nunique"), evaluated_rows=("evaluated_rows", "sum"),
        unsupported_rows=("unsupported_rows", "sum"), macro_f1=("macro_f1", "mean"),
        coverage=("coverage", "mean"), confident_accuracy=("confident_accuracy", "mean"),
    )
    lines = [
        "# Diagnosis experiment summary",
        "",
        "Diagnosis is evaluated on fault-labelled rows only; normal-versus-fault detection is a separate task.",
        "Confidence is the maximum XGBoost class score, with abstention below the configured threshold.",
        "",
        "| Design | Variant | Task | Folds | Evaluated rows | Unsupported rows | Macro F1 | Coverage | Confident accuracy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display_frame.itertuples(index=False):
        lines.append(
            f"| {row.design} | {row.variant} | {row.task} | {row.folds} | {row.evaluated_rows:,} | "
            f"{row.unsupported_rows:,} | {row.macro_f1:.3f} | {row.coverage:.3f} | {row.confident_accuracy:.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary": str(summary_path),
        "by_class": str(class_path),
        "by_building": str(building_path),
        "confusion_matrix": str(confusion_path),
        "predictions": str(predictions_path),
        "manifest": str(manifest_path),
        "markdown": str(markdown_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run leakage-safe AHU fault diagnosis experiments")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnosis"))
    parser.add_argument("--config", type=Path, default=Path("configs/diagnosis.yaml"))
    parser.add_argument("--building-config", type=Path, default=Path("configs/multibuilding_ahu.yaml"))
    parser.add_argument("--designs", help="Comma-separated subset of split designs; default is all designs")
    args = parser.parse_args(argv)
    result = run_diagnosis(
        args.data_file,
        args.splits_file,
        args.output_dir,
        args.config,
        args.building_config,
        args.designs,
    )
    print(f"Wrote diagnosis outputs to {args.output_dir.resolve()}")
    print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
