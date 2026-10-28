"""Matched pooled-versus-individual XGBoost building comparisons."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahu_fdd.baselines import _metric_row, _validate_assignments
from ahu_fdd.building_detection import _load_inputs
from ahu_fdd.detection import FoldTransformer, _xgboost, load_detection_config

MATCHED_DESIGNS = ("building_chronological", "building_ahu_held_out")
STRATEGIES = ("individual_xgboost", "pooled_adapted_xgboost", "pooled_transfer_xgboost")


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


def _fit_predict(
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    transformer = FoldTransformer(columns).fit(train_features)
    return _xgboost(
        transformer.transform(train_features),
        transformer.transform(eval_features),
        y_train,
        config,
    )


def _metric(
    building: str,
    design: str,
    fold: str,
    strategy: str,
    stage: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    train_rows: int,
    train_buildings: str,
) -> dict[str, Any]:
    row = _metric_row(design, fold, strategy, y_true, predictions)
    row["model"] = row.pop("baseline")
    row.update(
        {
            "building": building,
            "stage": stage,
            "train_rows": int(train_rows),
            "train_buildings": train_buildings,
        }
    )
    return row


def run_matched_comparison(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_file: Path,
) -> dict[str, Any]:
    """Compare local, pooled-adapted and zero-shot models on identical test rows."""
    config = load_detection_config(config_file)
    data, features, assignments, physical = _load_inputs(
        data_file, feature_file, feature_manifest_file, splits_file
    )
    _validate_assignments(assignments, len(data))
    assignments = assignments.loc[assignments["design"].isin(MATCHED_DESIGNS)].copy()
    labels = data["label"].astype("string").str.strip()
    normal_label = str(config["normal_label"]).strip()
    if labels.isna().any():
        raise ValueError("Matched comparison input contains missing labels")
    y = (~labels.eq(normal_label)).astype(int)

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    predictions_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    buildings = sorted(assignments["building_scope"].astype(str).unique())

    for (building, design, fold), target in assignments.groupby(
        ["building_scope", "design", "fold"], sort=True
    ):
        target = target.copy()
        target_ids = {
            split: target.loc[target["split"].eq(split), "row_id"].astype(int).tolist()
            for split in ("train", "validation", "test")
        }
        if not set(data.iloc[target_ids["train"]]["building"].unique()) <= {building}:
            raise ValueError("Target training rows contain another building")
        if not set(data.iloc[target_ids["test"]]["building"].unique()) <= {building}:
            raise ValueError("Target test rows contain another building")

        source = assignments.loc[
            assignments["design"].eq(design)
            & assignments["fold"].eq(fold)
            & assignments["building_scope"].ne(building)
            & assignments["split"].eq("train")
        ]
        source_ids = source["row_id"].astype(int).tolist()
        local_train_ids = target_ids["train"]
        local_final_ids = local_train_ids + target_ids["validation"]
        pooled_adapted_final_ids = local_final_ids + source_ids

        test_id_set = set(target_ids["test"])
        strategy_test_ids = {
            strategy: tuple(target_ids["test"]) for strategy in STRATEGIES
        }
        if len(set(strategy_test_ids.values())) != 1:
            raise ValueError(
                f"Matched strategies do not share test rows for {building}/{design}/{fold}"
            )
        for strategy in STRATEGIES:
            checks.append(
                {
                    "building": building,
                    "design": design,
                    "fold": fold,
                    "strategy": strategy,
                    "test_rows": len(test_id_set),
                    "test_row_ids_match": True,
                }
            )

        training_sets = {
            "individual_xgboost": local_train_ids,
            "pooled_adapted_xgboost": local_train_ids + source_ids,
            "pooled_transfer_xgboost": source_ids,
        }
        final_training_sets = {
            "individual_xgboost": local_final_ids,
            "pooled_adapted_xgboost": pooled_adapted_final_ids,
            "pooled_transfer_xgboost": source_ids,
        }
        train_buildings = {
            "individual_xgboost": str(building),
            "pooled_adapted_xgboost": "all_buildings",
            "pooled_transfer_xgboost": "other_buildings",
        }

        eval_frames = {
            "validation": (target_ids["validation"], "validation"),
            "test": (target_ids["test"], "test"),
        }
        for strategy in STRATEGIES:
            for ids, stage in eval_frames.values():
                fit_ids = training_sets[strategy] if stage == "validation" else final_training_sets[strategy]
                predictions = _fit_predict(
                    features.iloc[fit_ids],
                    features.iloc[ids],
                    y.iloc[fit_ids],
                    physical,
                    config,
                )
                metric = _metric(
                    str(building),
                    str(design),
                    str(fold),
                    strategy,
                    stage,
                    y.iloc[ids],
                    predictions,
                    len(fit_ids),
                    train_buildings[strategy],
                )
                (validation_rows if stage == "validation" else test_rows).append(metric)
                if stage == "test":
                    for row_id, true, predicted in zip(
                        ids, y.iloc[ids].to_numpy(), predictions, strict=True
                    ):
                        predictions_rows.append(
                            {
                                "row_id": int(row_id),
                                "building": str(building),
                                "design": str(design),
                                "fold": str(fold),
                                "strategy": strategy,
                                "y_true": int(true),
                                "y_pred": int(predicted),
                            }
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "matched_building_validation_metrics.csv"
    test_path = output_dir / "matched_building_test_metrics.csv"
    predictions_path = output_dir / "matched_building_test_predictions.csv"
    checks_path = output_dir / "matched_building_test_row_checks.csv"
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    pd.DataFrame(predictions_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(checks).to_csv(checks_path, index=False)

    test_frame = pd.DataFrame(test_rows)
    summary = (
        test_frame.groupby(["building", "design", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            test_rows=("rows", "sum"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_recall=("recall", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
        )
    )
    summary_path = output_dir / "matched_building_test_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "purpose": "Fair comparison of independent and pooled XGBoost models on identical building-specific test rows",
            "data_file": str(data_file.expanduser().resolve()),
            "feature_file": str(feature_file.expanduser().resolve()),
            "feature_manifest_file": str(feature_manifest_file.expanduser().resolve()),
            "splits_file": str(splits_file.expanduser().resolve()),
            "config_file": str(config_file.expanduser().resolve()),
            "row_count": int(len(data)),
            "buildings": buildings,
            "designs": list(MATCHED_DESIGNS),
            "strategies": list(STRATEGIES),
            "feature_set": "project_default_modelling_features",
            "protocol": {
                "individual": "target-building training rows only; refit on target train plus validation before test",
                "pooled_adapted": "target-building training rows plus other-building training rows; refit with target validation before test",
                "pooled_transfer": "other-building training rows only; target building is unseen during training",
                "test_rows": "identical target-building test row IDs for all three strategies",
            },
            "output_files": {
                "validation_metrics": str(validation_path),
                "test_metrics": str(test_path),
                "test_summary": str(summary_path),
                "test_predictions": str(predictions_path),
                "test_row_checks": str(checks_path),
            },
        }
    )
    manifest_path = output_dir / "matched_building_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Matched pooled-versus-individual comparison",
        "",
        "All strategies are evaluated on the same target-building test rows.",
        "Individual models use target-building data only; pooled-adapted models add other buildings' training rows; pooled-transfer models do not see the target building during training.",
        "",
        "| Building | Design | Strategy | Test F1 | Balanced accuracy | Recall | False-positive rate | Test rows |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_frame.sort_values(["building", "design", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.building} | {row.design} | {row.model} | {row.f1:.3f} | "
            f"{row.balanced_accuracy:.3f} | {row.recall:.3f} | {row.false_positive_rate:.3f} | {row.rows:,} |"
        )
    (output_dir / "matched_building_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "validation_metrics": str(validation_path),
        "test_metrics": str(test_path),
        "summary": str(summary_path),
        "predictions": str(predictions_path),
        "row_checks": str(checks_path),
        "manifest": str(manifest_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare individual and pooled XGBoost models on identical building test rows"
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    args = parser.parse_args(argv)
    result = run_matched_comparison(
        args.data_file,
        args.feature_file,
        args.feature_manifest,
        args.splits_file,
        args.output_dir,
        args.config,
    )
    print(f"Created matched comparison results in {args.output_dir.resolve()}")
    print(f"Test metrics: {result['test_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
