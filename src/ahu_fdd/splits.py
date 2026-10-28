"""Leakage-safe evaluation split generation for multibuilding AHU data."""

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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

REQUIRED_COLUMNS = {"building", "ahu", "timestamp"}
DESIGN_NAMES = (
    "ahu_held_out",
    "chronological",
    "building_held_out",
    "nested_grouped",
)
# ``ahu_held_out`` remains in the saved assignment table for traceability, but
# is deliberately disabled for the current pooled headline experiments.  The
# active designs below are the only designs that pooled runners may execute.
MODELING_DESIGN_NAMES = ("chronological", "building_held_out", "nested_grouped")
DISABLED_MODELING_DESIGN_NAMES = ("ahu_held_out",)
BUILDING_DESIGN_NAMES = ("building_ahu_held_out", "building_chronological")


def load_split_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The split configuration must be a mapping")
    if "random_state" not in config:
        raise ValueError("The split configuration must define random_state")
    for key in ("test_fraction", "validation_fraction", "inner_validation_fraction"):
        if key not in config:
            raise ValueError(f"The split configuration must define {key}")
        value = float(config[key])
        if not 0 < value < 1:
            raise ValueError(f"{key} must be between zero and one")
    if float(config["test_fraction"]) + float(config["validation_fraction"]) >= 1:
        raise ValueError("test_fraction plus validation_fraction must be below one")
    outer_folds = int(config.get("outer_folds", 5))
    if outer_folds < 2:
        raise ValueError("outer_folds must be at least two")
    config["outer_folds"] = outer_folds
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


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Split input is missing columns: {sorted(missing)}")
    result = frame.copy().reset_index(drop=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("Split input contains unparseable timestamps")
    if result[["building", "ahu"]].isna().any().any():
        raise ValueError("Split input contains missing building or AHU identifiers")
    result["_row_id"] = np.arange(len(result), dtype=np.int64)
    result["_group_key"] = (
        result["building"].astype("string") + "::" + result["ahu"].astype("string")
    )
    return result


def _assignments(
    frame: pd.DataFrame, design: str, fold: str, split_by_row: pd.Series
) -> pd.DataFrame:
    if split_by_row.isna().any():
        raise ValueError(f"{design}/{fold} left rows without a split assignment")
    return pd.DataFrame(
        {
            "row_id": frame["_row_id"],
            "building": frame["building"].astype("string"),
            "ahu": frame["ahu"].astype("string"),
            "timestamp": frame["timestamp"],
            "label": frame["label"] if "label" in frame else "",
            "group_key": frame["_group_key"].astype("string"),
            "design": design,
            "fold": fold,
            "split": split_by_row.astype("string"),
        }
    )


def _group_holdout(
    frame: pd.DataFrame, config: dict[str, Any], design: str
) -> pd.DataFrame:
    """Split AHU groups within each building, keeping every group in one split."""
    random_state = int(config["random_state"])
    test_fraction = float(config["test_fraction"])
    validation_fraction = float(config["validation_fraction"])
    split = pd.Series("train", index=frame.index, dtype="string")
    for offset, (_, building_frame) in enumerate(frame.groupby("building", sort=True)):
        groups = pd.DataFrame({"group_key": sorted(building_frame["_group_key"].unique())})
        if len(groups) < 3:
            raise ValueError(
                f"{design} requires at least three AHU groups in every building; "
                f"found {len(groups)}"
            )
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=random_state + offset
        )
        train_validation_idx, test_idx = next(
            splitter.split(groups, groups=groups["group_key"])
        )
        train_validation = groups.iloc[train_validation_idx]
        relative_validation = validation_fraction / (1 - test_fraction)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_validation,
            random_state=random_state + 100 + offset,
        )
        train_idx, validation_idx = next(
            splitter.split(train_validation, groups=train_validation["group_key"])
        )
        train_groups = set(train_validation.iloc[train_idx]["group_key"])
        validation_groups = set(train_validation.iloc[validation_idx]["group_key"])
        test_groups = set(groups.iloc[test_idx]["group_key"])
        split.loc[frame["_group_key"].isin(validation_groups)] = "validation"
        split.loc[frame["_group_key"].isin(test_groups)] = "test"
        if not train_groups or not validation_groups or not test_groups:
            raise ValueError(f"{design} produced an empty partition for a building")
    return _assignments(frame, design, "fold_001", split)


def _chronological(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    keep_duplicate_keys_together: bool = False,
) -> pd.DataFrame:
    """Split each building/AHU stream into past, validation and future rows."""
    test_fraction = float(config["test_fraction"])
    validation_fraction = float(config["validation_fraction"])
    split = pd.Series(pd.NA, index=frame.index, dtype="string")
    for _, indices in frame.groupby("_group_key", sort=True).groups.items():
        ordered = frame.loc[indices].sort_values(["timestamp", "_row_id"], kind="stable")
        count = len(ordered)
        train_target = int(count * (1 - validation_fraction - test_fraction))
        validation_target = int(count * (1 - test_fraction))
        if train_target < 1 or validation_target <= train_target or validation_target >= count:
            raise ValueError("chronological splitting requires at least three rows per AHU")
        if not keep_duplicate_keys_together:
            split.loc[ordered.index[:train_target]] = "train"
            split.loc[ordered.index[train_target:validation_target]] = "validation"
            split.loc[ordered.index[validation_target:]] = "test"
            continue
        key_columns = ["building", "ahu", "timestamp"]
        groups = ordered.groupby(key_columns, sort=False, dropna=False)
        cumulative = 0
        for _, group in groups:
            next_cumulative = cumulative + len(group)
            if next_cumulative <= train_target:
                partition = "train"
            elif next_cumulative <= validation_target:
                partition = "validation"
            else:
                partition = "test"
            split.loc[group.index] = partition
            cumulative = next_cumulative
    return _assignments(frame, "chronological", "fold_001", split)


def _building_held_out(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one transfer fold per held-out building."""
    buildings = sorted(frame["building"].unique())
    if len(buildings) < 3:
        raise ValueError("building_held_out requires at least three buildings")
    assignments: list[pd.DataFrame] = []
    for fold_number, held_out in enumerate(buildings, start=1):
        source_buildings = [building for building in buildings if building != held_out]
        validation_building = source_buildings[0]
        split = pd.Series("train", index=frame.index, dtype="string")
        split.loc[frame["building"].eq(validation_building)] = "validation"
        split.loc[frame["building"].eq(held_out)] = "test"
        assignments.append(
            _assignments(frame, "building_held_out", f"fold_{fold_number:03d}", split)
        )
    return pd.concat(assignments, ignore_index=True)


def _nested_grouped(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create outer grouped test folds and inner grouped validation partitions."""
    groups = frame["_group_key"]
    unique_groups = groups.nunique()
    outer_folds = min(int(config["outer_folds"]), unique_groups)
    if outer_folds < 3:
        raise ValueError("nested_grouped requires at least three AHU groups")
    splitter = GroupKFold(n_splits=outer_folds)
    assignments: list[pd.DataFrame] = []
    for fold_number, (outer_train_idx, outer_test_idx) in enumerate(
        splitter.split(frame, groups=groups), start=1
    ):
        outer_train = frame.iloc[outer_train_idx]
        outer_test = set(frame.iloc[outer_test_idx]["_group_key"])
        if outer_train["_group_key"].nunique() < 2:
            raise ValueError("nested_grouped needs at least two groups in each outer training set")
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=float(config["inner_validation_fraction"]),
            random_state=int(config["random_state"]) + fold_number,
        )
        inner_train_idx, inner_validation_idx = next(
            inner.split(outer_train, groups=outer_train["_group_key"])
        )
        train_groups = set(outer_train.iloc[inner_train_idx]["_group_key"])
        validation_groups = set(outer_train.iloc[inner_validation_idx]["_group_key"])
        split = pd.Series("train", index=frame.index, dtype="string")
        split.loc[frame["_group_key"].isin(validation_groups)] = "validation"
        split.loc[frame["_group_key"].isin(outer_test)] = "test"
        if not train_groups:
            raise ValueError("nested_grouped produced an empty inner training partition")
        assignments.append(
            _assignments(frame, "nested_grouped", f"fold_{fold_number:03d}", split)
        )
    return pd.concat(assignments, ignore_index=True)


def create_evaluation_splits(
    input_file: Path,
    output_dir: Path,
    config_path: Path,
    *,
    keep_duplicate_keys_together: bool = False,
) -> dict[str, Any]:
    config = load_split_config(config_path)
    frame = _prepare_frame(pd.read_csv(input_file, low_memory=False))
    outputs = [
        _group_holdout(frame, config, "ahu_held_out"),
        _chronological(
            frame, config, keep_duplicate_keys_together=keep_duplicate_keys_together
        ),
        _building_held_out(frame),
        _nested_grouped(frame, config),
    ]
    assignments = pd.concat(outputs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "evaluation_splits.csv"
    assignments.to_csv(output_csv, index=False)

    fold_summaries: list[dict[str, Any]] = []
    for (design, fold), group in assignments.groupby(["design", "fold"], sort=True):
        fold_summaries.append(
            {
                "design": design,
                "fold": fold,
                "row_counts": group["split"].value_counts().to_dict(),
                "group_counts": group.groupby("split")["group_key"].nunique().to_dict(),
                "building_counts": group.groupby("split")["building"].nunique().to_dict(),
            }
        )
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "input_file": str(input_file.expanduser().resolve()),
            "config_file": str(config_path.expanduser().resolve()),
            "row_count": int(len(frame)),
            "group_definition": "building::ahu",
            "designs": list(DESIGN_NAMES),
            "modeling_designs": list(MODELING_DESIGN_NAMES),
            "disabled_modeling_designs": list(DISABLED_MODELING_DESIGN_NAMES),
            "keep_duplicate_keys_together": keep_duplicate_keys_together,
            "fold_summaries": fold_summaries,
            "output_file": str(output_csv),
        }
    )
    manifest_path = output_dir / "evaluation_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output_dir / "evaluation_split_summary.md"
    lines = [
        "# Evaluation split summary",
        "",
        "The assignments are generated without random row-level headline splits.",
        "Every assignment is grouped by `building::ahu` where the design requires grouping.",
        "The pooled `ahu_held_out` assignment is retained for traceability but is disabled for current model runs.",
        "Active pooled designs: `chronological`, `building_held_out`, `nested_grouped`.",
        "",
        "| Design | Fold | Train rows | Validation rows | Test rows |", 
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in fold_summaries:
        counts = item["row_counts"]
        lines.append(
            f"| {item['design']} | {item['fold']} | {counts.get('train', 0):,} | "
            f"{counts.get('validation', 0):,} | {counts.get('test', 0):,} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_file": str(output_csv),
        "manifest_file": str(manifest_path),
        "summary_file": str(summary_path),
        "row_count": int(len(frame)),
        "designs": list(DESIGN_NAMES),
        "modeling_designs": list(MODELING_DESIGN_NAMES),
        "disabled_modeling_designs": list(DISABLED_MODELING_DESIGN_NAMES),
    }


def create_building_specific_splits(
    input_file: Path, output_dir: Path, config_path: Path
) -> dict[str, Any]:
    """Create independent within-building AHU and chronological assignments."""
    config = load_split_config(config_path)
    frame = _prepare_frame(pd.read_csv(input_file, low_memory=False))
    outputs: list[pd.DataFrame] = []
    for building, building_frame in frame.groupby("building", sort=True):
        ahu_split = _group_holdout(building_frame, config, "building_ahu_held_out")
        chronological = _chronological(
            building_frame, config, keep_duplicate_keys_together=True
        )
        for assignments in (ahu_split, chronological):
            assignments = assignments.copy()
            assignments["design"] = assignments["design"].map(
                {
                    "building_ahu_held_out": "building_ahu_held_out",
                    "chronological": "building_chronological",
                }
            )
            assignments["building_scope"] = building
            outputs.append(assignments)
    assignments = pd.concat(outputs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "building_splits.csv"
    assignments.to_csv(output_csv, index=False)
    fold_summaries: list[dict[str, Any]] = []
    for (building, design, fold), group in assignments.groupby(
        ["building_scope", "design", "fold"], sort=True
    ):
        fold_summaries.append(
            {
                "building": building,
                "design": design,
                "fold": fold,
                "row_counts": group["split"].value_counts().to_dict(),
                "group_counts": group.groupby("split")["group_key"].nunique().to_dict(),
            }
        )
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "input_file": str(input_file.expanduser().resolve()),
            "config_file": str(config_path.expanduser().resolve()),
            "row_count": int(len(frame)),
            "building_designs": list(BUILDING_DESIGN_NAMES),
            "buildings": sorted(frame["building"].unique()),
            "fold_summaries": fold_summaries,
            "output_file": str(output_csv),
        }
    )
    manifest_path = output_dir / "building_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Building-specific split summary",
        "",
        "Each building is split independently. AHU groups remain wholly within one partition.",
        "",
        "| Building | Design | Fold | Train rows | Validation rows | Test rows |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in fold_summaries:
        counts = item["row_counts"]
        lines.append(
            f"| {item['building']} | {item['design']} | {item['fold']} | "
            f"{counts.get('train', 0):,} | {counts.get('validation', 0):,} | "
            f"{counts.get('test', 0):,} |"
        )
    summary_path = output_dir / "building_split_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_file": str(output_csv),
        "manifest_file": str(manifest_path),
        "summary_file": str(summary_path),
        "row_count": int(len(frame)),
        "designs": list(BUILDING_DESIGN_NAMES),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create leakage-safe AHU evaluation splits")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/splits"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation_splits.yaml")
    )
    args = parser.parse_args(argv)
    result = create_evaluation_splits(args.input_file, args.output_dir, args.config)
    print(f"Created {result['row_count']:,} row assignments in {args.output_dir.resolve()}")
    return 0


def building_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create independent AHU and chronological splits per building"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/building_splits"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation_splits.yaml")
    )
    args = parser.parse_args(argv)
    result = create_building_specific_splits(args.input_file, args.output_dir, args.config)
    print(
        f"Created {result['row_count']:,} building-specific row assignments in "
        f"{args.output_dir.resolve()}"
    )
    return 0


def quality_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create quality-controlled pooled evaluation assignments"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation_splits.yaml")
    )
    args = parser.parse_args(argv)
    result = create_evaluation_splits(
        args.input_file,
        args.output_dir,
        args.config,
        keep_duplicate_keys_together=True,
    )
    print(
        f"Created {result['row_count']:,} quality-controlled row assignments in "
        f"{args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
