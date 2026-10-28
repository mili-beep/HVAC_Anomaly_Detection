from pathlib import Path

import pandas as pd
import pytest

from ahu_fdd.splits import (
    _building_held_out,
    _chronological,
    _group_holdout,
    _nested_grouped,
    _prepare_frame,
    create_evaluation_splits,
    load_split_config,
)
from ahu_fdd.validation import validate_assignments

CONFIG = load_split_config(Path("configs/evaluation_splits.yaml"))


def _frame() -> pd.DataFrame:
    rows = []
    for building in ("auditorium", "hospital", "office"):
        for ahu_number in range(1, 4):
            for hour in range(6):
                rows.append(
                    {
                        "building": building,
                        "ahu": f"AHU-{ahu_number}",
                        "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                        "label": "normal" if hour < 3 else "fault",
                    }
                )
    return pd.DataFrame(rows)


def _assert_group_disjoint(assignments: pd.DataFrame) -> None:
    for _, fold in assignments.groupby("fold"):
        groups = {
            split: set(rows["group_key"])
            for split, rows in fold.groupby("split")
        }
        assert groups.get("train", set()).isdisjoint(groups.get("validation", set()))
        assert groups.get("train", set()).isdisjoint(groups.get("test", set()))
        assert groups.get("validation", set()).isdisjoint(groups.get("test", set()))


def test_ahu_held_out_keeps_groups_disjoint() -> None:
    frame = _prepare_frame(_frame())
    result = _group_holdout(frame, CONFIG, "ahu_held_out")
    _assert_group_disjoint(result)


def test_chronological_split_has_no_future_rows_in_training() -> None:
    frame = _prepare_frame(_frame())
    result = _chronological(frame, CONFIG)
    for _, stream in result.groupby("group_key"):
        starts = stream.groupby("split")["timestamp"].min()
        ends = stream.groupby("split")["timestamp"].max()
        assert ends["train"] < starts["validation"] < starts["test"]


def test_building_held_out_uses_one_test_building_per_fold() -> None:
    frame = _prepare_frame(_frame())
    result = _building_held_out(frame)
    for _, fold in result.groupby("fold"):
        test_buildings = set(fold.loc[fold["split"].eq("test"), "building"])
        train_buildings = set(fold.loc[fold["split"].eq("train"), "building"])
        assert len(test_buildings) == 1
        assert test_buildings.isdisjoint(train_buildings)


def test_nested_grouped_keeps_outer_test_groups_unseen() -> None:
    frame = _prepare_frame(_frame())
    result = _nested_grouped(frame, CONFIG)
    _assert_group_disjoint(result)


def test_split_command_writes_one_assignment_table_and_manifest(tmp_path: Path) -> None:
    input_file = tmp_path / "canonical.csv"
    output_dir = tmp_path / "splits"
    _frame().to_csv(input_file, index=False)
    result = create_evaluation_splits(input_file, output_dir, Path("configs/evaluation_splits.yaml"))
    assignments = pd.read_csv(output_dir / "evaluation_splits.csv")
    assert result["row_count"] == len(_frame())
    assert set(assignments["design"]) == {
        "ahu_held_out",
        "chronological",
        "building_held_out",
        "nested_grouped",
    }
    assert (output_dir / "evaluation_split_manifest.json").is_file()
    assert (output_dir / "evaluation_split_summary.md").is_file()


def test_quality_splits_keep_duplicate_keys_together(tmp_path: Path) -> None:
    frame = _frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["timestamp"] = frame.iloc[0]["timestamp"]
    quality_input = pd.concat([frame, duplicate], ignore_index=True)
    input_file = tmp_path / "cleaned.csv"
    output_dir = tmp_path / "quality_splits"
    quality_input.to_csv(input_file, index=False)
    create_evaluation_splits(
        input_file,
        output_dir,
        Path("configs/evaluation_splits.yaml"),
        keep_duplicate_keys_together=True,
    )
    assignments = pd.read_csv(output_dir / "evaluation_splits.csv")
    chronological = assignments.loc[assignments["design"].eq("chronological")]
    duplicate_rows = chronological.loc[
        chronological["building"].eq("auditorium")
        & chronological["ahu"].eq("AHU-1")
        & chronological["timestamp"].eq("2024-01-01")
    ]
    assert duplicate_rows["split"].nunique() == 1


def test_grouped_split_rejects_too_few_ahus() -> None:
    frame = _prepare_frame(
        _frame().query("building == 'auditorium' and ahu != 'AHU-3'")
    )
    with pytest.raises(ValueError, match="at least three AHU groups"):
        _group_holdout(frame, CONFIG, "ahu_held_out")


def test_split_validation_requires_exact_row_coverage() -> None:
    frame = _prepare_frame(_frame())
    assignments = _chronological(frame, CONFIG)
    incomplete = assignments.loc[assignments["row_id"].ne(0)].copy()
    with pytest.raises(ValueError, match="does not cover its expected rows"):
        validate_assignments(incomplete, len(frame))


def test_split_validation_rejects_row_in_two_partitions() -> None:
    frame = _prepare_frame(_frame())
    assignments = _chronological(frame, CONFIG)
    first = assignments.iloc[[0]].copy()
    first["split"] = "test"
    invalid = pd.concat([assignments, first], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate a row"):
        validate_assignments(invalid, len(frame))
