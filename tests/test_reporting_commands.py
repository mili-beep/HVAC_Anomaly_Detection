"""Smoke tests for evidence-producing comparison and support commands."""

from pathlib import Path

import pandas as pd

from ahu_fdd.building_comparisons import (
    run_directional_transfer,
    run_matched_comparison,
)
from ahu_fdd.diagnosis_support import run_support_report
from ahu_fdd.multibuilding_features import engineer_multibuilding
from ahu_fdd.splits import create_building_specific_splits, create_evaluation_splits


def _small_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sensors = [
        "setpoint_temperature",
        "return_temperature",
        "supply_air_temperature",
        "supply_fan",
        "valve_position",
        "heating_exchanger",
        "heating_supply_temperature_1",
        "heating_supply_temperature_2",
        "heating_pump_total",
        "heating_pump_1",
        "heating_pump_2",
        "heating_pump_3",
        "cooling_supply_temperature_1",
        "cooling_supply_temperature_2",
        "cooling_pump_total",
        "cooling_pump_1",
        "cooling_pump_2",
        "cooling_pump_3",
        "cooling_pump_4",
    ]
    for building in ("auditorium", "hospital", "office"):
        for ahu_number in range(1, 4):
            for hour in range(24):
                row: dict[str, object] = {
                    "building": building,
                    "ahu": f"AHU-{ahu_number}",
                    "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                    "source_row_id": len(rows),
                    "label": "Normal condition" if hour < 12 else "Supply fault",
                }
                row.update({sensor: 1.0 for sensor in sensors})
                row.update(
                    {
                        "setpoint_temperature": 20.0,
                        "supply_air_temperature": 19.0 if hour < 12 else 25.0,
                        "return_temperature": 21.0,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _prepared(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_file = tmp_path / "cleaned.csv"
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    _small_frame().to_csv(data_file, index=False)
    engineer_multibuilding(
        data_file,
        feature_dir,
        Path("configs/multibuilding_ahu.yaml"),
        Path("configs/feature_engineering.yaml"),
    )
    split_result = create_building_specific_splits(
        data_file, split_dir, Path("configs/evaluation_splits.yaml")
    )
    return (
        data_file,
        feature_dir / "multibuilding_features.csv",
        feature_dir / "feature_manifest.json",
        Path(split_result["output_file"]),
    )


def test_matched_and_directional_reports_write_evidence(tmp_path: Path) -> None:
    data_file, feature_file, manifest_file, splits_file = _prepared(tmp_path)
    matched = run_matched_comparison(
        data_file,
        feature_file,
        manifest_file,
        splits_file,
        tmp_path / "matched",
        Path("configs/detection.yaml"),
    )
    directional = run_directional_transfer(
        data_file,
        feature_file,
        manifest_file,
        splits_file,
        tmp_path / "directional",
        Path("configs/detection.yaml"),
    )
    assert Path(matched["test_metrics"]).is_file()
    assert Path(matched["row_checks"]).is_file()
    assert Path(directional["summary"]).is_file()
    assert len(pd.read_csv(directional["summary"])) == 3


def test_diagnosis_support_report_writes_summary(tmp_path: Path) -> None:
    data_file = tmp_path / "cleaned.csv"
    split_dir = tmp_path / "splits"
    output_dir = tmp_path / "support"
    _small_frame().to_csv(data_file, index=False)
    split_result = create_evaluation_splits(
        data_file, split_dir, Path("configs/evaluation_splits.yaml")
    )
    result = run_support_report(
        data_file,
        Path(split_result["output_file"]),
        output_dir,
        minimum_fault_rows=1,
    )
    assert Path(result["summary"]).is_file()
    assert Path(result["manifest"]).is_file()
