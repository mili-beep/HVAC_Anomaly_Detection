from pathlib import Path

import pandas as pd

from ahu_fdd.multibuilding_features import engineer_multibuilding
from ahu_fdd.quality_detection import run_quality_detection
from ahu_fdd.splits import create_evaluation_splits


def _frame() -> pd.DataFrame:
    rows = []
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
                row = {
                    "building": building,
                    "ahu": f"AHU-{ahu_number}",
                    "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                    "source_row_id": len(rows),
                    "label": "Normal condition" if hour < 12 else "Supply fault",
                }
                row.update({sensor: 1.0 for sensor in sensors})
                row["setpoint_temperature"] = 20.0
                row["supply_air_temperature"] = 19.0 if hour < 12 else 25.0
                row["return_temperature"] = 21.0
                rows.append(row)
    return pd.DataFrame(rows)


def test_quality_detection_writes_metrics_and_manifest(tmp_path: Path) -> None:
    data_file = tmp_path / "cleaned.csv"
    feature_dir = tmp_path / "features"
    split_dir = tmp_path / "splits"
    output_dir = tmp_path / "results"
    _frame().to_csv(data_file, index=False)
    engineer_multibuilding(
        data_file,
        feature_dir,
        Path("configs/multibuilding_ahu.yaml"),
        Path("configs/feature_engineering.yaml"),
    )
    create_evaluation_splits(
        data_file, split_dir, Path("configs/evaluation_splits.yaml")
    )
    result = run_quality_detection(
        data_file,
        feature_dir / "multibuilding_features.csv",
        feature_dir / "feature_manifest.json",
        split_dir / "evaluation_splits.csv",
        output_dir,
        Path("configs/detection.yaml"),
        "exclude_ambiguous_keys",
        requested_models=["normal_range_rule"],
    )
    summary = pd.read_csv(result["summary"])
    assert set(summary["model"]) == {"normal_range_rule"}
    assert (output_dir / "quality_detection_manifest.json").is_file()
