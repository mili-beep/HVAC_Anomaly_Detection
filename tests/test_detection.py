from pathlib import Path

import numpy as np
import pandas as pd

from ahu_fdd.detection import (
    MODEL_NAMES,
    _sequence_windows,
    build_detection_features,
    run_detection,
)
from ahu_fdd.splits import create_evaluation_splits

SENSORS = [
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


def _frame() -> pd.DataFrame:
    rows = []
    for building in ("auditorium", "hospital", "office"):
        for ahu_number in range(1, 4):
            for hour in range(12):
                row = {
                    "building": building,
                    "ahu": f"AHU-{ahu_number}",
                    "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                    "label": "Normal condition" if hour < 6 else "Supply fan fault",
                }
                row.update({sensor: 1.0 for sensor in SENSORS})
                row.update(
                    {
                        "setpoint_temperature": 20.0,
                        "return_temperature": 21.0 if hour < 6 else 27.0,
                        "supply_air_temperature": 19.0,
                        "supply_fan": 1.0 if hour < 6 else 0.0,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_feature_variants_are_explicit_and_label_free() -> None:
    features, variants = build_detection_features(_frame(), SENSORS)
    assert set(variants) == {"physical_signal", "operational_context"}
    assert len(variants["operational_context"]) > len(variants["physical_signal"])
    assert not {"label", "ahu", "building", "timestamp"} & set(
        variants["physical_signal"] + variants["operational_context"]
    )
    assert set(features["row_id"]) == set(range(len(_frame())))


def test_lstm_windows_break_at_timestamp_gaps() -> None:
    frame = pd.DataFrame(
        {
            "building": ["auditorium"] * 5,
            "ahu": ["A-01"] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 05:00",
                    "2024-01-01 06:00",
                ]
            ),
            "row_id": range(5),
        }
    )
    matrix = np.arange(5, dtype=float).reshape(-1, 1)
    windows, rows = _sequence_windows(frame, matrix, sequence_length=3, max_gap_hours=1)
    assert windows.shape == (1, 3, 1)
    assert rows == [2]
    assert windows[:, :, 0].tolist() == [[0.0, 1.0, 2.0]]


def test_detection_runner_writes_fold_and_summary_outputs(tmp_path: Path) -> None:
    data_file = tmp_path / "canonical.csv"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "detection"
    _frame().to_csv(data_file, index=False)
    create_evaluation_splits(
        data_file, splits_dir, Path("configs/evaluation_splits.yaml")
    )
    result = run_detection(
        data_file,
        splits_dir / "evaluation_splits.csv",
        output_dir,
        Path("configs/detection.yaml"),
        Path("configs/multibuilding_ahu.yaml"),
    )
    summary = pd.read_csv(output_dir / "detection_metrics_summary.csv")
    assert set(result["models_requested"]) == set(MODEL_NAMES)
    assert {"normal_range_rule", "isolation_forest", "xgboost"} <= set(summary["model"])
    assert set(summary["variant"]) == {"physical_signal", "operational_context"}
    assert (output_dir / "detection_manifest.json").is_file()
    assert (output_dir / "detection_summary.md").is_file()
