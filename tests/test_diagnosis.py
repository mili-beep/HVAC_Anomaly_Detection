from pathlib import Path

import pandas as pd

from ahu_fdd.diagnosis import run_diagnosis
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
    labels = [
        "Normal condition",
        "Supply fan fault",
        "Valve position fault",
    ]
    for building in ("auditorium", "hospital", "office"):
        for ahu_number in range(1, 4):
            for hour in range(12):
                label = labels[hour // 4]
                row = {
                    "building": building,
                    "ahu": f"AHU-{ahu_number}",
                    "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                    "label": label,
                }
                row.update({sensor: 1.0 for sensor in SENSORS})
                row.update(
                    {
                        "setpoint_temperature": 20.0,
                        "return_temperature": 21.0 if label == "Normal condition" else 27.0,
                        "supply_air_temperature": 19.0,
                        "supply_fan": 1.0 if label != "Supply fan fault" else 0.0,
                        "valve_position": 1.0 if label != "Valve position fault" else 0.0,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_diagnosis_runner_writes_auditable_outputs(tmp_path: Path) -> None:
    data_file = tmp_path / "canonical.csv"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "diagnosis"
    _frame().to_csv(data_file, index=False)
    create_evaluation_splits(
        data_file, splits_dir, Path("configs/evaluation_splits.yaml")
    )
    result = run_diagnosis(
        data_file,
        splits_dir / "evaluation_splits.csv",
        output_dir,
        Path("configs/diagnosis.yaml"),
        Path("configs/multibuilding_ahu.yaml"),
        requested_designs="chronological",
    )
    summary = pd.read_csv(output_dir / "diagnosis_metrics_summary.csv")
    assert set(summary["task"]) == {
        "flat_type",
        "hierarchical_group",
        "hierarchical_type",
    }
    assert (output_dir / "diagnosis_predictions.csv").is_file()
    assert (output_dir / "diagnosis_manifest.json").is_file()
    assert result["markdown"].endswith("diagnosis_summary.md")
