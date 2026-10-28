from pathlib import Path

import pandas as pd

from ahu_fdd.baselines import BASELINE_NAMES, run_baselines
from ahu_fdd.splits import create_evaluation_splits


def _frame() -> pd.DataFrame:
    rows = []
    sensors = {
        "setpoint_temperature": 20.0,
        "return_temperature": 21.0,
        "supply_air_temperature": 19.0,
        "supply_fan": 1.0,
        "valve_position": 0.0,
        "heating_exchanger": 0.0,
        "heating_supply_temperature_1": 40.0,
        "heating_supply_temperature_2": 40.0,
        "heating_pump_total": 1.0,
        "heating_pump_1": 1.0,
        "heating_pump_2": 0.0,
        "heating_pump_3": 0.0,
        "cooling_supply_temperature_1": 12.0,
        "cooling_supply_temperature_2": 12.0,
        "cooling_pump_total": 0.0,
        "cooling_pump_1": 0.0,
        "cooling_pump_2": 0.0,
        "cooling_pump_3": 0.0,
        "cooling_pump_4": 0.0,
    }
    for building in ("auditorium", "hospital", "office"):
        for ahu_number in range(1, 4):
            for hour in range(12):
                row = {
                    "building": building,
                    "ahu": f"AHU-{ahu_number}",
                    "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                    "label": "Normal condition" if hour < 6 else "Supply fan fault",
                    **sensors,
                }
                if hour >= 6:
                    row["supply_fan"] = 0.0
                    row["return_temperature"] = 26.0
                rows.append(row)
    return pd.DataFrame(rows)


def test_baseline_run_writes_fold_and_summary_outputs(tmp_path: Path) -> None:
    assert BASELINE_NAMES == (
        "majority_class",
        "normal_range_rule",
        "actuator_pump_only",
    )
    data_file = tmp_path / "canonical.csv"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "baselines"
    _frame().to_csv(data_file, index=False)
    create_evaluation_splits(
        data_file, splits_dir, Path("configs/evaluation_splits.yaml")
    )
    result = run_baselines(
        data_file,
        splits_dir / "evaluation_splits.csv",
        output_dir,
        Path("configs/baselines.yaml"),
    )
    summary = pd.read_csv(output_dir / "baseline_metrics_summary.csv")
    assert set(summary["baseline"]) == set(BASELINE_NAMES)
    assert set(summary["design"]) == {
        "ahu_held_out",
        "chronological",
        "building_held_out",
        "nested_grouped",
    }
    assert (output_dir / "baseline_metrics_by_fold.csv").is_file()
    assert (output_dir / "baseline_manifest.json").is_file()
    assert result["row_count"] == len(_frame())


def test_baseline_summary_contains_required_metrics(tmp_path: Path) -> None:
    data_file = tmp_path / "canonical.csv"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "baselines"
    _frame().to_csv(data_file, index=False)
    create_evaluation_splits(
        data_file, splits_dir, Path("configs/evaluation_splits.yaml")
    )
    run_baselines(
        data_file,
        splits_dir / "evaluation_splits.csv",
        output_dir,
        Path("configs/baselines.yaml"),
    )
    summary = pd.read_csv(output_dir / "baseline_metrics_summary.csv")
    assert {
        "mean_f1",
        "mean_macro_f1",
        "mean_balanced_accuracy",
        "mean_false_positive_rate",
    } <= set(summary.columns)
