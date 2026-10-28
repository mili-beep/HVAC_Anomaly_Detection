from pathlib import Path

import numpy as np
import pandas as pd

from ahu_fdd.multibuilding import load_config
from ahu_fdd.multibuilding_features import (
    engineer_multibuilding,
    engineer_multibuilding_features,
    load_feature_settings,
)

BUILDING_CONFIG = load_config(Path("configs/multibuilding_ahu.yaml"))
SETTINGS = load_feature_settings(Path("configs/feature_engineering.yaml"))


def _frame() -> pd.DataFrame:
    sensors = BUILDING_CONFIG["canonical_sensor_columns"]
    rows: list[dict[str, object]] = []
    for building, ahu, hours in [("auditorium", "A-01", [0, 1, 2]), ("office", "O-01", [0, 1])]:
        for hour in hours:
            row: dict[str, object] = {
                "building": building,
                "ahu": ahu,
                "timestamp": f"2024-01-01 {hour:02d}:00:00",
                "source_row_id": len(rows),
                "label": "Normal condition" if hour == 0 else "Supply fault",
            }
            row.update({sensor: 1.0 for sensor in sensors})
            row["setpoint_temperature"] = 20.0
            row["supply_air_temperature"] = 21.0 + hour
            row["return_temperature"] = 22.0
            row["heating_pump_1"] = 1.0 if hour else 0.0
            row["is_missing__heating_pump_2"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def test_shared_features_are_past_only_and_grouped_by_building_and_ahu() -> None:
    features, lineage, variants = engineer_multibuilding_features(
        _frame(), BUILDING_CONFIG["canonical_sensor_columns"], SETTINGS
    )
    rolling = features["temporal__past_mean_3__return_temperature"]
    assert np.isnan(rolling.iloc[0])
    assert rolling.iloc[1] == 22.0
    assert np.isnan(rolling.iloc[3])
    assert lineage.loc[lineage["feature"].eq("temporal__past_mean_3__return_temperature"), "past_only"].iloc[0]
    assert set(variants["physical_signal"]).isdisjoint({"label", "ahu", "timestamp"})


def test_temporal_features_break_at_timestamp_gaps() -> None:
    frame = _frame().iloc[:3].copy()
    frame["timestamp"] = [
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
        "2024-01-01 04:00:00",
    ]
    features, _, _ = engineer_multibuilding_features(
        frame, BUILDING_CONFIG["canonical_sensor_columns"], SETTINGS
    )
    assert np.isnan(features["temporal__past_mean_3__return_temperature"].iloc[2])
    assert np.isnan(features["temporal__past_rate__return_temperature"].iloc[2])


def test_operational_context_is_additive_and_labels_are_not_predictors() -> None:
    features, _, variants = engineer_multibuilding_features(
        _frame(), BUILDING_CONFIG["canonical_sensor_columns"], SETTINGS
    )
    assert set(variants["physical_signal"]).issubset(variants["operational_context"])
    assert any(name.startswith("mode__") for name in variants["operational_context"])
    assert any(name.startswith("context__") for name in variants["operational_context"])
    assert not {"label", "building", "ahu", "timestamp"} & set(variants["operational_context"])
    assert features["label"].tolist() == [
        "Normal condition", "Supply fault", "Supply fault", "Normal condition", "Supply fault"
    ]


def test_duplicate_keys_are_rejected() -> None:
    duplicate = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    try:
        engineer_multibuilding_features(
            duplicate, BUILDING_CONFIG["canonical_sensor_columns"], SETTINGS
        )
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:  # pragma: no cover
        raise AssertionError("Duplicate keys must be rejected")


def test_multibuilding_command_writes_one_table_and_manifest(tmp_path: Path) -> None:
    input_file = tmp_path / "cleaned.csv"
    output_dir = tmp_path / "features"
    _frame().to_csv(input_file, index=False)
    manifest = engineer_multibuilding(
        input_file,
        output_dir,
        Path("configs/multibuilding_ahu.yaml"),
        Path("configs/feature_engineering.yaml"),
    )
    assert manifest["row_count"] == 5
    assert manifest["feature_counts"]["physical_signal"] == 88
    assert (output_dir / "multibuilding_features.csv").is_file()
    assert (output_dir / "feature_manifest.json").is_file()
