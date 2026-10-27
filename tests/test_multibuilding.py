from pathlib import Path

import pandas as pd
import pytest

from ahu_fdd.multibuilding import canonicalize_frame, load_config

CONFIG = load_config(Path("configs/multibuilding_ahu.yaml"))


def test_all_three_source_mappings_cover_the_canonical_schema() -> None:
    canonical = CONFIG["canonical_sensor_columns"]
    assert set(CONFIG["buildings"]) == {"auditorium", "hospital", "office"}
    for spec in CONFIG["buildings"].values():
        assert set(spec["sensor_columns"]) == set(canonical)


def test_hospital_timestamp_and_placeholder_are_preserved_and_flagged() -> None:
    spec = CONFIG["buildings"]["hospital"]
    frame = pd.DataFrame(
        {
            "AHU name": ["AHU-N01", "AHU-N01"],
            "DATE": ["20/10/2023", "20/10/2023"],
            "Time": ["08:00", "09:00"],
            "Set point temperature": [22.0, 22.0],
            "Return temperature": [".", 21.0],
            "Supply air temperature": [20.0, 20.0],
            "Supply fan": [1.0, 1.0],
            "Valve position": [0.0, 0.0],
            "Heat Exchanger": [1.0, 1.0],
            "Heating supply temperature": [40.0, 40.0],
            "Cooling supply temperature": [12.0, 12.0],
            "cooling pump": [1.0, 1.0],
            "labeling": ["Normal condition", "Return air temperature fault"],
        }
    )
    result, metadata = canonicalize_frame(
        frame, "hospital", spec, CONFIG["canonical_sensor_columns"], "hospital.csv"
    )
    assert result["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2023-10-20 08:00", "2023-10-20 09:00"
    ]
    assert result["return_temperature"].tolist() == [".", 21.0]
    assert result["has_placeholder__return_temperature"].tolist() == [True, False]
    assert metadata["placeholder_counts"]["return_temperature"] == 1


def test_canonicalizer_rejects_unparseable_timestamps() -> None:
    spec = CONFIG["buildings"]["auditorium"]
    frame = pd.DataFrame(
        {
            "AHU name": ["AHU-01"],
            "Time": ["not-a-timestamp"],
            "Set point temperature": [22.0],
            "Return temperature": [21.0],
            "Supply air temperature": [20.0],
            "Supply fan": [1.0],
            "Valve position": [0.0],
            "Heating supply temperature": [40.0],
            "Total heating pump": [1.0],
            "Heating pump 1": [1.0],
            "Heating pump 2": [0.0],
            "Heating pump 3": [0.0],
            "Cooling supply temperature 1": [12.0],
            "Cooling supply temperature 2": [12.0],
            "Total cooling pump": [1.0],
            "Cooling pump 1": [1.0],
            "Cooling pump 2": [0.0],
            "labeling": ["Normal condition"],
        }
    )
    with pytest.raises(ValueError, match="unparseable timestamps"):
        canonicalize_frame(frame, "auditorium", spec, CONFIG["canonical_sensor_columns"])


def test_office_schema_canonicalizes_all_declared_source_sensors() -> None:
    spec = CONFIG["buildings"]["office"]
    frame = {
        "AHU name": ["AHU-101"],
        "Time": ["2024-06-01 00"],
        "labeling": ["Normal condition"],
    }
    for source in spec["sensor_columns"].values():
        if source is not None:
            frame[source] = [1.0]
    result, _ = canonicalize_frame(
        pd.DataFrame(frame), "office", spec, CONFIG["canonical_sensor_columns"]
    )
    assert result["building"].tolist() == ["office"]
    assert result["timestamp"].dt.hour.tolist() == [0]
    assert result["cooling_pump_4"].tolist() == [1.0]
