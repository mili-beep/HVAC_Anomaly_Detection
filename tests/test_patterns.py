from pathlib import Path

import pandas as pd
import pytest

from ahu_fdd.patterns import discover_patterns, load_pattern_config

CONFIG = load_pattern_config(Path("configs/auditorium_patterns.yaml"))


def _pattern_input() -> pd.DataFrame:
    sensors = CONFIG["sensor_feature_columns"]
    rows = []
    for index, (hour, label, mode, value) in enumerate(
        [
            (0, "Normal condition", "heating", 18.0),
            (1, "Normal condition", "heating", 19.0),
            (2, "Supply fan fault", "heating", 99.0),
            (3, "Normal condition", "cooling", 22.0),
            (4, "Normal condition", "cooling", 23.0),
        ]
    ):
        row = {
            "building": "auditorium",
            "ahu": "AHU-01",
            "timestamp": f"2022-01-01 {hour:02d}:00:00",
            "source_row_id": index,
            "label_raw": label,
            "operating_mode": mode,
        }
        row.update({sensor: value + (offset * 0.1) for offset, sensor in enumerate(sensors)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_profiles_use_normal_rows_only_and_include_pca() -> None:
    outputs = discover_patterns(_pattern_input(), CONFIG)
    summary = outputs["summary"]
    assert summary["input_rows"] == 5
    assert summary["normal_rows"] == 4
    assert summary["non_normal_rows_excluded_from_profiles"] == 1
    assert outputs["hourly_profile"]["row_count"].unique().tolist() == [1]
    assert not outputs["pca_coordinates"].empty
    assert outputs["pca_variance"]["operating_mode"].nunique() == 2


def test_duplicate_ahu_timestamp_is_rejected() -> None:
    frame = pd.concat([_pattern_input(), _pattern_input().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate AHU/timestamp"):
        discover_patterns(frame, CONFIG)
