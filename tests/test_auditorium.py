import pandas as pd
import pytest

from ahu_fdd.auditorium import prepare_frame

POLICY = {
    "normal_label": "Normal condition",
    "binary_normal_label": "normal",
    "binary_fault_label": "fault",
    "sensor_feature_columns": ["Set point temperature", "Return temperature"],
    "metadata_columns": [
        "AHU name", "Time", "labeling", "building", "ahu", "source_row_id",
        "timestamp", "label_raw", "target_binary",
    ],
    "quality_columns": [
        "hours_since_previous_observation", "is_time_gap", "missing_sensor_count",
        "has_missing_sensor", "is_missing__Set point temperature",
        "is_missing__Return temperature",
    ],
}


def test_prepare_frame_maps_normal_and_fault_labels() -> None:
    frame = pd.DataFrame(
        {
            "AHU name": ["AHU-02", "AHU-01"],
            "Time": ["2022-01-01 01", "2022-01-01 00"],
            "Set point temperature": [22.0, 22.0],
            "Return temperature": [21.0, None],
            "labeling": ["Heating pump fault", "Normal condition"],
        }
    )
    prepared = prepare_frame(frame, POLICY)
    assert prepared["target_binary"].tolist() == ["normal", "fault"]
    assert prepared["ahu"].tolist() == ["AHU-01", "AHU-02"]
    assert prepared["label_raw"].tolist() == ["Normal condition", "Heating pump fault"]


def test_prepare_frame_marks_time_gaps_and_missingness() -> None:
    frame = pd.DataFrame(
        {
            "AHU name": ["AHU-01", "AHU-01"],
            "Time": ["2022-01-01 00", "2022-01-01 02"],
            "Set point temperature": [22.0, 22.0],
            "Return temperature": [21.0, None],
            "labeling": ["Normal condition", "Normal condition"],
        }
    )
    prepared = prepare_frame(frame, POLICY)
    assert prepared["is_time_gap"].tolist() == [False, True]
    assert prepared["is_missing__Return temperature"].tolist() == [False, True]
    assert prepared["missing_sensor_count"].tolist() == [0, 1]


def test_feature_roles_are_explicit_and_disjoint() -> None:
    prepared = prepare_frame(
        pd.DataFrame(
            {
                "AHU name": ["AHU-01"],
                "Time": ["2022-01-01 00"],
                "Set point temperature": [22.0],
                "Return temperature": [21.0],
                "labeling": ["Normal condition"],
            }
        ),
        POLICY,
    )
    sensor = set(POLICY["sensor_feature_columns"])
    metadata = set(POLICY["metadata_columns"])
    quality = set(POLICY["quality_columns"])
    assert sensor.isdisjoint(metadata | quality)
    assert set(POLICY["sensor_feature_columns"]) <= set(prepared.columns)
    assert not ({"target_binary", "label_raw", "AHU name"} & sensor)


def test_unclassified_source_column_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "AHU name": ["AHU-01"],
            "Time": ["2022-01-01 00"],
            "Set point temperature": [22.0],
            "Return temperature": [21.0],
            "unexpected_source_field": [1.0],
            "labeling": ["Normal condition"],
        }
    )
    with pytest.raises(ValueError, match="no declared feature role"):
        prepare_frame(frame, POLICY)
