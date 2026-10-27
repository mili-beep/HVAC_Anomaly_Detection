import pandas as pd

from ahu_fdd.auditorium import prepare_frame

POLICY = {
    "normal_label": "Normal condition",
    "binary_normal_label": "normal",
    "binary_fault_label": "fault",
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
