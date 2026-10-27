import pandas as pd

from ahu_fdd.audit import (
    duplicate_key_summary,
    parse_timestamps,
    placeholder_mask,
)


def test_parse_timestamps_for_hospital_date_and_time() -> None:
    frame = pd.DataFrame({"DATE": ["20/10/2023"], "Time": ["08:00"]})
    parsed = parse_timestamps(frame, "hospital")
    assert parsed.iloc[0] == pd.Timestamp("2023-10-20 08:00")


def test_parse_timestamps_marks_invalid_rows() -> None:
    frame = pd.DataFrame({"Time": ["2022-01-01 00", "not-a-time"]})
    parsed = parse_timestamps(frame, "auditorium")
    assert parsed.notna().tolist() == [True, False]


def test_duplicate_key_summary_counts_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "AHU name": ["AHU-1", "AHU-1", "AHU-1"],
            "labeling": ["Normal", "Valve fault", "Normal"],
        }
    )
    timestamps = pd.Series(pd.to_datetime(["2022-01-01 00"] * 3))
    result = duplicate_key_summary(frame, timestamps)
    assert result["duplicate_key_count"] == 1
    assert result["duplicate_row_count"] == 3
    assert result["conflicting_duplicate_label_keys"] == 1


def test_placeholder_mask_detects_literal_dot_but_not_missing() -> None:
    values = pd.Series([".", " NA ", None, "12.5"])
    assert placeholder_mask(values).tolist() == [True, True, False, False]
