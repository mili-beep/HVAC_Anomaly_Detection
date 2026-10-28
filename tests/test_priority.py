import pandas as pd
import pytest

from ahu_fdd.priority import group_priority_events, load_policy


def policy():
    return load_policy(__import__("pathlib").Path("configs/operational_priority.yaml"))


def test_groups_adjacent_rows_and_scores_event():
    rows = pd.DataFrame(
        {
            "building": ["auditorium", "auditorium", "auditorium"],
            "ahu": ["AHU-01"] * 3,
            "timestamp": ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00"],
            "predicted_type": ["Supply fan fault"] * 3,
            "type_confidence": [0.9, 0.8, 0.95],
            "type_abstained": [False, False, False],
        }
    )
    events = group_priority_events(rows, policy())
    assert len(events) == 2
    assert events.loc[0, "labelled_rows"] == 2
    assert events.loc[0, "priority_band"] == "High"


def test_missing_prediction_columns_are_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        group_priority_events(pd.DataFrame({"building": ["auditorium"]}), policy())


def test_abstention_is_reported_as_uncertain():
    rows = pd.DataFrame(
        {
            "building": ["office"],
            "ahu": ["AHU-01"],
            "timestamp": ["2024-01-01 00:00"],
            "predicted_type": ["Supply fan fault"],
            "type_confidence": [0.69],
            "type_abstained": [True],
        }
    )
    event = group_priority_events(rows, policy()).iloc[0]
    assert event["fault_type"] == "Uncertain"


def test_events_do_not_cross_fold_boundaries():
    rows = pd.DataFrame(
        {
            "building": ["office", "office"],
            "ahu": ["AHU-01", "AHU-01"],
            "design": ["nested_grouped", "nested_grouped"],
            "fold": ["fold_001", "fold_002"],
            "timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"],
            "predicted_type": ["Supply fan fault"] * 2,
            "type_confidence": [0.9, 0.9],
            "type_abstained": [False, False],
        }
    )
    events = group_priority_events(rows, policy())
    assert len(events) == 2
