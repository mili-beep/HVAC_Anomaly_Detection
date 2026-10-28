import json
from pathlib import Path

import pandas as pd
import pytest

from ahu_fdd.priority import (
    attach_diagnosis,
    group_priority_events,
    load_policy,
    run_priority_scoring,
)


def policy():
    return load_policy(Path("configs/operational_priority.yaml"))


def test_groups_adjacent_rows_and_scores_event():
    rows = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "building": ["auditorium", "auditorium", "auditorium"],
            "ahu": ["AHU-01"] * 3,
            "timestamp": ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00"],
            "prediction": [1, 1, 1],
            "predicted_type": ["Supply fan fault"] * 3,
            "type_model_score": [0.9, 0.8, 0.95],
            "type_abstained": [False, False, False],
        }
    )
    events = group_priority_events(rows, policy())
    assert len(events) == 2
    assert events.loc[0, "detected_rows"] == 2
    assert events.loc[0, "priority_band"] == "High"


def test_missing_prediction_columns_are_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        group_priority_events(pd.DataFrame({"building": ["auditorium"]}), policy())


def test_abstention_is_reported_as_uncertain():
    rows = pd.DataFrame(
        {
            "row_id": [1],
            "building": ["office"],
            "ahu": ["AHU-01"],
            "timestamp": ["2024-01-01 00:00"],
            "prediction": [1],
            "predicted_type": ["Supply fan fault"],
            "type_model_score": [0.69],
            "type_abstained": [True],
        }
    )
    event = group_priority_events(rows, policy()).iloc[0]
    assert event["fault_type"] == "Uncertain"


def test_events_do_not_cross_fold_boundaries():
    rows = pd.DataFrame(
        {
            "row_id": [1, 2],
            "building": ["office", "office"],
            "ahu": ["AHU-01", "AHU-01"],
            "design": ["nested_grouped", "nested_grouped"],
            "fold": ["fold_001", "fold_002"],
            "timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"],
            "prediction": [1, 1],
            "predicted_type": ["Supply fan fault"] * 2,
            "type_model_score": [0.9, 0.9],
            "type_abstained": [False, False],
        }
    )
    events = group_priority_events(rows, policy())
    assert len(events) == 2


def test_detector_alerts_are_filtered_and_diagnosis_is_optional():
    detections = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "building": ["office"] * 3,
            "ahu": ["AHU-01"] * 3,
            "timestamp": [
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-01 02:00",
            ],
            "design": ["chronological"] * 3,
            "fold": ["fold_001"] * 3,
            "prediction": [0, 1, 1],
        }
    )
    diagnosis = pd.DataFrame(
        {
            "row_id": [2],
            "design": ["chronological"],
            "fold": ["fold_001"],
            "task": ["fault_type"],
            "predicted_type": ["Supply fan fault"],
            "type_model_score": [0.9],
            "type_abstained": [False],
        }
    )
    attached = attach_diagnosis(detections, diagnosis)
    events = group_priority_events(attached, policy())
    assert len(events) == 1
    assert events.loc[0, "detected_rows"] == 2
    assert events.loc[0, "fault_type"] == "Supply fan fault"
    assert events.loc[0, "diagnosis_model_score"] == pytest.approx(0.45)


def test_detector_rows_are_required_even_without_diagnosis():
    rows = pd.DataFrame(
        {
            "row_id": [1],
            "building": ["office"],
            "ahu": ["AHU-01"],
            "timestamp": ["2024-01-01 00:00"],
            "prediction": [1],
        }
    )
    event = group_priority_events(rows, policy()).iloc[0]
    assert event["fault_type"] == "Uncertain"


def test_priority_runner_uses_detector_rows_and_writes_audit_manifest(tmp_path):
    detections = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "building": ["office"] * 3,
            "ahu": ["AHU-01"] * 3,
            "timestamp": [
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-01 02:00",
            ],
            "design": ["chronological"] * 3,
            "fold": ["fold_001"] * 3,
            "model": ["xgboost"] * 3,
            "prediction": [0, 1, 1],
        }
    )
    detection_path = tmp_path / "detector.csv"
    detections.to_csv(detection_path, index=False)
    output_dir = tmp_path / "operational"
    result = run_priority_scoring(
        detection_path,
        Path("configs/operational_priority.yaml"),
        output_dir,
    )
    events = pd.read_csv(result["events"])
    assert len(events) == 1
    assert events.loc[0, "detected_rows"] == 2
    manifest = json.loads((output_dir / "priority_manifest.json").read_text())
    assert manifest["detected_rows"] == 2
    assert manifest["diagnosis_matched_rows"] == 0
