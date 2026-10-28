from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ahu_fdd.multibuilding import load_config
from ahu_fdd.quality_cleaning import _verify_output, clean_canonical_frame
from ahu_fdd.quality_policy import POLICY_NAMES, load_duplicate_policy

CONFIG = Path("configs/duplicate_policies.yaml")


def test_duplicate_policies_are_frozen_and_non_aggregating() -> None:
    config = load_duplicate_policy(CONFIG)
    assert tuple(config["policies"]) == POLICY_NAMES
    assert config["policies"]["exclude_ambiguous_keys"]["action"] == (
        "exclude_all_rows_for_duplicate_keys"
    )
    assert config["policies"]["retain_flagged_keys"]["action"] == "retain_all_rows"


def test_duplicate_policy_rejects_selection_or_aggregation(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """duplicate_key_columns: [building, ahu, timestamp]
policies:
  exclude_ambiguous_keys:
    action: exclude_all_rows_for_duplicate_keys
    retain_excluded_rows_in_audit: true
    row_selection: first
    aggregation: none
  retain_flagged_keys:
    action: retain_all_rows
    retain_excluded_rows_in_audit: true
    row_selection: none
    aggregation: none
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot select or aggregate"):
        load_duplicate_policy(invalid)


def test_cleaning_excludes_or_retains_all_rows_for_an_ambiguous_key() -> None:
    building_config = load_config(Path("configs/multibuilding_ahu.yaml"))
    policy_config = load_duplicate_policy(CONFIG)
    sensors = building_config["canonical_sensor_columns"]
    rows = []
    for source_row_id, hour in enumerate([0, 0, 1]):
        row = {
            "building": "hospital",
            "ahu": "AHU-N01",
            "timestamp": f"2023-01-01 0{hour}:00:00",
            "label": "Normal condition",
            "source_row_id": source_row_id,
        }
        row.update({sensor: 1.0 for sensor in sensors})
        row["return_temperature"] = "." if source_row_id == 1 else 20.0
        row["heating_pump_total"] = np.inf if source_row_id == 2 else 1.0
        rows.append(row)
    frame = pd.DataFrame(rows)

    excluded, excluded_summary, _ = clean_canonical_frame(
        frame, sensors, "exclude_ambiguous_keys", policy_config
    )
    retained, retained_summary, _ = clean_canonical_frame(
        frame, sensors, "retain_flagged_keys", policy_config
    )

    assert len(excluded) == 1
    assert excluded_summary["rows_excluded"] == 2
    assert len(retained) == 3
    assert retained_summary["rows_excluded"] == 0
    assert retained_summary["output_duplicate_key_count"] == 1
    assert retained["duplicate_ahu_timestamp"].sum() == 2
    assert retained["return_temperature"].isna().sum() == 1
    assert retained["heating_pump_total"].isna().sum() == 1


def test_second_audit_accepts_the_two_declared_outputs() -> None:
    building_config = load_config(Path("configs/multibuilding_ahu.yaml"))
    policy_config = load_duplicate_policy(CONFIG)
    sensors = building_config["canonical_sensor_columns"]
    rows = []
    for source_row_id, hour in enumerate([0, 0, 1]):
        row = {
            "building": "hospital",
            "ahu": "AHU-N01",
            "timestamp": f"2023-01-01 0{hour}:00:00",
            "label": "Normal condition",
            "source_row_id": source_row_id,
        }
        row.update({sensor: 1.0 for sensor in sensors})
        rows.append(row)
    raw = pd.DataFrame(rows)
    retained, _, _ = clean_canonical_frame(
        raw, sensors, "retain_flagged_keys", policy_config
    )
    retained_summary, _ = _verify_output(
        raw, retained, sensors, "retain_flagged_keys", policy_config
    )
    assert retained_summary["checks_passed"] is True
    assert retained_summary["rows_imputed"] == 0
