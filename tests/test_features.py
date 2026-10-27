from pathlib import Path

import numpy as np
import pandas as pd

from ahu_fdd.features import (
    engineer_auditorium,
    engineer_features,
    load_feature_config,
    shortcut_audit,
)

CONFIG = load_feature_config(Path("configs/auditorium.yaml"))


def _feature_input() -> pd.DataFrame:
    sensors = CONFIG["sensor_feature_columns"]
    rows = []
    for ahu, hours in [("AHU-01", [0, 1, 2]), ("AHU-02", [0])]:
        for hour in hours:
            row = {
                "building": "auditorium",
                "ahu": ahu,
                "timestamp": f"2022-01-01 {hour:02d}:00:00",
                "source_row_id": len(rows),
                "label_raw": "Normal condition" if hour == 0 else "Supply fan fault",
                "target_binary": "normal" if hour == 0 else "fault",
            }
            row.update({sensor: 1.0 for sensor in sensors})
            row["Set point temperature"] = 20.0
            row["Supply air temperature"] = 21.0
            row["Return temperature"] = float(hour + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_features_are_past_only_and_do_not_cross_ahus() -> None:
    features, lineage, variants = engineer_features(_feature_input(), CONFIG)
    rolling = features["temporal__past_mean_3__return_temperature"].tolist()
    rates = features["temporal__past_rate__return_temperature"].tolist()
    assert np.isnan(rolling[0])
    assert rolling[1:3] == [1.0, 1.5]
    assert np.isnan(rolling[3])
    assert np.isnan(rates[0])
    assert rates[1:3] == [1.0, 1.0]
    assert np.isnan(rates[3])
    assert lineage.loc[lineage.feature.eq("temporal__past_mean_3__return_temperature"), "past_only"].iloc[0]
    assert set(variants["physical_signal"]).isdisjoint({"target_binary", "label_raw", "ahu"})


def test_shortcut_audit_is_review_only_and_has_no_metadata_features() -> None:
    features, lineage, variants = engineer_features(_feature_input(), CONFIG)
    audit = shortcut_audit(features, lineage, variants, CONFIG)
    assert len(audit) == sum(len(columns) for columns in variants.values())
    assert not {"target_binary", "label_raw", "ahu", "timestamp"} & set(audit["feature"])
    assert {"missingness_shortcut_flag", "value_separation_flag"} <= set(audit.columns)


def test_feature_engineering_writes_only_declared_variants(tmp_path: Path) -> None:
    input_file = tmp_path / "cleaned.csv"
    output_dir = tmp_path / "phase5"
    _feature_input().to_csv(input_file, index=False)

    manifest = engineer_auditorium(input_file, output_dir, Path("configs/auditorium.yaml"))

    assert set(manifest["output_files"]) == {
        "lineage",
        "shortcut_audit",
        "physical_signal",
        "operational_context",
    }
    assert not (output_dir / "auditorium_features.csv").exists()
