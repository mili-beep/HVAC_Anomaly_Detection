from pathlib import Path

import pandas as pd

from ahu_fdd.auditorium import load_policy, prepare_frame
from ahu_fdd.cleaning import (
    TrainingFoldMedianImputer,
    clean_frame,
    load_cleaning_config,
)

PHASE2 = load_policy(Path("configs/auditorium.yaml"))
PHASE4 = load_cleaning_config(Path("configs/auditorium.yaml"))


def _prepared_active_gap() -> pd.DataFrame:
    rows = []
    for hour, heating_supply in [(0, 40.0), (1, None), (2, 42.0)]:
        rows.append(
            {
                "AHU name": "AHU-01",
                "Time": f"2022-01-01 {hour:02d}",
                "Set point temperature": 22.0,
                "Return temperature": 21.0,
                "Supply air temperature": 20.0,
                "Supply fan": 1.0,
                "Valve position": 0.0,
                "Heating supply temperature": heating_supply,
                "Total heating pump": 1.0,
                "Heating pump 1": 1.0,
                "Heating pump 2": 0.0,
                "Heating pump 3": 0.0,
                "Cooling supply temperature 1": None,
                "Cooling supply temperature 2": None,
                "Total cooling pump": None,
                "Cooling pump 1": None,
                "Cooling pump 2": None,
                "labeling": "Normal condition",
            }
        )
    return prepare_frame(pd.DataFrame(rows), PHASE2)


def test_cleaning_classifies_structural_missingness_without_imputing_it() -> None:
    cleaned, _, summary = clean_frame(_prepared_active_gap(), PHASE4)
    assert summary["rows_imputed"] == 0
    assert summary["rows_with_short_gap_candidates"] == 1
    assert cleaned["operating_mode"].tolist() == ["heating"] * 3
    assert cleaned["structural_missing__Cooling supply temperature 1"].all()
    assert cleaned["short_gap_eligible__Heating supply temperature"].tolist() == [
        False, True, False
    ]


def test_training_fold_median_imputes_only_eligible_active_gap() -> None:
    cleaned, _, _ = clean_frame(_prepared_active_gap(), PHASE4)
    imputer = TrainingFoldMedianImputer(PHASE4["sensor_feature_columns"])
    transformed, report = imputer.fit(cleaned.iloc[[0, 2]]).transform(cleaned)
    assert transformed.loc[1, "Heating supply temperature"] == 41.0
    assert transformed.loc[1, "imputed__Heating supply temperature"]
    assert transformed["Cooling supply temperature 1"].isna().all()
    assert int(report.loc[report.sensor.eq("Heating supply temperature"), "imputed_count"].iloc[0]) == 1
