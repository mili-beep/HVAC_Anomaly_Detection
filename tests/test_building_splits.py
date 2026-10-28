from pathlib import Path

import pandas as pd

from ahu_fdd.splits import create_building_specific_splits


def _frame() -> pd.DataFrame:
    rows = []
    for building in ("auditorium", "office", "hospital"):
        for ahu_number in range(1, 4):
            for hour in range(10):
                rows.append(
                    {
                        "building": building,
                        "ahu": f"AHU-{ahu_number}",
                        "timestamp": f"2024-01-{hour + 1:02d} 00:00:00",
                        "label": "Normal condition",
                    }
                )
    return pd.DataFrame(rows)


def test_building_specific_splits_do_not_mix_buildings_or_ahus(tmp_path: Path) -> None:
    input_file = tmp_path / "canonical.csv"
    output_dir = tmp_path / "splits"
    _frame().to_csv(input_file, index=False)
    result = create_building_specific_splits(
        input_file, output_dir, Path("configs/evaluation_splits.yaml")
    )
    assignments = pd.read_csv(result["output_file"])
    assert set(assignments["building_scope"]) == {"auditorium", "hospital", "office"}
    assert set(assignments["design"]) == {"building_ahu_held_out", "building_chronological"}
    ahu_assignments = assignments.loc[assignments["design"].eq("building_ahu_held_out")]
    assert not ahu_assignments.groupby(
        ["building_scope", "fold", "group_key"]
    )["split"].nunique().gt(1).any()
