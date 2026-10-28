from pathlib import Path

import numpy as np
import pandas as pd

from ahu_fdd.utils import json_safe, numeric_series, placeholder_mask, write_json


def test_numeric_series_coerces_invalid_and_nonfinite_values() -> None:
    values = numeric_series(pd.Series(["1.5", "not-a-number", np.inf, -np.inf]))
    assert values.iloc[0] == 1.5
    assert values.iloc[1:].isna().all()


def test_placeholder_mask_detects_explicit_tokens_only() -> None:
    values = pd.Series([".", " NA ", None, "0"])
    assert placeholder_mask(values).tolist() == [True, True, False, False]


def test_json_safe_and_write_json_handle_project_values(tmp_path: Path) -> None:
    payload = {
        "path": tmp_path / "data.csv",
        "timestamp": pd.Timestamp("2026-01-02T03:04:05Z"),
        "integer": np.int64(3),
        "nonfinite": np.inf,
    }
    safe = json_safe(payload)
    assert safe["path"].endswith("data.csv")
    assert safe["integer"] == 3
    assert safe["nonfinite"] is None

    output = tmp_path / "manifest.json"
    write_json(output, payload)
    assert '"nonfinite": null' in output.read_text(encoding="utf-8")
