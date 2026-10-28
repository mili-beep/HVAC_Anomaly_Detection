from pathlib import Path

import numpy as np
import pandas as pd

from ahu_fdd.modeling import MODEL_NAMES, load_detection_config, sequence_windows


def test_detection_configuration_lists_shared_model_adapters() -> None:
    assert MODEL_NAMES == (
        "normal_range_rule",
        "isolation_forest",
        "xgboost",
        "lstm_autoencoder",
    )
    config = load_detection_config(Path("configs/detection.yaml"))
    assert set(MODEL_NAMES) <= set(config["models"])


def test_lstm_windows_break_at_timestamp_gaps() -> None:
    frame = pd.DataFrame(
        {
            "building": ["auditorium"] * 5,
            "ahu": ["A-01"] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 05:00",
                    "2024-01-01 06:00",
                ]
            ),
            "row_id": range(5),
        }
    )
    matrix = np.arange(5, dtype=float).reshape(-1, 1)
    windows, rows = sequence_windows(frame, matrix, sequence_length=3, max_gap_hours=1)
    assert windows.shape == (1, 3, 1)
    assert rows == [2]
    assert windows[:, :, 0].tolist() == [[0.0, 1.0, 2.0]]
