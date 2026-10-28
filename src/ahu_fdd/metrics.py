"""Shared evaluation metrics for binary detection runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_metric_row(
    design: str,
    fold: str,
    model: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Return one auditable binary classification metric record."""
    true = y_true.to_numpy(dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    true_negative = int(((true == 0) & (predicted == 0)).sum())
    false_positive = int(((true == 0) & (predicted == 1)).sum())
    false_negative = int(((true == 1) & (predicted == 0)).sum())
    true_positive = int(((true == 1) & (predicted == 1)).sum())
    return {
        "design": design,
        "fold": fold,
        "baseline": model,
        "rows": int(len(true)),
        "normal_support": int((true == 0).sum()),
        "fault_support": int((true == 1).sum()),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": (
            float(
                0.5
                * (
                    true_negative / (true_negative + false_positive)
                    + true_positive / (true_positive + false_negative)
                )
            )
            if (true_negative + false_positive) and (true_positive + false_negative)
            else np.nan
        ),
        "precision": float(
            precision_score(true, predicted, labels=[0, 1], zero_division=0)
        ),
        "recall": float(recall_score(true, predicted, labels=[0, 1], zero_division=0)),
        "f1": float(f1_score(true, predicted, labels=[0, 1], zero_division=0)),
        "macro_f1": float(
            f1_score(true, predicted, labels=[0, 1], average="macro", zero_division=0)
        ),
        "false_positive_rate": float(
            false_positive / (false_positive + true_negative)
            if true_negative + false_positive
            else 0.0
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }
