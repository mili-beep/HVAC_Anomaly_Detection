"""Small internal helpers shared by the audit and modelling modules."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLACEHOLDER_TOKENS = frozenset({".", "", "na", "n/a", "nan", "null", "none"})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def placeholder_mask(series: pd.Series) -> pd.Series:
    """Return a mask for explicit textual placeholder tokens."""
    values = series.astype("string").str.strip().str.lower()
    return values.isin(PLACEHOLDER_TOKENS) & series.notna()


def numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric and represent infinities as missing values."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def numeric_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return selected columns coerced to finite numeric values."""
    selected = frame if columns is None else frame[columns]
    return selected.apply(numeric_series)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool, str, int)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    """Write a JSON-safe value with stable formatting and a trailing newline."""
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
