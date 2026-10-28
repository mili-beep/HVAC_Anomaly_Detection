"""Validation helpers for frozen evaluation assignments."""

from __future__ import annotations

import pandas as pd


def validate_assignments(assignments: pd.DataFrame, row_count: int) -> None:
    """Validate complete, non-overlapping train/validation/test assignments.

    Pooled tables must cover every input row in each design/fold. Building-
    specific tables carry ``building_scope`` and must cover every row in that
    scope instead.
    """
    required = {"row_id", "design", "fold", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")
    if assignments["row_id"].isna().any():
        raise ValueError("Split assignments contain missing row IDs")
    numeric_ids = pd.to_numeric(assignments["row_id"], errors="coerce")
    if numeric_ids.isna().any() or (numeric_ids % 1 != 0).any():
        raise ValueError("Split assignments contain non-integer row IDs")
    row_ids = numeric_ids.astype("int64")
    if row_ids.lt(0).any() or row_ids.ge(row_count).any():
        raise ValueError("Split assignments contain row IDs outside the input table")
    checked = assignments.copy()
    checked["row_id"] = row_ids
    scope_column = "building_scope" if "building_scope" in checked.columns else None
    group_columns = ["design", "fold"]
    if scope_column:
        group_columns.append(scope_column)
    for group_key, group in checked.groupby(group_columns, dropna=False):
        if group.duplicated("row_id").any():
            raise ValueError(f"Split assignments duplicate a row within {group_key}")
        if scope_column:
            scope = group[scope_column].iloc[0]
            expected_ids = set(
                checked.loc[checked[scope_column].eq(scope), "row_id"]
            )
        else:
            expected_ids = set(range(row_count))
        actual_ids = set(group["row_id"])
        if actual_ids != expected_ids:
            missing_ids = sorted(expected_ids - actual_ids)[:5]
            extra_ids = sorted(actual_ids - expected_ids)[:5]
            raise ValueError(
                f"{group_key} does not cover its expected rows; "
                f"missing={missing_ids}, extra={extra_ids}"
            )
        if set(group["split"].astype(str)) != {"train", "validation", "test"}:
            raise ValueError(f"{group_key} must contain train, validation and test rows")
        if group["split"].isna().any():
            raise ValueError(f"{group_key} contains a missing split assignment")
