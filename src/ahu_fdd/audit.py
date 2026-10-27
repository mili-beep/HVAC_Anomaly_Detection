"""Raw-data audit for the multibuilding AHU dataset.

The audit is deliberately descriptive. It does not clean, impute, relabel, or
train models; those decisions belong to later project phases.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_FILES = {
    "auditorium": "auditorium_scientific_data.csv",
    "hospital": "hosptial_scientific_data.csv",
    "office": "office_scientific_data.csv",
}
LABEL_COLUMN = "labeling"
AHU_COLUMN = "AHU name"
PLACEHOLDER_TOKENS = frozenset({".", "", "na", "n/a", "nan", "null", "none"})
NUMERIC_RATIO_THRESHOLD = 0.8


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamps(frame: pd.DataFrame, building: str) -> pd.Series:
    """Parse source timestamps without changing the input frame."""
    if building == "hospital":
        if not {"DATE", "Time"}.issubset(frame.columns):
            raise ValueError("Hospital data requires DATE and Time columns")
        raw = (
            frame["DATE"].astype("string").str.strip()
            + " "
            + frame["Time"].astype("string").str.strip()
        )
        return pd.to_datetime(raw, format="%d/%m/%Y %H:%M", errors="coerce")

    if "Time" not in frame.columns:
        raise ValueError(f"{building} data requires a Time column")
    return pd.to_datetime(frame["Time"], errors="coerce")


def placeholder_mask(series: pd.Series) -> pd.Series:
    """Return a mask for explicit textual placeholder tokens."""
    values = series.astype("string").str.strip().str.lower()
    return values.isin(PLACEHOLDER_TOKENS) & series.notna()


def duplicate_key_summary(
    frame: pd.DataFrame, timestamps: pd.Series
) -> dict[str, Any]:
    """Summarise duplicate AHU/timestamp keys and conflicting labels."""
    keys = pd.DataFrame({"ahu": frame[AHU_COLUMN], "timestamp": timestamps})
    valid = keys["ahu"].notna() & keys["timestamp"].notna()
    valid_keys = keys.loc[valid]
    duplicate_rows = valid_keys.duplicated(keep=False)
    duplicate_key_values = valid_keys.loc[duplicate_rows].drop_duplicates()

    conflicting_labels = 0
    if LABEL_COLUMN in frame.columns and duplicate_key_values.shape[0]:
        labelled = pd.DataFrame(
            {
                "ahu": frame.loc[valid, AHU_COLUMN].to_numpy(),
                "timestamp": timestamps.loc[valid].to_numpy(),
                "label": frame.loc[valid, LABEL_COLUMN].astype("string").to_numpy(),
            }
        )
        duplicate_label_counts = labelled.groupby(["ahu", "timestamp"])["label"].nunique()
        conflicting_labels = int((duplicate_label_counts > 1).sum())

    return {
        "valid_key_rows": int(len(valid_keys)),
        "duplicate_key_count": int(len(duplicate_key_values)),
        "duplicate_row_count": int(duplicate_rows.sum()),
        "conflicting_duplicate_label_keys": conflicting_labels,
    }


def exact_duplicate_summary(frame: pd.DataFrame) -> dict[str, int]:
    duplicated = frame.duplicated(keep=False)
    return {
        "duplicate_row_count": int(duplicated.sum()),
        "duplicate_row_group_count": int(frame.loc[duplicated].drop_duplicates().shape[0]),
    }


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _safe_float(value)
    if isinstance(value, (np.bool_, bool, str, int)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)


def _counts_to_records(table: pd.DataFrame, names: list[str]) -> list[dict[str, Any]]:
    if table.empty:
        return []
    reset = table.rename_axis(index=names[: table.index.nlevels]).reset_index()
    return reset.to_dict(orient="records")


def _column_audit(frame: pd.DataFrame, building: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    column_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        missing = series.isna()
        placeholders = placeholder_mask(series)
        numeric = pd.to_numeric(series, errors="coerce")
        nonempty = series.notna() & ~placeholders
        numeric_nonempty = numeric.notna() & nonempty
        numeric_ratio = float(numeric_nonempty.sum() / nonempty.sum()) if nonempty.any() else None
        nonfinite = pd.Series(False, index=series.index)
        if numeric_ratio is not None and numeric_ratio >= NUMERIC_RATIO_THRESHOLD:
            nonfinite = numeric.notna() & ~np.isfinite(numeric)

        column_rows.append(
            {
                "building": building,
                "column": column,
                "pandas_dtype": str(series.dtype),
                "inferred_type": str(pd.api.types.infer_dtype(series, skipna=True)),
                "numeric_parse_ratio": numeric_ratio,
                "row_count": int(len(series)),
                "unique_non_null": int(series.nunique(dropna=True)),
                "constant_non_null": bool(series.nunique(dropna=True) <= 1),
                "missing_count": int(missing.sum()),
                "placeholder_count": int(placeholders.sum()),
                "nonfinite_count": int(nonfinite.sum()),
            }
        )
        if missing.any() or placeholders.any() or nonfinite.any():
            missing_rows.append(
                {
                    "building": building,
                    "column": column,
                    "missing_count": int(missing.sum()),
                    "missing_rate": float(missing.mean()),
                    "placeholder_count": int(placeholders.sum()),
                    "placeholder_rate": float(placeholders.mean()),
                    "nonfinite_count": int(nonfinite.sum()),
                }
            )
    return column_rows, missing_rows


def _numeric_audit(frame: pd.DataFrame, building: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric.notna()
        if not valid.any():
            continue
        values = numeric.loc[valid].astype(float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "building": building,
                "column": column,
                "numeric_value_count": int(len(values)),
                "finite_value_count": int(len(finite)),
                "minimum": _safe_float(finite.min()) if len(finite) else None,
                "maximum": _safe_float(finite.max()) if len(finite) else None,
                "mean": _safe_float(finite.mean()) if len(finite) else None,
                "std": _safe_float(finite.std()) if len(finite) > 1 else 0.0,
                "negative_count": int((finite < 0).sum()),
                "zero_count": int((finite == 0).sum()),
                "over_100_count": int((finite > 100).sum()),
                "over_1000_count": int((finite > 1000).sum()),
                "nonfinite_count": int((~np.isfinite(values)).sum()),
            }
        )
    return rows


def _coverage_audit(frame: pd.DataFrame, timestamps: pd.Series) -> dict[str, Any]:
    valid = timestamps.notna() & frame[AHU_COLUMN].notna()
    work = pd.DataFrame(
        {"ahu": frame.loc[valid, AHU_COLUMN].astype("string"), "timestamp": timestamps.loc[valid]}
    )
    if work.empty:
        return {"expected_definition": "inclusive calendar dates from each AHU minimum to maximum", "by_ahu": []}

    records: list[dict[str, Any]] = []
    for ahu, group in work.groupby("ahu", sort=True):
        observed_dates = pd.DatetimeIndex(group["timestamp"].dt.normalize().unique()).sort_values()
        start = observed_dates.min()
        end = observed_dates.max()
        expected_dates = pd.date_range(start, end, freq="D")
        missing_dates = expected_dates.difference(observed_dates)
        expected_hours = pd.date_range(start, end, freq="h")
        observed_hours = pd.DatetimeIndex(group["timestamp"].drop_duplicates()).sort_values()
        missing_hours = expected_hours.difference(observed_hours)
        records.append(
            {
                "ahu": str(ahu),
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "expected_calendar_days": int(len(expected_dates)),
                "observed_calendar_days": int(len(observed_dates)),
                "missing_calendar_days": int(len(missing_dates)),
                "coverage_rate": float(len(observed_dates) / len(expected_dates)),
                "missing_dates": [item.date().isoformat() for item in missing_dates],
                "expected_hourly_timestamps": int(len(expected_hours)),
                "observed_unique_timestamps": int(len(observed_hours)),
                "missing_hourly_timestamps": int(len(missing_hours)),
                "hourly_coverage_rate": float(len(observed_hours) / len(expected_hours)),
                "missing_timestamp_examples": [item.isoformat() for item in missing_hours[:20]],
            }
        )
    return {
        "expected_definition": "inclusive calendar dates from each AHU minimum to maximum",
        "by_ahu": records,
    }


def _ordering_audit(frame: pd.DataFrame, timestamps: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"ahu": frame[AHU_COLUMN], "timestamp": timestamps})
    records: list[dict[str, Any]] = []
    for ahu, group in work.groupby("ahu", sort=True):
        original_values = group["timestamp"].dropna()
        original_differences = original_values.diff().dropna()
        backwards = original_differences < pd.Timedelta(0)
        sorted_values = original_values.sort_values(kind="stable")
        sorted_differences = sorted_values.diff().dropna()
        positive = sorted_differences[sorted_differences > pd.Timedelta(0)]
        gaps = positive[positive > pd.Timedelta(hours=1)]
        records.append(
            {
                "ahu": str(ahu),
                "row_count_with_timestamp": int(len(original_values)),
                "backward_step_count": int(backwards.sum()),
                "chronological_non_decreasing": bool(not backwards.any()),
                "largest_forward_gap_hours": _safe_float(positive.max() / pd.Timedelta(hours=1))
                if len(positive)
                else 0.0,
                "gaps_over_one_hour_count": int(len(gaps)),
                "missing_hour_estimate_from_gaps": int(
                    sum(max(int(gap / pd.Timedelta(hours=1)) - 1, 0) for gap in gaps)
                ),
            }
        )
    return {
        "all_ahus_chronological_non_decreasing": bool(
            all(item["chronological_non_decreasing"] for item in records)
        ),
        "by_ahu": records,
    }


def _run_length_audit(frame: pd.DataFrame, timestamps: pd.Series) -> dict[str, Any]:
    if LABEL_COLUMN not in frame.columns:
        return {"by_label": [], "longest_runs": []}
    work = pd.DataFrame(
        {
            "ahu": frame[AHU_COLUMN],
            "timestamp": timestamps,
            "label": frame[LABEL_COLUMN].astype("string"),
        }
    ).dropna(subset=["ahu", "timestamp", "label"])
    work = work.sort_values(["ahu", "timestamp"], kind="stable")
    if work.empty:
        return {"by_label": [], "longest_runs": []}
    work["new_run"] = work["label"].ne(work.groupby("ahu", sort=False)["label"].shift())
    work["run_id"] = work.groupby("ahu", sort=False)["new_run"].cumsum()
    runs = (
        work.groupby(["ahu", "run_id", "label"], sort=False)
        .size()
        .rename("run_length")
        .reset_index()
    )
    summary = (
        runs.groupby("label")["run_length"]
        .agg(run_count="count", minimum="min", maximum="max", mean="mean", median="median")
        .reset_index()
        .to_dict(orient="records")
    )
    longest = runs.sort_values("run_length", ascending=False).head(20).to_dict(orient="records")
    return {"by_label": summary, "longest_runs": longest}


def _label_audit(frame: pd.DataFrame, timestamps: pd.Series) -> dict[str, Any]:
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Missing required label column {LABEL_COLUMN!r}")
    label = frame[LABEL_COLUMN].astype("string")
    overall = label.value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    by_ahu = pd.crosstab(frame[AHU_COLUMN].astype("string"), label)
    valid_time = timestamps.notna()
    month = timestamps.dt.month.where(valid_time)
    hour = timestamps.dt.hour.where(valid_time)
    by_month = pd.crosstab(month, label)
    by_hour = pd.crosstab(hour, label)
    return {
        "overall": overall.to_dict(orient="records"),
        "by_ahu": _counts_to_records(by_ahu, ["ahu"]),
        "by_month": _counts_to_records(by_month, ["month"]),
        "by_hour": _counts_to_records(by_hour, ["hour"]),
    }


def _label_quality_associations(
    frame: pd.DataFrame, label: pd.Series
) -> dict[str, list[dict[str, Any]]]:
    missing_rows: list[dict[str, Any]] = []
    placeholder_rows: list[dict[str, Any]] = []
    for column in frame.columns:
        missing = frame[column].isna()
        placeholders = placeholder_mask(frame[column])
        if missing.any():
            counts = label.loc[missing].value_counts(dropna=False)
            for value, count in counts.items():
                missing_rows.append(
                    {
                        "column": column,
                        "label": str(value),
                        "row_count": int(count),
                        "feature_missing_rate": float(missing.mean()),
                    }
                )
        if placeholders.any():
            counts = label.loc[placeholders].value_counts(dropna=False)
            for value, count in counts.items():
                placeholder_rows.append(
                    {
                        "column": column,
                        "label": str(value),
                        "row_count": int(count),
                        "feature_placeholder_rate": float(placeholders.mean()),
                    }
                )
    return {"missing_by_label": missing_rows, "placeholder_by_label": placeholder_rows}


def audit_file(path: Path, building: str) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    timestamps = parse_timestamps(frame, building)
    column_rows, missing_rows = _column_audit(frame, building)
    label = frame[LABEL_COLUMN].astype("string")
    placeholder_counts = [row for row in column_rows if row["placeholder_count"]]

    result: dict[str, Any] = {
        "building": building,
        "source_file": path.name,
        "source_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": column_rows,
        "ahu_column": AHU_COLUMN,
        "ahu_count": int(frame[AHU_COLUMN].nunique(dropna=True)),
        "ahu_row_counts": frame[AHU_COLUMN].value_counts(dropna=False)
        .rename_axis("ahu")
        .reset_index(name="row_count")
        .to_dict(orient="records"),
        "timestamp": {
            "source_columns": ["DATE", "Time"] if building == "hospital" else ["Time"],
            "parse_success_count": int(timestamps.notna().sum()),
            "parse_failure_count": int(timestamps.isna().sum()),
            "parse_success_rate": float(timestamps.notna().mean()),
            "minimum": timestamps.min(),
            "maximum": timestamps.max(),
            "invalid_examples": frame.loc[timestamps.isna(), [column for column in ["DATE", "Time"] if column in frame]]
            .head(10)
            .to_dict(orient="records"),
        },
        "chronological_order": _ordering_audit(frame, timestamps),
        "duplicate_ahu_timestamp": duplicate_key_summary(frame, timestamps),
        "exact_duplicates": exact_duplicate_summary(frame),
        "calendar_coverage": _coverage_audit(frame, timestamps),
        "missingness": missing_rows,
        "placeholder_columns": placeholder_counts,
        "nonfinite_columns": [row for row in column_rows if row["nonfinite_count"]],
        "numeric_summary": _numeric_audit(frame, building),
        "label": _label_audit(frame, timestamps),
        "label_quality_associations": _label_quality_associations(frame, label),
        "label_run_lengths": _run_length_audit(frame, timestamps),
    }
    return _jsonable(result)


def _summary_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Multibuilding AHU raw-data audit",
        "",
        f"Generated: `{manifest['generated_at']}`",
        f"Data root: `{manifest['data_root']}`",
        "",
        "> This is a descriptive audit. No cleaning, imputation, relabelling, or modelling was performed.",
        "",
        "## File overview",
        "",
        "| Building | Rows | Columns | AHUs | Timestamp success | SHA-256 (prefix) |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in manifest["files"]:
        timestamp = item["timestamp"]
        lines.append(
            f"| {item['building']} | {item['row_count']:,} | {item['column_count']} | "
            f"{item['ahu_count']} | {timestamp['parse_success_rate']:.2%} | "
            f"`{item['sha256'][:12]}…` |"
        )

    for item in manifest["files"]:
        lines.extend(["", f"## {item['building'].title()}", ""])
        timestamp = item["timestamp"]
        lines.extend(
            [
                f"- Source file: `{item['source_file']}` ({item['file_size_bytes']:,} bytes)",
                f"- Timestamp range: `{timestamp['minimum']}` to `{timestamp['maximum']}`",
                f"- Timestamp parse failures: **{timestamp['parse_failure_count']:,}**",
                f"- AHU/timestamp duplicate rows: **{item['duplicate_ahu_timestamp']['duplicate_row_count']:,}** "
                f"across **{item['duplicate_ahu_timestamp']['duplicate_key_count']:,}** keys",
                f"- Exact duplicate rows: **{item['exact_duplicates']['duplicate_row_count']:,}**",
                f"- Chronological within-AHU order: **{item['chronological_order']['all_ahus_chronological_non_decreasing']}**",
                f"- Forward gaps over one hour: **{sum(row['gaps_over_one_hour_count'] for row in item['chronological_order']['by_ahu']):,}**",
            ]
        )
        lines.append("\n### Labels\n")
        lines.extend(["| Label | Rows |", "| --- | ---: |"])
        for row in item["label"]["overall"]:
            lines.append(f"| {row['label']} | {row['count']:,} |")
        coverage = item["calendar_coverage"]["by_ahu"]
        missing_days = sum(row["missing_calendar_days"] for row in coverage)
        lines.extend(
            [
                "",
                f"Calendar coverage is measured against each AHU's inclusive observed date range. "
                f"Across AHUs, **{missing_days:,}** calendar days are absent from those inferred ranges.",
            ]
        )
        notable = [
            row
            for row in item["missingness"]
            if row["missing_count"] or row["placeholder_count"] or row["nonfinite_count"]
        ]
        lines.extend(["", "### Notable field-quality counts", ""])
        lines.extend(["| Field | Missing | Placeholders | Non-finite |", "| --- | ---: | ---: | ---: |"])
        for row in sorted(notable, key=lambda value: value["missing_count"], reverse=True)[:20]:
            lines.append(
                f"| `{row['column']}` | {row['missing_count']:,} | "
                f"{row['placeholder_count']:,} | {row['nonfinite_count']:,} |"
            )
        if not notable:
            lines.append("| None detected | 0 | 0 | 0 |")
    return "\n".join(lines) + "\n"


def _write_csv_outputs(manifest: dict[str, Any], output_dir: Path) -> None:
    columns: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for item in manifest["files"]:
        columns.extend(item["columns"])
        missingness.extend(item["missingness"])
        for dimension in ("overall", "by_ahu", "by_month", "by_hour"):
            for row in item["label"][dimension]:
                labels.append({"building": item["building"], "dimension": dimension, **row})
    pd.DataFrame(columns).to_csv(output_dir / "audit_columns.csv", index=False)
    pd.DataFrame(missingness).to_csv(output_dir / "audit_missingness.csv", index=False)
    pd.DataFrame(labels).to_csv(output_dir / "audit_label_counts.csv", index=False)


def run_audit(data_root: Path, output_dir: Path) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in DATASET_FILES.values() if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing expected source files under {data_root}: {', '.join(missing)}"
        )

    files = [audit_file(data_root / filename, building) for building, filename in DATASET_FILES.items()]
    manifest = _jsonable(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_root": str(data_root),
            "expected_files": DATASET_FILES,
            "expected_coverage_definition": "inclusive observed date span per AHU; source frequency is not assumed",
            "numeric_screening_definitions": {
                "negative_count": "finite numeric values less than 0",
                "over_100_count": "finite numeric values greater than 100",
                "over_1000_count": "finite numeric values greater than 1000",
                "interpretation": "screening counts only; these are not physical-validity judgements without a feature dictionary",
            },
            "files": files,
        }
    )
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_summary.md").write_text(
        _summary_markdown(manifest), encoding="utf-8"
    )
    _write_csv_outputs(manifest, output_dir)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit multibuilding AHU source CSV files")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing the three source CSV files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/audit"),
        help="Directory for the JSON, Markdown, and CSV audit outputs",
    )
    args = parser.parse_args(argv)
    run_audit(args.data_root, args.output_dir)
    print(f"Audit written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
