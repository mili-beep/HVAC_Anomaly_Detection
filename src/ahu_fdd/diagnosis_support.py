"""Generate train/test fault-class support tables from frozen assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ahu_fdd.splits import MODELING_DESIGN_NAMES


def run_support_report(
    data_file: Path,
    splits_file: Path,
    output_dir: Path,
    minimum_fault_rows: int = 30,
) -> dict[str, str]:
    data = pd.read_csv(data_file, low_memory=False).reset_index(drop=True)
    assignments = pd.read_csv(splits_file, low_memory=False)
    if "row_id" not in assignments or assignments["row_id"].isna().any():
        raise ValueError("Diagnosis support assignments must contain non-null row_id values")
    row_ids = assignments["row_id"].astype(int)
    if (row_ids < 0).any() or (row_ids >= len(data)).any():
        raise ValueError("Diagnosis support assignments contain row IDs outside the input table")
    labels = data["label"].astype("string").str.strip()
    counts = labels.loc[~labels.eq("Normal condition")].value_counts()
    eligible = set(counts[counts >= minimum_fault_rows].index.astype(str))
    active = assignments.loc[
        assignments["design"].isin(MODELING_DESIGN_NAMES)
        & assignments["split"].isin(["train", "test"])
    ].copy()
    active["fault_type"] = labels.iloc[active["row_id"].astype(int).to_numpy()].to_numpy()
    active = active.loc[active["fault_type"].isin(eligible)]
    group_keys = active[["design", "fold", "building"]].drop_duplicates()
    fault_frame = pd.DataFrame({"fault_type": sorted(eligible)})
    grid = group_keys.merge(fault_frame, how="cross")
    counts = (
        active.groupby(["design", "fold", "building", "fault_type", "split"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    rows: list[dict[str, object]] = []
    for record in grid.to_dict("records"):
        match = counts.loc[
            counts["design"].eq(record["design"])
            & counts["fold"].eq(record["fold"])
            & counts["building"].eq(record["building"])
            & counts["fault_type"].eq(record["fault_type"])
        ]
        train_rows = int(match.loc[match["split"].eq("train"), "rows"].sum())
        test_rows = int(match.loc[match["split"].eq("test"), "rows"].sum())
        rows.append(
            {
                **record,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "present_in_training": train_rows > 0,
                "unsupported_test_rows": test_rows if train_rows == 0 else 0,
            }
        )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "diagnosis_support_by_split.csv"
    frame.to_csv(detail_path, index=False)
    summary = (
        frame.groupby(["design", "fault_type"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            train_rows=("train_rows", "sum"),
            test_rows=("test_rows", "sum"),
            unsupported_test_rows=("unsupported_test_rows", "sum"),
        )
    )
    summary_path = output_dir / "diagnosis_support_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 1,
        "data_file": str(data_file.expanduser().resolve()),
        "splits_file": str(splits_file.expanduser().resolve()),
        "minimum_fault_rows": minimum_fault_rows,
        "eligible_fault_types": sorted(eligible),
        "output_files": {"by_split": str(detail_path), "summary": str(summary_path)},
    }
    manifest_path = output_dir / "diagnosis_support_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"by_split": str(detail_path), "summary": str(summary_path), "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise diagnosis class support by split")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnosis"))
    parser.add_argument("--minimum-fault-rows", type=int, default=30)
    args = parser.parse_args()
    run_support_report(args.data_file, args.splits_file, args.output_dir, args.minimum_fault_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
