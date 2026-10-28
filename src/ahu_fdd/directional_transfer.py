"""Small zero-shot directional transfer comparison for the building notebook."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ahu_fdd.baselines import _metric_row, _validate_assignments
from ahu_fdd.building_detection import _load_inputs
from ahu_fdd.detection import FoldTransformer, _xgboost, load_detection_config

TRANSFER_PAIRS = (
    ("auditorium", "office", "single_source"),
    ("office", "auditorium", "single_source"),
    ("auditorium+office", "hospital", "multi_source"),
)


def run_directional_transfer(
    data_file: Path,
    feature_file: Path,
    feature_manifest_file: Path,
    splits_file: Path,
    output_dir: Path,
    config_file: Path,
) -> dict[str, str]:
    config = load_detection_config(config_file)
    data, features, assignments, physical = _load_inputs(
        data_file, feature_file, feature_manifest_file, splits_file
    )
    _validate_assignments(assignments, len(data))
    chronological = assignments.loc[
        assignments["design"].eq("building_chronological")
        & assignments["fold"].eq("fold_001")
    ]
    labels = data["label"].astype("string").str.strip()
    y = (~labels.eq(str(config["normal_label"]).strip())).astype(int)
    rows: list[dict[str, object]] = []
    for source, target, transfer_type in TRANSFER_PAIRS:
        source_buildings = source.split("+")
        source_train = chronological.loc[
            chronological["building_scope"].isin(source_buildings)
            & chronological["split"].eq("train"),
            "row_id",
        ].astype(int)
        target_test = chronological.loc[
            chronological["building_scope"].eq(target)
            & chronological["split"].eq("test"),
            "row_id",
        ].astype(int)
        if source_train.empty or target_test.empty:
            raise ValueError(f"Missing source or target rows for {source} -> {target}")
        transformer = FoldTransformer(physical).fit(features.iloc[source_train])
        predictions = _xgboost(
            transformer.transform(features.iloc[source_train]),
            transformer.transform(features.iloc[target_test]),
            y.iloc[source_train],
            config,
        )
        row = _metric_row("directional_chronological", "fold_001", "xgboost", y.iloc[target_test], predictions)
        row.update(
            {
                "source_buildings": source,
                "target_building": target,
                "transfer_type": transfer_type,
                "train_rows": len(source_train),
                "test_rows": len(target_test),
                "target_labels_used_for_training": False,
                "model": row.pop("baseline"),
            }
        )
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "directional_transfer_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    manifest_path = output_dir / "directional_transfer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "design": "building_chronological",
                "fold": "fold_001",
                "zero_shot": True,
                "pairs": [
                    {"source": source, "target": target, "type": transfer_type}
                    for source, target, transfer_type in TRANSFER_PAIRS
                ],
                "output_file": str(output_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"summary": str(output_path), "manifest": str(manifest_path)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run small zero-shot directional transfer comparison")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    args = parser.parse_args(argv)
    run_directional_transfer(
        args.data_file,
        args.feature_file,
        args.feature_manifest,
        args.splits_file,
        args.output_dir,
        args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
