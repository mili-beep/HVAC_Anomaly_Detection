"""Deterministic cleaning for the canonical multibuilding AHU table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahu_fdd.audit import PLACEHOLDER_TOKENS, sha256_file
from ahu_fdd.multibuilding import load_config
from ahu_fdd.quality_policy import load_duplicate_policy

KEY_COLUMNS = ["building", "ahu", "timestamp"]


def _placeholder_mask(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.isin(PLACEHOLDER_TOKENS) & series.notna()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool, str, int)) or value is None:
        return value
    return str(value)


def _validate_input(frame: pd.DataFrame, sensors: list[str]) -> None:
    required = set(KEY_COLUMNS) | {"label", "source_row_id"} | set(sensors)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical input is missing columns: {sorted(missing)}")


def clean_canonical_frame(
    frame: pd.DataFrame,
    sensors: list[str],
    duplicate_policy: str,
    policy_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Apply deterministic quality handling without fitting an imputer."""
    _validate_input(frame, sensors)
    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    if cleaned["timestamp"].isna().any():
        raise ValueError("Canonical input contains unparseable timestamps")

    placeholder_cells = 0
    invalid_numeric_cells = 0
    nonfinite_cells = 0
    for sensor in sensors:
        raw = cleaned[sensor].copy()
        numeric = pd.to_numeric(raw, errors="coerce")
        nonfinite = numeric.isin([np.inf, -np.inf])
        existing_placeholder = cleaned.get(
            f"has_placeholder__{sensor}", pd.Series(False, index=cleaned.index)
        ).fillna(False).astype(bool)
        placeholder = existing_placeholder | _placeholder_mask(raw)
        invalid_numeric = numeric.isna() & raw.notna() & ~placeholder
        numeric = numeric.mask(nonfinite)
        cleaned[sensor] = numeric
        cleaned[f"is_missing__{sensor}"] = numeric.isna()
        cleaned[f"has_placeholder__{sensor}"] = placeholder
        placeholder_cells += int(placeholder.sum())
        invalid_numeric_cells += int(invalid_numeric.sum())
        nonfinite_cells += int(nonfinite.sum())

    duplicate_mask = cleaned.duplicated(KEY_COLUMNS, keep=False)
    duplicate_key_count = int(
        cleaned.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0]
    )
    duplicate_row_count = int(duplicate_mask.sum())
    cleaned["duplicate_ahu_timestamp"] = duplicate_mask
    cleaned["missing_sensor_count"] = cleaned[
        [f"is_missing__{sensor}" for sensor in sensors]
    ].sum(axis=1)
    cleaned["has_missing_sensor"] = cleaned["missing_sensor_count"].gt(0)
    cleaned["placeholder_sensor_count"] = cleaned[
        [f"has_placeholder__{sensor}" for sensor in sensors]
    ].sum(axis=1)
    cleaned["has_placeholder"] = cleaned["placeholder_sensor_count"].gt(0)
    cleaned = cleaned.sort_values(
        ["building", "ahu", "timestamp", "source_row_id"], kind="stable"
    ).reset_index(drop=True)

    action = policy_config["policies"][duplicate_policy]["action"]
    if action == "exclude_all_rows_for_duplicate_keys":
        output = cleaned.loc[~cleaned["duplicate_ahu_timestamp"]].copy()
    elif action == "retain_all_rows":
        output = cleaned.copy()
    else:  # pragma: no cover - policy loader rejects unknown actions
        raise ValueError(f"Unsupported duplicate policy action: {action}")
    output = output.reset_index(drop=True)
    summary = {
        "policy": duplicate_policy,
        "input_rows": int(len(cleaned)),
        "output_rows": int(len(output)),
        "rows_excluded": int(len(cleaned) - len(output)),
        "duplicate_key_count": duplicate_key_count,
        "duplicate_row_count": duplicate_row_count,
        "placeholder_cells_converted": placeholder_cells,
        "invalid_numeric_cells_converted": invalid_numeric_cells,
        "nonfinite_cells_converted": nonfinite_cells,
        "output_duplicate_key_count": int(
            output.loc[
                output.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS
            ].drop_duplicates().shape[0]
        ),
        "output_missing_sensor_rows": int(output["has_missing_sensor"].sum()),
        "output_placeholder_rows": int(output["has_placeholder"].sum()),
    }
    building_audit = []
    for building, group in cleaned.groupby("building", sort=True):
        output_group = output.loc[output["building"].eq(building)]
        building_audit.append(
            {
                "policy": duplicate_policy,
                "building": building,
                "input_rows": int(len(group)),
                "output_rows": int(len(output_group)),
                "rows_excluded": int(len(group) - len(output_group)),
                "duplicate_key_count": int(
                    group.loc[group["duplicate_ahu_timestamp"], KEY_COLUMNS]
                    .drop_duplicates()
                    .shape[0]
                ),
                "duplicate_row_count": int(group["duplicate_ahu_timestamp"].sum()),
                "missing_sensor_rows": int(output_group["has_missing_sensor"].sum()),
                "placeholder_rows": int(output_group["has_placeholder"].sum()),
            }
        )
    return output, summary, pd.DataFrame(building_audit)


def clean_multibuilding(
    input_file: Path,
    output_dir: Path,
    building_config_path: Path,
    duplicate_policy_path: Path,
) -> dict[str, Any]:
    building_config = load_config(building_config_path)
    policy_config = load_duplicate_policy(duplicate_policy_path)
    sensors = list(building_config["canonical_sensor_columns"])
    frame = pd.read_csv(input_file, low_memory=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[pd.DataFrame] = []
    policy_summaries: dict[str, Any] = {}
    output_files: dict[str, str] = {}
    for policy_name in policy_config["policies"]:
        cleaned, summary, building_audit = clean_canonical_frame(
            frame, sensors, policy_name, policy_config
        )
        filename = (
            "cleaned_ahu.csv"
            if policy_name == "exclude_ambiguous_keys"
            else f"cleaned_ahu_{policy_name}.csv"
        )
        output_path = output_dir / filename
        cleaned.to_csv(output_path, index=False)
        output_files[policy_name] = str(output_path)
        audit_rows.append(building_audit)
        policy_summaries[policy_name] = {
            **summary,
            "building_row_counts": cleaned["building"].value_counts().to_dict(),
            "output_sha256": sha256_file(output_path),
        }

    audit_path = output_dir / "cleaning_audit.csv"
    pd.concat(audit_rows, ignore_index=True).to_csv(audit_path, index=False)
    manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "input_file": str(input_file.expanduser().resolve()),
            "input_sha256": sha256_file(input_file),
            "building_config_file": str(building_config_path.expanduser().resolve()),
            "duplicate_policy_file": str(duplicate_policy_path.expanduser().resolve()),
            "duplicate_key_columns": KEY_COLUMNS,
            "canonical_sensor_columns": sensors,
            "output_files": {**output_files, "audit": str(audit_path)},
            "policies": policy_summaries,
            "imputation": {
                "performed_by_cli": False,
                "fit_scope": "training_fold_only",
            },
        }
    )
    manifest_path = output_dir / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Multibuilding cleaning summary",
        "",
        "No global imputation was performed. Imputation remains training-fold only.",
        "",
        "| Policy | Input rows | Output rows | Excluded rows | Duplicate keys | Output duplicate keys |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in policy_summaries.items():
        lines.append(
            f"| {name} | {summary['input_rows']:,} | {summary['output_rows']:,} | "
            f"{summary['rows_excluded']:,} | {summary['duplicate_key_count']:,} | "
            f"{summary['output_duplicate_key_count']:,} |"
        )
    (output_dir / "cleaning_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return manifest


def _row_changed_mask(frame: pd.DataFrame, sensors: list[str]) -> pd.Series:
    changed = pd.Series(False, index=frame.index)
    for sensor in sensors:
        raw = frame[sensor]
        numeric = pd.to_numeric(raw, errors="coerce")
        placeholder = frame.get(
            f"has_placeholder__{sensor}", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool) | _placeholder_mask(raw)
        nonfinite = numeric.isin([np.inf, -np.inf])
        invalid_numeric = numeric.isna() & raw.notna() & ~placeholder
        changed = changed | placeholder | nonfinite | invalid_numeric
    return changed


def _duplicate_counts(frame: pd.DataFrame) -> tuple[int, int]:
    duplicate_mask = frame.duplicated(KEY_COLUMNS, keep=False)
    keys = frame.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates()
    return int(len(keys)), int(duplicate_mask.sum())


def _verify_output(
    raw: pd.DataFrame,
    output: pd.DataFrame,
    sensors: list[str],
    policy: str,
    policy_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    _validate_input(raw, sensors)
    _validate_input(output, sensors)
    raw = raw.copy()
    output = output.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    if raw["timestamp"].isna().any() or output["timestamp"].isna().any():
        raise ValueError(f"{policy} output contains unparseable timestamps")

    raw_duplicate_mask = raw.duplicated(KEY_COLUMNS, keep=False)
    action = policy_config["policies"][policy]["action"]
    expected = raw.loc[~raw_duplicate_mask].copy() if action.startswith("exclude") else raw
    identity_columns = ["building", "source_row_id"]
    expected_ids = set(map(tuple, expected[identity_columns].to_numpy()))
    output_ids = set(map(tuple, output[identity_columns].to_numpy()))
    errors: list[str] = []
    if expected_ids != output_ids:
        errors.append("output source-row identities do not match the declared policy")
    if output[identity_columns].duplicated().any():
        errors.append("output contains duplicate source-row identities")

    duplicate_key_count, duplicate_row_count = _duplicate_counts(raw)
    output_duplicate_key_count, output_duplicate_row_count = _duplicate_counts(output)
    expected_output_duplicate_keys = 0 if action.startswith("exclude") else duplicate_key_count
    if output_duplicate_key_count != expected_output_duplicate_keys:
        errors.append("output duplicate-key count does not match the declared policy")
    if output["duplicate_ahu_timestamp"].astype(bool).ne(
        output.duplicated(KEY_COLUMNS, keep=False)
    ).any():
        errors.append("duplicate_ahu_timestamp flags are inconsistent")

    if not output.sort_values(KEY_COLUMNS, kind="stable").groupby(
        ["building", "ahu"], sort=False
    )["timestamp"].apply(lambda values: values.is_monotonic_increasing).all():
        errors.append("output is not chronological within an AHU")

    for sensor in sensors:
        numeric = pd.to_numeric(output[sensor], errors="coerce")
        if numeric.notna().any() and not np.isfinite(numeric.dropna()).all():
            errors.append(f"{sensor} contains non-finite output values")
        missing_flag = output[f"is_missing__{sensor}"].astype(bool)
        if missing_flag.ne(numeric.isna()).any():
            errors.append(f"{sensor} missingness flag is inconsistent")
        if _placeholder_mask(output[sensor]).any():
            errors.append(f"{sensor} still contains a literal placeholder")
    expected_labels = expected.groupby(["building", "label"], dropna=False).size().sort_index()
    output_labels = output.groupby(["building", "label"], dropna=False).size().sort_index()
    if not expected_labels.equals(output_labels):
        errors.append("output label counts do not match the declared policy")

    input_changed = _row_changed_mask(raw, sensors)
    output_changed = _row_changed_mask(output, sensors)
    quality_flags = output["has_missing_sensor"].astype(bool) | output[
        "has_placeholder"
    ].astype(bool) | output["duplicate_ahu_timestamp"].astype(bool)
    summary = {
        "policy": policy,
        "input_rows": int(len(raw)),
        "output_rows": int(len(output)),
        "rows_changed": int(input_changed.sum()),
        "rows_changed_retained": int(output_changed.sum()),
        "rows_excluded": int(len(raw) - len(output)),
        "rows_retained_with_quality_flags": int(quality_flags.sum()),
        "rows_imputed": 0,
        "imputed_cells": 0,
        "duplicate_key_count": duplicate_key_count,
        "duplicate_row_count": duplicate_row_count,
        "output_duplicate_key_count": output_duplicate_key_count,
        "output_duplicate_row_count": output_duplicate_row_count,
        "placeholder_cells_converted": int(
            sum(
                raw.get(
                    f"has_placeholder__{sensor}",
                    pd.Series(False, index=raw.index),
                )
                .fillna(False)
                .astype(bool)
                .sum()
                for sensor in sensors
            )
        ),
        "timestamps_valid": True,
        "chronological_within_ahu": not any(
            "output is not chronological within an AHU" == error for error in errors
        ),
        "labels_preserved": not any(
            "output label counts" in error for error in errors
        ),
        "source_rows_match": not any("source-row identities" in error for error in errors),
        "checks_passed": not errors,
        "errors": errors,
    }
    audit_rows = []
    for building, group in raw.groupby("building", sort=True):
        output_group = output.loc[output["building"].eq(building)]
        audit_rows.append(
            {
                "policy": policy,
                "building": building,
                "input_rows": int(len(group)),
                "output_rows": int(len(output_group)),
                "rows_changed": int(_row_changed_mask(group, sensors).sum()),
                "rows_excluded": int(len(group) - len(output_group)),
                "rows_retained_with_quality_flags": int(
                    (
                        output_group["has_missing_sensor"].astype(bool)
                        | output_group["has_placeholder"].astype(bool)
                        | output_group["duplicate_ahu_timestamp"].astype(bool)
                    ).sum()
                ),
                "rows_imputed": 0,
                "imputed_cells": 0,
                "duplicate_key_count": int(
                    group.loc[group.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
                    .drop_duplicates()
                    .shape[0]
                ),
                "duplicate_row_count": int(group.duplicated(KEY_COLUMNS, keep=False).sum()),
            }
        )
    return summary, pd.DataFrame(audit_rows)


def verify_cleaned_multibuilding(
    input_file: Path,
    output_dir: Path,
    building_config_path: Path,
    duplicate_policy_path: Path,
) -> dict[str, Any]:
    """Run the second audit against both saved cleaning-policy outputs."""
    building_config = load_config(building_config_path)
    policy_config = load_duplicate_policy(duplicate_policy_path)
    sensors = list(building_config["canonical_sensor_columns"])
    raw = pd.read_csv(input_file, low_memory=False)
    cleaning_manifest_path = output_dir / "cleaning_manifest.json"
    if not cleaning_manifest_path.is_file():
        raise FileNotFoundError(f"Missing cleaning manifest: {cleaning_manifest_path}")
    cleaning_manifest = json.loads(cleaning_manifest_path.read_text(encoding="utf-8"))
    if cleaning_manifest.get("input_sha256") != sha256_file(input_file):
        raise ValueError("Cleaning input checksum does not match the current canonical table")

    summaries: dict[str, Any] = {}
    audit_frames: list[pd.DataFrame] = []
    for policy in policy_config["policies"]:
        filename = (
            "cleaned_ahu.csv"
            if policy == "exclude_ambiguous_keys"
            else f"cleaned_ahu_{policy}.csv"
        )
        output_path = output_dir / filename
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing cleaned output: {output_path}")
        output = pd.read_csv(output_path, low_memory=False)
        summary, audit = _verify_output(
            raw, output, sensors, policy, policy_config
        )
        if summary["errors"]:
            raise ValueError(f"Second audit failed for {policy}: {summary['errors']}")
        summaries[policy] = {
            **summary,
            "output_file": str(output_path),
            "output_sha256": sha256_file(output_path),
        }
        audit_frames.append(audit)

    audit_path = output_dir / "cleaning_second_audit.csv"
    pd.concat(audit_frames, ignore_index=True).to_csv(audit_path, index=False)
    verification_manifest = _json_safe(
        {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "input_file": str(input_file.expanduser().resolve()),
            "input_sha256": sha256_file(input_file),
            "cleaning_manifest": str(cleaning_manifest_path.resolve()),
            "summaries": summaries,
            "output_files": {
                "audit_csv": str(audit_path),
            },
            "all_checks_passed": True,
        }
    )
    manifest_path = output_dir / "cleaning_verification_manifest.json"
    manifest_path.write_text(
        json.dumps(verification_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Multibuilding cleaning second audit",
        "",
        "All checks passed. No imputation was performed by the cleaning command.",
        "",
        "| Policy | Input rows | Output rows | Changed rows | Excluded rows | Retained with quality flags | Imputed rows | Imputed cells |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries.values():
        lines.append(
            f"| {summary['policy']} | {summary['input_rows']:,} | {summary['output_rows']:,} | "
            f"{summary['rows_changed']:,} | {summary['rows_excluded']:,} | "
            f"{summary['rows_retained_with_quality_flags']:,} | "
            f"{summary['rows_imputed']:,} | {summary['imputed_cells']:,} |"
        )
    (output_dir / "cleaning_second_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return verification_manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply frozen deterministic cleaning policies to canonical AHU data"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/cleaned"))
    parser.add_argument(
        "--building-config", type=Path, default=Path("configs/multibuilding_ahu.yaml")
    )
    parser.add_argument(
        "--duplicate-policy-config",
        type=Path,
        default=Path("configs/duplicate_policies.yaml"),
    )
    args = parser.parse_args(argv)
    manifest = clean_multibuilding(
        args.input_file,
        args.output_dir,
        args.building_config,
        args.duplicate_policy_config,
    )
    print(
        f"Cleaned {manifest['policies']['exclude_ambiguous_keys']['output_rows']:,} "
        f"rows into {args.output_dir.resolve()}"
    )
    return 0


def verify_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the cleaned multibuilding AHU policy outputs"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/cleaned"))
    parser.add_argument(
        "--building-config", type=Path, default=Path("configs/multibuilding_ahu.yaml")
    )
    parser.add_argument(
        "--duplicate-policy-config",
        type=Path,
        default=Path("configs/duplicate_policies.yaml"),
    )
    args = parser.parse_args(argv)
    manifest = verify_cleaned_multibuilding(
        args.input_file,
        args.output_dir,
        args.building_config,
        args.duplicate_policy_config,
    )
    print(
        f"Verified {len(manifest['summaries'])} cleaned policies into "
        f"{args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
