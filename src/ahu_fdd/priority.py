"""Create configurable, auditable priority scores from diagnosis predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

REQUIRED_COLUMNS = {
    "building",
    "ahu",
    "timestamp",
    "predicted_type",
    "type_confidence",
    "type_abstained",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def load_policy(path: Path) -> dict[str, Any]:
    """Load and validate the small operational scoring policy."""
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("Priority policy must be a YAML mapping")
    weights = policy.get("weights", {})
    if not isinstance(weights, dict) or set(weights) != {"severity", "duration", "criticality"}:
        raise ValueError("Priority policy must define severity, duration and criticality weights")
    total = sum(float(value) for value in weights.values())
    if abs(total - 1.0) > 1e-9 or any(float(value) < 0 for value in weights.values()):
        raise ValueError("Priority weights must be non-negative and sum to 1")
    for name in ("severity", "duration", "criticality"):
        if not 1 <= float(policy.get(name, {}).get("default", 3)) <= 5:
            raise ValueError(f"{name} default score must be between 1 and 5")
    return policy


def _score_duration(minutes: float, policy: dict[str, Any]) -> int:
    for item in policy["duration"]["bins"]:
        upper = item.get("max_minutes")
        if upper is None or minutes <= float(upper):
            return int(item["score"])
    raise ValueError("Duration policy has no matching bin")


def _score_severity(fault_type: str, policy: dict[str, Any]) -> int:
    settings = policy["severity"]
    return int(settings.get("fault_types", {}).get(fault_type, settings["default"]))


def _score_criticality(building: str, ahu: str, policy: dict[str, Any]) -> int:
    settings = policy["criticality"]
    by_building = settings.get("by_building", {})
    by_ahu = settings.get("by_ahu", {})
    return int(by_ahu.get(f"{building}/{ahu}", by_building.get(building, settings["default"])))


def _band(score: float, policy: dict[str, Any]) -> tuple[str, str]:
    for band in policy["priority_bands"]:
        if score <= float(band["maximum"]):
            return str(band["name"]), str(band["action"])
    raise ValueError(f"Priority score is outside configured bands: {score}")


def group_priority_events(predictions: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """Group consecutive flagged prediction rows into scored operational events."""
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["type_confidence"] = pd.to_numeric(frame["type_confidence"], errors="raise")
    frame["type_abstained"] = frame["type_abstained"].map(_as_bool)
    frame["reported_type"] = frame["predicted_type"].where(~frame["type_abstained"], "Uncertain")
    frame = frame.sort_values(["building", "ahu", "timestamp"]).reset_index(drop=True)
    gap = pd.to_timedelta(float(policy.get("max_gap_minutes", 60)), unit="m")
    keys = ["building", "ahu"]
    context_keys = [column for column in ("design", "fold") if column in frame.columns]
    keys.extend(context_keys)
    previous = frame.groupby(keys, sort=False)["timestamp"].shift()
    starts = previous.isna() | (frame["timestamp"] - previous > gap)
    frame["event_number"] = starts.groupby([frame[key] for key in keys], sort=False).cumsum()

    rows: list[dict[str, Any]] = []
    interval = float(policy.get("sampling_interval_minutes", 60))
    for group_key, group in frame.groupby(keys + ["event_number"], sort=False):
        key_values = dict(zip(keys, group_key[:-1], strict=True))
        building = key_values["building"]
        ahu = key_values["ahu"]
        start = group["timestamp"].min()
        end = group["timestamp"].max()
        duration_minutes = max(interval, (end - start).total_seconds() / 60 + interval)
        known = group.loc[group["reported_type"].ne("Uncertain"), "reported_type"]
        fault_type = str(known.mode().iloc[0]) if not known.empty else "Uncertain"
        confidence = float(group["type_confidence"].mean())
        severity = _score_severity(fault_type, policy)
        duration = _score_duration(duration_minutes, policy)
        criticality = _score_criticality(str(building), str(ahu), policy)
        weights = policy["weights"]
        score = 100 * (
            float(weights["severity"]) * severity / 5
            + float(weights["duration"]) * duration / 5
            + float(weights["criticality"]) * criticality / 5
        )
        band, action = _band(score, policy)
        rows.append(
            {
                "event_id": "/".join([str(key_values[key]) for key in keys] + [str(int(group_key[-1]))]),
                "building": building,
                "ahu": ahu,
                "start_time": start,
                "end_time": end,
                "labelled_duration_minutes": duration_minutes,
                "labelled_rows": len(group),
                "fault_type": fault_type,
                "diagnosis_confidence": confidence,
                "severity_score": severity,
                "duration_score": duration,
                "criticality_score": criticality,
                "priority_score": round(score, 3),
                "priority_band": band,
                "recommended_action": action,
                **{key: key_values[key] for key in context_keys},
            }
        )
    return pd.DataFrame(rows)


def run_priority_scoring(
    predictions_path: Path,
    policy_path: Path,
    output_dir: Path,
    design: str = "chronological",
) -> dict[str, str]:
    """Build scored events from one declared evaluation design."""
    policy = load_policy(policy_path)
    predictions = pd.read_csv(predictions_path, low_memory=False)
    if "design" in predictions and design:
        predictions = predictions.loc[predictions["design"].eq(design)].copy()
    if predictions.empty:
        raise ValueError(f"No prediction rows remain for design {design!r}")
    events = group_priority_events(predictions, policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "priority_events.csv"
    events.to_csv(events_path, index=False)
    summary = events["priority_band"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    summary_path = output_dir / "priority_summary.csv"
    pd.DataFrame({"priority_band": summary.index, "events": summary.values}).to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 1,
        "predictions_file": str(predictions_path.expanduser().resolve()),
        "policy_file": str(policy_path.expanduser().resolve()),
        "design": design,
        "input_rows": int(len(predictions)),
        "event_count": int(len(events)),
        "duration_definition": "labelled event duration: end-start plus the configured sampling interval",
        "limitation": "The dataset labels complete sessions; duration does not establish physical fault onset or physical duration.",
        "output_files": {"events": str(events_path), "summary": str(summary_path)},
    }
    manifest_path = output_dir / "priority_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Operational priority scoring",
        "",
        "Scores combine policy-defined severity, labelled duration and equipment criticality.",
        "They are decision-support outputs, not learned probabilities.",
        "",
        "The source labels complete sessions, so durations are labelled event durations rather than verified physical fault durations.",
        "",
        f"Events: {len(events):,}",
        "",
        "| Priority band | Events |",
        "|---|---:|",
    ]
    markdown.extend(f"| {band} | {int(summary[band])} |" for band in summary.index)
    markdown_path = output_dir / "priority_summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"events": str(events_path), "summary": str(summary_path), "manifest": str(manifest_path), "markdown": str(markdown_path)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create auditable AHU event priority scores")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/operational_priority.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/operational"))
    parser.add_argument("--design", default="chronological")
    args = parser.parse_args(argv)
    result = run_priority_scoring(args.predictions, args.policy, args.output_dir, args.design)
    print(f"Wrote priority outputs to {args.output_dir.resolve()}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
