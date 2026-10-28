"""Validation for frozen multibuilding data-quality policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

POLICY_NAMES = ("exclude_ambiguous_keys", "retain_flagged_keys")
REQUIRED_POLICY_FIELDS = {
    "action",
    "retain_excluded_rows_in_audit",
    "row_selection",
    "aggregation",
}


def load_duplicate_policy(path: Path) -> dict[str, Any]:
    """Load and validate the duplicate-policy configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The duplicate-policy configuration must be a mapping")
    key_columns = config.get("duplicate_key_columns")
    if key_columns != ["building", "ahu", "timestamp"]:
        raise ValueError(
            "duplicate_key_columns must be exactly building, ahu, timestamp"
        )
    policies = config.get("policies")
    if not isinstance(policies, dict) or set(policies) != set(POLICY_NAMES):
        raise ValueError(
            "policies must define exactly exclude_ambiguous_keys and retain_flagged_keys"
        )
    for name in POLICY_NAMES:
        policy = policies[name]
        if not isinstance(policy, dict) or set(policy) != REQUIRED_POLICY_FIELDS:
            raise ValueError(
                f"{name} must define exactly {sorted(REQUIRED_POLICY_FIELDS)}"
            )
        if policy["row_selection"] != "none" or policy["aggregation"] != "none":
            raise ValueError(f"{name} cannot select or aggregate duplicate rows")
        if policy["retain_excluded_rows_in_audit"] is not True:
            raise ValueError(f"{name} must retain affected rows in the audit")
    if policies["exclude_ambiguous_keys"]["action"] != "exclude_all_rows_for_duplicate_keys":
        raise ValueError("exclude_ambiguous_keys must exclude every row for a duplicate key")
    if policies["retain_flagged_keys"]["action"] != "retain_all_rows":
        raise ValueError("retain_flagged_keys must retain every row")
    return config
