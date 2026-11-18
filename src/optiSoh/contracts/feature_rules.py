"""Enforces feature hygiene and forbids label/leakage proxies."""

from typing import Sequence

FORBIDDEN_FEATURES = {
    "cycle_index",
    "cycle",
    "cycle_number",
    "time",
    "timestamp",
    "time_elapsed",
    "cumulative_ah",
    "cumulative_capacity",
    "cell_id",
    "battery_id",
    "dataset",
    "dataset_name",
    "initial_capacity",
    "q0",
}


def validate_feature_names(feature_names: Sequence[str]) -> None:
    """Raise ValueError if any forbidden proxy feature is present in the feature set."""
    for name in feature_names:
        norm = name.lower().strip()
        if norm in FORBIDDEN_FEATURES:
            raise ValueError(f"Feature '{name}' is strictly forbidden as it trivially leaks SOH or domain identity.")
