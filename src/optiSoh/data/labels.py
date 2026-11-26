"""Label alignment, SOH computation, regeneration, and outlier flagging."""

from typing import Optional, Sequence, Tuple

import numpy as np

from .config_resolver import get_preprocessing_spec


def compute_soh_labels(
    capacities: Sequence[float],
    initial_capacity_rule: str = "first_valid",
) -> Tuple[np.ndarray, float]:
    """
    Compute SOH (%) = 100 * (Q_k / Q_0) ensuring zero target leakage into feature space.

    Parameters
    ----------
    capacities : Sequence[float]
        Measured discharge capacities (Ah) ordered chronologically for a single cell.
    initial_capacity_rule : str, default="first_valid"
        How to determine initial capacity Q_0:
        - "first_valid": first strictly positive finite discharge capacity.
        - "max_first_3": maximum among the first 3 valid discharge capacities.

    Returns
    -------
    Tuple[np.ndarray, float]
        (soh_labels_pct, q0_ah)
    """
    caps = np.asarray(capacities, dtype=float)
    valid_mask = np.isfinite(caps) & (caps > 0.0)
    if not np.any(valid_mask):
        raise ValueError("No valid positive discharge capacities found to compute SOH labels.")

    valid_caps = caps[valid_mask]
    if initial_capacity_rule == "first_valid":
        q0 = float(valid_caps[0])
    elif initial_capacity_rule == "max_first_3":
        q0 = float(np.max(valid_caps[:3]))
    else:
        raise ValueError(f"Unknown initial_capacity_rule: {initial_capacity_rule}")

    soh = np.where(valid_mask, 100.0 * (caps / q0), np.nan)
    return soh, q0


def detect_label_outliers(
    soh_labels: Sequence[float],
    window: Optional[int] = None,
    threshold_pct: Optional[float] = None,
) -> np.ndarray:
    """
    Flag isolated SOH capacity outliers deviating from their local rolling median
    by more than `threshold_pct` percentage points (e.g., Oxford Cell2 cyc7400, Cell5 cyc5000).

    Cycles are kept (never dropped) and flagged via `CanonicalCycleGrid.is_label_outlier`.
    """
    spec = get_preprocessing_spec()
    win = int(spec["label_outlier_window"] if window is None else window)
    thresh = float(spec["label_outlier_threshold_pct"] if threshold_pct is None else threshold_pct)

    soh = np.asarray(soh_labels, dtype=float)
    n = len(soh)
    flags = np.zeros(n, dtype=bool)
    if n < 3:
        return flags

    half = max(win // 2, 1)
    for i in range(n):
        if not np.isfinite(soh[i]):
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neigh = soh[lo:hi]
        neigh = neigh[np.isfinite(neigh)]
        if len(neigh) >= 3:
            local_med = float(np.median(neigh))
            if abs(soh[i] - local_med) > thresh:
                flags[i] = True
    return flags


def detect_capacity_regeneration(
    soh_labels: Sequence[float],
    jump_threshold_pct: Optional[float] = None,
    outlier_mask: Optional[Sequence[bool]] = None,
    baseline_window: int = 3,
) -> np.ndarray:
    """
    Flag cycles exhibiting genuine capacity regeneration (e.g., after rest periods)
    while ignoring isolated outlier dips and their immediate recovery back to baseline.

    A non-outlier cycle `i` is flagged if its SOH rises by more than `jump_threshold_pct`
    above both the immediately preceding non-outlier cycle and the rolling median of up to
    `baseline_window` preceding non-outlier cycles.
    """
    thresh = (
        float(get_preprocessing_spec()["regeneration_jump_pct"])
        if jump_threshold_pct is None
        else float(jump_threshold_pct)
    )
    soh = np.asarray(soh_labels, dtype=float)
    n = len(soh)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags

    if outlier_mask is None:
        outliers = detect_label_outliers(soh)
    else:
        outliers = np.asarray(outlier_mask, dtype=bool)

    recent_clean: list[float] = []
    for i in range(n):
        val = soh[i]
        if not np.isfinite(val) or outliers[i]:
            continue
        if recent_clean:
            prev_val = recent_clean[-1]
            prev_med = float(np.median(recent_clean[-baseline_window:]))
            if (val - prev_val > thresh) and (val - prev_med > thresh):
                flags[i] = True
        recent_clean.append(float(val))
    return flags
