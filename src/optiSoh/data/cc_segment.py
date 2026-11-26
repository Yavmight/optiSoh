"""Constant-current (CC) segment detection and quality control."""

from typing import Optional

import numpy as np

from .config_resolver import get_preprocessing_spec


def extract_cc_segment(
    voltage: np.ndarray,
    current: np.ndarray,
    target_current: float,
    tolerance: Optional[float] = None,
    min_points: Optional[int] = None,
    min_voltage_span: Optional[float] = None,
    v_min_valid: Optional[float] = None,
    v_max_valid: Optional[float] = None,
) -> Optional[slice]:
    """
    Detect the longest contiguous constant-current (CC) charging segment and verify quality.
    Any omitted parameters default to the fixed a-priori values in `configs/default.yaml`.
    """
    spec = get_preprocessing_spec()
    tol = float(spec["cc_current_tolerance"] if tolerance is None else tolerance)
    min_pts = int(spec["cc_min_points"] if min_points is None else min_points)
    min_span = float(spec["cc_min_voltage_span"] if min_voltage_span is None else min_voltage_span)
    v_lo = float(spec["cc_v_min_valid"] if v_min_valid is None else v_min_valid)
    v_hi = float(spec["cc_v_max_valid"] if v_max_valid is None else v_max_valid)

    v = np.asarray(voltage, dtype=float)
    cur = np.asarray(current, dtype=float)

    if v.ndim != 1 or cur.ndim != 1 or len(v) != len(cur) or len(v) < min_pts:
        return None

    abs_tol = max(abs(target_current) * tol, 0.02)
    mask = np.isfinite(v) & np.isfinite(cur) & (v >= v_lo) & (v <= v_hi) & (np.abs(cur - target_current) <= abs_tol)

    if int(np.sum(mask)) < min_pts:
        return None

    # Find longest contiguous True run
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts

    best_idx = int(np.argmax(lengths))
    start = int(starts[best_idx])
    end = int(ends[best_idx])

    if (end - start) < min_pts:
        return None

    # Trim any trailing flat/falling points after voltage reaches its peak (CC -> CV transition)
    v_seg = v[start:end]
    peak_offset = int(np.argmax(v_seg))
    if peak_offset + 1 >= min_pts:
        end = start + peak_offset + 1
        v_seg = v[start:end]

    if float(v_seg[-1] - v_seg[0]) < min_span:
        return None

    return slice(start, end)
