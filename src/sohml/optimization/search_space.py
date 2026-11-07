"""Decision vector encoding, decoding, and repair operators."""

from dataclasses import dataclass

import numpy as np

from ..contracts.constants import (
    WINDOW_END_MAX,
    WINDOW_END_MIN,
    WINDOW_MIN_WIDTH,
    WINDOW_START_MAX,
    WINDOW_START_MIN,
)


@dataclass
class CoOptDecisionVector:
    v_start: float
    v_end: float
    feature_mask: np.ndarray  # Boolean array of length D
    hyperparams: dict


def repair_decision_vector(
    v_start: float,
    v_end: float,
    feature_mask: np.ndarray,
    min_width: float = WINDOW_MIN_WIDTH,
):
    """Enforce minimum window width and ensure at least one feature is active."""
    v_start = float(np.clip(v_start, WINDOW_START_MIN, WINDOW_START_MAX))
    v_end = float(np.clip(v_end, WINDOW_END_MIN, WINDOW_END_MAX))
    if v_end - v_start < min_width:
        v_end = min(WINDOW_END_MAX, v_start + min_width)
    if not np.any(feature_mask):
        feature_mask[0] = True
    return v_start, v_end, feature_mask
