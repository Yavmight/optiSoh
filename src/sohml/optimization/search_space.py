"""Decision vector encoding, decoding, and repair operators."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CoOptDecisionVector:
    v_start: float
    v_end: float
    feature_mask: np.ndarray  # Boolean array of length D
    hyperparams: dict


def repair_decision_vector(v_start: float, v_end: float, feature_mask: np.ndarray, min_width: float = 0.15):
    """Enforce minimum window width and ensure at least one feature is active."""
    v_start = float(np.clip(v_start, 3.60, 3.95))
    v_end = float(np.clip(v_end, 3.75, 4.10))
    if v_end - v_start < min_width:
        v_end = min(4.10, v_start + min_width)
    if not np.any(feature_mask):
        feature_mask[0] = True
    return v_start, v_end, feature_mask
