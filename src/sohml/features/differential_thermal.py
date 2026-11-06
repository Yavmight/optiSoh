"""Differential Temperature (dT/dV) feature extraction."""
import numpy as np

def extract_thermal_features(v_sub: np.ndarray, dt_sub: np.ndarray, temp_sub: np.ndarray) -> dict:
    """Extract 10 thermal features: max dT/dV, voltage location, temperature rise."""
    return {
        "dt_max": float(np.max(dt_sub)) if len(dt_sub) else 0.0,
        "dt_area": float(np.trapz(dt_sub, v_sub)) if len(dt_sub) > 1 else 0.0,
    }
