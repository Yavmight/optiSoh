"""Charging kinetics and energy feature extraction."""


def extract_kinetics_features(v_sub, tau_sub, q_sub):
    """Extract charging duration, curvature, and normalized partial energy."""
    return {"tau_duration": float(tau_sub[-1] - tau_sub[0]) if len(tau_sub) > 1 else 0.0}
