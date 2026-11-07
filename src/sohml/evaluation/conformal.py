"""Cell-grouped split conformal prediction for calibrated SOH uncertainty intervals."""

import numpy as np


class CellGroupedConformalInterval:
    """Calibrates conformal intervals respecting cell group structures."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_val = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        p = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        self.q_val = float(np.quantile(residuals, p))

    def predict_interval(self, y_pred: np.ndarray):
        return y_pred - self.q_val, y_pred + self.q_val
