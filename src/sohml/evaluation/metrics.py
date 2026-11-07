"""Per-cell and banded evaluation metrics."""

import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_err": float(np.max(np.abs(err))),
    }
