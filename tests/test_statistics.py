"""Tests for evaluation metrics and statistical tests."""

import numpy as np
from sohml.evaluation.metrics import compute_regression_metrics


def test_metrics():
    y_true = np.array([100.0, 90.0, 80.0])
    y_pred = np.array([99.0, 91.0, 80.0])
    m = compute_regression_metrics(y_true, y_pred)
    assert "rmse" in m
    assert m["rmse"] < 1.0
