"""Tests for StackedSOHRegressor."""
import numpy as np
from sohml.models.stacked_ensemble import StackedSOHRegressor

def test_stacked_ensemble_fit_predict():
    X = np.random.randn(30, 5)
    y = np.linspace(100, 70, 30)
    model = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)
