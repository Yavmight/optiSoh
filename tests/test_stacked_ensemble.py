"""Tests for StackedSOHRegressor and model complexity calculation."""

import numpy as np
from sklearn.base import clone
from sohml.models.complexity import compute_model_complexity
from sohml.models.stacked_ensemble import StackedSOHRegressor


def test_stacked_ensemble_fit_predict():
    X = np.random.randn(30, 5)
    y = np.linspace(100, 70, 30)
    model = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)


def test_stacked_ensemble_grouped_splits():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 4)
    y = np.linspace(100, 80, 40)
    cell_indices = np.repeat(["Cell_A", "Cell_B", "Cell_C", "Cell_D"], 10)

    model = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10, n_splits=4, random_state=42)
    model.fit(X, y, cell_indices=cell_indices)
    preds = model.predict(X)
    assert preds.shape == (40,)


def test_stacked_ensemble_sample_weight():
    X = np.random.randn(30, 5)
    y = np.linspace(100, 70, 30)
    weights = np.ones(30)
    weights[:5] = 5.0

    model = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10, random_state=42)
    model.fit(X, y, sample_weight=weights)
    preds = model.predict(X)
    assert preds.shape == (30,)


def test_stacked_ensemble_set_params_and_clone():
    model = StackedSOHRegressor(rf_n_estimators=15, rf_max_depth=4)
    model.set_params(rf_max_depth=7, ridge_alpha=2.5)
    assert model.rf_max_depth == 7
    assert model.ridge_alpha == 2.5

    # Test cloning compatibility
    cloned = clone(model)
    assert cloned.rf_max_depth == 7
    assert cloned.ridge_alpha == 2.5


def test_compute_model_complexity():
    X = np.random.randn(50, 4)
    y = np.linspace(100, 80, 50)
    model = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10, random_state=42)
    model.fit(X, y)

    # Complexity should be normalized in [0, 1]
    comp = compute_model_complexity(model, n_train_samples=len(X))
    assert 0.0 <= comp <= 1.0

    # Fallback when n_train_samples is None (inferred from model.n_samples_fit_)
    comp_inferred = compute_model_complexity(model)
    assert 0.0 <= comp_inferred <= 1.0
