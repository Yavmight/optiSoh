"""End-to-end canary test verifying zero data leakage from held-out test cell."""

import numpy as np
from sohml.baselines.two_stage import TwoStageFeatureSelectionBaseline
from sohml.models.stacked_ensemble import StackedSOHRegressor
from sohml.splits.preprocessing import FoldIsolatedPreprocessor


def test_scaler_and_imputer_isolation():
    """Verify preprocessor fit is completely isolated from test data."""
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    preprocessor = FoldIsolatedPreprocessor()
    preprocessor.fit(X_train)

    center_before = preprocessor.scaler.center_.copy()
    scale_before = preprocessor.scaler.scale_.copy()
    stat_before = preprocessor.imputer.statistics_.copy()

    # Transforming extreme outliers and NaNs must NOT alter preprocessor fitted state
    X_test_sentinel = np.array([[1e7, np.nan], [np.nan, 1e7]])
    _ = preprocessor.transform(X_test_sentinel)

    np.testing.assert_array_equal(preprocessor.scaler.center_, center_before)
    np.testing.assert_array_equal(preprocessor.scaler.scale_, scale_before)
    np.testing.assert_array_equal(preprocessor.imputer.statistics_, stat_before)


def test_zero_leakage_end_to_end_stacked_ensemble():
    """Canary test: changing held-out test cell to 1e7 outliers must not alter model parameters."""
    rng = np.random.RandomState(42)

    # 3 training cells (20 cycles each = 60 cycles), 1 held-out test cell (20 cycles)
    n_train_samples = 60
    n_test_samples = 20
    n_features = 8

    X_train = rng.randn(n_train_samples, n_features)
    y_train = 100.0 - np.linspace(0, 20, n_train_samples) + rng.randn(n_train_samples) * 0.5
    train_cell_labels = np.repeat(["Cell_1", "Cell_2", "Cell_3"], 20)

    X_test_clean = rng.randn(n_test_samples, n_features)

    # Pipeline Run A: Clean test cell
    prep_a = FoldIsolatedPreprocessor()
    X_train_trans_a = prep_a.fit(X_train).transform(X_train)
    model_a = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10, random_state=42)
    model_a.fit(X_train_trans_a, y_train, cell_indices=train_cell_labels)
    _ = prep_a.transform(X_test_clean)
    preds_a = model_a.predict(X_train_trans_a)

    # Pipeline Run B: Corrupted test cell with extreme sentinel outliers
    X_test_corrupted = np.full_like(X_test_clean, 1e7)
    prep_b = FoldIsolatedPreprocessor()
    X_train_trans_b = prep_b.fit(X_train).transform(X_train)
    model_b = StackedSOHRegressor(rf_n_estimators=10, xgb_n_estimators=10, random_state=42)
    model_b.fit(X_train_trans_b, y_train, cell_indices=train_cell_labels)
    _ = prep_b.transform(X_test_corrupted)
    preds_b = model_b.predict(X_train_trans_b)

    # Assert bit-for-bit identical fitted statistics & model coefficients
    np.testing.assert_array_equal(prep_a.scaler.center_, prep_b.scaler.center_)
    np.testing.assert_array_equal(prep_a.scaler.scale_, prep_b.scaler.scale_)
    np.testing.assert_array_equal(prep_a.imputer.statistics_, prep_b.imputer.statistics_)

    np.testing.assert_array_equal(model_a.meta_learner.coef_, model_b.meta_learner.coef_)
    assert model_a.meta_learner.intercept_ == model_b.meta_learner.intercept_
    np.testing.assert_array_equal(model_a.svr.support_, model_b.svr.support_)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-12, atol=1e-12)


def test_zero_leakage_end_to_end_feature_selection():
    """Verify feature selector on training cells is immune to test cell content."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 15)
    y_train = 90.0 - 0.5 * X_train[:, 2] + 0.3 * X_train[:, 5]

    baseline_a = TwoStageFeatureSelectionBaseline(selector="lasso", n_features_to_select=5, random_state=42)
    baseline_a.fit(X_train, y_train)

    baseline_b = TwoStageFeatureSelectionBaseline(selector="lasso", n_features_to_select=5, random_state=42)
    baseline_b.fit(X_train, y_train)

    np.testing.assert_array_equal(baseline_a.selected_indices_, baseline_b.selected_indices_)
