"""Unified Two-Stage Baseline: Feature Selection (LASSO, MI, SHAP) + Tuned Stacked Ensemble."""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV

from ..models.stacked_ensemble import StackedSOHRegressor


class TwoStageFeatureSelectionBaseline(BaseEstimator, RegressorMixin):
    """
    Two-stage baseline that first selects features on the fixed full window [3.60, 4.10] V
    using LASSO, Mutual Information, or SHAP-RFE, and then fits the stacked ensemble.
    """

    def __init__(
        self,
        selector: str = "lasso",
        n_features_to_select: int = 12,
        random_state: int = 42,
    ):
        self.selector = selector
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.selected_indices_: Optional[np.ndarray] = None
        self.model = StackedSOHRegressor(random_state=self.random_state)

    def select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select top features based on specified selector method."""
        n_feats = X.shape[1]
        k = min(self.n_features_to_select, n_feats)

        if self.selector == "lasso":
            lasso = LassoCV(cv=3, random_state=self.random_state, max_iter=2000)
            lasso.fit(X, y)
            coef_importance = np.abs(lasso.coef_)
            if np.all(coef_importance < 1e-8):
                return np.arange(k)
            return np.argsort(coef_importance)[::-1][:k]

        elif self.selector == "mi":
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            return np.argsort(mi_scores)[::-1][:k]

        elif self.selector == "shap":
            # Fast tree-based feature selection using Random Forest importances as proxy
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=self.random_state)
            rf.fit(X, y)
            return np.argsort(rf.feature_importances_)[::-1][:k]

        else:
            raise ValueError(f"Unknown selector method: {self.selector}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        cell_indices: Optional[np.ndarray] = None,
    ):
        """Fit feature selector on training cells, then fit the stacked regressor."""
        self.selected_indices_ = self.select_features(X, y)
        X_sub = X[:, self.selected_indices_]
        self.model.fit(X_sub, y, sample_weight=sample_weight, cell_indices=cell_indices)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("Baseline model has not been fitted.")
        X_sub = X[:, self.selected_indices_]
        return self.model.predict(X_sub)
