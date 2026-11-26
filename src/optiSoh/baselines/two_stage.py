"""Unified Two-Stage Baseline: Cell-Grouped Feature Selection (LASSO, MI, SHAP) + Tuned Stacked Ensemble."""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold

from ..models.stacked_ensemble import StackedSOHRegressor
from ..splits.weights import compute_cell_sample_weights


class TwoStageFeatureSelectionBaseline(BaseEstimator, RegressorMixin):
    """
    Two-stage baseline that first selects features on the training-derived reference window
    using LASSO (with GroupKFold), Mutual Information, or RF/SHAP proxy, and then fits the
    cell-grouped, cell-weighted stacked ensemble.
    """

    def __init__(
        self,
        selector: str = "lasso",
        n_features_to_select: int = 12,
        random_state: int = 42,
        allow_random_kfold: bool = False,
    ):
        self.selector = selector
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.allow_random_kfold = allow_random_kfold
        self.selected_indices_: Optional[np.ndarray] = None
        self.model = StackedSOHRegressor(
            random_state=self.random_state,
            allow_random_kfold=self.allow_random_kfold,
        )

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cell_indices: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select top features using cell-grouped CV (for LASSO) and cell-balanced sample weights."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_feats = X.shape[1]
        k = min(self.n_features_to_select, n_feats)

        if self.selector == "lasso":
            if cell_indices is not None:
                groups = np.asarray(cell_indices)
                n_unique = len(np.unique(groups))
                if n_unique < 2:
                    raise ValueError(
                        f"Need at least 2 unique cells in cell_indices for grouped LASSO CV, got {n_unique}."
                    )
                n_splits = min(3, n_unique)
                cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
            elif self.allow_random_kfold:
                cv = 3
            else:
                raise ValueError(
                    "cell_indices must be provided for TwoStageFeatureSelectionBaseline (selector='lasso') "
                    "to prevent cycle-level CV leakage unless allow_random_kfold=True."
                )

            lasso = LassoCV(cv=cv, random_state=self.random_state, max_iter=2000)
            if sample_weight is not None:
                lasso.fit(X, y, sample_weight=sample_weight)
            else:
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
            rf.fit(X, y, sample_weight=sample_weight)
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
        """Fit feature selector on training cells, then fit the cell-grouped stacked regressor."""
        if cell_indices is None and not self.allow_random_kfold:
            raise ValueError(
                "cell_indices must be provided to TwoStageFeatureSelectionBaseline.fit() "
                "to prevent within-cell OOF leakage unless allow_random_kfold=True."
            )

        if sample_weight is None and cell_indices is not None:
            sample_weight = compute_cell_sample_weights(np.asarray(cell_indices))

        self.selected_indices_ = self.select_features(X, y, cell_indices=cell_indices, sample_weight=sample_weight)
        X_sub = np.asarray(X)[:, self.selected_indices_]
        self.model = StackedSOHRegressor(
            random_state=self.random_state,
            allow_random_kfold=self.allow_random_kfold,
        )
        self.model.fit(X_sub, y, sample_weight=sample_weight, cell_indices=cell_indices)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("Baseline model has not been fitted.")
        X_sub = np.asarray(X)[:, self.selected_indices_]
        return self.model.predict(X_sub)
