"""Stacked SOH Regressor: SVR + Random Forest + XGBoost with Ridge meta-learner."""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.svm import SVR
from xgboost import XGBRegressor


class StackedSOHRegressor(BaseEstimator, RegressorMixin):
    """Stacked ensemble using cell-grouped out-of-fold predictions for Ridge meta-learning."""

    def __init__(
        self,
        svr_c: float = 10.0,
        svr_gamma: float = 0.01,
        svr_epsilon: float = 0.1,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 5,
        rf_min_samples_leaf: int = 2,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 4,
        xgb_learning_rate: float = 0.05,
        xgb_reg_lambda: float = 1.0,
        ridge_alpha: float = 1.0,
        n_splits: int = 3,
        random_state: int = 42,
        allow_random_kfold: bool = False,
    ):
        self.svr_c = svr_c
        self.svr_gamma = svr_gamma
        self.svr_epsilon = svr_epsilon
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_reg_lambda = xgb_reg_lambda
        self.ridge_alpha = ridge_alpha
        self.n_splits = n_splits
        self.random_state = random_state
        self.allow_random_kfold = allow_random_kfold

    def _make_svr(self) -> SVR:
        return SVR(C=self.svr_c, gamma=self.svr_gamma, epsilon=self.svr_epsilon)

    def _make_rf(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=self.random_state,
        )

    def _make_xgb(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            reg_lambda=self.xgb_reg_lambda,
            random_state=self.random_state,
            verbosity=0,
        )

    def _make_meta_learner(self) -> Ridge:
        return Ridge(alpha=self.ridge_alpha, random_state=self.random_state)

    @property
    def svr(self) -> SVR:
        if hasattr(self, "svr_"):
            return self.svr_
        return self._make_svr()

    @property
    def rf(self) -> RandomForestRegressor:
        if hasattr(self, "rf_"):
            return self.rf_
        return self._make_rf()

    @property
    def xgb(self) -> XGBRegressor:
        if hasattr(self, "xgb_"):
            return self.xgb_
        return self._make_xgb()

    @property
    def meta_learner(self) -> Ridge:
        if hasattr(self, "meta_learner_"):
            return self.meta_learner_
        return self._make_meta_learner()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        cell_indices: Optional[np.ndarray] = None,
    ):
        """
        Fit base learners and meta-learner using strictly out-of-fold (OOF) predictions.
        If cell_indices are provided, GroupKFold is used so the meta-learner learns
        how base models generalize across unseen cells.
        Supports sample_weight weighting across base learners and meta-learner.
        Raises ValueError if cell_indices is None unless allow_random_kfold=True.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        n_samples = len(X)
        self.n_samples_fit_ = n_samples
        self.n_features_in_ = X.shape[1]

        oof_preds = np.zeros((n_samples, 3), dtype=np.float64)

        # 1. Determine splitting strategy (cell-disjoint if cell_indices available)
        use_groups = False
        splits = []
        if cell_indices is not None:
            unique_cells = np.unique(cell_indices)
            if len(unique_cells) >= 2:
                use_groups = True
                effective_splits = min(self.n_splits, len(unique_cells))
                cv = GroupKFold(n_splits=effective_splits)
                splits = list(cv.split(X, y, groups=cell_indices))
            elif not self.allow_random_kfold:
                raise ValueError(
                    f"cell_indices contains only {len(unique_cells)} unique cell(s). "
                    "At least 2 cells are required for cell-disjoint stacking. "
                    "Set allow_random_kfold=True to permit cycle-level KFold."
                )

        if not use_groups:
            if not self.allow_random_kfold and cell_indices is None:
                raise ValueError(
                    "cell_indices must be provided to fit() to enforce cell-disjoint splits and prevent intra-cell "
                    "cycle leakage across folds. Set allow_random_kfold=True to override for synthetic/unlabeled benchmarks."
                )

            effective_splits = min(self.n_splits, n_samples)
            if effective_splits < 2:
                oof_preds[:, 0] = y
                oof_preds[:, 1] = y
                oof_preds[:, 2] = y
                splits = []
            else:
                cv = KFold(n_splits=effective_splits, shuffle=True, random_state=self.random_state)
                splits = list(cv.split(X, y))

        # 2. Generate out-of-fold predictions
        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None

            fold_svr = self._make_svr().fit(X_tr, y_tr, sample_weight=sw_tr)
            fold_rf = self._make_rf().fit(X_tr, y_tr, sample_weight=sw_tr)
            fold_xgb = self._make_xgb().fit(X_tr, y_tr, sample_weight=sw_tr)

            oof_preds[val_idx, 0] = fold_svr.predict(X_val)
            oof_preds[val_idx, 1] = fold_rf.predict(X_val)
            oof_preds[val_idx, 2] = fold_xgb.predict(X_val)

        # 3. Fit meta-learner strictly on out-of-fold predictions
        self.meta_learner_ = self._make_meta_learner()
        self.meta_learner_.fit(oof_preds, y, sample_weight=sample_weight)

        # 4. Refit base learners on the full dataset for deployment
        self.svr_ = self._make_svr().fit(X, y, sample_weight=sample_weight)
        self.rf_ = self._make_rf().fit(X, y, sample_weight=sample_weight)
        self.xgb_ = self._make_xgb().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "meta_learner_"):
            raise RuntimeError("Model has not been fitted yet.")
        base_preds = np.column_stack([self.svr_.predict(X), self.rf_.predict(X), self.xgb_.predict(X)])
        return self.meta_learner_.predict(base_preds)
