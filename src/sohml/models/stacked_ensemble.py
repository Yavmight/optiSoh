"""Stacked SOH Regressor: SVR + Random Forest + XGBoost with Ridge meta-learner."""
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold

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

        self.svr = SVR(C=self.svr_c, gamma=self.svr_gamma, epsilon=self.svr_epsilon)
        self.rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=42,
        )
        self.xgb = XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            reg_lambda=self.xgb_reg_lambda,
            random_state=42,
            verbosity=0,
        )
        self.meta_learner = Ridge(alpha=self.ridge_alpha)

    def fit(self, X: np.ndarray, y: np.ndarray, cell_indices: Optional[np.ndarray] = None):
        """
        Fit base learners and meta-learner using strictly out-of-fold (OOF) predictions.
        If cell_indices are provided, GroupKFold is used so the meta-learner learns
        how base models generalize across unseen cells.
        """
        n_samples = len(X)
        oof_preds = np.zeros((n_samples, 3), dtype=np.float64)

        # 1. Determine splitting strategy
        use_groups = False
        if cell_indices is not None:
            unique_cells = np.unique(cell_indices)
            if len(unique_cells) >= 2:
                use_groups = True
                effective_splits = min(self.n_splits, len(unique_cells))
                cv = GroupKFold(n_splits=effective_splits)
                splits = list(cv.split(X, y, groups=cell_indices))

        if not use_groups:
            effective_splits = min(self.n_splits, n_samples)
            if effective_splits < 2:
                oof_preds[:, 0] = y
                oof_preds[:, 1] = y
                oof_preds[:, 2] = y
                splits = []
            else:
                cv = KFold(n_splits=effective_splits, shuffle=True, random_state=42)
                splits = list(cv.split(X, y))

        # 2. Generate out-of-fold predictions
        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            fold_svr = clone(self.svr).fit(X_tr, y_tr)
            fold_rf = clone(self.rf).fit(X_tr, y_tr)
            fold_xgb = clone(self.xgb).fit(X_tr, y_tr)

            oof_preds[val_idx, 0] = fold_svr.predict(X_val)
            oof_preds[val_idx, 1] = fold_rf.predict(X_val)
            oof_preds[val_idx, 2] = fold_xgb.predict(X_val)

        # 3. Fit meta-learner strictly on out-of-fold predictions
        self.meta_learner.fit(oof_preds, y)

        # 4. Refit base learners on the full dataset for deployment
        self.svr.fit(X, y)
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack([self.svr.predict(X), self.rf.predict(X), self.xgb.predict(X)])
        return self.meta_learner.predict(base_preds)
