"""Fold-isolated scaling and imputation ensuring test cells remain untouched."""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class FoldIsolatedPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor fitted strictly on training cells: Median Imputation + Robust Scaling."""

    def __init__(
        self,
        impute_strategy: str = "median",
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        self.impute_strategy = impute_strategy
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        self.scaler = RobustScaler(with_centering=self.with_centering, with_scaling=self.with_scaling)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit imputer and scaler strictly on training data."""
        self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
        self.scaler_ = RobustScaler(with_centering=self.with_centering, with_scaling=self.with_scaling)
        X_arr = np.asarray(X)
        X_imp = self.imputer_.fit_transform(X_arr)
        self.scaler_.fit(X_imp)

        # Expose references on self.scaler and self.imputer for compatibility
        self.imputer = self.imputer_
        self.scaler = self.scaler_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted imputer and scaler to validation/test data."""
        if not hasattr(self, "scaler_") or not hasattr(self, "imputer_"):
            raise RuntimeError("Preprocessor has not been fitted yet.")
        X_arr = np.asarray(X)
        X_imp = self.imputer_.transform(X_arr)
        return self.scaler_.transform(X_imp)
