"""Canary test verifying zero data leakage from held-out cell."""
import numpy as np
from sohml.splits.preprocessing import FoldIsolatedPreprocessor

def test_zero_leakage_canary():
    # Training cells data
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    scaler = FoldIsolatedPreprocessor()
    scaler.fit(X_train)
    
    # Replacing test cell with 1e6 outlier MUST NOT affect fitted training scaler
    center_before = scaler.scaler.center_.copy()
    scale_before = scaler.scaler.scale_.copy()
    
    X_test_sentinel = np.array([[1e6, 1e6]])
    # Test cell is transformed, never fit
    _ = scaler.transform(X_test_sentinel)
    
    assert np.allclose(scaler.scaler.center_, center_before)
    assert np.allclose(scaler.scaler.scale_, scale_before)
