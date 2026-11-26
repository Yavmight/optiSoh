"""Tests for 10 mV quantized window cache with disk persistence."""

import numpy as np
from optiSoh.contracts.schemas import FeatureMatrixBundle
from optiSoh.features.window_cache import WindowFeatureCache, enumerate_quantized_windows


def test_cache_quantization():
    cache = WindowFeatureCache(cache_dir=None, enable_disk=False)
    s, e = cache.quantize(3.604, 4.098)
    assert s == 3600
    assert e == 4100


def test_cache_in_memory_and_disk_persistence(tmp_path):
    cache = WindowFeatureCache(cache_dir=tmp_path / "feat_cache", enable_disk=True)

    dummy_matrix = np.ones((5, 12))
    key = cache.put(3.60, 4.10, dummy_matrix)
    assert key == (3600, 4100)
    assert cache.contains(3.60, 4.10)
    assert len(cache) == 1

    # Verify retrieval from memory
    retrieved = cache.get(3.60, 4.10)
    assert retrieved is not None
    np.testing.assert_array_equal(retrieved, dummy_matrix)

    # Clear memory cache and verify retrieval from disk
    cache.memory_cache.clear()
    assert len(cache.memory_cache) == 0
    retrieved_from_disk = cache.get(3.60, 4.10)
    assert retrieved_from_disk is not None
    np.testing.assert_array_equal(retrieved_from_disk, dummy_matrix)


def test_cache_get_or_compute():
    cache = WindowFeatureCache(cache_dir=None, enable_disk=False)
    computed_counter = [0]

    def compute_fn(v_start: float, v_end: float) -> np.ndarray:
        computed_counter[0] += 1
        return np.full((3, 4), v_end - v_start)

    # First call: computes and stores
    mat1 = cache.get_or_compute(3.65, 4.05, compute_fn)
    assert computed_counter[0] == 1

    # Second call with same quantized window: returns cached without recomputation
    mat2 = cache.get_or_compute(3.652, 4.048, compute_fn)
    assert computed_counter[0] == 1
    np.testing.assert_array_equal(mat1, mat2)


def test_p2_4_feature_matrix_bundle_cache_disk_roundtrip(synthetic_bundle, tmp_path):
    """Verify full FeatureMatrixBundle (X, y_soh, cell_indices, cycle_indices) round-trips via .npz."""
    cache = WindowFeatureCache(cache_dir=tmp_path / "bundle_cache", enable_disk=True)

    # Mark 3 cycles incomplete below 3.80 V to verify aligned metadata filtering
    for c in synthetic_bundle.cycles[:3]:
        low_mask = c.v_grid < 3.80 - 1e-6
        c.q_norm[low_mask] = np.nan
        c.dq_dv[low_mask] = np.nan

    fmb_mem = cache.get_or_compute_bundle(3.604, 4.098, synthetic_bundle.cycles)
    assert isinstance(fmb_mem, FeatureMatrixBundle)
    assert (fmb_mem.window.v_start, fmb_mem.window.v_end) == (3.60, 4.10)
    expected_n = len(synthetic_bundle.cycles) - 3
    assert fmb_mem.X.shape == (expected_n, 42)
    assert fmb_mem.y_soh.shape == (expected_n,)
    assert fmb_mem.cell_indices.shape == (expected_n,)
    assert fmb_mem.cycle_indices.shape == (expected_n,)
    assert np.all(np.isfinite(fmb_mem.X))

    # Reload strictly from disk (.npz) after clearing RAM cache
    cache.clear(clear_disk=False)
    assert len(cache) == 0
    assert cache.contains(3.60, 4.10)

    fmb_disk = cache.get(3.60, 4.10)
    assert isinstance(fmb_disk, FeatureMatrixBundle)
    assert fmb_disk.feature_names == fmb_mem.feature_names
    np.testing.assert_allclose(fmb_disk.X, fmb_mem.X)
    np.testing.assert_allclose(fmb_disk.y_soh, fmb_mem.y_soh)
    np.testing.assert_array_equal(fmb_disk.cell_indices, fmb_mem.cell_indices)
    np.testing.assert_array_equal(fmb_disk.cycle_indices, fmb_mem.cycle_indices)

    # Verify quantized window enumeration count on [3.60, 4.10] V, 150 mV min width, 10 mV step
    windows = enumerate_quantized_windows(3.60, 4.10, 0.15, 10)
    assert len(windows) == 666
    assert windows[0] == (3.60, 3.75)
    assert windows[-1] == (3.95, 4.10)
