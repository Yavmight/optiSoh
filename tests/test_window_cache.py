"""Tests for 10 mV quantized window cache with disk persistence."""

import numpy as np
from sohml.features.window_cache import WindowFeatureCache


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
