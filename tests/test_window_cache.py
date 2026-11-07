"""Tests for 10 mV quantized window cache."""

from pathlib import Path

from sohml.features.window_cache import WindowFeatureCache


def test_cache_quantization():
    cache = WindowFeatureCache(cache_dir=Path("data/cache"), step_mv=10)
    s, e = cache.quantize(3.604, 4.098)
    assert s == 3600
    assert e == 4100
