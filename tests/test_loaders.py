"""Tests for raw data loaders."""
def test_loader_instantiation():
    from sohml.data.nasa_loader import NasaBatteryLoader
    from pathlib import Path
    loader = NasaBatteryLoader(raw_dir=Path("data/raw/nasa"))
    assert loader.nominal_capacity == 2.0
