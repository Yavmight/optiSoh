"""Tests for raw data loaders."""


def test_loader_instantiation():
    from pathlib import Path

    from sohml.data.nasa_loader import NasaBatteryLoader

    loader = NasaBatteryLoader(raw_dir=Path("data/raw/nasa"))
    assert loader.nominal_capacity == 2.0
