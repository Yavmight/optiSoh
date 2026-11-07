"""Tests for raw data loaders and dataset bundles."""

from pathlib import Path

from sohml.data.nasa_loader import NasaBatteryLoader
from sohml.data.sandia_loader import SandiaBatteryLoader

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_loader_instantiation():
    loader = NasaBatteryLoader(raw_dir=REPO_ROOT / "data" / "raw" / "nasa")
    assert loader.nominal_capacity == 2.0


def test_sandia_heterogeneous_bundle():
    loader = SandiaBatteryLoader(raw_dir=REPO_ROOT / "data" / "raw" / "sandia")
    bundle = loader.load_cohort(["NCA", "NMC"])
    assert bundle.name == "sandia_preger_2020"
    assert bundle.get_nominal_capacity("nca_default") == 3.2
    assert bundle.get_nominal_capacity("nmc_default") == 3.0
    assert bundle.get_ambient_temp() == 25.0
