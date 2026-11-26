"""Tests for raw data loaders, QC report, and dataset bundles."""

from pathlib import Path

import numpy as np
import pytest
from optiSoh.data.nasa_audit import is_core_grade_cell
from optiSoh.data.nasa_loader import NasaBatteryLoader
from optiSoh.data.oxford_loader import OxfordBatteryLoader
from optiSoh.data.qc_report import generate_qc_markdown_report
from optiSoh.data.sandia_loader import SandiaBatteryLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
NASA_RAW_DIR = REPO_ROOT / "data" / "raw" / "nasa"
OXFORD_RAW_DIR = REPO_ROOT / "data" / "raw" / "oxford"
SANDIA_RAW_DIR = REPO_ROOT / "data" / "raw" / "sandia"
HAS_NASA_MAT = any(NASA_RAW_DIR.rglob("B0005.mat"))
HAS_OXFORD_MAT = (OXFORD_RAW_DIR / "Oxford_Battery_Degradation_Dataset_1.mat").exists()
HAS_SANDIA_PKL = any(SANDIA_RAW_DIR.rglob("SNL_18650_NCA_25C_0-100_0.5-1C_a.pkl"))


def test_loader_instantiation():
    loader = NasaBatteryLoader(raw_dir=NASA_RAW_DIR)
    assert loader.nominal_capacity == 2.0
    assert loader.eol_soh_threshold == 80.0


def test_sandia_heterogeneous_bundle(tmp_path):
    loader = SandiaBatteryLoader(raw_dir=tmp_path)
    bundle = loader.load_cohort(["NCA", "NMC"])
    assert bundle.name == "sandia_preger_2020"
    assert bundle.get_nominal_capacity("nca_default") == 3.2
    assert bundle.get_nominal_capacity("nmc_default") == 3.0
    assert bundle.get_ambient_temp() == 25.0
    assert bundle.metadata["cycle_index_unit"] == "sequential_cycles"


def test_qc_report_generation(synthetic_bundle, tmp_path):
    out_md = tmp_path / "qc.md"
    report = generate_qc_markdown_report(synthetic_bundle, out_md)
    assert out_md.exists()
    assert "Data Quality Control Report" in report
    assert "Per-Cell SOH Band Distribution" in report
    assert "All cells achieve >=95% coverage" in report
    assert "SYN_01" in report


def test_core_grade_cell_requires_window_coverage():
    # Valid B0005-like cell passes
    assert is_core_grade_cell(24.0, 167, 1.85, 71.4, cov_385_410=166, cov_390_410=166) is True
    # B0034-like cell with poor window coverage (9/196 on [3.85, 4.10] V) must be rejected
    assert is_core_grade_cell(24.0, 196, 1.66, 77.0, cov_385_410=9, cov_390_410=120) is False


@pytest.mark.skipif(not HAS_NASA_MAT, reason="Raw NASA .mat files not present locally")
def test_nasa_loader_real_b0005():
    loader = NasaBatteryLoader(raw_dir=NASA_RAW_DIR)
    bundle = loader.load_cohort(["B0005"])
    assert bundle.name == "nasa_pcoe"
    assert bundle.cell_ids == ["B0005"]
    assert len(bundle.cycles) >= 160
    assert np.isclose(bundle.cycles[0].soh_true, 100.0)
    assert bundle.cycles[-1].soh_true < 75.0
    # Verify post-rest capacity regeneration cycles are kept and flagged
    n_regen = sum(1 for c in bundle.cycles if c.is_regeneration)
    assert n_regen >= 5
    # Verify below-EOL (< 80% SOH) cycles are kept and flagged consistently with eol_soh_threshold
    assert bundle.metadata["eol_soh_threshold"] == 80.0
    assert bundle.metadata["cycle_index_unit"] == "sequential_cc_cycles"
    below_eol = [c for c in bundle.cycles if c.is_below_eol]
    assert len(below_eol) > 0
    assert all(c.soh_true < 80.0 and c.soh_band(80.0) == "<80%" for c in below_eol)
    assert all(c.soh_true >= 80.0 for c in bundle.cycles if not c.is_below_eol)
    # Verify [3.90, 4.10] V has finite curves on post-shipping cycle 1
    cyc = bundle.cycles[1]
    mask = (cyc.v_grid >= 3.90) & (cyc.v_grid <= 4.10)
    assert np.all(np.isfinite(cyc.q_norm[mask]))
    assert np.all(np.isfinite(cyc.dq_dv[mask]))


def test_oxford_loader_instantiation():
    loader = OxfordBatteryLoader(mat_path=OXFORD_RAW_DIR)
    assert loader.nominal_capacity == 0.74
    assert loader.ambient_temp_c == 40.0
    assert loader.eol_soh_threshold == 80.0


@pytest.mark.skipif(not HAS_OXFORD_MAT, reason="Raw Oxford .mat file not present locally")
def test_oxford_loader_real_cell1():
    loader = OxfordBatteryLoader(mat_path=OXFORD_RAW_DIR)
    bundle = loader.load(cell_ids=["Cell1"])
    assert bundle.name == "oxford_degradation_1"
    assert bundle.cell_ids == ["Cell1"]
    assert len(bundle.cycles) == 78
    assert np.isclose(bundle.cycles[0].soh_true, 100.0)
    assert bundle.cycles[-1].soh_true < 80.0
    assert bundle.metadata["eol_soh_threshold"] == 80.0
    assert bundle.metadata["cycle_index_unit"] == "drive_cycles"
    # Verify 100% coverage on [3.60, 4.10] V across all Cell1 cycles
    for cyc in bundle.cycles:
        mask = (cyc.v_grid >= 3.60) & (cyc.v_grid <= 4.10)
        assert np.all(np.isfinite(cyc.q_norm[mask]))
        assert np.all(np.isfinite(cyc.dq_dv[mask]))


@pytest.mark.skipif(not HAS_OXFORD_MAT, reason="Raw Oxford .mat file not present locally")
def test_oxford_label_outliers_flagged_not_dropped():
    loader = OxfordBatteryLoader(mat_path=OXFORD_RAW_DIR)
    bundle = loader.load(cell_ids=["Cell2", "Cell5"])
    c2_outliers = [c.cycle_index for c in bundle.cycles if c.cell_id == "Cell2" and c.is_label_outlier]
    c2_regens = [c.cycle_index for c in bundle.cycles if c.cell_id == "Cell2" and c.is_regeneration]
    c5_outliers = [c.cycle_index for c in bundle.cycles if c.cell_id == "Cell5" and c.is_label_outlier]
    assert 7400 in c2_outliers
    assert 7500 not in c2_regens  # recovery from cyc7400 outlier is not regeneration
    assert 5000 in c5_outliers


@pytest.mark.skipif(not HAS_SANDIA_PKL, reason="Raw Sandia .pkl files not present locally")
def test_sandia_loader_real_nca_and_nmc(tmp_path):
    loader = SandiaBatteryLoader(raw_dir=SANDIA_RAW_DIR)
    target_ids = ["SNL_18650_NCA_25C_0-100_0.5-1C_a", "SNL_18650_NMC_25C_0-100_0.5-2C_b"]
    bundle = loader.load_cohort(cell_ids=target_ids)
    assert bundle.name == "sandia_preger_2020"
    assert set(bundle.cell_ids) == set(target_ids)
    assert bundle.get_nominal_capacity(target_ids[0]) == 3.2
    assert bundle.get_nominal_capacity(target_ids[1]) == 3.0
    assert bundle.metadata["cycle_index_unit"] == "sequential_cycles"
    # Verify zero remaining clone cycles (consecutive or non-consecutive) across harmonized curves
    assert bundle.metadata["per_cell_stats"]["SNL_18650_NMC_25C_0-100_0.5-2C_b"]["n_deduplicated_cycles"] > 0
    for cid in target_ids:
        cc = [c for c in bundle.cycles if c.cell_id == cid]
        sigs = [
            (
                np.nan_to_num(np.round(c.q_norm, 6), nan=-999.0).tobytes(),
                np.nan_to_num(np.round(c.delta_t, 6), nan=-999.0).tobytes(),
                round(c.discharge_capacity, 6),
            )
            for c in cc
        ]
        assert len(set(sigs)) == len(sigs)
    # Verify heterogeneous QC report header
    report = generate_qc_markdown_report(bundle, tmp_path / "snl_qc.md")
    assert "3 Ah, 3.2 Ah (heterogeneous per-cell)" in report
