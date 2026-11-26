"""Audits all 34 NASA battery files to determine valid 1.5A CC charging cells."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .nasa_loader import NasaBatteryLoader


def is_core_grade_cell(
    ambient_temp_c: float,
    n_valid_cycles: int,
    q0_initial_ah: float,
    soh_final_pct: float,
    cov_385_410: int,
    cov_390_410: int,
) -> bool:
    """
    Determine whether a NASA cell meets room-temperature core-grade criteria:
    - ~24 deg C ambient temperature
    - >= 100 valid CC charge-discharge pairs
    - Healthy initial capacity (>= 1.5 Ah) and real degradation (final SOH <= 85%)
    - Strong partial-window coverage (>= 80% on [3.85, 4.10] V and >= 95% on [3.90, 4.10] V)
    """
    if n_valid_cycles < 100:
        return False
    if not (np.isfinite(ambient_temp_c) and abs(ambient_temp_c - 24.0) <= 2.0):
        return False
    if not (np.isfinite(q0_initial_ah) and q0_initial_ah >= 1.5):
        return False
    if not (np.isfinite(soh_final_pct) and soh_final_pct <= 85.0):
        return False
    cov_385_frac = cov_385_410 / n_valid_cycles
    cov_390_frac = cov_390_410 / n_valid_cycles
    return cov_385_frac >= 0.80 and cov_390_frac >= 0.95


def audit_nasa_archive(raw_dir: Union[str, Path]) -> pd.DataFrame:
    """Scan all raw NASA .mat files and return a summary DataFrame of usable CC cycles and coverage."""
    raw_path = Path(raw_dir)
    mat_files = sorted(raw_path.rglob("*.mat"))
    loader = NasaBatteryLoader(raw_dir=raw_path)

    rows = []
    for mat_file in mat_files:
        cid = mat_file.stem
        cycles, meta = loader.load_cell(cid)

        # Compute coverage across key candidate partial-charging windows
        cov_360_410 = 0
        cov_380_410 = 0
        cov_385_410 = 0
        cov_390_410 = 0
        for cyc in cycles:
            m360 = (cyc.v_grid >= 3.60 - 1e-6) & (cyc.v_grid <= 4.10 + 1e-6)
            m380 = (cyc.v_grid >= 3.80 - 1e-6) & (cyc.v_grid <= 4.10 + 1e-6)
            m385 = (cyc.v_grid >= 3.85 - 1e-6) & (cyc.v_grid <= 4.10 + 1e-6)
            m390 = (cyc.v_grid >= 3.90 - 1e-6) & (cyc.v_grid <= 4.10 + 1e-6)
            if np.all(np.isfinite(cyc.q_norm[m360])) and np.all(np.isfinite(cyc.dq_dv[m360])):
                cov_360_410 += 1
            if np.all(np.isfinite(cyc.q_norm[m380])) and np.all(np.isfinite(cyc.dq_dv[m380])):
                cov_380_410 += 1
            if np.all(np.isfinite(cyc.q_norm[m385])) and np.all(np.isfinite(cyc.dq_dv[m385])):
                cov_385_410 += 1
            if np.all(np.isfinite(cyc.q_norm[m390])) and np.all(np.isfinite(cyc.dq_dv[m390])):
                cov_390_410 += 1

        n_cyc = meta["n_valid_cycles"]
        q0 = meta.get("q0_initial_ah", float("nan"))
        q_end = meta.get("q_final_ah", float("nan"))
        soh_end = meta.get("soh_final_pct", float("nan"))
        amb_t = meta.get("ambient_temp_c", float("nan"))

        core_grade = is_core_grade_cell(
            ambient_temp_c=amb_t,
            n_valid_cycles=n_cyc,
            q0_initial_ah=q0,
            soh_final_pct=soh_end,
            cov_385_410=cov_385_410,
            cov_390_410=cov_390_410,
        )

        rows.append(
            {
                "cell_id": cid,
                "ambient_temp_c": amb_t,
                "n_valid_cycles": n_cyc,
                "n_regeneration_cycles": meta.get("n_regeneration_cycles", 0),
                "n_below_eol": meta.get("n_below_eol", 0),
                "pct_below_eol": round(100.0 * meta.get("n_below_eol", 0) / n_cyc, 1) if n_cyc > 0 else 0.0,
                "dropped_charge_ops": meta["dropped_charge_ops"],
                "q0_initial_ah": round(q0, 4) if np.isfinite(q0) else np.nan,
                "q_final_ah": round(q_end, 4) if np.isfinite(q_end) else np.nan,
                "soh_final_pct": round(soh_end, 2) if np.isfinite(soh_end) else np.nan,
                "v_cc_min_median": round(meta.get("v_cc_min_median", float("nan")), 4),
                "v_cc_min_p95": round(meta.get("v_cc_min_p95", float("nan")), 4),
                "v_cc_max_min": round(meta.get("v_cc_max_min", float("nan")), 4),
                "cov_360_410": cov_360_410,
                "cov_380_410": cov_380_410,
                "cov_385_410": cov_385_410,
                "cov_390_410": cov_390_410,
                "cov_385_410_pct": round(100.0 * cov_385_410 / n_cyc, 1) if n_cyc > 0 else 0.0,
                "cov_390_410_pct": round(100.0 * cov_390_410 / n_cyc, 1) if n_cyc > 0 else 0.0,
                "core_grade": core_grade,
            }
        )

    return pd.DataFrame(rows)
