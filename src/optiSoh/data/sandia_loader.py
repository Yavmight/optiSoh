"""Sandia National Laboratories (SNL Preger 2020) Battery Archive loader."""

import hashlib
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from ..contracts.schemas import CanonicalCycleGrid, DatasetBundle
from .cc_segment import extract_cc_segment
from .config_resolver import get_dataset_spec, get_preprocessing_spec
from .harmonize import build_canonical_v_grid, resample_to_grid
from .labels import compute_soh_labels, detect_capacity_regeneration, detect_label_outliers

_TEMP_PATTERN = re.compile(r"_(\d+)C_")


def _parse_cell_meta_from_id(cell_id: str) -> Tuple[str, float]:
    """Extract chemistry ('NCA'/'NMC'/'LFP') and ambient temperature (deg C) from SNL cell ID."""
    if "_NCA_" in cell_id:
        chem = "NCA"
    elif "_NMC_" in cell_id:
        chem = "NMC"
    elif "_LFP_" in cell_id:
        chem = "LFP"
    else:
        chem = "UNKNOWN"

    m = _TEMP_PATTERN.search(cell_id)
    temp_c = float(m.group(1)) if m else 25.0
    return chem, temp_c


class SandiaBatteryLoader:
    """Loads 18650 NCA and NMC cells cycled over 0-100% SOC with thermocouple logs."""

    CHEMISTRY_NOMINAL_CAPACITIES: Dict[str, float] = {
        "NCA": 3.2,
        "NMC": 3.0,
    }

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.spec = get_dataset_spec("sandia")
        self.prep_spec = get_preprocessing_spec()
        self.nominal_cap_nca = float(self.spec.get("nominal_capacity_nca", 3.2))
        self.nominal_cap_nmc = float(self.spec.get("nominal_capacity_nmc", 3.0))
        self.cc_rate = float(self.spec.get("cc_rate", 0.5))
        self.filter_soc_range = str(self.spec.get("filter_soc_range", "0-100"))
        self.include_temperatures = [float(t) for t in self.spec.get("include_temperatures", [15, 25, 35])]
        self.eol_soh_threshold = float(self.spec.get("eol_soh_threshold", 80.0))

    def _nominal_capacity_for_chemistry(self, chemistry: str) -> float:
        if chemistry == "NCA":
            return self.nominal_cap_nca
        if chemistry == "NMC":
            return self.nominal_cap_nmc
        return self.CHEMISTRY_NOMINAL_CAPACITIES.get(chemistry, 3.0)

    def discover_cell_files(
        self,
        chemistries: Optional[List[str]] = None,
        temperatures: Optional[List[float]] = None,
        cell_ids: Optional[List[str]] = None,
    ) -> List[Path]:
        """Find Sandia .pkl cell files matching chemistry, SOC range, and temperature filters."""
        if not self.raw_dir.exists():
            return []

        allowed_chems = set(chemistries or self.spec.get("chemistries", ["NCA", "NMC"]))
        allowed_temps = set(float(t) for t in (temperatures if temperatures is not None else self.include_temperatures))
        allowed_ids = set(cell_ids) if cell_ids is not None else None

        matched: List[Path] = []
        for p in sorted(self.raw_dir.rglob("*.pkl")):
            cid = p.stem
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if self.filter_soc_range and f"_{self.filter_soc_range}_" not in cid:
                continue
            chem, temp_c = _parse_cell_meta_from_id(cid)
            if chem not in allowed_chems:
                continue
            if temp_c not in allowed_temps:
                continue
            matched.append(p)
        return matched

    def load_cell(
        self,
        pkl_path: Path,
        v_grid: Optional[np.ndarray] = None,
        initial_capacity_rule: str = "first_valid",
    ) -> Tuple[List[CanonicalCycleGrid], Dict]:
        """Parse and harmonize one Sandia SNL .pkl cell file, deduplicating imputed clones and bad-temp cycles."""
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        cell_id = pkl_path.stem
        chem, amb_temp_c = _parse_cell_meta_from_id(cell_id)
        nominal_cap = self._nominal_capacity_for_chemistry(chem)
        target_cc_current = self.cc_rate * nominal_cap

        if v_grid is None:
            v_grid = build_canonical_v_grid()

        raw_cycles = sorted(obj.get("cycle_data", []), key=lambda c: int(c.get("cycle_number", 0)))

        paired_records = []
        n_deduplicated = 0
        dropped_bad_temp = 0
        dropped_cc = 0
        seen_full_sigs: set = set()
        seen_qt_sigs: set = set()
        seen_cc_sigs: set = set()

        for cyc in raw_cycles:
            v = np.asarray(cyc.get("voltage_in_V", []), dtype=float)
            cur = np.asarray(cyc.get("current_in_A", []), dtype=float)
            qch = np.asarray(cyc.get("charge_capacity_in_Ah", []), dtype=float)
            qdc = np.asarray(cyc.get("discharge_capacity_in_Ah", []), dtype=float)
            t = np.asarray(cyc.get("time_in_s", []), dtype=float)
            temp = np.asarray(cyc.get("temperature_in_C", []), dtype=float)

            if len(v) < 20 or len(qdc) == 0 or not np.any(np.isfinite(qdc)):
                dropped_cc += 1
                continue

            cap_ah = float(np.nanmax(qdc))
            if not (np.isfinite(cap_ah) and cap_ah > 0.1):
                dropped_cc += 1
                continue

            # Deduplicate exact full-cycle clones (consecutive or non-consecutive) from BatteryML imputation
            v_clean = np.ascontiguousarray(np.nan_to_num(v, nan=-999.0))
            full_sig = (hashlib.sha1(v_clean.tobytes()).digest(), round(cap_ah, 6))
            if full_sig in seen_full_sigs:
                n_deduplicated += 1
                continue
            seen_full_sigs.add(full_sig)

            # Deduplicate BatteryML clones where charge capacity + thermocouple arrays are identical
            if len(qch) == len(v) and len(temp) == len(v):
                qt_hash = hashlib.sha1(
                    np.ascontiguousarray(np.round(np.nan_to_num(qch, nan=-999.0), 6)).tobytes()
                    + np.ascontiguousarray(np.round(np.nan_to_num(temp, nan=-999.0), 6)).tobytes()
                ).digest()
                qt_sig = (qt_hash, round(cap_ah, 6))
                if qt_sig in seen_qt_sigs:
                    n_deduplicated += 1
                    continue
                seen_qt_sigs.add(qt_sig)

            cc_slice = extract_cc_segment(v, cur, target_current=target_cc_current)
            if cc_slice is None or len(qch) != len(v) or len(temp) != len(v) or len(t) != len(v):
                dropped_cc += 1
                continue

            temp_cc = temp[cc_slice]
            finite_temp_frac = float(np.mean(np.isfinite(temp_cc)))
            if finite_temp_frac < 0.80 or float(np.nanstd(temp_cc)) < 1e-6:
                dropped_bad_temp += 1
                continue

            # Also deduplicate CC-segment clones where only CV/discharge transition samples differed
            cc_hash = hashlib.sha1(
                np.ascontiguousarray(np.round(v[cc_slice], 6)).tobytes()
                + np.ascontiguousarray(np.round(qch[cc_slice], 6)).tobytes()
                + np.ascontiguousarray(np.round(np.nan_to_num(temp_cc, nan=-999.0), 6)).tobytes()
            ).digest()
            cc_sig = (cc_hash, round(cap_ah, 6))
            if cc_sig in seen_cc_sigs:
                n_deduplicated += 1
                continue
            seen_cc_sigs.add(cc_sig)

            cyc_num = int(cyc.get("cycle_number", len(paired_records)))
            paired_records.append((cyc_num, v[cc_slice], qch[cc_slice], temp_cc, t[cc_slice], cap_ah))

        if not paired_records:
            return [], {
                "cell_id": cell_id,
                "chemistry": chem,
                "ambient_temp_c": amb_temp_c,
                "nominal_capacity": nominal_cap,
                "n_valid_cycles": 0,
                "n_deduplicated_cycles": n_deduplicated,
                "dropped_bad_temp_cycles": dropped_bad_temp,
                "dropped_cc_cycles": dropped_cc,
                "n_regeneration_cycles": 0,
                "n_label_outliers": 0,
                "n_below_eol": 0,
            }

        discharge_caps = [rec[5] for rec in paired_records]
        soh_labels, q0 = compute_soh_labels(discharge_caps, initial_capacity_rule=initial_capacity_rule)
        outlier_flags = detect_label_outliers(soh_labels)
        regen_flags = detect_capacity_regeneration(soh_labels, outlier_mask=outlier_flags)

        canonical_cycles: List[CanonicalCycleGrid] = []
        v_cc_mins: List[float] = []
        v_cc_maxs: List[float] = []

        for idx, (cyc_num, v_cc, q_cc, temp_cc, t_cc, cap_ah) in enumerate(paired_records):
            grid_data = resample_to_grid(
                voltage=v_cc,
                capacity=q_cc,
                temp=temp_cc,
                time=t_cc,
                nominal_capacity=nominal_cap,
                v_grid=v_grid,
                c_rate=self.cc_rate,
            )

            if not np.any(np.isfinite(grid_data["q_norm"])):
                dropped_cc += 1
                continue

            v_cc_mins.append(float(np.min(v_cc)))
            v_cc_maxs.append(float(np.max(v_cc)))
            soh_val = float(soh_labels[idx])

            canonical_cycles.append(
                CanonicalCycleGrid(
                    cell_id=cell_id,
                    cycle_index=cyc_num,
                    v_grid=grid_data["v_grid"],
                    q_norm=grid_data["q_norm"],
                    dq_dv=grid_data["dq_dv"],
                    delta_t=grid_data["delta_t"],
                    dt_dv=grid_data["dt_dv"],
                    tau=grid_data["tau"],
                    soh_true=soh_val,
                    discharge_capacity=float(cap_ah),
                    nominal_capacity=nominal_cap,
                    is_regeneration=bool(regen_flags[idx]),
                    is_below_eol=bool(soh_val < self.eol_soh_threshold),
                    is_label_outlier=bool(outlier_flags[idx]),
                )
            )

        band_counts: Dict[str, int] = {}
        for c in canonical_cycles:
            b = c.soh_band(self.eol_soh_threshold)
            band_counts[b] = band_counts.get(b, 0) + 1

        cell_meta = {
            "cell_id": cell_id,
            "source_file": str(pkl_path),
            "chemistry": chem,
            "ambient_temp_c": amb_temp_c,
            "nominal_capacity": nominal_cap,
            "n_valid_cycles": len(canonical_cycles),
            "n_deduplicated_cycles": n_deduplicated,
            "dropped_bad_temp_cycles": dropped_bad_temp,
            "dropped_cc_cycles": dropped_cc,
            "n_regeneration_cycles": int(sum(c.is_regeneration for c in canonical_cycles)),
            "n_label_outliers": int(sum(c.is_label_outlier for c in canonical_cycles)),
            "n_below_eol": int(sum(c.is_below_eol for c in canonical_cycles)),
            "soh_band_counts": band_counts,
            "q0_initial_ah": float(q0),
            "q_final_ah": float(discharge_caps[-1]),
            "soh_final_pct": float(soh_labels[-1]),
            "v_cc_min_median": float(np.median(v_cc_mins)) if v_cc_mins else float("nan"),
            "v_cc_min_p95": float(np.quantile(v_cc_mins, 0.95)) if v_cc_mins else float("nan"),
            "v_cc_max_min": float(np.min(v_cc_maxs)) if v_cc_maxs else float("nan"),
        }
        return canonical_cycles, cell_meta

    def load_cohort(
        self,
        chemistries: Optional[List[str]] = None,
        temperatures: Optional[List[float]] = None,
        cell_ids: Optional[List[str]] = None,
        initial_capacity_rule: str = "first_valid",
        n_jobs: int = -1,
    ) -> DatasetBundle:
        """Filter, harmonize, and return full-range cycled NCA/NMC Sandia cells."""
        default_temp = float(self.spec.get("temperature_c", 25.0))
        cell_capacities: Dict[str, float] = {
            "nca_default": self.nominal_cap_nca,
            "nmc_default": self.nominal_cap_nmc,
        }
        cell_temps: Dict[str, float] = {"default": default_temp}
        cell_chemistries: Dict[str, str] = {}
        per_cell_meta: Dict[str, Dict] = {}
        loaded_ids: List[str] = []
        all_cycles: List[CanonicalCycleGrid] = []

        v_grid = build_canonical_v_grid()
        cell_files = self.discover_cell_files(
            chemistries=chemistries,
            temperatures=temperatures,
            cell_ids=cell_ids,
        )

        if len(cell_files) > 2 and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.load_cell)(pkl_path, v_grid=v_grid, initial_capacity_rule=initial_capacity_rule)
                for pkl_path in cell_files
            )
        else:
            results = [
                self.load_cell(pkl_path, v_grid=v_grid, initial_capacity_rule=initial_capacity_rule)
                for pkl_path in cell_files
            ]

        for cycles, meta in results:
            cid = meta["cell_id"]
            if not cycles:
                per_cell_meta[cid] = meta
                continue
            loaded_ids.append(cid)
            all_cycles.extend(cycles)
            cell_capacities[cid] = meta["nominal_capacity"]
            cell_temps[cid] = meta["ambient_temp_c"]
            cell_chemistries[cid] = meta["chemistry"]
            per_cell_meta[cid] = meta

        return DatasetBundle(
            name="sandia_preger_2020",
            cell_ids=loaded_ids,
            cycles=all_cycles,
            nominal_capacity=self.nominal_cap_nca,
            ambient_temp_c=default_temp,
            cell_nominal_capacities=cell_capacities,
            cell_ambient_temps=cell_temps,
            metadata={
                "cycle_index_unit": "sequential_cycles",
                "cc_rate": self.cc_rate,
                "eol_soh_threshold": self.eol_soh_threshold,
                "initial_capacity_rule": initial_capacity_rule,
                "allowed_chemistries": chemistries or list(self.spec.get("chemistries", ["NCA", "NMC"])),
                "supported_temperatures_c": temperatures
                if temperatures is not None
                else list(self.include_temperatures),
                "cell_chemistries": cell_chemistries,
                "total_deduplicated_cycles": int(
                    sum(m.get("n_deduplicated_cycles", 0) for m in per_cell_meta.values())
                ),
                "total_dropped_bad_temp_cycles": int(
                    sum(m.get("dropped_bad_temp_cycles", 0) for m in per_cell_meta.values())
                ),
                "preprocessing_config": dict(self.prep_spec),
                "per_cell_stats": per_cell_meta,
            },
        )
