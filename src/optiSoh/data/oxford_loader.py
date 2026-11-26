"""Oxford Battery Degradation Dataset 1 loader."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from ..contracts.schemas import CanonicalCycleGrid, DatasetBundle
from .cc_segment import extract_cc_segment
from .config_resolver import get_dataset_spec, get_preprocessing_spec
from .harmonize import build_canonical_v_grid, resample_to_grid
from .labels import compute_soh_labels, detect_capacity_regeneration, detect_label_outliers


def _parse_cycle_number(cyc_key: str) -> int:
    """Convert Oxford cycle key like 'cyc0100' into integer 100."""
    digits = "".join(ch for ch in cyc_key if ch.isdigit())
    return int(digits) if digits else 0


def _estimate_current_a(q_ah: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Estimate instantaneous charging current (A) = dQ(Ah)/dt(h) from cumulative charge."""
    if len(q_ah) < 2:
        return np.zeros_like(q_ah, dtype=float)
    dt = np.gradient(t_s)
    dq = np.gradient(q_ah)
    valid_dt = np.abs(dt) > 1e-6
    cur = np.zeros_like(q_ah, dtype=float)
    cur[valid_dt] = dq[valid_dt] / (dt[valid_dt] / 3600.0)
    return cur


class OxfordBatteryLoader:
    """Parses Oxford_Battery_Degradation_Dataset_1.mat C1ch/C1dc characterization cycles."""

    DEFAULT_FILENAME = "Oxford_Battery_Degradation_Dataset_1.mat"

    def __init__(
        self,
        mat_path: Path,
        nominal_capacity: Optional[float] = None,
        cc_current: Optional[float] = None,
    ):
        self.mat_path = Path(mat_path)
        self.spec = get_dataset_spec("oxford")
        self.prep_spec = get_preprocessing_spec()
        self.nominal_capacity = (
            float(nominal_capacity) if nominal_capacity is not None else float(self.spec.get("nominal_capacity", 0.74))
        )
        self.cc_current = float(cc_current) if cc_current is not None else float(self.spec.get("cc_current", 0.74))
        self.ambient_temp_c = float(self.spec.get("temperature_c", 40.0))
        self.charge_key = str(self.spec.get("charge_key", "C1ch"))
        self.discharge_key = str(self.spec.get("discharge_key", "C1dc"))
        self.eol_soh_threshold = float(self.spec.get("eol_soh_threshold", 80.0))

    def _resolve_mat_file(self) -> Path:
        """Accept either direct .mat path or a directory containing the Oxford .mat file."""
        if self.mat_path.is_file():
            return self.mat_path
        direct = self.mat_path / self.DEFAULT_FILENAME
        if direct.exists():
            return direct
        matches = [p for p in self.mat_path.rglob("*.mat") if "Oxford_Battery_Degradation" in p.name]
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not locate {self.DEFAULT_FILENAME} at {self.mat_path}")

    def _parse_cell(
        self,
        cell_id: str,
        cell_dict: dict,
        v_grid: np.ndarray,
        initial_capacity_rule: str = "first_valid",
    ) -> Tuple[List[CanonicalCycleGrid], Dict]:
        """Parse and harmonize all characterization cycles for one Oxford cell."""
        cyc_keys = sorted(
            [k for k in cell_dict.keys() if isinstance(cell_dict[k], dict)],
            key=_parse_cycle_number,
        )

        paired_records = []
        dropped_cycles = 0

        for ck in cyc_keys:
            entry = cell_dict[ck]
            ch = entry.get(self.charge_key)
            dc = entry.get(self.discharge_key)
            if not isinstance(ch, dict) or not isinstance(dc, dict):
                dropped_cycles += 1
                continue

            v = np.atleast_1d(ch.get("v", [])).astype(float)
            q_ah = np.atleast_1d(ch.get("q", [])).astype(float) / 1000.0  # mAh -> Ah
            temp = np.atleast_1d(ch.get("T", [])).astype(float)
            t_raw = np.atleast_1d(ch.get("t", [])).astype(float)

            # Oxford .mat timestamps are stored in MATLAB serial days (despite Readme typo)
            if len(t_raw) < 2:
                dropped_cycles += 1
                continue
            t_span = float(t_raw[-1] - t_raw[0])
            t_s = (t_raw - t_raw[0]) * 86400.0 if t_span < 10.0 else (t_raw - t_raw[0])

            dc_q_ah = np.atleast_1d(dc.get("q", [])).astype(float) / 1000.0
            cap_ah = float(np.max(np.abs(dc_q_ah))) if len(dc_q_ah) > 0 else float("nan")

            if not (np.isfinite(cap_ah) and cap_ah > 0.05):
                dropped_cycles += 1
                continue

            i_eff = _estimate_current_a(q_ah, t_s)
            cc_slice = extract_cc_segment(v, i_eff, target_current=self.cc_current)
            if cc_slice is None or len(temp) != len(v) or len(t_s) != len(v):
                dropped_cycles += 1
                continue

            cyc_num = _parse_cycle_number(ck)
            paired_records.append((cyc_num, v[cc_slice], q_ah[cc_slice], temp[cc_slice], t_s[cc_slice], cap_ah))

        if not paired_records:
            return [], {
                "cell_id": cell_id,
                "n_valid_cycles": 0,
                "n_regeneration_cycles": 0,
                "n_below_eol": 0,
                "dropped_cycles": dropped_cycles,
                "ambient_temp_c": self.ambient_temp_c,
            }

        discharge_caps = [rec[5] for rec in paired_records]
        soh_labels, q0 = compute_soh_labels(discharge_caps, initial_capacity_rule=initial_capacity_rule)
        outlier_flags = detect_label_outliers(soh_labels)
        regen_flags = detect_capacity_regeneration(soh_labels, outlier_mask=outlier_flags)
        c_rate = self.cc_current / self.nominal_capacity

        canonical_cycles: List[CanonicalCycleGrid] = []
        v_cc_mins: List[float] = []
        v_cc_maxs: List[float] = []

        for idx, (cyc_num, v_cc, q_cc, temp_cc, t_cc, cap_ah) in enumerate(paired_records):
            grid_data = resample_to_grid(
                voltage=v_cc,
                capacity=q_cc,
                temp=temp_cc,
                time=t_cc,
                nominal_capacity=self.nominal_capacity,
                v_grid=v_grid,
                c_rate=c_rate,
            )

            if not np.any(np.isfinite(grid_data["q_norm"])):
                dropped_cycles += 1
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
                    nominal_capacity=self.nominal_capacity,
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
            "n_valid_cycles": len(canonical_cycles),
            "n_regeneration_cycles": int(sum(c.is_regeneration for c in canonical_cycles)),
            "n_label_outliers": int(sum(c.is_label_outlier for c in canonical_cycles)),
            "n_below_eol": int(sum(c.is_below_eol for c in canonical_cycles)),
            "soh_band_counts": band_counts,
            "dropped_cycles": dropped_cycles,
            "q0_initial_ah": float(q0),
            "q_final_ah": float(discharge_caps[-1]),
            "soh_final_pct": float(soh_labels[-1]),
            "ambient_temp_c": self.ambient_temp_c,
            "v_cc_min_median": float(np.median(v_cc_mins)) if v_cc_mins else float("nan"),
            "v_cc_min_p95": float(np.quantile(v_cc_mins, 0.95)) if v_cc_mins else float("nan"),
            "v_cc_max_min": float(np.min(v_cc_maxs)) if v_cc_maxs else float("nan"),
        }
        return canonical_cycles, cell_meta

    def load(
        self,
        cell_ids: Optional[List[str]] = None,
        initial_capacity_rule: str = "first_valid",
    ) -> DatasetBundle:
        """Extract C1ch curves and paired C1dc discharge capacities into a DatasetBundle."""
        mat_file = self._resolve_mat_file()
        mat = scipy.io.loadmat(mat_file, simplify_cells=True)

        target_cells = (
            list(cell_ids) if cell_ids is not None else list(self.spec.get("cells", [f"Cell{i}" for i in range(1, 9)]))
        )

        v_grid = build_canonical_v_grid()
        all_cycles: List[CanonicalCycleGrid] = []
        cell_capacities: Dict[str, float] = {}
        cell_temps: Dict[str, float] = {}
        per_cell_meta: Dict[str, Dict] = {}

        for cid in target_cells:
            if cid not in mat:
                raise KeyError(f"Cell '{cid}' not found in {mat_file}")
            cycles, meta = self._parse_cell(
                cid,
                mat[cid],
                v_grid=v_grid,
                initial_capacity_rule=initial_capacity_rule,
            )
            all_cycles.extend(cycles)
            cell_capacities[cid] = self.nominal_capacity
            cell_temps[cid] = self.ambient_temp_c
            per_cell_meta[cid] = meta

        return DatasetBundle(
            name="oxford_degradation_1",
            cell_ids=target_cells,
            cycles=all_cycles,
            nominal_capacity=self.nominal_capacity,
            ambient_temp_c=self.ambient_temp_c,
            cell_nominal_capacities=cell_capacities,
            cell_ambient_temps=cell_temps,
            metadata={
                "source_file": str(mat_file),
                "cycle_index_unit": "drive_cycles",
                "cc_current_a": self.cc_current,
                "eol_soh_threshold": self.eol_soh_threshold,
                "initial_capacity_rule": initial_capacity_rule,
                "preprocessing_config": dict(self.prep_spec),
                "per_cell_stats": per_cell_meta,
            },
        )

    def load_cohort(
        self,
        cell_ids: Optional[List[str]] = None,
        initial_capacity_rule: str = "first_valid",
    ) -> DatasetBundle:
        """Alias for load() to match NasaBatteryLoader/SandiaBatteryLoader interface."""
        return self.load(cell_ids=cell_ids, initial_capacity_rule=initial_capacity_rule)
