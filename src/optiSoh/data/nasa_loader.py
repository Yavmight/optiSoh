"""NASA PCoE Li-ion Battery Aging Dataset loader."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from ..contracts.schemas import CanonicalCycleGrid, DatasetBundle
from .cc_segment import extract_cc_segment
from .config_resolver import get_dataset_spec, get_preprocessing_spec
from .harmonize import build_canonical_v_grid, resample_to_grid
from .labels import compute_soh_labels, detect_capacity_regeneration, detect_label_outliers


def _extract_scalar_capacity(data_dict: dict) -> float:
    """Safely extract scalar discharge capacity (Ah) from a NASA MATLAB cycle dict."""
    cap = data_dict.get("Capacity", np.nan)
    if isinstance(cap, np.ndarray):
        if cap.size == 0:
            return float("nan")
        return float(cap.flat[0])
    try:
        return float(cap)
    except (TypeError, ValueError):
        return float("nan")


def _integrate_charge_ah(current: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Compute cumulative charge capacity Q(t) in Ah via trapezoidal integration."""
    if len(time_s) < 2:
        return np.zeros_like(time_s, dtype=float)
    dt_h = np.diff(time_s) / 3600.0
    dt_h = np.maximum(dt_h, 0.0)
    dq = 0.5 * (current[:-1] + current[1:]) * dt_h
    return np.concatenate(([0.0], np.cumsum(dq)))


class NasaBatteryLoader:
    """Parses NASA PCoE MATLAB (.mat) battery aging structures into CanonicalCycleGrid bundles."""

    def __init__(
        self,
        raw_dir: Path,
        nominal_capacity: Optional[float] = None,
        cc_current: Optional[float] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.spec = get_dataset_spec("nasa")
        self.prep_spec = get_preprocessing_spec()
        self.nominal_capacity = (
            float(nominal_capacity) if nominal_capacity is not None else float(self.spec.get("nominal_capacity", 2.0))
        )
        self.cc_current = float(cc_current) if cc_current is not None else float(self.spec.get("cc_current", 1.5))
        self.ambient_temp_c = float(self.spec.get("temperature_c", 24.0))
        self.eol_soh_threshold = float(self.spec.get("eol_soh_threshold", 80.0))

    def _find_cell_file(self, cell_id: str) -> Path:
        """Locate <cell_id>.mat directly or inside subdirectories of raw_dir."""
        direct = self.raw_dir / f"{cell_id}.mat"
        if direct.exists():
            return direct
        matches = list(self.raw_dir.rglob(f"{cell_id}.mat"))
        if not matches:
            raise FileNotFoundError(f"Could not find {cell_id}.mat under {self.raw_dir}")
        return matches[0]

    def load_cell(
        self,
        cell_id: str,
        v_grid: Optional[np.ndarray] = None,
        initial_capacity_rule: str = "first_valid",
    ) -> Tuple[List[CanonicalCycleGrid], Dict]:
        """Parse a single NASA .mat cell file into harmonized CanonicalCycleGrid cycles."""
        mat_path = self._find_cell_file(cell_id)
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        if cell_id not in mat:
            raise KeyError(f"Expected top-level key '{cell_id}' in {mat_path}")

        raw_cycles = mat[cell_id]["cycle"]
        if not isinstance(raw_cycles, list):
            raw_cycles = [raw_cycles]

        if v_grid is None:
            v_grid = build_canonical_v_grid()

        # Pair each valid discharge capacity with its most recent preceding valid CC charge
        paired_records = []
        pending_charge = None
        dropped_charge_ops = 0
        ambient_temps = []

        for op_idx, op in enumerate(raw_cycles):
            op_type = op.get("type", "")
            amb_t = op.get("ambient_temperature", np.nan)
            if isinstance(amb_t, (int, float, np.floating)) and np.isfinite(amb_t):
                ambient_temps.append(float(amb_t))

            data = op.get("data", {})
            if not isinstance(data, dict):
                continue

            if op_type == "charge":
                v = np.atleast_1d(data.get("Voltage_measured", [])).astype(float)
                cur = np.atleast_1d(data.get("Current_measured", [])).astype(float)
                temp = np.atleast_1d(data.get("Temperature_measured", [])).astype(float)
                t = np.atleast_1d(data.get("Time", [])).astype(float)

                cc_slice = extract_cc_segment(v, cur, target_current=self.cc_current)
                if cc_slice is not None and len(temp) == len(v) and len(t) == len(v):
                    pending_charge = (op_idx, v[cc_slice], cur[cc_slice], temp[cc_slice], t[cc_slice])
                else:
                    dropped_charge_ops += 1

            elif op_type == "discharge":
                cap = _extract_scalar_capacity(data)
                if np.isfinite(cap) and cap > 0.1 and pending_charge is not None:
                    paired_records.append((pending_charge, cap))
                    pending_charge = None

        if not paired_records:
            return [], {
                "cell_id": cell_id,
                "source_file": str(mat_path),
                "n_valid_cycles": 0,
                "dropped_charge_ops": dropped_charge_ops,
                "ambient_temp_c": float(np.median(ambient_temps)) if ambient_temps else self.ambient_temp_c,
            }

        discharge_caps = [rec[1] for rec in paired_records]
        soh_labels, q0 = compute_soh_labels(discharge_caps, initial_capacity_rule=initial_capacity_rule)
        outlier_flags = detect_label_outliers(soh_labels)
        regen_flags = detect_capacity_regeneration(soh_labels, outlier_mask=outlier_flags)
        c_rate = self.cc_current / self.nominal_capacity

        canonical_cycles: List[CanonicalCycleGrid] = []
        v_cc_mins: List[float] = []
        v_cc_maxs: List[float] = []

        for cyc_idx, ((_, v_cc, cur_cc, temp_cc, t_cc), cap) in enumerate(paired_records):
            q_cc = _integrate_charge_ah(cur_cc, t_cc)
            grid_data = resample_to_grid(
                voltage=v_cc,
                capacity=q_cc,
                temp=temp_cc,
                time=t_cc,
                nominal_capacity=self.nominal_capacity,
                v_grid=v_grid,
                c_rate=c_rate,
            )

            # Skip cycle if resampling yielded no finite grid points
            if not np.any(np.isfinite(grid_data["q_norm"])):
                dropped_charge_ops += 1
                continue

            v_cc_mins.append(float(np.min(v_cc)))
            v_cc_maxs.append(float(np.max(v_cc)))

            soh_val = float(soh_labels[cyc_idx])
            canonical_cycles.append(
                CanonicalCycleGrid(
                    cell_id=cell_id,
                    cycle_index=cyc_idx,
                    v_grid=grid_data["v_grid"],
                    q_norm=grid_data["q_norm"],
                    dq_dv=grid_data["dq_dv"],
                    delta_t=grid_data["delta_t"],
                    dt_dv=grid_data["dt_dv"],
                    tau=grid_data["tau"],
                    soh_true=soh_val,
                    discharge_capacity=float(cap),
                    nominal_capacity=self.nominal_capacity,
                    is_regeneration=bool(regen_flags[cyc_idx]),
                    is_below_eol=bool(soh_val < self.eol_soh_threshold),
                    is_label_outlier=bool(outlier_flags[cyc_idx]),
                )
            )

        band_counts: Dict[str, int] = {}
        for c in canonical_cycles:
            b = c.soh_band(self.eol_soh_threshold)
            band_counts[b] = band_counts.get(b, 0) + 1

        cell_meta = {
            "cell_id": cell_id,
            "source_file": str(mat_path),
            "n_valid_cycles": len(canonical_cycles),
            "n_regeneration_cycles": int(sum(c.is_regeneration for c in canonical_cycles)),
            "n_label_outliers": int(sum(c.is_label_outlier for c in canonical_cycles)),
            "n_below_eol": int(sum(c.is_below_eol for c in canonical_cycles)),
            "soh_band_counts": band_counts,
            "dropped_charge_ops": dropped_charge_ops,
            "q0_initial_ah": float(q0),
            "q_final_ah": float(discharge_caps[-1]),
            "soh_final_pct": float(soh_labels[-1]),
            "ambient_temp_c": float(np.median(ambient_temps)) if ambient_temps else self.ambient_temp_c,
            "v_cc_min_median": float(np.median(v_cc_mins)) if v_cc_mins else float("nan"),
            "v_cc_min_p95": float(np.quantile(v_cc_mins, 0.95)) if v_cc_mins else float("nan"),
            "v_cc_max_min": float(np.min(v_cc_maxs)) if v_cc_maxs else float("nan"),
        }
        return canonical_cycles, cell_meta

    def load_cohort(
        self,
        cell_ids: Optional[List[str]] = None,
        initial_capacity_rule: str = "first_valid",
    ) -> DatasetBundle:
        """Load, harmonize, and return a DatasetBundle for the specified NASA cell cohort."""
        if cell_ids is None:
            cell_ids = list(self.spec.get("core_cells", ["B0005", "B0006", "B0007", "B0018"]))

        v_grid = build_canonical_v_grid()
        all_cycles: List[CanonicalCycleGrid] = []
        cell_capacities: Dict[str, float] = {}
        cell_temps: Dict[str, float] = {}
        per_cell_meta: Dict[str, Dict] = {}

        for cid in cell_ids:
            cycles, meta = self.load_cell(cid, v_grid=v_grid, initial_capacity_rule=initial_capacity_rule)
            all_cycles.extend(cycles)
            cell_capacities[cid] = self.nominal_capacity
            cell_temps[cid] = meta["ambient_temp_c"]
            per_cell_meta[cid] = meta

        return DatasetBundle(
            name="nasa_pcoe",
            cell_ids=list(cell_ids),
            cycles=all_cycles,
            nominal_capacity=self.nominal_capacity,
            ambient_temp_c=self.ambient_temp_c,
            cell_nominal_capacities=cell_capacities,
            cell_ambient_temps=cell_temps,
            metadata={
                "cycle_index_unit": "sequential_cc_cycles",
                "cc_current_a": self.cc_current,
                "eol_soh_threshold": self.eol_soh_threshold,
                "initial_capacity_rule": initial_capacity_rule,
                "preprocessing_config": dict(self.prep_spec),
                "per_cell_stats": per_cell_meta,
            },
        )
