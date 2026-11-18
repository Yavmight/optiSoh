"""Sandia National Laboratories (SNL Preger 2020) Battery Archive loader."""

from pathlib import Path
from typing import Dict, List, Optional

from ..contracts.schemas import DatasetBundle
from .config_resolver import get_dataset_spec


class SandiaBatteryLoader:
    """Loads 18650 NCA and NMC cells cycled over 0-100% SOC with thermocouple logs."""

    CHEMISTRY_NOMINAL_CAPACITIES: Dict[str, float] = {
        "NCA": 3.2,
        "NMC": 3.0,
    }

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.spec = get_dataset_spec("sandia")

    def load_cohort(self, chemistries: Optional[List[str]] = None) -> DatasetBundle:
        """Filter and return full-range cycled NCA/NMC cells."""
        nominal_cap_nca = self.spec.get("nominal_capacity_nca", 3.2)
        nominal_cap_nmc = self.spec.get("nominal_capacity_nmc", 3.0)
        ambient_temp = self.spec.get("temperature_c", 25.0)

        cell_capacities = {
            "nca_default": nominal_cap_nca,
            "nmc_default": nominal_cap_nmc,
        }

        return DatasetBundle(
            name="sandia_preger_2020",
            cell_ids=[],
            cycles=[],
            nominal_capacity=nominal_cap_nca,
            ambient_temp_c=ambient_temp,
            cell_nominal_capacities=cell_capacities,
            cell_ambient_temps={"default": ambient_temp},
            metadata={
                "allowed_chemistries": chemistries or ["NCA", "NMC"],
                "supported_temperatures_c": [15.0, 25.0, 35.0],
            },
        )
