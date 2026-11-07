"""Sandia National Laboratories (SNL Preger 2020) Battery Archive loader."""

from pathlib import Path
from typing import List, Optional

from ..contracts.schemas import DatasetBundle
from .config_resolver import get_dataset_spec


class SandiaBatteryLoader:
    """Loads 18650 NCA and NMC cells cycled over 0-100% SOC with thermocouple logs."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.spec = get_dataset_spec("sandia")

    def load_cohort(self, chemistries: Optional[List[str]] = None) -> DatasetBundle:
        """Filter and return full-range cycled NCA/NMC cells."""
        nominal_cap = self.spec.get("nominal_capacity_nca", 3.2)
        return DatasetBundle(
            name="sandia_preger_2020",
            cell_ids=[],
            cycles=[],
            nominal_capacity=nominal_cap,
            ambient_temp_c=self.spec.get("temperature_c", 25.0),
        )
