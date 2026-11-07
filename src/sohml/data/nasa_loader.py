"""NASA PCoE Li-ion Battery Aging Dataset loader."""

from pathlib import Path
from typing import List, Optional

from ..contracts.schemas import DatasetBundle
from .config_resolver import get_dataset_spec


class NasaBatteryLoader:
    """Parses NASA PCoE MATLAB structures and cleaned CSVs."""

    def __init__(self, raw_dir: Path, nominal_capacity: Optional[float] = None):
        self.raw_dir = Path(raw_dir)
        spec = get_dataset_spec("nasa")
        self.nominal_capacity = nominal_capacity if nominal_capacity is not None else spec.get("nominal_capacity", 2.0)
        self.ambient_temp_c = spec.get("temperature_c", 24.0)

    def load_cohort(self, cell_ids: List[str]) -> DatasetBundle:
        """Load and parse specified cell cohort."""
        return DatasetBundle(
            name="nasa_pcoe",
            cell_ids=cell_ids,
            cycles=[],
            nominal_capacity=self.nominal_capacity,
            ambient_temp_c=self.ambient_temp_c,
        )
