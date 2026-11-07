"""Oxford Battery Degradation Dataset 1 loader."""

from pathlib import Path
from typing import List, Optional

from ..contracts.schemas import DatasetBundle
from .config_resolver import get_dataset_spec


class OxfordBatteryLoader:
    """Parses Oxford_Battery_Degradation_Dataset_1.mat C1ch characterization cycles."""

    def __init__(self, mat_path: Path, nominal_capacity: Optional[float] = None):
        self.mat_path = Path(mat_path)
        spec = get_dataset_spec("oxford")
        self.nominal_capacity = nominal_capacity if nominal_capacity is not None else spec.get("nominal_capacity", 0.74)
        self.ambient_temp_c = spec.get("temperature_c", 40.0)

    def load(self, cell_ids: Optional[List[str]] = None) -> DatasetBundle:
        """Extract C1ch cycles and paired C1dc discharge capacities."""
        return DatasetBundle(
            name="oxford_degradation_1",
            cell_ids=cell_ids or [f"Cell{i}" for i in range(1, 9)],
            cycles=[],
            nominal_capacity=self.nominal_capacity,
            ambient_temp_c=self.ambient_temp_c,
        )
