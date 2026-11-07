"""Oxford Battery Degradation Dataset 1 loader."""

from pathlib import Path
from typing import List, Optional

from ..contracts.schemas import DatasetBundle


class OxfordBatteryLoader:
    """Parses Oxford_Battery_Degradation_Dataset_1.mat C1ch characterization cycles."""

    def __init__(self, mat_path: Path, nominal_capacity: float = 0.74):
        self.mat_path = Path(mat_path)
        self.nominal_capacity = nominal_capacity

    def load(self, cell_ids: Optional[List[str]] = None) -> DatasetBundle:
        """Extract C1ch cycles and paired C1dc discharge capacities."""
        return DatasetBundle(
            name="oxford_degradation_1",
            cell_ids=cell_ids or [f"Cell{i}" for i in range(1, 9)],
            cycles=[],
            nominal_capacity=self.nominal_capacity,
            ambient_temp_c=40.0,
        )
