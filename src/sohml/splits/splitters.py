"""Outer Leave-One-Cell-Out and Inner GroupKFold splitters."""

from typing import Generator, List, Tuple

import numpy as np


class OuterLOCOSplitter:
    """Outer Leave-One-Cell-Out cross-validator."""

    def __init__(self, cell_ids: List[str]):
        self.cell_ids = np.array(cell_ids)

    def split(self) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """Yield (train_cells, test_cells, held_out_cell_id) for each outer fold."""
        for i, held_out in enumerate(self.cell_ids):
            train_cells = np.delete(self.cell_ids, i)
            yield train_cells, np.array([held_out]), str(held_out)
