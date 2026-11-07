"""Outer Leave-One-Cell-Out, Grouped Outer K-Fold, and Inner Cell-Disjoint splitters."""

from typing import Generator, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold


class OuterLOCOSplitter:
    """Outer Leave-One-Cell-Out cross-validator."""

    def __init__(self, cell_ids: List[str]):
        self.cell_ids = np.array(cell_ids)

    def split(self) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """Yield (train_cells, test_cells, held_out_cell_id) for each outer fold."""
        for i, held_out in enumerate(self.cell_ids):
            train_cells = np.delete(self.cell_ids, i)
            yield train_cells, np.array([held_out]), str(held_out)

    def split_indices(self, cell_labels: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """Yield (train_indices, test_indices, held_out_cell_id) based on cycle cell labels."""
        cell_labels = np.asarray(cell_labels)
        for train_cells, test_cells, held_out in self.split():
            train_mask = np.isin(cell_labels, train_cells)
            test_mask = np.isin(cell_labels, test_cells)
            yield np.where(train_mask)[0], np.where(test_mask)[0], held_out


class InnerCellDisjointSplitter:
    """Inner cross-validator that splits training cells disjointly across folds to prevent leakage."""

    def __init__(
        self,
        cell_ids: List[str],
        n_splits: int = 3,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.cell_ids = np.array(cell_ids)
        self.n_splits = min(n_splits, len(self.cell_ids))
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (inner_train_cells, inner_val_cells) for each inner fold."""
        if self.n_splits < 2:
            yield self.cell_ids, self.cell_ids
            return

        indices = np.arange(len(self.cell_ids))
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)
        for fold_idx in range(self.n_splits):
            val_indices = folds[fold_idx]
            train_indices = np.concatenate([folds[j] for j in range(self.n_splits) if j != fold_idx])
            yield self.cell_ids[train_indices], self.cell_ids[val_indices]

    def split_indices(self, cell_labels: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Given cycle-level cell labels, yield (train_indices, val_indices) without cell leakage."""
        cell_labels = np.asarray(cell_labels)
        unique_cells = np.unique(cell_labels)
        n_splits = min(self.n_splits, len(unique_cells))
        if n_splits < 2:
            all_idx = np.arange(len(cell_labels))
            yield all_idx, all_idx
            return

        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(cell_labels, groups=cell_labels):
            yield train_idx, val_idx


class GroupedOuterKFoldSplitter:
    """Outer Grouped K-Fold cross-validator for cohorts with many cells (e.g. Sandia).

    Partitions cell IDs into K mutually exclusive evaluation cohorts.
    """

    def __init__(
        self,
        cell_ids: List[str],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.cell_ids = np.array(cell_ids)
        self.n_splits = min(n_splits, len(self.cell_ids))
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """Yield (train_cells, test_cells, fold_index) for each outer fold."""
        if self.n_splits < 2:
            yield self.cell_ids, self.cell_ids, 0
            return

        rs = self.random_state if self.shuffle else None
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=rs)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.cell_ids)):
            yield self.cell_ids[train_idx], self.cell_ids[test_idx], fold_idx

    def split_indices(self, cell_labels: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """Yield (train_indices, test_indices, fold_index) based on cycle cell labels."""
        cell_labels = np.asarray(cell_labels)
        for train_cells, test_cells, fold_idx in self.split():
            train_mask = np.isin(cell_labels, train_cells)
            test_mask = np.isin(cell_labels, test_cells)
            yield np.where(train_mask)[0], np.where(test_mask)[0], fold_idx
