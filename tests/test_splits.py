"""Tests for outer LOCO, grouped outer K-fold, and inner cell-disjoint splitters."""

import numpy as np
from sohml.splits.splitters import (
    GroupedOuterKFoldSplitter,
    InnerCellDisjointSplitter,
    OuterLOCOSplitter,
)


def test_outer_loco():
    splitter = OuterLOCOSplitter(cell_ids=["C1", "C2", "C3"])
    splits = list(splitter.split())
    assert len(splits) == 3
    train, test, held_out = splits[0]
    assert held_out not in train
    assert len(train) == 2
    assert len(test) == 1

    # Test split_indices
    cell_labels = np.array(["C1", "C1", "C2", "C2", "C3", "C3"])
    idx_splits = list(splitter.split_indices(cell_labels))
    assert len(idx_splits) == 3
    tr_idx, te_idx, held_out_id = idx_splits[0]
    assert held_out_id == "C1"
    assert len(te_idx) == 2
    assert len(tr_idx) == 4
    assert len(set(tr_idx).intersection(set(te_idx))) == 0


def test_inner_cell_disjoint():
    cell_ids = ["C1", "C2", "C3", "C4"]
    splitter = InnerCellDisjointSplitter(cell_ids=cell_ids, n_splits=2, shuffle=False)
    splits = list(splitter.split())
    assert len(splits) == 2
    for tr, val in splits:
        assert len(set(tr).intersection(set(val))) == 0

    # Test split_indices
    cell_labels = np.array(["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4"])
    idx_splits = list(splitter.split_indices(cell_labels))
    assert len(idx_splits) == 2
    for tr_idx, val_idx in idx_splits:
        tr_cells = set(cell_labels[tr_idx])
        val_cells = set(cell_labels[val_idx])
        assert len(tr_cells.intersection(val_cells)) == 0


def test_grouped_outer_kfold():
    cell_ids = [f"C{i}" for i in range(10)]
    splitter = GroupedOuterKFoldSplitter(cell_ids=cell_ids, n_splits=5, shuffle=False)
    splits = list(splitter.split())
    assert len(splits) == 5

    all_test_cells = []
    for tr, te, fold_idx in splits:
        assert len(set(tr).intersection(set(te))) == 0
        all_test_cells.extend(te)
    assert set(all_test_cells) == set(cell_ids)
