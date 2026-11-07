"""Tests for outer LOCO and inner GroupKFold splitters."""

from sohml.splits.splitters import OuterLOCOSplitter


def test_outer_loco():
    splitter = OuterLOCOSplitter(cell_ids=["C1", "C2", "C3"])
    splits = list(splitter.split())
    assert len(splits) == 3
    train, test, held_out = splits[0]
    assert held_out not in train
