"""Cell sample weighting to prevent cells with many cycles from dominating."""
import numpy as np

def compute_cell_sample_weights(cell_indices: np.ndarray) -> np.ndarray:
    """Return sample weights inversely proportional to each cell's cycle count."""
    unique_cells, counts = np.unique(cell_indices, return_counts=True)
    count_dict = dict(zip(unique_cells, counts))
    weights = np.array([1.0 / count_dict[c] for c in cell_indices], dtype=float)
    return weights / np.mean(weights)
