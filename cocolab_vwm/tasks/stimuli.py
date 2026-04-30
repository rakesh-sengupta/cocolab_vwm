"""Stimulus generation for change-detection and related tasks.

Provides reproducible (seeded) functions to generate sets of object positions
on the OCOS grid for `close` and `far` spatial distributions, matching the
two conditions in Sengupta (2025) Fig. 2/3.
"""
from __future__ import annotations

import numpy as np


def close_inputs(
    grid_shape: tuple[int, int], set_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample `set_size` distinct nodes from a 2x2 sub-grid in the corner.

    Matches the 'close inputs' condition of Sengupta (2025), where stimuli
    cluster within a small region.
    """
    rows, cols = grid_shape
    if set_size > 4:
        raise ValueError("close_inputs supports set_size <= 4 (2x2 region).")
    candidates = [r * cols + c for r in range(2) for c in range(2)]
    return rng.choice(candidates, size=set_size, replace=False)


def far_inputs(
    grid_shape: tuple[int, int], set_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample `set_size` nodes spread radially around the grid.

    Matches the 'far inputs' condition: stimuli placed >= 3 nodes apart.
    """
    rows, cols = grid_shape
    cr, cc = rows // 2, cols // 2
    candidates = []
    for r in range(rows):
        for c in range(cols):
            if abs(r - cr) >= 3 or abs(c - cc) >= 3:
                candidates.append(r * cols + c)
    if set_size > len(candidates):
        raise ValueError(
            f"set_size {set_size} exceeds available far positions "
            f"{len(candidates)} for grid {grid_shape}"
        )
    return rng.choice(candidates, size=set_size, replace=False)


def make_target_pattern(
    grid_shape: tuple[int, int], indices: np.ndarray, amplitude: float = 1.0
) -> np.ndarray:
    """Create the ideal target activity vector for a stimulus."""
    n = grid_shape[0] * grid_shape[1]
    target = np.zeros(n)
    target[indices] = amplitude
    return target


def changed_indices(
    indices: np.ndarray,
    grid_shape: tuple[int, int],
    n_changes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a perturbed index set: replace `n_changes` items with new ones.

    Used to construct the probe array in the change-detection task.
    """
    n = grid_shape[0] * grid_shape[1]
    if n_changes > len(indices):
        raise ValueError("Cannot change more items than are present.")
    keep = rng.choice(indices, size=len(indices) - n_changes, replace=False)
    available = np.setdiff1d(np.arange(n), indices)
    new = rng.choice(available, size=n_changes, replace=False)
    return np.concatenate([keep, new])
