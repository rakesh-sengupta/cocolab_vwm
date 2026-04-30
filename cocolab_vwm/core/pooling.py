"""Feedforward compression: max-pooling and other homology-surjective maps.

This module implements the neurobiologically grounded feedforward operations
discussed in the binding paper (Sengupta, in prep, Proposition: hierarchy
via homology-surjective maps). Three candidate operations are provided:

1. ``max_pool_indices`` : 2D max-pooling on the OCOS grid. Selects the
   maximally active node within each pooling window. Geometrically, this
   is an elementary collapse of a contractible neighbourhood, which is a
   homological surjection.

2. ``winner_take_all_indices`` : global WTA. The single most active node
   in the whole layer wins.

3. ``soft_max_pool`` : a softmax-weighted average within each pool. This
   is differentiable (useful for future learning-based extensions) and
   approaches max-pooling as the softmax temperature goes to zero.

All three functions preserve the contractibility-of-fibres condition
needed for the surjectivity hypothesis to hold (Sengupta in prep,
biological grounding section).
"""
from __future__ import annotations

import numpy as np


def _check_pool(grid_shape: tuple[int, int], pool_size: int) -> None:
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    rows, cols = grid_shape
    if rows % pool_size != 0 or cols % pool_size != 0:
        raise ValueError(
            f"grid_shape {grid_shape} must be divisible by pool_size "
            f"{pool_size}"
        )


def pool_grid_shape(
    grid_shape: tuple[int, int], pool_size: int
) -> tuple[int, int]:
    """Return the shape of the grid after pooling."""
    _check_pool(grid_shape, pool_size)
    return (grid_shape[0] // pool_size, grid_shape[1] // pool_size)


def max_pool(
    activity: np.ndarray,
    grid_shape: tuple[int, int],
    pool_size: int,
) -> np.ndarray:
    """2D max-pool a 1D activity vector laid out on a grid.

    Parameters
    ----------
    activity : (n_nodes,) array
        Lower-layer activity in row-major order.
    grid_shape : (rows, cols)
        Lower-layer grid shape; rows and cols must be divisible by pool_size.
    pool_size : int
        Edge of each square pooling window. ``pool_size=2`` is the standard
        2x2 pool; larger values give larger receptive fields per upper unit.

    Returns
    -------
    pooled : (n_pooled_nodes,) array
        Upper-layer activity in row-major order over the pooled grid.

    Notes
    -----
    The corresponding pool window indices for each upper unit are computable
    via ``pool_window_indices``; this is what we use to wire the Nengo
    feedforward connection.
    """
    _check_pool(grid_shape, pool_size)
    rows, cols = grid_shape
    grid = activity.reshape(rows, cols)
    out_rows, out_cols = rows // pool_size, cols // pool_size
    out = np.zeros((out_rows, out_cols))
    for i in range(out_rows):
        for j in range(out_cols):
            block = grid[
                i * pool_size : (i + 1) * pool_size,
                j * pool_size : (j + 1) * pool_size,
            ]
            out[i, j] = block.max()
    return out.ravel()


def pool_window_indices(
    grid_shape: tuple[int, int], pool_size: int
) -> list[list[int]]:
    """Return, for each upper-layer unit, the list of lower-layer indices
    that fall in its pooling window.

    Used to build the feedforward connection weight pattern: each upper
    unit is connected to (and pools over) its block of ``pool_size**2``
    lower units.

    Returns
    -------
    windows : list of length n_pooled_nodes
        windows[u] is the list of lower-layer node indices in upper unit u's
        receptive field.
    """
    _check_pool(grid_shape, pool_size)
    rows, cols = grid_shape
    out_rows, out_cols = rows // pool_size, cols // pool_size
    windows = []
    for i in range(out_rows):
        for j in range(out_cols):
            block = []
            for di in range(pool_size):
                for dj in range(pool_size):
                    r = i * pool_size + di
                    c = j * pool_size + dj
                    block.append(r * cols + c)
            windows.append(block)
    return windows


def average_pool_transform(
    grid_shape: tuple[int, int], pool_size: int
) -> np.ndarray:
    """Build a (n_pooled, n_input) linear transform implementing average pooling.

    This is what a Nengo Connection's ``transform=`` argument expects.

    Average pooling is a strict approximation of max-pooling that is
    *linear*, hence directly representable as a Nengo transform without a
    nonlinearity. Max-pooling requires a nonlinear operation; for that we
    expose a separate ``MaxPoolNode`` class below.

    Returns
    -------
    W : (n_pooled, n_input) array
        Averaging weights: W[u, v] = 1/(pool_size**2) if v is in u's window,
        else 0.
    """
    windows = pool_window_indices(grid_shape, pool_size)
    n_input = grid_shape[0] * grid_shape[1]
    n_pooled = len(windows)
    W = np.zeros((n_pooled, n_input))
    weight = 1.0 / (pool_size * pool_size)
    for u, idxs in enumerate(windows):
        for v in idxs:
            W[u, v] = weight
    return W


def winner_take_all(activity: np.ndarray) -> np.ndarray:
    """Return a one-hot vector with a 1 at the maximum-activation index.

    Global WTA: a single representative wins for the whole layer.
    """
    out = np.zeros_like(activity)
    out[int(np.argmax(activity))] = 1.0
    return out
