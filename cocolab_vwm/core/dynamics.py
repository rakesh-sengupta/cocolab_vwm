"""OCOS dynamics: pure-numpy reference implementation.

This module implements the additive recurrent on-center off-surround network
of Sengupta (2025), Eqs. (1)-(3). The pure-numpy implementation serves three
purposes:

1. **Specification**: it is the unambiguous mathematical reference against
   which any Nengo realisation is compared (regression-tested).
2. **Speed**: parameter sweeps over hundreds of trials run faster in numpy
   than in spiking Nengo.
3. **Independence**: lets users run the model without installing Nengo.

For the spiking Nengo version see `cocolab_vwm.layers.nengo_layer`.
"""
from __future__ import annotations

import numpy as np

from cocolab_vwm.core.params import OCOSParams


def activation(x: np.ndarray) -> np.ndarray:
    """Half-rectified saturating activation F(x) = x/(1+x) for x > 0.

    Eq. (3) of Sengupta (2025).
    """
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = x[mask] / (1.0 + x[mask])
    return out


def activation_derivative(x: np.ndarray) -> np.ndarray:
    """F'(x) = 1/(1+x)^2 for x > 0, used in stability checks."""
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = 1.0 / (1.0 + x[mask]) ** 2
    return out


def grid_positions(grid_shape: tuple[int, int]) -> np.ndarray:
    """Return (n_nodes, 2) array of (row, col) positions for a 2D grid."""
    rows, cols = grid_shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(float)


def inhibition_matrix(params: OCOSParams) -> np.ndarray:
    """Compute beta_ij per Eq. (1) of Sengupta (2025).

    Returns
    -------
    beta : (n_nodes, n_nodes) array
        Off-diagonal beta_ij = beta_0 * exp(-d_ij^2 / (2 * rf_size^2)).
        Diagonal is zero (no self-inhibition; self-excitation handled by alpha).
    """
    pos = grid_positions(params.grid_shape)
    diffs = pos[:, None, :] - pos[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)
    beta = params.beta_0 * np.exp(-sq_dists / (2.0 * params.rf_size ** 2))
    np.fill_diagonal(beta, 0.0)
    return beta


def stability_bound(params: OCOSParams, x_star: float = 0.0) -> float:
    """Approximate non-divergence bound on alpha (Sengupta 2025).

    Returns the maximum alpha for which the linearised system has only
    non-positive real eigenvalues (Gershgorin estimate).

    At x_star = 0, F'(0) = 1, giving the conservative bound
        alpha_max = 1 / (n_RF_eff)
    where n_RF_eff is the largest row sum of the inhibition matrix.
    """
    beta = inhibition_matrix(params)
    fprime = activation_derivative(np.array([x_star]))[0]
    if fprime == 0:
        return float("inf")
    row_sum_max = beta.sum(axis=1).max() * fprime
    if row_sum_max == 0:
        return float("inf")
    return 1.0 / row_sum_max


def simulate(
    params: OCOSParams,
    input_indices: np.ndarray,
    input_amplitude: float = 1.0,
    rng: np.random.Generator | None = None,
    return_history: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the OCOS network for `params.t_total` seconds.

    Parameters
    ----------
    params : OCOSParams
    input_indices : array of node indices to which input is applied
    input_amplitude : amplitude of the binary input during t_input
    rng : numpy Generator for reproducibility (passed by caller)
    return_history : if True, also return the full activity time series

    Returns
    -------
    x_final : (n_nodes,) final activation
    history : (n_steps, n_nodes) array if return_history else None
    """
    if rng is None:
        rng = np.random.default_rng()

    n_nodes = params.n_nodes
    n_steps = int(round(params.t_total / params.dt))
    n_input_steps = int(round(params.t_input / params.dt))

    beta = inhibition_matrix(params)
    x = np.zeros(n_nodes)

    history = np.zeros((n_steps, n_nodes)) if return_history else None

    input_vec = np.zeros(n_nodes)
    input_vec[input_indices] = input_amplitude

    for step in range(n_steps):
        # I in Eq. (2) of Sengupta (2025); renamed input_t to satisfy ruff E741.
        input_t = input_vec if step < n_input_steps else np.zeros(n_nodes)
        Fx = activation(x)
        # Eq. (2): dx/dt = -x + alpha*F(x) - sum_j beta_ij F(x_j) + I + noise
        dxdt = (
            -x
            + params.alpha * Fx
            - beta @ Fx
            + input_t
            + rng.normal(0.0, params.noise_std, size=n_nodes)
        )
        x = x + dxdt * params.dt
        if history is not None:
            history[step] = x

    return x, history
