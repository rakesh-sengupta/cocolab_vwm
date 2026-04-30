"""Core dynamics, parameters, and cross-talk function."""
from cocolab_vwm.core.crosstalk import cross_talk, uncertainty
from cocolab_vwm.core.dynamics import (
    activation,
    activation_derivative,
    grid_positions,
    inhibition_matrix,
    simulate,
    stability_bound,
)
from cocolab_vwm.core.params import HierarchyParams, OCOSParams
from cocolab_vwm.core.pooling import (
    average_pool_transform,
    max_pool,
    pool_grid_shape,
    pool_window_indices,
    winner_take_all,
)

__all__ = [
    "OCOSParams",
    "HierarchyParams",
    "activation",
    "activation_derivative",
    "grid_positions",
    "inhibition_matrix",
    "stability_bound",
    "simulate",
    "cross_talk",
    "uncertainty",
    "max_pool",
    "pool_grid_shape",
    "pool_window_indices",
    "average_pool_transform",
    "winner_take_all",
]
