"""Change-detection task driver.

Runs the canonical VWM change-detection paradigm (Luck & Vogel 1997 style)
on either the pure-numpy OCOS reference implementation or the spiking
Nengo version. Produces per-trial Hamming distances and aggregate accuracy
suitable for figure-replication scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cocolab_vwm.core.dynamics import simulate
from cocolab_vwm.core.params import OCOSParams
from cocolab_vwm.tasks.stimuli import close_inputs, far_inputs, make_target_pattern
from cocolab_vwm.utils.metrics import hamming_distance

SpatialCondition = Literal["close", "far"]


@dataclass
class ChangeDetectionResult:
    """Container for one batch of change-detection trials."""
    set_size: int
    spatial: SpatialCondition
    rf_size: float
    beta_0: float
    hamming_distances: list[int] = field(default_factory=list)

    @property
    def mean_hamming(self) -> float:
        return float(np.mean(self.hamming_distances))

    @property
    def sem_hamming(self) -> float:
        if len(self.hamming_distances) <= 1:
            return 0.0
        return float(np.std(self.hamming_distances, ddof=1)
                     / np.sqrt(len(self.hamming_distances)))


def run_change_detection(
    params: OCOSParams,
    set_size: int,
    spatial: SpatialCondition = "far",
    n_trials: int = 100,
    seed: int | None = None,
) -> ChangeDetectionResult:
    """Run a batch of change-detection trials with the numpy OCOS model.

    Each trial: encode `set_size` items at positions sampled per `spatial`
    condition, run the network, compare final state against the target pattern
    via Hamming distance.

    Parameters
    ----------
    params : OCOSParams
    set_size : int
        Number of items in the memory array.
    spatial : 'close' or 'far'
        Spatial-distribution condition (Sengupta 2025).
    n_trials : int
    seed : int or None
        Reproducibility seed for the per-batch RNG.
    """
    rng = np.random.default_rng(seed)
    sampler = close_inputs if spatial == "close" else far_inputs

    result = ChangeDetectionResult(
        set_size=set_size, spatial=spatial,
        rf_size=params.rf_size, beta_0=params.beta_0,
    )

    for _ in range(n_trials):
        idx = sampler(params.grid_shape, set_size, rng)
        target = make_target_pattern(params.grid_shape, idx)
        x_final, _ = simulate(params, idx, rng=rng)
        result.hamming_distances.append(
            hamming_distance(x_final, target)
        )

    return result
