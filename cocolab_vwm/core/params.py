"""Parameter dataclasses for OCOS layers and hierarchies.

All experiment-relevant parameters live here as frozen dataclasses, so that
(a) every simulation result can be reproduced from a single config object,
(b) parameter changes appear in version control as diffs, and
(c) tests can exercise edge cases by constructing Param objects directly.

Defaults follow Sengupta (2025), Table 1 (single-layer) and Table 2 (hierarchy).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class OCOSParams:
    """Parameters for a single OCOS recurrent layer with distance-dependent
    inhibition.

    Parameters
    ----------
    grid_shape
        Spatial layout of nodes; total nodes = grid_shape[0] * grid_shape[1].
    alpha
        Self-excitation gain. Stability requires alpha <= 1 / max F'(x*).
    beta_0
        Base lateral inhibition strength. Effective inhibition between nodes
        i, j is beta_0 * exp(-d_ij^2 / (2 * rf_size^2)).
    rf_size
        Receptive-field size sigma in grid units (controls inhibition spread).
    noise_std
        Gaussian noise standard deviation added per timestep.
    dt
        Simulation timestep in seconds.
    t_total
        Total simulation duration in seconds.
    t_input
        Input stimulus duration in seconds (typically 0.3 * t_total).

    Notes
    -----
    Stability condition (Gershgorin / Lyapunov, Sengupta 2025):
        alpha * beta_0 <= 1 / (n_RF * max_j F'(x*_j))
    where n_RF ~ pi * (rf_size / grid_spacing)^2 is the number of neurons
    within one receptive field.
    """
    grid_shape: tuple[int, int] = (8, 8)
    alpha: float = 2.0
    beta_0: float = 0.3
    rf_size: float = 1.0
    noise_std: float = 0.03
    dt: float = 0.01
    t_total: float = 10.0
    t_input: float = 1.0

    @property
    def n_nodes(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class HierarchyParams:
    """Parameters for the hierarchical VWM model.

    Models the chain Scene -> Object -> Features (orientation, size).
    Cross-talk function C(L) modulates feedback strength across layers.

    Parameters
    ----------
    layer_params
        Default OCOS parameters used at any layer index for which
        ``per_layer_params`` does not provide an override. Maintained for
        backward compatibility with v0.1.
    per_layer_params
        Optional tuple of per-layer OCOSParams. When ``per_layer_params[i]``
        is provided it overrides ``layer_params`` for layer ``i``; otherwise
        the default ``layer_params`` is used. Empty tuple (default) means
        all layers share ``layer_params``.

        Biologically, this lets us model the V1 -> V4 -> IT progression of
        increasing receptive-field size:

            HierarchyParams(
                layer_params=OCOSParams(rf_size=1.0),  # V1-like default
                per_layer_params=(
                    OCOSParams(rf_size=1.0),  # R_V1
                    OCOSParams(rf_size=2.0),  # R_V4
                    OCOSParams(rf_size=4.0),  # R_IT
                ),
            )
    feedback_levels
        Tuple of (level, A_value) pairs. Defaults match Sengupta (2025):
        A(1)=0.3, A(2)=0.7, A(3)=1.0.
    cross_talk_B
        Scaling constant in C(L) = A(L) * exp(-B * (1 - A(L))^2).
    n_features
        Number of feature dimensions (orientation, size = 2 by default).
    """
    layer_params: OCOSParams = field(default_factory=OCOSParams)
    per_layer_params: tuple[OCOSParams, ...] = ()
    feedback_levels: tuple[tuple[int, float], ...] = (
        (1, 0.3), (2, 0.7), (3, 1.0)
    )
    cross_talk_B: float = 5.0
    n_features: int = 2

    def get_layer_params(self, layer_idx: int) -> OCOSParams:
        """Return the OCOSParams for layer ``layer_idx``.

        Falls back to ``self.layer_params`` if no per-layer override exists.
        """
        if 0 <= layer_idx < len(self.per_layer_params):
            return self.per_layer_params[layer_idx]
        return self.layer_params

    def A(self, L: int) -> float:
        """Fraction of p-lattice visible at feedback level L."""
        for level, a in self.feedback_levels:
            if level == L:
                return a
        raise ValueError(
            f"Feedback level {L} not in configured levels "
            f"{[lv for lv, _ in self.feedback_levels]}"
        )

    def cross_talk(self, L: int) -> float:
        """C(L) = A(L) * exp(-B * (1 - A(L))^2)."""
        import math
        a = self.A(L)
        return a * math.exp(-self.cross_talk_B * (1.0 - a) ** 2)
