"""Hierarchical VWM with explicit feedforward pooling.

Builds on ``hierarchy.py`` by replacing the identity feedforward connection
with an average-pool transform. This realises the homology-surjective
compression discussed in the binding paper (Sengupta, in prep): each upper
unit pools over a contractible neighbourhood of lower units, which is a
homological surjection that cannot increase Betti numbers across stages.

Why average pool, not max pool?
-------------------------------
True max-pooling is nonlinear and would require a Nengo Node implementing
``np.max``. Average pooling is linear, hence directly representable as a
Nengo Connection ``transform=`` weight matrix. For most analyses the
distinction is qualitative (both implement a homological surjection); a
true-max-pool variant is provided in v0.3 if needed.

Backward compatibility
----------------------
This is a NEW function, ``make_pooled_hierarchy``, separate from
``make_hierarchy``. v0.1 code that calls ``make_hierarchy`` is not affected.
"""
from __future__ import annotations

import nengo
import numpy as np

from cocolab_vwm.core.crosstalk import cross_talk
from cocolab_vwm.core.params import HierarchyParams
from cocolab_vwm.core.pooling import average_pool_transform, pool_grid_shape
from cocolab_vwm.layers.nengo_layer import make_ocos_layer


def make_pooled_hierarchy(
    params: HierarchyParams,
    pool_size: int = 2,
    feedback_level: int = 3,
    n_neurons_per_node: int = 30,
    label: str = "pooled_hierarchy",
) -> nengo.Network:
    """Build a two-layer OCOS hierarchy with average-pool feedforward.

    The lower layer has ``params.get_layer_params(0).grid_shape``; the upper
    layer's grid is determined by the pool size:
        upper_grid = (lower_rows // pool_size, lower_cols // pool_size)

    The upper-layer OCOSParams is taken from
    ``params.get_layer_params(1)`` if provided, but its grid_shape is
    overridden to match the pooled output grid.

    Parameters
    ----------
    params : HierarchyParams
    pool_size : int
        Edge of the square pooling window (>= 1; 2 = standard 2x2 pool).
    feedback_level : int
    n_neurons_per_node : int
    label : str

    Returns
    -------
    net : nengo.Network with attributes:
        - net.input          : feedforward input to lower layer
        - net.lower          : R_lower OCOS layer
        - net.upper          : R_upper OCOS layer (smaller grid)
        - net.feedback_gain  : Node with scalar c_talk gain
        - net.output_lower   : output of R_lower (n_lower,)
        - net.output_upper   : output of R_upper (n_upper,)

    Notes
    -----
    The feedback path from upper to lower is implemented with the
    *transpose* of the pooling transform: each upper unit broadcasts back
    to all lower units in its pool, weighted equally. This is the simplest
    transposed-projection compatible with backprop intuitions and matches
    how recurrent biological hierarchies broadcast feedback into their
    receptive fields.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")

    c_talk_value = cross_talk(params.A(feedback_level), params.cross_talk_B)

    lower_params = params.get_layer_params(0)
    n_lower = lower_params.n_nodes

    # Compute pooled (upper) grid shape and override upper-layer grid.
    upper_grid = pool_grid_shape(lower_params.grid_shape, pool_size)
    n_upper = upper_grid[0] * upper_grid[1]
    base_upper = params.get_layer_params(1)
    # Construct upper params with overridden grid_shape (frozen dataclass:
    # we use dataclasses.replace to produce a new instance).
    from dataclasses import replace
    upper_params = replace(base_upper, grid_shape=upper_grid)

    # Pooling transforms.
    W_ff = average_pool_transform(lower_params.grid_shape, pool_size)
    # Feedback uses the transpose (broadcast back, equally weighted).
    W_fb = W_ff.T  # shape (n_lower, n_upper)

    with nengo.Network(label=label) as net:
        net.input = nengo.Node(size_in=n_lower, label="ff_input")

        net.lower = make_ocos_layer(
            lower_params,
            n_neurons_per_node=n_neurons_per_node,
            label="R_lower",
        )
        net.upper = make_ocos_layer(
            upper_params,
            n_neurons_per_node=n_neurons_per_node,
            label="R_upper",
        )

        # Feedforward: pooled lower -> upper.
        nengo.Connection(
            net.lower.output, net.upper.input,
            transform=W_ff, synapse=0.05,
        )

        # Cross-talk gain.
        net.feedback_gain = nengo.Node(
            output=lambda t, gain=c_talk_value: gain,
            size_out=1,
            label="c_talk",
        )

        # Feedback: upper -> lower, gated by c_talk and broadcast through
        # the pooling transpose.
        gate = nengo.networks.Product(
            n_neurons=n_neurons_per_node, dimensions=n_upper,
            label="feedback_gate",
        )
        nengo.Connection(net.upper.output, gate.input_a, synapse=0.05)
        nengo.Connection(
            net.feedback_gain, gate.input_b,
            transform=np.ones((n_upper, 1)), synapse=None,
        )
        nengo.Connection(
            gate.output, net.lower.input,
            transform=W_fb, synapse=0.05,
        )

        nengo.Connection(net.input, net.lower.input, synapse=None)

        net.output_lower = nengo.Node(size_in=n_lower, label="out_lower")
        net.output_upper = nengo.Node(size_in=n_upper, label="out_upper")
        nengo.Connection(net.lower.output, net.output_lower, synapse=None)
        nengo.Connection(net.upper.output, net.output_upper, synapse=None)

    return net
