"""OCOS as a Nengo network.

Wraps the additive recurrent on-center off-surround dynamics in a Nengo
``nengo.Network``. The network exposes an input port (a ``nengo.Node``) and
an output probe-able ensemble; this means it composes naturally with
``nengo_spa`` modules and with the BG-thalamus action-selection circuit
(see ``cocolab_vwm.control``).

Design choice
-------------
We use a single ensemble of n_nodes neurons with a custom recurrent connection
function rather than n_nodes separate one-neuron ensembles. This is faster to
build and simulate, and it keeps the topology explicit (the recurrent weight
matrix W = alpha*I - beta encodes the OCOS structure directly).

For a topographic / convolutional version that scales to large grids see
the future ``cocolab_vwm.layers.topographic`` module (deferred to v0.2).
"""
from __future__ import annotations

import nengo
import numpy as np

from cocolab_vwm.core.dynamics import inhibition_matrix
from cocolab_vwm.core.params import OCOSParams


def make_ocos_layer(
    params: OCOSParams,
    n_neurons_per_node: int = 30,
    label: str = "ocos",
) -> nengo.Network:
    """Build a Nengo network implementing one OCOS layer.

    The state vector x of dimension `params.n_nodes` is represented by
    `n_neurons_per_node * n_nodes` neurons grouped into `n_nodes` ensembles
    (one per grid node). Recurrent excitation/inhibition is implemented as
    a connection between the ensembles with weights derived from the OCOS
    parameters.

    Parameters
    ----------
    params : OCOSParams
    n_neurons_per_node : int
        Neurons per grid node in the spiking ensemble. 30 is a reasonable
        default for prototyping; raise for higher representational fidelity.
    label : str
        Network label for visualisation.

    Returns
    -------
    net : nengo.Network with attributes:
        - net.input  : nengo.Node, shape (n_nodes,)
        - net.state  : nengo.networks.EnsembleArray representing x
        - net.output : nengo.Node, shape (n_nodes,) (passes F(x))
    """
    n_nodes = params.n_nodes
    beta = inhibition_matrix(params)

    with nengo.Network(label=label) as net:
        # Input port: external drive I_i applied during stimulus interval.
        net.input = nengo.Node(size_in=n_nodes, label="input")

        # State representation: one small ensemble per grid node.
        # Radius 1.5 covers the typical post-input fixed point of x in [0, 1].
        net.state = nengo.networks.EnsembleArray(
            n_neurons=n_neurons_per_node,
            n_ensembles=n_nodes,
            radius=1.5,
            label="state",
        )

        # Add a non-default output that computes F(x) on each ensemble.
        # In Nengo, EnsembleArray functions must be applied per-ensemble
        # via add_output, not via Connection(function=...) on the array's
        # passthrough output node.
        def Fx_scalar(x):
            return x / (1.0 + x) if x > 0 else 0.0

        net.state.add_output("Fx", Fx_scalar)

        # Recurrent connection implementing
        #   dx/dt = -x + alpha*F(x) - beta @ F(x) + I
        # Nengo's default synapse provides the integrator dynamics; we supply
        # the linear transform W = alpha*I - beta on top of F(x).
        W = params.alpha * np.eye(n_nodes) - beta

        nengo.Connection(
            net.state.Fx,
            net.state.input,
            transform=W,
            synapse=0.05,  # 50 ms exponential synapse, tau-style integrator
        )

        # External input adds directly to the ensemble inputs.
        nengo.Connection(net.input, net.state.input, synapse=None)

        # Convenience output: F(x) (so downstream modules see the activation,
        # not the raw membrane-like variable).
        net.output = nengo.Node(size_in=n_nodes, label="output")
        nengo.Connection(net.state.Fx, net.output, synapse=None)

    return net
