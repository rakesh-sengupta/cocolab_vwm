"""Hierarchical VWM network with cross-talk gated feedback.

Stacks two OCOS layers (R_lower, R_upper) and adds a feedforward path and a
cross-talk-modulated feedback path. This is the v0.1 prototype: two layers,
one feature dimension, single-object encoding. The architecture extends
cleanly to the full R_V1...R_IT...PFC chain by stacking more layers and
adding semantic-pointer binding at the top (deferred to v0.2).

Cross-talk modulation
---------------------
Feedback transmission from R_upper -> R_lower is multiplied by a scalar
gain ``c_talk`` representing C(L) at the current feedback level. At L=1
(feedforward only) c_talk ~ 0 and the lower layer receives no top-down
information; at L=3 c_talk = 1 and feedback is at full strength.

The gain is exposed as a Nengo Node so it can be driven dynamically by a
controller (e.g., the BG-thalamus circuit gating attention).
"""
from __future__ import annotations

import nengo
import numpy as np

from cocolab_vwm.core.crosstalk import cross_talk
from cocolab_vwm.core.params import HierarchyParams
from cocolab_vwm.layers.nengo_layer import make_ocos_layer


def make_hierarchy(
    params: HierarchyParams,
    feedback_level: int = 3,
    n_neurons_per_node: int = 30,
    label: str = "hierarchy",
) -> nengo.Network:
    """Build a two-layer OCOS hierarchy with cross-talk-gated feedback.

    Parameters
    ----------
    params : HierarchyParams
    feedback_level : int
        Initial feedback level L; sets the static cross-talk gain.
        Override at runtime by writing to ``net.feedback_gain.input``.
    n_neurons_per_node : int
    label : str

    Returns
    -------
    net : nengo.Network with attributes:
        - net.input          : feedforward input to lower layer
        - net.lower          : R_lower OCOS layer (e.g. R_V4-equivalent)
        - net.upper          : R_upper OCOS layer (e.g. R_IT-equivalent)
        - net.feedback_gain  : Node holding scalar c_talk gain
        - net.output_lower   : output of R_lower
        - net.output_upper   : output of R_upper
    """
    c_talk_value = cross_talk(params.A(feedback_level), params.cross_talk_B)
    # Per-layer params: layer 0 = lower (V4-like), layer 1 = upper (IT-like).
    lower_params = params.get_layer_params(0)
    upper_params = params.get_layer_params(1)
    n_nodes = lower_params.n_nodes
    if upper_params.n_nodes != n_nodes:
        raise ValueError(
            "v0.2 still requires upper and lower layers to have the same "
            "n_nodes (no spatial pooling between layers yet). "
            f"Got lower n_nodes={n_nodes}, upper n_nodes={upper_params.n_nodes}."
        )

    with nengo.Network(label=label) as net:
        net.input = nengo.Node(size_in=n_nodes, label="ff_input")

        # Two OCOS layers, each with potentially different OCOSParams
        # (e.g. larger RF in upper layer to model V4 -> IT progression).
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

        # Feedforward: lower -> upper (full strength, identity transform).
        # In v0.2 this becomes a learned/structured projection (e.g.
        # max-pooling implementing the homology-surjective compression of
        # Sengupta in prep, Proposition: hierarchy via homology-surjective
        # maps).
        nengo.Connection(
            net.lower.output, net.upper.input,
            transform=1.0, synapse=0.05,
        )

        # Cross-talk gain node. Drives the feedback connection multiplicatively.
        net.feedback_gain = nengo.Node(
            output=lambda t, gain=c_talk_value: gain,
            size_out=1,
            label="c_talk",
        )

        # Feedback: upper -> lower, gated by c_talk. We implement the gate
        # as an EnsembleArray that computes elementwise product with the gain.
        gate = nengo.networks.Product(
            n_neurons=n_neurons_per_node, dimensions=n_nodes,
            label="feedback_gate",
        )
        nengo.Connection(net.upper.output, gate.input_a, synapse=0.05)
        # Broadcast scalar gain to each node dimension.
        nengo.Connection(
            net.feedback_gain, gate.input_b,
            transform=np.ones((n_nodes, 1)),
            synapse=None,
        )
        nengo.Connection(gate.output, net.lower.input, synapse=0.05)

        # External feedforward input adds to lower layer.
        nengo.Connection(net.input, net.lower.input, synapse=None)

        # Convenience output passthroughs.
        net.output_lower = nengo.Node(size_in=n_nodes, label="out_lower")
        net.output_upper = nengo.Node(size_in=n_nodes, label="out_upper")
        nengo.Connection(net.lower.output, net.output_lower, synapse=None)
        nengo.Connection(net.upper.output, net.output_upper, synapse=None)

    return net
