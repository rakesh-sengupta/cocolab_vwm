"""Basal ganglia / thalamus action selection for VWM task control.

Wraps Nengo SPA's BasalGanglia + Thalamus action-selection circuit
(Stewart, Choo & Eliasmith 2010) into a controller that gates which
``operation'' is currently active over the VWM workspace.

In the v0.2 design the BG receives utility values for k candidate actions
and produces a one-hot output that selects which downstream gate is
opened. This is the substrate for implementing the four primitive WM
operations in NSPFB-Op (Sengupta in prep, Definition: Primitive WM
operations):

    Q   - Query: read a binding from the workspace
    U   - Update: write a new binding into the workspace
    Cmp - Compare: compare workspace state against a probe
    Prop- Propagate: arc-consistency / unit propagation

In v0.2 we wire these as four output channels; the actual implementation
of each operation against the workspace is deferred to v0.3 (semantic
pointers).

Usage
-----
::

    bg = make_bg_gate(n_actions=4, n_neurons_per_action=50)
    # bg.utility takes a 4-vector of action utilities
    # bg.action  is a 4-vector approximately one-hot over the winner

When wired into a hierarchy, ``bg.action`` becomes the gate-vector that
multiplies the input to whichever downstream module is meant to receive
the active operation.
"""
from __future__ import annotations

import nengo
import nengo.spa as legacy_spa  # noqa: F401  (older API; we use modern spa)
import numpy as np


def make_bg_gate(
    n_actions: int,
    n_neurons_per_action: int = 50,
    label: str = "bg_gate",
) -> nengo.Network:
    """Build a BG-thalamus action-selection network.

    Inspired by ``nengo_spa.networks.selection.BasalGanglia`` and the
    classic Stewart-Choo-Eliasmith (2010) BG model. We use Nengo's
    built-in ``BasalGanglia`` and ``Thalamus`` modules, which implement
    the canonical striatum -> GPi/SNr -> thalamus disinhibition circuit.

    Parameters
    ----------
    n_actions : int
        Number of competing actions/operations (e.g. 4 for the four
        NSPFB-Op primitives Q, U, Cmp, Prop).
    n_neurons_per_action : int
        Neurons per BG channel.
    label : str

    Returns
    -------
    net : nengo.Network with attributes:
        - net.utility : Node, size_in=n_actions
            Write to this with the current utility scores for each action.
        - net.action  : Node, size_in=n_actions
            Read from this; values approximate a one-hot vector with the
            winning action set near 1 and others near 0.
        - net.bg, net.thalamus : underlying Nengo networks (for inspection)
    """
    if n_actions < 2:
        raise ValueError(
            f"BG action selection needs at least 2 competitors; "
            f"got n_actions={n_actions}"
        )

    with nengo.Network(label=label) as net:
        # Utility input: external code writes per-action utility scores here.
        net.utility = nengo.Node(size_in=n_actions, label="utility")

        # Basal ganglia: implements the action-selection competition.
        # Nengo's networks.actionselection.BasalGanglia is the canonical
        # implementation of the Stewart-Choo-Eliasmith BG model.
        net.bg = nengo.networks.actionselection.BasalGanglia(
            n_neurons_per_ensemble=n_neurons_per_action,
            dimensions=n_actions,
        )

        # Thalamus: gates the BG output, producing approximately one-hot.
        net.thalamus = nengo.networks.actionselection.Thalamus(
            n_neurons_per_ensemble=n_neurons_per_action,
            dimensions=n_actions,
        )

        # Wire utility -> BG -> Thalamus -> action.
        nengo.Connection(net.utility, net.bg.input, synapse=None)
        nengo.Connection(net.bg.output, net.thalamus.input)

        net.action = nengo.Node(size_in=n_actions, label="action")
        nengo.Connection(net.thalamus.output, net.action, synapse=None)

    return net


def make_op_selector(
    op_names: tuple = ("query", "update", "compare", "propagate"),
    n_neurons_per_action: int = 50,
    label: str = "op_selector",
) -> nengo.Network:
    """Build a named-operation selector wrapping ``make_bg_gate``.

    This is a thin convenience layer over the BG gate that exposes the
    four NSPFB-Op primitives as named output channels. Each output channel
    is high (~1) when its corresponding action is selected, low (~0)
    otherwise.

    The output channels are intended to drive multiplicative gates on
    downstream operation modules (deferred to v0.3).

    Parameters
    ----------
    op_names : tuple of str
        Names for the operation channels. Default = NSPFB-Op primitives.
    n_neurons_per_action : int

    Returns
    -------
    net : nengo.Network with:
        - net.utility : Node, size_in=len(op_names)
        - For each name in op_names: a Node ``getattr(net, name)``,
          size_in=1, carrying the gate signal for that operation.
    """
    n = len(op_names)
    with nengo.Network(label=label) as net:
        net.bg_gate = make_bg_gate(
            n_actions=n,
            n_neurons_per_action=n_neurons_per_action,
        )
        net.utility = net.bg_gate.utility

        # Per-operation output nodes (size 1 each, one entry from `action`).
        for i, name in enumerate(op_names):
            ch = nengo.Node(size_in=1, label=name)
            # Pull the i-th component of `action` into the per-channel node.
            transform = np.zeros((1, n))
            transform[0, i] = 1.0
            nengo.Connection(
                net.bg_gate.action, ch,
                transform=transform, synapse=None,
            )
            setattr(net, name, ch)

        net.op_names = op_names

    return net
