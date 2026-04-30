"""Control layer: BG-thalamus gating, pulvinar blackboard.

Deferred to v0.2. v0.1 ships with a stub so that imports from
``cocolab_vwm.control`` don't fail in user code that is forward-prepared
for the BG gating scheduler.

Planned structure for v0.2:
    bg_gate.py         : nengo_spa.networks.BasalGanglia + Thalamus wired to
                         select which feature dimension is updated each step.
    pulvinar.py        : shared semantic-pointer workspace (the blackboard).
    pfc_slot.py        : top-level abstract memory sample, biases BG selection.
"""

__all__ = []
