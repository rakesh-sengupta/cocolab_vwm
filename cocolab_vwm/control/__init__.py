"""Control layer: BG-thalamus gating, pulvinar blackboard.

v0.2 introduces the BG-thalamus action selector (`bg_gate.py`) as the
substrate for routing the four NSPFB-Op primitives (Query, Update, Compare,
Propagate). The pulvinar blackboard and PFC slot are deferred to v0.3.
"""
from cocolab_vwm.control.bg_gate import make_bg_gate, make_op_selector

__all__ = ["make_bg_gate", "make_op_selector"]
