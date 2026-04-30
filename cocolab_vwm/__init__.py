"""cocolab_vwm: Hierarchical Visual Working Memory module for Nengo.

Built on the OCOS recurrent network with distance-dependent inhibition
(Sengupta, 2025) and the NSPFB-Op binding/operations formalism
(Sengupta, in prep).

Architectural layers
--------------------
core      : OCOS dynamics, cross-talk function, parameter dataclasses
layers    : single-layer and hierarchical layer constructors
control   : BG-thalamus gating, pulvinar blackboard (later versions)
tasks     : change-detection, delayed match-to-sample protocols
utils     : metrics (Hamming distance, recall probability)
"""
from cocolab_vwm._version import __version__

__all__ = ["__version__"]
