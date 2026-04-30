"""Cross-talk function C(L) and feedback gating.

Implements the cross-talk function of Sengupta (2025):

    C(L) = A(L) * exp(-B * (1 - A(L))^2)

where A(L) is the fraction of the p-lattice visible at feedback level L.
This factor multiplies feedback transmission in the hierarchical model,
implementing the feedback-depth modulation that links the OCOS dynamics
to the FPT runtime in the binding paper (Sengupta, in prep).
"""
from __future__ import annotations

import math


def cross_talk(A: float, B: float = 5.0) -> float:
    """C(L) = A * exp(-B * (1 - A)^2).

    Parameters
    ----------
    A : float in [0, 1]
        Fraction of p-lattice visible.
    B : float
        Scaling constant; larger B = sharper transition near A = 1.

    Returns
    -------
    C : float in [0, 1]
        Cross-talk modulation factor.
    """
    if not 0.0 <= A <= 1.0:
        raise ValueError(f"A must be in [0, 1]; got {A}")
    if B < 0:
        raise ValueError(f"B must be non-negative; got {B}")
    return A * math.exp(-B * (1.0 - A) ** 2)


def uncertainty(A: float) -> float:
    """U(L) = 1 - A(L). Equation in Sengupta (2025), Section 4."""
    return 1.0 - A
