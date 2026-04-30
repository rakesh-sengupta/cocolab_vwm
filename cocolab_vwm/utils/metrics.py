"""Performance metrics for VWM simulations.

Hamming distance and recall probability mirror the metrics reported in
Sengupta (2025) so that any new architectural variant can be benchmarked
against the published single-layer and hierarchical results.
"""
from __future__ import annotations
import numpy as np


def binarize(activity: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Binarise an activity vector at `threshold` (inclusive)."""
    return (activity >= threshold).astype(int)


def hamming_distance(
    a: np.ndarray, b: np.ndarray, threshold: float = 0.1
) -> int:
    """Hamming distance between binarised activity patterns.

    This is the metric used in Figs. 2 and 3 of Sengupta (2025) for
    error-rate analysis in the change-detection task.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return int(np.sum(binarize(a, threshold) != binarize(b, threshold)))


def recall_probability(
    final: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.1,
    overlap_min: float = 0.5,
) -> float:
    """Probability that the binarised final state overlaps the target enough.

    Returns 1.0 if (intersect / |target|) >= overlap_min, else 0.0.
    Suitable as a per-trial 0/1 score for averaging over many trials,
    matching the recall-probability metric in Sengupta (2025) Sec. 5.4.
    """
    fb = binarize(final, threshold)
    tb = binarize(target, threshold)
    n_target = int(tb.sum())
    if n_target == 0:
        return 1.0 if fb.sum() == 0 else 0.0
    overlap = int(np.sum(fb & tb)) / n_target
    return 1.0 if overlap >= overlap_min else 0.0
