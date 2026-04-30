"""Task drivers (change-detection, etc.)."""
from cocolab_vwm.tasks.change_detection import (
    ChangeDetectionResult,
    run_change_detection,
)
from cocolab_vwm.tasks.stimuli import (
    changed_indices,
    close_inputs,
    far_inputs,
    make_target_pattern,
)

__all__ = [
    "close_inputs",
    "far_inputs",
    "make_target_pattern",
    "changed_indices",
    "run_change_detection",
    "ChangeDetectionResult",
]
