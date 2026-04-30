"""Task drivers (change-detection, etc.)."""
from cocolab_vwm.tasks.stimuli import (
    close_inputs,
    far_inputs,
    make_target_pattern,
    changed_indices,
)
from cocolab_vwm.tasks.change_detection import (
    run_change_detection,
    ChangeDetectionResult,
)

__all__ = [
    "close_inputs",
    "far_inputs",
    "make_target_pattern",
    "changed_indices",
    "run_change_detection",
    "ChangeDetectionResult",
]
