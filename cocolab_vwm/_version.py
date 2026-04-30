"""Single source of truth for package version.

Follow semantic versioning: MAJOR.MINOR.PATCH
- MAJOR: incompatible API change
- MINOR: backward-compatible feature additions
- PATCH: backward-compatible bug fixes

v0.1.0 = initial OCOS layer + simple hierarchy + change detection task.
v0.2.0 = per-layer params, average-pool feedforward (homology-surjective
         compression), BG-thalamus action selection for NSPFB-Op gating.
"""
__version__ = "0.2.0"
