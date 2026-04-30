# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.3.0
- True (nonlinear) max-pool variant via `MaxPoolNode`.
- Pulvinar blackboard with semantic-pointer slots.
- PFC slot module (top-level abstract memory sample).
- Semantic-pointer feature binding implementing the four NSPFB-Op primitives
  Q, U, Cmp, Prop over the workspace.
- Topographic ensemble option for large grids.

## [0.2.0] - 2026-04-30

### Added
- `HierarchyParams.per_layer_params` for per-layer OCOSParams overrides
  (e.g. larger RF in upper layers to model V4 -> IT progression).
- `HierarchyParams.get_layer_params(idx)` accessor.
- `core/pooling.py`: numpy reference for max-pool, average-pool transform,
  pool window indices, and global WTA. These realise the homology-surjective
  feedforward maps from the binding paper (Sengupta in prep).
- `layers/pooled_hierarchy.py`: `make_pooled_hierarchy` builds a two-layer
  hierarchy with average-pool feedforward and transposed broadcast feedback.
  Upper-layer grid is automatically reduced by `pool_size`.
- `control/bg_gate.py`: `make_bg_gate` (BG-thalamus action selector) and
  `make_op_selector` (named four-channel selector for the NSPFB-Op primitives
  Q, U, Cmp, Prop). Built on `nengo.networks.actionselection`.
- 26 new tests (13 pooling unit + 6 pooled-hierarchy integration + 7 BG-gate
  integration). Total: 75 tests across the suite.
- `examples/03_bg_op_selection.py` demonstrates BG sequencing
  `update -> query -> compare` over a 1 s simulation.

### Changed
- `make_hierarchy` now uses `params.get_layer_params(0/1)` internally.
  Backward-compatible: existing callers that rely on `layer_params` see
  no behaviour change.
- README and CONTRIBUTING document the `--slow` flag for regression tests
  (interaction with the Nengo pytest plugin).

### Fixed
- Eliminated divide-by-zero `RuntimeWarning` in `activation` and
  `activation_derivative` by masking before the division (semantically
  equivalent; cleans up test logs).
- Stability test split into `x_star=0` (trivially stable) and
  `x_star>0` (finite bound) cases.

## [0.1.0] - 2026-04-30

### Added
- `OCOSParams` and `HierarchyParams` frozen dataclasses (`core/params.py`).
- Pure-numpy reference dynamics: `simulate`, `inhibition_matrix`,
  `stability_bound` (`core/dynamics.py`).
- Cross-talk function `C(L)` (`core/crosstalk.py`).
- Single OCOS Nengo layer (`layers/nengo_layer.py`).
- Two-layer hierarchy with cross-talk gated feedback (`layers/hierarchy.py`).
- Change-detection task driver and stimulus generators (`tasks/`).
- Hamming distance and recall metrics (`utils/metrics.py`).
- Unit, integration, and regression test suites.
- pyproject.toml, ruff/mypy/pytest config, GitHub Actions CI.
