# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0
- Max-pooling feedforward connections (homology-surjective compression).
- BG-thalamus action selection module (`control/bg_gate.py`).
- Pulvinar blackboard with semantic-pointer slots.
- Per-layer OCOSParams (currently shared in v0.1).
- Topographic ensemble option for large grids.

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
