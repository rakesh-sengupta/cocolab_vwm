# Contributing to cocolab_vwm

This document captures the development workflow used to keep `cocolab_vwm`
extensible without breaking existing behaviour. The rules are pragmatic, not
ceremonial — every one of them earns its keep.

## Branching model

We use a lightweight version of git-flow:

- `main`: always corresponds to the latest tagged release. Never commit
  directly. Only release tags merge in.
- `develop`: integration branch. Feature branches merge here. CI must be
  green on every push.
- `feature/<short-name>`: feature branches off `develop`. Merge back via PR.
- `fix/<short-name>`: bug-fix branches.
- `experiment/<name>`: throwaway branches for exploratory work that may
  never land. Tag them with the date and description; delete when done.

## Versioning

Semantic versioning, single source in `cocolab_vwm/_version.py`:

- **PATCH** (0.1.0 → 0.1.1): bug fix, no API change, no behaviour change in
  the published parameter regime.
- **MINOR** (0.1.0 → 0.2.0): new features (new module, new task, new layer
  type) that are *additive*. Existing scripts using the old API must keep
  working.
- **MAJOR** (0.x → 1.0): breaking API change. Avoid until the architecture
  has settled.

Update `CHANGELOG.md` in the same commit that bumps the version.

## Adding a feature without breaking older work

This is the central question. Three rules:

1. **Add, don't replace.** New code path → new module or new function. If
   `make_ocos_layer` ever needs a different signature, add
   `make_ocos_layer_v2` and deprecate the old one with a `DeprecationWarning`
   for at least one minor version before removal.
2. **Lock the old behaviour with a regression test.** Before changing any
   numerics, add a test in `tests/regression/` that captures the current
   output. Then refactor. If the test fails, decide explicitly whether the
   change is intentional (update the test + bump version) or a bug
   (revert).
3. **Keep parameter dataclasses backward-compatible.** Add new fields with
   safe defaults. Never reorder or rename existing fields without a major
   version bump.

## Testing protocol

Three test tiers, each serving a different purpose:

| Tier            | Location                  | Speed   | What it catches                          |
|-----------------|---------------------------|---------|------------------------------------------|
| Unit            | `tests/unit/`             | <1 s ea | API correctness, edge cases              |
| Integration     | `tests/integration/`      | 1-5 s   | Nengo build/sim works end-to-end         |
| Regression      | `tests/regression/`       | 10-60 s | Numerical results match published figures|

Mark slow tests with `@pytest.mark.slow`. CI runs fast tests on every push;
slow tests run on one Python version per PR and on every release.

**Before any commit:**

```bash
pytest                         # fast tier; must pass
ruff check cocolab_vwm tests   # must pass
```

Or even better — install `pre-commit` once and let it run these checks
automatically on every `git commit`:

```bash
pip install pre-commit
pre-commit install             # one-time setup per clone
```

The hooks defined in `.pre-commit-config.yaml` run ruff (lint + format),
trailing-whitespace cleanup, and a guard against accidentally committing
large binary files. To bypass them in an emergency: `git commit --no-verify`.

**Before any release:**

```bash
pytest --slow                  # full suite, including regression tier
ruff check cocolab_vwm tests
# update CHANGELOG.md and bump _version.py
git tag v0.x.y && git push --tags
```

The `--slow` flag is provided by the Nengo pytest plugin; without it, slow
regression tests are skipped by default to keep developer-loop iteration fast.

## Coding conventions

- **Type hints** on all public functions. `mypy` is in the dev deps; we don't
  enforce strict mode in CI yet, but please don't introduce regressions.
- **Docstrings**: NumPy style, with `Parameters` / `Returns` / `Notes`
  sections. The docstring should cite the equation/proposition number from
  the source paper when relevant — this is the trail back to the theory.
- **Keep `core/` Nengo-free**. Pure numpy. `layers/` and `control/` are the
  only places that import nengo. This separation is what lets us
  unit-test the science without booting a Nengo simulator.
- **Reproducibility**: every function that uses randomness takes an `rng`
  argument or a `seed`. No global numpy seeds inside library code.

## Adding a new layer type or controller

Skeleton:

1. Create `cocolab_vwm/<subpackage>/<new_module>.py`.
2. Add a numpy reference function in `cocolab_vwm/core/` if the maths is
   nontrivial — this is the testable specification.
3. Add unit tests in `tests/unit/test_<module>.py`.
4. Add an integration test if the new module composes with Nengo.
5. Add an example script in `examples/` showing canonical usage.
6. Update `__init__.py` to expose the public API.
7. Update README and CHANGELOG.

## Pull request checklist

- [ ] Tests added/updated for new behaviour.
- [ ] CHANGELOG.md updated.
- [ ] Docstrings reference the relevant paper equation/proposition.
- [ ] No mutation of existing public function signatures (or version bump if so).
- [ ] CI green.
