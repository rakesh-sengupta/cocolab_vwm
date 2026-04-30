# cocolab_vwm

A Nengo-compatible module for hierarchical visual working memory, built on
the OCOS (On-Centre Off-Surround) recurrent neural network with
distance-dependent inhibition (Sengupta, 2025) and the NSPFB-Op
binding/operations formalism (Sengupta, in prep).

## Status

**v0.2.0** — adds per-layer OCOSParams, average-pool feedforward
(homology-surjective compression), and BG-thalamus action selection for the
four NSPFB-Op primitives (Q, U, Cmp, Prop).

Planned for v0.3: pulvinar blackboard, PFC slot module, semantic-pointer
feature binding implementing the NSPFB-Op operations over the workspace.

## Installation

From source (editable), recommended for development:

```bash
git clone https://github.com/yourusername/cocolab_vwm.git
cd cocolab_vwm
pip install -e ".[dev,plot]"
```

## Quick start

```python
from cocolab_vwm import OCOSParams
from cocolab_vwm.tasks import run_change_detection

params = OCOSParams(grid_shape=(8, 8), beta_0=0.3, rf_size=1.0)
result = run_change_detection(
    params, set_size=4, spatial="far", n_trials=100, seed=42,
)
print(f"Mean Hamming distance: {result.mean_hamming:.2f} "
      f"+/- {result.sem_hamming:.2f}")
```

## Project layout

```
cocolab_vwm/
    core/        # numpy reference dynamics, params, cross-talk
    layers/      # Nengo single-layer + hierarchical networks
    control/     # BG-thalamus gating, pulvinar blackboard (v0.2)
    tasks/       # change detection, stimuli generators
    utils/       # metrics
tests/
    unit/        # fast pure-numpy unit tests
    integration/ # Nengo simulation tests (slower)
    regression/  # lock published numerical results
examples/        # runnable demos
docs/            # sphinx documentation (v0.2)
scripts/         # parameter sweeps, figure-replication
```

## Running the tests

```bash
pytest                          # fast suite (unit + integration), ~10 s
pytest --slow                   # include slow regression tests, ~15 s
pytest tests/unit/              # unit tests only
pytest --cov=cocolab_vwm        # with coverage report
```

The `--slow` flag is provided by the Nengo pytest plugin and gates regression
tests that lock in published numerical results (these are slower because they
run many trials of the full numpy OCOS simulation).

## References

- Sengupta, R. (2025). Modeling visual working memory using recurrent
  on-center off-surround neural network with distance dependent inhibition.
- Sengupta, R. (in prep). When binding becomes easy: a topological account
  of tractability in working memory.

## License

MIT.
