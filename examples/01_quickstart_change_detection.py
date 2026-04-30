"""Quick-start example: change-detection task with the numpy OCOS reference.

Reproduces the qualitative pattern from Sengupta (2025), Fig. 2: at small
receptive-field size, error rate is lower for far-spaced stimuli than for
close-spaced stimuli at modest set sizes.

Run with:
    python examples/01_quickstart_change_detection.py
"""
import numpy as np
from cocolab_vwm.core.params import OCOSParams
from cocolab_vwm.tasks import run_change_detection


def main():
    params = OCOSParams(
        grid_shape=(8, 8),
        alpha=2.0,
        beta_0=0.3,
        rf_size=1.0,
        noise_std=0.03,
        dt=0.01,
        t_total=10.0,
        t_input=1.0,
    )

    print(f"Running change-detection task with {params.n_nodes}-node grid")
    print("=" * 60)

    set_sizes = (2, 3, 4, 6)
    for spatial in ("close", "far"):
        print(f"\n  Spatial condition: {spatial}")
        print("  set_size  mean_hamming  +/- SEM")
        for n in set_sizes:
            try:
                res = run_change_detection(
                    params, set_size=n, spatial=spatial,
                    n_trials=30, seed=10 * n,
                )
                print(f"  {n:>8}  {res.mean_hamming:>12.2f}  {res.sem_hamming:.2f}")
            except ValueError as e:
                # close_inputs only supports up to 4 items
                print(f"  {n:>8}  (skipped: {e})")


if __name__ == "__main__":
    main()
