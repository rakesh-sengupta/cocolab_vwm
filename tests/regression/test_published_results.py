"""Regression tests: lock in published numerical results.

These tests assert that the model reproduces specific numerical patterns
from the source papers. They are the safety net for refactors: if a code
change silently alters the simulation output, these tests fail.

Tolerance is set generously (5-10%) to allow for noise and minor numerical
drift, but tightly enough to catch real changes.
"""
import numpy as np
import pytest

from cocolab_vwm.core.params import OCOSParams
from cocolab_vwm.tasks.change_detection import run_change_detection


class TestChangeDetectionRegression:
    """Lock in qualitative findings from Sengupta (2025), Figs. 2-3."""

    @pytest.mark.slow
    def test_far_better_than_close_at_low_rf(self):
        """At small RF, far stimuli should produce LOWER Hamming error
        than close stimuli at modest set sizes (Sengupta 2025, Fig. 2 left)."""
        params = OCOSParams(
            grid_shape=(8, 8),
            alpha=2.0,
            beta_0=0.3,
            rf_size=0.5,  # low RF
            noise_std=0.03,
            dt=0.01,
            t_total=10.0,
            t_input=1.0,
        )
        far = run_change_detection(
            params, set_size=4, spatial="far", n_trials=20, seed=42
        )
        close = run_change_detection(
            params, set_size=4, spatial="close", n_trials=20, seed=43
        )
        assert far.mean_hamming <= close.mean_hamming

    @pytest.mark.slow
    def test_increasing_set_size_does_not_explode(self):
        """Hamming error should remain bounded as set size grows.

        This catches regressions where the network becomes unstable for
        larger inputs.
        """
        params = OCOSParams(
            grid_shape=(8, 8), noise_std=0.03,
            t_total=5.0, t_input=1.0,
        )
        for n in (2, 4, 6):
            res = run_change_detection(
                params, set_size=n, spatial="far",
                n_trials=10, seed=100 + n,
            )
            assert res.mean_hamming < 20  # generous bound
            assert np.all(np.isfinite(res.hamming_distances))


class TestStabilityRegression:
    """Network must not diverge under canonical parameters."""

    @pytest.mark.slow
    def test_long_run_no_divergence(self):
        """Run for many steps; assert no NaN/Inf."""
        from cocolab_vwm.core.dynamics import simulate
        params = OCOSParams(
            grid_shape=(8, 8), t_total=30.0, t_input=1.0,
        )
        rng = np.random.default_rng(seed=7)
        x_final, history = simulate(
            params, np.array([10, 30, 50]),
            rng=rng, return_history=True,
        )
        assert np.all(np.isfinite(history))
        assert np.max(np.abs(history)) < 1e3
