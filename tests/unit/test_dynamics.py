"""Unit tests for the OCOS dynamics module.

These exercise the pure-numpy reference implementation, since it is the
specification against which the Nengo realisation is compared.
"""
import numpy as np

from cocolab_vwm.core.dynamics import (
    activation,
    activation_derivative,
    grid_positions,
    inhibition_matrix,
    simulate,
    stability_bound,
)
from cocolab_vwm.core.params import OCOSParams


class TestActivation:

    def test_zero_below_threshold(self):
        assert activation(np.array([-1.0, 0.0])).tolist() == [0.0, 0.0]

    def test_saturating_above(self):
        # F(1) = 0.5, F(inf) -> 1
        assert np.isclose(activation(np.array([1.0]))[0], 0.5)
        assert activation(np.array([1e6]))[0] < 1.0

    def test_derivative_matches_numerical(self):
        x = np.array([0.5, 1.0, 2.0])
        eps = 1e-6
        analytic = activation_derivative(x)
        numerical = (activation(x + eps) - activation(x - eps)) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, atol=1e-4)


class TestGrid:

    def test_grid_shape_correctness(self):
        pos = grid_positions((3, 4))
        assert pos.shape == (12, 2)
        # First row should be (0,0), (0,1), ...
        np.testing.assert_array_equal(pos[0], [0, 0])
        np.testing.assert_array_equal(pos[3], [0, 3])
        np.testing.assert_array_equal(pos[4], [1, 0])


class TestInhibitionMatrix:

    def test_diagonal_is_zero(self):
        beta = inhibition_matrix(OCOSParams(grid_shape=(3, 3)))
        np.testing.assert_allclose(np.diag(beta), 0.0)

    def test_symmetric(self):
        beta = inhibition_matrix(OCOSParams(grid_shape=(3, 3)))
        np.testing.assert_allclose(beta, beta.T)

    def test_decreases_with_distance(self):
        """Eq. (1): closer nodes inhibit each other more strongly."""
        p = OCOSParams(grid_shape=(5, 5), rf_size=1.0)
        beta = inhibition_matrix(p)
        # Node 0 = (0,0). Node 1 = (0,1) distance 1; node 4 = (0,4) distance 4.
        assert beta[0, 1] > beta[0, 4]

    def test_rf_size_scales_inhibition(self):
        """Larger RF spreads inhibition further."""
        p_small = OCOSParams(grid_shape=(5, 5), rf_size=0.5)
        p_large = OCOSParams(grid_shape=(5, 5), rf_size=2.0)
        b_small = inhibition_matrix(p_small)
        b_large = inhibition_matrix(p_large)
        # Distant nodes should have stronger inhibition under large RF.
        assert b_large[0, 4] > b_small[0, 4]


class TestStabilityBound:

    def test_returns_positive(self):
        bound = stability_bound(OCOSParams())
        assert bound > 0

    def test_x_star_zero_is_trivially_stable(self):
        """At x*=0, F'(0)=0 (the network sits on the activation threshold),
        so the linearised system has eigenvalues -1 regardless of alpha and
        the bound is unbounded. This is correct behaviour."""
        bound = stability_bound(OCOSParams(), x_star=0.0)
        assert bound == float("inf")

    def test_x_star_positive_is_finite(self):
        """At positive x*, F'>0 and the stability bound is finite."""
        bound = stability_bound(OCOSParams(), x_star=0.5)
        assert np.isfinite(bound)
        assert bound > 0


class TestSimulate:

    def test_zero_input_zero_steady_state(self, small_params, rng):
        """No input, no noise => activity stays near zero."""
        p = OCOSParams(
            grid_shape=small_params.grid_shape,
            alpha=small_params.alpha,
            beta_0=small_params.beta_0,
            rf_size=small_params.rf_size,
            noise_std=0.0,
            dt=small_params.dt,
            t_total=small_params.t_total,
            t_input=small_params.t_input,
        )
        x_final, _ = simulate(p, np.array([], dtype=int), rng=rng)
        assert np.max(np.abs(x_final)) < 1e-6

    def test_input_drives_activity(self, small_params, rng):
        """Stimulating a node leaves it more active than its neighbours."""
        x_final, _ = simulate(
            small_params, np.array([5]), rng=rng
        )
        assert x_final[5] > 0  # stimulated node has positive activity
        # Most non-stimulated nodes should be near zero or inhibited.
        nonstim_mean = np.mean(np.delete(x_final, 5))
        assert x_final[5] > nonstim_mean

    def test_history_shape(self, small_params, rng):
        n_steps = int(round(small_params.t_total / small_params.dt))
        _, history = simulate(
            small_params, np.array([0]),
            rng=rng, return_history=True,
        )
        assert history.shape == (n_steps, small_params.n_nodes)

    def test_reproducibility_with_seed(self, small_params):
        """Same seed => identical trajectory."""
        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)
        x1, _ = simulate(small_params, np.array([0]), rng=rng1)
        x2, _ = simulate(small_params, np.array([0]), rng=rng2)
        np.testing.assert_allclose(x1, x2)

    def test_no_divergence_under_published_params(self, canonical_params):
        """Smoke test: published parameters never produce NaN or Inf."""
        rng = np.random.default_rng(seed=0)
        x_final, _ = simulate(
            canonical_params, np.array([10, 20, 30]), rng=rng
        )
        assert np.all(np.isfinite(x_final))
        assert np.max(np.abs(x_final)) < 100  # loose sanity bound
