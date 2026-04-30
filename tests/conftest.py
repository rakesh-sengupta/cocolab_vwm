"""Shared pytest fixtures.

Anything reused across multiple test files lives here so that test setup
remains DRY and changes to default parameter sets propagate uniformly.
"""
import numpy as np
import pytest

from cocolab_vwm.core.params import HierarchyParams, OCOSParams


@pytest.fixture
def small_params():
    """A small, fast OCOS parameter set for unit testing.

    4x4 grid keeps simulations <100 ms each, suitable for tests that loop
    over many parameter combinations.
    """
    return OCOSParams(
        grid_shape=(4, 4),
        alpha=2.0,
        beta_0=0.3,
        rf_size=1.0,
        noise_std=0.01,  # low noise for deterministic-ish tests
        dt=0.01,
        t_total=2.0,
        t_input=0.5,
    )


@pytest.fixture
def canonical_params():
    """The 8x8 grid used in Sengupta (2025) Figs. 2-3.

    Used for regression tests that lock in published numerical results.
    """
    return OCOSParams(
        grid_shape=(8, 8),
        alpha=2.0,
        beta_0=0.3,
        rf_size=1.0,
        noise_std=0.03,
        dt=0.01,
        t_total=10.0,
        t_input=1.0,
    )


@pytest.fixture
def hierarchy_params(small_params):
    """A small HierarchyParams for fast tests."""
    return HierarchyParams(
        layer_params=small_params,
        cross_talk_B=5.0,
        n_features=2,
    )


@pytest.fixture
def rng():
    """Seeded RNG for reproducible test runs."""
    return np.random.default_rng(seed=42)
