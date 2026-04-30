"""Integration tests for the pooled hierarchy."""
import nengo
import numpy as np
import pytest

from cocolab_vwm.core.params import HierarchyParams, OCOSParams
from cocolab_vwm.layers.pooled_hierarchy import make_pooled_hierarchy


@pytest.fixture
def small_pooled_params():
    """4x4 lower -> 2x2 upper (pool_size=2)."""
    return HierarchyParams(
        layer_params=OCOSParams(
            grid_shape=(4, 4),
            alpha=2.0, beta_0=0.3, rf_size=1.0,
            noise_std=0.0, dt=0.001,
            t_total=0.5, t_input=0.15,
        ),
        cross_talk_B=5.0,
    )


class TestPooledHierarchy:

    def test_builds(self, small_pooled_params):
        net = make_pooled_hierarchy(
            small_pooled_params, pool_size=2,
            feedback_level=3, n_neurons_per_node=20,
        )
        assert net.feedback_gain is not None

    def test_pool_size_one_equivalent_to_identity(self, small_pooled_params):
        """pool_size=1 should give upper grid identical to lower grid."""
        net = make_pooled_hierarchy(
            small_pooled_params, pool_size=1,
            feedback_level=3, n_neurons_per_node=20,
        )
        # net.output_upper should have the same size as net.output_lower
        assert net.output_lower.size_in == net.output_upper.size_in == 16

    def test_pool_size_two_halves_each_dim(self, small_pooled_params):
        net = make_pooled_hierarchy(
            small_pooled_params, pool_size=2,
            feedback_level=3, n_neurons_per_node=20,
        )
        assert net.output_lower.size_in == 16
        assert net.output_upper.size_in == 4  # 4x4 -> 2x2

    def test_runs_end_to_end(self, small_pooled_params):
        p = small_pooled_params
        n_lower = p.get_layer_params(0).n_nodes
        with nengo.Network() as model:
            net = make_pooled_hierarchy(
                p, pool_size=2, feedback_level=3, n_neurons_per_node=20,
            )
            stim = nengo.Node(
                lambda t: [
                    1.0 if (i == 5 and t < p.layer_params.t_input) else 0.0
                    for i in range(n_lower)
                ]
            )
            nengo.Connection(stim, net.input, synapse=None)
            probe_lower = nengo.Probe(net.output_lower, synapse=0.05)
            probe_upper = nengo.Probe(net.output_upper, synapse=0.05)
        with nengo.Simulator(
            model, dt=p.layer_params.dt, progress_bar=False
        ) as sim:
            sim.run(p.layer_params.t_total)
        assert np.all(np.isfinite(sim.data[probe_lower]))
        assert np.all(np.isfinite(sim.data[probe_upper]))

    def test_pool_size_zero_raises(self, small_pooled_params):
        with pytest.raises(ValueError):
            make_pooled_hierarchy(small_pooled_params, pool_size=0)

    def test_non_divisible_pool_raises(self):
        """5x5 grid + pool_size=2 should raise (5 not divisible by 2)."""
        bad = HierarchyParams(
            layer_params=OCOSParams(
                grid_shape=(5, 5), t_total=0.2, dt=0.01, t_input=0.1
            )
        )
        with pytest.raises(ValueError):
            make_pooled_hierarchy(bad, pool_size=2)
