"""Integration tests for the two-layer hierarchy with cross-talk feedback."""
import numpy as np
import pytest
import nengo
from cocolab_vwm.core.params import OCOSParams, HierarchyParams
from cocolab_vwm.layers.hierarchy import make_hierarchy


@pytest.fixture
def small_hierarchy_params():
    return HierarchyParams(
        layer_params=OCOSParams(
            grid_shape=(3, 3),
            alpha=2.0, beta_0=0.3, rf_size=1.0,
            noise_std=0.0,
            dt=0.001,
            t_total=0.4,
            t_input=0.15,
        ),
        cross_talk_B=5.0,
    )


class TestHierarchy:

    def test_builds_at_each_feedback_level(self, small_hierarchy_params):
        """Should construct without error at L=1, 2, 3."""
        for L in (1, 2, 3):
            net = make_hierarchy(
                small_hierarchy_params,
                feedback_level=L,
                n_neurons_per_node=20,
            )
            assert net.feedback_gain is not None

    def test_runs_end_to_end(self, small_hierarchy_params):
        p = small_hierarchy_params
        n_nodes = p.layer_params.n_nodes
        with nengo.Network() as model:
            h = make_hierarchy(
                p, feedback_level=3, n_neurons_per_node=20
            )
            stim = nengo.Node(
                lambda t: [
                    1.0 if (i == 4 and t < p.layer_params.t_input) else 0.0
                    for i in range(n_nodes)
                ]
            )
            nengo.Connection(stim, h.input, synapse=None)
            probe_lower = nengo.Probe(h.output_lower, synapse=0.05)
            probe_upper = nengo.Probe(h.output_upper, synapse=0.05)

        with nengo.Simulator(model, dt=p.layer_params.dt,
                             progress_bar=False) as sim:
            sim.run(p.layer_params.t_total)

        assert np.all(np.isfinite(sim.data[probe_lower]))
        assert np.all(np.isfinite(sim.data[probe_upper]))

    def test_feedback_level_changes_dynamics(self, small_hierarchy_params):
        """Different feedback levels should produce measurably different
        post-stimulus activity (the C(L) gain matters)."""
        p = small_hierarchy_params
        n_nodes = p.layer_params.n_nodes

        results = {}
        for L in (1, 3):
            with nengo.Network() as model:
                h = make_hierarchy(
                    p, feedback_level=L, n_neurons_per_node=30
                )
                stim = nengo.Node(
                    lambda t: [
                        1.0 if (i == 4 and t < p.layer_params.t_input) else 0.0
                        for i in range(n_nodes)
                    ]
                )
                nengo.Connection(stim, h.input, synapse=None)
                probe_lower = nengo.Probe(h.output_lower, synapse=0.05)
            with nengo.Simulator(model, dt=p.layer_params.dt,
                                 progress_bar=False) as sim:
                sim.run(p.layer_params.t_total)
            # Average over the post-stimulus interval.
            stim_steps = int(p.layer_params.t_input / p.layer_params.dt)
            results[L] = sim.data[probe_lower][stim_steps:].mean(axis=0)

        # The two activation patterns should differ (C(1) << C(3)).
        diff = float(np.max(np.abs(results[1] - results[3])))
        assert diff > 0.01, (
            f"Feedback levels 1 and 3 produced indistinguishable activity "
            f"(max diff = {diff:.4f}); cross-talk gain may not be wired."
        )
