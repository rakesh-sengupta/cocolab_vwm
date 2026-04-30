"""Integration tests: Nengo OCOS layer matches numpy reference qualitatively.

These run actual Nengo simulations, so they're slower than unit tests
(~1-3 seconds each). They live in tests/integration/ so the unit suite
stays fast.
"""
import numpy as np
import pytest
import nengo
from cocolab_vwm.core.params import OCOSParams
from cocolab_vwm.core.dynamics import simulate as numpy_simulate
from cocolab_vwm.layers.nengo_layer import make_ocos_layer


@pytest.fixture
def integration_params():
    """A small grid for fast Nengo runs."""
    return OCOSParams(
        grid_shape=(3, 3),
        alpha=2.0,
        beta_0=0.3,
        rf_size=1.0,
        noise_std=0.0,
        dt=0.001,
        t_total=0.5,
        t_input=0.2,
    )


class TestNengoOCOSLayer:

    def test_network_builds(self, integration_params):
        """Smoke test: the network constructs without error."""
        net = make_ocos_layer(integration_params)
        with net:
            assert net.input is not None
            assert net.output is not None

    def test_network_simulates(self, integration_params):
        """Smoke test: a Nengo Simulator runs end-to-end."""
        with nengo.Network() as model:
            ocos = make_ocos_layer(integration_params)
            input_node = nengo.Node(
                output=lambda t: [1.0 if i == 4 and t < 0.2 else 0.0
                                  for i in range(integration_params.n_nodes)]
            )
            nengo.Connection(input_node, ocos.input, synapse=None)
            probe = nengo.Probe(ocos.output, synapse=0.05)

        with nengo.Simulator(model, dt=integration_params.dt,
                             progress_bar=False) as sim:
            sim.run(integration_params.t_total)

        assert sim.data[probe].shape[0] > 0
        assert np.all(np.isfinite(sim.data[probe]))

    def test_stimulated_node_dominant(self, integration_params):
        """The stimulated node should end up with the highest activation
        (qualitative match to numpy reference)."""
        target_idx = 4  # centre of 3x3 grid
        with nengo.Network() as model:
            ocos = make_ocos_layer(
                integration_params, n_neurons_per_node=50
            )
            input_node = nengo.Node(
                output=lambda t: [
                    1.0 if i == target_idx and t < 0.2 else 0.0
                    for i in range(integration_params.n_nodes)
                ]
            )
            nengo.Connection(input_node, ocos.input, synapse=None)
            probe = nengo.Probe(ocos.output, synapse=0.05)

        with nengo.Simulator(model, dt=integration_params.dt,
                             progress_bar=False) as sim:
            sim.run(integration_params.t_total)

        final = sim.data[probe][-50:].mean(axis=0)  # average last 50 ms
        # Stimulated node should be at least as active as any other node.
        # (Spiking noise prevents an exact maximum, so allow a small slack.)
        assert final[target_idx] >= final.max() - 0.1
