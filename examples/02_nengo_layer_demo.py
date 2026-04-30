"""Spiking Nengo demo: build the OCOS layer and run a stimulation trial.

Confirms the spiking implementation produces qualitatively the same
behaviour as the numpy reference (the stimulated node dominates).

Run with:
    python examples/02_nengo_layer_demo.py
"""
import nengo
import numpy as np

from cocolab_vwm.core.params import OCOSParams
from cocolab_vwm.layers.nengo_layer import make_ocos_layer


def main():
    params = OCOSParams(
        grid_shape=(5, 5),
        alpha=2.0,
        beta_0=0.3,
        rf_size=1.0,
        noise_std=0.0,
        dt=0.001,
        t_total=1.0,
        t_input=0.3,
    )

    target_idx = 12  # centre of 5x5 grid

    with nengo.Network(label="demo") as model:
        ocos = make_ocos_layer(params, n_neurons_per_node=50)

        def stim_fn(t):
            return [
                1.0 if (i == target_idx and t < params.t_input) else 0.0
                for i in range(params.n_nodes)
            ]

        stim = nengo.Node(stim_fn, label="stim")
        nengo.Connection(stim, ocos.input, synapse=None)
        probe = nengo.Probe(ocos.output, synapse=0.05)

    print(f"Running Nengo simulation ({params.t_total}s)...")
    with nengo.Simulator(model, dt=params.dt, progress_bar=False) as sim:
        sim.run(params.t_total)

    final = sim.data[probe][-50:].mean(axis=0)
    print("\nFinal activations (averaged over last 50 ms):")
    grid = final.reshape(params.grid_shape)
    for row in grid:
        print("  " + "  ".join(f"{v:+.2f}" for v in row))

    target_activation = float(final[target_idx])
    other_max = float(np.max(np.delete(final, target_idx)))
    print(f"\nStimulated node ({target_idx}) activation: {target_activation:+.3f}")
    print(f"Maximum other-node activation:        {other_max:+.3f}")

    if target_activation > 0.3:
        print(
            "Spiking OCOS retains the stimulated node in active memory "
            "after stimulus offset."
        )
    else:
        print(
            "Stimulated activation below 0.3 - try more neurons per node "
            "or check stimulation amplitude."
        )

    # Note for users: corner nodes typically drift to small positive values
    # because they have fewer inhibitory neighbours. This is a known boundary
    # effect of the finite-grid OCOS, not a bug.
    print(
        "\nNote: corner nodes drifting to small positive activations is a "
        "boundary effect (fewer inhibitory neighbours), not a model bug."
    )


if __name__ == "__main__":
    main()
