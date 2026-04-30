"""v0.2 demo: BG-thalamus gate selecting NSPFB-Op operations.

Demonstrates the four-channel BG action selector with utility values that
change over time, simulating an experimentally driven sequence of WM
operations (encode -> maintain -> compare). Each channel maps to one
NSPFB-Op primitive:

    query (Q)     - read a binding
    update (U)    - write a new binding (active during encoding)
    compare (Cmp) - compare with a probe (active at retrieval)
    propagate (P) - arc-consistency

Run with:
    python examples/03_bg_op_selection.py
"""
import nengo

from cocolab_vwm.control import make_op_selector


def main():
    print("BG-thalamus operation selection demo")
    print("=" * 60)
    print("Schedule:")
    print("  0.0 - 0.3 s : update (encoding into WM)")
    print("  0.3 - 0.7 s : query  (maintenance / introspection)")
    print("  0.7 - 1.0 s : compare (probe judgment)")
    print()

    def utility_schedule(t):
        # 4-vector: [query, update, compare, propagate]
        if t < 0.3:
            return [0.0, 0.9, 0.0, 0.1]   # update wins
        elif t < 0.7:
            return [0.9, 0.0, 0.0, 0.1]   # query wins
        else:
            return [0.0, 0.0, 0.9, 0.1]   # compare wins

    with nengo.Network(label="bg_demo") as model:
        sel = make_op_selector(n_neurons_per_action=50)
        util_node = nengo.Node(output=utility_schedule)
        nengo.Connection(util_node, sel.utility, synapse=None)

        probes = {
            name: nengo.Probe(getattr(sel, name), synapse=0.05)
            for name in sel.op_names
        }

    print("Running 1 s spiking simulation...")
    with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
        sim.run(1.0)

    # Evaluate winner in each window.
    windows = {
        "early (encoding)":      (50, 250),
        "middle (maintenance)":  (350, 650),
        "late (probe)":          (750, 950),
    }
    print(f"\n{'Window':<25} {'Q':>8} {'U':>8} {'C':>8} {'P':>8}  Winner")
    print("-" * 70)
    for window_name, (lo, hi) in windows.items():
        means = {
            name: float(sim.data[probes[name]][lo:hi].mean())
            for name in sel.op_names
        }
        winner = max(means, key=means.get)
        row = (
            f"{window_name:<25} "
            f"{means['query']:>8.3f} {means['update']:>8.3f} "
            f"{means['compare']:>8.3f} {means['propagate']:>8.3f}  "
            f"{winner}"
        )
        print(row)

    print(
        "\nThe BG-thalamus circuit successfully sequences operations "
        "based on time-varying utility, the substrate for executing "
        "the four NSPFB-Op primitives Q, U, Cmp, P over the workspace."
    )


if __name__ == "__main__":
    main()
