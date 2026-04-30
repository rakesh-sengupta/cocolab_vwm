"""Integration tests for the BG-thalamus action-selection gate."""
import nengo
import numpy as np
import pytest

from cocolab_vwm.control.bg_gate import make_bg_gate, make_op_selector


class TestBGGate:

    def test_builds(self):
        """Smoke test: BG gate constructs."""
        net = make_bg_gate(n_actions=4, n_neurons_per_action=30)
        assert net.utility is not None
        assert net.action is not None

    def test_minimum_two_actions(self):
        """A BG with one action makes no sense; should raise."""
        with pytest.raises(ValueError):
            make_bg_gate(n_actions=1)

    def test_winning_action_selected(self):
        """Action with highest utility should win in steady state."""
        with nengo.Network() as model:
            bg = make_bg_gate(n_actions=4, n_neurons_per_action=30)
            # Action 2 has the highest utility -> should win.
            util = nengo.Node(output=lambda t: [0.0, 0.2, 0.9, 0.1])
            nengo.Connection(util, bg.utility, synapse=None)
            probe = nengo.Probe(bg.action, synapse=0.05)
        with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
            sim.run(0.5)

        # Mean over the last 100 ms (steady state).
        final = sim.data[probe][-100:].mean(axis=0)
        winner = int(np.argmax(final))
        assert winner == 2, f"Expected action 2 to win; got {winner}"
        # Winner activation should dominate non-winners.
        non_winners = np.delete(final, 2)
        assert final[2] > 5 * non_winners.max() + 0.1

    def test_changing_input_changes_winner(self):
        """If the highest-utility action switches mid-run, the BG should
        eventually select the new winner."""
        with nengo.Network() as model:
            bg = make_bg_gate(n_actions=3, n_neurons_per_action=30)

            def util_fn(t):
                # First half: action 0 wins. Second half: action 2 wins.
                if t < 0.4:
                    return [0.9, 0.1, 0.0]
                else:
                    return [0.0, 0.1, 0.9]
            util = nengo.Node(output=util_fn)
            nengo.Connection(util, bg.utility, synapse=None)
            probe = nengo.Probe(bg.action, synapse=0.05)
        with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
            sim.run(0.8)

        # Sample around 0.3 s (action 0 should win) and 0.7 s
        # (action 2 should win).
        early = sim.data[probe][250:350].mean(axis=0)
        late = sim.data[probe][650:750].mean(axis=0)
        assert int(np.argmax(early)) == 0
        assert int(np.argmax(late)) == 2


class TestOpSelector:

    def test_default_op_names(self):
        """Default op_names matches the four NSPFB-Op primitives."""
        sel = make_op_selector(n_neurons_per_action=30)
        assert sel.op_names == ("query", "update", "compare", "propagate")
        for name in sel.op_names:
            assert hasattr(sel, name)

    def test_custom_op_names(self):
        sel = make_op_selector(
            op_names=("act_a", "act_b"),
            n_neurons_per_action=20,
        )
        assert hasattr(sel, "act_a")
        assert hasattr(sel, "act_b")

    def test_correct_channel_activates(self):
        """Setting utility for one operation should activate only its channel."""
        with nengo.Network() as model:
            sel = make_op_selector(n_neurons_per_action=30)
            # Utility favours 'compare' (index 2).
            util = nengo.Node(output=lambda t: [0.0, 0.0, 0.9, 0.0])
            nengo.Connection(util, sel.utility, synapse=None)
            probes = {
                name: nengo.Probe(getattr(sel, name), synapse=0.05)
                for name in sel.op_names
            }
        with nengo.Simulator(model, dt=0.001, progress_bar=False) as sim:
            sim.run(0.5)

        means = {name: sim.data[probes[name]][-100:].mean()
                 for name in sel.op_names}
        # Compare channel should dominate.
        winner = max(means, key=means.get)
        assert winner == "compare", f"Expected compare to win; means={means}"
        # And dominate clearly.
        for name, val in means.items():
            if name != "compare":
                assert val < 0.3, (
                    f"Non-winner {name} activation too high: {val:.3f}"
                )
