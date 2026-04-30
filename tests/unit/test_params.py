"""Unit tests for parameter dataclasses.

These tests guard the public parameter API. Since experiment scripts
instantiate OCOSParams/HierarchyParams directly, any breakage here breaks
every downstream user of the package.
"""
import math
import json
import pytest
from cocolab_vwm.core.params import OCOSParams, HierarchyParams


class TestOCOSParams:

    def test_default_construction(self):
        p = OCOSParams()
        assert p.grid_shape == (8, 8)
        assert p.n_nodes == 64

    def test_custom_grid_shape(self):
        p = OCOSParams(grid_shape=(4, 4))
        assert p.n_nodes == 16

    def test_immutability(self):
        p = OCOSParams()
        with pytest.raises(Exception):
            p.alpha = 99.0  # frozen dataclass

    def test_serialisation_roundtrip(self):
        p = OCOSParams(grid_shape=(4, 4), alpha=1.5)
        d = p.to_dict()
        assert d["grid_shape"] == (4, 4) or d["grid_shape"] == [4, 4]
        # JSON serialisation should at least produce valid JSON
        j = p.to_json()
        parsed = json.loads(j)
        assert parsed["alpha"] == 1.5


class TestHierarchyParams:

    def test_default_levels(self):
        h = HierarchyParams()
        assert h.A(1) == 0.3
        assert h.A(2) == 0.7
        assert h.A(3) == 1.0

    def test_unknown_level_raises(self):
        h = HierarchyParams()
        with pytest.raises(ValueError):
            h.A(99)

    def test_cross_talk_at_full_feedback(self):
        """At A=1, C(L)=1*exp(0)=1 regardless of B."""
        h = HierarchyParams(cross_talk_B=5.0)
        assert math.isclose(h.cross_talk(3), 1.0)

    def test_cross_talk_monotonic_in_L(self):
        """C should increase with feedback level."""
        h = HierarchyParams()
        c1, c2, c3 = h.cross_talk(1), h.cross_talk(2), h.cross_talk(3)
        assert c1 < c2 < c3
