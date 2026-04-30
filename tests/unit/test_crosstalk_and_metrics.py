"""Unit tests for cross-talk and metrics."""
import math

import numpy as np
import pytest

from cocolab_vwm.core.crosstalk import cross_talk, uncertainty
from cocolab_vwm.utils.metrics import binarize, hamming_distance, recall_probability


class TestCrossTalk:

    def test_full_visibility_is_one(self):
        assert math.isclose(cross_talk(1.0, B=5.0), 1.0)
        assert math.isclose(cross_talk(1.0, B=100.0), 1.0)  # B-independent

    def test_zero_visibility_is_zero(self):
        assert cross_talk(0.0) == 0.0

    def test_monotonic_in_A(self):
        # For B > 0, C(A) is strictly increasing on [0, 1].
        for A1, A2 in [(0.1, 0.2), (0.3, 0.7), (0.5, 0.99)]:
            assert cross_talk(A1) < cross_talk(A2)

    def test_invalid_A_raises(self):
        with pytest.raises(ValueError):
            cross_talk(-0.1)
        with pytest.raises(ValueError):
            cross_talk(1.5)

    def test_invalid_B_raises(self):
        with pytest.raises(ValueError):
            cross_talk(0.5, B=-1.0)

    def test_uncertainty(self):
        assert uncertainty(0.3) == pytest.approx(0.7)


class TestMetrics:

    def test_binarize_threshold(self):
        a = np.array([0.0, 0.05, 0.1, 0.5, 1.0])
        np.testing.assert_array_equal(
            binarize(a, threshold=0.1), [0, 0, 1, 1, 1]
        )

    def test_hamming_zero_for_identical(self):
        a = np.array([0.0, 0.5, 1.0])
        assert hamming_distance(a, a) == 0

    def test_hamming_counts_differences(self):
        a = np.array([0.0, 0.5, 1.0])
        b = np.array([1.0, 0.5, 0.0])
        # binarised: [0,1,1] vs [1,1,0] => 2 differences
        assert hamming_distance(a, b) == 2

    def test_hamming_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            hamming_distance(np.zeros(3), np.zeros(4))

    def test_recall_perfect_match(self):
        a = np.array([0.0, 1.0, 1.0])
        assert recall_probability(a, a) == 1.0

    def test_recall_no_match(self):
        target = np.array([1.0, 1.0, 0.0])
        final = np.array([0.0, 0.0, 1.0])
        assert recall_probability(final, target) == 0.0

    def test_recall_partial_meets_threshold(self):
        target = np.array([1.0, 1.0, 1.0, 1.0])
        final = np.array([1.0, 1.0, 0.0, 0.0])  # 50% overlap
        assert recall_probability(final, target, overlap_min=0.5) == 1.0
        assert recall_probability(final, target, overlap_min=0.51) == 0.0
