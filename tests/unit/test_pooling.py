"""Unit tests for the pooling / feedforward compression module."""
import numpy as np
import pytest

from cocolab_vwm.core.pooling import (
    average_pool_transform,
    max_pool,
    pool_grid_shape,
    pool_window_indices,
    winner_take_all,
)


class TestMaxPool:

    def test_2x2_pool_on_4x4_grid(self):
        """4x4 grid pooled with size 2 -> 2x2 grid; each cell = max of block."""
        # Block (0,0) = [0,1,4,5]; max = 5
        # Block (0,1) = [2,3,6,7]; max = 7
        # Block (1,0) = [8,9,12,13]; max = 13
        # Block (1,1) = [10,11,14,15]; max = 15
        a = np.arange(16).astype(float)
        out = max_pool(a, (4, 4), 2)
        np.testing.assert_array_equal(out, [5, 7, 13, 15])

    def test_pool_size_one_is_identity(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(max_pool(a, (2, 2), 1), a)

    def test_grid_shape_must_divide(self):
        with pytest.raises(ValueError):
            max_pool(np.zeros(9), (3, 3), 2)


class TestPoolGridShape:

    def test_basic(self):
        assert pool_grid_shape((8, 8), 2) == (4, 4)
        assert pool_grid_shape((4, 4), 4) == (1, 1)

    def test_must_divide(self):
        with pytest.raises(ValueError):
            pool_grid_shape((5, 5), 2)


class TestPoolWindowIndices:

    def test_4x4_pool_2_windows(self):
        windows = pool_window_indices((4, 4), 2)
        assert len(windows) == 4
        # First window: top-left 2x2 = indices 0, 1, 4, 5.
        assert sorted(windows[0]) == [0, 1, 4, 5]
        # Second window: top-right 2x2 = indices 2, 3, 6, 7.
        assert sorted(windows[1]) == [2, 3, 6, 7]
        # Third window: bottom-left = 8, 9, 12, 13.
        assert sorted(windows[2]) == [8, 9, 12, 13]
        # Fourth: bottom-right = 10, 11, 14, 15.
        assert sorted(windows[3]) == [10, 11, 14, 15]

    def test_windows_partition_input(self):
        """Every input index appears in exactly one window."""
        windows = pool_window_indices((6, 6), 3)
        all_idx = []
        for w in windows:
            all_idx.extend(w)
        assert sorted(all_idx) == list(range(36))


class TestAveragePoolTransform:

    def test_shape(self):
        W = average_pool_transform((4, 4), 2)
        assert W.shape == (4, 16)

    def test_row_sums_to_one(self):
        """Each upper unit's row sums to 1 (it's a true average)."""
        W = average_pool_transform((4, 4), 2)
        np.testing.assert_allclose(W.sum(axis=1), np.ones(4))

    def test_matches_average_pool(self):
        """W @ activity should equal block-wise mean."""
        a = np.arange(16).astype(float)
        W = average_pool_transform((4, 4), 2)
        # Block 0,0 = mean([0, 1, 4, 5]) = 2.5; block 0,1 mean([2,3,6,7])=4.5
        # Block 1,0 mean([8,9,12,13])=10.5; block 1,1 mean([10,11,14,15])=12.5
        np.testing.assert_allclose(W @ a, [2.5, 4.5, 10.5, 12.5])

    def test_pool_size_one_is_identity(self):
        W = average_pool_transform((3, 3), 1)
        np.testing.assert_allclose(W, np.eye(9))


class TestWinnerTakeAll:

    def test_one_hot(self):
        out = winner_take_all(np.array([0.1, 0.5, 0.3]))
        np.testing.assert_array_equal(out, [0.0, 1.0, 0.0])

    def test_ties_pick_first(self):
        """np.argmax returns the lowest tied index by convention."""
        out = winner_take_all(np.array([0.5, 0.5, 0.3]))
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0])
