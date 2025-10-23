"""
Comprehensive tests for utils module.
"""

import numpy as np
import pytest

from calibre.utils import check_arrays, sort_by_x


class TestCheckArrays:
    """Test check_arrays function."""

    def test_valid_inputs(self):
        """Test with valid inputs."""
        X = [0.1, 0.3, 0.5, 0.7, 0.9]
        y = [0, 0, 1, 1, 1]

        X_valid, y_valid = check_arrays(X, y)

        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
        assert len(X_valid) == len(y_valid) == 5
        assert X_valid.shape == (5,)
        assert y_valid.shape == (5,)

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        X = np.array([0.2, 0.4, 0.6, 0.8])
        y = np.array([0, 1, 1, 0])

        X_valid, y_valid = check_arrays(X, y)

        np.testing.assert_array_equal(X_valid, X)
        np.testing.assert_array_equal(y_valid, y)

    def test_2d_arrays(self):
        """Test with 2D arrays (should be flattened to 1D)."""
        X = np.array([[0.1], [0.3], [0.5]])
        y = np.array([[0], [1], [1]])

        X_valid, y_valid = check_arrays(X, y)

        assert X_valid.shape == (3,)
        assert y_valid.shape == (3,)
        np.testing.assert_array_equal(X_valid, [0.1, 0.3, 0.5])
        np.testing.assert_array_equal(y_valid, [0, 1, 1])

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        X = [0.1, 0.3, 0.5]
        y = [0, 1]  # Different length

        with pytest.raises(
            ValueError, match="Input arrays X and y must have the same length"
        ):
            check_arrays(X, y)

    def test_empty_arrays(self):
        """Test with empty arrays."""
        X = []
        y = []

        with pytest.raises(ValueError, match="Found array with 0 sample"):
            check_arrays(X, y)

    def test_single_element(self):
        """Test with single-element arrays."""
        X = [0.5]
        y = [1]

        X_valid, y_valid = check_arrays(X, y)

        assert len(X_valid) == len(y_valid) == 1
        assert X_valid[0] == 0.5
        assert y_valid[0] == 1

    def test_nan_values(self):
        """Test with NaN values (should be allowed with force_all_finite='allow-nan')."""
        X = [0.1, np.nan, 0.5]
        y = [0, 1, 1]

        X_valid, y_valid = check_arrays(X, y)

        assert len(X_valid) == 3
        assert np.isnan(X_valid[1])
        np.testing.assert_array_equal(y_valid, [0, 1, 1])


class TestSortByX:
    """Test sort_by_x function."""

    def test_basic_sorting(self):
        """Test basic sorting functionality."""
        X = np.array([0.3, 0.1, 0.5, 0.2, 0.4])
        y = np.array([1, 0, 1, 0, 1])

        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        expected_X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected_y = np.array([0, 0, 1, 1, 1])

        np.testing.assert_array_equal(X_sorted, expected_X)
        np.testing.assert_array_equal(y_sorted, expected_y)

        # Check that sort indices are correct
        np.testing.assert_array_equal(X[sort_idx], X_sorted)
        np.testing.assert_array_equal(y[sort_idx], y_sorted)

    def test_already_sorted(self):
        """Test with already sorted arrays."""
        X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = np.array([0, 0, 1, 1, 1])

        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        np.testing.assert_array_equal(X_sorted, X)
        np.testing.assert_array_equal(y_sorted, y)
        np.testing.assert_array_equal(sort_idx, np.arange(len(X)))

    def test_reverse_sorted(self):
        """Test with reverse sorted arrays."""
        X = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        y = np.array([1, 1, 1, 0, 0])

        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        expected_X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected_y = np.array([0, 0, 1, 1, 1])

        np.testing.assert_array_equal(X_sorted, expected_X)
        np.testing.assert_array_equal(y_sorted, expected_y)

    def test_duplicate_x_values(self):
        """Test with duplicate X values."""
        X = np.array([0.3, 0.1, 0.3, 0.2, 0.1])
        y = np.array([1, 0, 2, 0, 3])

        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        # X should be sorted
        assert np.all(X_sorted[:-1] <= X_sorted[1:])
        # Check that we got all values
        assert len(X_sorted) == len(X)
        assert len(y_sorted) == len(y)

    def test_single_element(self):
        """Test with single element arrays."""
        X = np.array([0.5])
        y = np.array([1])

        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        np.testing.assert_array_equal(X_sorted, X)
        np.testing.assert_array_equal(y_sorted, y)
        np.testing.assert_array_equal(sort_idx, [0])

    def test_with_nans(self):
        """Test with NaN values (they should be sorted to the end)."""
        X = np.array([0.3, np.nan, 0.1, 0.5])
        y = np.array([1, 2, 0, 1])

        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        # Non-NaN values should be sorted, NaN should be at the end
        assert X_sorted[0] == 0.1
        assert X_sorted[1] == 0.3
        assert X_sorted[2] == 0.5
        assert np.isnan(X_sorted[3])

        # Check corresponding y values
        assert y_sorted[0] == 0  # corresponding to X=0.1
        assert y_sorted[1] == 1  # corresponding to X=0.3
        assert y_sorted[2] == 1  # corresponding to X=0.5
        assert y_sorted[3] == 2  # corresponding to X=nan


class TestValidationEdgeCases:
    """Test edge cases for validation functions."""

    def test_large_arrays(self):
        """Test with large arrays."""
        n = 10000
        X = np.random.rand(n)
        y = np.random.randint(0, 2, n)

        X_valid, y_valid = check_arrays(X, y)
        sort_idx, X_sorted, y_sorted = sort_by_x(X, y)

        assert len(X_valid) == n
        assert len(y_valid) == n
        assert len(X_sorted) == n
        assert len(y_sorted) == n

        # Check that sorting worked
        assert np.all(X_sorted[:-1] <= X_sorted[1:])

    def test_type_consistency(self):
        """Test that output types are consistent."""
        X_list = [0.1, 0.2, 0.3]
        y_list = [0, 1, 1]

        X_valid, y_valid = check_arrays(X_list, y_list)
        sort_idx, X_sorted, y_sorted = sort_by_x(X_valid, y_valid)

        # All outputs should be numpy arrays
        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
        assert isinstance(sort_idx, np.ndarray)
        assert isinstance(X_sorted, np.ndarray)
        assert isinstance(y_sorted, np.ndarray)
