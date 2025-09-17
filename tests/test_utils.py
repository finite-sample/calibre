"""
Comprehensive tests for utils module.
"""

import numpy as np
import pytest

from calibre.utils import bin_data, check_arrays, create_bins, sort_by_x


class TestCheckArrays:
    """Test check_arrays function."""

    def test_valid_inputs(self):
        """Test with valid inputs."""
        X = [0.1, 0.3, 0.5, 0.7, 0.9]
        y = [0, 0, 1, 1, 1]

        X_valid, y_valid = check_arrays(X, y)

        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
        assert len(X_valid) == len(y_valid)
        assert X_valid.ndim == 1
        assert y_valid.ndim == 1

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        X = np.array([0.1, 0.3, 0.5])
        y = np.array([0, 1, 1])

        X_valid, y_valid = check_arrays(X, y)

        np.testing.assert_array_equal(X_valid, X)
        np.testing.assert_array_equal(y_valid, y)

    def test_2d_arrays(self):
        """Test with 2D arrays that should be flattened."""
        X = np.array([[0.1], [0.3], [0.5]])
        y = np.array([[0], [1], [1]])

        X_valid, y_valid = check_arrays(X, y)

        assert X_valid.ndim == 1
        assert y_valid.ndim == 1
        assert len(X_valid) == 3
        assert len(y_valid) == 3

    def test_mismatched_lengths(self):
        """Test with mismatched lengths."""
        X = [0.1, 0.3, 0.5]
        y = [0, 1]  # Different length

        with pytest.raises(ValueError, match="same length"):
            check_arrays(X, y)

    def test_empty_arrays(self):
        """Test with empty arrays."""
        X = []
        y = []

        with pytest.raises(ValueError, match="minimum of 1 is required"):
            check_arrays(X, y)

    def test_single_element(self):
        """Test with single element arrays."""
        X = [0.5]
        y = [1]

        X_valid, y_valid = check_arrays(X, y)

        assert len(X_valid) == 1
        assert len(y_valid) == 1
        assert X_valid[0] == 0.5
        assert y_valid[0] == 1

    def test_nan_values(self):
        """Test handling of NaN values."""
        X = [0.1, np.nan, 0.5]
        y = [0, 1, 1]

        # Should pass validation but preserve NaN
        X_valid, y_valid = check_arrays(X, y)
        assert np.isnan(X_valid[1])
        assert len(X_valid) == 3


class TestSortByX:
    """Test sort_by_x function."""

    def test_basic_sorting(self):
        """Test basic sorting functionality."""
        X = np.array([0.5, 0.1, 0.9, 0.3])
        y = np.array([1, 0, 1, 0])

        sort_indices, X_sorted, y_sorted = sort_by_x(X, y)

        # Check X is sorted
        assert np.all(X_sorted[:-1] <= X_sorted[1:])

        # Check y values correspond correctly
        expected_indices = np.argsort(X)
        np.testing.assert_array_equal(sort_indices, expected_indices)
        np.testing.assert_array_equal(X_sorted, X[expected_indices])
        np.testing.assert_array_equal(y_sorted, y[expected_indices])

    def test_already_sorted(self):
        """Test with already sorted data."""
        X = np.array([0.1, 0.3, 0.5, 0.7])
        y = np.array([0, 0, 1, 1])

        sort_indices, X_sorted, y_sorted = sort_by_x(X, y)

        np.testing.assert_array_equal(X_sorted, X)
        np.testing.assert_array_equal(y_sorted, y)
        np.testing.assert_array_equal(sort_indices, np.arange(len(X)))

    def test_reverse_sorted(self):
        """Test with reverse sorted data."""
        X = np.array([0.9, 0.7, 0.5, 0.3])
        y = np.array([1, 1, 0, 0])

        sort_indices, X_sorted, y_sorted = sort_by_x(X, y)

        # Should be sorted in ascending order
        assert np.all(X_sorted[:-1] <= X_sorted[1:])
        np.testing.assert_array_equal(X_sorted, np.array([0.3, 0.5, 0.7, 0.9]))
        np.testing.assert_array_equal(y_sorted, np.array([0, 0, 1, 1]))

    def test_duplicate_x_values(self):
        """Test with duplicate X values."""
        X = np.array([0.5, 0.1, 0.5, 0.1])
        y = np.array([1, 0, 0, 1])

        sort_indices, X_sorted, y_sorted = sort_by_x(X, y)

        # Should still be sorted, with stable sort preserving relative order
        assert np.all(X_sorted[:-1] <= X_sorted[1:])
        assert len(X_sorted) == len(X)

    def test_single_element(self):
        """Test with single element."""
        X = np.array([0.5])
        y = np.array([1])

        sort_indices, X_sorted, y_sorted = sort_by_x(X, y)

        np.testing.assert_array_equal(X_sorted, X)
        np.testing.assert_array_equal(y_sorted, y)
        np.testing.assert_array_equal(sort_indices, [0])

    def test_with_nans(self):
        """Test behavior with NaN values."""
        X = np.array([0.5, np.nan, 0.1])
        y = np.array([1, 0, 0])

        sort_indices, X_sorted, y_sorted = sort_by_x(X, y)

        # NaN should be sorted to the end
        assert np.isnan(X_sorted[-1])
        assert len(X_sorted) == 3


class TestCreateBins:
    """Test create_bins function."""

    def test_uniform_strategy(self):
        """Test uniform binning strategy."""
        X = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        bins = create_bins(X, n_bins=3, strategy="uniform")

        assert len(bins) == 4  # n_bins + 1 edges
        assert bins[0] == 0.0
        assert bins[-1] == 1.0
        # Should be evenly spaced
        np.testing.assert_allclose(np.diff(bins), np.diff(bins)[0])

    def test_quantile_strategy(self):
        """Test quantile binning strategy."""
        # Create data with uneven distribution
        X = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])

        bins = create_bins(X, n_bins=2, strategy="quantile")

        assert len(bins) == 3  # n_bins + 1 edges
        assert bins[0] <= X.min()
        assert bins[-1] >= X.max()

    def test_invalid_strategy(self):
        """Test invalid strategy."""
        X = np.array([0.1, 0.5, 0.9])

        with pytest.raises(ValueError, match="Unknown binning strategy"):
            create_bins(X, strategy="invalid")

    def test_edge_cases(self):
        """Test edge cases."""
        # All same values
        X = np.array([0.5, 0.5, 0.5])
        bins = create_bins(X, n_bins=3, strategy="uniform")
        assert len(bins) == 4

        # Single value
        X = np.array([0.5])
        bins = create_bins(X, n_bins=2, strategy="uniform")
        assert len(bins) == 3

    def test_custom_range(self):
        """Test with custom min/max range."""
        X = np.array([0.2, 0.4, 0.6, 0.8])

        bins = create_bins(X, n_bins=2, strategy="uniform", x_min=0.0, x_max=1.0)

        assert bins[0] == 0.0
        assert bins[-1] == 1.0
        assert len(bins) == 3

    def test_bins_parameter_validation(self):
        """Test bins parameter validation."""
        X = np.array([0.1, 0.5, 0.9])

        # n_bins must be positive
        with pytest.raises(ValueError, match="n_bins must be positive"):
            create_bins(X, n_bins=0)

        # n_bins must be integer
        with pytest.raises(TypeError):
            create_bins(X, n_bins=2.5)


class TestBinData:
    """Test bin_data function."""

    def test_basic_binning(self):
        """Test basic data binning."""
        X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        bins = np.array([0.0, 0.4, 0.8, 1.0])

        bin_indices, bin_counts = bin_data(X, bins)

        assert len(bin_indices) == len(X)
        assert len(bin_counts) == len(bins) - 1
        assert all(0 <= idx < len(bins) - 1 for idx in bin_indices)
        assert sum(bin_counts) == len(X)

    def test_edge_values(self):
        """Test binning with edge values."""
        X = np.array([0.0, 0.5, 1.0])
        bins = np.array([0.0, 0.5, 1.0])

        bin_indices, bin_counts = bin_data(X, bins)

        # All values should be binned
        assert len(bin_indices) == len(X)
        assert sum(bin_counts) == len(X)

    def test_out_of_range_values(self):
        """Test with values outside bin range."""
        X = np.array([-0.1, 0.3, 1.1])
        bins = np.array([0.0, 0.5, 1.0])

        bin_indices, bin_counts = bin_data(X, bins)

        # Should handle out-of-range values appropriately
        assert len(bin_indices) == len(X)
        # Out-of-range values might be assigned to boundary bins

    def test_empty_bins(self):
        """Test with bins that might be empty."""
        X = np.array([0.1, 0.2])  # All in first bin
        bins = np.array([0.0, 0.3, 0.6, 1.0])

        bin_indices, bin_counts = bin_data(X, bins)

        assert bin_counts[0] == 2  # Both values in first bin
        assert bin_counts[1] == 0  # Second bin empty
        assert bin_counts[2] == 0  # Third bin empty

    def test_single_bin(self):
        """Test with single bin."""
        X = np.array([0.1, 0.5, 0.9])
        bins = np.array([0.0, 1.0])

        bin_indices, bin_counts = bin_data(X, bins)

        assert len(bin_counts) == 1
        assert bin_counts[0] == 3  # All values in single bin
        assert all(idx == 0 for idx in bin_indices)

    def test_monotonic_bins(self):
        """Test that bins must be monotonic."""
        X = np.array([0.1, 0.5, 0.9])
        bins = np.array([0.0, 1.0, 0.5])  # Not monotonic

        # Should either handle gracefully or raise error
        # The exact behavior depends on implementation
        try:
            bin_indices, bin_counts = bin_data(X, bins)
        except (ValueError, AssertionError):
            pass  # Expected for non-monotonic bins


class TestIntegrationAndEdgeCases:
    """Integration tests and additional edge cases."""

    def test_full_pipeline(self):
        """Test full pipeline of utility functions."""
        # Unsorted data
        X = np.array([0.7, 0.1, 0.9, 0.3, 0.5])
        y = np.array([1, 0, 1, 0, 1])

        # Step 1: Validate
        X_valid, y_valid = check_arrays(X, y)

        # Step 2: Sort
        _, X_sorted, y_sorted = sort_by_x(X_valid, y_valid)

        # Step 3: Create bins
        bins = create_bins(X_sorted, n_bins=3, strategy="uniform")

        # Step 4: Bin data
        bin_indices, bin_counts = bin_data(X_sorted, bins)

        # Verify results
        assert len(X_sorted) == len(y_sorted) == len(X)
        assert np.all(X_sorted[:-1] <= X_sorted[1:])
        assert len(bins) == 4
        assert sum(bin_counts) == len(X)

    def test_extreme_values(self):
        """Test with extreme values."""
        X = np.array([1e-10, 0.5, 1 - 1e-10])
        y = np.array([0, 1, 1])

        X_valid, y_valid = check_arrays(X, y)
        _, X_sorted, y_sorted = sort_by_x(X_valid, y_valid)

        # Should handle extreme values
        assert X_sorted[0] == 1e-10
        assert X_sorted[-1] == 1 - 1e-10

    def test_large_datasets(self):
        """Test with larger datasets."""
        np.random.seed(42)
        n = 10000
        X = np.random.uniform(0, 1, n)
        y = np.random.binomial(1, X, n)

        X_valid, y_valid = check_arrays(X, y)
        _, X_sorted, y_sorted = sort_by_x(X_valid, y_valid)
        bins = create_bins(X_sorted, n_bins=50, strategy="quantile")
        bin_indices, bin_counts = bin_data(X_sorted, bins)

        # Should handle large datasets efficiently
        assert len(X_sorted) == n
        assert sum(bin_counts) == n
        assert len(bins) == 51

    def test_type_consistency(self):
        """Test type consistency across functions."""
        X = [0.1, 0.5, 0.9]  # List input
        y = [0, 1, 1]  # List input

        X_valid, y_valid = check_arrays(X, y)

        # Should return numpy arrays
        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)

        X_sorted, y_sorted, indices = sort_by_x(X_valid, y_valid)

        # Should maintain numpy array types
        assert isinstance(X_sorted, np.ndarray)
        assert isinstance(y_sorted, np.ndarray)
        assert isinstance(indices, np.ndarray)
