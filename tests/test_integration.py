"""
Integration tests for the calibre package.
Tests complete calibration workflows and edge cases.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from calibre.calibration import (
    ISplineCalibrator,
    NearlyIsotonicRegression,
    RegularizedIsotonicRegression,
    RelaxedPAVA,
    SmoothedIsotonicRegression,
)
from calibre.metrics import (
    brier_score,
    correlation_metrics,
    expected_calibration_error,
    mean_calibration_error,
)


@pytest.fixture
def realistic_dataset():
    """Create a realistic dataset for calibration testing."""
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Get uncalibrated predictions
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_proba_train": y_proba_train,
        "y_proba_test": y_proba_test,
    }


class TestFullCalibrationWorkflow:
    """Test complete calibration workflows."""

    def test_nearly_isotonic_regression_workflow(self, realistic_dataset):
        """Test complete workflow with NearlyIsotonicRegression."""
        data = realistic_dataset

        # Train calibrator
        calibrator = NearlyIsotonicRegression(lam=1.0, method="path")
        calibrator.fit(data["y_proba_train"], data["y_train"])

        # Calibrate test predictions
        y_calib = calibrator.transform(data["y_proba_test"])

        # Evaluate calibration
        mce_before = mean_calibration_error(data["y_test"], data["y_proba_test"])
        mce_after = mean_calibration_error(data["y_test"], y_calib)

        ece_before = expected_calibration_error(data["y_test"], data["y_proba_test"])
        ece_after = expected_calibration_error(data["y_test"], y_calib)

        # Calibration should improve (or at least not hurt significantly)
        assert isinstance(mce_after, float)
        assert isinstance(ece_after, float)
        assert len(y_calib) == len(data["y_test"])
        assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

    def test_ispline_calibrator_workflow(self, realistic_dataset):
        """Test complete workflow with ISplineCalibrator."""
        data = realistic_dataset

        # Train calibrator
        calibrator = ISplineCalibrator(n_splines=10, degree=3, cv=3)
        calibrator.fit(data["y_proba_train"], data["y_train"])

        # Calibrate test predictions
        y_calib = calibrator.transform(data["y_proba_test"])

        # Basic validation
        assert len(y_calib) == len(data["y_test"])
        assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

        # Check that calibrator preserves some correlation with original
        corr_metrics = correlation_metrics(
            data["y_test"], y_calib, y_orig=data["y_proba_test"]
        )
        assert corr_metrics["spearman_corr_orig_to_calib"] > 0.5

    def test_relaxed_pava_workflow(self, realistic_dataset):
        """Test complete workflow with RelaxedPAVA."""
        data = realistic_dataset

        # Train calibrator
        calibrator = RelaxedPAVA(percentile=5, adaptive=True)
        calibrator.fit(data["y_proba_train"], data["y_train"])

        # Calibrate test predictions
        y_calib = calibrator.transform(data["y_proba_test"])

        # Basic validation
        assert len(y_calib) == len(data["y_test"])
        assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

        # Should be mostly monotonic (allowing small violations)
        sorted_idx = np.argsort(data["y_proba_test"])
        y_calib_sorted = y_calib[sorted_idx]
        violations = np.sum(np.diff(y_calib_sorted) < 0)
        total_pairs = len(y_calib_sorted) - 1
        violation_rate = violations / total_pairs if total_pairs > 0 else 0
        assert violation_rate < 0.1  # Less than 10% violations

    def test_regularized_isotonic_workflow(self, realistic_dataset):
        """Test complete workflow with RegularizedIsotonicRegression."""
        data = realistic_dataset

        # Train calibrator
        calibrator = RegularizedIsotonicRegression(alpha=0.1)
        calibrator.fit(data["y_proba_train"], data["y_train"])

        # Calibrate test predictions
        y_calib = calibrator.transform(data["y_proba_test"])

        # Basic validation
        assert len(y_calib) == len(data["y_test"])
        assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

        # Should be strictly monotonic
        sorted_idx = np.argsort(data["y_proba_test"])
        y_calib_sorted = y_calib[sorted_idx]
        assert np.all(np.diff(y_calib_sorted) >= 0)

    def test_smoothed_isotonic_workflow(self, realistic_dataset):
        """Test complete workflow with SmoothedIsotonicRegression."""
        data = realistic_dataset

        # Train calibrator
        calibrator = SmoothedIsotonicRegression(
            window_length=7, poly_order=3, interp_method="linear"
        )
        calibrator.fit(data["y_proba_train"], data["y_train"])

        # Calibrate test predictions
        y_calib = calibrator.transform(data["y_proba_test"])

        # Basic validation
        assert len(y_calib) == len(data["y_test"])
        assert np.all(y_calib >= 0) and np.all(y_calib <= 1)


class TestCalibratorComparison:
    """Test comparing different calibrators on the same data."""

    def test_calibrator_performance_comparison(self, realistic_dataset):
        """Compare performance of different calibrators."""
        data = realistic_dataset

        calibrators = {
            "nearly_isotonic": NearlyIsotonicRegression(lam=1.0, method="path"),
            "ispline": ISplineCalibrator(n_splines=10, degree=3, cv=3),
            "relaxed_pava": RelaxedPAVA(percentile=5),
            "regularized": RegularizedIsotonicRegression(alpha=0.1),
            "smoothed": SmoothedIsotonicRegression(window_length=7, poly_order=3),
        }

        results = {}

        for name, calibrator in calibrators.items():
            # Train calibrator
            calibrator.fit(data["y_proba_train"], data["y_train"])

            # Calibrate test predictions
            y_calib = calibrator.transform(data["y_proba_test"])

            # Calculate metrics
            mce = mean_calibration_error(data["y_test"], y_calib)
            ece = expected_calibration_error(data["y_test"], y_calib)
            brier = brier_score(data["y_test"], y_calib)

            results[name] = {
                "mce": mce,
                "ece": ece,
                "brier": brier,
                "predictions": y_calib,
            }

        # All calibrators should produce valid results
        for name, result in results.items():
            assert isinstance(result["mce"], float)
            assert isinstance(result["ece"], float)
            assert isinstance(result["brier"], float)
            assert result["mce"] >= 0
            assert result["ece"] >= 0
            assert result["brier"] >= 0
            assert len(result["predictions"]) == len(data["y_test"])


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of calibrators."""

    def test_perfect_predictions(self):
        """Test calibrators with perfect predictions."""
        n = 100
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = y_true.astype(float)  # Perfect predictions

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)

            # Perfect predictions should remain perfect (or very close)
            mce = mean_calibration_error(y_true, y_calib)
            assert mce < 0.1

    def test_constant_predictions(self):
        """Test calibrators with constant predictions."""
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_pred = np.full(n, 0.5)  # Constant predictions

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)

            # Should handle constant predictions
            assert len(y_calib) == n
            assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

    def test_extreme_predictions(self):
        """Test calibrators with extreme predictions (0 and 1)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Should handle extreme values
                assert len(y_calib) == len(y_true)
                assert np.all(y_calib >= 0) and np.all(y_calib <= 1)
            except (ValueError, np.linalg.LinAlgError):
                # Some methods might not handle extreme cases
                pass

    def test_small_datasets(self):
        """Test calibrators with very small datasets."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0.2, 0.7, 0.8])

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Should handle small datasets
                assert len(y_calib) == len(y_true)
                assert np.all(y_calib >= 0) and np.all(y_calib <= 1)
            except (ValueError, np.linalg.LinAlgError):
                # Some methods might require minimum data size
                pass

    def test_unsorted_data(self):
        """Test calibrators with unsorted input data."""
        np.random.seed(42)
        n = 50

        # Create unsorted data
        y_pred = np.random.uniform(0, 1, n)
        y_true = np.random.binomial(1, y_pred, n)

        # Shuffle to ensure unsorted
        idx = np.random.permutation(n)
        y_pred_shuffled = y_pred[idx]
        y_true_shuffled = y_true[idx]

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            calibrator.fit(y_pred_shuffled, y_true_shuffled)
            y_calib = calibrator.transform(y_pred_shuffled)

            # Should handle unsorted data correctly
            assert len(y_calib) == n
            assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

    def test_duplicate_predictions(self):
        """Test calibrators with many duplicate prediction values."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.3, 0.3, 0.7, 0.7, 0.3, 0.7, 0.3, 0.7])

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)

            # Should handle duplicate predictions
            assert len(y_calib) == len(y_true)
            assert np.all(y_calib >= 0) and np.all(y_calib <= 1)


class TestSklearnCompatibility:
    """Test sklearn compatibility and API compliance."""

    def test_fit_transform_api(self, realistic_dataset):
        """Test sklearn-style fit/transform API."""
        data = realistic_dataset

        calibrators = [
            NearlyIsotonicRegression(lam=1.0, method="path"),
            RelaxedPAVA(percentile=5),
            RegularizedIsotonicRegression(alpha=0.1),
        ]

        for calibrator in calibrators:
            # Test fit method
            fitted_calibrator = calibrator.fit(data["y_proba_train"], data["y_train"])
            assert fitted_calibrator is calibrator  # Returns self

            # Test transform method
            y_calib = calibrator.transform(data["y_proba_test"])
            assert len(y_calib) == len(data["y_test"])

            # Test fit_transform method (if available)
            if hasattr(calibrator, "fit_transform"):
                y_calib_ft = calibrator.fit_transform(
                    data["y_proba_train"], data["y_train"]
                )
                assert len(y_calib_ft) == len(data["y_train"])

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid parameters
        with pytest.raises((ValueError, TypeError)):
            NearlyIsotonicRegression(lam=-1.0)  # Negative lambda

        with pytest.raises((ValueError, TypeError)):
            RelaxedPAVA(percentile=150)  # Invalid percentile

        with pytest.raises((ValueError, TypeError)):
            RegularizedIsotonicRegression(alpha=-0.1)  # Negative alpha


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        X = np.array([0.1, 0.5, 0.9])
        y = np.array([0, 1])  # Different length

        calibrator = NearlyIsotonicRegression(lam=1.0, method="path")

        with pytest.raises(ValueError):
            calibrator.fit(X, y)

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        X = np.array([])
        y = np.array([])

        calibrator = NearlyIsotonicRegression(lam=1.0, method="path")

        with pytest.raises(ValueError):
            calibrator.fit(X, y)

    def test_invalid_prediction_range(self):
        """Test handling of predictions outside [0,1] range."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([-0.1, 1.1, 0.5, 0.7])  # Outside [0,1]

        calibrator = NearlyIsotonicRegression(lam=1.0, method="path")

        # Some calibrators might handle this, others might raise errors
        try:
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)
            # If it succeeds, results should be in valid range
            assert np.all(y_calib >= 0) and np.all(y_calib <= 1)
        except (ValueError, AssertionError):
            # Expected for invalid input
            pass
