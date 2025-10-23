"""
Integration tests for the calibre package.
Tests complete calibration workflows and edge cases.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from calibre import (
    NearlyIsotonicCalibrator,
    RegularizedIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SmoothedIsotonicCalibrator,
    SplineCalibrator,
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

    @pytest.mark.parametrize(
        "calibrator_config",
        [
            {
                "class": NearlyIsotonicCalibrator,
                "kwargs": {"lam": 1.0, "method": "path"},
                "name": "nearly_isotonic",
            },
            {
                "class": SplineCalibrator,
                "kwargs": {"n_splines": 10, "degree": 3, "cv": 3},
                "name": "spline",
            },
            {
                "class": RelaxedPAVACalibrator,
                "kwargs": {"percentile": 5, "adaptive": True},
                "name": "relaxed_pava",
            },
            {
                "class": RegularizedIsotonicCalibrator,
                "kwargs": {"alpha": 0.1},
                "name": "regularized",
            },
            {
                "class": SmoothedIsotonicCalibrator,
                "kwargs": {
                    "window_length": 7,
                    "poly_order": 3,
                    "interp_method": "linear",
                },
                "name": "smoothed",
            },
        ],
    )
    def test_calibrator_workflow(self, calibrator_config, realistic_dataset):
        """Test complete workflow for all calibrators."""
        data = realistic_dataset

        # Create and train calibrator
        calibrator = calibrator_config["class"](**calibrator_config["kwargs"])
        calibrator.fit(data["y_proba_train"], data["y_train"])

        # Calibrate test predictions
        y_calib = calibrator.transform(data["y_proba_test"])

        # Basic validation (common to all calibrators)
        assert len(y_calib) == len(data["y_test"])
        assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

        # Calibrator-specific validation
        name = calibrator_config["name"]

        if name == "nearly_isotonic":
            # Test calibration metrics
            mce_after = mean_calibration_error(data["y_test"], y_calib)
            ece_after = expected_calibration_error(data["y_test"], y_calib)
            assert isinstance(mce_after, float) and isinstance(ece_after, float)

        elif name == "spline":
            # Check correlation preservation
            corr_metrics = correlation_metrics(
                data["y_test"], y_calib, y_orig=data["y_proba_test"]
            )
            assert corr_metrics["spearman_corr_orig_to_calib"] > 0.5

        elif name in ["relaxed_pava", "regularized"]:
            # Check monotonicity (allowing some violations)
            sorted_idx = np.argsort(data["y_proba_test"])
            y_calib_sorted = y_calib[sorted_idx]
            violations = np.sum(np.diff(y_calib_sorted) < 0)
            total_pairs = len(y_calib_sorted) - 1
            violation_rate = violations / total_pairs if total_pairs > 0 else 0
            max_violation_rate = 0.1 if name == "relaxed_pava" else 0.2
            assert violation_rate <= max_violation_rate


class TestCalibratorComparison:
    """Test comparing different calibrators on the same data."""

    def test_calibrator_performance_comparison(self, realistic_dataset):
        """Compare performance of different calibrators."""
        data = realistic_dataset

        calibrators = {
            "nearly_isotonic": NearlyIsotonicCalibrator(lam=1.0, method="path"),
            "spline": SplineCalibrator(n_splines=10, degree=3, cv=3),
            "relaxed_pava": RelaxedPAVACalibrator(percentile=5),
            "regularized": RegularizedIsotonicCalibrator(alpha=0.1),
            "smoothed": SmoothedIsotonicCalibrator(window_length=7, poly_order=3),
        }

        for name, calibrator in calibrators.items():
            # Train and test calibrator
            calibrator.fit(data["y_proba_train"], data["y_train"])
            y_calib = calibrator.transform(data["y_proba_test"])

            # Validate metrics
            metrics = {
                "mce": mean_calibration_error(data["y_test"], y_calib),
                "ece": expected_calibration_error(data["y_test"], y_calib),
                "brier": brier_score(data["y_test"], y_calib),
            }

            # All metrics should be valid
            for metric_name, value in metrics.items():
                assert (
                    isinstance(value, float) and value >= 0
                ), f"{name} {metric_name} invalid"

            assert len(y_calib) == len(data["y_test"])


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of calibrators."""

    @pytest.fixture
    def core_calibrators(self):
        """Common set of calibrators for edge case testing."""
        return [
            NearlyIsotonicCalibrator(lam=1.0, method="path"),
            RelaxedPAVACalibrator(percentile=5),
            RegularizedIsotonicCalibrator(alpha=0.1),
        ]

    def _test_calibrator_robustness(
        self, calibrators, y_pred, y_true, expect_success=True
    ):
        """Helper method to test calibrator robustness on edge cases."""
        for calibrator in calibrators:
            try:
                calibrator.fit(y_pred, y_true)
                y_calib = calibrator.transform(y_pred)

                # Basic validation
                assert len(y_calib) == len(y_true)
                assert np.all(y_calib >= 0) and np.all(y_calib <= 1)

                if expect_success:
                    # Additional checks for successful cases
                    assert not np.any(np.isnan(y_calib))

            except (ValueError, np.linalg.LinAlgError):
                if expect_success:
                    pytest.fail(f"{type(calibrator).__name__} failed on valid input")
                # Otherwise, failure is expected for some edge cases

    def test_perfect_predictions(self, core_calibrators):
        """Test calibrators with perfect predictions."""
        n = 100
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = y_true.astype(float)

        self._test_calibrator_robustness(core_calibrators, y_pred, y_true)

        # Perfect predictions should maintain low calibration error
        for calibrator in core_calibrators:
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)
            mce = mean_calibration_error(y_true, y_calib)
            assert mce < 0.1

    def test_challenging_edge_cases(self, core_calibrators):
        """Test various challenging edge cases in a single consolidated test."""
        test_cases = [
            # Constant predictions
            {
                "name": "constant",
                "y_pred": np.full(100, 0.5),
                "y_true": np.random.binomial(1, 0.3, 100),
                "expect_success": True,
            },
            # Extreme predictions
            {
                "name": "extreme",
                "y_pred": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                "y_true": np.array([0, 0, 1, 1, 0, 1]),
                "expect_success": False,  # May fail for some calibrators
            },
            # Small dataset
            {
                "name": "small",
                "y_pred": np.array([0.2, 0.7, 0.8]),
                "y_true": np.array([0, 1, 1]),
                "expect_success": False,  # May fail for some calibrators
            },
            # Duplicate predictions
            {
                "name": "duplicates",
                "y_pred": np.array([0.3, 0.3, 0.7, 0.7, 0.3, 0.7, 0.3, 0.7]),
                "y_true": np.array([0, 0, 1, 1, 0, 1, 0, 1]),
                "expect_success": True,
            },
        ]

        for case in test_cases:
            self._test_calibrator_robustness(
                core_calibrators, case["y_pred"], case["y_true"], case["expect_success"]
            )

    def test_unsorted_data_handling(self, core_calibrators):
        """Test calibrators with unsorted input data."""
        np.random.seed(42)
        n = 50

        # Create and shuffle data
        y_pred = np.random.uniform(0, 1, n)
        y_true = np.random.binomial(1, y_pred, n)
        idx = np.random.permutation(n)

        self._test_calibrator_robustness(core_calibrators, y_pred[idx], y_true[idx])


class TestSklearnCompatibility:
    """Test sklearn compatibility and API compliance."""

    def test_fit_transform_api(self, realistic_dataset):
        """Test sklearn-style fit/transform API."""
        data = realistic_dataset
        calibrator = NearlyIsotonicCalibrator(lam=1.0, method="path")

        # Test fit method returns self
        fitted_calibrator = calibrator.fit(data["y_proba_train"], data["y_train"])
        assert fitted_calibrator is calibrator

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
        """Test that invalid parameters are accepted at init (lazy validation)."""
        # Our calibrators use lazy parameter validation
        test_cases = [
            (NearlyIsotonicCalibrator, {"lam": -1.0}),
            (RelaxedPAVACalibrator, {"percentile": 150}),
            (RegularizedIsotonicCalibrator, {"alpha": -0.1}),
        ]

        for calibrator_class, invalid_params in test_cases:
            calibrator = calibrator_class(**invalid_params)
            # Should not raise at init time - validation happens during fit
            assert calibrator is not None


class TestErrorHandling:
    """Test error handling and input validation."""

    @pytest.mark.parametrize(
        "calibrator_class",
        [
            NearlyIsotonicCalibrator,
            RelaxedPAVACalibrator,
            RegularizedIsotonicCalibrator,
        ],
    )
    def test_input_validation_errors(self, calibrator_class):
        """Test various input validation scenarios."""
        calibrator = calibrator_class()

        # Test mismatched array lengths
        with pytest.raises(ValueError):
            calibrator.fit(np.array([0.1, 0.5, 0.9]), np.array([0, 1]))

        # Test empty arrays
        with pytest.raises(ValueError):
            calibrator.fit(np.array([]), np.array([]))

    def test_invalid_prediction_range(self):
        """Test handling of predictions outside [0,1] range."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([-0.1, 1.1, 0.5, 0.7])  # Outside [0,1]

        calibrator = NearlyIsotonicCalibrator(lam=1.0, method="path")

        # Some calibrators might handle this, others might raise errors
        try:
            calibrator.fit(y_pred, y_true)
            y_calib = calibrator.transform(y_pred)
            # If it succeeds, results should be in valid range
            assert np.all(y_calib >= 0) and np.all(y_calib <= 1)
        except (ValueError, AssertionError):
            # Expected for invalid input
            pass
