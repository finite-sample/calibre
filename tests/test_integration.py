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
                "kwargs": {"n_knots": 10, "degree": 3, "cv": 3},
                "name": "spline",
            },
            {
                "class": RelaxedPAVACalibrator,
                "kwargs": {"epsilon": 0.01},
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
        assert np.all(y_calib >= 0)
        assert np.all(y_calib <= 1)

        # Calibrator-specific validation
        name = calibrator_config["name"]

        if name == "nearly_isotonic":
            # Test calibration metrics
            mce_after = mean_calibration_error(data["y_test"], y_calib)
            ece_after = expected_calibration_error(data["y_test"], y_calib)
            assert isinstance(mce_after, float)
            assert isinstance(ece_after, float)

        elif name == "spline":
            # Check correlation preservation
            corr_metrics = correlation_metrics(
                data["y_test"], y_calib, y_orig=data["y_proba_test"]
            )
            assert corr_metrics["spearman_corr_orig_to_calib"] > 0.5

        elif name == "regularized":
            # Monotonicity is a hard constraint here, so nothing may go backwards.
            sorted_idx = np.argsort(data["y_proba_test"])
            y_calib_sorted = y_calib[sorted_idx]
            assert np.all(np.diff(y_calib_sorted) >= -1e-9), (
                "regularized isotonic constrains monotonicity, so a violation is "
                "a bug, not a tolerance to be widened"
            )

        elif name == "relaxed_pava":
            # This estimator deliberately permits decreases, so counting them is
            # not the test. What it guarantees is their *size*: no single decrease
            # between adjacent unique scores may exceed epsilon. A rate-based
            # bound only ever passed here because the old percentile threshold
            # collapsed to zero on binary labels, making the estimator silently
            # equal to plain PAVA and hence trivially monotone.
            # The bound is per adjacent pair of *fitted knots*, which is where the
            # increment constraint lives. Measuring between arbitrary test scores
            # would be wrong: two neighbouring test scores can span many knot
            # intervals, and the permitted decreases accumulate across them.
            epsilon = calibrator.get_params()["epsilon"]
            knots = calibrator.calibration_curve_.y
            worst_drop = float(np.max(np.maximum(0.0, -np.diff(knots))))
            assert worst_drop <= epsilon + 1e-9, (
                f"largest per-knot decrease {worst_drop:.6f} exceeds epsilon={epsilon}"
            )


class TestCalibratorComparison:
    """Test comparing different calibrators on the same data."""

    def test_calibrator_performance_comparison(self, realistic_dataset):
        """Compare performance of different calibrators."""
        data = realistic_dataset

        calibrators = {
            "nearly_isotonic": NearlyIsotonicCalibrator(lam=1.0, method="path"),
            "spline": SplineCalibrator(n_knots=10, degree=3, cv=3),
            "relaxed_pava": RelaxedPAVACalibrator(epsilon=0.01),
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
                assert isinstance(value, float), (
                    f"{name} {metric_name} is not a float: {value!r}"
                )
                assert value >= 0, f"{name} {metric_name} is negative: {value}"

            assert len(y_calib) == len(data["y_test"])


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of calibrators."""

    @pytest.fixture
    def core_calibrators(self):
        """Common set of calibrators for edge case testing."""
        return [
            NearlyIsotonicCalibrator(lam=1.0, method="path"),
            RelaxedPAVACalibrator(epsilon=0.01),
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
                assert np.all(y_calib >= 0)
                assert np.all(y_calib <= 1)

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
        """Invalid parameters are accepted at init but must be rejected by fit.

        Deferring validation to ``fit`` is the scikit-learn convention: ``__init__``
        only records parameters so that ``get_params``/``clone`` round-trip. But
        deferring is not the same as skipping -- previously nothing anywhere
        checked that ``fit`` rejects a negative penalty, so an invalid setting
        would silently produce a fit.
        """
        # Each case pins the message too, so the test checks that the *right*
        # error is raised rather than merely that something went wrong.
        test_cases = [
            (NearlyIsotonicCalibrator, {"lam": -1.0}, r"lam must be non-negative"),
            (RelaxedPAVACalibrator, {"epsilon": -0.5}, r"epsilon must be non-negative"),
            (
                RelaxedPAVACalibrator,
                {"min_slope": -0.5},
                r"min_slope must be non-negative",
            ),
            (
                RelaxedPAVACalibrator,
                {"epsilon": 0.1, "min_slope": 0.1},
                r"opposite directions",
            ),
            (
                RegularizedIsotonicCalibrator,
                {"alpha": -0.1},
                r"alpha must be non-negative",
            ),
        ]

        x = np.linspace(0.05, 0.95, 40)
        y = (np.arange(40) % 3 == 0).astype(float)

        for calibrator_class, invalid_params, expected in test_cases:
            calibrator = calibrator_class(**invalid_params)
            assert calibrator is not None, "init must not raise"

            # get_params must echo back exactly what was passed, unmodified.
            params = calibrator.get_params()
            for key, value in invalid_params.items():
                assert params[key] == value, (
                    f"{calibrator_class.__name__} mutated {key}: "
                    f"{params[key]!r} != {value!r}"
                )

            with pytest.raises(ValueError, match=expected):
                calibrator.fit(x, y)


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
        with pytest.raises(ValueError, match=r"(?i)must|length|shape|empty"):
            calibrator.fit(np.array([0.1, 0.5, 0.9]), np.array([0, 1]))

        # Test empty arrays
        with pytest.raises(ValueError, match=r"(?i)must|length|shape|empty"):
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
            assert np.all(y_calib >= 0)
            assert np.all(y_calib <= 1)
        except (ValueError, AssertionError):
            # Expected for invalid input
            pass
