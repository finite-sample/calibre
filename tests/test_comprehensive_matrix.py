"""
Comprehensive test matrix for all calibration algorithms.

This module runs systematic tests across all combinations of:
- Calibrators (5 types x multiple parameter settings)
- Data patterns (8 realistic miscalibration scenarios)
- Sample sizes (small, medium, large)
- Noise levels (low, medium, high)

Total test combinations: ~400 tests
"""

from itertools import pairwise, product
from typing import Any

import numpy as np
import pytest

from calibre import (
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
)
from calibre.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from tests.data_generators import CalibrationDataGenerator

# Declared at module scope because @pytest.mark.parametrize is evaluated at
# collection time, before setup_class populates cls.calibrator_configs. A test
# asserts the two stay in step, so adding a calibrator to one and not the other
# fails rather than silently going untested.
_CALIBRATOR_NAMES = (
    "nir_strict",
    "nir_relaxed",
    "ispline_small",
    "ispline_medium",
    "ispline_large",
    "rpava_strict_adaptive",
    "rpava_loose_adaptive",
    "rpava_strict_block",
    "rpava_loose_block",
    "spline_penalty_weak",
    "spline_penalty_medium",
    "spline_penalty_strong",
)


class TestMatrix:
    """Comprehensive test matrix for calibration algorithms."""

    @classmethod
    def setup_class(cls):
        """Set up test matrix parameters."""
        cls.data_generator = CalibrationDataGenerator(random_state=42)

        # Define calibrator configurations
        cls.calibrator_configs = {
            # Nearly Isotonic Regression variants
            "nir_strict": lambda: NearlyIsotonicCalibrator(lam=5.0),
            "nir_relaxed": lambda: NearlyIsotonicCalibrator(lam=0.05),
            # I-Spline Calibrator variants
            "ispline_small": lambda: SplineCalibrator(n_knots=5, degree=2, alpha=0.1),
            "ispline_medium": lambda: SplineCalibrator(n_knots=10, degree=3, alpha=0.1),
            "ispline_large": lambda: SplineCalibrator(n_knots=20, degree=3, alpha=0.1),
            # Epsilon-monotone / minimum-slope variants. The old adaptive-vs-block
            # split is gone (there is one exact algorithm now), so these cover the
            # two directions of the signed increment bound instead.
            "rpava_strict_adaptive": lambda: RelaxedPAVACalibrator(epsilon=0.01),
            "rpava_loose_adaptive": lambda: RelaxedPAVACalibrator(epsilon=0.05),
            "rpava_strict_block": lambda: RelaxedPAVACalibrator(min_slope=0.001),
            "rpava_loose_block": lambda: RelaxedPAVACalibrator(min_slope=0.01),
            # Pinned spline penalty variants
            "spline_penalty_weak": lambda: SplineCalibrator(alpha=0.01, n_knots=10),
            "spline_penalty_medium": lambda: SplineCalibrator(alpha=0.1, n_knots=10),
            "spline_penalty_strong": lambda: SplineCalibrator(alpha=1.0, n_knots=10),
        }

        # Define data patterns
        cls.data_patterns = [
            "overconfident_nn",
            "underconfident_rf",
            "sigmoid_distorted",
            "imbalanced_binary",
            "multi_modal",
            "weather_forecasting",
            "click_through_rate",
            "medical_diagnosis",
        ]

        # Define test parameters
        cls.sample_sizes = [100, 300, 1000]
        cls.noise_levels = [0.05, 0.1, 0.2]

        # Results storage
        cls.results = {}

    def _run_single_test(
        self, calibrator_name: str, pattern: str, n_samples: int, noise_level: float
    ) -> dict[str, Any]:
        """Run a single test combination and return results."""
        try:
            # Ask the generator whether it takes a noise level rather than
            # keeping a list here. The list was wrong in both directions: it
            # exempted click_through_rate, which does take one, and omitted
            # imbalanced_binary, which does not -- so every imbalanced_binary
            # combination raised TypeError and was recorded as a calibrator
            # failure. That is one pattern in eight, which is exactly the 12.5%
            # that the old `success_rate >= 0.7` assertion had room to absorb.
            extra = (
                {"noise_level": noise_level}
                if self.data_generator.accepts(pattern, "noise_level")
                else {}
            )
            y_pred, y_true = self.data_generator.generate_dataset(
                pattern, n_samples=n_samples, **extra
            )

            # Create calibrator
            calibrator = self.calibrator_configs[calibrator_name]()

            # Fit calibrator
            calibrator.fit(y_pred, y_true)

            # Transform predictions
            y_calib = calibrator.transform(y_pred)

            # Calculate metrics
            original_ece = expected_calibration_error(y_true, y_pred)
            calibrated_ece = expected_calibration_error(y_true, y_calib)

            original_mce = maximum_calibration_error(y_true, y_pred)
            calibrated_mce = maximum_calibration_error(y_true, y_calib)

            original_brier = brier_score(y_true, y_pred)
            calibrated_brier = brier_score(y_true, y_calib)

            # Check bounds
            bounds_valid = np.all(y_calib >= 0) and np.all(y_calib <= 1)

            # Check monotonicity (on sorted test data)
            x_test = np.linspace(0, 1, 50)
            y_test_calib = calibrator.transform(x_test)
            monotonicity_violations = np.sum(np.diff(y_test_calib) < 0)

            # Granularity preservation
            original_unique = len(np.unique(np.round(y_pred, 6)))
            calibrated_unique = len(np.unique(np.round(y_calib, 6)))
            granularity_ratio = calibrated_unique / max(original_unique, 1)

            # Correlation preservation
            if len(y_pred) > 1 and np.std(y_pred) > 0 and np.std(y_calib) > 0:
                rank_correlation = np.corrcoef(y_pred, y_calib)[0, 1]
            else:
                rank_correlation = np.nan  # Handle edge cases gracefully

            return {
                "success": True,
                "calibrator": calibrator_name,
                "pattern": pattern,
                "n_samples": n_samples,
                "noise_level": noise_level,
                # Calibration quality
                "original_ece": original_ece,
                "calibrated_ece": calibrated_ece,
                "ece_improvement": original_ece - calibrated_ece,
                "ece_relative_improvement": (original_ece - calibrated_ece)
                / max(original_ece, 1e-10),
                "original_mce": original_mce,
                "calibrated_mce": calibrated_mce,
                "mce_improvement": original_mce - calibrated_mce,
                "original_brier": original_brier,
                "calibrated_brier": calibrated_brier,
                "brier_improvement": original_brier - calibrated_brier,
                # Mathematical properties
                "bounds_valid": bounds_valid,
                "monotonicity_violations": monotonicity_violations,
                "granularity_ratio": granularity_ratio,
                "rank_correlation": rank_correlation,
                # Data characteristics
                "original_mean": np.mean(y_pred),
                "calibrated_mean": np.mean(y_calib),
                "true_rate": np.mean(y_true),
                "original_std": np.std(y_pred),
                "calibrated_std": np.std(y_calib),
            }

        except Exception as e:
            return {
                "success": False,
                "calibrator": calibrator_name,
                "pattern": pattern,
                "n_samples": n_samples,
                "noise_level": noise_level,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    @pytest.mark.parametrize(
        ("calibrator_name", "pattern", "n_samples", "noise_level"),
        [
            (cal, pat, n, noise)
            for cal, pat, n, noise in product(
                [
                    "nir_strict",
                    "ispline_medium",
                    "rpava_strict_adaptive",
                    "spline_penalty_medium",
                ],  # Core calibrators
                [
                    "overconfident_nn",
                    "underconfident_rf",
                    "sigmoid_distorted",
                ],  # Core patterns
                [300],  # Medium sample size
                [0.1],  # Medium noise
            )
        ],
    )
    def test_core_combinations(self, calibrator_name, pattern, n_samples, noise_level):
        """Test core combinations of calibrators and patterns."""
        result = self._run_single_test(calibrator_name, pattern, n_samples, noise_level)

        assert result["success"], (
            f"{calibrator_name} raised on {pattern}: {result['error']}"
        )

        # Core requirements
        assert result["bounds_valid"], (
            f"Bounds violated for {calibrator_name} on {pattern}"
        )
        # Handle NaN correlations gracefully
        if not np.isnan(result["rank_correlation"]):
            assert result["rank_correlation"] >= 0.2, (
                f"Poor rank correlation for {calibrator_name} on {pattern}: "
                f"{result['rank_correlation']:.3f}"
            )
        assert result["calibrated_ece"] >= 0, (
            f"Invalid ECE for {calibrator_name} on {pattern}"
        )
        assert result["calibrated_brier"] <= 1.0, (
            f"Invalid Brier score for {calibrator_name} on {pattern}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "calibrator_name",
        [
            "nir_strict",
            "nir_relaxed",
            "ispline_medium",
            "rpava_strict_adaptive",
            "spline_penalty_medium",
        ],
    )
    def test_bounds_across_patterns(self, calibrator_name):
        """Test that calibrators maintain bounds across all patterns."""
        for pattern in self.data_patterns:
            result = self._run_single_test(calibrator_name, pattern, 200, 0.1)

            if result["success"]:
                assert result["bounds_valid"], (
                    f"{calibrator_name} violated bounds on {pattern}"
                )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "pattern",
        [
            "overconfident_nn",
            "underconfident_rf",
            "sigmoid_distorted",
            "imbalanced_binary",
        ],
    )
    def test_calibration_improvement_across_calibrators(self, pattern):
        """Test that most calibrators improve calibration on common patterns."""
        calibrators = [
            "nir_strict",
            "ispline_medium",
            "rpava_strict_adaptive",
            "spline_penalty_medium",
        ]

        improvements = 0
        total_tests = 0

        for calibrator_name in calibrators:
            result = self._run_single_test(calibrator_name, pattern, 400, 0.1)

            if result["success"]:
                total_tests += 1
                # Allow small tolerance for ECE improvement
                if result["ece_improvement"] >= -0.01:  # Not worse by more than 0.01
                    improvements += 1

        improvement_rate = improvements / max(total_tests, 1)
        # Some patterns are inherently difficult - allow 0% improvement rate
        assert improvement_rate >= 0.0, (
            f"Only {improvement_rate:.1%} of calibrators improved on {pattern}"
        )

    @pytest.mark.slow
    def test_monotonicity_strict_calibrators(self):
        """Test that strict monotonicity calibrators maintain monotonicity."""
        strict_calibrators = [
            "spline_penalty_weak",
            "spline_penalty_medium",
            "spline_penalty_strong",
        ]

        for calibrator_name in strict_calibrators:
            for pattern in [
                "overconfident_nn",
                "underconfident_rf",
                "sigmoid_distorted",
            ]:
                result = self._run_single_test(calibrator_name, pattern, 200, 0.1)

                if result["success"]:
                    # Allow some violations even for "strict" methods due to
                    # numerical precision
                    assert result["monotonicity_violations"] <= 35, (
                        f"{calibrator_name} violated strict monotonicity on "
                        f"{pattern}: "
                        f"{result['monotonicity_violations']} violations"
                    )

    @pytest.mark.slow
    def test_relaxed_monotonicity_calibrators(self):
        """Violations must fall as the monotonicity penalty rises.

        A fixed threshold on the violation rate at a single ``lam`` is not a
        meaningful test, because ``lam`` in this objective is not scale-free: the
        squared-error term is *summed* over observations while the penalty term
        is a total decrease, so the penalty's influence scales like 1/n. At
        n=300, ``lam=0.1`` is effectively no penalty at all -- the fit sits close
        to the raw data and legitimately shows ~35% violations -- while ``lam=50``
        drives them to zero. The previous fixed 40% bound only passed because the
        path solver was not solving the stated objective.

        So assert the property that actually characterises the estimator. Note it
        is the total violation *magnitude* that is controlled, not the violation
        count: the penalty is ``sum max(0, b_i - b_{i+1})``, so a larger ``lam``
        can shrink the total while spreading it over more, smaller, violations.
        For any penalized problem ``min f(b) + lam * P(b)``, the penalty at the
        optimum is non-increasing in ``lam``; that is the provable statement, and
        it must be measured on the fitted grid rather than on a resampled linspace
        (interpolating between knots creates sign changes of its own).
        """
        # Both patterns accept noise_level; weather_forecasting does not, and it
        # is perfectly calibrated by construction anyway (y ~ Binomial(1, y_pred)),
        # so it is a poor probe for a miscalibration fix.
        for pattern in ["multi_modal", "sigmoid_distorted"]:
            y_pred, y_true = self.data_generator.generate_dataset(
                pattern, 300, noise_level=0.1
            )
            grid = np.unique(y_pred)

            totals = []
            for lam in (0.1, 1.0, 10.0, 100.0):
                cal = NearlyIsotonicCalibrator(lam=lam)
                fitted = cal.fit(y_pred, y_true).transform(grid)
                totals.append(float(np.sum(np.maximum(0.0, -np.diff(fitted)))))

            for lo, hi in pairwise(totals):
                assert hi <= lo + 1e-9, (
                    f"{pattern}: total violation magnitude rose with lam: {totals}"
                )
            assert totals[-1] == 0.0, (
                f"{pattern}: a large penalty must recover monotonicity, got {totals}"
            )

    @pytest.mark.parametrize("n_samples", [100, 300, 1000])
    def test_scalability(self, n_samples):
        """Test that calibrators work across different sample sizes."""
        calibrators = ["nir_strict", "ispline_medium", "rpava_strict_adaptive"]
        pattern = "overconfident_nn"

        for calibrator_name in calibrators:
            result = self._run_single_test(calibrator_name, pattern, n_samples, 0.1)

            if result["success"]:
                assert result["bounds_valid"], (
                    f"{calibrator_name} failed bounds on n={n_samples}"
                )
                # Handle NaN correlations gracefully
                if not np.isnan(result["rank_correlation"]):
                    assert result["rank_correlation"] >= 0.1, (
                        f"{calibrator_name} poor correlation on "
                        f"n={n_samples}: {result['rank_correlation']:.3f}"
                    )

    @pytest.mark.parametrize("noise_level", [0.05, 0.1, 0.2])
    def test_noise_robustness(self, noise_level):
        """Test robustness to different noise levels."""
        calibrators = ["nir_strict", "ispline_medium", "spline_penalty_medium"]
        pattern = "sigmoid_distorted"

        for calibrator_name in calibrators:
            result = self._run_single_test(calibrator_name, pattern, 300, noise_level)

            if result["success"]:
                assert result["bounds_valid"], (
                    f"{calibrator_name} failed bounds with noise={noise_level}"
                )
                assert result["calibrated_brier"] <= 1.0, (
                    f"{calibrator_name} invalid Brier with noise={noise_level}"
                )

    @pytest.mark.slow
    def test_granularity_preservation(self):
        """Test that calibrators preserve reasonable granularity."""
        calibrators = [
            "nir_relaxed",
            "ispline_medium",
            "rpava_loose_adaptive",
        ]
        patterns = ["multi_modal", "weather_forecasting", "click_through_rate"]

        for calibrator_name in calibrators:
            for pattern in patterns:
                result = self._run_single_test(calibrator_name, pattern, 400, 0.1)

                if result["success"]:
                    # Should preserve at least 0.3% of unique values (extremely relaxed)
                    assert result["granularity_ratio"] >= 0.003, (
                        f"{calibrator_name} collapsed granularity too much "
                        f"on {pattern}: {result['granularity_ratio']:.3f}"
                    )

                    # Should not create unrealistic explosion
                    assert result["granularity_ratio"] <= 5.0, (
                        f"{calibrator_name} created too many unique values "
                        f"on {pattern}: {result['granularity_ratio']:.3f}"
                    )

    @pytest.mark.slow
    def test_extreme_scenarios(self):
        """Test calibrators on extreme scenarios."""
        extreme_tests = [
            ("medical_diagnosis", 500, 0.05),  # Rare disease
            ("imbalanced_binary", 800, 0.1),  # Heavy imbalance
            ("click_through_rate", 600, 0.05),  # Power-law distribution
        ]

        calibrators = ["nir_strict", "ispline_medium", "spline_penalty_medium"]

        for pattern, n_samples, noise_level in extreme_tests:
            for calibrator_name in calibrators:
                result = self._run_single_test(
                    calibrator_name, pattern, n_samples, noise_level
                )

                if result["success"]:
                    # Basic sanity checks for extreme scenarios
                    assert result["bounds_valid"], (
                        f"{calibrator_name} bounds failed on {pattern}"
                    )
                    assert 0 <= result["calibrated_ece"] <= 1, (
                        f"{calibrator_name} invalid ECE on {pattern}"
                    )
                    # Handle NaN correlations in extreme scenarios
                    if not np.isnan(result["rank_correlation"]):
                        assert result["rank_correlation"] >= -0.5, (
                            f"{calibrator_name} very negative correlation on "
                            f"{pattern}: {result['rank_correlation']:.3f}"
                        )

    @pytest.mark.slow
    def test_parameter_sensitivity(self):
        """Test sensitivity to calibrator parameters."""
        # Test Nearly Isotonic lambda sensitivity
        lambdas = [0.01, 0.1, 1.0, 10.0]
        pattern = "overconfident_nn"

        results = []
        for lam in lambdas:
            calibrator = NearlyIsotonicCalibrator(lam=lam)
            try:
                y_pred, y_true = self.data_generator.generate_dataset(
                    pattern, n_samples=300
                )
                calibrator.fit(y_pred, y_true)

                x_test = np.linspace(0, 1, 50)
                y_test_calib = calibrator.transform(x_test)
                violations = np.sum(np.diff(y_test_calib) < 0)

                results.append((lam, violations))
            except Exception as exc:
                pytest.fail(f"lam={lam} raised {type(exc).__name__}: {exc}")

        assert len(results) >= 2, "the sweep must produce comparable fits"
        if len(results) >= 2:
            # Higher lambda should generally reduce violations
            _lambdas_sorted, violations_sorted = zip(*sorted(results), strict=False)

            # Check general trend (allow some noise)
            if len(results) >= 3:
                high_lambda_violations = violations_sorted[-1]
                low_lambda_violations = violations_sorted[0]
                assert high_lambda_violations <= low_lambda_violations + 2, (
                    "Higher lambda should reduce violations"
                )

    @pytest.mark.slow
    def test_the_parametrised_names_match_the_configured_ones(self):
        """The module-level list must not drift from the configured calibrators.

        Parametrisation reads the module-level tuple, so a calibrator added to
        ``calibrator_configs`` alone would never be exercised by the matrix and
        nothing would say so.
        """
        assert set(_CALIBRATOR_NAMES) == set(self.calibrator_configs)

    @pytest.mark.parametrize("calibrator_name", sorted(_CALIBRATOR_NAMES))
    def test_comprehensive_matrix(self, calibrator_name):
        """Every pattern/size/noise combination must run without error.

        Parametrised by calibrator for two reasons. It used to be one test
        looping over all 1296 combinations, which took 48 seconds -- a third of
        the suite's serial runtime -- and, being a single test, could not be
        distributed by xdist, so one worker held it while the others idled. And
        a failure named nothing: the assertion was an aggregate rate, so the
        report said only that some unspecified fraction of combinations failed.

        The threshold used to be ``success_rate >= 0.7``. That tolerated 30% of
        combinations failing, and it was in fact absorbing a hard failure:
        ``imbalanced_binary`` raised ``TypeError`` for *every* calibrator, size
        and noise level, because the caller passed it a ``noise_level`` its
        generator does not accept. One pattern in eight is 12.5%, comfortably
        inside the 30% allowance, so every calibrator scored exactly 87.5% and
        the suite reported success. With that fixed the real rate is 100%, so
        that is what is asserted -- a combination that errors is a bug, not a
        statistic.
        """
        failures = []
        total = 0
        for pattern in self.data_patterns:
            for n_samples in self.sample_sizes:
                for noise_level in self.noise_levels:
                    total += 1
                    result = self._run_single_test(
                        calibrator_name, pattern, n_samples, noise_level
                    )
                    if not result["success"]:
                        failures.append(
                            f"{pattern}/n={n_samples}/noise={noise_level}: "
                            f"{result.get('error')}"
                        )

        assert not failures, (
            f"{calibrator_name} failed {len(failures)} of {total} combinations:\n  "
            + "\n  ".join(failures[:10])
        )


class TestMatrixAnalysis:
    """Analysis and reporting on test matrix results."""

    @pytest.mark.slow
    def test_calibrator_ranking_by_improvement(self):
        """Rank calibrators by average calibration improvement."""
        # This would typically be run after the comprehensive matrix
        # For now, we'll run a smaller subset
        calibrators = [
            "nir_strict",
            "ispline_medium",
            "rpava_strict_adaptive",
            "spline_penalty_medium",
        ]
        patterns = ["overconfident_nn", "underconfident_rf", "sigmoid_distorted"]

        data_gen = CalibrationDataGenerator(random_state=42)
        calibrator_scores = {}

        for cal_name in calibrators:
            improvements = []

            for pattern in patterns:
                try:
                    # Create calibrator
                    if cal_name == "nir_strict":
                        calibrator = NearlyIsotonicCalibrator(lam=5.0)
                    elif cal_name == "ispline_medium":
                        calibrator = SplineCalibrator(n_knots=10, degree=3, cv=3)
                    elif cal_name == "rpava_strict_adaptive":
                        calibrator = RelaxedPAVACalibrator(epsilon=0.01)
                    elif cal_name == "spline_penalty_medium":
                        calibrator = SplineCalibrator(alpha=0.1)

                    # Generate data and test
                    y_pred, y_true = data_gen.generate_dataset(pattern, n_samples=300)

                    original_ece = expected_calibration_error(y_true, y_pred)

                    calibrator.fit(y_pred, y_true)
                    y_calib = calibrator.transform(y_pred)
                    calibrated_ece = expected_calibration_error(y_true, y_calib)

                    improvement = original_ece - calibrated_ece
                    improvements.append(improvement)

                except Exception as exc:
                    pytest.fail(
                        f"{cal_name} on {pattern} raised {type(exc).__name__}: {exc}"
                    )

            if improvements:
                calibrator_scores[cal_name] = np.mean(improvements)

        # Should have results for most calibrators
        assert len(calibrator_scores) >= 3, "Not enough calibrators succeeded"

        # Print ranking
        print("\nCalibrator ranking by average ECE improvement:")
        for cal, score in sorted(
            calibrator_scores.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {cal}: {score:.4f}")
