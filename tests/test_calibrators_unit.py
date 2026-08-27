"""
Unit tests for individual calibrator classes.

This module provides focused unit tests for each calibrator implementation,
testing basic functionality, parameter validation, and core behavior.
"""

import itertools

import numpy as np
import pytest

from calibre import (
    CDIIsotonicCalibrator,
    IsotonicCalibrator,
    NearlyIsotonicCalibrator,
    RelaxedPAVACalibrator,
    SplineCalibrator,
)
from calibre.metrics import brier_score, root_mean_squared_calibration_error


@pytest.fixture
def calibration_data():
    """Generate synthetic test data for calibration with realistic bias."""
    np.random.seed(42)
    n = 100
    # Sorted x values as the underlying true signal
    x = np.sort(np.random.uniform(0, 1, n))
    # True underlying probabilities (monotonic)
    y_true = x.copy()
    # Introduce non-linear bias: quadratic term that increases for higher x
    bias = 0.5 * x**2
    # Add Gaussian noise (small relative to the bias)
    noise = np.random.normal(0, 0.05, size=n)
    # Observed predictions are biased: true + bias + noise
    y_observed = y_true + bias + noise
    return x, y_observed, y_true


@pytest.fixture
def miscalibrated_data():
    """A calibration problem a calibrator can be judged on.

    ``calibration_data`` above cannot serve for accuracy claims. It sets
    ``y_true = x`` exactly, and every calibrator here is monotone in ``x`` by
    construction, so any comparison of ``transform(x)`` against ``y_true``
    reduces to "a monotone function of x correlates with x" -- which is true of
    the input itself, and of anything that does nothing. Its targets also reach
    ~1.52, so they are not probabilities and two calibrators reject them
    outright under their defaults.

    This one is the real thing: probabilities in (0, 1), outcomes drawn from
    them, and ``score`` a monotone distortion of the truth. Being monotone, the
    distortion leaves the ranking untouched and every rank-based statistic
    blind to it -- which is exactly the failure a calibrator exists to repair,
    and exactly what ``corr`` could not see.

    Returns:
        tuple: ``(score, y, p_true)`` -- the miscalibrated score, the binary
        outcomes, and the probabilities that generated them.
    """
    rng = np.random.default_rng(0)
    n = 2000
    p_true = rng.uniform(0.02, 0.98, n)
    score = p_true**1.8
    y = rng.binomial(1, p_true).astype(float)
    return score, y, p_true


class DoesNothing:
    """A calibrator-shaped object that returns its input untouched.

    The control. Every accuracy assertion in this file is checked against it,
    because an assertion no implementation can fail is not evidence. The six
    ``corr(y_true, y_calib) > 0.5`` checks this replaced all passed with this
    object scoring 0.9837.
    """

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


class PredictsTheBaseRate:
    """Returns the training prevalence for every input.

    The second control, and the one that caught a hole in the first version of
    this file's replacement assertions. It is *perfectly calibrated* -- every
    prediction equals the observed frequency -- and scores 0.0220 held-out
    calibration error against the uncalibrated 0.1483. A reliability-only bar
    waves it through, while it has discarded every scrap of ranking.

    Brier is what stops it: 0.2502 against the uncalibrated 0.1980.
    """

    def __init__(self):
        self.rate = 0.5

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def transform(self, X):
        return np.full(len(np.asarray(X)), self.rate)


def _calibration_error(predictions, y):
    """Root mean squared calibration error against outcomes ``y``.

    Args:
        predictions: Predicted probabilities.
        y: Binary outcomes.

    Returns:
        float: Mass-weighted RMS calibration error over ten equal-width bins.
    """
    return root_mean_squared_calibration_error(
        y, np.clip(predictions, 0.0, 1.0), n_bins=10
    )


def assert_calibrates(calibrator, miscalibrated_data, factor=2.0):
    """Assert a calibrator calibrates, on data it was not fitted to.

    **Two legs, and both are needed.** Calibration error alone is not a
    sufficient oracle: a predictor returning ``mean(y)`` for every input is
    perfectly reliable and scores 0.0220 against the uncalibrated 0.1483, so it
    would sail past a reliability-only bar while having discarded every scrap
    of ranking information. Brier is a proper score and charges it for that --
    0.2502 against the uncalibrated 0.1980 -- so the second leg is what makes
    the first mean something.

    That is the reliability-resolution split: the first leg asks whether the
    predictions mean what they say, the second whether they still say anything.

    **Fitted and evaluated on disjoint halves**, because a flexible calibrator
    can drive in-sample error to zero by interpolating the sampled labels. This
    is not hypothetical here -- see
    ``TestNearlyIsotonicOverfitsAtLowLambda``, where the in-sample error is
    *anti-correlated* with the held-out error across the useful range of
    ``lam``.

    Args:
        calibrator: An unfitted calibrator exposing fit/transform.
        miscalibrated_data: The ``(score, y, p_true)`` fixture tuple.
        factor: How many times smaller the held-out calibration error must be.

    Returns:
        float: The achieved held-out calibration error.
    """
    score, y, _ = miscalibrated_data
    half = len(score) // 2
    fit, hold = slice(0, half), slice(half, None)

    before = _calibration_error(score[hold], y[hold])
    before_brier = brier_score(y[hold], np.clip(score[hold], 0.0, 1.0))

    predictions = calibrator.fit(score[fit], y[fit]).transform(score[hold])
    after = _calibration_error(predictions, y[hold])
    after_brier = brier_score(y[hold], np.clip(predictions, 0.0, 1.0))

    assert after * factor < before, (
        f"held-out calibration error {after:.4f} is not {factor}x below the "
        f"uncalibrated {before:.4f}; a calibrator that did nothing would "
        f"score {before:.4f}"
    )
    assert after_brier <= before_brier, (
        f"held-out Brier {after_brier:.4f} is worse than the uncalibrated "
        f"{before_brier:.4f}; reliability was bought by throwing away "
        f"resolution, which a constant prediction does perfectly"
    )
    return after


@pytest.fixture
def binary_data():
    """Generate binary classification data for testing."""
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 1, n)
    y = (x + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    return x, y


class TestTheAccuracyBarCanBeFailed:
    """The control for every ``assert_calibrates`` call in this file.

    Six tests here used to assert ``corr(y_true, y_calib) > 0.5`` and nothing
    else about accuracy. No implementation could fail it: correlation is
    invariant to increasing affine maps, and miscalibration *is* an increasing
    map, so a calibrator returning half the true probability scores exactly
    1.0. Measured on the old fixture, ``DoesNothing`` scored 0.9837 and the raw
    uncalibrated observations 0.7209 -- both comfortably past the bar.

    An assertion that nothing can fail is not evidence, so the replacement
    carries a control that does fail.
    """

    def test_a_calibrator_that_does_nothing_fails_the_bar(self):
        """Without this, the six replacements would be as vacuous as the six
        assertions they replaced.
        """
        rng = np.random.default_rng(0)
        n = 2000
        p_true = rng.uniform(0.02, 0.98, n)
        data = (p_true**1.8, rng.binomial(1, p_true).astype(float), p_true)

        with pytest.raises(AssertionError, match=r"not 2\.0x below"):
            assert_calibrates(DoesNothing(), data)

    def test_a_constant_prediction_fails_the_bar(self, miscalibrated_data):
        """Perfect reliability, zero resolution -- must not pass.

        The first version of these replacements asserted only that calibration
        error fell, and this object passed it: predicting the base rate is
        exactly calibrated. Reliability without resolution is not calibration
        in any useful sense, which is why the assertion has a Brier leg.
        """
        with pytest.raises(AssertionError, match="resolution"):
            assert_calibrates(PredictsTheBaseRate(), miscalibrated_data)

    def test_a_real_calibrator_clears_it(self, miscalibrated_data):
        """The other half: the bar must not be so high that nothing passes."""
        assert assert_calibrates(IsotonicCalibrator(), miscalibrated_data) >= 0.0

    def test_correlation_cannot_see_miscalibration(self):
        """Why the statistic was changed, not just the threshold.

        Rank-preserving distortion is invisible to correlation by construction.
        Anyone tempted to reinstate a correlation check should read this first.
        """
        truth = np.linspace(0.05, 0.95, 200)
        for distorted in (0.5 * truth, 0.4 + 0.2 * truth):
            assert np.corrcoef(truth, distorted)[0, 1] == pytest.approx(1.0)
            assert np.mean(np.abs(distorted - truth)) > 0.1


class TestIsotonicCalibrator:
    """Test IsotonicCalibrator functionality."""

    def test_basic_fitting(self, binary_data):
        """Test basic fit and transform operations."""
        x, y = binary_data
        cal = IsotonicCalibrator()
        cal.fit(x, y)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))

    def test_with_diagnostics(self, binary_data):
        """Test calibrator with diagnostics enabled."""
        x, y = binary_data
        cal = IsotonicCalibrator(enable_diagnostics=True)
        cal.fit(x, y)

        assert cal.has_diagnostics()
        diagnostics = cal.get_diagnostics()
        assert isinstance(diagnostics, dict)
        assert "n_plateaus" in diagnostics

    def test_parameter_bounds(self, binary_data):
        """Test with y_min and y_max parameters."""
        x, y = binary_data
        cal = IsotonicCalibrator(y_min=0.1, y_max=0.9)
        cal.fit(x, y)
        y_calib = cal.transform(x)

        assert np.all(y_calib >= 0.1)
        assert np.all(y_calib <= 0.9)


class TestNearlyIsotonicCalibrator:
    """Test NearlyIsotonicCalibrator functionality."""

    def test_basic_fit(self, calibration_data, miscalibrated_data):
        """Fit the exact path implementation and exercise held-out behavior."""
        x, y_observed, _y_true = calibration_data
        cal = NearlyIsotonicCalibrator(lam=5.0)
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert_calibrates(NearlyIsotonicCalibrator(lam=5.0), miscalibrated_data)

    def test_low_lambda_fit(self, calibration_data):
        """A low penalty still returns a complete finite fit."""
        x, y_observed, _y_true = calibration_data
        cal = NearlyIsotonicCalibrator(lam=0.05)
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all(np.isfinite(y_calib))


class TestNearlyIsotonicOverfitsAtLowLambda:
    """Small ``lam`` fits the labels, not the probabilities.

    ``lam`` penalises monotonicity violations, so a small one leaves the fit
    nearly unconstrained and it interpolates the sampled outcomes. Measured on
    ``miscalibrated_data``, fitted on one half and evaluated on the other,
    against an uncalibrated baseline of 0.1483 calibration error and 0.1980
    Brier:

    ====== ================ =============== ==============
    lam    in-sample calib  held-out calib  held-out Brier
    ====== ================ =============== ==============
    0.1              0.0177          0.2250         0.2862
    1.0              0.0000          0.1854         0.2493
    5.0              0.0313          0.0354         0.1812
    10.0             0.0123          0.0362         0.1769
    50.0             0.0000          0.0393         0.1769
    ====== ================ =============== ==============

    At lam <= 1 the calibrator is **worse than not calibrating at all**, on
    both measures, while its in-sample error is at or near zero. In-sample
    error is not merely a weak signal here; over this range it points the wrong
    way.

    This is why ``assert_calibrates`` holds data back. The file's previous
    accuracy tests fitted and scored on the same array and had been asserting
    ``lam=0.1`` was fine.
    """

    def test_a_low_lambda_does_not_calibrate_out_of_sample(self, miscalibrated_data):
        with pytest.raises(AssertionError):
            assert_calibrates(NearlyIsotonicCalibrator(lam=0.05), miscalibrated_data)

    def test_a_useful_lambda_does(self, miscalibrated_data):
        """The other half -- the solver is fine, the setting was not."""
        assert_calibrates(NearlyIsotonicCalibrator(lam=5.0), miscalibrated_data)

    def test_in_sample_error_hides_it(self, miscalibrated_data):
        """The mechanism, asserted rather than described.

        Fitted and scored on the same data, the overfitting configuration looks
        *better* than the sound one. Anyone tempted to drop the holdout should
        run this.
        """
        score, y, _ = miscalibrated_data
        overfit = NearlyIsotonicCalibrator(lam=0.05).fit(score, y)
        sound = NearlyIsotonicCalibrator(lam=5.0).fit(score, y)

        assert _calibration_error(overfit.transform(score), y) < _calibration_error(
            sound.transform(score), y
        )


class TestPinnedSplinePenalty:
    """Test SplineCalibrator functionality."""

    def test_basic_functionality(self, calibration_data, miscalibrated_data):
        """Test SplineCalibrator basic operations."""
        x, y_observed, _y_true = calibration_data
        # The `calibration_data` fixture's targets run outside [0, 1] (up to
        # ~1.52), so they are not probabilities and the Bernoulli likelihood
        # does not apply -- fit on the identity scale instead.
        cal = SplineCalibrator(n_knots=10, degree=3, cv=5, link="identity")
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        assert_calibrates(SplineCalibrator(), miscalibrated_data)

    def test_parameter_variations(self, calibration_data):
        """Test different parameter combinations."""
        x, y_observed, _y_true = calibration_data

        # Fitted on the identity scale: see test_basic_functionality.
        configs = [
            {"n_knots": 5, "degree": 2},
            {"n_knots": 15, "degree": 3},
            {"n_knots": 8, "degree": 1},
        ]

        for config in configs:
            cal = SplineCalibrator(**config, link="identity")
            cal.fit(x, y_observed)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert np.all((y_calib >= 0) & (y_calib <= 1))


class TestRelaxedPAVACalibrator:
    """Test RelaxedPAVACalibrator functionality."""

    def test_basic_functionality(self, calibration_data, miscalibrated_data):
        """Test RelaxedPAVACalibrator basic operations."""
        x, y_observed, _y_true = calibration_data
        cal = RelaxedPAVACalibrator(min_increment=-0.02)
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        assert_calibrates(
            RelaxedPAVACalibrator(min_increment=-0.02),
            miscalibrated_data,
            factor=2.0,
        )

    def test_negative_bound_relaxes_monotonicity_monotonically(self, calibration_data):
        """A more negative bound must permit at least as much total decrease.

        ``min_increment`` bounds changes between adjacent unique scores. Zero
        reproduces isotonic regression; moving the bound downward enlarges the
        feasible set.
        """
        x, y_observed, _ = calibration_data
        grid = np.unique(x)

        totals = []
        for bound in [0.0, -0.01, -0.05, -0.2]:
            cal = RelaxedPAVACalibrator(min_increment=bound, clip_output=False)
            fitted = cal.fit(x, y_observed).transform(grid)
            assert len(cal.transform(x)) == len(x)
            assert np.all(np.diff(fitted) >= bound - 1e-12)
            totals.append(float(np.sum(np.maximum(0.0, -np.diff(fitted)))))

        assert totals[0] == 0.0, "a zero bound must be exactly monotone"
        for lo, hi in itertools.pairwise(totals):
            assert hi >= lo - 1e-12, f"total decrease fell as bound relaxed: {totals}"

    def test_zero_bound_equals_isotonic(self, calibration_data):
        """A zero bound is standard isotonic regression and must match sklearn."""
        from sklearn.isotonic import IsotonicRegression

        x, y_observed, _ = calibration_data
        grid = np.unique(x)

        # clip_output=False so the comparison is against the same estimator:
        # sklearn does not clip, and this fixture's targets dip below 0.
        got = (
            RelaxedPAVACalibrator(min_increment=0.0, clip_output=False)
            .fit(x, y_observed)
            .transform(grid)
        )
        expected = (
            IsotonicRegression(out_of_bounds="clip").fit(x, y_observed).transform(grid)
        )
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)

    def test_positive_bound_removes_plateaus(self, calibration_data):
        """A positive increment bound must leave no plateau on the fitted grid."""
        x, y_observed, _ = calibration_data
        grid = np.unique(x)

        plain = (
            RelaxedPAVACalibrator(min_increment=0.0).fit(x, y_observed).transform(grid)
        )
        separated = (
            RelaxedPAVACalibrator(min_increment=1e-4, clip_output=False)
            .fit(x, y_observed)
            .transform(grid)
        )

        assert np.any(np.diff(plain) == 0), "fixture should produce plateaus"
        assert np.all(np.diff(separated) >= 1e-4 - 1e-12)

    def test_clipping_can_flatten_a_positive_bound(self):
        """Probability clipping must not be advertised as strictly increasing."""
        x = np.linspace(0.0, 1.0, 20)
        y = np.r_[np.zeros(10), np.ones(10)]

        fitted = RelaxedPAVACalibrator(min_increment=0.1).fit(x, y)
        knots = fitted.calibration_curve_.y

        assert np.all((knots >= 0.0) & (knots <= 1.0))
        assert np.any(np.diff(knots) == 0.0)

    def test_bound_is_required_and_must_be_finite(self):
        """The estimator must not invent an application-specific constraint."""
        with pytest.raises(TypeError, match="min_increment"):
            RelaxedPAVACalibrator()

        x = np.linspace(0.0, 1.0, 10)
        y = (x > 0.5).astype(float)
        for invalid in (np.nan, np.inf, -np.inf, "auto"):
            with pytest.raises(ValueError, match="min_increment must be finite"):
                RelaxedPAVACalibrator(min_increment=invalid).fit(x, y)


class TestSplineCalibrator:
    """Test SplineCalibrator functionality."""

    def test_basic_functionality(self, calibration_data, miscalibrated_data):
        """Test SplineCalibrator basic operations."""
        x, y_observed, _y_true = calibration_data
        # The `calibration_data` fixture's targets run outside [0, 1] (up to
        # ~1.52), so they are not probabilities and the Bernoulli likelihood
        # does not apply -- fit on the identity scale instead.
        cal = SplineCalibrator(alpha=0.1, link="identity")
        cal.fit(x, y_observed)
        y_calib = cal.transform(x)

        assert len(y_calib) == len(x)
        assert np.all((y_calib >= 0) & (y_calib <= 1))
        assert_calibrates(SplineCalibrator(alpha=0.1), miscalibrated_data)

    def test_regularization_strength(self, calibration_data):
        """Test different regularization strengths."""
        x, y_observed, _y_true = calibration_data

        for alpha in [0.01, 0.1, 1.0]:
            cal = SplineCalibrator(alpha=alpha, link="identity")
            cal.fit(x, y_observed)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert np.all((y_calib >= 0) & (y_calib <= 1))


class TestCalibratorErrorHandling:
    """Test error handling across all calibrators."""

    def test_mismatched_array_lengths(self):
        """Test error handling for mismatched array lengths."""
        np.array([1, 2, 3, 4, 5])
        y_good = np.array([1, 2, 3, 4, 5])
        x_bad = np.array([1, 2, 3])  # mismatched length

        calibrators = [
            NearlyIsotonicCalibrator(lam=0.5),
            SplineCalibrator(n_knots=5),
            RelaxedPAVACalibrator(min_increment=-0.02),
            SplineCalibrator(alpha=0.1),
        ]

        for cal in calibrators:
            with pytest.raises(ValueError, match=r"(?i)length|shape|empty"):
                cal.fit(x_bad, y_good)

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        x_empty = np.array([])
        y_empty = np.array([])

        calibrators = [
            IsotonicCalibrator(),
            NearlyIsotonicCalibrator(lam=0.5),
            SplineCalibrator(n_knots=5),
        ]

        for cal in calibrators:
            with pytest.raises(ValueError, match=r"(?i)length|shape|empty"):
                cal.fit(x_empty, y_empty)

    def test_transform_before_fit(self):
        """Test that transform raises error when called before fit."""
        x = np.array([0.1, 0.2, 0.3])

        calibrators = [
            IsotonicCalibrator(),
            NearlyIsotonicCalibrator(),
            SplineCalibrator(),
        ]

        for cal in calibrators:
            with pytest.raises((ValueError, AttributeError)):
                cal.transform(x)

    @pytest.mark.parametrize(
        "sample_weight",
        [
            np.zeros(3),
            np.array([1.0, np.nan, 1.0]),
            np.array([1.0, np.inf, 1.0]),
        ],
    )
    def test_cdi_rejects_weights_that_cannot_define_a_fit(self, sample_weight):
        """Invalid weights must not silently produce zeros or NaNs."""
        with pytest.raises(ValueError, match="sample_weight"):
            CDIIsotonicCalibrator(thresholds=[0.5]).fit(
                np.array([0.1, 0.5, 0.9]),
                np.array([0.0, 1.0, 1.0]),
                sample_weight=sample_weight,
            )

    @pytest.mark.parametrize(
        ("scores", "targets"),
        [
            (np.array([0.1, np.nan, 0.9]), np.array([0.0, 1.0, 1.0])),
            (np.array([0.1, 0.5, 0.9]), np.array([0.0, np.inf, 1.0])),
        ],
    )
    def test_cdi_rejects_nonfinite_positive_mass_data(self, scores, targets):
        """Positive-mass scores and targets must be finite."""
        with pytest.raises(ValueError, match="finite"):
            CDIIsotonicCalibrator(thresholds=[0.5]).fit(scores, targets)


class TestCalibratorCommonInterface:
    """Test that all calibrators follow the common interface."""

    @pytest.fixture
    def all_calibrators(self):
        """Fixture providing all calibrator instances."""
        return [
            IsotonicCalibrator(),
            NearlyIsotonicCalibrator(lam=0.5),
            SplineCalibrator(n_knots=8, cv=3),
            RelaxedPAVACalibrator(min_increment=-0.02),
            SplineCalibrator(alpha=0.1),
        ]

    def test_fit_returns_self(self, all_calibrators, binary_data):
        """Test that fit() returns self for method chaining."""
        x, y = binary_data

        for cal in all_calibrators:
            result = cal.fit(x, y)
            assert result is cal

    def test_transform_output_shape(self, all_calibrators, binary_data):
        """Test that transform() returns correct output shape."""
        x, y = binary_data

        for cal in all_calibrators:
            cal.fit(x, y)
            y_calib = cal.transform(x)
            assert len(y_calib) == len(x)
            assert isinstance(y_calib, np.ndarray)

    def test_fit_transform_equivalence(self, all_calibrators, binary_data):
        """Test that fit_transform gives same result as fit + transform."""
        x, y = binary_data

        for cal in all_calibrators:
            # Test fit_transform
            cal1 = cal.__class__(**cal.get_params())
            y_calib_1 = cal1.fit_transform(x, y)

            # Test fit + transform
            cal2 = cal.__class__(**cal.get_params())
            cal2.fit(x, y)
            y_calib_2 = cal2.transform(x)

            np.testing.assert_array_almost_equal(y_calib_1, y_calib_2)

    def test_diagnostics_toggle(self, all_calibrators, binary_data):
        """Test that diagnostics can be enabled/disabled."""
        x, y = binary_data

        for cal in all_calibrators:
            # Get base parameters
            params = cal.get_params()

            # Test with diagnostics disabled
            params_no_diag = params.copy()
            params_no_diag.pop("enable_diagnostics", None)  # Remove to avoid conflict
            params_no_diag["enable_diagnostics"] = False
            cal_no_diag = cal.__class__(**params_no_diag)
            cal_no_diag.fit(x, y)
            assert not cal_no_diag.has_diagnostics()

            # Test with diagnostics enabled
            params_with_diag = params.copy()
            params_with_diag.pop("enable_diagnostics", None)  # Remove to avoid conflict
            params_with_diag["enable_diagnostics"] = True
            cal_with_diag = cal.__class__(**params_with_diag)
            cal_with_diag.fit(x, y)
            assert cal_with_diag.has_diagnostics()
            assert isinstance(cal_with_diag.get_diagnostics(), dict)
