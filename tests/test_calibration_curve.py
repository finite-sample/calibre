"""Reference and scenario tests for calibration_curve."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.calibration import calibration_curve as sklearn_calibration_curve

from calibre import calibration_curve


def _exact_grouped_sample() -> tuple[np.ndarray, np.ndarray]:
    """Return observations with exact event rates at five probabilities."""
    probabilities = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 100)
    outcomes = np.concatenate(
        [
            np.r_[
                np.ones(int(probability * 100)), np.zeros(100 - int(probability * 100))
            ]
            for probability in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
    )
    return outcomes, probabilities


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_unweighted_bin_means_match_sklearn_reference(strategy):
    """Unweighted occupied-bin summaries match scikit-learn exactly."""
    y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=float)
    y_pred = np.array([0.02, 0.08, 0.21, 0.32, 0.49, 0.61, 0.74, 0.91, 1.0])

    prob_true, prob_pred, _ = calibration_curve(
        y_true, y_pred, n_bins=4, strategy=strategy
    )
    expected_true, expected_pred = sklearn_calibration_curve(
        y_true, y_pred, n_bins=4, strategy=strategy
    )

    np.testing.assert_allclose(prob_true, expected_true, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(prob_pred, expected_pred, atol=0.0, rtol=0.0)


def test_known_uniform_bins_have_exact_means_and_counts():
    """A hand-calculated example fixes the return order and count semantics."""
    y_true = np.array([0, 1, 0, 1, 1], dtype=float)
    y_pred = np.array([0.05, 0.15, 0.55, 0.75, 0.95])

    prob_true, prob_pred, counts = calibration_curve(y_true, y_pred, n_bins=2)

    np.testing.assert_allclose(prob_true, [0.5, 2.0 / 3.0])
    np.testing.assert_allclose(prob_pred, [0.1, 0.75])
    np.testing.assert_array_equal(counts, [2, 3])


def test_empty_bins_are_omitted_instead_of_fabricating_zero_points():
    """No observation means no point on the reliability diagram."""
    y_true = np.r_[np.zeros(5), np.ones(5)]
    y_pred = np.r_[np.full(5, 0.1), np.full(5, 0.9)]

    prob_true, prob_pred, counts = calibration_curve(y_true, y_pred, n_bins=5)

    np.testing.assert_array_equal(prob_true, [0.0, 1.0])
    np.testing.assert_allclose(prob_pred, [0.1, 0.9])
    np.testing.assert_array_equal(counts, [5, 5])


def test_probability_endpoints_belong_to_boundary_bins():
    """Both documented endpoints are included without clipping."""
    prob_true, prob_pred, counts = calibration_curve(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), n_bins=2
    )

    np.testing.assert_array_equal(prob_true, [0.0, 1.0])
    np.testing.assert_array_equal(prob_pred, [0.0, 1.0])
    np.testing.assert_array_equal(counts, [1, 1])


def test_weighted_means_match_literal_frequency_replication():
    """Integer observation weights have their literal frequency meaning."""
    y_true = np.array([0.0, 1.0, 1.0])
    y_pred = np.array([0.2, 0.2, 0.8])
    weight = np.array([1, 3, 2])

    weighted_true, weighted_pred, counts = calibration_curve(
        y_true, y_pred, n_bins=2, sample_weight=weight
    )
    repeated_true, repeated_pred, _ = calibration_curve(
        np.repeat(y_true, weight), np.repeat(y_pred, weight), n_bins=2
    )

    np.testing.assert_allclose(weighted_true, repeated_true)
    np.testing.assert_allclose(weighted_pred, repeated_pred)
    np.testing.assert_allclose(weighted_true, [0.75, 1.0])
    np.testing.assert_allclose(weighted_pred, [0.2, 0.8])
    np.testing.assert_array_equal(counts, [2, 1])


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_common_weight_scaling_changes_neither_means_nor_counts(strategy):
    """Evaluation weights represent relative, not absolute, mass."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    weight = np.array([1.0, 2.0, 3.0, 4.0])

    first = calibration_curve(
        y_true, y_pred, n_bins=2, strategy=strategy, sample_weight=weight
    )
    second = calibration_curve(
        y_true, y_pred, n_bins=2, strategy=strategy, sample_weight=17 * weight
    )

    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_zero_weight_observations_are_absent_from_support():
    """A zero-weight row affects neither bin means nor raw active-row counts."""
    result = calibration_curve(
        np.array([0.0, 1.0, 7.0]),
        np.array([0.1, 0.9, 8.0]),
        n_bins=2,
        sample_weight=np.array([1.0, 1.0, 0.0]),
    )

    np.testing.assert_array_equal(result[0], [0.0, 1.0])
    np.testing.assert_allclose(result[1], [0.1, 0.9])
    np.testing.assert_array_equal(result[2], [1, 1])


def test_weighted_quantile_bins_preserve_tied_scores():
    """A repeated prediction is never split to meet an equal-mass target."""
    y_pred = np.repeat([0.1, 0.5, 0.9], [4, 4, 2]).astype(float)
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=float)

    prob_true, prob_pred, counts = calibration_curve(
        y_true,
        y_pred,
        n_bins=3,
        strategy="quantile",
        sample_weight=np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1], dtype=float),
    )

    np.testing.assert_array_equal(counts, [8, 2])
    np.testing.assert_allclose(prob_true, [5.0 / 12.0, 1.0])
    np.testing.assert_allclose(prob_pred, [11.0 / 30.0, 0.9])


def test_bin_count_is_capped_at_positive_weight_sample_size():
    """A huge request cannot allocate bins beyond the active sample size."""
    result = calibration_curve(
        np.array([0.0, 1.0, 1.0]), np.array([0.1, 0.6, 0.9]), n_bins=10_000_000
    )

    assert all(values.size <= 3 for values in result)
    assert np.sum(result[2]) == 3


def test_curve_separates_exact_calibration_from_monotone_distortion():
    """A realistic negative control is far from the diagonal under distortion."""
    y_true, calibrated = _exact_grouped_sample()

    true_good, pred_good, _ = calibration_curve(y_true, calibrated, n_bins=5)
    true_bad, pred_bad, _ = calibration_curve(y_true, calibrated**2, n_bins=5)

    assert np.max(np.abs(true_good - pred_good)) == pytest.approx(0.0, abs=2e-15)
    assert np.max(np.abs(true_bad - pred_bad)) > 0.2


@pytest.mark.parametrize(
    ("y_true", "y_pred", "sample_weight", "match"),
    [
        ([], [], None, "must not be empty"),
        ([[0.0], [1.0]], [0.1, 0.9], None, "one-dimensional"),
        ([0.0, 1.0], [[0.1], [0.9]], None, "one-dimensional"),
        ([0.0], [0.1, 0.9], None, "same shape"),
        ([0.0, 2.0], [0.1, 0.9], None, "binary outcomes"),
        ([0.0, 1.0], [-0.1, 0.9], None, "probabilities"),
        ([0.0, 1.0], [0.1, 1.1], None, "probabilities"),
        ([0.0, np.nan], [0.1, 0.9], None, "binary outcomes"),
        ([0.0, 1.0], [0.1, np.inf], None, "probabilities"),
        (["no", "yes"], [0.1, 0.9], None, "numeric"),
        ([0.0, 1.0], [0.1, 0.9], [-1.0, 1.0], "non-negative"),
        ([0.0, 1.0], [0.1, 0.9], [0.0, 0.0], "positive weight"),
        ([0.0, 1.0], [0.1, 0.9], [1.0], "same shape"),
    ],
)
def test_rejects_invalid_binary_probability_inputs(
    y_true, y_pred, sample_weight, match
):
    """Malformed inputs fail at the public boundary with a useful message."""
    with pytest.raises(ValueError, match=match):
        calibration_curve(
            np.asarray(y_true),
            np.asarray(y_pred),
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True, "5"])
def test_rejects_invalid_bin_counts(n_bins):
    """The bin count must be an integer count, not a coercible value."""
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve(np.array([0.0]), np.array([0.5]), n_bins=n_bins)


def test_rejects_unknown_strategy():
    """Only the two documented binning strategies are accepted."""
    with pytest.raises(ValueError, match="Unknown binning strategy"):
        calibration_curve(
            np.array([0.0]),
            np.array([0.5]),
            strategy="adaptive",  # type: ignore[arg-type]
        )


def test_binning_options_are_keyword_only():
    """The public signature follows the package metric convention."""
    signature = inspect.signature(calibration_curve)

    for name in ("n_bins", "strategy", "sample_weight"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        calibration_curve(np.array([0.0]), np.array([0.5]), 2)  # type: ignore[call-arg]
