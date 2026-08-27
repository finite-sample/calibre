"""Reference and scenario tests for root mean squared calibration error."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import calibre
from calibre import IsotonicCalibrator, root_mean_squared_calibration_error
from calibre.metrics import brier_score


def _uniform_rmsce_reference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int,
    sample_weight: np.ndarray,
) -> float:
    """Direct implementation of the published mass-weighted formula."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.searchsorted(edges[1:-1], y_pred, side="right")
    total = 0.0
    for index in range(n_bins):
        mask = bin_id == index
        mass = float(np.sum(sample_weight[mask]))
        if mass == 0.0:
            continue
        mean_pred = float(np.average(y_pred[mask], weights=sample_weight[mask]))
        mean_true = float(np.average(y_true[mask], weights=sample_weight[mask]))
        total += mass / np.sum(sample_weight) * (mean_pred - mean_true) ** 2
    return float(np.sqrt(total))


def _exact_grouped_sample() -> tuple[np.ndarray, np.ndarray]:
    """Return separate observations with exact rates at five probabilities."""
    probabilities = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 100)
    outcomes = np.concatenate(
        [
            np.r_[
                np.ones(int(probability * 100)), np.zeros(100 - int(probability * 100))
            ]
            for probability in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
    )
    return probabilities, outcomes


def test_matches_independent_mass_weighted_reference_formula():
    """The implementation must match RMSCE, including unequal bin masses."""
    y_true = np.array([0, 0, 1, 0, 1, 1, 1], dtype=float)
    y_pred = np.array([0.05, 0.15, 0.22, 0.41, 0.62, 0.87, 0.95])
    weight = np.array([1.0, 2.0, 4.0, 1.5, 0.5, 3.0, 7.0])
    expected = _uniform_rmsce_reference(y_true, y_pred, 4, weight)

    actual = root_mean_squared_calibration_error(
        y_true, y_pred, n_bins=4, sample_weight=weight
    )

    assert actual == pytest.approx(expected, abs=1e-15)


def test_matches_published_torchmetrics_l2_example():
    """The public TorchMetrics RMSCE example is an external reference fixture."""
    y_true = np.array([0, 0, 1, 1, 1], dtype=float)
    y_pred = np.array([0.25, 0.25, 0.55, 0.75, 0.75])

    result = root_mean_squared_calibration_error(y_true, y_pred, n_bins=2)

    assert result == pytest.approx(0.2918332857414772, abs=1e-15)


def test_sparse_bad_bin_has_only_its_sample_mass():
    """One bad forecast must not receive the weight of a thousand forecasts."""
    y_true = np.r_[np.tile([0.0, 1.0], 500), 0.0]
    y_pred = np.r_[np.full(1000, 0.5), 0.99]

    result = root_mean_squared_calibration_error(y_true, y_pred, n_bins=10)

    assert result == pytest.approx(0.99 / np.sqrt(1001), abs=1e-15)
    assert result == pytest.approx(0.031290907291, abs=1e-12)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (np.zeros(20), np.full(20, 0.8), 0.8),
        (np.ones(20), np.full(20, 0.8), 0.2),
        (np.r_[np.zeros(10), np.ones(10)], np.full(20, 0.5), 0.0),
    ],
)
def test_known_constant_prediction_values(y_true, y_pred, expected):
    """Single occupied-bin cases have exact, interpretable answers."""
    assert root_mean_squared_calibration_error(y_true, y_pred) == pytest.approx(
        expected
    )


def test_quantile_bins_preserve_ties_and_report_actual_bins():
    """A repeated score is never split merely to hit a requested bin count."""
    y_pred = np.repeat([0.1, 0.5, 0.9], [4, 4, 2]).astype(float)
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=float)

    details = root_mean_squared_calibration_error(
        y_true, y_pred, n_bins=3, strategy="quantile", return_details=True
    )

    np.testing.assert_array_equal(details["bin_counts"], [4, 4, 2])
    np.testing.assert_allclose(details["bin_score_minimums"], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(details["bin_score_maximums"], [0.1, 0.5, 0.9])
    assert details["root_mean_squared_calibration_error"] == pytest.approx(
        np.sqrt((4 * 0.15**2 + 4 * 0.0**2 + 2 * 0.1**2) / 10)
    )


def test_bin_count_is_capped_at_positive_weight_sample_size():
    """Requesting empty bins cannot allocate work beyond the evaluation sample."""
    y_true = np.array([0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.6, 0.9])

    capped = root_mean_squared_calibration_error(y_true, y_pred, n_bins=3)
    oversized = root_mean_squared_calibration_error(y_true, y_pred, n_bins=1_000_000)

    assert oversized == pytest.approx(capped)


def test_weight_scaling_and_integer_replication_are_invariant():
    """Weights behave as observation mass under quantile binning and scoring."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    weight = np.array([1, 3, 2, 4])
    expected = root_mean_squared_calibration_error(
        y_true, y_pred, n_bins=3, strategy="quantile", sample_weight=weight
    )

    scaled = root_mean_squared_calibration_error(
        y_true,
        y_pred,
        n_bins=3,
        strategy="quantile",
        sample_weight=weight * 17,
    )
    replicated = root_mean_squared_calibration_error(
        np.repeat(y_true, weight),
        np.repeat(y_pred, weight),
        n_bins=3,
        strategy="quantile",
    )

    assert scaled == pytest.approx(expected, abs=1e-15)
    assert replicated == pytest.approx(expected, abs=1e-15)


def test_zero_weight_rows_are_ignored_completely():
    """A zero-mass row cannot affect domains, bins, summaries, or the score."""
    details = root_mean_squared_calibration_error(
        np.array([0.0, 1.0, 99.0]),
        np.array([0.2, 0.8, np.nan]),
        sample_weight=np.array([1.0, 1.0, 0.0]),
        return_details=True,
    )

    assert details["root_mean_squared_calibration_error"] == pytest.approx(0.2)
    assert np.sum(details["bin_counts"]) == 2
    assert np.sum(details["bin_weights"]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ([0, 2], [0.2, 0.8], "binary outcomes"),
        ([0, np.nan], [0.2, 0.8], "binary outcomes"),
        ([0, 1], [-0.1, 0.8], "probabilities"),
        ([0, 1], [0.2, 1.1], "probabilities"),
        ([0, 1], [0.2, np.inf], "probabilities"),
    ],
)
def test_rejects_values_outside_binary_probability_domains(y_true, y_pred, match):
    """The metric has binary-outcome and probability input semantics."""
    with pytest.raises(ValueError, match=match):
        root_mean_squared_calibration_error(y_true, y_pred)


@pytest.mark.parametrize("n_bins", [True, np.bool_(False), 2.5, "10", 0, -1])
def test_rejects_invalid_bin_counts(n_bins):
    """Only positive integers denote a number of bins."""
    with pytest.raises(ValueError, match="n_bins"):
        root_mean_squared_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            n_bins=cast("Any", n_bins),
        )


@pytest.mark.parametrize(
    ("sample_weight", "match"),
    [
        (np.ones((2, 1)), "one-dimensional"),
        (np.ones(3), "same shape"),
        (np.array([1.0, -1.0]), "non-negative"),
        (np.array([1.0, np.inf]), "finite"),
        (np.zeros(2), "positive"),
    ],
)
def test_rejects_invalid_sample_weights(sample_weight, match):
    """Malformed evaluation weights fail with actionable errors."""
    with pytest.raises(ValueError, match=match):
        root_mean_squared_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            sample_weight=sample_weight,
        )


def test_rejects_empty_mismatched_and_multidimensional_inputs():
    """The documented one-dimensional nonempty contract is enforced."""
    with pytest.raises(ValueError, match="must not be empty"):
        root_mean_squared_calibration_error(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same shape"):
        root_mean_squared_calibration_error(np.array([0.0]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="one-dimensional"):
        root_mean_squared_calibration_error(
            np.array([[0.0], [1.0]]), np.array([[0.2], [0.8]])
        )
    with pytest.raises(TypeError, match="return_details must be boolean"):
        root_mean_squared_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            return_details=cast("Any", 1),
        )


def test_rejects_unknown_strategy():
    """An unknown binning rule cannot silently fall back to a default."""
    with pytest.raises(ValueError, match="Unknown binning strategy"):
        root_mean_squared_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            strategy=cast("Any", "adaptive"),
        )


def test_heldout_pre_post_and_resolution_controls():
    """Calibration improves held-out RMSCE while Brier rejects a collapsed model."""
    probability, train_outcomes = _exact_grouped_sample()
    score = probability**2
    eval_probability, eval_outcomes = _exact_grouped_sample()
    eval_score = eval_probability**2

    calibrated = IsotonicCalibrator().fit(score, train_outcomes).transform(eval_score)
    bad = (
        IsotonicCalibrator()
        .fit(score, np.ones_like(train_outcomes))
        .transform(eval_score)
    )
    base_rate = np.full_like(eval_score, np.mean(train_outcomes))

    before = root_mean_squared_calibration_error(eval_outcomes, eval_score, n_bins=5)
    after = root_mean_squared_calibration_error(eval_outcomes, calibrated, n_bins=5)
    bad_error = root_mean_squared_calibration_error(eval_outcomes, bad, n_bins=5)
    collapsed_error = root_mean_squared_calibration_error(
        eval_outcomes, base_rate, n_bins=5
    )

    assert after == pytest.approx(0.0, abs=1e-15)
    assert before > 0.15
    assert bad_error > before
    assert collapsed_error == pytest.approx(0.0, abs=1e-15)
    assert brier_score(eval_outcomes, calibrated) < brier_score(
        eval_outcomes, base_rate
    )


def test_removed_ambiguous_name_has_no_compatibility_alias():
    """The old nonstandard function does not remain in the public namespace."""
    assert not hasattr(calibre, "binned_calibration_error")
