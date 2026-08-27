"""Reference and scenario tests for maximum calibration error."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from calibre import IsotonicCalibrator, maximum_calibration_error
from calibre.metrics import brier_score


def _uniform_mce_reference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int,
    sample_weight: np.ndarray,
) -> float:
    """Direct implementation of the published weighted infinity norm."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.searchsorted(edges[1:-1], y_pred, side="right")
    gaps = []
    for index in range(n_bins):
        mask = bin_id == index
        if not np.any(mask):
            continue
        mean_pred = float(np.average(y_pred[mask], weights=sample_weight[mask]))
        mean_true = float(np.average(y_true[mask], weights=sample_weight[mask]))
        gaps.append(abs(mean_pred - mean_true))
    return max(gaps)


def _exact_grouped_sample() -> tuple[np.ndarray, np.ndarray]:
    """Return observations with exact rates at five forecast values."""
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


def test_matches_independent_weighted_reference_formula():
    """Weights affect within-bin means but not a bin's maximum-norm influence."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.2, 0.7, 0.8])
    weight = np.array([9.0, 1.0, 1.0, 9.0])
    expected = _uniform_mce_reference(y_true, y_pred, 2, weight)

    actual = maximum_calibration_error(y_true, y_pred, n_bins=2, sample_weight=weight)

    assert actual == pytest.approx(expected, abs=1e-15)


def test_matches_published_torchmetrics_max_example():
    """The public TorchMetrics MCE example is an external reference fixture."""
    y_true = np.array([0, 0, 1, 1, 1], dtype=float)
    y_pred = np.array([0.25, 0.25, 0.55, 0.75, 0.75])

    assert maximum_calibration_error(y_true, y_pred, n_bins=2) == pytest.approx(
        19 / 60, abs=1e-15
    )


def test_equal_and_opposite_errors_can_cancel_inside_a_bin():
    """The test records the fixed-bin limitation instead of hiding it."""
    y_pred = np.r_[np.full(100, 0.01), np.full(100, 0.09)]
    y_true = np.r_[np.ones(9), np.zeros(91), np.ones(1), np.zeros(99)]

    hidden = maximum_calibration_error(y_true, y_pred, n_bins=10)
    resolved = maximum_calibration_error(y_true, y_pred, n_bins=100)

    assert hidden == pytest.approx(0.0, abs=1e-15)
    assert resolved == pytest.approx(0.08, abs=1e-15)


def test_one_low_mass_bad_bin_determines_the_entire_score():
    """Worst-bin weighting is an intentional property, not sample-mass weighting."""
    y_true = np.r_[np.ones(90), np.zeros(810), 0.0]
    y_pred = np.r_[np.full(900, 0.1), 0.99]

    result = maximum_calibration_error(y_true, y_pred, n_bins=10)

    assert result == pytest.approx(0.99, abs=1e-15)


def test_weight_scaling_and_integer_replication_are_invariant():
    """Evaluation weights have frequency-weight semantics."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    weight = np.array([1, 3, 2, 4])
    expected = maximum_calibration_error(y_true, y_pred, n_bins=4, sample_weight=weight)

    scaled = maximum_calibration_error(
        y_true, y_pred, n_bins=4, sample_weight=weight * 17
    )
    replicated = maximum_calibration_error(
        np.repeat(y_true, weight), np.repeat(y_pred, weight), n_bins=4
    )

    assert scaled == pytest.approx(expected, abs=1e-15)
    assert replicated == pytest.approx(expected, abs=1e-15)


def test_zero_weight_rows_are_ignored_completely():
    """A zero-mass row cannot affect validation, bins, or the score."""
    result = maximum_calibration_error(
        np.array([0.0, 1.0, 99.0]),
        np.array([0.2, 0.8, np.nan]),
        sample_weight=np.array([1.0, 1.0, 0.0]),
    )

    assert result == pytest.approx(0.2)


def test_bin_count_is_capped_at_positive_weight_sample_size():
    """Requesting empty bins cannot allocate beyond the evaluation sample."""
    y_true = np.array([0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.6, 0.9])

    capped = maximum_calibration_error(y_true, y_pred, n_bins=3)
    oversized = maximum_calibration_error(y_true, y_pred, n_bins=1_000_000)

    assert oversized == pytest.approx(capped)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ([0, 2], [0.2, 0.8], "binary outcomes"),
        ([0, np.nan], [0.2, 0.8], "binary outcomes"),
        ([0, 1], [-0.1, 0.8], "probabilities"),
        ([0, 1], [0.2, 1.1], "probabilities"),
        ([0, 1], [0.2, np.inf], "probabilities"),
        ([0, 1], ["low", "high"], "numeric"),
    ],
)
def test_rejects_values_outside_binary_probability_domains(y_true, y_pred, match):
    """MCE has binary-outcome and probability input semantics."""
    with pytest.raises(ValueError, match=match):
        maximum_calibration_error(y_true, y_pred)


@pytest.mark.parametrize("n_bins", [True, np.bool_(False), 2.5, "10", 0, -1])
def test_rejects_invalid_bin_counts(n_bins):
    """Only positive integers denote a number of bins."""
    with pytest.raises(ValueError, match="n_bins"):
        maximum_calibration_error(
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
        (np.array(["one", "two"]), "numeric"),
    ],
)
def test_rejects_invalid_sample_weights(sample_weight, match):
    """Malformed evaluation weights fail with actionable errors."""
    with pytest.raises(ValueError, match=match):
        maximum_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            sample_weight=sample_weight,
        )


def test_rejects_empty_mismatched_and_multidimensional_inputs():
    """The documented one-dimensional nonempty contract is enforced."""
    with pytest.raises(ValueError, match="must not be empty"):
        maximum_calibration_error(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same shape"):
        maximum_calibration_error(np.array([0.0]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="one-dimensional"):
        maximum_calibration_error(np.array([[0.0], [1.0]]), np.array([[0.2], [0.8]]))


def test_bin_count_is_keyword_only():
    """MCE, ECE, and RMSCE expose one bin-count calling convention."""
    call = cast("Any", maximum_calibration_error)
    with pytest.raises(TypeError):
        call(np.array([0.0, 1.0]), np.array([0.2, 0.8]), 2)


def test_heldout_pre_post_and_resolution_controls():
    """Calibration improves held-out MCE while Brier rejects a collapsed model."""
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

    before = maximum_calibration_error(eval_outcomes, eval_score, n_bins=5)
    after = maximum_calibration_error(eval_outcomes, calibrated, n_bins=5)
    bad_error = maximum_calibration_error(eval_outcomes, bad, n_bins=5)
    collapsed_error = maximum_calibration_error(eval_outcomes, base_rate, n_bins=5)

    assert after == pytest.approx(0.0, abs=2e-15)
    assert before > 0.15
    assert bad_error > before
    assert collapsed_error == pytest.approx(0.0, abs=1e-15)
    assert brier_score(eval_outcomes, calibrated) < brier_score(
        eval_outcomes, base_rate
    )
