"""Reference and scenario tests for the binary Brier score."""

from __future__ import annotations

import numpy as np
import pytest

from calibre import IsotonicCalibrator, brier_score


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


def test_matches_independent_weighted_formula():
    """Unequal evaluation weights produce the literal weighted mean square."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.6, 0.8, 0.9])
    weight = np.array([1.0, 2.0, 7.0, 3.0])
    expected = float(np.average((y_pred - y_true) ** 2, weights=weight))

    actual = brier_score(y_true, y_pred, sample_weight=weight)

    assert actual == pytest.approx(expected, abs=1e-15)


def test_matches_sklearn_documented_binary_example():
    """A published scikit-learn example is an external reference fixture."""
    y_true = np.array([0, 1, 1, 0], dtype=float)
    y_pred = np.array([0.1, 0.9, 0.8, 0.3])

    assert brier_score(y_true, y_pred) == pytest.approx(0.0375, abs=1e-15)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        ([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0], 0.0),
        ([0, 1, 0, 1], [1.0, 0.0, 1.0, 0.0], 1.0),
        ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.25),
        ([0, 1, 1, 0, 1], [0.2, 0.7, 0.8, 0.4, 0.6], 0.098),
    ],
)
def test_known_binary_scores(y_true, y_pred, expected):
    """Boundary, baseline, and existing documented values are exact."""
    assert brier_score(y_true, y_pred) == pytest.approx(expected, abs=1e-15)


def test_weight_scaling_and_integer_replication_are_invariant():
    """Evaluation weights have frequency-weight semantics."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    weight = np.array([1, 3, 2, 4])
    expected = brier_score(y_true, y_pred, sample_weight=weight)

    scaled = brier_score(y_true, y_pred, sample_weight=weight * 17)
    replicated = brier_score(np.repeat(y_true, weight), np.repeat(y_pred, weight))

    assert scaled == pytest.approx(expected, abs=1e-15)
    assert replicated == pytest.approx(expected, abs=1e-15)


def test_zero_weight_rows_are_ignored_completely():
    """A zero-mass row cannot affect validation or the score."""
    result = brier_score(
        np.array([0.0, 1.0, 99.0]),
        np.array([0.2, 0.8, np.nan]),
        sample_weight=np.array([1.0, 1.0, 0.0]),
    )

    assert result == pytest.approx(0.04)


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
    """The documented binary-outcome and probability domains are enforced."""
    with pytest.raises(ValueError, match=match):
        brier_score(y_true, y_pred)


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
        brier_score(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            sample_weight=sample_weight,
        )


def test_rejects_empty_mismatched_and_multidimensional_inputs():
    """The documented one-dimensional nonempty contract is enforced."""
    with pytest.raises(ValueError, match="must not be empty"):
        brier_score(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same shape"):
        brier_score(np.array([0.0]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="one-dimensional"):
        brier_score(np.array([[0.0], [1.0]]), np.array([[0.2], [0.8]]))


def test_heldout_pre_post_and_resolution_controls():
    """Brier rewards calibration and rejects destructive base-rate collapse."""
    probability, train_outcomes = _exact_grouped_sample()
    score = probability**2
    eval_probability, eval_outcomes = _exact_grouped_sample()
    eval_score = eval_probability**2

    calibrated = IsotonicCalibrator().fit(score, train_outcomes).transform(eval_score)
    harmful = (
        IsotonicCalibrator()
        .fit(score, np.ones_like(train_outcomes))
        .transform(eval_score)
    )
    base_rate = np.full_like(eval_score, np.mean(train_outcomes))

    before = brier_score(eval_outcomes, eval_score)
    after = brier_score(eval_outcomes, calibrated)
    harmful_score = brier_score(eval_outcomes, harmful)
    collapsed = brier_score(eval_outcomes, base_rate)

    assert after < before
    assert harmful_score > before
    assert collapsed > after


def test_calibrated_forecast_minimises_expected_score():
    """The proper-score ordering holds exactly on a deterministic population."""
    probability, outcomes = _exact_grouped_sample()
    overconfident = np.clip(1.4 * probability - 0.2, 0.0, 1.0)
    underconfident = 0.5 + 0.5 * (probability - 0.5)

    calibrated = brier_score(outcomes, probability)

    assert calibrated < brier_score(outcomes, overconfident)
    assert calibrated < brier_score(outcomes, underconfident)
