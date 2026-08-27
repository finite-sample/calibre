"""Reference, invariant, and realistic-scenario tests for CDI-ISO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss

from calibre import BaseCalibrator, CDIIsotonicCalibrator

if TYPE_CHECKING:
    from collections.abc import Callable


def _grouped_binary_data(
    scores: np.ndarray, rates: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build score groups with exact empirical binary event rates."""
    score_parts = []
    outcome_parts = []
    for score, rate, count in zip(scores, rates, counts, strict=True):
        events = round(float(rate * count))
        score_parts.append(np.full(count, score, dtype=float))
        outcome_parts.append(
            np.r_[np.ones(events, dtype=float), np.zeros(count - events, dtype=float)]
        )
    return np.concatenate(score_parts), np.concatenate(outcome_parts)


def _paper_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four adjacent blocks with exactly known rates and sizes."""
    scores = np.array([0.1, 0.3, 0.6, 0.9])
    rates = np.array([0.1, 0.35, 0.4, 0.8])
    counts = np.array([100, 80, 120, 100])
    X, y = _grouped_binary_data(scores, rates, counts)
    return X, y, rates, counts.astype(float)


def test_local_bounds_match_the_published_adjacent_block_formula():
    """Every term in the paper's CDI bound must be represented literally."""
    X, y, rates, counts = _paper_fixture()
    calibrator = CDIIsotonicCalibrator(
        thresholds=[0.25, 0.75],
        threshold_weights=[1.0, 3.0],
        bandwidth=0.4,
        alpha=0.05,
        gamma=0.5,
        clip_output=False,
    ).fit(X, y)

    economics_weight = np.array([0.21875, 0.3125, 0.75])
    pooled_rate = (counts[:-1] * rates[:-1] + counts[1:] * rates[1:]) / (
        counts[:-1] + counts[1:]
    )
    standard_error = np.sqrt(
        pooled_rate * (1.0 - pooled_rate) * (1.0 / counts[:-1] + 1.0 / counts[1:])
    )
    z_value = 1.959963984540054
    lower_difference = rates[1:] - rates[:-1] - z_value * standard_error
    expected = (
        0.5 * economics_weight * np.maximum(lower_difference, 0.0)
        - (1.0 - economics_weight) * z_value * standard_error
    )

    np.testing.assert_allclose(calibrator.block_rate_, rates, atol=1e-15)
    np.testing.assert_allclose(calibrator.effective_sample_size_, counts)
    np.testing.assert_allclose(calibrator.economics_weight_, economics_weight)
    np.testing.assert_allclose(calibrator.adjacency_bounds_, expected, atol=1e-15)


def test_fit_matches_an_independent_quadratic_program():
    """The complete estimator must attain the documented constrained optimum."""
    import cvxpy as cp

    X, y, _, _ = _paper_fixture()
    calibrator = CDIIsotonicCalibrator(
        thresholds=[0.25, 0.75],
        threshold_weights=[1.0, 3.0],
        bandwidth=0.4,
        alpha=0.05,
        gamma=0.5,
        clip_output=False,
    ).fit(X, y)

    fitted = cp.Variable(calibrator.block_rate_.size)
    objective = cp.sum(
        cp.multiply(
            calibrator.block_weight_, cp.square(calibrator.block_rate_ - fitted)
        )
    )
    problem = cp.Problem(
        cp.Minimize(objective),
        [cp.diff(fitted) >= calibrator.adjacency_bounds_],
    )
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1e-10,
        eps_rel=1e-10,
        polishing=True,
    )

    assert problem.status == "optimal"
    np.testing.assert_allclose(
        calibrator.calibration_curve_.y,
        np.asarray(fitted.value, dtype=float),
        rtol=0,
        atol=1e-8,
    )


def test_common_weight_rescaling_does_not_change_inference_or_fit():
    """Relative sample weights cannot acquire meaning from their arbitrary units."""
    X, y, _, _ = _paper_fixture()
    weight = np.linspace(0.5, 2.0, X.size)

    def fit(scale: float) -> CDIIsotonicCalibrator:
        return CDIIsotonicCalibrator(
            thresholds=[0.3, 0.7], bandwidth=0.25, gamma=0.4
        ).fit(X, y, sample_weight=scale * weight)

    baseline = fit(1.0)
    scaled = fit(100.0)
    grid = np.linspace(0.0, 1.0, 201)
    np.testing.assert_allclose(
        scaled.effective_sample_size_, baseline.effective_sample_size_, atol=1e-12
    )
    np.testing.assert_allclose(
        scaled.adjacency_bounds_, baseline.adjacency_bounds_, atol=1e-12
    )
    np.testing.assert_allclose(scaled.transform(grid), baseline.transform(grid))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: CDIIsotonicCalibrator(thresholds=[]), "non-empty"),
        (lambda: CDIIsotonicCalibrator(thresholds=[np.nan]), "finite"),
        (lambda: CDIIsotonicCalibrator(thresholds=[-0.1]), r"\[0, 1\]"),
        (lambda: CDIIsotonicCalibrator(thresholds=[1.1]), r"\[0, 1\]"),
        (
            lambda: CDIIsotonicCalibrator(
                thresholds=[0.2, 0.8], threshold_weights=[1.0]
            ),
            "match",
        ),
        (
            lambda: CDIIsotonicCalibrator(thresholds=[0.2], threshold_weights=[np.nan]),
            "finite",
        ),
        (
            lambda: CDIIsotonicCalibrator(thresholds=[0.2], threshold_weights=[0.0]),
            "positive",
        ),
        (lambda: CDIIsotonicCalibrator(thresholds=[0.5], bandwidth=0.0), "greater"),
        (lambda: CDIIsotonicCalibrator(thresholds=[0.5], bandwidth=np.nan), "finite"),
        (lambda: CDIIsotonicCalibrator(thresholds=[0.5], alpha=0.0), "strictly"),
        (lambda: CDIIsotonicCalibrator(thresholds=[0.5], alpha=1.0), "strictly"),
        (lambda: CDIIsotonicCalibrator(thresholds=[0.5], gamma=-0.1), "between"),
        (lambda: CDIIsotonicCalibrator(thresholds=[0.5], gamma=1.1), "between"),
        (
            lambda: CDIIsotonicCalibrator(thresholds=[0.5], clip_output="yes"),
            "boolean",
        ),
    ],
)
def test_invalid_design_parameters_are_rejected(
    factory: Callable[[], CDIIsotonicCalibrator], match: str
):
    """Invalid statistical designs must fail instead of being silently coerced."""
    with pytest.raises(ValueError, match=match):
        factory().fit(np.array([0.2, 0.8]), np.array([0.0, 1.0]))


@pytest.mark.parametrize("scores", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
def test_probability_score_scale_is_enforced_at_fit_and_transform(scores):
    """Threshold distances are meaningful only on the documented score scale."""
    calibrator = CDIIsotonicCalibrator(thresholds=[0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrator.fit(scores, np.array([0.0, 1.0]))

    fitted = CDIIsotonicCalibrator(thresholds=[0.5]).fit(
        np.array([0.2, 0.8]), np.array([0.0, 1.0])
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fitted.transform(scores)


def test_binomial_uncertainty_rejects_fractional_targets():
    """The two-proportion calculation cannot silently accept non-binary labels."""
    with pytest.raises(ValueError, match="binary"):
        CDIIsotonicCalibrator(thresholds=[0.5]).fit(
            np.array([0.2, 0.8]), np.array([0.25, 0.75])
        )


def test_sklearn_api_and_fitted_attributes_are_consistent():
    """CDI uses the shared estimator lifecycle and sklearn's fitted-state protocol."""
    calibrator = CDIIsotonicCalibrator(
        thresholds=[0.2, 0.8], threshold_weights=[1.0, 2.0]
    )
    assert isinstance(calibrator, BaseCalibrator)
    cloned = clone(calibrator)
    assert cloned.get_params() == calibrator.get_params()
    with pytest.raises(NotFittedError):
        calibrator.transform(np.array([0.5]))

    X, y, _, _ = _paper_fixture()
    returned = calibrator.fit(X, y)
    assert returned is calibrator
    assert calibrator.fit_transform(X, y).shape == X.shape


def _net_benefit(y: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    """Compute decision-curve net benefit for one validation threshold."""
    positive = prediction >= threshold
    true_positive = np.sum(positive & (y == 1.0))
    false_positive = np.sum(positive & (y == 0.0))
    odds = threshold / (1.0 - threshold)
    return float((true_positive - false_positive * odds) / y.size)


def test_held_out_proper_scores_and_decision_utility_on_supported_data():
    """CDI must pass positive and negative controls on a realistic grouped design."""
    probability = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    raw_score = probability**1.8
    X_fit, y_fit = _grouped_binary_data(
        raw_score, probability, np.full(probability.size, 1000)
    )
    X_test, y_test = _grouped_binary_data(
        raw_score, probability, np.full(probability.size, 2000)
    )
    calibrated = (
        CDIIsotonicCalibrator(thresholds=[0.3], bandwidth=0.35, gamma=0.15)
        .fit(X_fit, y_fit)
        .transform(X_test)
    )

    brier_before = float(np.mean((y_test - X_test) ** 2))
    brier_after = float(np.mean((y_test - calibrated) ** 2))
    log_loss_before = float(log_loss(y_test, X_test))
    log_loss_after = float(log_loss(y_test, calibrated))
    benefit_before = _net_benefit(y_test, X_test, 0.3)
    benefit_after = _net_benefit(y_test, calibrated, 0.3)

    assert brier_after < brier_before
    assert log_loss_after < log_loss_before
    assert benefit_after > benefit_before

    constant = np.full(y_test.size, np.mean(y_fit))
    assert float(np.mean((y_test - constant) ** 2)) > brier_after
    assert float(log_loss(y_test, constant)) > log_loss_after
    assert _net_benefit(y_test, constant, 0.3) < benefit_after
