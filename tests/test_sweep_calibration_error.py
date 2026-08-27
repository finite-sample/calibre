"""Reference and scenario tests for monotonic sweep calibration error."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

from calibre import sweep_calibration_error


def _google_reference_sweep(
    y_true: np.ndarray, y_pred: np.ndarray, p: int
) -> tuple[float, int]:
    """Reproduce the authors' equal-mass sweep for untied predictions."""
    order = np.argsort(y_pred, kind="stable")
    previous = np.zeros(y_true.size, dtype=int)

    def summarize(bin_id: np.ndarray) -> tuple[float, bool]:
        n_used = int(np.max(bin_id)) + 1
        total = 0.0
        previous_rate = -np.inf
        monotone = True
        for bin_number in range(n_used):
            member = bin_id == bin_number
            mean_true = float(np.mean(y_true[member]))
            mean_pred = float(np.mean(y_pred[member]))
            monotone = monotone and mean_true >= previous_rate
            previous_rate = mean_true
            total += int(np.sum(member)) * abs(mean_true - mean_pred) ** p
        return (total / y_true.size) ** (1.0 / p), monotone

    for n_bins in range(2, y_true.size + 1):
        current = np.empty(y_true.size, dtype=int)
        current[order] = np.minimum(
            n_bins - 1,
            np.floor(np.arange(y_true.size) / y_true.size * n_bins),
        ).astype(int)
        _, monotone = summarize(current)
        if not monotone:
            value, _ = summarize(previous)
            return value, int(np.max(previous)) + 1
        previous = current
    value, _ = summarize(previous)
    return value, int(np.max(previous)) + 1


def test_matches_google_reference_on_uneven_equal_mass_bins():
    """Remainder rows go into the same bins as the authors' implementation."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.038, 0.067, 0.399, 0.535, 0.734])
    expected = _google_reference_sweep(y_true, y_pred, p=2)

    actual = sweep_calibration_error(y_true, y_pred, norm=2, return_n_bins=True)

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == expected[1] == 3


def test_default_is_the_research_reference_l2_norm():
    """The paper's experiments and accompanying implementation use L2."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.4, 0.6, 0.7])

    default = sweep_calibration_error(y_true, y_pred)
    l2 = sweep_calibration_error(y_true, y_pred, norm=2)
    l1 = sweep_calibration_error(y_true, y_pred, norm=1)

    assert default == pytest.approx(l2)
    assert default != pytest.approx(l1)


def test_one_observation_is_a_one_bin_plugin_estimate():
    """With no sweep possible, the only bin still has a measurable gap."""
    result = sweep_calibration_error(
        np.array([1.0]), np.array([0.3]), return_n_bins=True
    )

    assert result == pytest.approx((0.7, 1))


def test_exact_group_calibration_scores_zero_and_reports_supported_bins():
    """Tied score groups with matching event rates are calibrated exactly."""
    y_true = np.concatenate(
        [np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]]
    )
    y_pred = np.repeat([0.2, 0.8], 100)

    error, n_bins = sweep_calibration_error(y_true, y_pred, return_n_bins=True)

    assert error == pytest.approx(0.0)
    assert n_bins == 2


def test_tied_predictions_are_not_split_by_input_order():
    """The sweep cannot manufacture monotonicity from ordering within a tie."""
    rng = np.random.default_rng(11)
    y_true = np.tile([0.0, 0.0, 1.0, 1.0], 100)
    y_pred = np.repeat([0.2, 0.5, 0.8, 0.9], 100)
    order = rng.permutation(y_true.size)

    expected = sweep_calibration_error(y_true, y_pred, return_n_bins=True)
    actual = sweep_calibration_error(y_true[order], y_pred[order], return_n_bins=True)

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == expected[1]


def test_detects_known_monotone_miscalibration():
    """A monotone affine distortion remains visible at supported resolution."""
    rng = np.random.default_rng(23)
    true_probability = rng.beta(2.0, 5.0, size=20_000)
    y_true = rng.binomial(1, true_probability).astype(float)
    y_pred = 0.5 * true_probability + 0.25
    population_error = float(np.sqrt(np.mean((y_pred - true_probability) ** 2)))

    actual = sweep_calibration_error(y_true, y_pred)

    assert actual == pytest.approx(population_error, abs=0.02)


def test_nonmonotone_calibration_curve_is_a_negative_control():
    """The cited monotonicity assumption is load-bearing, not a universal fact."""
    y_true = np.concatenate(
        [np.r_[np.ones(80), np.zeros(20)], np.r_[np.ones(20), np.zeros(80)]]
    )
    y_pred = np.repeat([0.2, 0.8], 100)

    error, n_bins = sweep_calibration_error(y_true, y_pred, return_n_bins=True)

    assert error == pytest.approx(0.0)
    assert n_bins == 1
    assert np.sqrt(
        np.mean((y_pred - np.repeat([0.8, 0.2], 100)) ** 2)
    ) == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ([], [], "must not be empty"),
        ([[0.0], [1.0]], [0.1, 0.9], "one-dimensional"),
        ([0.0, 1.0], [[0.1], [0.9]], "one-dimensional"),
        ([0.0], [0.1, 0.9], "same shape"),
        ([0.0, 2.0], [0.1, 0.9], "binary outcomes"),
        ([0.0, 1.0], [-0.1, 0.9], "probabilities"),
        ([0.0, 1.0], [0.1, 1.1], "probabilities"),
        ([0.0, np.nan], [0.1, 0.9], "binary outcomes"),
        ([0.0, 1.0], [0.1, np.inf], "probabilities"),
        (["no", "yes"], [0.1, 0.9], "numeric"),
    ],
)
def test_rejects_invalid_binary_probability_inputs(
    y_true: Any, y_pred: Any, match: str
):
    """Malformed inputs fail at the public boundary with a useful message."""
    with pytest.raises(ValueError, match=match):
        sweep_calibration_error(np.asarray(y_true), np.asarray(y_pred))


@pytest.mark.parametrize("p", [True, np.bool_(False), 0, -1, 1.5, "2", np.inf])
def test_rejects_invalid_norms(p: Any):
    """The cited implementation defines positive integer L-p norms."""
    with pytest.raises(ValueError, match="norm"):
        sweep_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            norm=cast("Any", p),
        )


@pytest.mark.parametrize("return_n_bins", [0, 1, "yes", None, 1.0])
def test_rejects_non_boolean_return_option(return_n_bins: Any):
    """The return-shape switch does not silently use Python truthiness."""
    with pytest.raises(ValueError, match="return_n_bins"):
        sweep_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            return_n_bins=cast("Any", return_n_bins),
        )


def test_options_are_keyword_only_and_weights_are_not_advertised():
    """The signature exposes only the statistical design the method supports."""
    signature = inspect.signature(sweep_calibration_error)

    assert signature.parameters["norm"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["return_n_bins"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "sample_weight" not in signature.parameters

    with pytest.raises(TypeError):
        cast("Any", sweep_calibration_error)(
            np.array([0.0, 1.0]), np.array([0.2, 0.8]), 2
        )
