"""Reference and scenario tests for the debiased calibration-error estimator."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

from calibre import debiased_calibration_error, plugin_calibration_error


def _reference_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int
) -> float:
    """Implement Kumar, Liang, and Ma's estimator independently."""
    groups = np.array_split(np.argsort(y_pred, kind="stable"), min(n_bins, y_true.size))
    total = 0.0
    for group in groups:
        if group.size < 2:
            continue
        mean_true = float(np.mean(y_true[group]))
        mean_pred = float(np.mean(y_pred[group]))
        correction = mean_true * (1.0 - mean_true) / (group.size - 1)
        total += group.size / y_true.size * ((mean_pred - mean_true) ** 2 - correction)
    return total


def test_matches_the_authors_debiased_squared_formula():
    """An independent rank split and formula reproduce the public result."""
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    y_pred = np.array([0.03, 0.11, 0.22, 0.38, 0.47, 0.59, 0.71, 0.84, 0.96])
    expected = _reference_squared_error(y_true, y_pred, n_bins=3)

    actual = debiased_calibration_error(y_true, y_pred, n_bins=3, squared=True)

    assert actual == pytest.approx(expected)


def test_root_output_is_the_square_root_of_the_positive_squared_estimate():
    """The two output scales differ only by the documented floor and root."""
    y_true = np.tile([0.0, 0.0, 0.0, 1.0], 100)
    y_pred = np.full(y_true.size, 0.55)

    squared = debiased_calibration_error(y_true, y_pred, n_bins=1, squared=True)
    root = debiased_calibration_error(y_true, y_pred, n_bins=1)

    assert squared > 0.0
    assert root == pytest.approx(np.sqrt(squared))


def test_negative_squared_estimate_is_preserved_but_root_output_is_zero():
    """Well-calibrated finite samples may produce a negative unbiased estimate."""
    y_true = np.tile([0.0, 1.0], 50)
    y_pred = np.full(y_true.size, 0.5)

    squared = debiased_calibration_error(y_true, y_pred, n_bins=1, squared=True)

    assert squared == pytest.approx(-0.25 / 99.0)
    assert debiased_calibration_error(y_true, y_pred, n_bins=1) == 0.0


def test_tied_predictions_are_never_split_between_bins():
    """Requested singleton bins cannot erase a measurable tied-group error."""
    first = np.r_[np.ones(20), np.zeros(80)]
    second = np.r_[np.ones(80), np.zeros(20)]
    y_true = np.concatenate([first, second])
    y_pred = np.repeat([0.3, 0.7], 100)
    correction = 0.2 * 0.8 / 99.0
    expected = np.sqrt(0.1**2 - correction)

    actual = debiased_calibration_error(y_true, y_pred, n_bins=200)

    assert actual == pytest.approx(expected)


def test_singleton_bins_follow_the_reference_and_contribute_zero():
    """A bin with no estimable label variance has no correctable contribution."""
    y_true = np.array([0.0, 1.0, 0.0])
    y_pred = np.array([0.1, 0.6, 0.9])

    assert debiased_calibration_error(
        y_true, y_pred, n_bins=3, squared=True
    ) == pytest.approx(0.0)
    assert debiased_calibration_error(
        y_true, y_pred, n_bins=10_000_000, squared=True
    ) == pytest.approx(0.0)


def test_debiasing_removes_plugin_noise_on_realistic_calibrated_data():
    """Known-truth simulation distinguishes finite-sample noise from error."""
    rng = np.random.default_rng(20260826)
    y_pred = rng.beta(2.0, 5.0, size=20_000)
    y_true = rng.binomial(1, y_pred).astype(float)

    plugin = plugin_calibration_error(y_true, y_pred, n_bins=50, norm=2)
    debiased = debiased_calibration_error(y_true, y_pred, n_bins=50)

    assert plugin > 0.01
    assert debiased < plugin / 2.0


def test_detects_real_miscalibration_on_realistic_data():
    """The correction must retain a population-scale probability distortion."""
    rng = np.random.default_rng(7)
    true_probability = rng.beta(2.0, 5.0, size=20_000)
    y_true = rng.binomial(1, true_probability).astype(float)
    y_pred = 0.5 * true_probability + 0.25

    actual = debiased_calibration_error(y_true, y_pred, n_bins=50)
    population_error = float(np.sqrt(np.mean((y_pred - true_probability) ** 2)))

    assert actual == pytest.approx(population_error, abs=0.01)


def test_result_is_invariant_to_row_order_even_with_ties():
    """Input ordering is not an undocumented tie breaker."""
    rng = np.random.default_rng(19)
    y_true = np.tile([0.0, 0.0, 1.0, 1.0], 100)
    y_pred = np.repeat([0.2, 0.5, 0.8, 0.9], 100)
    order = rng.permutation(y_true.size)

    expected = debiased_calibration_error(y_true, y_pred, n_bins=15, squared=True)
    actual = debiased_calibration_error(
        y_true[order], y_pred[order], n_bins=15, squared=True
    )

    assert actual == pytest.approx(expected)


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
        debiased_calibration_error(np.asarray(y_true), np.asarray(y_pred))


@pytest.mark.parametrize("n_bins", [True, np.bool_(False), 2.5, "10", 0, -1])
def test_rejects_invalid_bin_counts(n_bins: Any):
    """Only positive integers denote a number of bins."""
    with pytest.raises(ValueError, match="n_bins"):
        debiased_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            n_bins=cast("Any", n_bins),
        )


@pytest.mark.parametrize("squared", [0, 1, "yes", None, 1.0])
def test_rejects_non_boolean_squared_option(squared: Any):
    """The output-scale switch does not silently use Python truthiness."""
    with pytest.raises(ValueError, match="squared"):
        debiased_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            squared=cast("Any", squared),
        )


def test_options_are_keyword_only_and_weights_are_not_advertised():
    """The public signature is explicit about its supported statistical design."""
    signature = inspect.signature(debiased_calibration_error)

    assert signature.parameters["n_bins"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["squared"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "sample_weight" not in signature.parameters

    with pytest.raises(TypeError):
        cast("Any", debiased_calibration_error)(
            np.array([0.0, 1.0]), np.array([0.2, 0.8]), 2
        )
