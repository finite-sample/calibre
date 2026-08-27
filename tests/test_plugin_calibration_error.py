"""Reference and scenario tests for the plugin calibration-error estimator."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

from calibre import plugin_calibration_error


def _grouped_outcomes(event_counts: list[int], group_size: int = 10) -> np.ndarray:
    """Build deterministic binary groups with exact requested event counts."""
    return np.concatenate(
        [np.r_[np.ones(count), np.zeros(group_size - count)] for count in event_counts]
    )


def test_matches_independent_equal_mass_plugin_formula():
    """A direct rank split and formula reproduce the public implementation."""
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    y_pred = np.array([0.03, 0.11, 0.22, 0.38, 0.47, 0.59, 0.71, 0.84, 0.96])
    groups = np.array_split(np.argsort(y_pred, kind="stable"), 3)
    expected = np.sqrt(
        sum(
            len(group)
            / y_true.size
            * abs(np.mean(y_pred[group]) - np.mean(y_true[group])) ** 2
            for group in groups
        )
    )

    actual = plugin_calibration_error(y_true, y_pred, n_bins=3, norm=2)

    assert actual == pytest.approx(expected)


def test_exact_calibration_and_known_miscalibration_have_exact_values():
    """Grouped event rates give zero, then a known common gap of 0.1."""
    y_true = _grouped_outcomes([2, 8])
    calibrated = np.repeat([0.2, 0.8], 10)
    distorted = np.repeat([0.3, 0.7], 10)

    assert plugin_calibration_error(y_true, calibrated, n_bins=2) == pytest.approx(0.0)
    assert plugin_calibration_error(y_true, distorted, n_bins=2) == pytest.approx(0.1)


def test_l1_and_l2_norms_match_hand_calculation():
    """The exposed norm changes only the documented aggregation."""
    y_true = _grouped_outcomes([2, 8])
    y_pred = np.repeat([0.3, 0.5], 10)

    assert plugin_calibration_error(y_true, y_pred, n_bins=2, norm=1) == pytest.approx(
        0.2
    )
    assert plugin_calibration_error(y_true, y_pred, n_bins=2, norm=2) == pytest.approx(
        np.sqrt(0.05)
    )


def test_calibrated_constant_negative_control_scores_zero():
    """The metric intentionally cannot distinguish calibration from resolution."""
    y_true = np.tile([0.0, 1.0], 50)
    y_pred = np.full(y_true.size, 0.5)

    assert plugin_calibration_error(y_true, y_pred, n_bins=15) == pytest.approx(0.0)


def test_weighted_result_matches_literal_frequency_replication():
    """Integer weights have the same result as repeating observations."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    weight = np.array([1, 3, 2, 4])

    weighted = plugin_calibration_error(
        y_true, y_pred, n_bins=3, norm=2, sample_weight=weight
    )
    repeated = plugin_calibration_error(
        np.repeat(y_true, weight),
        np.repeat(y_pred, weight),
        n_bins=3,
        norm=2,
    )

    assert weighted == pytest.approx(repeated)


def test_common_weight_scaling_does_not_change_result():
    """Sample weights express relative evaluation mass."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    weight = np.array([1.0, 2.0, 3.0, 4.0])

    first = plugin_calibration_error(y_true, y_pred, n_bins=2, sample_weight=weight)
    second = plugin_calibration_error(
        y_true, y_pred, n_bins=2, sample_weight=19 * weight
    )

    assert first == pytest.approx(second)


def test_zero_weight_rows_are_outside_the_evaluation_population():
    """An ignored row affects neither validation, bins, nor the estimate."""
    actual = plugin_calibration_error(
        np.array([0.0, 1.0, 7.0]),
        np.array([0.2, 0.8, 9.0]),
        n_bins=2,
        sample_weight=np.array([1.0, 1.0, 0.0]),
    )

    assert actual == pytest.approx(0.2)


def test_requested_bins_never_split_a_tied_prediction_group():
    """Many requested bins cannot turn an arbitrary ordering into error."""
    y_true = np.tile([0.0, 1.0], 50)
    y_pred = np.full(y_true.size, 0.2)

    assert plugin_calibration_error(y_true, y_pred, n_bins=100) == pytest.approx(0.3)


def test_bins_are_capped_at_the_positive_weight_sample_size():
    """An enormous request has bounded work and the same singleton-bin result."""
    y_true = np.array([0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.6, 0.9])

    expected = plugin_calibration_error(y_true, y_pred, n_bins=3)
    actual = plugin_calibration_error(y_true, y_pred, n_bins=10_000_000)

    assert actual == pytest.approx(expected)


def test_plugin_bias_increases_with_excessive_bins_on_calibrated_data():
    """A realistic calibrated sample exposes the estimator's finite-sample bias."""
    rng = np.random.default_rng(20260826)
    y_pred = rng.uniform(size=20_000)
    y_true = rng.binomial(1, y_pred).astype(float)

    coarse = plugin_calibration_error(y_true, y_pred, n_bins=5)
    fine = plugin_calibration_error(y_true, y_pred, n_bins=200)

    assert fine > 3 * coarse


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
    y_true: Any, y_pred: Any, sample_weight: Any, match: str
):
    """Malformed inputs fail at the public boundary with a useful message."""
    with pytest.raises(ValueError, match=match):
        plugin_calibration_error(
            np.asarray(y_true),
            np.asarray(y_pred),
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )


@pytest.mark.parametrize("n_bins", [True, np.bool_(False), 2.5, "10", 0, -1])
def test_rejects_invalid_bin_counts(n_bins: Any):
    """Only positive integers denote a number of bins."""
    with pytest.raises(ValueError, match="n_bins"):
        plugin_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            n_bins=cast("Any", n_bins),
        )


@pytest.mark.parametrize("p", [True, np.bool_(False), 0, -1, 1.5, "2", np.inf])
def test_rejects_invalid_norms(p: Any):
    """The cited estimator defines integer L-p norms at least one."""
    with pytest.raises(ValueError, match="norm"):
        plugin_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            norm=cast("Any", p),
        )


def test_options_are_keyword_only():
    """The public signature follows the package metric convention."""
    signature = inspect.signature(plugin_calibration_error)

    for name in ("n_bins", "norm", "sample_weight"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        cast("Any", plugin_calibration_error)(
            np.array([0.0, 1.0]), np.array([0.2, 0.8]), 2, 2
        )
