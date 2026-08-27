"""Reference, contract, and realistic tests for ``mean_calibration_error``."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from calibre import brier_score, mean_calibration_error


def test_matches_the_probability_scale_definition_with_weights():
    """Mean calibration is the weighted observed-versus-predicted rate gap."""
    y_true = np.array([0.0, 1.0, 1.0])
    y_pred = np.array([0.2, 0.4, 0.9])
    sample_weight = np.array([1.0, 2.0, 3.0])
    expected = abs(
        np.average(y_pred, weights=sample_weight)
        - np.average(y_true, weights=sample_weight)
    )
    assert mean_calibration_error(
        y_true, y_pred, sample_weight=sample_weight
    ) == pytest.approx(expected)
    assert expected == pytest.approx(13.0 / 60.0)


def test_weights_obey_unit_replication_and_scaling_identities():
    """Evaluation weights must behave like frequency weights for this mean."""
    y_true = np.array([0.0, 1.0, 1.0])
    y_pred = np.array([0.2, 0.4, 0.9])
    weights = np.array([1, 2, 3])

    unweighted = mean_calibration_error(y_true, y_pred)
    assert mean_calibration_error(
        y_true, y_pred, sample_weight=np.ones(3)
    ) == pytest.approx(unweighted)

    repeated = mean_calibration_error(
        np.repeat(y_true, weights), np.repeat(y_pred, weights)
    )
    weighted = mean_calibration_error(y_true, y_pred, sample_weight=weights)
    assert weighted == pytest.approx(repeated)
    assert mean_calibration_error(
        y_true, y_pred, sample_weight=10.0 * weights
    ) == pytest.approx(weighted)


def test_zero_weight_rows_are_outside_the_evaluation_population():
    """An excluded row cannot invalidate or move the weighted estimand."""
    y_true = np.array([0.0, 1.0, 99.0])
    y_pred = np.array([0.2, 0.8, -10.0])
    weights = np.array([1.0, 1.0, 0.0])
    assert mean_calibration_error(
        y_true, y_pred, sample_weight=weights
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "sample_weight", "match"),
    [
        ([], [], None, "must not be empty"),
        ([[0.0, 1.0]], [[0.2, 0.8]], None, "one-dimensional"),
        ([0.0, 1.0], [0.2], None, "same shape"),
        ([0.0, 2.0], [0.2, 0.8], None, "binary outcomes"),
        ([0.0, 1.0], [-0.1, 0.8], None, "probabilities"),
        ([0.0, 1.0], [0.2, 1.1], None, "probabilities"),
        ([0.0, np.nan], [0.2, 0.8], None, "binary outcomes"),
        ([0.0, 1.0], [0.2, np.inf], None, "probabilities"),
        (["no", "yes"], [0.2, 0.8], None, "numeric"),
        ([0.0, 1.0], [0.2, 0.8], [[1.0, 1.0]], "one-dimensional"),
        ([0.0, 1.0], [0.2, 0.8], [1.0], "same shape"),
        ([0.0, 1.0], [0.2, 0.8], [1.0, -1.0], "non-negative"),
        ([0.0, 1.0], [0.2, 0.8], [1.0, np.inf], "non-negative"),
        ([0.0, 1.0], [0.2, 0.8], [0.0, 0.0], "positive weight"),
        ([0.0, 1.0], [0.2, 0.8], ["one", "two"], "numeric"),
    ],
    ids=[
        "empty",
        "two-dimensional",
        "shape",
        "nonbinary",
        "negative-probability",
        "probability-over-one",
        "nonfinite-outcome",
        "nonfinite-probability",
        "nonnumeric-outcome",
        "weight-dimension",
        "weight-shape",
        "negative-weight",
        "nonfinite-weight",
        "zero-weight",
        "nonnumeric-weight",
    ],
)
def test_rejects_values_outside_the_binary_probability_contract(
    y_true,
    y_pred,
    sample_weight,
    match,
):
    """Malformed evaluation data must fail at the public boundary."""
    with pytest.raises(ValueError, match=match):
        mean_calibration_error(y_true, y_pred, sample_weight=sample_weight)


def test_sample_weight_is_keyword_only():
    """Optional controls follow the package-wide metrics convention."""
    signature = inspect.signature(mean_calibration_error)
    assert list(signature.parameters) == ["y_true", "y_pred", "sample_weight"]
    assert signature.parameters["sample_weight"].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        mean_calibration_error([0, 1], [0.2, 0.8], [1.0, 1.0])


def test_exactly_reversed_predictions_are_the_negative_control():
    """Mean calibration can hide maximally wrong individual probabilities."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = 1.0 - y_true
    assert mean_calibration_error(y_true, y_pred) == pytest.approx(0.0)
    assert brier_score(y_true, y_pred) == pytest.approx(1.0)


def test_heldout_pre_post_control_improves_bias_and_proper_score():
    """Correcting a planted global shift should help on both diagnostics."""
    calibrated = np.repeat(np.array([0.2, 0.5, 0.8]), 100)
    y_true = np.concatenate(
        [
            np.r_[np.ones(20), np.zeros(80)],
            np.r_[np.ones(50), np.zeros(50)],
            np.r_[np.ones(80), np.zeros(20)],
        ]
    )
    shifted = calibrated + 0.1

    assert mean_calibration_error(y_true, calibrated) == pytest.approx(0.0)
    assert mean_calibration_error(y_true, shifted) == pytest.approx(0.1)
    assert brier_score(y_true, calibrated) < brier_score(y_true, shifted)
