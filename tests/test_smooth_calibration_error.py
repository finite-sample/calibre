"""Contract and known-truth tests for smooth calibration error."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

import calibre.metrics as metrics
from calibre import smooth_calibration_error


def test_zero_residuals_have_zero_error_and_reference_bandwidth() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = y_true.copy()

    error, bandwidth = smooth_calibration_error(y_true, y_pred, return_bandwidth=True)

    assert error == 0.0
    assert bandwidth == pytest.approx(1.0 / 512.0)


def test_fixed_bandwidth_detects_a_known_constant_gap() -> None:
    y_true = np.zeros(200)
    y_pred = np.full(200, 0.25)

    error = smooth_calibration_error(y_true, y_pred, bandwidth=0.05)

    assert error == pytest.approx(0.25, abs=5e-4)


def test_row_permutation_does_not_change_the_result() -> None:
    rng = np.random.default_rng(20260826)
    y_pred = rng.uniform(0.0, 1.0, 400)
    y_true = rng.binomial(1, y_pred).astype(float)
    order = rng.permutation(y_true.size)

    expected = smooth_calibration_error(y_true, y_pred, return_bandwidth=True)
    actual = smooth_calibration_error(
        y_true[order], y_pred[order], return_bandwidth=True
    )

    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([], [], "must not be empty"),
        ([[0.0, 1.0]], [[0.2, 0.8]], "one-dimensional"),
        ([0.0, 1.0], [0.2], "same shape"),
        ([0.0, 0.5], [0.2, 0.8], "binary outcomes"),
        ([0.0, 2.0], [0.2, 0.8], "binary outcomes"),
        ([0.0, np.nan], [0.2, 0.8], "binary outcomes"),
        ([0.0, 1.0], [-0.1, 0.8], "probabilities"),
        ([0.0, 1.0], [0.2, 1.1], "probabilities"),
        ([0.0, 1.0], [0.2, np.nan], "probabilities"),
        (["no", "yes"], [0.2, 0.8], "numeric"),
        ([0.0, 1.0], ["low", "high"], "numeric"),
    ],
)
def test_rejects_invalid_binary_probability_inputs(
    y_true: list[Any], y_pred: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_calibration_error(np.asarray(y_true), np.asarray(y_pred))


@pytest.mark.parametrize("bandwidth", [0.0, -0.1, np.nan, np.inf, True, "0.05"])
def test_rejects_invalid_bandwidth(bandwidth: Any) -> None:
    with pytest.raises(ValueError, match="bandwidth must be a finite number"):
        smooth_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.25, 0.75]),
            bandwidth=bandwidth,
        )


def test_rejects_bandwidth_below_reference_floor_before_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_smoothing(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("invalid bandwidth reached the smoothing grid")

    monkeypatch.setattr(metrics, "_smooth_at", unexpected_smoothing)

    with pytest.raises(ValueError, match=r"at least 0\.001"):
        smooth_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.25, 0.75]),
            bandwidth=0.0009,
        )


@pytest.mark.parametrize("return_bandwidth", [0, 1, "yes", None])
def test_rejects_non_boolean_return_bandwidth(return_bandwidth: Any) -> None:
    with pytest.raises(TypeError, match="return_bandwidth must be boolean"):
        smooth_calibration_error(
            np.array([0.0, 1.0]),
            np.array([0.25, 0.75]),
            return_bandwidth=return_bandwidth,
        )


def test_options_are_keyword_only_and_weights_are_not_claimed() -> None:
    signature = inspect.signature(smooth_calibration_error)

    assert signature.parameters["bandwidth"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["return_bandwidth"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert "sample_weight" not in signature.parameters

    with pytest.raises(TypeError):
        cast("Any", smooth_calibration_error)(
            np.array([0.0, 1.0]), np.array([0.25, 0.75]), 0.05
        )
