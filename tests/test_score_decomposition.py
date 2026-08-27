"""Contract, numerical, and validation tests for the CORP decomposition."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

from calibre import score_decomposition


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            {
                "mean_score": 0.0,
                "miscalibration": 0.0,
                "discrimination": 0.25,
                "uncertainty": 0.25,
            },
        ),
        (
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            {
                "mean_score": 1.0,
                "miscalibration": 0.75,
                "discrimination": 0.0,
                "uncertainty": 0.25,
            },
        ),
        (
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            {
                "mean_score": 0.25,
                "miscalibration": 0.0,
                "discrimination": 0.0,
                "uncertainty": 0.25,
            },
        ),
    ],
)
def test_exact_brier_controls(
    y_true: np.ndarray, y_pred: np.ndarray, expected: dict[str, float]
) -> None:
    actual = score_decomposition(y_true, y_pred)

    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=1e-15)


def test_vectorized_proper_score_callable_uses_true_then_predicted_order() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.7, 0.6, 0.8])
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def twice_brier(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
        calls.append((true.copy(), pred.copy()))
        return 2.0 * (pred - true) ** 2

    custom = score_decomposition(y_true, y_pred, score=twice_brier)
    standard = score_decomposition(y_true, y_pred, score="brier")

    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0][0], y_true)
    np.testing.assert_array_equal(calls[0][1], y_pred)
    for key in standard:
        assert custom[key] == pytest.approx(2.0 * standard[key], abs=1e-15)


def test_log_score_matches_direct_clipped_formula() -> None:
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([0.1, 0.9, 0.6, 0.2])
    expected = np.mean(-y_true * np.log(y_pred) - (1.0 - y_true) * np.log1p(-y_pred))

    result = score_decomposition(y_true, y_pred, score="log")

    assert result["mean_score"] == pytest.approx(expected, abs=1e-15)
    assert result["miscalibration"] >= -1e-15
    assert result["discrimination"] >= -1e-15


def test_frequency_weights_equal_literal_row_replication() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.6, 0.9])
    weights = np.array([1, 3, 2, 4])

    weighted = score_decomposition(y_true, y_pred, sample_weight=weights)
    repeated = score_decomposition(
        np.repeat(y_true, weights), np.repeat(y_pred, weights)
    )

    for key in repeated:
        assert weighted[key] == pytest.approx(repeated[key], abs=1e-15)


def test_zero_weight_rows_are_outside_the_evaluation_population() -> None:
    y_true = np.array([0.0, 1.0, 99.0])
    y_pred = np.array([0.2, 0.8, np.nan])
    weights = np.array([1.0, 1.0, 0.0])

    weighted = score_decomposition(y_true, y_pred, sample_weight=weights)
    expected = score_decomposition(y_true[:2], y_pred[:2])

    assert weighted == expected


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
        score_decomposition(np.asarray(y_true), np.asarray(y_pred))


@pytest.mark.parametrize(
    ("sample_weight", "message"),
    [
        ([[1.0, 1.0]], "one-dimensional"),
        ([1.0], "same shape"),
        ([1.0, -1.0], "non-negative"),
        ([1.0, np.nan], "finite"),
        ([0.0, 0.0], "positive weight"),
        (["one", "two"], "numeric"),
    ],
)
def test_rejects_invalid_sample_weights(sample_weight: list[Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score_decomposition(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            sample_weight=np.asarray(sample_weight),
        )


@pytest.mark.parametrize(
    "bad_score",
    [
        lambda y_true, y_pred: 1.0,
        lambda y_true, y_pred: np.ones(y_true.size + 1),
        lambda y_true, y_pred: np.full(y_true.shape, np.nan),
    ],
)
def test_rejects_malformed_custom_score_output(bad_score: Any) -> None:
    with pytest.raises(ValueError, match="score callable must return"):
        score_decomposition(np.array([0.0, 1.0]), np.array([0.2, 0.8]), score=bad_score)


def test_signature_and_result_names_are_explicit() -> None:
    signature = inspect.signature(score_decomposition)

    assert list(signature.parameters) == [
        "y_true",
        "y_pred",
        "score",
        "sample_weight",
    ]
    assert signature.parameters["score"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sample_weight"].kind is inspect.Parameter.KEYWORD_ONLY

    result = score_decomposition(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    assert "MCB" not in result
    assert "DSC" not in result
    assert "UNC" not in result

    with pytest.raises(TypeError):
        cast("Any", score_decomposition)(
            np.array([0.0, 1.0]), np.array([0.2, 0.8]), "brier"
        )
