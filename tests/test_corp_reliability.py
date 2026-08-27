"""Contract, reference-identity, and validation tests for CORP diagrams."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

from calibre import corp_reliability


def test_hand_computed_pava_diagram_has_clear_public_fields() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.2, 0.4, 0.6, 0.8])

    diagram = corp_reliability(y_true, y_pred)

    np.testing.assert_array_equal(diagram.prediction_values, y_pred)
    np.testing.assert_array_equal(
        diagram.event_probabilities, np.array([0.0, 0.5, 0.5, 1.0])
    )
    np.testing.assert_array_equal(diagram.prediction_weights, np.ones(4))


def test_frequency_weights_equal_literal_row_replication() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.6, 0.9])
    weights = np.array([1, 3, 2, 4])

    weighted = corp_reliability(y_true, y_pred, sample_weight=weights)
    repeated = corp_reliability(np.repeat(y_true, weights), np.repeat(y_pred, weights))

    np.testing.assert_array_equal(
        weighted.prediction_values, repeated.prediction_values
    )
    np.testing.assert_allclose(
        weighted.event_probabilities, repeated.event_probabilities, atol=1e-15
    )
    np.testing.assert_array_equal(
        weighted.prediction_weights, repeated.prediction_weights
    )


def test_zero_weight_rows_are_absent_from_the_diagram() -> None:
    y_true = np.array([0.0, 1.0, 0.0])
    y_pred = np.array([0.2, 0.8, 0.5])
    weights = np.array([1.0, 1.0, 0.0])

    weighted = corp_reliability(y_true, y_pred, sample_weight=weights)
    expected = corp_reliability(y_true[:2], y_pred[:2])

    np.testing.assert_array_equal(
        weighted.prediction_values, expected.prediction_values
    )
    np.testing.assert_array_equal(
        weighted.event_probabilities, expected.event_probabilities
    )
    np.testing.assert_array_equal(
        weighted.prediction_weights, expected.prediction_weights
    )


def test_row_permutation_does_not_change_a_weighted_diagram() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.2, 0.2, 0.5, 0.8, 0.8])
    weights = np.array([1.0, 2.0, 3.0, 1.0, 4.0])
    order = np.array([4, 0, 3, 1, 2])

    expected = corp_reliability(y_true, y_pred, sample_weight=weights)
    actual = corp_reliability(
        y_true[order], y_pred[order], sample_weight=weights[order]
    )

    np.testing.assert_array_equal(actual.prediction_values, expected.prediction_values)
    np.testing.assert_allclose(
        actual.event_probabilities, expected.event_probabilities, atol=1e-15
    )
    np.testing.assert_array_equal(
        actual.prediction_weights, expected.prediction_weights
    )


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
        corp_reliability(np.asarray(y_true), np.asarray(y_pred))


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
        corp_reliability(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.8]),
            sample_weight=np.asarray(sample_weight),
        )


def test_signature_is_consistent_and_has_no_compatibility_fields() -> None:
    signature = inspect.signature(corp_reliability)

    assert list(signature.parameters) == ["y_true", "y_pred", "sample_weight"]
    assert signature.parameters["sample_weight"].kind is inspect.Parameter.KEYWORD_ONLY

    diagram = corp_reliability(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    assert not hasattr(diagram, "x")
    assert not hasattr(diagram, "cep")
    assert not hasattr(diagram, "weight")

    with pytest.raises(TypeError):
        cast("Any", corp_reliability)(
            np.array([0.0, 1.0]), np.array([0.2, 0.8]), np.ones(2)
        )
