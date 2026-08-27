"""Scenario tests for exact distinct-value counts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from calibre import brier_score, unique_value_counts


def test_unchanged_predictions_have_ratio_one():
    """An unchanged prediction vector retains every distinct value."""
    predictions = np.array([0.1, 0.2, 0.2, 0.7, 0.9, 0.9])

    result = unique_value_counts(predictions, original_predictions=predictions.copy())

    assert result == {
        "n_unique_predictions": 4,
        "n_unique_original_predictions": 4,
        "unique_prediction_ratio": 1.0,
    }


def test_known_plateau_collapse_has_exact_ratio():
    """Six original scores collapsed into two plateaus have ratio one-third."""
    original = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    calibrated = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])

    result = unique_value_counts(calibrated, original_predictions=original)

    assert result == {
        "n_unique_predictions": 2,
        "n_unique_original_predictions": 6,
        "unique_prediction_ratio": pytest.approx(1 / 3),
    }


def test_nearby_floating_point_values_remain_distinct():
    """No hidden decimal rounding changes the structural count."""
    result = unique_value_counts(np.array([0.1, 0.1000004, 0.1]))

    assert result == {"n_unique_predictions": 2}


def test_repeating_every_observation_does_not_change_counts():
    """Frequency replication changes mass but not the set of outputs."""
    original = np.array([0.1, 0.2, 0.3, 0.4])
    calibrated = np.array([0.2, 0.2, 0.8, 0.8])
    expected = unique_value_counts(calibrated, original_predictions=original)

    repeated = unique_value_counts(
        np.repeat(calibrated, 7), original_predictions=np.repeat(original, 7)
    )

    assert repeated == expected


def test_identical_counts_do_not_imply_identical_forecast_quality():
    """A negative control prevents counts being interpreted as resolution or skill."""
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])
    useful = np.array([0.1, 0.1, 0.9, 0.9])
    reversed_forecast = np.array([0.9, 0.9, 0.1, 0.1])

    assert unique_value_counts(useful) == unique_value_counts(reversed_forecast)
    assert brier_score(outcomes, useful) == pytest.approx(0.01)
    assert brier_score(outcomes, reversed_forecast) == pytest.approx(0.81)


@pytest.mark.parametrize(
    ("predictions", "original", "match"),
    [
        ([], None, "must not be empty"),
        ([[0.1], [0.2]], None, "one-dimensional"),
        ([0.1, np.nan], None, "finite"),
        ([0.1, np.inf], None, "finite"),
        (["low", "high"], None, "numeric"),
        ([0.1, 0.2], [0.1], "same length"),
        ([0.1, 0.2], [[0.1], [0.2]], "one-dimensional"),
        ([0.1, 0.2], [0.1, np.nan], "finite"),
    ],
)
def test_rejects_invalid_or_misaligned_vectors(predictions, original, match):
    """Both vectors must be finite, numeric, one-dimensional, and aligned."""
    with pytest.raises(ValueError, match=match):
        unique_value_counts(predictions, original_predictions=original)


def test_original_predictions_are_keyword_only():
    """Before and after vectors cannot be silently reversed by position."""
    call = cast("Any", unique_value_counts)
    with pytest.raises(TypeError):
        call(np.arange(3), np.arange(3))
