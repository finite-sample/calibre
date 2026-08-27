"""Reference and scenario tests for Spearman rank diagnostics."""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import pytest

from calibre import brier_score, correlation_metrics


def test_known_increasing_decreasing_and_tied_correlations():
    """Exact rank relationships produce their known coefficients."""
    increasing = correlation_metrics(np.arange(5), np.arange(5))
    decreasing = correlation_metrics(np.arange(5), np.arange(4, -1, -1))
    tied = correlation_metrics(
        np.array([0, 1, 1, 0, 1]), np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    )

    assert increasing["spearman_corr_to_y_true"] == pytest.approx(1.0)
    assert decreasing["spearman_corr_to_y_true"] == pytest.approx(-1.0)
    assert tied["spearman_corr_to_y_true"] == pytest.approx(
        0.8660254037844386, abs=1e-15
    )


def test_all_comparison_vectors_have_explicit_results():
    """Each optional vector is compared with the predicted values."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    result = correlation_metrics(
        y_true,
        y_pred,
        input_scores=np.array([-2.0, -1.0, 1.0, 2.0]),
        original_predictions=np.array([0.2, 0.4, 0.6, 0.8]),
    )

    assert set(result) == {
        "spearman_corr_to_y_true",
        "spearman_corr_to_input_scores",
        "spearman_corr_to_original_predictions",
    }
    assert result["spearman_corr_to_input_scores"] == pytest.approx(1.0)
    assert result["spearman_corr_to_original_predictions"] == pytest.approx(1.0)


def test_monotone_miscalibration_is_invisible_to_rank_correlation():
    """A negative control prevents correlation being presented as calibration."""
    probability = np.repeat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 100)
    outcomes = np.concatenate(
        [
            np.r_[np.ones(int(value * 100)), np.zeros(100 - int(value * 100))]
            for value in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
    )
    distorted = probability**2

    calibrated_rank = correlation_metrics(outcomes, probability)
    distorted_rank = correlation_metrics(outcomes, distorted)

    assert distorted_rank == calibrated_rank
    assert brier_score(outcomes, distorted) > brier_score(outcomes, probability)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (np.ones(4), np.arange(4, dtype=float)),
        (np.arange(4, dtype=float), np.ones(4)),
        (np.ones(4), np.ones(4)),
        (np.ones(1), np.ones(1)),
    ],
)
def test_constant_pairs_return_nan_without_a_warning(y_true, y_pred):
    """Undefined correlations are quiet, explicit NaNs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = correlation_metrics(y_true, y_pred)

    assert np.isnan(result["spearman_corr_to_y_true"])
    assert caught == []


@pytest.mark.parametrize(
    ("y_true", "y_pred", "kwargs", "match"),
    [
        ([], [], {}, "must not be empty"),
        ([0, 1], [0.2], {}, "same length"),
        ([[0], [1]], [[0.2], [0.8]], {}, "one-dimensional"),
        ([0, np.nan], [0.2, 0.8], {}, "finite"),
        ([0, 1], [0.2, np.inf], {}, "finite"),
        ([0, 1], ["low", "high"], {}, "numeric"),
        ([0, 1], [0.2, 0.8], {"input_scores": [1.0]}, "same length"),
        (
            [0, 1],
            [0.2, 0.8],
            {"original_predictions": [[0.2], [0.8]]},
            "one-dimensional",
        ),
    ],
)
def test_rejects_invalid_paired_vectors(y_true, y_pred, kwargs, match):
    """Every vector must be numeric, finite, one-dimensional, and aligned."""
    with pytest.raises(ValueError, match=match):
        correlation_metrics(y_true, y_pred, **kwargs)


def test_comparison_vectors_are_keyword_only():
    """Optional vectors cannot be silently swapped by position."""
    call = cast("Any", correlation_metrics)
    with pytest.raises(TypeError):
        call(np.arange(3), np.arange(3), np.arange(3))
