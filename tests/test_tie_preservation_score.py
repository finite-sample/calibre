"""Reference and scenario tests for tie_preservation_score."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from sklearn.metrics import rand_score

from calibre import brier_score, tie_preservation_score


def _partition(values: np.ndarray) -> np.ndarray:
    """Convert exact values to integer partition labels."""
    return np.unique(values, return_inverse=True)[1]


def test_matches_sklearn_rand_index_reference():
    """The public score equals the reference across nontrivial tie partitions."""
    original = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.8, 0.9])
    calibrated = np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.5, 0.8, 0.8])
    expected = rand_score(_partition(original), _partition(calibrated))

    assert tie_preservation_score(original, calibrated) == pytest.approx(expected)


def test_identical_tie_partition_scores_one():
    """Numerical values can change without changing the tie partition."""
    original = np.array([0.1, 0.1, 0.4, 0.8, 0.8])
    calibrated = np.array([0.2, 0.2, 0.5, 0.9, 0.9])

    assert tie_preservation_score(original, calibrated) == 1.0


def test_complete_collapse_of_unique_predictions_scores_zero():
    """Every originally distinct pair changes status after constant collapse."""
    original = np.linspace(0.0, 1.0, 20)
    calibrated = np.full(20, 0.5)

    assert tie_preservation_score(original, calibrated) == 0.0


@pytest.mark.parametrize(
    ("original", "calibrated"),
    [
        ([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 2.0, 3.0]),
        ([0.0, 0.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
    ],
)
def test_one_changed_pair_out_of_six_scores_five_sixths(original, calibrated):
    """Creating and breaking one tie receive the same pairwise penalty."""
    assert tie_preservation_score(
        np.asarray(original), np.asarray(calibrated)
    ) == pytest.approx(5.0 / 6.0)


def test_ties_use_exact_equality_without_a_hidden_tolerance():
    """Nearby unequal floating-point outputs are distinct values."""
    original = np.array([0.0, 0.0, 1.0])
    calibrated = np.array([0.0, 5e-11, 1.0])

    assert tie_preservation_score(original, calibrated) == pytest.approx(2.0 / 3.0)


def test_score_is_invariant_to_common_row_permutation():
    """Pairwise partition agreement does not depend on row order."""
    original = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.4])
    calibrated = np.array([0.2, 0.2, 0.2, 0.5, 0.6, 0.7])
    permutation = np.array([4, 1, 5, 0, 3, 2])

    expected = tie_preservation_score(original, calibrated)
    actual = tie_preservation_score(original[permutation], calibrated[permutation])

    assert actual == expected


def test_same_tie_score_does_not_imply_same_forecast_quality():
    """The score is structural and cannot detect probability miscalibration."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = np.array([0.9, 0.8, 0.2, 0.1])

    assert tie_preservation_score(good, bad) == 1.0
    assert brier_score(y_true, good) == pytest.approx(0.025)
    assert brier_score(y_true, bad) == pytest.approx(0.725)


def test_large_input_uses_partition_counts_instead_of_pair_enumeration():
    """The implementation handles an input that makes a pair loop impractical."""
    size = 100_000
    original = np.arange(size, dtype=float)
    calibrated = original // 2
    changed_pairs = size // 2
    expected = 1.0 - changed_pairs / math.comb(size, 2)

    assert tie_preservation_score(original, calibrated) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("original", "calibrated", "match"),
    [
        ([], [], "must not be empty"),
        ([[0.1], [0.2]], [0.1, 0.2], "one-dimensional"),
        ([0.1, 0.2], [[0.1], [0.2]], "one-dimensional"),
        ([0.1], [0.1, 0.2], "same length"),
        ([0.1, np.nan], [0.1, 0.2], "finite"),
        ([0.1, 0.2], [0.1, np.inf], "finite"),
        (["low", "high"], [0.1, 0.2], "numeric"),
    ],
)
def test_rejects_invalid_or_misaligned_vectors(original, calibrated, match):
    """Malformed vectors fail at the public boundary with a useful message."""
    with pytest.raises(ValueError, match=match):
        tie_preservation_score(np.asarray(original), np.asarray(calibrated))


def test_public_signature_uses_descriptive_names_and_no_tolerance():
    """The API has two required, consistently named prediction vectors."""
    signature = inspect.signature(tie_preservation_score)

    assert list(signature.parameters) == [
        "original_predictions",
        "calibrated_predictions",
    ]
    with pytest.raises(TypeError):
        tie_preservation_score(
            np.array([0.1]),
            np.array([0.1]),
            1e-10,  # type: ignore[call-arg]
        )
