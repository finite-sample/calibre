"""Diagnostic analysis tools for calibration.

This module provides diagnostic analysis to help understand calibration behavior,
particularly detecting plateaus (flat regions) and identifying potential data
quality issues.
"""

from __future__ import annotations

import numpy as np

# Plateau widths, in number of tied samples, at which the reported
# `support` label changes. Conventional cut points for a human-readable
# summary, not thresholds anything is inferred from.
SPARSE_PLATEAU_WIDTH = 5
MODERATE_PLATEAU_WIDTH = 10


def run_plateau_diagnostics(
    input_scores: np.ndarray,
    calibrated_predictions: np.ndarray,
) -> dict:
    """Detect and analyze plateaus (flat regions) in calibration curves.

    This function identifies flat regions where the calibrator outputs the same
    value for multiple inputs, and flags potentially problematic plateaus based
    on simple, interpretable criteria like sample count.

    The diagnosis is purely structural: it counts how many samples support each
    flat region. It does not take the true labels, because it makes no claim
    about whether a plateau is *justified* by the outcomes -- only about whether
    enough data sits underneath it to say anything at all.

    Args:
        input_scores: Original predicted probabilities.
        calibrated_predictions: Calibrated probabilities.

    Returns:
        diagnostics: Dictionary containing:

            - ``'n_plateaus'``: Number of plateaus detected.
            - ``'plateaus'``: List of plateau information dicts, each
              containing:

            - ``'plateau_id'``: Unique identifier (0-indexed).
            - ``'input_score_range'``: Minimum and maximum input score.
            - ``'calibrated_value'``: Constant output value of the plateau.
            - ``'n_observations'``: Number of observations in the plateau.
            - ``'support'``: ``'adequate'``, ``'sparse'`` or
              ``'very_sparse'``.

            - ``'warnings'``: List of warning messages about problematic plateaus.

    Raises:
        ValueError: If the two arrays have different lengths.

    Examples:
        >>> input_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        >>> calibrated = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
        >>> diagnostics = run_plateau_diagnostics(input_scores, calibrated)
        >>> print(diagnostics['n_plateaus'])
        2
        >>> for warning in diagnostics['warnings']:
        ...     print(warning)
        Plateau 1 at [0.100, 0.300] has only 3 observations - may be unreliable
        Plateau 2 at [0.700, 0.900] has only 3 observations - may be unreliable
    """
    input_scores = np.asarray(input_scores, dtype=float).ravel()
    calibrated_predictions = np.asarray(calibrated_predictions, dtype=float).ravel()
    if input_scores.size != calibrated_predictions.size:
        raise ValueError(
            "input_scores and calibrated_predictions must have the same length; "
            f"got {input_scores.size} and {calibrated_predictions.size}"
        )

    # A plateau is a flat region along the score axis. Sorting by the output
    # instead would bring equal but separated values together and invent a flat
    # region spanning the different value between them.
    sorted_indices = np.argsort(input_scores, kind="mergesort")
    calibrated_sorted = calibrated_predictions[sorted_indices]
    input_scores_sorted = input_scores[sorted_indices]

    # Detect plateaus
    plateau_indices = detect_plateaus(calibrated_sorted)

    # Analyze each plateau
    plateaus = []
    warnings = []

    for i, (start_idx, end_idx, value) in enumerate(plateau_indices):
        plateau_info = analyze_plateau_simple(
            input_scores_sorted, start_idx, end_idx, value, i
        )
        plateaus.append(plateau_info)

        # Generate warnings for problematic plateaus
        if plateau_info["support"] == "very_sparse":
            warnings.append(
                f"Plateau {i + 1} at "
                f"[{plateau_info['input_score_range'][0]:.3f}, "
                f"{plateau_info['input_score_range'][1]:.3f}] has only "
                f"{plateau_info['n_observations']} observations - may be unreliable"
            )
        elif plateau_info["support"] == "sparse":
            warnings.append(
                f"Plateau {i + 1} at "
                f"[{plateau_info['input_score_range'][0]:.3f}, "
                f"{plateau_info['input_score_range'][1]:.3f}] has "
                f"{plateau_info['n_observations']} observations - consider "
                "collecting more data in this range"
            )

    # Summary
    return {
        "n_plateaus": len(plateaus),
        "plateaus": plateaus,
        "warnings": warnings,
    }


def detect_plateaus(
    calibrated_predictions: np.ndarray, *, min_width: int = 2
) -> list[tuple[int, int, float]]:
    """Detect plateaus (consecutive identical values) in calibrated predictions.

    Args:
        calibrated_predictions: Calibrated probabilities ordered by input score.
        min_width: Minimum number of consecutive identical values to count as
            a plateau.

    Returns:
        plateaus: List of (start_index, end_index, value) tuples for each
            detected plateau. Indices are inclusive.

    Examples:
        >>> y_cal = np.array([0.2, 0.2, 0.2, 0.5, 0.8, 0.8])
        >>> plateaus = detect_plateaus(y_cal)
        >>> [(lo, hi, float(v)) for lo, hi, v in plateaus]
        [(0, 2, 0.2), (4, 5, 0.8)]
    """
    if len(calibrated_predictions) == 0:
        return []

    plateaus = []
    start_idx = 0
    current_value = calibrated_predictions[0]

    for i in range(1, len(calibrated_predictions)):
        if not np.isclose(calibrated_predictions[i], current_value):
            # End of current plateau
            width = i - start_idx
            if width >= min_width:
                plateaus.append((start_idx, i - 1, current_value))

            # Start new potential plateau
            start_idx = i
            current_value = calibrated_predictions[i]

    # Check final plateau
    width = len(calibrated_predictions) - start_idx
    if width >= min_width:
        plateaus.append((start_idx, len(calibrated_predictions) - 1, current_value))

    return plateaus


def analyze_plateau_simple(
    input_scores: np.ndarray,
    start_idx: int,
    end_idx: int,
    value: float,
    plateau_id: int,
) -> dict:
    """Analyze a single plateau with simple, interpretable metrics.

    Args:
        input_scores: Sorted input predictions.
        start_idx: Start index of plateau (inclusive).
        end_idx: End index of plateau (inclusive).
        value: The constant value of the plateau.
        plateau_id: Unique identifier for this plateau.

    Returns:
        plateau_info: Dictionary with plateau information:

            - plateau_id
            - input_score_range: (minimum, maximum) input value
            - calibrated_value: output value
            - n_observations: number of observations
            - support: 'adequate', 'sparse', or 'very_sparse'
    """
    # Extract plateau region
    plateau_scores = input_scores[start_idx : end_idx + 1]

    # Basic statistics
    width = end_idx - start_idx + 1
    score_minimum = float(np.min(plateau_scores))
    score_maximum = float(np.max(plateau_scores))

    # The labels are a convenience for reading a diagnostic, not a statistical
    # statement, and the cut points are conventional rather than derived -- which
    # is exactly why they are named here instead of sitting bare in the branch.
    if width < SPARSE_PLATEAU_WIDTH:
        support = "very_sparse"
    elif width < MODERATE_PLATEAU_WIDTH:
        support = "sparse"
    else:
        support = "adequate"

    return {
        "plateau_id": plateau_id,
        "input_score_range": (score_minimum, score_maximum),
        "calibrated_value": float(value),
        "n_observations": width,
        "support": support,
    }
