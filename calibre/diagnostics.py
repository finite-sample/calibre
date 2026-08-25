"""Diagnostic analysis tools for calibration.

This module provides diagnostic analysis to help understand calibration behavior,
particularly detecting plateaus (flat regions) and identifying potential data
quality issues.
"""

from __future__ import annotations

import numpy as np

# Plateau widths, in number of tied samples, at which the reported
# `sample_density` label changes. Conventional cut points for a human-readable
# summary, not thresholds anything is inferred from.
SPARSE_PLATEAU_WIDTH = 5
MODERATE_PLATEAU_WIDTH = 10


def run_plateau_diagnostics(
    X: np.ndarray,
    y_calibrated: np.ndarray,
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
        X: Original predicted probabilities.
        y_calibrated: Calibrated probabilities.

    Returns:
        diagnostics: Dictionary containing:

            - ``'n_plateaus'``: Number of plateaus detected.
            - ``'plateaus'``: List of plateau information dicts, each
              containing:

            - ``'plateau_id'``: Unique identifier (0-indexed).
            - ``'x_range'``: Tuple of (min, max) input values in the plateau.
            - ``'value'``: The constant output value of the plateau.
            - ``'width'``: Number of samples in the plateau.
            - ``'n_samples'``: Number of samples (same as width).
            - ``'sample_density'``: ``'adequate'``, ``'sparse'`` or
              ``'very_sparse'``.

            - ``'warnings'``: List of warning messages about problematic plateaus.

    Raises:
        ValueError: If ``X`` and ``y_calibrated`` have different lengths.

    Examples:
        >>> X = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        >>> y_cal = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
        >>> diagnostics = run_plateau_diagnostics(X, y_cal)
        >>> print(diagnostics['n_plateaus'])
        2
        >>> for warning in diagnostics['warnings']:
        ...     print(warning)
        Plateau 1 at [0.100, 0.300] has only 3 samples - may be unreliable
        Plateau 2 at [0.700, 0.900] has only 3 samples - may be unreliable
    """
    X = np.asarray(X, dtype=float).ravel()
    y_calibrated = np.asarray(y_calibrated, dtype=float).ravel()
    if X.size != y_calibrated.size:
        raise ValueError(
            "X and y_calibrated must have the same length; "
            f"got {X.size} and {y_calibrated.size}"
        )

    # A plateau is a flat region along the score axis. Sorting by the output
    # instead would bring equal but separated values together and invent a flat
    # region spanning the different value between them.
    sorted_indices = np.argsort(X, kind="mergesort")
    y_cal_sorted = y_calibrated[sorted_indices]
    X_sorted = X[sorted_indices]

    # Detect plateaus
    plateau_indices = detect_plateaus(y_cal_sorted)

    # Analyze each plateau
    plateaus = []
    warnings = []

    for i, (start_idx, end_idx, value) in enumerate(plateau_indices):
        plateau_info = analyze_plateau_simple(X_sorted, start_idx, end_idx, value, i)
        plateaus.append(plateau_info)

        # Generate warnings for problematic plateaus
        if plateau_info["sample_density"] == "very_sparse":
            warnings.append(
                f"Plateau {i + 1} at [{plateau_info['x_range'][0]:.3f}, "
                f"{plateau_info['x_range'][1]:.3f}] has only "
                f"{plateau_info['n_samples']} samples - may be unreliable"
            )
        elif plateau_info["sample_density"] == "sparse":
            warnings.append(
                f"Plateau {i + 1} at [{plateau_info['x_range'][0]:.3f}, "
                f"{plateau_info['x_range'][1]:.3f}] has {plateau_info['n_samples']} "
                f"samples - consider collecting more data in this range"
            )

    # Summary
    return {
        "n_plateaus": len(plateaus),
        "plateaus": plateaus,
        "warnings": warnings,
    }


def detect_plateaus(
    y_calibrated: np.ndarray, min_width: int = 2
) -> list[tuple[int, int, float]]:
    """Detect plateaus (consecutive identical values) in calibrated predictions.

    Args:
        y_calibrated: Calibrated probabilities ordered by their input score.
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
    if len(y_calibrated) == 0:
        return []

    plateaus = []
    start_idx = 0
    current_value = y_calibrated[0]

    for i in range(1, len(y_calibrated)):
        if not np.isclose(y_calibrated[i], current_value):
            # End of current plateau
            width = i - start_idx
            if width >= min_width:
                plateaus.append((start_idx, i - 1, current_value))

            # Start new potential plateau
            start_idx = i
            current_value = y_calibrated[i]

    # Check final plateau
    width = len(y_calibrated) - start_idx
    if width >= min_width:
        plateaus.append((start_idx, len(y_calibrated) - 1, current_value))

    return plateaus


def analyze_plateau_simple(
    X: np.ndarray,
    start_idx: int,
    end_idx: int,
    value: float,
    plateau_id: int,
) -> dict:
    """Analyze a single plateau with simple, interpretable metrics.

    Args:
        X: Sorted input predictions.
        start_idx: Start index of plateau (inclusive).
        end_idx: End index of plateau (inclusive).
        value: The constant value of the plateau.
        plateau_id: Unique identifier for this plateau.

    Returns:
        plateau_info: Dictionary with plateau information:

            - plateau_id
            - x_range: (min, max) of input values
            - value: output value
            - width: number of samples
            - n_samples: same as width
            - sample_density: 'adequate', 'sparse', or 'very_sparse'
    """
    # Extract plateau region
    X_plateau = X[start_idx : end_idx + 1]

    # Basic statistics
    width = end_idx - start_idx + 1
    x_min = float(np.min(X_plateau))
    x_max = float(np.max(X_plateau))

    # The labels are a convenience for reading a diagnostic, not a statistical
    # statement, and the cut points are conventional rather than derived -- which
    # is exactly why they are named here instead of sitting bare in the branch.
    if width < SPARSE_PLATEAU_WIDTH:
        sample_density = "very_sparse"
    elif width < MODERATE_PLATEAU_WIDTH:
        sample_density = "sparse"
    else:
        sample_density = "adequate"

    return {
        "plateau_id": plateau_id,
        "x_range": (x_min, x_max),
        "value": float(value),
        "width": width,
        "n_samples": width,
        "sample_density": sample_density,
    }
