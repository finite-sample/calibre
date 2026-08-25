"""Diagnostic analysis tools for calibration.

This module provides diagnostic analysis to help understand calibration behavior,
particularly detecting plateaus (flat regions) and identifying potential data
quality issues.
"""

from __future__ import annotations

from typing import Any

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


def diversity_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    calibrator: Any = None,
    sample_sizes: list[int] | None = None,
    n_trials: int = 10,
    random_state: int | None = None,
) -> tuple[list[int], list[float]]:
    """Measure how calibration diversity changes with training sample size.

    This diagnostic tool helps determine whether you have sufficient training
    data for stable calibration. If diversity continues increasing with sample
    size, more data would likely improve calibration granularity.

    Args:
        X: Input features (predicted probabilities).
        y: True binary labels.
        calibrator: Calibrator to test. If None, uses IsotonicCalibrator.
        sample_sizes: Sample sizes to test. If None, uses default range
            covering 10% to 100% of available data.
        n_trials: Number of random trials per sample size for averaging.
        random_state: Random state for reproducibility.

    Returns:
        sizes: Sample sizes tested.
        diversities: Mean fraction of unique calibrated values at each size.

    Raises:
        ValueError: If X and y have different lengths, a sample size lies
            outside the data, or ``n_trials`` is below 1.

    Notes:
        This function is computationally expensive as it fits the calibrator
        multiple times (n_trials x len(sample_sizes) fits). Use for diagnostic
        analysis, not routine evaluation.

        The diversity metric measures granularity: higher diversity means more
        unique calibrated values, indicating better discrimination. If diversity
        plateaus, you have sufficient data. If it keeps increasing, more data
        would help.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(0, 1, 200)
        >>> y = (X > 0.5).astype(int)
        >>>
        >>> sizes, divs = diversity_learning_curve(
        ...     X, y, sample_sizes=[50, 100, 200], n_trials=2, random_state=0
        ... )
        >>> sizes
        [50, 100, 200]
        >>> len(divs) == 3 and all(0.0 <= d <= 1.0 for d in divs)
        True

        Rising diversity suggests more data would buy more granularity; a flat tail
        suggests the calibrator has the resolution the data can support.

    See Also:
        unique_value_counts : Count unique values in calibrated predictions
        run_plateau_diagnostics : Detect and analyze plateaus
    """
    from sklearn.utils.validation import check_array

    X = check_array(X, ensure_2d=False)
    y = check_array(y, ensure_2d=False)

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n_total = len(X)

    # Default calibrator
    if calibrator is None:
        from .calibrators.isotonic import IsotonicCalibrator

        calibrator = IsotonicCalibrator()

    # Default sample sizes
    if sample_sizes is None:
        sample_sizes = sorted(
            {
                max(1, min(n_total, int(n_total * fraction)))
                for fraction in (0.1, 0.2, 1 / 3, 0.5, 0.8, 1.0)
            }
        )
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if any(size < 1 or size > n_total for size in sample_sizes):
        raise ValueError(
            f"sample_sizes must lie between 1 and {n_total}, got {sample_sizes}"
        )

    rng = np.random.RandomState(random_state)
    diversities = []

    for size in sample_sizes:
        trial_diversities = []

        for _trial in range(n_trials):
            # Random subsample
            indices = rng.choice(n_total, size=size, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]

            # Fit calibrator
            # Create fresh instance for each trial
            cal = calibrator.__class__(**calibrator.get_params())
            cal.fit(X_sub, y_sub)
            y_cal = cal.transform(X_sub)

            # Compute diversity
            n_unique = len(np.unique(y_cal))
            diversity = n_unique / len(y_cal)
            trial_diversities.append(diversity)

        diversities.append(float(np.mean(trial_diversities)))

    return sample_sizes, diversities
