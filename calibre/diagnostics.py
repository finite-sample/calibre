"""
Diagnostic analysis tools for calibration.

This module provides diagnostic analysis to help understand calibration behavior,
particularly detecting plateaus (flat regions) and identifying potential data quality issues.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_plateau_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    y_calibrated: np.ndarray,
    n_bootstraps: int = 100,  # Kept for API compatibility but unused
    random_state: int | None = None,  # Kept for API compatibility but unused
) -> dict:
    """
    Detect and analyze plateaus (flat regions) in calibration curves.

    This function identifies flat regions where the calibrator outputs the same
    value for multiple inputs, and flags potentially problematic plateaus based
    on simple, interpretable criteria like sample count.

    Parameters
    ----------
    X
        Original predicted probabilities.
    y
        True labels.
    y_calibrated
        Calibrated probabilities.
    n_bootstraps
        Kept for API compatibility, currently unused.
    random_state
        Kept for API compatibility, currently unused.

    Returns
    -------
    diagnostics : dict
        Dictionary containing:
        - 'n_plateaus': Number of plateaus detected
        - 'plateaus': List of plateau information dicts, each containing:
            - 'plateau_id': Unique identifier (0-indexed)
            - 'x_range': Tuple of (min, max) input values in plateau
            - 'value': The constant output value of the plateau
            - 'width': Number of samples in the plateau
            - 'n_samples': Number of samples (same as width)
            - 'sample_density': 'adequate', 'sparse', or 'very_sparse'
        - 'warnings': List of warning messages about problematic plateaus

    Examples
    --------
    >>> X = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> y_cal = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
    >>> diagnostics = run_plateau_diagnostics(X, y, y_cal)
    >>> print(diagnostics['n_plateaus'])
    2
    >>> print(diagnostics['warnings'])
    []
    """
    # Sort by calibrated values to find consecutive identical values
    sorted_indices = np.argsort(y_calibrated)
    y_cal_sorted = y_calibrated[sorted_indices]
    X_sorted = X[sorted_indices]

    # Detect plateaus
    plateau_indices = detect_plateaus(y_cal_sorted)

    # Analyze each plateau
    plateaus = []
    warnings = []

    for i, (start_idx, end_idx, value) in enumerate(plateau_indices):
        plateau_info = analyze_plateau_simple(
            X_sorted, y_cal_sorted, start_idx, end_idx, value, i
        )
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
    diagnostics = {
        "n_plateaus": len(plateaus),
        "plateaus": plateaus,
        "warnings": warnings,
    }

    return diagnostics


def detect_plateaus(
    y_calibrated: np.ndarray, min_width: int = 2
) -> list[tuple[int, int, float]]:
    """
    Detect plateaus (consecutive identical values) in calibrated predictions.

    Parameters
    ----------
    y_calibrated
        Sorted calibrated probabilities.
    min_width
        Minimum number of consecutive identical values to count as a plateau.

    Returns
    -------
    plateaus : list of tuples
        List of (start_index, end_index, value) tuples for each detected plateau.
        Indices are inclusive.

    Examples
    --------
    >>> y_cal = np.array([0.2, 0.2, 0.2, 0.5, 0.8, 0.8])
    >>> plateaus = detect_plateaus(y_cal)
    >>> print(plateaus)
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
    y_calibrated: np.ndarray,
    start_idx: int,
    end_idx: int,
    value: float,
    plateau_id: int,
) -> dict:
    """
    Analyze a single plateau with simple, interpretable metrics.

    Parameters
    ----------
    X
        Sorted input predictions.
    y_calibrated
        Sorted calibrated predictions.
    start_idx
        Start index of plateau (inclusive).
    end_idx
        End index of plateau (inclusive).
    value
        The constant value of the plateau.
    plateau_id
        Unique identifier for this plateau.

    Returns
    -------
    plateau_info : dict
        Dictionary with plateau information:
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

    # Assess sample density (simple thresholds)
    if width < 5:
        sample_density = "very_sparse"
    elif width < 10:
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
    """
    Measure how calibration diversity changes with training sample size.

    This diagnostic tool helps determine whether you have sufficient training
    data for stable calibration. If diversity continues increasing with sample
    size, more data would likely improve calibration granularity.

    Parameters
    ----------
    X
        Input features (predicted probabilities).
    y
        True binary labels.
    calibrator
        Calibrator to test. If None, uses IsotonicCalibrator.
    sample_sizes
        Sample sizes to test. If None, uses default range covering
        10% to 100% of available data.
    n_trials
        Number of random trials per sample size for averaging.
    random_state
        Random state for reproducibility.

    Returns
    -------
    sizes : list of int
        Sample sizes tested.
    diversities : list of float
        Average diversity at each sample size, where diversity is
        the fraction of unique calibrated values.

    Raises
    ------
    ValueError
        If X and y have different lengths.

    Notes
    -----
    This function is computationally expensive as it fits the calibrator
    multiple times (n_trials × len(sample_sizes) fits). Use for diagnostic
    analysis, not routine evaluation.

    The diversity metric measures granularity: higher diversity means more
    unique calibrated values, indicating better discrimination. If diversity
    plateaus, you have sufficient data. If it keeps increasing, more data
    would help.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import IsotonicCalibrator
    >>>
    >>> # Generate example data
    >>> np.random.seed(42)
    >>> X = np.random.uniform(0, 1, 500)
    >>> y = (X > 0.5).astype(int)
    >>>
    >>> # Test data sufficiency
    >>> sizes, divs = diversity_learning_curve(X, y,
    ...     sample_sizes=[50, 100, 200, 300, 400, 500])
    >>>
    >>> # Check if converged
    >>> if divs[-1] - divs[-2] < 0.05:
    ...     print("✓ Diversity has converged - sufficient data")
    ... else:
    ...     print("⚠ Diversity still increasing - more data may help")
    >>>
    >>> # Compare methods
    >>> from calibre import SplineCalibrator
    >>> sizes, iso_divs = diversity_learning_curve(X, y,
    ...     calibrator=IsotonicCalibrator())
    >>> sizes, spl_divs = diversity_learning_curve(X, y,
    ...     calibrator=SplineCalibrator())
    >>> # Compare which method preserves more diversity at each sample size

    See Also
    --------
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
        sample_sizes = [
            max(10, n_total // 10),
            max(20, n_total // 5),
            max(30, n_total // 3),
            max(50, n_total // 2),
            min(n_total - 10, int(n_total * 0.8)),
            n_total,
        ]
        sample_sizes = [s for s in sample_sizes if s <= n_total]

    rng = np.random.RandomState(random_state)
    diversities = []

    for size in sample_sizes:
        trial_diversities = []

        for trial in range(n_trials):
            # Random subsample
            indices = rng.choice(n_total, size=size, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]

            # Fit calibrator
            try:
                # Create fresh instance for each trial
                cal = calibrator.__class__(**calibrator.get_params())
                cal.fit(X_sub, y_sub)
                y_cal = cal.transform(X_sub)

                # Compute diversity
                n_unique = len(np.unique(y_cal))
                diversity = n_unique / len(y_cal)
                trial_diversities.append(diversity)
            except Exception as e:
                logger.warning(
                    f"Failed to fit calibrator at size {size}, trial {trial}: {e}"
                )
                continue

        if trial_diversities:
            diversities.append(float(np.mean(trial_diversities)))
        else:
            diversities.append(0.0)

    return sample_sizes, diversities
