"""
Evaluation metrics for calibration.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.utils import check_array


def mean_calibration_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean calibration error.

    Parameters
    ----------
    y_true
        Ground truth values (0 or 1 for binary classification).
    y_pred
        Predicted probabilities.

    Returns
    -------
    mce : float
        Mean calibration error.

    Raises
    ------
    ValueError
        If arrays have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    >>> mean_calibration_error(y_true, y_pred)
    0.26
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    # Ensure inputs have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred should have the same shape")

    # Simple mean absolute difference between predictions and outcomes
    return float(np.mean(np.abs(y_pred - y_true)))


def binned_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray | None = None, n_bins: int = 10, strategy: str = "uniform", return_details: bool = False
) -> float | dict:
    """
    Calculate binned calibration error.

    Parameters
    ----------
    y_true
        Ground truth values.
    y_pred
        Predicted values.
    x
        Input features for binning. If None, y_pred is used for binning.
    n_bins
        Number of bins.
    strategy
        Strategy for binning:
        - 'uniform': Bins with uniform widths.
        - 'quantile': Bins with approximately equal counts.
    return_details
        If True, return bin details (bin centers, counts, mean predictions, mean truths).

    Returns
    -------
    bce : float or dict
        Binned calibration error. If return_details is True, returns a dictionary
        with BCE and bin details.

    Raises
    ------
    ValueError
        If arrays have different lengths or unknown binning strategy.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    >>> binned_calibration_error(y_true, y_pred, n_bins=2)
    0.05
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    # Check that arrays have matching lengths
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # If x is not provided, use y_pred for binning
    if x is None:
        x = y_pred
    else:
        x = check_array(x, ensure_2d=False)
        # Check that x has matching length
        if len(x) != len(y_true):
            raise ValueError("x must have the same length as y_true and y_pred")

    # Create bins based on strategy
    if strategy == "uniform":
        bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_ids = np.digitize(x, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)  # Ensure valid bin indices

    # Calculate error for each bin
    error = 0
    valid_bins = 0

    bin_centers = []
    bin_counts = []
    bin_pred_means = []
    bin_true_means = []

    for i in range(n_bins):
        bin_mask = bin_ids == i
        if np.any(bin_mask):
            avg_pred = np.mean(y_pred[bin_mask])
            avg_true = np.mean(y_true[bin_mask])
            bin_count = np.sum(bin_mask)

            error += (avg_pred - avg_true) ** 2
            valid_bins += 1

            if return_details:
                bin_center = (bins[i] + bins[i + 1]) / 2
                bin_centers.append(bin_center)
                bin_counts.append(bin_count)
                bin_pred_means.append(avg_pred)
                bin_true_means.append(avg_true)

    # Calculate root mean squared error across bins
    if valid_bins > 0:
        bce = np.sqrt(error / valid_bins)
    else:
        bce = 0.0

    if return_details:
        return {
            "bce": bce,
            "bin_centers": np.array(bin_centers),
            "bin_counts": np.array(bin_counts),
            "bin_pred_means": np.array(bin_pred_means),
            "bin_true_means": np.array(bin_true_means),
        }
    else:
        return float(bce)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    The ECE is a weighted average of the absolute calibration error across bins,
    where each bin's weight is proportional to the number of samples in the bin.

    Parameters
    ----------
    y_true
        Ground truth values (0 or 1 for binary classification).
    y_pred
        Predicted probabilities.
    n_bins
        Number of bins for discretizing predictions.

    Returns
    -------
    ece : float
        Expected Calibration Error.

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    >>> expected_calibration_error(y_true, y_pred, n_bins=2)
    0.12
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Create bins and assign each prediction to a bin
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_samples = len(y_true)
    ece = 0.0

    for bin_idx in range(n_bins):
        # Get indices of samples in this bin
        mask = bin_indices == bin_idx
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_confidence = np.mean(y_pred[mask])
            bin_accuracy = np.mean(y_true[mask])

            # Weighted absolute calibration error
            ece += (bin_count / n_samples) * np.abs(bin_confidence - bin_accuracy)

    return ece


def maximum_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    The MCE is the maximum absolute difference between the average predicted
    probability and the fraction of positive samples in any bin.

    Parameters
    ----------
    y_true
        Ground truth values (0 or 1 for binary classification).
    y_pred
        Predicted probabilities.
    n_bins
        Number of bins for discretizing predictions.

    Returns
    -------
    mce : float
        Maximum Calibration Error.

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    >>> maximum_calibration_error(y_true, y_pred, n_bins=2)
    0.2
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Create bins and assign each prediction to a bin
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    max_error = 0.0

    for bin_idx in range(n_bins):
        # Get indices of samples in this bin
        mask = bin_indices == bin_idx
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_confidence = np.mean(y_pred[mask])
            bin_accuracy = np.mean(y_true[mask])

            # Update maximum calibration error
            max_error = max(max_error, np.abs(bin_confidence - bin_accuracy))

    return max_error


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Brier score.

    The Brier score is a proper scoring rule that measures the mean squared
    difference between predicted probabilities and the actual outcomes.

    Parameters
    ----------
    y_true
        Ground truth values (0 or 1 for binary classification).
    y_pred
        Predicted probabilities.

    Returns
    -------
    score : float
        Brier score (lower is better).

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    >>> brier_score(y_true, y_pred)
    0.142
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return float(brier_score_loss(y_true, y_pred))


def correlation_metrics(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray | None = None, y_orig: np.ndarray | None = None) -> dict:
    """
    Calculate correlation metrics between various signals.

    Parameters
    ----------
    y_true
        Ground truth values.
    y_pred
        Predicted/calibrated values.
    x
        Input features.
    y_orig
        Original uncalibrated predictions.

    Returns
    -------
    correlations : dict
        Dictionary of correlation metrics.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
    >>> y_orig = np.array([0.1, 0.6, 0.9, 0.3, 0.5])
    >>> correlation_metrics(y_true, y_pred, y_orig=y_orig)
    {'spearman_corr_to_y_true': 0.6708203932499371, 'spearman_corr_to_y_orig': 0.9}
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    results = {"spearman_corr_to_y_true": spearmanr(y_true, y_pred).correlation}

    if y_orig is not None:
        y_orig = check_array(y_orig, ensure_2d=False)
        results["spearman_corr_to_y_orig"] = spearmanr(y_orig, y_pred).correlation
        results["spearman_corr_orig_to_calib"] = spearmanr(
            y_orig, y_pred
        ).correlation  # Alias for backward compatibility

    if x is not None:
        x = check_array(x, ensure_2d=False)
        results["spearman_corr_to_x"] = spearmanr(x, y_pred).correlation

    return results


def unique_value_counts(y_pred: np.ndarray, y_orig: np.ndarray | None = None, precision: int = 6) -> dict:
    """
    Count unique values in predictions.

    Parameters
    ----------
    y_pred
        Predicted/calibrated values.
    y_orig
        Original uncalibrated predictions.
    precision
        Decimal precision for rounding.

    Returns
    -------
    counts : dict
        Dictionary with counts of unique values.

    Examples
    --------
    >>> import numpy as np
    >>> y_pred = np.array([0.2, 0.7, 0.8, 0.2, 0.7])
    >>> y_orig = np.array([0.1, 0.6, 0.9, 0.2, 0.5])
    >>> unique_value_counts(y_pred, y_orig)
    {'n_unique_y_pred': 3, 'n_unique_y_orig': 5, 'unique_value_ratio': 0.6}
    """
    y_pred = check_array(y_pred, ensure_2d=False)

    results: dict[str, int | float] = {"n_unique_y_pred": len(np.unique(np.round(y_pred, precision)))}

    if y_orig is not None:
        y_orig = check_array(y_orig, ensure_2d=False)
        results["n_unique_y_orig"] = len(np.unique(np.round(y_orig, precision)))
        results["unique_value_ratio"] = float(results["n_unique_y_pred"]) / max(
            1, int(results["n_unique_y_orig"])
        )

    return results


def calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the calibration curve for binary classification.

    Parameters
    ----------
    y_true
        Ground truth values (0 or 1 for binary classification).
    y_pred
        Predicted probabilities.
    n_bins
        Number of bins for discretizing predictions.
    strategy
        Strategy for binning:
        - 'uniform': Bins with uniform widths.
        - 'quantile': Bins with approximately equal counts.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,)
        The true fraction of positive samples in each bin.
    prob_pred : ndarray of shape (n_bins,)
        The mean predicted probability in each bin.
    counts : ndarray of shape (n_bins,)
        The number of samples in each bin.

    Raises
    ------
    ValueError
        If arrays have different lengths or unknown binning strategy.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.9, 0.1])
    >>> prob_true, prob_pred, counts = calibration_curve(y_true, y_pred, n_bins=5)
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Create bins based on strategy
    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    # Assign predictions to bins
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_pred_sums = np.bincount(bin_indices, weights=y_pred, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Avoid division by zero
    nonzero = bin_counts > 0
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)

    prob_true[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    prob_pred[nonzero] = bin_pred_sums[nonzero] / bin_counts[nonzero]

    return prob_true, prob_pred, bin_counts


def tie_preservation_score(
    y_original: np.ndarray, y_calibrated: np.ndarray, tolerance: float = 1e-10
) -> float:
    """
    Measure how well calibration preserves genuine ties while removing spurious ones.

    Parameters
    ----------
    y_original
        Original predicted probabilities before calibration.
    y_calibrated
        Calibrated probabilities.
    tolerance
        Tolerance for considering values as tied.

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Returns
    -------
    score : float
        Tie preservation score between 0 and 1.
        Higher values indicate better preservation of meaningful ties.

    Examples
    --------
    >>> import numpy as np
    >>> y_orig = np.array([0.1, 0.15, 0.2, 0.6, 0.65, 0.7])
    >>> y_cal = np.array([0.1, 0.15, 0.2, 0.65, 0.65, 0.65])
    >>> score = tie_preservation_score(y_orig, y_cal)
    >>> 0 <= score <= 1
    True
    """
    y_original = check_array(y_original, ensure_2d=False)
    y_calibrated = check_array(y_calibrated, ensure_2d=False)

    if len(y_original) != len(y_calibrated):
        raise ValueError("Arrays must have the same length")

    n = len(y_original)
    if n < 2:
        return 1.0

    # Count tied pairs in original and calibrated data
    tied_orig = 0
    tied_cal = 0
    preserved_ties = 0

    for i in range(n):
        for j in range(i + 1, n):
            orig_tied = abs(y_original[i] - y_original[j]) <= tolerance
            cal_tied = abs(y_calibrated[i] - y_calibrated[j]) <= tolerance

            if orig_tied:
                tied_orig += 1
                if cal_tied:
                    preserved_ties += 1

            if cal_tied:
                tied_cal += 1

    # Score based on preservation of original ties and avoidance of spurious ties
    if tied_orig == 0:
        # No original ties to preserve
        preservation_rate = 1.0
    else:
        preservation_rate = preserved_ties / tied_orig

    # Penalty for creating too many new ties
    if tied_cal == 0:
        spurious_penalty = 0.0
    else:
        spurious_ties = tied_cal - preserved_ties
        spurious_penalty = spurious_ties / (n * (n - 1) / 2)  # Normalize by total pairs

    score = preservation_rate - spurious_penalty
    return max(0.0, min(1.0, score))


def plateau_quality_score(
    X: np.ndarray, y: np.ndarray, y_calibrated: np.ndarray
) -> float:
    """
    Overall quality score for plateaus in calibrated predictions.

    Parameters
    ----------
    X
        Input features.
    y
        True target values.
    y_calibrated
        Calibrated predictions.

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Returns
    -------
    score : float
        Quality score between 0 and 1.
        Higher values indicate better plateau quality.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> y = np.array([0, 0, 1, 1, 1])
    >>> y_cal = np.array([0.1, 0.25, 0.25, 0.4, 0.6])
    >>> score = plateau_quality_score(X, y, y_cal)
    >>> 0 <= score <= 1
    True
    """
    from .diagnostics import detect_plateaus

    X = check_array(X, ensure_2d=False)
    y = check_array(y, ensure_2d=False)
    y_calibrated = check_array(y_calibrated, ensure_2d=False)

    if not (len(X) == len(y) == len(y_calibrated)):
        raise ValueError("All arrays must have the same length")

    # Sort by X
    sort_idx = np.argsort(X)
    y_sorted = y[sort_idx]
    y_cal_sorted = y_calibrated[sort_idx]

    # Extract plateaus
    plateaus = detect_plateaus(y_cal_sorted)

    if not plateaus:
        return 1.0  # No plateaus is good

    scores = []

    for start_idx, end_idx, _value in plateaus:
        # Check if plateau represents genuine flatness
        plateau_y = y_sorted[start_idx : end_idx + 1]
        plateau_var = np.var(plateau_y)

        # Good plateaus have low variance in true outcomes
        # and appropriate size (not too small or too large)
        size = end_idx - start_idx + 1
        size_penalty = abs(size - len(X) * 0.1) / (
            len(X) * 0.1
        )  # Penalize very large plateaus

        plateau_score = np.exp(-plateau_var - size_penalty)
        scores.append(plateau_score)

    return np.mean(scores) if scores else 1.0


def calibration_diversity_index(
    y_calibrated: np.ndarray, reference_diversity: float | None = None
) -> float:
    """
    Measure granularity preservation in calibrated predictions.

    Parameters
    ----------
    y_calibrated
        Calibrated predictions.
    reference_diversity
        Reference diversity to compare against (e.g., diversity of original predictions).
        If None, returns absolute diversity.

    Returns
    -------
    diversity : float
        Diversity index. Higher values indicate more granular predictions.
        If reference_diversity is provided, returns relative diversity.

    Examples
    --------
    >>> import numpy as np
    >>> y_cal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> diversity = calibration_diversity_index(y_cal)
    >>> diversity > 0
    True
    """
    y_calibrated = check_array(y_calibrated, ensure_2d=False)

    # Number of unique values normalized by total samples
    n_unique = len(np.unique(y_calibrated))
    n_total = len(y_calibrated)

    diversity = n_unique / n_total

    if reference_diversity is not None:
        if reference_diversity == 0:
            return np.inf if diversity > 0 else 1.0
        return diversity / reference_diversity

    return diversity


def progressive_sampling_diversity(
    X: np.ndarray,
    y: np.ndarray,
    sample_sizes: list[int] | None = None,
    n_trials: int = 10,
    random_state: int | None = None,
) -> tuple[list[int], list[float]]:
    """
    Compute diversity vs sample size curve for progressive sampling analysis.

    Parameters
    ----------
    X
        Input features.
    y
        Target values.
    sample_sizes
        Sample sizes to test. If None, uses default range.
    n_trials
        Number of trials per sample size.
    random_state
        Random state for reproducibility.

    Raises
    ------
    ValueError
        If X and y have different lengths.

    Returns
    -------
    sizes : list of int
        Sample sizes tested.
    diversities : list of float
        Average diversity at each sample size.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.linspace(0, 1, 100)
    >>> y = np.random.binomial(1, X, 100)
    >>> sizes, divs = progressive_sampling_diversity(X, y, sample_sizes=[20, 50, 80])
    >>> len(sizes) == len(divs) == 3
    True
    """
    X = check_array(X, ensure_2d=False)
    y = check_array(y, ensure_2d=False)

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n_total = len(X)

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

        for _trial in range(n_trials):
            # Random subsample
            indices = rng.choice(n_total, size=size, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]

            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso_reg.fit(X_sub, y_sub)
            y_cal = iso_reg.transform(X_sub)

            # Compute diversity
            diversity = calibration_diversity_index(y_cal)
            trial_diversities.append(diversity)

        diversities.append(float(np.mean(trial_diversities)))

    return sample_sizes, diversities


__all__ = [
    "mean_calibration_error",
    "binned_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "correlation_metrics",
    "unique_value_counts",
    "calibration_curve",
    "tie_preservation_score",
    "plateau_quality_score",
    "calibration_diversity_index",
    "progressive_sampling_diversity",
]
