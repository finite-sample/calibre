"""Evaluation metrics for calibration."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.utils import check_array


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two arrays, as a plain float.

    Wrapped because ``scipy.stats.spearmanr`` returns a result object whose
    ``correlation`` attribute type checkers cannot see, and whose tuple form is
    typed too loosely to convert directly.

    Args:
        a: First array.
        b: Second array.

    Returns:
        float: The correlation coefficient, or NaN if either input is constant.
    """
    result = spearmanr(a, b)
    coefficient = getattr(result, "statistic", None)
    if coefficient is None:  # pragma: no cover - older scipy
        coefficient = getattr(result, "correlation", None)
    if coefficient is None:  # pragma: no cover - unexpected scipy shape
        coefficient = next(iter(result))  # type: ignore[call-overload]
    return float(coefficient)  # type: ignore[arg-type]


def mean_calibration_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Calculate the mean calibration error: the bias of the predictions.

    .. math:: \left| \mathbb{E}[\hat{p}] - \mathbb{E}[y] \right|

    This is calibration *in the large* -- whether the predictions are right on
    average. It is zero for any predictor whose mean matches the base rate, and
    it says nothing about calibration within subgroups; for that use
    :func:`expected_calibration_error`.

    Args:
        y_true: Ground truth values (0 or 1 for binary classification).
        y_pred: Predicted probabilities.

    Returns:
        float: Absolute difference between the mean prediction and the base rate.

    Raises:
        ValueError: If arrays have different shapes.

    Notes:
        .. versionchanged:: 0.7.0
           Previously this returned ``mean(|y_pred - y_true|)``, which is mean
           absolute error, not a calibration error at all: it is minimised by hard
           0/1 predictions and is nonzero for a perfectly calibrated model -- a
           perfectly calibrated constant predictor of 0.5 scored 0.5. Use
           :func:`sklearn.metrics.mean_absolute_error` if you want the old quantity.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> round(mean_calibration_error(y_true, y_pred), 4)   # mean 0.54 vs base 0.6
        0.06

        A perfectly calibrated predictor scores zero, however unsharp it is:

        >>> y = np.array([0, 0, 1, 1])
        >>> mean_calibration_error(y, np.full(4, 0.5))
        0.0
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    # Ensure inputs have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred should have the same shape")

    return float(abs(np.mean(y_pred) - np.mean(y_true)))


def binned_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x: np.ndarray | None = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    return_details: bool = False,
) -> float | dict:
    """Calculate binned calibration error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        x: Input features for binning. If None, y_pred is used for binning.
        n_bins: Number of bins.
        strategy: Strategy for binning:

            - 'uniform': Bins with uniform widths.
            - 'quantile': Bins with approximately equal counts.
        return_details: If True, return bin details (bin centers, counts, mean
            predictions, mean truths).

    Returns:
        bce: Binned calibration error. If return_details is True, returns a
            dictionary with BCE and bin details.

    Raises:
        ValueError: If arrays have different lengths, ``n_bins`` is below 1,
            or the binning strategy is unknown.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> binned_calibration_error(y_true, y_pred, n_bins=2)
        0.3
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    # Check that arrays have matching lengths
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Bind to a separate name so the type is narrowed: reassigning `x` would leave
    # `None` in its inferred union for every use below.
    if x is None:
        bin_on = y_pred
    else:
        bin_on = check_array(x, ensure_2d=False)
        if len(bin_on) != len(y_true):
            raise ValueError("x must have the same length as y_true and y_pred")

    # Create bins based on strategy
    if strategy == "uniform":
        bins = np.linspace(np.min(bin_on), np.max(bin_on), n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(bin_on, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_ids = np.digitize(bin_on, bins) - 1
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
    bce = np.sqrt(error / valid_bins) if valid_bins > 0 else 0.0

    if return_details:
        return {
            "bce": bce,
            "bin_centers": np.array(bin_centers),
            "bin_counts": np.array(bin_counts),
            "bin_pred_means": np.array(bin_pred_means),
            "bin_true_means": np.array(bin_true_means),
        }
    return float(bce)


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE).

    The ECE is a weighted average of the absolute calibration error across bins,
    where each bin's weight is proportional to the number of samples in the bin.

    Args:
        y_true: Ground truth values (0 or 1 for binary classification).
        y_pred: Predicted probabilities.
        n_bins: Number of bins for discretizing predictions.

    Returns:
        ece: Expected Calibration Error.

    Raises:
        ValueError: If arrays have different lengths or ``n_bins`` is below 1.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> float(expected_calibration_error(y_true, y_pred, n_bins=2))
        0.3
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
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


def maximum_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Maximum Calibration Error (MCE).

    The MCE is the maximum absolute difference between the average predicted
    probability and the fraction of positive samples in any bin.

    Args:
        y_true: Ground truth values (0 or 1 for binary classification).
        y_pred: Predicted probabilities.
        n_bins: Number of bins for discretizing predictions.

    Returns:
        mce: Maximum Calibration Error.

    Raises:
        ValueError: If arrays have different lengths or ``n_bins`` is below 1.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> round(float(maximum_calibration_error(y_true, y_pred, n_bins=2)), 4)
        0.3
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
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
    """Calculate the Brier score.

    The Brier score is a proper scoring rule that measures the mean squared
    difference between predicted probabilities and the actual outcomes.

    Args:
        y_true: Ground truth values (0 or 1 for binary classification).
        y_pred: Predicted probabilities.

    Returns:
        score: Brier score (lower is better).

    Raises:
        ValueError: If arrays have different lengths.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> brier_score(y_true, y_pred)
        0.098
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return float(brier_score_loss(y_true, y_pred))


def correlation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x: np.ndarray | None = None,
    y_orig: np.ndarray | None = None,
) -> dict:
    """Calculate correlation metrics between various signals.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted/calibrated values.
        x: Input features.
        y_orig: Original uncalibrated predictions.

    Returns:
        correlations: Dictionary of correlation metrics.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.4, 0.6])
        >>> y_orig = np.array([0.1, 0.6, 0.9, 0.3, 0.5])
        >>> corr = correlation_metrics(y_true, y_pred, y_orig=y_orig)
        >>> sorted(corr)
        ['spearman_corr_orig_to_calib', 'spearman_corr_to_y_orig',
         'spearman_corr_to_y_true']
        >>> round(float(corr["spearman_corr_to_y_true"]), 4)
        0.866
        >>> round(float(corr["spearman_corr_to_y_orig"]), 4)
        1.0
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    results = {"spearman_corr_to_y_true": _spearman(y_true, y_pred)}

    if y_orig is not None:
        orig = check_array(y_orig, ensure_2d=False)
        corr_orig = _spearman(orig, y_pred)
        results["spearman_corr_to_y_orig"] = corr_orig
        results["spearman_corr_orig_to_calib"] = corr_orig  # backward-compatible alias

    if x is not None:
        scores = check_array(x, ensure_2d=False)
        results["spearman_corr_to_x"] = _spearman(scores, y_pred)

    return results


def unique_value_counts(
    y_pred: np.ndarray, y_orig: np.ndarray | None = None, precision: int = 6
) -> dict:
    """Count unique values in predictions.

    Args:
        y_pred: Predicted/calibrated values.
        y_orig: Original uncalibrated predictions.
        precision: Decimal precision for rounding.

    Returns:
        counts: Dictionary with counts of unique values.

    Examples:
        >>> import numpy as np
        >>> y_pred = np.array([0.2, 0.7, 0.8, 0.2, 0.7])
        >>> y_orig = np.array([0.1, 0.6, 0.9, 0.2, 0.5])
        >>> unique_value_counts(y_pred, y_orig)
        {'n_unique_y_pred': 3, 'n_unique_y_orig': 5, 'unique_value_ratio': 0.6}
    """
    y_pred = check_array(y_pred, ensure_2d=False)

    results: dict[str, int | float] = {
        "n_unique_y_pred": len(np.unique(np.round(y_pred, precision)))
    }

    if y_orig is not None:
        orig = check_array(y_orig, ensure_2d=False)
        results["n_unique_y_orig"] = len(np.unique(np.round(orig, precision)))
        results["unique_value_ratio"] = float(results["n_unique_y_pred"]) / max(
            1, int(results["n_unique_y_orig"])
        )

    return results


def calibration_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the calibration curve for binary classification.

    Args:
        y_true: Ground truth values (0 or 1 for binary classification).
        y_pred: Predicted probabilities.
        n_bins: Number of bins for discretizing predictions.
        strategy: Strategy for binning:

            - 'uniform': Bins with uniform widths.
            - 'quantile': Bins with approximately equal counts.

    Returns:
        prob_true: The true fraction of positive samples in each bin.
        prob_pred: The mean predicted probability in each bin.
        counts: The number of samples in each bin.

    Raises:
        ValueError: If arrays have different lengths, ``n_bins`` is below 1,
            or the binning strategy is unknown.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.9, 0.1])
        >>> prob_true, prob_pred, counts = calibration_curve(y_true, y_pred, n_bins=5)
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
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
    """Measure how well calibration preserves genuine ties while removing spurious ones.

    Args:
        y_original: Original predicted probabilities before calibration.
        y_calibrated: Calibrated probabilities.
        tolerance: Tolerance for considering values as tied.

    Raises:
        ValueError: If arrays have different lengths.

    Returns:
        score: Tie preservation score between 0 and 1. Higher values indicate
            better preservation of meaningful ties.

    Examples:
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
    # With no original ties there is nothing to preserve, so the rate is vacuously 1.
    preservation_rate = 1.0 if tied_orig == 0 else preserved_ties / tied_orig

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
    """Overall quality score for plateaus in calibrated predictions.

    Args:
        X: Input features.
        y: True target values.
        y_calibrated: Calibrated predictions.

    Raises:
        ValueError: If arrays have different lengths.

    Returns:
        score: Quality score between 0 and 1. Higher values indicate better
            plateau quality.

    Examples:
        >>> import numpy as np
        >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> y_cal = np.array([0.1, 0.25, 0.25, 0.4, 0.6])
        >>> score = plateau_quality_score(X, y, y_cal)
        >>> bool(0 <= score <= 1)
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

    return float(np.mean(scores)) if scores else 1.0


def calibration_diversity_index(
    y_calibrated: np.ndarray, reference_diversity: float | None = None
) -> float:
    """Measure granularity preservation in calibrated predictions.

    Args:
        y_calibrated: Calibrated predictions.
        reference_diversity: Reference diversity to compare against (e.g.,
            diversity of original predictions). If None, returns absolute
            diversity.

    Returns:
        diversity: Diversity index. Higher values indicate more granular
            predictions. If reference_diversity is provided, returns relative
            diversity.

    Examples:
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
    """Compute diversity vs sample size curve for progressive sampling analysis.

    Args:
        X: Input features.
        y: Target values.
        sample_sizes: Sample sizes to test. If None, uses default range.
        n_trials: Number of trials per sample size.
        random_state: Random state for reproducibility.

    Raises:
        ValueError: If X and y have different lengths, a sample size lies
            outside the data, or ``n_trials`` is below 1.

    Returns:
        sizes: Sample sizes tested.
        diversities: Average diversity at each sample size.

    Examples:
        >>> import numpy as np
        >>> X = np.linspace(0, 1, 100)
        >>> y = np.random.binomial(1, X, 100)
        >>> sizes, divs = progressive_sampling_diversity(
        ...     X, y, sample_sizes=[20, 50, 80]
        ... )
        >>> len(sizes) == len(divs) == 3
        True
    """
    X = check_array(X, ensure_2d=False)
    y = check_array(y, ensure_2d=False)

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n_total = len(X)

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

            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso_reg.fit(X_sub, y_sub)
            y_cal = iso_reg.transform(X_sub)

            # Compute diversity
            diversity = calibration_diversity_index(y_cal)
            trial_diversities.append(diversity)

        diversities.append(float(np.mean(trial_diversities)))

    return sample_sizes, diversities


def _equal_mass_bins(y_pred: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Assign each prediction to an approximately equal-mass bin.

    Args:
        y_pred: Predicted probabilities.
        n_bins: Requested number of bins. Fewer are returned when ties prevent it.

    Returns:
        bin_id: Bin index per observation.
        n_used: Number of bins actually produced.

    Notes:
        Equal-mass rather than equal-width because Roelofs et al. (2022) measure
        consistently smaller bias for equal-mass binning, a point they note is "not
        well appreciated in the literature" -- equal width is the common practice,
        including in the debiased estimator's original presentation.

        Bin edges are snapped outward to the end of each run of tied predictions, so
        identical scores always share a bin. Splitting a tie group would compare a
        bin's mean prediction against a mean label drawn from an arbitrary subset of
        observations carrying that same prediction, which measures the sort order
        rather than calibration. Clipped or rounded scores make this common: a
        forecast clipped into [0, 1] can put hundreds of observations on a single
        value. The cost is that bins are only approximately equal in mass, and that
        heavily tied data supports fewer bins than requested.
    """
    order = np.argsort(y_pred, kind="mergesort")
    sorted_pred = y_pred[order]
    n = y_pred.size

    # Ideal rank cut points. A cut is moved only when a run of tied predictions
    # straddles it, and then forward to that run's end; an unconditional snap
    # would shift every cut by one even on data with no ties at all.
    ideal = (np.arange(1, n_bins) * n) // n_bins
    ideal = ideal[(ideal > 0) & (ideal < n)]
    straddles = sorted_pred[ideal - 1] == sorted_pred[ideal]
    snapped = np.where(
        straddles,
        np.searchsorted(sorted_pred, sorted_pred[ideal], side="right"),
        ideal,
    )
    edges = np.unique(snapped)
    edges = edges[(edges > 0) & (edges < n)]

    bin_id = np.empty(n, dtype=int)
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [n]])
    for k, (lo, hi) in enumerate(zip(starts, stops, strict=True)):
        bin_id[order[lo:hi]] = k
    return bin_id, len(starts)


def _bin_summaries(
    y_true: np.ndarray, y_pred: np.ndarray, bin_id: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin counts, mean prediction and mean label.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted probabilities.
        bin_id: Bin index per observation.
        n_bins: Number of bins.

    Returns:
        counts: Observations per bin.
        mean_pred: Mean prediction per bin.
        mean_true: Mean label per bin.
    """
    counts = np.bincount(bin_id, minlength=n_bins).astype(float)
    safe = np.where(counts > 0, counts, 1.0)
    mean_pred = np.bincount(bin_id, weights=y_pred, minlength=n_bins) / safe
    mean_true = np.bincount(bin_id, weights=y_true, minlength=n_bins) / safe
    return counts, mean_pred, mean_true


def plugin_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15, p: int = 2
) -> float:
    r"""Calculate the uncorrected :math:`\ell_p` binned calibration error.

    .. math::
        \widehat{\mathrm{CE}}_p = \left[ \sum_k \frac{n_k}{n}
        \left| \bar{f}_k - \bar{y}_k \right|^p \right]^{1/p}

    This is the plain plugin estimator: the quantity
    :func:`debiased_calibration_error` corrects and
    :func:`sweep_calibration_error` chooses a bin count for. It exists so those
    three can be compared on equal terms.

    That comparison is otherwise a trap. :func:`expected_calibration_error` is
    :math:`\ell_1` on **uniform-width** bins, :func:`debiased_calibration_error`
    is :math:`\ell_2` on **equal-mass** bins, and
    :func:`sweep_calibration_error` is :math:`\ell_1` on equal-mass bins by
    default -- so plotting them against each other compares three different
    quantities and reads as disagreement between estimators. This function takes
    both the norm and the bin count as arguments and uses the same equal-mass,
    tie-safe binning as the bias-aware estimators, so the only thing that differs
    is the bias correction.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        n_bins: Number of equal-mass bins. Fewer are used when ties prevent it.
        p: Norm. 1 gives the familiar weighted mean absolute gap; 2 matches
            :func:`debiased_calibration_error`.

    Returns:
        float: The uncorrected calibration error. Biased upward, and
            increasingly so as ``n_bins`` grows.

    Raises:
        ValueError: If the arrays disagree in length, ``n_bins`` is below 1,
            or ``p`` is below 1.

    See Also:
        debiased_calibration_error : The same quantity at ``p=2``, bias-corrected.
        sweep_calibration_error : Chooses ``n_bins`` rather than fixing it.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p_hat = rng.uniform(0, 1, 4000)
        >>> y = rng.binomial(1, p_hat).astype(float)

        These are calibrated by construction, so the true error is zero and whatever
        the plugin reports is bias -- which grows with the bin count:

        >>> coarse = plugin_calibration_error(y, p_hat, n_bins=5)
        >>> fine = plugin_calibration_error(y, p_hat, n_bins=50)
        >>> bool(fine > coarse)
        True

        Debiasing removes it:

        >>> bool(debiased_calibration_error(y, p_hat, n_bins=50) < fine)
        True
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if p < 1:
        raise ValueError(f"p must be at least 1, got {p}")

    n = len(y_true)
    if n == 0:
        return 0.0

    n_bins = min(n_bins, n)
    bin_id, n_used = _equal_mass_bins(y_pred, n_bins)
    counts, mean_pred, mean_true = _bin_summaries(y_true, y_pred, bin_id, n_used)

    occupied = counts > 0
    gaps = np.abs(mean_pred[occupied] - mean_true[occupied]) ** p
    total = float(np.sum(counts[occupied] / n * gaps))
    return float(total ** (1.0 / p))


def debiased_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15, squared: bool = False
) -> float:
    r"""Calculate the debiased :math:`\ell_2` calibration error.

    The plugin binned estimator is biased upward: each bin contributes the
    squared gap between mean prediction and mean label, and part of that gap is
    sampling noise in the label mean rather than miscalibration. The bias is
    roughly ``n_bins / n``, so it grows as bins are added -- which is exactly
    when a finer picture of the calibration curve is wanted. Subtracting the
    per-bin Bernoulli variance removes it.

    .. math::
        \widehat{\mathrm{CE}}^2 = \sum_k \frac{n_k}{n}
        \left[ (\bar{f}_k - \bar{y}_k)^2
             - \frac{\bar{y}_k (1 - \bar{y}_k)}{n_k - 1} \right]

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        n_bins: Number of equal-mass bins. Defaults to 15, following Guo et
            al. (2017) as used by Roelofs et al.
        squared: Return the estimate of the **squared** error instead, without
            the square root or the floor at zero. This is the quantity the
            correction actually makes unbiased, and it may legitimately come
            out negative -- see Notes.

    Returns:
        float: Debiased calibration error. Floored at zero: the correction can
            drive the sum negative on well-calibrated data, which is evidence
            of no detectable miscalibration rather than of negative error.
            With ``squared=True`` the unfloored sum is returned instead.

    Raises:
        ValueError: If the arrays disagree in length or ``n_bins`` is below 1.

    Notes:
        This is the :math:`\ell_2` error, so it is not comparable in magnitude to
        :func:`expected_calibration_error`, which is :math:`\ell_1`.

        **The correction is unbiased on the squared scale, not on the error scale.**
        Measured on 400 perfectly calibrated samples of 1500 observations, where the
        true error is exactly zero (``tests/test_monte_carlo.py``):

        ==========================================  =========  ===================
        quantity                                    mean       distance from zero
        ==========================================  =========  ===================
        ``squared=True`` (the sum itself)           +4.5e-05   1.3 standard errors
        default (``sqrt`` of the floored sum)       +0.0106    15.7 standard errors
        ==========================================  =========  ===================

        The sum is unbiased, exactly as intended, and comes out **negative on 53% of
        calibrated samples** -- what an unbiased estimate of zero should do. The floor
        then discards that half, and no amount of data removes the resulting upward
        bias in the reported error. (The square root pulls the other way, being
        concave: ``E[sqrt(W)]`` of 0.0106 against ``sqrt(E[W])`` of 0.0172.)

        So: to *report* an error, use the default, and read a small positive value on
        well-calibrated data as the floor rather than as evidence. To **average across
        folds, compare two models, or do anything else that assumes unbiasedness**,
        use ``squared=True`` and take the square root at the very end, if at all.

    References:
        Bröcker (2012); Ferro & Fricker (2012); Kumar, Liang & Ma (2019),
        "Verified Uncertainty Calibration", NeurIPS.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 4000)
        >>> y = rng.binomial(1, p).astype(float)

        These predictions are calibrated, so the plugin estimator reports error that
        is not there while the debiased one does not:

        >>> plugin = expected_calibration_error(y, p, n_bins=15)
        >>> debiased = debiased_calibration_error(y, p, n_bins=15)
        >>> bool(plugin > 0.01), bool(debiased < 0.01)
        (True, True)

    See Also:
        sweep_calibration_error : Chooses the bin count instead of fixing it.
        calibre.evaluation.score_decomposition : Avoids binning altogether.
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    n_bins = min(n_bins, len(y_true))
    bin_id, n_used = _equal_mass_bins(y_pred, n_bins)
    counts, mean_pred, mean_true = _bin_summaries(y_true, y_pred, bin_id, n_used)

    # A bin holding one observation has no within-bin variance estimate, so its
    # plugin term is pure noise with nothing to subtract. It contributes zero
    # rather than an uncorrectable term, matching the reference implementation
    # accompanying Kumar et al. (2019).
    correctable = counts > 1
    per_bin = np.zeros_like(counts)
    variance = (
        mean_true[correctable]
        * (1.0 - mean_true[correctable])
        / (counts[correctable] - 1.0)
    )
    per_bin[correctable] = (
        mean_pred[correctable] - mean_true[correctable]
    ) ** 2 - variance

    total = float(np.sum(counts / len(y_true) * per_bin))
    if squared:
        return total
    return float(np.sqrt(max(total, 0.0)))


@overload
def sweep_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: int = ...,
    return_n_bins: Literal[False] = ...,
) -> float: ...


@overload
def sweep_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: int = ...,
    *,
    return_n_bins: Literal[True],
) -> tuple[float, int]: ...


def sweep_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: int = 1,
    return_n_bins: bool = False,
) -> float | tuple[float, int]:
    r"""Calculate the monotonic sweep calibration error (``ECE_sweep``).

    Fixing the bin count is the weak point of binned calibration error: too few
    bins hide miscalibration, too many measure noise, and the best choice depends
    on the sample size and the score distribution. This estimator chooses instead.

    A true calibration curve is non-decreasing -- a model's accuracy should not
    fall as its confidence rises. So bins are added while the observed bin heights
    stay monotone, and the sweep stops at the largest bin count for which they do.
    Non-monotonicity is the signal that the bins have become fine enough to be
    reading noise.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities.
        p: Norm. 1 gives the familiar weighted mean absolute gap.
        return_n_bins: Also return the bin count the sweep settled on. That
            number is half of what the estimator has to say -- it is the
            sweep's answer to "how fine can these data support?" -- and
            reporting only the error hides it.

    Returns:
        float or tuple of (float, int): Binned calibration error at the
            selected bin count, and that bin count when ``return_n_bins`` is
            True. The count is the number of bins actually occupied, which
            ties can hold below the number the sweep reached.

    Raises:
        ValueError: If the arrays disagree in length or ``p`` is below 1.

    References:
        Roelofs, Cain, Shlens & Mozer (2022), "Mitigating Bias in Calibration Error
        Estimation", AISTATS. Algorithm 1.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 4000)
        >>> y = rng.binomial(1, p).astype(float)
        >>> float(sweep_calibration_error(y, p)) < 0.05
        True

    See Also:
        debiased_calibration_error : Fixes the bin count and corrects the bias.
        calibre.evaluation.score_decomposition : Lets isotonic regression bin.
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if p < 1:
        raise ValueError(f"p must be at least 1, got {p}")

    n = len(y_true)
    if n < 2:
        return (0.0, int(n)) if return_n_bins else 0.0

    def error_at(n_bins: int) -> tuple[float, bool, int]:
        bin_id, n_used = _equal_mass_bins(y_pred, n_bins)
        counts, mean_pred, mean_true = _bin_summaries(y_true, y_pred, bin_id, n_used)
        occupied = counts > 0
        monotone = bool(np.all(np.diff(mean_true[occupied]) >= 0.0))
        gaps = np.abs(mean_pred[occupied] - mean_true[occupied]) ** p
        error = float(np.sum(counts[occupied] / n * gaps) ** (1.0 / p))
        return error, monotone, int(np.count_nonzero(occupied))

    # b = 2 is guaranteed monotone only in the sense that the sweep needs a
    # starting point; if even it is not, one bin is all the data supports.
    best, _, best_bins = error_at(1)
    for n_bins in range(2, n + 1):
        error, monotone, occupied = error_at(n_bins)
        if not monotone:
            break
        best, best_bins = error, occupied
    return (best, best_bins) if return_n_bins else best


def _reflect_and_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve on the unit interval with reflecting boundaries.

    Args:
        values: Gridded mass, length ``m``.
        kernel: Kernel evaluated on the same grid, length ``m``.

    Returns:
        ndarray: The smoothed grid, length ``m``.

    Notes:
        Predictions pile up at 0 and 1 -- a clipped or confident model puts real mass
        exactly on the bounds -- and a plain convolution would let that mass leak off
        the ends, understating error precisely where models are most overconfident.
        Reflecting the grid at both ends keeps it inside.
    """
    m = values.size
    extended = np.concatenate([np.flip(values)[:-1], values, np.flip(values)[1:]])
    return np.convolve(extended, kernel, "valid")[m // 2 : m // 2 + m]


def _gaussian_kernel(sigma: float, n_points: int) -> np.ndarray:
    """Evaluate a Gaussian kernel of width ``sigma`` on a unit grid.

    Args:
        sigma: Kernel bandwidth.
        n_points: Grid size.

    Returns:
        ndarray: Kernel values, centred on the grid.
    """
    t = np.linspace(0.0, 1.0, n_points)
    return np.exp(-((t - 0.5) ** 2) / (2.0 * sigma**2)) / (np.sqrt(2.0 * np.pi) * sigma)


def _spread_to_grid(y_pred: np.ndarray, values: np.ndarray, m: int) -> np.ndarray:
    """Bin ``values`` onto a regular grid, splitting each linearly between neighbours.

    Args:
        y_pred: Positions in ``[0, 1]``.
        values: Mass carried by each position.
        m: Grid size.

    Returns:
        ndarray: Gridded mass, length ``m``.
    """
    grid = np.zeros(m)
    scaled = y_pred * (m - 1)
    lower = scaled.astype(int).clip(0, m - 2)
    frac = scaled - lower
    np.add.at(grid, lower, (1.0 - frac) * values)
    np.add.at(grid, lower + 1, frac * values)
    return grid


def _interpolate_grid(t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate a gridded function at ``t``.

    Args:
        t: Evaluation points in ``[0, 1]``.
        grid: Function values on a regular grid over ``[0, 1]``.

    Returns:
        ndarray: Interpolated values.
    """
    n = grid.size
    index = (t * (n - 1)).astype(int).clip(0, n - 2)
    residual = t * (n - 1) - index
    return grid[index] * (1.0 - residual) + grid[index + 1] * residual


def _smooth_at(
    y_pred: np.ndarray, values: np.ndarray, t: np.ndarray, sigma: float
) -> np.ndarray:
    """Kernel-smooth ``values`` located at ``y_pred``, evaluated at ``t``.

    Args:
        y_pred: Positions in ``[0, 1]``.
        values: Mass carried by each position.
        t: Evaluation points.
        sigma: Kernel bandwidth.

    Returns:
        ndarray: Smoothed values at ``t``.
    """
    m = max(2000, round(20.0 / sigma)) // 2 + 1
    gridded = _spread_to_grid(y_pred, values, m)
    smoothed = _reflect_and_convolve(gridded, _gaussian_kernel(sigma, m))
    return _interpolate_grid(t, smoothed)


@overload
def smooth_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float | None = ...,
    return_sigma: Literal[False] = ...,
) -> float: ...


@overload
def smooth_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float | None = ...,
    *,
    return_sigma: Literal[True],
) -> tuple[float, float]: ...


def smooth_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: float | None = None,
    return_sigma: bool = False,
) -> float | tuple[float, float]:
    r"""Calculate the smooth calibration error (smECE).

    Binned calibration error has no consistent limit: refine the bins and the
    estimate keeps climbing on data that is perfectly calibrated. smECE replaces
    the bins with a Gaussian kernel and, crucially, chooses its own bandwidth, so
    there is no knob at all -- not a bin count, not a bandwidth.

    .. math::
        \mathrm{smECE}_\sigma(f, y) =
        \frac{\int \left| (K_\sigma \star \nu)(t) \right| \, dt}
             {\int (K_\sigma \star \rho)(t) \, dt},
        \qquad
        \nu = \sum_i (f_i - y_i)\, \delta_{f_i},
        \quad \rho = \sum_i \delta_{f_i}

    The bandwidth is the fixed point :math:`\sigma = \mathrm{smECE}_\sigma`, found
    by bisection. Below that width the kernel is resolving noise; above it, it is
    smoothing away real miscalibration.

    This is a *consistent* calibration measure in the sense of Blasiok, Gopalan,
    Hu and Nakkiran (2023): it is bounded above and below by polynomial functions
    of the true distance to the nearest perfectly calibrated predictor. Binned ECE
    is not, which is why it can report a large error for a predictor that is
    almost calibrated and a small one for a predictor that is not.

    Args:
        y_true: Ground truth values (0 or 1).
        y_pred: Predicted probabilities in ``[0, 1]``.
        sigma: Kernel bandwidth. When None, the fixed point above is used,
            which is the recommended behaviour and what makes the estimator
            hyperparameter-free.
        return_sigma: Also return the bandwidth used. Worth reporting: it is
            an interpretable scale, roughly the resolution at which
            miscalibration is detectable.

    Returns:
        float or tuple of (float, float): The smooth calibration error, and
            the bandwidth when ``return_sigma``.

    Raises:
        ValueError: If the arrays disagree in length, ``y_pred`` falls outside
            ``[0, 1]``, or ``sigma`` is not positive.

    See Also:
        - :func:`debiased_calibration_error` -- bias-corrected, but still needs a
          bin count.
        - :func:`calibre.evaluation.score_decomposition` -- avoids binning by using
          isotonic regression, and decomposes the score rather than summarising
          the error.

    References:
        Blasiok & Nakkiran (2024), "Smooth ECE: Principled Reliability Diagrams via
        Kernel Smoothing", ICLR. Blasiok, Gopalan, Hu & Nakkiran (2023), "A Unifying
        Theory of Distance from Calibration", STOC.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = rng.uniform(0, 1, 2000)
        >>> y = rng.binomial(1, p).astype(float)

        Calibrated by construction, so the error is near zero:

        >>> bool(smooth_calibration_error(y, p) < 0.03)
        True

        An overconfident predictor is caught:

        >>> squashed = np.clip(2.0 * (p - 0.5) + 0.5, 0, 1)
        >>> bool(smooth_calibration_error(y, squashed) > 0.05)
        True

        Unlike a binned estimator there is no bin count to justify; the bandwidth is
        chosen by the data:

        >>> error, width = smooth_calibration_error(y, p, return_sigma=True)
        >>> bool(0.0 < width <= 1.0)
        True
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if sigma is not None and sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_pred.size == 0:
        return (0.0, 1.0) if return_sigma else 0.0
    if np.any(y_pred < 0.0) or np.any(y_pred > 1.0):
        raise ValueError("y_pred must lie in [0, 1]")

    residual = y_pred - y_true

    def error_at(width: float) -> float:
        n_eval = max(round(10.0 / width), 200)
        t = np.linspace(0.0, 1.0, n_eval)
        smoothed = _smooth_at(y_pred, residual, t, width)
        # The 1e-4 floor keeps the ratio finite where the kernel reaches a region
        # holding no predictions at all.
        density = _smooth_at(y_pred, np.ones_like(residual), t, width) + 1e-4
        return float(np.sum(np.abs(smoothed)) / np.sum(density))

    if sigma is not None:
        value = error_at(sigma)
        return (value, float(sigma)) if return_sigma else value

    # Bisect for the fixed point sigma = smECE_sigma. `resolved(w)` is True once
    # the kernel is at least as wide as the error it measures, so the smallest
    # such width is the self-consistent one.
    def resolved(width: float) -> bool:
        return width < 1e-3 or width < error_at(width)

    width = 1.0
    if not resolved(width):
        low, high = 1.0, 0.0
        for _ in range(10):
            mid = (low + high) / 2.0
            if resolved(mid):
                high = mid
            else:
                low = mid
        width = low

    value = error_at(width)
    return (value, float(width)) if return_sigma else value


# Declared at the end of the module so that every public name above is already
# defined. Declared near the top, this list silently omitted the two bias-aware
# estimators from ``from calibre.metrics import *``.
__all__ = [
    "binned_calibration_error",
    "brier_score",
    "calibration_curve",
    "calibration_diversity_index",
    "correlation_metrics",
    "debiased_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "mean_calibration_error",
    "plateau_quality_score",
    "plugin_calibration_error",
    "progressive_sampling_diversity",
    "smooth_calibration_error",
    "sweep_calibration_error",
    "tie_preservation_score",
    "unique_value_counts",
]
