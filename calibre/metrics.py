"""
Evaluation metrics for calibration.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss
from sklearn.utils import check_array


def mean_calibration_error(y_true, y_pred):
    """
    Calculate the mean calibration error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values (0 or 1 for binary classification).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.

    Returns
    -------
    mce : float
        Mean calibration error.

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
    return np.mean(np.abs(y_pred - y_true))


def binned_calibration_error(
    y_true, y_pred, x=None, n_bins=10, strategy="uniform", return_details=False
):
    """
    Calculate binned calibration error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    x : array-like of shape (n_samples,), optional
        Input features for binning. If None, y_pred is used for binning.
    n_bins : int, default=10
        Number of bins.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy for binning:
        - 'uniform': Bins with uniform widths.
        - 'quantile': Bins with approximately equal counts.
    return_details : bool, default=False
        If True, return bin details (bin centers, counts, mean predictions, mean truths).

    Returns
    -------
    bce : float or dict
        Binned calibration error. If return_details is True, returns a dictionary
        with BCE and bin details.

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
        return bce


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    The ECE is a weighted average of the absolute calibration error across bins,
    where each bin's weight is proportional to the number of samples in the bin.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values (0 or 1 for binary classification).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for discretizing predictions.

    Returns
    -------
    ece : float
        Expected Calibration Error.

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


def maximum_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate Maximum Calibration Error (MCE).

    The MCE is the maximum absolute difference between the average predicted
    probability and the fraction of positive samples in any bin.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values (0 or 1 for binary classification).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for discretizing predictions.

    Returns
    -------
    mce : float
        Maximum Calibration Error.

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


def brier_score(y_true, y_pred):
    """
    Calculate the Brier score.

    The Brier score is a proper scoring rule that measures the mean squared
    difference between predicted probabilities and the actual outcomes.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values (0 or 1 for binary classification).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.

    Returns
    -------
    score : float
        Brier score (lower is better).

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

    return brier_score_loss(y_true, y_pred)


def correlation_metrics(y_true, y_pred, x=None, y_orig=None):
    """
    Calculate correlation metrics between various signals.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted/calibrated values.
    x : array-like of shape (n_samples,), optional
        Input features.
    y_orig : array-like of shape (n_samples,), optional
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

    if x is not None:
        x = check_array(x, ensure_2d=False)
        results["spearman_corr_to_x"] = spearmanr(x, y_pred).correlation

    return results


def unique_value_counts(y_pred, y_orig=None, precision=6):
    """
    Count unique values in predictions.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted/calibrated values.
    y_orig : array-like of shape (n_samples,), optional
        Original uncalibrated predictions.
    precision : int, default=6
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

    results = {"n_unique_y_pred": len(np.unique(np.round(y_pred, precision)))}

    if y_orig is not None:
        y_orig = check_array(y_orig, ensure_2d=False)
        results["n_unique_y_orig"] = len(np.unique(np.round(y_orig, precision)))
        results["unique_value_ratio"] = results["n_unique_y_pred"] / max(
            1, results["n_unique_y_orig"]
        )

    return results


def calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform"):
    """
    Compute the calibration curve for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values (0 or 1 for binary classification).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for discretizing predictions.
    strategy : {'uniform', 'quantile'}, default='uniform'
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
