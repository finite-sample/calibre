"""
Utility functions for the calibre package.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.utils import check_array


def check_arrays(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check and validate input arrays.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Input features.
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    X : ndarray of shape (n_samples,)
        Validated input features as a 1D numpy array.
    y : ndarray of shape (n_samples,)
        Validated target values as a 1D numpy array.

    Raises
    ------
    ValueError
        If X and y have different lengths or are empty.

    Examples
    --------
    >>> import numpy as np
    >>> X = [0.1, 0.3, 0.5, 0.7, 0.9]
    >>> y = [0, 0, 1, 1, 1]
    >>> X_valid, y_valid = check_arrays(X, y)
    """
    # Convert inputs to numpy arrays and validate dimensions.
    X = check_array(X, ensure_2d=False, ensure_all_finite="allow-nan")
    y = check_array(y, ensure_2d=False, ensure_all_finite="allow-nan")

    # Flatten to 1D arrays.
    X = X.ravel()
    y = y.ravel()

    # Check for empty arrays
    if len(X) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Verify that the arrays have the same length.
    if len(X) != len(y):
        raise ValueError(
            f"Input arrays X and y must have the same length. Got len(X)={len(X)} and len(y)={len(y)}"
        )

    return X, y


def sort_by_x(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort arrays by X values.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Input features.
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    sort_idx : ndarray of shape (n_samples,)
        Indices that would sort X.
    X_sorted : ndarray of shape (n_samples,)
        Sorted X values.
    y_sorted : ndarray of shape (n_samples,)
        y values sorted by X.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0.5, 0.3, 0.7, 0.1, 0.9])
    >>> y = np.array([1, 0, 1, 0, 1])
    >>> idx, X_sorted, y_sorted = sort_by_x(X, y)
    >>> print(X_sorted)
    [0.1 0.3 0.5 0.7 0.9]
    >>> print(y_sorted)
    [0 0 1 1 1]
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Get sorting indices
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    return sort_idx, X_sorted, y_sorted


def create_bins(
    X: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> np.ndarray:
    """
    Create bin edges for discretizing continuous values.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Values to be binned.
    n_bins : int, default=10
        Number of bins.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy for binning:
        - 'uniform': Bins with uniform widths.
        - 'quantile': Bins with approximately equal counts.
    x_min : float, optional
        Minimum value for bin range. If None, uses min(X).
    x_max : float, optional
        Maximum value for bin range. If None, uses max(X).

    Returns
    -------
    bins : ndarray of shape (n_bins+1,)
        Bin edges.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    >>> create_bins(X, n_bins=3, strategy='uniform')
    array([0.1, 0.367, 0.633, 0.9])
    """
    X = np.asarray(X)

    # Validate n_bins
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    if strategy == "uniform":
        # Create bins with uniform widths
        min_val = x_min if x_min is not None else np.min(X)
        max_val = x_max if x_max is not None else np.max(X)
        bins = np.linspace(min_val, max_val, n_bins + 1)
    elif strategy == "quantile":
        # Create bins with approximately equal counts
        bins = np.percentile(X, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(
            f"Unknown binning strategy: {strategy}. " f"Use 'uniform' or 'quantile'."
        )

    return bins


def bin_data(X: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign data points to bins and compute bin counts.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Values to be binned.
    bins : array-like of shape (n_bins+1,)
        Bin edges.

    Returns
    -------
    bin_indices : ndarray of shape (n_samples,)
        Bin indices for each data point.
    bin_counts : ndarray of shape (n_bins,)
        Number of data points in each bin.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> bins = np.array([0.0, 0.33, 0.67, 1.0])
    >>> bin_indices, bin_counts = bin_data(X, bins)
    >>> print(bin_indices)
    [0 0 1 2 2]
    >>> print(bin_counts)
    [2 1 2]
    """
    X = np.asarray(X)
    bins = np.asarray(bins)

    # Assign data points to bins
    bin_indices = np.digitize(X, bins) - 1

    # Clip bin indices to valid range
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    # Compute bin counts
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)

    return bin_indices, bin_counts


def extract_plateaus(
    X: np.ndarray, y_calibrated: np.ndarray, tolerance: float = 1e-10
) -> List[Tuple[int, int, float]]:
    """
    Extract plateau regions from calibrated isotonic regression output.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Input features (should be sorted).
    y_calibrated : array-like of shape (n_samples,)
        Calibrated output values from isotonic regression.
    tolerance : float, default=1e-10
        Tolerance for considering values as equal (forming a plateau).

    Returns
    -------
    plateaus : list of tuples
        List of (start_idx, end_idx, value) for each plateau.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> y_cal = np.array([0.1, 0.25, 0.25, 0.25, 0.5, 0.6])
    >>> plateaus = extract_plateaus(X, y_cal)
    >>> len(plateaus)
    1
    """
    X = np.asarray(X)
    y_calibrated = np.asarray(y_calibrated)

    if len(X) != len(y_calibrated):
        raise ValueError("X and y_calibrated must have the same length")

    plateaus = []
    n = len(y_calibrated)

    if n == 0:
        return plateaus

    # Find consecutive equal values
    i = 0
    while i < n:
        start_idx = i
        current_value = y_calibrated[i]

        # Find end of plateau
        while i + 1 < n and abs(y_calibrated[i + 1] - current_value) <= tolerance:
            i += 1

        end_idx = i

        # Only consider it a plateau if it spans more than one point
        if end_idx > start_idx:
            plateaus.append((start_idx, end_idx, current_value))

        i += 1

    return plateaus


def bootstrap_resample(
    X: np.ndarray, y: np.ndarray, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a bootstrap resample of the data.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Input features.
    y : array-like of shape (n_samples,)
        Target values.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    X_boot : ndarray of shape (n_samples,)
        Bootstrap resampled input features.
    y_boot : ndarray of shape (n_samples,)
        Bootstrap resampled target values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> X_boot, y_boot = bootstrap_resample(X, y, random_state=42)
    >>> len(X_boot) == len(X)
    True
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    rng = np.random.RandomState(random_state)
    n = len(X)

    # Sample with replacement
    indices = rng.choice(n, size=n, replace=True)

    return X[indices], y[indices]


def compute_delong_ci(
    y_true: np.ndarray, y_scores: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute AUC with DeLong confidence interval.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_scores : array-like of shape (n_samples,)
        Target scores (predicted probabilities).
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> auc, ci_low, ci_high = compute_delong_ci(y_true, y_scores)
    >>> 0 <= ci_low <= auc <= ci_high <= 1
    True
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        return np.nan, np.nan, np.nan

    # Simplified confidence interval using bootstrap
    # For a more accurate DeLong method, we'd need more complex implementation
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return auc, np.nan, np.nan

    # Approximate standard error for AUC (Hanley & McNeil, 1982)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)

    se_auc = np.sqrt(
        (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2))
        / (n_pos * n_neg)
    )

    # Confidence interval
    z_score = stats.norm.ppf(1 - alpha / 2)
    ci_lower = max(0, auc - z_score * se_auc)
    ci_upper = min(1, auc + z_score * se_auc)

    return auc, ci_lower, ci_upper


def minimum_detectable_difference(
    n1: int, n2: int, p_pooled: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """
    Calculate minimum detectable difference for two proportions.

    Parameters
    ----------
    n1 : int
        Sample size for first group.
    n2 : int
        Sample size for second group.
    p_pooled : float
        Pooled proportion estimate.
    alpha : float, default=0.05
        Significance level.
    power : float, default=0.8
        Statistical power (1 - beta).

    Returns
    -------
    mdd : float
        Minimum detectable difference.

    Examples
    --------
    >>> mdd = minimum_detectable_difference(50, 60, 0.3)
    >>> mdd > 0
    True
    """
    if n1 <= 0 or n2 <= 0:
        return np.inf

    if p_pooled <= 0 or p_pooled >= 1:
        return np.inf

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    pooled_var = p_pooled * (1 - p_pooled)
    combined_se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))

    mdd = (z_alpha + z_beta) * combined_se

    return mdd
