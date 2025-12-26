"""
Array operation utilities.

This module provides functions for common array operations used in calibration,
such as sorting, transforming, and manipulating arrays.
"""

from __future__ import annotations

import numpy as np


def sort_by_x(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort arrays by X values and return sort indices.

    Parameters
    ----------
    X
        Values to sort by.
    y
        Values to sort along with X.

    Returns
    -------
    sort_idx : ndarray of shape (n_samples,)
        Indices that would sort X.
    X_sorted : ndarray of shape (n_samples,)
        Sorted X array.
    y_sorted : ndarray of shape (n_samples,)
        Sorted y array.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import sort_by_x
    >>>
    >>> X = np.array([0.3, 0.1, 0.2])
    >>> y = np.array([1, 0, 0])
    >>> idx, X_sorted, y_sorted = sort_by_x(X, y)
    >>> print(X_sorted)
    [0.1 0.2 0.3]
    >>> print(y_sorted)
    [0 0 1]
    """
    X = np.asarray(X)
    y = np.asarray(y)

    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    return sort_idx, X_sorted, y_sorted


def clip_to_range(X: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """
    Clip array values to a specified range.

    Parameters
    ----------
    X
        Array to clip.
    lower
        Lower bound.
    upper
        Upper bound.

    Returns
    -------
    X_clipped : ndarray
        Clipped array.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import clip_to_range
    >>>
    >>> X = np.array([-0.1, 0.5, 1.2])
    >>> X_clipped = clip_to_range(X, 0.0, 1.0)
    >>> print(X_clipped)
    [0.  0.5 1. ]
    """
    return np.clip(X, lower, upper)


def ensure_1d(X: np.ndarray) -> np.ndarray:
    """
    Ensure array is 1-dimensional by raveling.

    Parameters
    ----------
    X
        Array to ensure is 1D.

    Returns
    -------
    X_1d : ndarray of shape (n_samples,)
        1-dimensional array.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import ensure_1d
    >>>
    >>> X = np.array([[0.1, 0.2, 0.3]])
    >>> X_1d = ensure_1d(X)
    >>> print(X_1d.shape)
    (3,)
    """
    return np.asarray(X).ravel()


def restore_order(X_sorted: np.ndarray, original_order: np.ndarray) -> np.ndarray:
    """
    Restore original order of a sorted array.

    Parameters
    ----------
    X_sorted
        Sorted array.
    original_order
        Original sort indices from sort_by_x.

    Returns
    -------
    X_restored : ndarray
        Array in original order.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import sort_by_x, restore_order
    >>>
    >>> X = np.array([0.3, 0.1, 0.2])
    >>> y = np.array([1, 0, 0])
    >>> idx, X_sorted, y_sorted = sort_by_x(X, y)
    >>>
    >>> # Do something with sorted arrays
    >>> y_sorted_modified = y_sorted * 2
    >>>
    >>> # Restore original order
    >>> y_restored = restore_order(y_sorted_modified, idx)
    >>> print(y_restored)
    [2 0 0]
    """
    X_sorted = np.asarray(X_sorted)
    X_restored = np.empty_like(X_sorted)
    X_restored[original_order] = X_sorted
    return X_restored


def find_unique_sorted(X: np.ndarray, tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """
    Find unique values in a sorted array with tolerance.

    Parameters
    ----------
    X
        Sorted array to find unique values in.
    tolerance
        Tolerance for considering values equal.

    Returns
    -------
    unique_values : ndarray
        Unique values.
    unique_indices : ndarray
        Indices of first occurrence of each unique value.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import find_unique_sorted
    >>>
    >>> X = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.3])
    >>> unique_vals, unique_idx = find_unique_sorted(X)
    >>> print(unique_vals)
    [0.1 0.2 0.3]
    >>> print(unique_idx)
    [0 2 5]
    """
    X = np.asarray(X)

    if len(X) == 0:
        return np.array([]), np.array([])

    # Find where consecutive elements differ by more than tolerance
    diffs = np.diff(X)
    change_points = np.where(np.abs(diffs) > tolerance)[0] + 1

    # Include first element
    unique_indices = np.concatenate([[0], change_points])
    unique_values = X[unique_indices]

    return unique_values, unique_indices


def group_by_value(X: np.ndarray, y: np.ndarray, tolerance: float = 1e-10) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Group y values by unique X values.

    Parameters
    ----------
    X
        Values to group by (must be sorted).
    y
        Values to group.
    tolerance
        Tolerance for considering X values equal.

    Returns
    -------
    groups : list of ndarray
        List of y value groups.
    group_indices : list of ndarray
        List of indices for each group.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import group_by_value
    >>>
    >>> X = np.array([0.1, 0.1, 0.2, 0.2, 0.3])
    >>> y = np.array([0, 1, 1, 1, 0])
    >>> groups, indices = group_by_value(X, y)
    >>> print([g.tolist() for g in groups])
    [[0, 1], [1, 1], [0]]
    """
    X = np.asarray(X)
    y = np.asarray(y)

    unique_vals, unique_indices = find_unique_sorted(X, tolerance)

    groups = []
    group_indices = []

    for i in range(len(unique_indices)):
        start = unique_indices[i]
        end = unique_indices[i + 1] if i + 1 < len(unique_indices) else len(X)

        groups.append(y[start:end])
        group_indices.append(np.arange(start, end))

    return groups, group_indices


def interpolate_monotonic(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, bounds_error: bool = False, fill_value: float | tuple[float, float] | None = None
) -> np.ndarray:
    """
    Interpolate monotonic function at new points.

    Parameters
    ----------
    X_train
        Training X values (must be sorted).
    y_train
        Training y values (should be monotonic).
    X_test
        Test X values to interpolate at.
    bounds_error
        Whether to raise error for out-of-bounds values.
    fill_value
        Value(s) to use for out-of-bounds points.
        If None, uses (y_train[0], y_train[-1]).

    Returns
    -------
    y_test : ndarray of shape (n_test,)
        Interpolated values.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.array_ops import interpolate_monotonic
    >>>
    >>> X_train = np.array([0.0, 0.5, 1.0])
    >>> y_train = np.array([0.0, 0.3, 1.0])
    >>> X_test = np.array([0.25, 0.75])
    >>> y_test = interpolate_monotonic(X_train, y_train, X_test)
    >>> print(y_test)
    [0.15 0.65]
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    if fill_value is None:
        fill_value = (float(y_train[0]), float(y_train[-1]))

    # Use numpy interp (fast and efficient)
    y_test = np.interp(X_test, X_train, y_train)

    # Handle bounds if needed
    if not bounds_error and fill_value is not None:
        if isinstance(fill_value, tuple):
            lower_fill, upper_fill = fill_value
        else:
            lower_fill = upper_fill = fill_value

        y_test = np.where(X_test < X_train[0], lower_fill, y_test)
        y_test = np.where(X_test > X_train[-1], upper_fill, y_test)

    return np.asarray(y_test)


__all__ = [
    "sort_by_x",
    "clip_to_range",
    "ensure_1d",
    "restore_order",
    "find_unique_sorted",
    "group_by_value",
    "interpolate_monotonic",
]
