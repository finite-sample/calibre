"""
Input validation utilities.

This module provides functions for validating and checking input arrays
to ensure they meet the requirements for calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_array


def check_arrays(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Check and validate input arrays for calibration.

    This function ensures that X and y are valid numpy arrays with
    compatible shapes and no invalid values.

    Parameters
    ----------
    X
        The input predictions/probabilities.
    y
        The target values/labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (validated_X, validated_y).
        validated_X
        validated_y

    Raises
    ------
    ValueError
        If arrays are empty or have incompatible lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.validation import check_arrays
    >>>
    >>> X = np.array([0.1, 0.2, 0.3])
    >>> y = np.array([0, 1, 1])
    >>> X_checked, y_checked = check_arrays(X, y)
    >>> print(X_checked.shape, y_checked.shape)
    (3,) (3,)
    """
    # Use sklearn's check_array for initial validation
    X = check_array(X, ensure_2d=False, ensure_all_finite="allow-nan")
    y = check_array(y, ensure_2d=False, ensure_all_finite="allow-nan")

    # Ensure 1D arrays
    X = X.ravel()
    y = y.ravel()

    # Check for empty arrays
    if len(X) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check for compatible lengths
    if len(X) != len(y):
        raise ValueError(
            f"Input arrays X and y must have the same length. "
            f"Got X: {len(X)}, y: {len(y)}"
        )

    return X, y


def check_array_1d(X: np.ndarray, name: str = "X") -> np.ndarray:
    """
    Check that an array is 1-dimensional.

    Parameters
    ----------
    X
        The array to check.
    name
        Name of the array for error messages.

    Returns
    -------
    Validated 1D array.

    Raises
    ------
    ValueError
        If array is not 1-dimensional or is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.validation import check_array_1d
    >>>
    >>> X = np.array([0.1, 0.2, 0.3])
    >>> X_checked = check_array_1d(X)
    >>> print(X_checked.shape)
    (3,)
    """
    X = check_array(X, ensure_2d=False, ensure_all_finite="allow-nan")
    X = X.ravel()

    if len(X) == 0:
        raise ValueError(f"Array '{name}' cannot be empty")

    return X


def check_fitted(calibrator: object, attributes: list[str] | None = None) -> None:
    """
    Check if a calibrator has been fitted.

    Parameters
    ----------
    calibrator
        The calibrator to check.
    attributes
        List of attribute names that should exist if fitted.
        If None, checks for common fitted attributes.

    Raises
    ------
    ValueError
        If the calibrator has not been fitted.

    Examples
    --------
    >>> from calibre import IsotonicCalibrator
    >>> from calibre.utils.validation import check_fitted
    >>>
    >>> cal = IsotonicCalibrator()
    >>> try:
    ...     check_fitted(cal)
    ... except ValueError as e:
    ...     print("Not fitted:", e)
    """
    if attributes is None:
        # Common fitted attribute names
        attributes = ["isotonic_", "X_", "y_", "calibrator_", "spline_", "model_"]

    # Check if any of the expected attributes exist
    has_fitted_attr = any(hasattr(calibrator, attr) for attr in attributes)

    if not has_fitted_attr:
        raise ValueError(
            f"{calibrator.__class__.__name__} must be fitted before transform. "
            f"Call fit(X, y) first."
        )


def check_consistent_length(*arrays: np.ndarray) -> None:
    """
    Check that all arrays have consistent first dimension.

    Parameters
    ----------
    *arrays
        Arrays to check for consistent length.

    Raises
    ------
    ValueError
        If arrays have inconsistent lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre.utils.validation import check_consistent_length
    >>>
    >>> X = np.array([0.1, 0.2, 0.3])
    >>> y = np.array([0, 1, 1])
    >>> check_consistent_length(X, y)  # No error
    >>>
    >>> z = np.array([0, 1])  # Different length
    >>> try:
    ...     check_consistent_length(X, z)
    ... except ValueError as e:
    ...     print("Error:", e)
    """
    lengths = [len(X) for X in arrays if X is not None]

    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent array lengths: {lengths}. "
            f"All arrays must have the same length."
        )


def validate_parameters(**params: object) -> None:
    """
    Validate common calibrator parameters.

    Parameters
    ----------
    **params
        Parameter names and values to validate.

    Raises
    ------
    ValueError
        If any parameter is invalid.

    Examples
    --------
    >>> from calibre.utils.validation import validate_parameters
    >>>
    >>> validate_parameters(alpha=0.1, n_bootstraps=100)  # OK
    >>>
    >>> try:
    ...     validate_parameters(alpha=-0.5)  # Negative
    ... except ValueError as e:
    ...     print("Error:", e)
    """
    for name, value in params.items():
        if name in ["alpha", "lam"] and value is not None:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f"Parameter '{name}' must be non-negative, got {value}"
                )

        elif name in ["n_bootstraps", "n_splits", "n_splines"] and value is not None:
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Parameter '{name}' must be a positive integer, got {value}"
                )

        elif name in ["window_length", "min_window"] and value is not None:
            if not isinstance(value, int) or value < 3:
                raise ValueError(
                    f"Parameter '{name}' must be an integer >= 3, got {value}"
                )

        elif name == "percentile" and value is not None:
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(
                    f"Parameter 'percentile' must be in [0, 100], got {value}"
                )


__all__ = [
    "check_arrays",
    "check_array_1d",
    "check_fitted",
    "check_consistent_length",
    "validate_parameters",
]
