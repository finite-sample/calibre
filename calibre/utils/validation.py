"""Input validation utilities.

This module provides functions for validating and checking input arrays
to ensure they meet the requirements for calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_array


def _validate_probability_vector(y_pred: np.ndarray) -> np.ndarray:
    """Validate one non-empty vector of finite forecast probabilities."""
    raw = np.asarray(y_pred)
    if raw.ndim != 1:
        raise ValueError("y_pred must be one-dimensional")
    if raw.size == 0:
        raise ValueError("y_pred must not be empty")
    try:
        pred = raw.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("y_pred must be numeric") from error
    if not np.all(np.isfinite(pred)) or not np.all((pred >= 0.0) & (pred <= 1.0)):
        raise ValueError("y_pred must contain finite probabilities in [0, 1]")
    return pred


def _validate_binary_probability_metric_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate binary outcomes, probabilities, and evaluation weights."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    try:
        true = true.astype(float, copy=False)
        pred = pred.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("y_true and y_pred must be numeric") from error

    if sample_weight is None:
        weight = np.ones(true.size, dtype=float)
    else:
        raw_weight = np.asarray(sample_weight)
        if raw_weight.ndim != 1:
            raise ValueError("sample_weight must be one-dimensional")
        if raw_weight.shape != true.shape:
            raise ValueError("sample_weight must have the same shape as y_true")
        try:
            weight = raw_weight.astype(float, copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError("sample_weight must be numeric") from error
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values")
        if not np.any(weight > 0.0):
            raise ValueError("sample_weight must contain at least one positive weight")

    active = weight > 0.0
    true = true[active]
    pred = pred[active]
    weight = weight[active]
    if not np.all(np.isfinite(true)) or not np.all((true == 0.0) | (true == 1.0)):
        raise ValueError("y_true must contain binary outcomes in {0, 1}")
    if not np.all(np.isfinite(pred)) or not np.all((pred >= 0.0) & (pred <= 1.0)):
        raise ValueError("y_pred must contain finite probabilities in [0, 1]")
    return true, pred, weight


def _as_float_1d(a: np.ndarray, name: str) -> np.ndarray:
    """Validate an array and return it as a finite 1-D ``float64`` array.

    Wrapped for two reasons. sklearn's ``check_array`` preserves an integer dtype,
    and integer targets are a trap for any estimator that averages labels: pooling
    a 0 and a 1 into an int array stores 0, not 0.5. And its stubs type
    ``ensure_all_finite`` as ``bool`` even though the documented API also accepts
    ``"allow-nan"``, so the call is funnelled through one place.

    Args:
        a: Array-like input.
        name: Name used in validation errors.

    Returns:
        ndarray: A 1-D float64 array.

    Raises:
        ValueError: If the input is not one-dimensional and finite.
    """
    checked = check_array(
        a,
        ensure_2d=False,
        ensure_all_finite=False,
        dtype="numeric",
    )
    result = np.asarray(checked, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def check_arrays(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check and validate input arrays for calibration.

    This function ensures that X and y are valid numpy arrays with
    compatible shapes and no invalid values.

    Args:
        X: The input predictions/probabilities.
        y: The target values/labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (validated_X, validated_y).
            validated_X validated_y

    Raises:
        ValueError: If arrays are empty or have incompatible lengths.

    Notes:
        Both arrays are returned as ``float64``. sklearn's ``check_array`` preserves an
        integer dtype, and integer targets are a trap for any estimator that averages
        labels: pooling a 0 and a 1 into an int array stores 0, not 0.5.

    Examples:
        >>> import numpy as np
        >>> from calibre.utils.validation import check_arrays
        >>>
        >>> X = np.array([0.1, 0.2, 0.3])
        >>> y = np.array([0, 1, 1])
        >>> X_checked, y_checked = check_arrays(X, y)
        >>> print(X_checked.shape, y_checked.shape)
        (3,) (3,)
        >>> y_checked.dtype                       # integer labels are widened
        dtype('float64')
    """
    X = _as_float_1d(X, "X")
    y = _as_float_1d(y, "y")

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
    """Check that an array is 1-dimensional.

    Args:
        X: The array to check.
        name: Name of the array for error messages.

    Returns:
        Validated 1D array.

    Raises:
        ValueError: If array is not 1-dimensional or is empty.

    Examples:
        >>> import numpy as np
        >>> from calibre.utils.validation import check_array_1d
        >>>
        >>> X = np.array([0.1, 0.2, 0.3])
        >>> X_checked = check_array_1d(X)
        >>> print(X_checked.shape)
        (3,)
    """
    X = _as_float_1d(X, name)

    if len(X) == 0:
        raise ValueError(f"Array '{name}' cannot be empty")

    return X


def check_fitted(calibrator: object, attributes: list[str] | None = None) -> None:
    """Check if a calibrator has been fitted.

    Args:
        calibrator: The calibrator to check.
        attributes: List of attribute names that should exist if fitted. If
            None, checks for common fitted attributes.

    Raises:
        NotFittedError: If the calibrator has not been fitted.

    Examples:
        >>> from calibre import IsotonicCalibrator
        >>> from calibre.utils.validation import check_fitted
        >>>
        >>> cal = IsotonicCalibrator()
        >>> try:
        ...     check_fitted(cal)
        ... except Exception as e:
        ...     print("Not fitted:", e)
        Not fitted: IsotonicCalibrator must be fitted...Call fit(X, y) first.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    if attributes is not None and any(
        not hasattr(calibrator, attr) or getattr(calibrator, attr) is None
        for attr in attributes
    ):
        raise NotFittedError(
            f"{calibrator.__class__.__name__} must be fitted before transform. "
            "Call fit(X, y) first."
        )

    try:
        check_is_fitted(calibrator, attributes=attributes)
    except (NotFittedError, TypeError) as exc:
        raise NotFittedError(
            f"{calibrator.__class__.__name__} must be fitted before transform. "
            "Call fit(X, y) first."
        ) from exc


def check_consistent_length(*arrays: np.ndarray) -> None:
    """Check that all arrays have consistent first dimension.

    Args:
        *arrays: Arrays to check for consistent length.

    Raises:
        ValueError: If arrays have inconsistent lengths.

    Examples:
        >>> import numpy as np
        >>> from calibre.utils.validation import check_consistent_length
        >>>
        >>> X = np.array([0.1, 0.2, 0.3])
        >>> y = np.array([0, 1, 1])
        >>> check_consistent_length(X, y)  # No error
        >>>
        >>> z = np.array([0, 1])  # Different length
        >>> check_consistent_length(X, z)
        Traceback (most recent call last):
            ...
        ValueError: Inconsistent array lengths: [3, 2]. All arrays must have
        the same length.
    """
    lengths = [len(X) for X in arrays if X is not None]

    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent array lengths: {lengths}. "
            f"All arrays must have the same length."
        )


def validate_parameters(**params: object) -> None:
    """Validate common calibrator parameters.

    Args:
        **params: Parameter names and values to validate.

    Raises:
        ValueError: If any parameter is invalid.

    Examples:
        >>> from calibre.utils.validation import validate_parameters
        >>>
        >>> validate_parameters(alpha=0.1, n_bootstraps=100)  # OK
        >>>
        >>> validate_parameters(alpha=-0.5)  # Negative
        Traceback (most recent call last):
            ...
        ValueError: Parameter 'alpha' must be non-negative, got -0.5
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

        elif (
            name in ["window_length", "min_window"]
            and value is not None
            and (not isinstance(value, int) or value < 3)
        ):
            raise ValueError(f"Parameter '{name}' must be an integer >= 3, got {value}")

        elif (
            name == "percentile"
            and value is not None
            and (not isinstance(value, (int, float)) or not 0 <= value <= 100)
        ):
            raise ValueError(f"Parameter 'percentile' must be in [0, 100], got {value}")


__all__ = [
    "check_array_1d",
    "check_arrays",
    "check_consistent_length",
    "check_fitted",
    "validate_parameters",
]
