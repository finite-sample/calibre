"""
Utility functions for calibre.

This package provides utility functions for validation, array operations,
and other common tasks used throughout the calibre library.

Modules
-------
validation
    Input validation and parameter checking.
array_ops
    Array operations and transformations.

Examples
--------
>>> from calibre.utils import check_arrays, sort_by_x
>>> import numpy as np
>>>
>>> X = np.array([0.3, 0.1, 0.2])
>>> y = np.array([1, 0, 0])
>>>
>>> # Validate arrays
>>> X, y = check_arrays(X, y)
>>>
>>> # Sort by X
>>> idx, X_sorted, y_sorted = sort_by_x(X, y)
>>> print(X_sorted)
[0.1 0.2 0.3]
"""

from __future__ import annotations

# Import from array_ops module
from .array_ops import (
    clip_to_range,
    ensure_1d,
    find_unique_sorted,
    group_by_value,
    interpolate_monotonic,
    restore_order,
    sort_by_x,
)

# Import from validation module
from .validation import (
    check_array_1d,
    check_arrays,
    check_consistent_length,
    check_fitted,
    validate_parameters,
)

__all__ = [
    # Validation
    "check_arrays",
    "check_array_1d",
    "check_fitted",
    "check_consistent_length",
    "validate_parameters",
    # Array operations
    "sort_by_x",
    "clip_to_range",
    "ensure_1d",
    "restore_order",
    "find_unique_sorted",
    "group_by_value",
    "interpolate_monotonic",
]

# No version needed here - use main package version from calibre.__version__
