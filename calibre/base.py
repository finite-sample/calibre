"""
Base classes and interfaces for calibration.

This module provides the foundational classes that all calibrators inherit from,
as well as optional mixins for additional functionality like diagnostics.
Supports a modular architecture with concrete implementations in the
calibrators package.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class BaseCalibrator(BaseEstimator, TransformerMixin):
    """Base class for all calibrators.

    All calibrator classes should inherit from this base class to ensure
    consistent API and functionality. This follows the scikit-learn transformer
    interface with fit/transform/fit_transform methods.

    Parameters
    ----------
    enable_diagnostics
        Whether to run plateau diagnostics after fitting.

    Notes
    -----
    Subclasses must implement the fit() and transform() methods.
    The fit_transform() method is provided by default.

    Examples
    --------
    >>> import numpy as np
    >>> from calibre import BaseCalibrator
    >>>
    >>> class SimpleCalibrator(BaseCalibrator):
    ...     def __init__(self, enable_diagnostics=False):
    ...         super().__init__(enable_diagnostics=enable_diagnostics)
    ...     def fit(self, X, y):
    ...         self.mean_ = np.mean(y)
    ...         return self
    ...
    ...     def transform(self, X):
    ...         return np.full_like(X, self.mean_)
    >>>
    >>> X = np.array([0.1, 0.3, 0.5])
    >>> y = np.array([0, 1, 1])
    >>>
    >>> cal = SimpleCalibrator()
    >>> cal.fit(X, y)
    >>> cal.transform(X)
    array([0.667, 0.667, 0.667])
    """

    def __init__(self, enable_diagnostics: bool = False) -> None:
        self.enable_diagnostics = enable_diagnostics
        self.diagnostics_: dict | None = None
        self._fit_data_X: np.ndarray | None = None
        self._fit_data_y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseCalibrator:
        """Fit the calibrator.

        This method implements the template method pattern: it handles
        data storage and diagnostics, while delegating the actual fitting
        logic to the abstract _fit_impl() method that subclasses must implement.

        Parameters
        ----------
        X
            The values to be calibrated (e.g., predicted probabilities).
        y
            The target values (e.g., true labels).

        Returns
        -------
        BaseCalibrator
            Returns self for method chaining.
        """
        # Store fit data for potential diagnostics
        self._fit_data_X = X
        self._fit_data_y = y

        # Delegate actual fitting to subclass implementation
        self._fit_impl(X, y)

        # Run diagnostics if enabled
        self._run_diagnostics()

        return self

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Implement the actual fitting logic.

        This abstract method must be implemented by subclasses to perform
        the calibration-specific fitting logic. The base fit() method handles
        data storage, diagnostics, and return value automatically.

        Parameters
        ----------
        X
            The values to be calibrated (e.g., predicted probabilities).
        y
            The target values (e.g., true labels).
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        Notes
        -----
        Subclasses should implement this method instead of overriding fit().
        The fit() method in the base class ensures consistent behavior
        for diagnostics and data storage.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the _fit_impl() method"
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply calibration to new data.

        Parameters
        ----------
        X
            The values to be calibrated.

        Returns
        -------
        Calibrated values.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the transform() method"
        )

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the calibrator and then transform the data.

        This is a convenience method that combines fit() and transform()
        in a single call. The default implementation simply calls fit()
        followed by transform().

        Parameters
        ----------
        X
            The values to be calibrated.
        y
            The target values.

        Returns
        -------
        Calibrated values.

        Examples
        --------
        >>> import numpy as np
        >>> from calibre import IsotonicCalibrator
        >>> X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>> cal = IsotonicCalibrator()
        >>> X_calibrated = cal.fit_transform(X, y)
        """
        return self.fit(X, y).transform(X)

    def _run_diagnostics(self) -> None:
        """
        Run diagnostic analysis on the fitted calibrator.

        This method is called automatically after fitting if enable_diagnostics=True.
        It calls the standalone diagnostic functions from the diagnostics module.
        """
        if (
            not self.enable_diagnostics
            or self._fit_data_X is None
            or self._fit_data_y is None
        ):
            return

        from .diagnostics import run_plateau_diagnostics

        try:
            y_calibrated = self.transform(self._fit_data_X)
            self.diagnostics_ = run_plateau_diagnostics(
                self._fit_data_X, self._fit_data_y, y_calibrated
            )
        except Exception as e:
            logger.warning(f"Diagnostic analysis failed: {e}")
            self.diagnostics_ = None

    def has_diagnostics(self) -> bool:
        """
        Check if diagnostic information is available.

        Returns
        -------
        has_diag : bool
            True if diagnostics have been computed and are available.

        Examples
        --------
        >>> from calibre import IsotonicCalibrator
        >>> import numpy as np
        >>>
        >>> X = np.array([0.1, 0.3, 0.5])
        >>> y = np.array([0, 1, 1])
        >>>
        >>> cal = IsotonicCalibrator(enable_diagnostics=True)
        >>> cal.fit(X, y)
        >>> if cal.has_diagnostics():
        ...     print("Diagnostics available!")
        """
        return self.diagnostics_ is not None

    def get_diagnostics(self) -> dict | None:
        """
        Get diagnostic results.

        Returns
        -------
        dict | None
            Diagnostic results from plateau analysis, or None if diagnostics
            were not computed or are not available.

        Examples
        --------
        >>> from calibre import IsotonicCalibrator
        >>> import numpy as np
        >>>
        >>> X = np.array([0.1, 0.3, 0.5])
        >>> y = np.array([0, 1, 1])
        >>>
        >>> cal = IsotonicCalibrator(enable_diagnostics=True)
        >>> cal.fit(X, y)
        >>> diagnostics = cal.get_diagnostics()
        >>> if diagnostics:
        ...     print(f"Found {diagnostics['n_plateaus']} plateaus")
        """
        return self.diagnostics_

    def diagnostic_summary(self) -> str:
        """
        Get a human-readable summary of diagnostic analysis.

        Returns
        -------
        summary : str
            Human-readable plateau summary.

        Examples
        --------
        >>> from calibre import IsotonicCalibrator
        >>> import numpy as np
        >>>
        >>> X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> y = np.array([0, 0, 1, 1, 1])
        >>>
        >>> cal = IsotonicCalibrator(enable_diagnostics=True)
        >>> cal.fit(X, y)
        >>> print(cal.diagnostic_summary())
        """
        if not self.enable_diagnostics or self.diagnostics_ is None:
            return "Diagnostics not available. Set enable_diagnostics=True to enable."

        if self.diagnostics_["n_plateaus"] == 0:
            return "No plateaus detected in calibration curve."

        lines = [f"Detected {self.diagnostics_['n_plateaus']} plateau(s):"]

        if self.diagnostics_["warnings"]:
            lines.append("\nWarnings:")
            for warning in self.diagnostics_["warnings"]:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)


class MonotonicMixin:
    """Mixin for calibrators that maintain monotonicity.

    This mixin provides utility methods for calibrators that aim to
    preserve or enforce monotonic relationships between inputs and outputs.

    Methods
    -------
    check_monotonicity(y)
        Check if an array is monotonically increasing.
    enforce_monotonicity(y)
        Enforce monotonicity on an array.

    Notes
    -----
    This is a utility mixin that doesn't require any specific attributes.
    It's designed to be mixed in with BaseCalibrator subclasses that
    need monotonicity guarantees.
    """

    @staticmethod
    def check_monotonicity(y: np.ndarray, strict: bool = False) -> bool:
        """Check if an array is monotonically increasing.

        Parameters
        ----------
        y
            Values to check for monotonicity.
        strict
            If True, check for strictly increasing (no equal consecutive values).
            If False, check for non-decreasing (allows equal consecutive values).

        Returns
        -------
        True if the array is monotonic according to the specified criteria.

        Examples
        --------
        >>> import numpy as np
        >>> from calibre.base import MonotonicMixin
        >>>
        >>> y1 = np.array([0.1, 0.2, 0.3, 0.4])
        >>> MonotonicMixin.check_monotonicity(y1)
        True
        >>>
        >>> y2 = np.array([0.1, 0.3, 0.2, 0.4])
        >>> MonotonicMixin.check_monotonicity(y2)
        False
        >>>
        >>> y3 = np.array([0.1, 0.2, 0.2, 0.3])
        >>> MonotonicMixin.check_monotonicity(y3, strict=False)
        True
        >>> MonotonicMixin.check_monotonicity(y3, strict=True)
        False
        """
        y = np.asarray(y)
        if len(y) <= 1:
            return True

        diffs = np.diff(y)

        if strict:
            return bool(np.all(diffs > 0))
        else:
            return bool(np.all(diffs >= 0))

    @staticmethod
    def enforce_monotonicity(y: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Enforce monotonicity on an array.

        This method ensures the array is non-decreasing by replacing any
        value that is less than the previous value with the previous value.

        Parameters
        ----------
        y
            Values to make monotonic.
        inplace
            If True, modify the array in place. Otherwise, return a copy.

        Returns
        -------
        Monotonically increasing version of the input array.

        Examples
        --------
        >>> import numpy as np
        >>> from calibre.base import MonotonicMixin
        >>>
        >>> y = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
        >>> y_mono = MonotonicMixin.enforce_monotonicity(y)
        >>> print(y_mono)
        [0.1 0.3 0.3 0.5 0.5]
        >>>
        >>> # Original array unchanged
        >>> print(y)
        [0.1 0.3 0.2 0.5 0.4]
        """
        if inplace:
            y_result = y
        else:
            y_result = np.asarray(y).copy()

        for i in range(1, len(y_result)):
            if y_result[i] < y_result[i - 1]:
                y_result[i] = y_result[i - 1]

        return y_result


# Module constants for validation and default values
DEFAULT_MIN_WINDOW = 5
DEFAULT_POLY_ORDER = 3
DEFAULT_N_BOOTSTRAPS = 100
DEFAULT_N_SPLITS = 5
MIN_VARIANCE_THRESHOLD = 1e-6
WINDOW_DIVISOR = 10
ADAPTIVE_WINDOW_DIVISOR = 5

__all__ = [
    "BaseCalibrator",
    "MonotonicMixin",
    "DEFAULT_MIN_WINDOW",
    "DEFAULT_POLY_ORDER",
    "DEFAULT_N_BOOTSTRAPS",
    "DEFAULT_N_SPLITS",
    "MIN_VARIANCE_THRESHOLD",
    "WINDOW_DIVISOR",
    "ADAPTIVE_WINDOW_DIVISOR",
]
