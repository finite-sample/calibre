"""
Implementation of calibration techniques for model probability calibration.

This module provides various algorithms for calibrating model predictions,
with a focus on monotonic and nearly-monotonic calibration methods.
"""
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

class BaseCalibrator(BaseEstimator, TransformerMixin):
    """Base class for all calibrators.
    
    All calibrator classes should inherit from this base class to ensure
    consistent API and functionality.
    """
    
    def fit(self, X, y=None):
        """Fit the calibrator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
        y : array-like of shape (n_samples,), default=None
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError
        
    def transform(self, X):
        """Apply calibration to new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        raise NotImplementedError
        
    def fit_transform(self, X, y=None):
        """Fit and then transform.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
        y : array-like of shape (n_samples,), default=None
            The target values.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        return self.fit(X, y).transform(X)


def check_arrays(X, y):
    """Ensure inputs are numpy arrays of the correct shape.
    
    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Target values.
        
    Returns
    -------
    X : numpy.ndarray
        Input features as a 1D numpy array.
    y : numpy.ndarray
        Target values as a 1D numpy array.
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
    
    return X, y


def sort_by_x(X, y):
    """Sort X and y by X values and return sorting information.
    
    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Target values.
        
    Returns
    -------
    sort_idx : numpy.ndarray
        Indices that would sort X.
    X_sorted : numpy.ndarray
        Sorted X values.
    y_sorted : numpy.ndarray
        Y values sorted by X.
    """
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    return sort_idx, X_sorted, y_sorted


class NearlyIsotonicRegression(BaseCalibrator):
    """Nearly-isotonic regression for flexible monotonic calibration.
    
    This calibrator implements nearly-isotonic regression, which relaxes the
    strict monotonicity constraint of standard isotonic regression by penalizing
    rather than prohibiting violations. This allows for a more flexible fit
    while still maintaining a generally monotonic trend.
    
    Parameters
    ----------
    lam : float, default=1.0
        Regularization parameter controlling the strength of monotonicity constraint.
        Higher values enforce stricter monotonicity.
    method : {'cvx', 'path'}, default='cvx'
        Method to use for solving the optimization problem:
        - 'cvx': Uses convex optimization with CVXPY
        - 'path': Uses a path algorithm similar to the original nearly-isotonic paper
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.
    
    Notes
    -----
    Nearly-isotonic regression solves the following optimization problem:
    
        minimize sum((y_i - beta_i)^2) + lambda * sum(max(0, beta_i - beta_{i+1}))
        
    This formulation penalizes violations of monotonicity proportionally to their magnitude,
    allowing small violations when they significantly improve the fit.
    """
    
    def __init__(self, lam=1.0, method='cvx'):
        self.lam = lam
        self.method = method
    
    def fit(self, X, y):
        """Fit the nearly-isotonic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y)
        self.X_ = X
        self.y_ = y
        
        return self
    
    def transform(self, X):
        """Apply nearly-isotonic calibration to new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        X = np.asarray(X).ravel()
        
        if self.method == 'cvx':
            return self._transform_cvx(X)
        elif self.method == 'path':
            return self._transform_path(X)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'cvx' or 'path'.")
    
    def _transform_cvx(self, X):
        """Implement nearly-isotonic regression using convex optimization."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        
        # Define variables
        beta = cp.Variable(len(y_sorted))
        
        # Penalty for non-monotonicity: sum of positive parts of decreases
        monotonicity_penalty = cp.sum(cp.maximum(0, beta[:-1] - beta[1:]))
        
        # Objective: minimize squared error + lambda * monotonicity penalty
        obj = cp.Minimize(cp.sum_squares(beta - y_sorted) + self.lam * monotonicity_penalty)
        
        # Create and solve the problem
        prob = cp.Problem(obj)
        
        try:
            prob.solve(solver=cp.OSQP, polishing=True)
            
            # Check if solution is found and is optimal
            if prob.status in ["optimal", "optimal_inaccurate"]:
                # Create interpolation function based on sorted values
                cal_func = interp1d(
                    X_sorted, 
                    beta.value, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=(beta.value[0], beta.value[-1])
                )
                
                # Apply interpolation to get values at X points
                return cal_func(X)
            
        except Exception as e:
            print(f"Optimization failed: {e}")
        
        # Fallback to standard isotonic regression if optimization fails
        print("Falling back to standard isotonic regression")
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(self.X_, self.y_)
        return ir.transform(X)
    
    def _transform_path(self, X):
        """Implement nearly-isotonic regression using a path algorithm."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        n = len(y_sorted)
        
        # Initialize solution with original values
        beta = y_sorted.copy()
        
        # Initialize groups and number of groups
        groups = [[i] for i in range(n)]
        
        # Initialize current lambda
        lambda_curr = 0
        
        while True:
            # Compute collision times
            collisions = []
            
            for i in range(len(groups) - 1):
                g1 = groups[i]
                g2 = groups[i + 1]
                
                # Calculate average values for each group
                avg1 = np.mean([beta[j] for j in g1])
                avg2 = np.mean([beta[j] for j in g2])
                
                # Check if collision will occur (if first group has higher value)
                if avg1 > avg2:
                    # Calculate collision time
                    t = avg1 - avg2
                    collisions.append((i, t))
                else:
                    # No collision will occur
                    collisions.append((i, np.inf))
            
            # Check termination condition
            if all(t[1] > self.lam - lambda_curr for t in collisions):
                break
            
            # Find minimum collision time
            valid_times = [(i, t) for i, t in collisions if t < np.inf]
            if not valid_times:
                break
                
            idx, t_min = min(valid_times, key=lambda x: x[1])
            
            # Compute new lambda value (critical point)
            lambda_star = lambda_curr + t_min
            
            # Check if we've exceeded lambda
            if lambda_star > self.lam:
                break
                
            # Update current lambda
            lambda_curr = lambda_star
            
            # Merge groups
            new_group = groups[idx] + groups[idx + 1]
            avg = np.mean([beta[j] for j in new_group])
            for j in new_group:
                beta[j] = avg
            
            groups = groups[:idx] + [new_group] + groups[idx + 2:]
        
        # Create interpolation function based on sorted values
        cal_func = interp1d(
            X_sorted, 
            beta, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(beta[0], beta[-1])
        )
        
        # Apply interpolation to get values at X points
        return cal_func(X)


class ISplineCalibrator(BaseCalibrator):
    """I-Spline calibration with cross-validation.
    
    This calibrator uses monotonic I-splines with non-negative coefficients
    to ensure monotonicity while providing a smooth calibration function.
    Cross-validation is used to find the best model.
    
    Parameters
    ----------
    n_splines : int, default=10
        Number of spline basis functions.
    degree : int, default=3
        Polynomial degree of spline basis functions.
    cv : int, default=5
        Number of cross-validation folds.
    
    Attributes
    ----------
    spline_ : SplineTransformer
        Fitted spline transformer.
    model_ : LinearRegression
        Fitted linear model with non-negative coefficients.
    """
    
    def __init__(self, n_splines=10, degree=3, cv=5):
        self.n_splines = n_splines
        self.degree = degree
        self.cv = cv
    
    def fit(self, X, y):
        """Fit the I-Spline calibration model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y)
        
        # Reshape X to 2D if needed
        X_2d = np.array(X).reshape(-1, 1)
        
        # Create spline transformer with monotonicity constraints
        spline = SplineTransformer(
            n_knots=self.n_splines,
            degree=self.degree,
            extrapolation='constant',
            include_bias=True
        )
        
        # Perform cross-validation to find the best model
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        best_score = -np.inf
        best_model = None
        
        for train_idx, val_idx in kf.split(X_2d):
            X_train, y_train = X_2d[train_idx], y[train_idx]
            X_val, y_val = X_2d[val_idx], y[val_idx]
            
            # Fit spline transformer
            X_train_spline = spline.fit_transform(X_train)
            
            # Fit linear model with non-negative coefficients (monotonicity constraint)
            model = LinearRegression(positive=True, fit_intercept=True)
            model.fit(X_train_spline, y_train)
            
            # Evaluate on validation set
            X_val_spline = spline.transform(X_val)
            score = model.score(X_val_spline, y_val)
            
            if score > best_score:
                best_score = score
                best_model = (spline, model)
        
        # If no best model was found, use simple isotonic regression
        if best_model is None:
            self.fallback_ = IsotonicRegression(out_of_bounds='clip')
            self.fallback_.fit(X, y)
            self.spline_ = None
            self.model_ = None
        else:
            self.spline_, self.model_ = best_model
            self.fallback_ = None
        
        return self
    
    def transform(self, X):
        """Apply I-Spline calibration to new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        X = np.asarray(X).ravel()
        X_2d = X.reshape(-1, 1)
        
        if self.fallback_ is not None:
            return self.fallback_.transform(X)
        
        X_spline = self.spline_.transform(X_2d)
        return self.model_.predict(X_spline)


class RelaxedPAVA(BaseCalibrator):
    """Relaxed Pool Adjacent Violators Algorithm (PAVA) for calibration.
    
    This calibrator implements a relaxed version of PAVA that allows small
    monotonicity violations up to a threshold determined by the percentile
    of differences between adjacent sorted points.
    
    Parameters
    ----------
    percentile : float, default=10
        Percentile of absolute differences to use as threshold.
        Lower values enforce stricter monotonicity.
    adaptive : bool, default=True
        Whether to use the adaptive implementation (recommended) or the
        block-merging implementation.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.
    """
    
    def __init__(self, percentile=10, adaptive=True):
        self.percentile = percentile
        self.adaptive = adaptive
    
    def fit(self, X, y):
        """Fit the relaxed PAVA model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y)
        self.X_ = X
        self.y_ = y
        
        return self
    
    def transform(self, X):
        """Apply relaxed PAVA calibration to new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        X = np.asarray(X).ravel()
        
        # Apply relaxed PAVA to get calibrated values for training data
        if self.adaptive:
            y_calibrated = self._relaxed_pava_adaptive()
        else:
            y_calibrated = self._relaxed_pava_block()
        
        # Create interpolation function
        cal_func = interp1d(
            self.X_, 
            y_calibrated, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(np.min(y_calibrated), np.max(y_calibrated))
        )
        
        # Apply interpolation to get values at X points
        return cal_func(X)
    
    def _relaxed_pava_adaptive(self):
        """Implement relaxed PAVA with adaptive threshold."""
        X, y = self.X_, self.y_
        
        # Sort by X values
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Calculate absolute differences between adjacent points
        diffs = np.abs(np.diff(y_sorted))
        
        # Handle edge case: if only one data point or all points have same value
        if len(diffs) == 0 or np.all(diffs == 0):
            return y.copy()
        
        # Find relaxation threshold based on percentile of differences
        relaxation = np.percentile(diffs, self.percentile)
        
        n = len(y_sorted)
        y_smoothed = y_sorted.copy()
        
        # Iteratively pool adjacent violators that exceed the relaxation threshold
        for iteration in range(n):  # At most n iterations needed
            changed = False
            for i in range(n-1):
                # Check if monotonicity is violated by more than the threshold
                if y_smoothed[i] > y_smoothed[i+1] + relaxation:
                    # Average adjacent violators
                    avg = (y_smoothed[i] + y_smoothed[i+1]) / 2
                    y_smoothed[i] = avg
                    y_smoothed[i+1] = avg
                    changed = True
            
            # If no changes in this iteration, we've converged
            if not changed:
                break
        
        # Restore original order
        y_result = np.empty_like(y)
        y_result[sort_idx] = y_smoothed
        
        return y_result
    
    def _relaxed_pava_block(self):
        """Implement relaxed PAVA with block merging approach."""
        X, y = self.X_, self.y_
        
        # Sort by X values
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        n = len(y_sorted)
        
        # Calculate threshold based on the percentile of sorted differences
        diffs = np.abs(np.diff(y_sorted))
        if len(diffs) > 0:
            epsilon = np.percentile(diffs, self.percentile)
        else:
            epsilon = 0.0
        
        # Apply modified PAVA with epsilon threshold
        y_fit = y_sorted.copy()
        
        # Track indices of blocks that have been averaged
        blocks = [[i] for i in range(n)]
        
        i = 0
        while i < n - 1:
            if y_fit[i] > y_fit[i + 1] + epsilon:
                # Violation detected, merge blocks
                block1 = next(b for b in blocks if i in b)
                block2 = next(b for b in blocks if (i + 1) in b)
                
                # If blocks are different, merge them
                if block1 != block2:
                    # Calculate weighted average
                    merged_block = block1 + block2
                    merged_avg = sum(y_sorted[j] for j in merged_block) / len(merged_block)
                    
                    # Update values
                    for j in merged_block:
                        y_fit[j] = merged_avg
                    
                    # Update block structure
                    blocks.remove(block1)
                    blocks.remove(block2)
                    blocks.append(merged_block)
                    
                    # Move back to check if new violations occur
                    i = min(merged_block)
                else:
                    i += 1
            else:
                i += 1
        
        # Restore original order
        y_result = np.empty_like(y_fit)
        y_result[sort_idx] = y_fit
        
        return y_result


class RegularizedIsotonicRegression(BaseCalibrator):
    """Regularized isotonic regression with L2 regularization.
    
    This calibrator adds L2 regularization to standard isotonic regression to
    prevent overfitting and produce smoother calibration curves.
    
    Parameters
    ----------
    alpha : float, default=0.1
        Regularization strength. Higher values result in smoother curves.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.
    """
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def fit(self, X, y):
        """Fit the regularized isotonic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y)
        self.X_ = X
        self.y_ = y
        
        return self
    
    def transform(self, X):
        """Apply regularized isotonic calibration to new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        X = np.asarray(X).ravel()
        
        # Calculate calibration function
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        
        # Define variables
        beta = cp.Variable(len(y_sorted))
        
        # Monotonicity constraints: each value should be greater than or equal to the previous
        constraints = [beta[i] <= beta[i+1] for i in range(len(beta)-1)]
        
        # Objective: minimize squared error + alpha * L2 regularization
        obj = cp.Minimize(cp.sum_squares(beta - y_sorted) + self.alpha * cp.sum_squares(beta))
        
        # Create and solve the problem
        prob = cp.Problem(obj, constraints)
        
        try:
            # Solve the problem
            prob.solve(solver=cp.OSQP, polishing=True)
            
            # Check if solution is found and is optimal
            if prob.status in ["optimal", "optimal_inaccurate"]:
                # Create interpolation function
                cal_func = interp1d(
                    X_sorted, 
                    beta.value, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=(beta.value[0], beta.value[-1])
                )
                
                # Apply interpolation to get values at X points
                return cal_func(X)
            
        except Exception as e:
            print(f"Regularized isotonic optimization failed: {e}")
        
        # Fallback to standard isotonic regression
        print("Falling back to standard isotonic regression")
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(self.X_, self.y_)
        return ir.transform(X)


class SmoothedIsotonicRegression(BaseCalibrator):
    """Locally smoothed isotonic regression.
    
    This calibrator applies standard isotonic regression and then smooths
    the result using a Savitzky-Golay filter, which preserves the monotonicity
    properties while reducing jaggedness.
    
    Parameters
    ----------
    window_length : int or None, default=None
        Window length for Savitzky-Golay filter. Should be odd.
        If None, window_length is set to max(5, len(X)//10)
    poly_order : int, default=3
        Polynomial order for the Savitzky-Golay filter.
        Must be less than window_length.
    interp_method : str, default='linear'
        Interpolation method to use ('linear', 'cubic', etc.)
    adaptive : bool, default=False
        Whether to use adaptive window sizes based on local density.
    min_window : int, default=5
        Minimum window length when using adaptive=True.
    max_window : int or None, default=None
        Maximum window length when using adaptive=True.
        If None, max_window is set to len(X)//5.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The training input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.
    """
    
    def __init__(self, window_length=None, poly_order=3, interp_method='linear', 
                 adaptive=False, min_window=5, max_window=None):
        self.window_length = window_length
        self.poly_order = poly_order
        self.interp_method = interp_method
        self.adaptive = adaptive
        self.min_window = min_window
        self.max_window = max_window
    
    def fit(self, X, y):
        """Fit the smoothed isotonic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y)
        self.X_ = X
        self.y_ = y
        
        return self
    
    def transform(self, X):
        """Apply smoothed isotonic calibration to new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The values to be calibrated.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated values.
        """
        X = np.asarray(X).ravel()
        
        # Calculate smoothed calibration values
        if self.adaptive:
            y_smoothed = self._transform_adaptive()
        else:
            y_smoothed = self._transform_fixed()
        
        # Create interpolation function
        cal_func = interp1d(
            self.X_, 
            y_smoothed, 
            kind=self.interp_method, 
            bounds_error=False, 
            fill_value=(np.min(y_smoothed), np.max(y_smoothed))
        )
        
        # Apply interpolation to get values at X points
        return cal_func(X)
    
    def _transform_fixed(self):
        """Implement smoothed isotonic regression with fixed window size."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        
        # Apply standard isotonic regression
        ir = IsotonicRegression(out_of_bounds='clip')
        y_iso = ir.fit_transform(X_sorted, y_sorted)
        
        # Determine window length if not provided
        n = len(X_sorted)
        window_length = self.window_length
        if window_length is None:
            window_length = max(5, n // 10)
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure window_length is not too large
        window_length = min(window_length, n - (n % 2 == 0))
        
        # Ensure poly_order is valid
        poly_order = min(self.poly_order, window_length - 1)
        
        # Apply Savitzky-Golay filter for smoothing if we have enough points
        if n >= window_length:
            try:
                y_smoothed = savgol_filter(y_iso, window_length, poly_order)
                
                # The smoothing might violate monotonicity, so we need to correct it
                for i in range(1, len(y_smoothed)):
                    if y_smoothed[i] < y_smoothed[i-1]:
                        y_smoothed[i] = y_smoothed[i-1]
                        
            except Exception as e:
                print(f"Savitzky-Golay smoothing failed: {e}")
                y_smoothed = y_iso
        else:
            # Not enough points for smoothing
            y_smoothed = y_iso
        
        # Restore original order
        y_result = np.empty_like(y_smoothed)
        y_result[order] = y_smoothed
        
        return y_result
    
    def _transform_adaptive(self):
        """Implement smoothed isotonic regression with adaptive window size."""
        order, X_sorted, y_sorted = sort_by_x(self.X_, self.y_)
        
        # Apply standard isotonic regression
        ir = IsotonicRegression(out_of_bounds='clip')
        y_iso = ir.fit_transform(X_sorted, y_sorted)
        
        n = len(X_sorted)
        
        # Set default max_window if not provided
        max_window = self.max_window
        if max_window is None:
            max_window = max(self.min_window, n // 5)
            # Ensure it's odd
            if max_window % 2 == 0:
                max_window += 1
        
        # Initialize result array
        y_smoothed = np.array(y_iso)
        
        # Not enough points for smoothing
        if n <= 1:
            return y_iso
            
        # Normalize x to [0,1] range for density calculation
        x_range = X_sorted[-1] - X_sorted[0]
        if x_range <= 0:
            return y_iso
            
        x_norm = (X_sorted - X_sorted[0]) / x_range
        
        # For each point, apply adaptive smoothing
        for i in range(n):
            # Calculate distances to all other points
            distances = np.abs(x_norm[i] - x_norm)
            
            # Determine optimal window size based on local density
            window_size = self._find_optimal_window_size(
                distances, self.min_window, max_window, n
            )
            
            # Apply smoothing if window is large enough
            if window_size >= 5:  # Minimum required for savgol_filter
                y_smoothed[i] = self._apply_local_smoothing(
                    i, window_size, X_sorted, y_iso, n
                )
        
        # Ensure monotonicity is preserved
        for i in range(1, len(y_smoothed)):
            if y_smoothed[i] < y_smoothed[i-1]:
                y_smoothed[i] = y_smoothed[i-1]
        
        # Restore original order
        y_result = np.empty_like(y_smoothed)
        y_result[order] = y_smoothed
        
        return y_result
    
    def _find_optimal_window_size(self, distances, min_window, max_window, n):
        """Find the optimal window size based on local point density."""
        window_size = min_window
        
        for w in range(min_window, max_window + 2, 2):
            # Calculate normalized window width
            width = w / n
            
            # Count points within this width
            count = np.sum(distances <= width)
            
            # If we have enough points, use this window size
            if count >= w:
                window_size = w
            else:
                break
                
        return window_size
    
    def _apply_local_smoothing(self, i, window_size, X_sorted, y_iso, n):
        """Apply local smoothing around point i with specified window size."""
        # Create a local window around point i
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        
        # If we don't have enough points in the window
        if end_idx - start_idx < 5:
            return y_iso[i]
            
        # Create temp arrays for local smoothing
        x_local = X_sorted[start_idx:end_idx]
        y_local = y_iso[start_idx:end_idx]
        
        # Ensure window length is odd
        window_len = len(x_local)
        if window_len % 2 == 0:
            window_len -= 1
        
        if window_len < 5:
            return y_iso[i]
            
        # Apply Savitzky-Golay filter
        poly_ord = min(self.poly_order, window_len - 1)
        try:
            y_local_smooth = savgol_filter(y_local, window_len, poly_ord)
            
            # Get the smoothed value for the center point
            local_idx = i - start_idx
            if 0 <= local_idx < len(y_local_smooth):
                return y_local_smooth[local_idx]
        except Exception:
            pass
            
        # Fall back to original value if smoothing fails
        return y_iso[i]
