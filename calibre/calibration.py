"""
Implementation of calibration techniques.
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

from .utils import check_arrays, sort_by_x

class BaseCalibrator(BaseEstimator, TransformerMixin):
    """Base class for all calibrators."""
    
    def fit(self, X, y=None):
        """Fit the calibrator."""
        raise NotImplementedError
        
    def transform(self, X):
        """Apply calibration to new data."""
        raise NotImplementedError
        
    def fit_transform(self, X, y=None):
        """Fit and then transform."""
        return self.fit(X, y).transform(X)

def nearly_isotonic_opt(x, y, lam=1.0):
    """
    Implementation of nearly isotonic regression using convex optimization.
    
    Parameters:
    -----------
    x : array-like
        Input features (for sorting)
    y : array-like
        Target values to calibrate
    lam : float
        Regularization parameter controlling the strength of monotonicity constraint
        
    Returns:
    --------
    y_calibrated : array-like
        Calibrated values with nearly-isotonic property
    """
    x, y = check_arrays(x, y)
    order, x_sorted, y_sorted = sort_by_x(x, y)
    
    # Define variables
    beta = cp.Variable(len(y_sorted))
    
    # Penalty for non-monotonicity: sum of positive parts of decreases
    monotonicity_penalty = cp.sum(cp.maximum(0, beta[:-1] - beta[1:]))
    
    # Objective: minimize squared error + lambda * monotonicity penalty
    obj = cp.Minimize(cp.sum_squares(beta - y_sorted) + lam * monotonicity_penalty)
    
    # Create and solve the problem
    prob = cp.Problem(obj)
    
    try:
        prob.solve(solver=cp.OSQP, polishing=True)
        
        # Check if solution is found and is optimal
        if prob.status in ["optimal", "optimal_inaccurate"]:
            # Unsort to original x order
            y_calibrated = np.empty_like(y)
            y_calibrated[order] = beta.value
            return y_calibrated
        
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Fallback to original values if optimization fails
    return y


def nearly_isotonic_path(x, y, lam=1.0):
    """
    Implementation of nearly isotonic regression using a path algorithm.
    
    Parameters:
    -----------
    x : array-like
        Input features (for sorting)
    y : array-like
        Target values to calibrate
    lam : float
        Regularization parameter controlling the strength of monotonicity constraint
        
    Returns:
    --------
    y_calibrated : array-like
        Calibrated values with nearly-isotonic property
    """
    x, y = check_arrays(x, y)
    order, x_sorted, y_sorted = sort_by_x(x, y)
    n = len(y_sorted)
    
    # Initialize solution with original values
    beta = y_sorted.copy()
    
    # Initialize groups and number of groups
    groups = [[i] for i in range(n)]
    
    # Function to compute slopes for each group
    # In the nearly isotonic case, slopes are 0 (constant within groups)
    def compute_slopes():
        return [0] * len(groups)
    
    # Function to compute collision times between adjacent groups
    def compute_collisions():
        slopes = compute_slopes()
        collisions = []
        
        for i in range(len(groups) - 1):
            g1 = groups[i]
            g2 = groups[i + 1]
            
            # Calculate average values for each group
            avg1 = np.mean([beta[j] for j in g1])
            avg2 = np.mean([beta[j] for j in g2])
            
            # Calculate slope difference (should be 0 in this case)
            slope_diff = slopes[i] - slopes[i + 1]
            
            # Check if collision will occur (if first group has higher value)
            if avg1 > avg2:
                # Calculate collision time
                if abs(slope_diff) > 1e-10:  # Avoid division by zero
                    t = (avg1 - avg2) / slope_diff
                else:
                    t = avg1 - avg2
                
                collisions.append((i, t))
            else:
                # No collision will occur
                collisions.append((i, np.inf))
                
        return collisions
    
    # Initialize current lambda
    lambda_curr = 0
    
    while True:
        # Compute collision times
        collisions = compute_collisions()
        
        # Check termination condition
        if all(t[1] > lam - lambda_curr for t in collisions):
            break
        
        # Find minimum collision time
        valid_times = [(i, t) for i, t in collisions if t < np.inf]
        if not valid_times:
            break
            
        idx, t_min = min(valid_times, key=lambda x: x[1])
        
        # Compute new lambda value (critical point)
        lambda_star = lambda_curr + t_min
        
        # Check if we've exceeded lambda
        if lambda_star > lam:
            break
            
        # Update current lambda
        lambda_curr = lambda_star
        
        # Merge groups
        new_group = groups[idx] + groups[idx + 1]
        avg = np.mean([beta[j] for j in new_group])
        for j in new_group:
            beta[j] = avg
        
        groups = groups[:idx] + [new_group] + groups[idx + 2:]
    
    # Unsort to original x order
    y_calibrated = np.empty_like(y)
    y_calibrated[order] = beta
    
    return y_calibrated


def ispline_calib(x, y, n_splines=10, degree=3, cv=5):
    """
    Implements I-Spline calibration with cross-validation.
    
    Parameters:
    -----------
    x : array-like
        Input features (predictions to calibrate)
    y : array-like
        Target values
    n_splines : int
        Number of spline basis functions
    degree : int
        Polynomial degree of spline basis functions
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    y_calibrated : array-like
        Calibrated values
    """
    x, y = check_arrays(x, y)
    
    # Reshape x to 2D if needed
    X = np.array(x).reshape(-1, 1)
    
    # Create spline transformer with monotonicity constraints
    spline = SplineTransformer(
        n_knots=n_splines,
        degree=degree,
        extrapolation='constant',
        include_bias=True
    )
    
    # Perform cross-validation to find the best model
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_score = -np.inf
    best_model = None
    
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
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
        ir = IsotonicRegression(out_of_bounds='clip')
        return ir.fit_transform(X.ravel(), y)
    
    # Apply the best model to the entire dataset
    best_spline, best_linear = best_model
    X_spline = best_spline.transform(X)
    y_calibrated = best_linear.predict(X_spline)
    
    return y_calibrated


def relax_pava(x, y, percentile=10):
    """
    Relaxed PAVA: allows small non-monotonic deviations up to a threshold
    determined by the percentile of differences.
    
    Parameters:
    -----------
    x : array-like
        Input features (for sorting)
    y : array-like
        Target values to calibrate
    percentile : float
        Percentile of absolute differences to use as threshold
        
    Returns:
    --------
    y_relaxed : array-like
        Calibrated values with relaxed monotonicity
    """
    x, y = check_arrays(x, y)
    order, x_sorted, y_sorted = sort_by_x(x, y)
    n = len(y_sorted)
    
    # Calculate threshold based on the percentile of sorted differences
    diffs = np.abs(np.diff(y_sorted))
    if len(diffs) > 0:
        epsilon = np.percentile(diffs, percentile)
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
    
    # Unsort to original x order
    y_relaxed = np.empty_like(y_fit)
    y_relaxed[order] = y_fit
    
    return y_relaxed

def regularized_isotonic(x, y, alpha=0.1):
    """
    Implementation of regularized isotonic regression using convex optimization.
    
    This method adds L2 regularization to the standard isotonic regression problem,
    which helps prevent overfitting and produces a smoother calibration curve.
    
    Parameters:
    -----------
    x : array-like
        Input features (for sorting)
    y : array-like
        Target values to calibrate
    alpha : float
        Regularization strength parameter. Higher values result in smoother curves.
        
    Returns:
    --------
    y_calibrated : array-like
        Calibrated values with regularized isotonic property
    """
    x, y = check_arrays(x, y)
    order, x_sorted, y_sorted = sort_by_x(x, y)
    
    # Define variables
    beta = cp.Variable(len(y_sorted))
    
    # Monotonicity constraints: each value should be greater than or equal to the previous
    constraints = [beta[i] <= beta[i+1] for i in range(len(beta)-1)]
    
    # Objective: minimize squared error + alpha * L2 regularization
    obj = cp.Minimize(cp.sum_squares(beta - y_sorted) + alpha * cp.sum_squares(beta))
    
    # Create and solve the problem
    prob = cp.Problem(obj, constraints)
    
    # Solve the problem
    prob.solve(solver=cp.OSQP, polishing=True)
    
    # Check if solution is found and is optimal
    if prob.status in ["optimal", "optimal_inaccurate"]:
        # Unsort to original x order
        y_calibrated = np.empty_like(y)
        y_calibrated[order] = beta.value
        return y_calibrated
    else:
        # Raise an exception if optimization was not successful
        raise ValueError(f"Regularized isotonic optimization failed with status: {prob.status}")

def locally_smoothed_isotonic(x, y, window_length=None, polyorder=3, interp_method='linear'):
    """
    Implementation of locally smoothed isotonic regression.
    
    This method first applies standard isotonic regression and then smooths
    the result using a Savitzky-Golay filter, which preserves the monotonicity
    properties while reducing jaggedness.
    
    Parameters:
    -----------
    x : array-like
        Input features (predictions to calibrate)
    y : array-like
        Target values
    window_length : int or None
        Window length for Savitzky-Golay filter. Should be odd.
        If None, window_length is set to max(5, len(x)//10)
    polyorder : int
        Polynomial order for the Savitzky-Golay filter.
        Must be less than window_length.
    interp_method : str
        Interpolation method to use ('linear', 'cubic', etc.)
        
    Returns:
    --------
    y_calibrated : array-like
        Calibrated values with smoothed isotonic property
    """
    x, y = check_arrays(x, y)
    order, x_sorted, y_sorted = sort_by_x(x, y)
    
    # Apply standard isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    y_iso = ir.fit_transform(x_sorted, y_sorted)
    
    # Determine window length if not provided
    n = len(x_sorted)
    if window_length is None:
        window_length = max(5, n // 10)
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure window_length is not too large
    window_length = min(window_length, n - (n % 2 == 0))
    
    # Ensure polyorder is valid
    polyorder = min(polyorder, window_length - 1)
    
    # Apply Savitzky-Golay filter for smoothing if we have enough points
    if n >= window_length:
        try:
            y_smoothed = savgol_filter(y_iso, window_length, polyorder)
            
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
    
    # Create interpolation function based on smoothed values
    interpolator = interp1d(
        x_sorted, 
        y_smoothed, 
        kind=interp_method, 
        bounds_error=False, 
        fill_value=(y_smoothed[0], y_smoothed[-1])
    )
    
    # Apply interpolation to get values at original x points
    y_calibrated = interpolator(x)
    
    return y_calibrated


def adaptive_smoothed_isotonic(x, y, min_window=5, max_window=None, polyorder=3):
    """
    Implementation of adaptive locally smoothed isotonic regression.
    
    This method applies isotonic regression followed by adaptive smoothing,
    where the window size varies based on local density of points.
    
    Parameters:
    -----------
    x : array-like
        Input features (predictions to calibrate)
    y : array-like
        Target values
    min_window : int
        Minimum window length for smoothing
    max_window : int or None
        Maximum window length for smoothing
        If None, max_window is set to len(x)//5
    polyorder : int
        Polynomial order for the filter
        
    Returns:
    --------
    y_calibrated : array-like
        Calibrated values with adaptively smoothed isotonic property
    """
    x, y = check_arrays(x, y)
    order, x_sorted, y_sorted = sort_by_x(x, y)
    
    # Apply standard isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    y_iso = ir.fit_transform(x_sorted, y_sorted)
    
    n = len(x_sorted)
    
    # Set default max_window if not provided
    if max_window is None:
        max_window = max(min_window, n // 5)
        # Ensure it's odd
        if max_window % 2 == 0:
            max_window += 1
    
    # Initialize result array
    y_smoothed = np.array(y_iso)
    
    # Calculate point density to determine adaptive window size
    if n > 1:
        # Normalize x to [0,1] range
        x_range = x_sorted[-1] - x_sorted[0]
        if x_range > 0:
            x_norm = (x_sorted - x_sorted[0]) / x_range
            
            # For each point, calculate local density
            for i in range(n):
                # Calculate distances to all other points
                distances = np.abs(x_norm[i] - x_norm)
                
                # Count points within different window sizes
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
                
                # Apply local smoothing with adaptive window
                if window_size >= 5:  # Minimum required for savgol_filter
                    # Create a local window around point i
                    half_window = window_size // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(n, i + half_window + 1)
                    
                    # If we have enough points in the window
                    if end_idx - start_idx >= 5:
                        # Create temp arrays for local smoothing
                        x_local = x_sorted[start_idx:end_idx]
                        y_local = y_iso[start_idx:end_idx]
                        
                        # Apply smoothing
                        window_len = len(x_local)
                        if window_len % 2 == 0:
                            window_len -= 1
                        
                        if window_len >= 5:
                            poly_ord = min(polyorder, window_len - 1)
                            y_local_smooth = savgol_filter(y_local, window_len, poly_ord)
                            
                            # Update only the center point
                            local_idx = i - start_idx
                            if 0 <= local_idx < len(y_local_smooth):
                                y_smoothed[i] = y_local_smooth[local_idx]
    
    # Ensure monotonicity is preserved
    for i in range(1, len(y_smoothed)):
        if y_smoothed[i] < y_smoothed[i-1]:
            y_smoothed[i] = y_smoothed[i-1]
    
    # Unsort to original x order
    y_calibrated = np.empty_like(y)
    y_calibrated[order] = y_smoothed
    
    return y_calibrated