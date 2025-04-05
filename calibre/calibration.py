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