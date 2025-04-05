"""
Evaluation metrics for calibration.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.utils import check_array

def mean_calibration_error(y_true, y_pred):
    """
    Calculate the mean calibration error.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values (0 or 1 for binary classification)
    y_pred : array-like
        Predicted probabilities
        
    Returns:
    --------
    mce : float
        Mean calibration error
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    
    # Ensure inputs have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred should have the same shape")
    
    # Simple mean absolute difference between predictions and outcomes
    return np.mean(np.abs(y_pred - y_true))


def binned_calibration_error(y_true, y_pred, x=None, n_bins=10):
    """
    Calculate binned calibration error.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    x : array-like, optional
        Input features for binning. If None, y_pred is used for binning.
    n_bins : int
        Number of bins
        
    Returns:
    --------
    bce : float
        Binned calibration error
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
    
    # Create bins and assign each point to a bin
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    bin_ids = np.digitize(x, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)  # Ensure valid bin indices
    
    # Calculate error for each bin
    error = 0
    valid_bins = 0
    for i in range(n_bins):
        bin_mask = bin_ids == i
        if np.any(bin_mask):
            avg_pred = np.mean(y_pred[bin_mask])
            avg_true = np.mean(y_true[bin_mask])
            error += (avg_pred - avg_true) ** 2
            valid_bins += 1
    
    # Return root mean squared error across bins
    if valid_bins > 0:
        return np.sqrt(error / valid_bins)
    else:
        return 0.0

def correlation_metrics(y_true, y_pred, x=None, y_orig=None):
    """
    Calculate Spearman's correlation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted/calibrated values
    x : array-like, optional
        Input features
    y_orig : array-like, optional
        Original uncalibrated predictions
        
    Returns:
    --------
    correlations : dict
        Dictionary of correlation metrics
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    
    results = {
        'spearman_corr_to_y_true': spearmanr(y_true, y_pred).correlation
    }
    
    if y_orig is not None:
        y_orig = check_array(y_orig, ensure_2d=False)
        results['spearman_corr_to_y_orig'] = spearmanr(y_orig, y_pred).correlation
    
    if x is not None:
        x = check_array(x, ensure_2d=False)
        results['spearman_corr_to_x'] = spearmanr(x, y_pred).correlation
    
    return results


def unique_value_counts(y_pred, y_orig=None, precision=6):
    """
    Count unique values in predictions.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted/calibrated values
    y_orig : array-like, optional
        Original uncalibrated predictions
    precision : int
        Decimal precision for rounding
        
    Returns:
    --------
    counts : dict
        Dictionary with counts of unique values
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    
    results = {
        'n_unique_y_pred': len(np.unique(np.round(y_pred, precision)))
    }
    
    if y_orig is not None:
        y_orig = check_array(y_orig, ensure_2d=False)
        results['n_unique_y_orig'] = len(np.unique(np.round(y_orig, precision)))
        results['unique_value_ratio'] = results['n_unique_y_pred'] / max(1, results['n_unique_y_orig'])
    
    return results