"""
Utility functions for the calibre package.
"""
import numpy as np
from sklearn.utils import check_array

def check_arrays(x, y):
    """
    Check and validate input arrays.
    
    Parameters:
    -----------
    x : array-like
        Input features
    y : array-like
        Target values
        
    Returns:
    --------
    x, y : array-like
        Validated arrays
    """
    x = check_array(x, ensure_2d=False)
    y = check_array(y, ensure_2d=False)
    
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length")
    
    return x, y

def sort_by_x(x, y):
    """
    Sort arrays by x values.
    
    Parameters:
    -----------
    x : array-like
        Input features
    y : array-like
        Target values
        
    Returns:
    --------
    order : array-like
        Indices that would sort x
    x_sorted : array-like
        Sorted x values
    y_sorted : array-like
        y values sorted by x
    """
    order = np.argsort(x)
    x_sorted = np.array(x)[order]
    y_sorted = np.array(y)[order]
    
    return order, x_sorted, y_sorted