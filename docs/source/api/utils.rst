Utility Functions  
=================

This module provides utility functions for data validation and array operations.

Data Validation
---------------

.. autofunction:: calibre.utils.check_arrays

.. autofunction:: calibre.utils.check_array_1d

.. autofunction:: calibre.utils.check_consistent_length

Array Operations
----------------

.. autofunction:: calibre.utils.sort_by_x

.. autofunction:: calibre.utils.clip_to_range

.. autofunction:: calibre.utils.ensure_1d

Usage Examples
--------------

Input Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre.utils import check_arrays
   import numpy as np
   
   # Valid input
   X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
   y = np.array([0, 0, 1, 1, 1])
   
   try:
       X_checked, y_checked = check_arrays(X, y)
       print("Arrays are valid")
   except ValueError as e:
       print(f"Validation error: {e}")

Sorting Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre.utils import sort_by_x
   
   # Unsorted data
   X = np.array([0.7, 0.1, 0.9, 0.3, 0.5])
   y = np.array([1, 0, 1, 0, 1])
   
   # Sort by X values
   sort_indices, X_sorted, y_sorted = sort_by_x(X, y)
   
   print(f"Original X: {X}")
   print(f"Sorted X: {X_sorted}")
   print(f"Sorted y: {y_sorted}")
   print(f"Sort indices: {sort_indices}")

Array Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre.utils import ensure_1d, clip_to_range
   import numpy as np
   
   # Ensure array is 1D
   arr_2d = np.array([[1], [2], [3]])
   arr_1d = ensure_1d(arr_2d)
   print(f"1D array: {arr_1d}")
   
   # Clip values to valid range
   values = np.array([-0.1, 0.5, 1.2])
   clipped = clip_to_range(values, 0.0, 1.0)
   print(f"Clipped: {clipped}")

Note
~~~~

These utility functions are primarily for internal use within calibration algorithms. 
For typical calibration workflows, use the main calibrator classes directly:

.. code-block:: python

   from calibre import IsotonicCalibrator, expected_calibration_error
   import numpy as np
   
   # This is the recommended approach for users
   X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
   y = np.array([0, 0, 1, 1, 1])
   
   cal = IsotonicCalibrator()
   cal.fit(X, y)
   X_calibrated = cal.transform(X)
   
   ece = expected_calibration_error(y, X_calibrated)
   print(f"ECE: {ece:.4f}")