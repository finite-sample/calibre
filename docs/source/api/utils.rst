Utility Functions
=================

This module provides utility functions for data processing and validation.

Data Validation
---------------

Array Checking
~~~~~~~~~~~~~~

.. autofunction:: calibre.check_arrays

Data Processing
---------------

Sorting Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.sort_by_x

Binning Operations
~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.create_bins

.. autofunction:: calibre.bin_data

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

Creating Bins
~~~~~~~~~~~~~

.. code-block:: python

   from calibre.utils import create_bins, bin_data
   import numpy as np
   
   # Create uniform bins
   X = np.random.uniform(0, 1, 1000)
   bins_uniform = create_bins(X, n_bins=10, strategy='uniform')
   print(f"Uniform bins: {bins_uniform}")
   
   # Create quantile bins
   bins_quantile = create_bins(X, n_bins=10, strategy='quantile')
   print(f"Quantile bins: {bins_quantile}")
   
   # Assign data to bins
   bin_indices, bin_counts = bin_data(X, bins_uniform)
   print(f"Bin counts: {bin_counts}")

Advanced Binning
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom bin range
   bins_custom = create_bins(
       X, 
       n_bins=5, 
       strategy='uniform',
       x_min=0.2,  # Custom range
       x_max=0.8
   )
   
   # Bin with custom bins
   bin_indices, bin_counts = bin_data(X, bins_custom)
   
   # Analyze bin distribution
   for i, count in enumerate(bin_counts):
       bin_start = bins_custom[i]
       bin_end = bins_custom[i + 1]
       print(f"Bin [{bin_start:.2f}, {bin_end:.2f}): {count} samples")