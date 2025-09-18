Calibration Metrics
===================

This module provides various metrics for evaluating calibration quality.

Calibration Error Metrics
--------------------------

Mean Calibration Error
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.mean_calibration_error

Binned Calibration Error
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.binned_calibration_error

Expected Calibration Error
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.expected_calibration_error

Maximum Calibration Error
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.maximum_calibration_error

Scoring Metrics
---------------

Brier Score
~~~~~~~~~~~

.. autofunction:: calibre.brier_score

Calibration Curve
~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.calibration_curve

Statistical Metrics
-------------------

Correlation Metrics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.correlation_metrics

Unique Value Counts
~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.unique_value_counts

Usage Examples
--------------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import (
       mean_calibration_error,
       expected_calibration_error,
       brier_score
   )
   import numpy as np
   
   # Example data
   y_true = np.array([0, 0, 1, 1, 1])
   y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
   
   # Calculate metrics
   mce = mean_calibration_error(y_true, y_pred)
   ece = expected_calibration_error(y_true, y_pred, n_bins=5)
   bs = brier_score(y_true, y_pred)
   
   print(f"Mean Calibration Error: {mce:.4f}")
   print(f"Expected Calibration Error: {ece:.4f}")
   print(f"Brier Score: {bs:.4f}")

Comprehensive Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import (
       binned_calibration_error,
       correlation_metrics,
       unique_value_counts
   )
   
   # Binned calibration with details
   bce, details = binned_calibration_error(
       y_true, y_pred, 
       n_bins=10, 
       return_details=True
   )
   
   print(f"Binned Calibration Error: {bce:.4f}")
   print(f"Bin centers: {details['bin_centers']}")
   print(f"Bin accuracies: {details['bin_accuracies']}")
   
   # Correlation analysis
   corr = correlation_metrics(y_true, y_pred)
   print(f"Spearman correlation: {corr['spearman_corr_to_y_true']:.4f}")
   
   # Granularity analysis
   counts = unique_value_counts(y_pred)
   print(f"Unique values: {counts['n_unique_y_pred']}")

Plotting Calibration Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from calibre import calibration_curve
   
   # Generate calibration curve data
   fraction_pos, mean_pred, counts = calibration_curve(
       y_true, y_pred, n_bins=10
   )
   
   # Plot reliability diagram
   plt.figure(figsize=(8, 6))
   plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
   plt.plot(mean_pred, fraction_pos, 'bo-', label='Model')
   plt.xlabel('Mean Predicted Probability')
   plt.ylabel('Fraction of Positives')
   plt.title('Calibration Plot')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()