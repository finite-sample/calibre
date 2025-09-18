Quick Start Guide
=================

This guide will help you get started with Calibre's calibration methods.

Basic Concepts
--------------

**Probability calibration** is the process of adjusting the output probabilities of a machine learning model to better reflect the true likelihood of the predicted class. A well-calibrated model's predicted probabilities should match the observed frequencies.

For example, if a model predicts 100 instances with 70% probability of being positive, approximately 70 of them should actually be positive.

Common Calibration Problems
---------------------------

1. **Overconfidence**: Model outputs probabilities that are too extreme (close to 0 or 1)
2. **Underconfidence**: Model outputs probabilities that are too conservative (close to 0.5)
3. **Loss of granularity**: Traditional methods collapse many distinct probabilities into few unique values

Basic Usage
-----------

Nearly Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

The most versatile method that allows controlled violations of monotonicity:

.. code-block:: python

   import numpy as np
   from calibre import NearlyIsotonicRegression
   
   # Example data: model predictions and true binary outcomes
   np.random.seed(42)
   y_pred = np.sort(np.random.uniform(0, 1, 1000))
   y_true = np.random.binomial(1, y_pred, 1000)
   
   # Strict monotonicity (high lambda)
   cal_strict = NearlyIsotonicRegression(lam=10.0, method='cvx')
   cal_strict.fit(y_pred, y_true)
   y_calibrated_strict = cal_strict.transform(y_pred)
   
   # More flexible (low lambda) - preserves more granularity
   cal_flexible = NearlyIsotonicRegression(lam=0.1, method='cvx')
   cal_flexible.fit(y_pred, y_true)
   y_calibrated_flexible = cal_flexible.transform(y_pred)

I-Spline Calibration
~~~~~~~~~~~~~~~~~~~~

For smooth calibration curves using monotonic splines:

.. code-block:: python

   from calibre import ISplineCalibrator
   
   # Smooth calibration with cross-validation
   cal_ispline = ISplineCalibrator(n_splines=10, degree=3, cv=5)
   cal_ispline.fit(y_pred, y_true)
   y_calibrated_smooth = cal_ispline.transform(y_pred)

Relaxed PAVA
~~~~~~~~~~~~

Ignores small violations while correcting larger ones:

.. code-block:: python

   from calibre import RelaxedPAVA
   
   # Allow violations below 10th percentile threshold
   cal_relaxed = RelaxedPAVA(percentile=10, adaptive=True)
   cal_relaxed.fit(y_pred, y_true)
   y_calibrated_relaxed = cal_relaxed.transform(y_pred)

Regularized Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds L2 regularization for smoother curves:

.. code-block:: python

   from calibre import RegularizedIsotonicRegression
   
   # L2 regularized isotonic regression
   cal_reg = RegularizedIsotonicRegression(alpha=0.1)
   cal_reg.fit(y_pred, y_true)
   y_calibrated_reg = cal_reg.transform(y_pred)

Smoothed Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces the "staircase" effect of standard isotonic regression:

.. code-block:: python

   from calibre import SmoothedIsotonicRegression
   
   # Apply Savitzky-Golay smoothing
   cal_smooth = SmoothedIsotonicRegression(
       window_length=7, 
       poly_order=3, 
       interp_method='linear'
   )
   cal_smooth.fit(y_pred, y_true)
   y_calibrated_smooth = cal_smooth.transform(y_pred)

Evaluating Calibration Quality
------------------------------

Calibre provides several metrics to evaluate calibration quality:

.. code-block:: python

   from calibre import (
       mean_calibration_error,
       expected_calibration_error,
       binned_calibration_error,
       correlation_metrics,
       unique_value_counts
   )
   
   # Calculate calibration errors
   mce = mean_calibration_error(y_true, y_calibrated_strict)
   ece = expected_calibration_error(y_true, y_calibrated_strict, n_bins=10)
   bce = binned_calibration_error(y_true, y_calibrated_strict, n_bins=10)
   
   print(f"Mean Calibration Error: {mce:.4f}")
   print(f"Expected Calibration Error: {ece:.4f}")
   print(f"Binned Calibration Error: {bce:.4f}")
   
   # Check correlations and granularity preservation
   corr = correlation_metrics(y_true, y_calibrated_strict, y_orig=y_pred)
   counts = unique_value_counts(y_calibrated_strict, y_orig=y_pred)
   
   print(f"Correlation with true values: {corr['spearman_corr_to_y_true']:.4f}")
   print(f"Granularity preservation: {counts['unique_value_ratio']:.2f}")

Choosing the Right Method
-------------------------

Use this guide to select the appropriate calibration method:

- **NearlyIsotonicRegression (method='cvx')**: When you want precise control over the monotonicity/granularity trade-off and can afford the computational cost
- **NearlyIsotonicRegression (method='path')**: For larger datasets where you need efficiency but still want some control
- **ISplineCalibrator**: When you want smooth calibration curves for visualization and interpretation
- **RelaxedPAVA**: For a simple, efficient approach that ignores small violations
- **RegularizedIsotonicRegression**: When you need smoother curves with L2 regularization
- **SmoothedIsotonicRegression**: To reduce the "staircase effect" while preserving monotonicity

Example Workflow
----------------

Here's a complete example showing how to calibrate a model and evaluate the results:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_classification
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from calibre import NearlyIsotonicRegression, mean_calibration_error
   
   # Generate example data
   X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
   
   # Train a model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   
   # Get uncalibrated predictions
   y_pred_uncal = model.predict_proba(X_test)[:, 1]
   
   # Calibrate the predictions
   calibrator = NearlyIsotonicRegression(lam=1.0, method='path')
   calibrator.fit(y_pred_uncal, y_test)
   y_pred_cal = calibrator.transform(y_pred_uncal)
   
   # Evaluate calibration
   mce_before = mean_calibration_error(y_test, y_pred_uncal)
   mce_after = mean_calibration_error(y_test, y_pred_cal)
   
   print(f"Mean Calibration Error before: {mce_before:.4f}")
   print(f"Mean Calibration Error after: {mce_after:.4f}")
   print(f"Improvement: {((mce_before - mce_after) / mce_before * 100):.1f}%")

Next Steps
----------

- Explore the :doc:`api/index` for detailed method documentation
- Check out the :doc:`examples/index` for more advanced usage patterns
- Read about the theory behind each method in the algorithm documentation