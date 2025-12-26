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
   from calibre import NearlyIsotonicCalibrator
   
   # Example data: model predictions and true binary outcomes
   np.random.seed(42)
   y_pred = np.sort(np.random.uniform(0, 1, 1000))
   y_true = np.random.binomial(1, y_pred, 1000)
   
   # Strict monotonicity (high lambda)
   cal_strict = NearlyIsotonicCalibrator(lam=10.0, method='cvx')
   cal_strict.fit(y_pred, y_true)
   y_calibrated_strict = cal_strict.transform(y_pred)
   
   # More flexible (low lambda) - preserves more granularity
   cal_flexible = NearlyIsotonicCalibrator(lam=0.1, method='cvx')
   cal_flexible.fit(y_pred, y_true)
   y_calibrated_flexible = cal_flexible.transform(y_pred)

I-Spline Calibration
~~~~~~~~~~~~~~~~~~~~

For smooth calibration curves using monotonic splines:

.. code-block:: python

   from calibre import SplineCalibrator
   
   # Smooth calibration with cross-validation
   cal_ispline = SplineCalibrator(n_splines=10, degree=3, cv=5)
   cal_ispline.fit(y_pred, y_true)
   y_calibrated_smooth = cal_ispline.transform(y_pred)

Relaxed PAVA
~~~~~~~~~~~~

Ignores small violations while correcting larger ones:

.. code-block:: python

   from calibre import RelaxedPAVACalibrator
   
   # Allow violations below 10th percentile threshold
   cal_relaxed = RelaxedPAVACalibrator(percentile=10, adaptive=True)
   cal_relaxed.fit(y_pred, y_true)
   y_calibrated_relaxed = cal_relaxed.transform(y_pred)

Regularized Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adds L2 regularization for smoother curves:

.. code-block:: python

   from calibre import RegularizedIsotonicCalibrator
   
   # L2 regularized isotonic regression
   cal_reg = RegularizedIsotonicCalibrator(alpha=0.1)
   cal_reg.fit(y_pred, y_true)
   y_calibrated_reg = cal_reg.transform(y_pred)

Smoothed Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces the "staircase" effect of standard isotonic regression:

.. code-block:: python

   from calibre import SmoothedIsotonicCalibrator
   
   # Apply Savitzky-Golay smoothing
   cal_smooth = SmoothedIsotonicCalibrator(
       window_length=7, 
       poly_order=3, 
       interp_method='linear'
   )
   cal_smooth.fit(y_pred, y_true)
   y_calibrated_smooth = cal_smooth.transform(y_pred)

Quick Evaluation
----------------

Calibre provides metrics to evaluate calibration quality:

.. code-block:: python

   from calibre import mean_calibration_error
   
   # Calculate calibration error
   mce = mean_calibration_error(y_true, y_calibrated_strict)
   print(f"Mean Calibration Error: {mce:.4f}")

Choosing the Right Method
-------------------------

Quick selection guide:

- **IsotonicCalibrator**: Start here - standard, reliable approach
- **NearlyIsotonicCalibrator**: When you need more granularity preservation
- **SplineCalibrator**: For smooth curves and visualization
- **RegularizedIsotonicCalibrator**: Often best calibration quality
- **RelaxedPAVACalibrator**: Good for small datasets
- **SmoothedIsotonicCalibrator**: Reduces staircase effects

Next Steps
----------

- Explore the :doc:`api/index` for detailed method documentation
- Check out the :doc:`examples/index` for more advanced usage patterns
- Read about the theory behind each method in the algorithm documentation