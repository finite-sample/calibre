Calibration Methods
===================

This module contains all calibration algorithms implemented in Calibre.

Base Classes
------------

.. autoclass:: calibre.BaseCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Calibration Algorithms
----------------------

Nearly Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.NearlyIsotonicRegression
   :members:
   :undoc-members:
   :show-inheritance:

I-Spline Calibrator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.ISplineCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Relaxed PAVA
~~~~~~~~~~~~

.. autoclass:: calibre.RelaxedPAVA
   :members:
   :undoc-members:
   :show-inheritance:

Regularized Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.RegularizedIsotonicRegression
   :members:
   :undoc-members:
   :show-inheritance:

Smoothed Isotonic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.SmoothedIsotonicRegression
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from calibre import NearlyIsotonicRegression
   import numpy as np
   
   # Generate example data
   np.random.seed(42)
   X = np.random.uniform(0, 1, 1000)
   y = np.random.binomial(1, X, 1000)
   
   # Fit calibrator
   calibrator = NearlyIsotonicRegression(lam=1.0)
   calibrator.fit(X, y)
   
   # Transform predictions
   X_new = np.random.uniform(0, 1, 100)
   y_calibrated = calibrator.transform(X_new)

Comparing Methods
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import (
       NearlyIsotonicRegression,
       ISplineCalibrator,
       RelaxedPAVA,
       RegularizedIsotonicRegression
   )
   
   # Initialize different calibrators
   calibrators = {
       'Nearly Isotonic': NearlyIsotonicRegression(lam=1.0),
       'I-Spline': ISplineCalibrator(n_splines=10),
       'Relaxed PAVA': RelaxedPAVA(percentile=10),
       'Regularized': RegularizedIsotonicRegression(alpha=0.1)
   }
   
   # Fit and compare
   results = {}
   for name, cal in calibrators.items():
       cal.fit(X, y)
       y_cal = cal.transform(X_new)
       results[name] = y_cal