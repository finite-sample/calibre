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

Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.IsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Nearly Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.NearlyIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Spline Calibrator
~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.SplineCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Relaxed PAVA Calibrator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.RelaxedPAVACalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Regularized Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.RegularizedIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Smoothed Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.SmoothedIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from calibre import IsotonicCalibrator
   import numpy as np
   
   # Generate example data
   np.random.seed(42)
   X = np.random.uniform(0, 1, 1000)
   y = np.random.binomial(1, X, 1000)
   
   # Fit calibrator
   calibrator = IsotonicCalibrator()
   calibrator.fit(X, y)
   
   # Transform predictions
   X_new = np.random.uniform(0, 1, 100)
   y_calibrated = calibrator.transform(X_new)

Comparing Methods
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import (
       IsotonicCalibrator,
       NearlyIsotonicCalibrator,
       SplineCalibrator,
       RelaxedPAVACalibrator,
       RegularizedIsotonicCalibrator
   )
   
   # Initialize different calibrators
   calibrators = {
       'Isotonic': IsotonicCalibrator(),
       'Nearly Isotonic': NearlyIsotonicCalibrator(lam=1.0),
       'Spline': SplineCalibrator(n_splines=10),
       'Relaxed PAVA': RelaxedPAVACalibrator(percentile=10),
       'Regularized': RegularizedIsotonicCalibrator(alpha=0.1)
   }
   
   # Fit and compare
   results = {}
   for name, cal in calibrators.items():
       cal.fit(X, y)
       y_cal = cal.transform(X_new)
       results[name] = y_cal
       
   print(f"Calibrated {len(X_new)} predictions using {len(calibrators)} methods")