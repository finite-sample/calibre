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

Cost- and Data-Informed Isotonic Calibrator (Research)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.CDIIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   CDI-ISO is a research-grade calibrator that uses economic decision theory 
   and statistical evidence to make informed monotonicity decisions. It requires
   specification of operating thresholds where discrimination matters most.

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
       CDIIsotonicCalibrator,
       IsotonicCalibrator,
       NearlyIsotonicCalibrator,
       SplineCalibrator,
       RelaxedPAVACalibrator,
       RegularizedIsotonicCalibrator
   )
   
   # Initialize different calibrators
   calibrators = {
       'CDI-ISO': CDIIsotonicCalibrator(thresholds=[0.3, 0.7], gamma=0.15),
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

CDI-ISO Usage Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from calibre import CDIIsotonicCalibrator
   import numpy as np
   
   # Generate example data
   np.random.seed(42)
   X = np.random.uniform(0, 1, 1000)
   y = np.random.binomial(1, X, 1000)
   
   # CDI-ISO with economic thresholds (e.g., decision points at 0.3 and 0.7)
   cdi_cal = CDIIsotonicCalibrator(
       thresholds=[0.3, 0.7],          # Operating decision thresholds
       threshold_weights=[0.6, 0.4],   # Relative importance
       bandwidth=0.1,                   # Kernel bandwidth around thresholds
       gamma=0.2,                       # Minimum slope strength
       alpha=0.05,                      # Statistical significance level
       window=30                        # Evidence window size
   )
   
   # Fit the calibrator
   cdi_cal.fit(X, y)
   
   # Get calibrated predictions
   X_test = np.random.uniform(0, 1, 100)
   y_calibrated = cdi_cal.transform(X_test)
   
   # Access diagnostic information
   bounds = cdi_cal.adjacency_bounds_()
   breakpoints = cdi_cal.breakpoints_()
   
   print(f"CDI calibrator learned {len(bounds)} local bounds")
   print(f"Calibration function has {len(breakpoints[0])} breakpoints")