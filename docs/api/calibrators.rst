Calibration Methods
===================

Every calibrator follows the scikit-learn transformer API: ``.fit(scores,
labels)`` and ``.transform(scores)``, plus ``sample_weight`` where it is
meaningful. All are binary; for multiclass see :doc:`multiclass`.

Which calibrator should I use?
------------------------------

**If you don't want to think about it:**
:class:`~calibre.CenteredIsotonicCalibrator`. It is non-parametric, has nothing
to tune, is monotone, and has no plateaus.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - You want
     - Use
     - Notes
   * - A drop-in isotonic replacement, no tuning
     - :class:`~calibre.CenteredIsotonicCalibrator`
     - Collapses isotonic's flat steps to points and interpolates. O(n).
   * - A smooth curve, and you can afford cross-validation
     - :class:`~calibre.SplineCalibrator`
     - Monotone spline; picks its own smoothing by link-appropriate prediction loss.
   * - A smooth curve with smoothing you control
     - :class:`~calibre.RegularizedIsotonicCalibrator`
     - Same model, you set ``alpha`` instead of tuning it. Fast.
   * - Exactly scikit-learn's isotonic behaviour
     - :class:`~calibre.IsotonicCalibrator`
     - Thin wrapper, plus optional plateau diagnostics.
   * - Guaranteed strictly increasing output
     - :class:`~calibre.RelaxedPAVACalibrator`
     - ``min_slope`` forces a minimum step between adjacent scores.
   * - To allow small ranking violations if they fit better
     - :class:`~calibre.NearlyIsotonicCalibrator`
     - ``lam`` trades monotonicity against fit.
   * - Accuracy near specific decision thresholds
     - :class:`~calibre.CDIIsotonicCalibrator`
     - Research-grade; needs your operating thresholds.

Base Classes
------------

.. autoclass:: calibre.BaseCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: calibre.MonotonicMixin
   :members:
   :no-index:

Recommended Default
-------------------

Centered Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.CenteredIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Centered isotonic regression (Oron & Flournoy 2017) runs PAVA, then collapses
each flat block to its weighted centroid and interpolates linearly between the
collapsed points. The result is calibrated as well as isotonic regression but
keeps almost all of the input's distinct values.

Other Calibrators
-----------------

Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.IsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Spline Calibrator
~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.SplineCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Regularized Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.RegularizedIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   This is a monotone spline with a second-difference (curvature) penalty. It is
   **not** ridge regression, and ``alpha=0`` is not isotonic regression.

Relaxed PAVA Calibrator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.RelaxedPAVACalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Bounds each adjacent increment: ``epsilon`` permits small decreases, while
``min_slope`` forbids plateaus outright. Solved by shift-to-PAVA in O(n).

Nearly Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.NearlyIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   Penalises rather than forbids monotonicity violations. Two exact solvers:
   ``method="path"`` (default, pure NumPy) and ``method="cvx"`` (CVXPY). Note
   that ``lam`` is twice the source paper's lambda.

Smoothed Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.SmoothedIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   Savitzky-Golay smoothing of an isotonic fit. Retained for compatibility;
   prefer :class:`~calibre.SplineCalibrator` or
   :class:`~calibre.RegularizedIsotonicCalibrator` for a smooth curve.

Research
--------

Cost- and Data-Informed Isotonic Calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: calibre.CDIIsotonicCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   CDI-ISO is research-grade. It uses economic decision theory and statistical
   evidence to decide where monotonicity should be enforced strictly, and
   requires you to specify the operating thresholds where discrimination
   matters most.

Usage Examples
--------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from calibre import CenteredIsotonicCalibrator

   rng = np.random.default_rng(42)
   X = rng.uniform(0, 1, 1000)
   y = rng.binomial(1, X).astype(float)

   calibrator = CenteredIsotonicCalibrator().fit(X, y)

   X_new = rng.uniform(0, 1, 100)
   y_calibrated = calibrator.transform(X_new)

.. warning::
   Always fit the calibrator on data the model did not train on. A model's
   scores on its own training data are already too good, so a calibrator fitted
   there learns the wrong correction. Use a held-out split, or
   :func:`~calibre.cross_val_calibrate` for out-of-fold predictions.

Comparing Methods
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from calibre import (
       CenteredIsotonicCalibrator,
       IsotonicCalibrator,
       RegularizedIsotonicCalibrator,
       RelaxedPAVACalibrator,
       SplineCalibrator,
       unique_value_counts,
   )

   calibrators = {
       "Isotonic": IsotonicCalibrator(),
       "Centered": CenteredIsotonicCalibrator(),
       "Spline": SplineCalibrator(),
       "Relaxed PAVA": RelaxedPAVACalibrator(min_slope=1e-5),
       "Regularized": RegularizedIsotonicCalibrator(alpha=0.1),
   }

   for name, cal in calibrators.items():
       out = cal.fit(X, y).transform(X)
       n = unique_value_counts(out)["n_unique_y_pred"]
       print(f"{name:14s} {n:5d} distinct values")

CDI-ISO Usage Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from calibre import CDIIsotonicCalibrator

   cdi_cal = CDIIsotonicCalibrator(
       thresholds=[0.3, 0.7],           # operating decision thresholds
       threshold_weights=[0.6, 0.4],    # relative importance
       bandwidth=0.1,                   # kernel bandwidth around thresholds
       gamma=0.2,                       # minimum slope strength
       alpha=0.05,                      # significance level
       window=30,                       # evidence window size
   )
   cdi_cal.fit(X, y)
   y_calibrated = cdi_cal.transform(X_new)

   bounds = cdi_cal.adjacency_bounds_()
   breakpoints = cdi_cal.breakpoints_()

   print(f"CDI calibrator learned {len(bounds)} local bounds")
   print(f"Calibration function has {len(breakpoints[0])} breakpoints")
