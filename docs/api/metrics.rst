Calibration Metrics
===================

This module provides metrics for evaluating calibration quality.

For the CORP reliability diagram and the ``MCB``/``DSC``/``UNC`` score
decomposition — which need no bin count and cannot be tuned in your favour —
see :doc:`evaluation`.

Bias-aware calibration error
----------------------------

The plugin binned estimator is biased upward: part of each bin's gap is sampling
noise in the label mean rather than miscalibration, and the bias grows with the
bin count. These two estimators correct for that, and are the ones to reach for
when the number will be reported.

Smooth Calibration Error (smECE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.smooth_calibration_error

Unlike everything below it, smECE has no bin count *and* no bandwidth to choose,
and it is a consistent measure of distance from calibration. It is the one to
reach for when the number will be quoted without qualification.

Debiased Calibration Error
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.debiased_calibration_error

Sweep Calibration Error
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.sweep_calibration_error

Plugin Calibration Error
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.plugin_calibration_error

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

.. note::

   These estimators are not interchangeable, and their magnitudes are not
   comparable. :func:`~calibre.expected_calibration_error` and
   :func:`~calibre.sweep_calibration_error` are :math:`\ell_1`;
   :func:`~calibre.debiased_calibration_error` is :math:`\ell_2`.
   :func:`~calibre.expected_calibration_error` uses uniform-width bins, while the
   two bias-aware estimators use equal-mass bins. Compare like with like.

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

Granularity Metrics
-------------------

These measure what a calibrator did to the *resolution* of your scores — the
thing isotonic regression quietly destroys. No other calibration package
reports them.

Unique Value Counts
~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.unique_value_counts

Tie Preservation Score
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calibre.tie_preservation_score

Usage Examples
--------------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from calibre import (
       brier_score,
       expected_calibration_error,
       mean_calibration_error,
   )

   y_true = np.array([0, 0, 1, 1, 1])
   y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9])

   print(f"Brier score {brier_score(y_true, y_pred):.4f}")
   print(f"ECE         {expected_calibration_error(y_true, y_pred, n_bins=5):.4f}")
   print(f"bias        {mean_calibration_error(y_true, y_pred):.4f}")

``brier_score`` is a proper scoring rule and the one to optimise.
``mean_calibration_error`` is calibration in the large, ``|mean(prediction) −
base rate|``.

Reporting an honest calibration error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On data that is calibrated by construction the true error is zero, so whatever
the plugin estimator reports is bias:

.. code-block:: python

   import numpy as np

   from calibre import debiased_calibration_error, sweep_calibration_error
   from calibre.metrics import expected_calibration_error

   rng = np.random.default_rng(0)
   p = rng.uniform(0, 1, 4000)
   y = rng.binomial(1, p).astype(float)

   print(f"plugin ECE  {expected_calibration_error(y, p, n_bins=15):.4f}")
   print(f"debiased    {debiased_calibration_error(y, p, n_bins=15):.4f}")
   print(f"sweep       {sweep_calibration_error(y, p):.4f}")

Measuring what calibration cost you in resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   from calibre import (
       CenteredIsotonicCalibrator,
       IsotonicCalibrator,
       unique_value_counts,
   )

   rng = np.random.default_rng(0)
   scores = rng.uniform(0, 1, 2000)
   labels = rng.binomial(1, scores).astype(float)

   for name, cal in (
       ("isotonic", IsotonicCalibrator()),
       ("centered", CenteredIsotonicCalibrator()),
   ):
       out = cal.fit(scores, labels).transform(scores)
       counts = unique_value_counts(out, y_orig=scores)
       print(f"{name:9s} {counts['n_unique_y_pred']:5d} distinct values")

Both are well calibrated. Only one of them still tells you which of two cases
is the riskier bet.
