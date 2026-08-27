Calibration Report
==================

One call that gathers the CORP decomposition, four calibration-error estimators,
and prediction granularity. It summarizes independent held-out evaluation rows;
it does not perform a hypothesis test or issue a calibration verdict. Nothing
here is new: it is assembled from :doc:`evaluation` and :doc:`metrics`.

.. autofunction:: calibre.calibration_report

.. autoclass:: calibre.CalibrationReport
   :members: to_dict
   :exclude-members: n, base_rate, mean_prediction, mean_calibration_error,
      brier_score, miscalibration, discrimination, uncertainty,
      smooth_calibration_error, smooth_calibration_bandwidth,
      debiased_calibration_error, plugin_calibration_error,
      sweep_calibration_error, sweep_n_bins, n_bins, n_unique_predictions,
      unique_prediction_ratio, intervals

Confidence intervals
--------------------

.. autofunction:: calibre.bootstrap_ci

Usage
-----

.. code-block:: python

   import numpy as np

   from calibre import calibration_report

   rng = np.random.default_rng(0)
   p = rng.uniform(0, 1, 3000)
   y = rng.binomial(1, p).astype(float)
   overconfident = np.clip(1.8 * (p - 0.5) + 0.5, 0, 1)

   print(calibration_report(y, overconfident))

.. code-block:: text

   CalibrationReport  n=3,000  base rate 0.4933

     Brier            0.1849
       = MCB          0.0183   (recalibration recovers this)
       - DSC          0.0834   (earned by the forecasts)
       + UNC          0.2500   (irreducible)

     mean cal. error  0.0039   (mean forecast 0.4973)
     smECE            0.1173   (bandwidth 0.1182, chosen)
     debiased ECE     0.1240   (15 bins)
     plugin ECE       0.1267   (15 bins, uncorrected)
     sweep ECE        0.1222   (6 bins; assumes a monotone calibration curve)

     prediction granularity  1,657 of 3,000 values unique (55.2%)

Read ``MCB`` first: it is what recalibration would recover, and here it is a
fifth of what the forecasts earn in ``DSC``. The four error estimators agree on
the magnitude but not the number, which is the point of showing all four.

.. warning::

   Run this on independent, **held-out** predictions. On the data a calibrator was
   fitted to, any isotonic-family method reports ``MCB`` of exactly zero by
   construction -- the calibrator and this diagnostic are the same PAV projection,
   and PAV is idempotent -- no matter how badly the model generalizes. Use
   :func:`~calibre.cross_val_calibrate` for out-of-fold probabilities.

   Sweep ECE assumes a non-decreasing population calibration curve. It can be near
   zero for strongly nonmonotone miscalibration, so compare it with the other
   diagnostics rather than treating it as a general verdict.
