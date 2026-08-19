Calibration Report
==================

One call that gathers the CORP decomposition, three calibration-error estimators
that disagree in instructive ways, the bias, and the resolution the forecasts
retain. Nothing here is new: it is assembled from :doc:`evaluation` and
:doc:`metrics`.

.. autofunction:: calibre.calibration_report

.. autoclass:: calibre.CalibrationReport
   :members: to_dict
   :exclude-members: n, base_rate, mean_prediction, bias, brier, mcb, dsc, unc,
      smece, smece_sigma, debiased_ece, plugin_ece, sweep_ece, sweep_bins,
      n_bins, n_distinct, distinct_ratio, intervals

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

     bias             0.0039   (mean forecast 0.4973)
     smECE            0.1173   (bandwidth 0.1182, chosen)
     debiased ECE     0.1240   (15 bins)
     plugin ECE       0.1267   (15 bins, uncorrected)
     sweep ECE        0.1180   (6 bins, chosen)

     distinct values  1,657 of 3,000 (55.2%)

Read ``MCB`` first: it is what recalibration would recover, and here it is a
fifth of what the forecasts earn in ``DSC``. The three error estimators agree on
the magnitude but not the number, which is the point of showing all three.

.. warning::

   Run this on **held-out** predictions. On the data a calibrator was fitted to,
   any isotonic-family method reports ``MCB`` of exactly zero by construction --
   the calibrator and this diagnostic are the same PAV projection, and PAV is
   idempotent -- no matter how badly the model generalises. Use
   :func:`~calibre.cross_val_calibrate` for out-of-fold probabilities.
