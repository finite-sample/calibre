Cross-Validation
================

Shared cross-validation machinery. Every calibrator with an ``"auto"``
hyperparameter resolves it through :func:`~calibre.select_by_cv`, so the
selection rule is the same everywhere and is implemented once.

Selection is always on a **proper scoring rule** — log loss or Brier. Calibration
error is deliberately rejected as a selection criterion: it is not proper, and a
calibrator tuned to minimise ECE can win by discarding resolution. There is a
test asserting the rejection.

Out-of-Fold Calibration
-----------------------

.. autofunction:: calibre.cross_val_calibrate

Model Selection
---------------

.. autofunction:: calibre.select_by_cv

.. autofunction:: calibre.make_folds

.. autofunction:: calibre.selection.resolve_auto

Usage
-----

.. code-block:: python

   import numpy as np

   from calibre import CenteredIsotonicCalibrator, cross_val_calibrate

   rng = np.random.default_rng(0)
   scores = rng.uniform(0, 1, 1500)
   labels = rng.binomial(1, scores).astype(float)

   # Every returned probability comes from a model that never saw that row.
   out_of_fold = cross_val_calibrate(
       CenteredIsotonicCalibrator(), scores, labels, cv=5
   )
   print(out_of_fold.shape)
