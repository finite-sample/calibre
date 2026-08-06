API Reference
=============

This section provides detailed documentation for all Calibre classes and
functions.

.. toctree::
   :maxdepth: 2

   calibrators
   evaluation
   metrics
   report
   multiclass
   selection
   diagnostics
   plots
   utils

Overview
--------

**Calibrating.**

- :doc:`calibrators` — the calibration algorithms and base classes. Start with
  :class:`~calibre.CenteredIsotonicCalibrator`.
- :doc:`selection` — cross-validation shared by every calibrator, including
  :func:`~calibre.cross_val_calibrate` for out-of-fold probabilities.

**Measuring.**

- :doc:`evaluation` — CORP reliability diagrams and the ``MCB``/``DSC``/``UNC``
  score decomposition. No bin count to choose, and none to tune in your favour.
- :doc:`metrics` — calibration error estimators, including the bias-aware ones and
  smECE, plus the granularity metrics.
- :doc:`report` — one call that gathers all of it, with optional bootstrap intervals.
- :doc:`multiclass` — class-wise evaluation and the diagnostic that tells you
  which multiclass method you need.
- :doc:`diagnostics` — where a fitted curve went flat, and how much data each
  plateau rests on.

**Drawing.**

- :doc:`plots` — reliability diagrams, the score decomposition, and the
  resolution barcode. Needs ``pip install 'calibre[plots]'``.

**Supporting.**

- :doc:`utils` — validation and array helpers.
