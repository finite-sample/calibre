Calibre: Advanced Calibration Models
=====================================

.. image:: https://img.shields.io/pypi/v/calibre.svg
   :target: https://pypi.org/project/calibre/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/calibre.svg
   :target: https://pypi.org/project/calibre/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

**Probability calibration that doesn't flatten your scores.**

Your classifier's probabilities are usually wrong — a model that says "80%" may
be right 60% of the time. Isotonic regression is the standard fix, and it works,
but it pays for accuracy with resolution: it is a step function, so it collapses
many distinct scores into a handful of values. On a 2,000-point held-out set,
isotonic regression turns 2,000 distinct scores into **82**. Everything inside a
step becomes indistinguishable — which matters as soon as you rank, threshold,
or bucket the output.

Calibre gives you calibration methods that retain much more of that ordering
while correcting the probabilities, together with tools to measure both.

Calibrating
-----------

- **Centered isotonic regression**: collapses PAVA's flat blocks to their
  centroid and interpolates. Non-parametric, nothing to tune, and preserves
  score ordering between pooled blocks. The recommended default.
- **Monotone I-splines**: smooth calibration curves, with the smoothing either
  chosen by cross-validation using the loss appropriate for the link or set by
  you.
- **Relaxed PAVA**: bounds each adjacent increment. ``epsilon`` permits small
  decreases; ``min_slope`` forbids plateaus when output clipping is disabled. O(n).
- **Nearly-isotonic regression**: penalises rather than forbids monotonicity
  violations, when a small reordering buys a better fit.
- **Regularized isotonic regression**: a monotone spline with an explicit
  curvature penalty.

Measuring
---------

- **CORP reliability diagrams and score decompositions**: split a proper score
  into ``MCB`` (what recalibration would save you), ``DSC`` (what your scores
  buy over the base rate) and ``UNC`` (irreducible difficulty). No bin count to
  choose. Pinned against R's ``reliabilitydiag`` to 1e-16.
- **Bias-aware calibration error**: binned ECE is biased upward, and its value
  depends on the bin count. ``debiased_calibration_error`` corrects the
  within-bin bias; ``sweep_calibration_error`` chooses the bin count.
- **Multiclass diagnostics**: ``miscalibration_profile`` helps distinguish
  global from class-specific miscalibration before you choose a method.
- **Granularity metrics**: how much resolution the calibration cost you — the
  thing no other calibration package reports.

Quick Start
-----------

Install Calibre:

.. code-block:: bash

   pip install calibre

.. code-block:: python

   import numpy as np
   from sklearn.model_selection import train_test_split

   from calibre import CenteredIsotonicCalibrator, IsotonicCalibrator

   # An overconfident model: true log-odds z, but the model reports 1.8 * z.
   rng = np.random.default_rng(0)
   z = rng.normal(0, 2, 4000)
   y = (rng.random(4000) < 1 / (1 + np.exp(-z))).astype(float)
   scores = 1 / (1 + np.exp(-1.8 * z))

   # Always fit the calibrator on data the model did not train on.
   s_fit, s_test, y_fit, y_test = train_test_split(
       scores, y, test_size=0.5, random_state=0
   )

   isotonic = IsotonicCalibrator().fit(s_fit, y_fit)
   centered = CenteredIsotonicCalibrator().fit(s_fit, y_fit)

   print("distinct values, isotonic:", len(np.unique(isotonic.transform(s_test))))
   print("distinct values, calibre: ", len(np.unique(centered.transform(s_test))))
   # > distinct values, isotonic: 82
   # > distinct values, calibre:  1863

Both are well calibrated. Only one of them still tells you which of two cases is
the riskier bet.

Interactive Examples
--------------------

🚀 **Start here**: :doc:`examples/index` provides hands-on Jupyter notebooks covering:

- **Getting Started**: Basic workflows and method selection
- **Validation & Evaluation**: Comprehensive quality assessment
- **Diagnostics & Troubleshooting**: Advanced plateau analysis
- **Performance Comparison**: Systematic method benchmarking

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
