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

Calibration is a critical step in deploying machine learning models. While techniques like isotonic regression have been standard for this task, they come with significant limitations:

1. **Loss of granularity**: Traditional isotonic regression often collapses many distinct probability values into a small number of unique values, which can be problematic for decision-making.

2. **Rigid monotonicity**: Perfect monotonicity might not always be necessary or beneficial; small violations might be acceptable if they better preserve the information content of the original predictions.

Calibre addresses these limitations by implementing a suite of advanced calibration techniques that provide more nuanced control over model probability calibration. Its methods are designed to preserve granularity while still favoring a generally monotonic trend.

Features
--------

- **Nearly-isotonic regression**: Allows controlled violations of monotonicity to better preserve data granularity
- **I-spline calibration**: Uses monotonic splines for smooth calibration functions
- **Relaxed PAVA**: Ignores "small" violations based on percentile thresholds in the data
- **Regularized isotonic regression**: Adds L2 regularization to standard isotonic regression for smoother calibration curves while maintaining monotonicity
- **Locally smoothed isotonic**: Applies Savitzky-Golay filtering to isotonic regression results to reduce the "staircase effect" while preserving monotonicity
- **Adaptive smoothed isotonic**: Uses variable-sized smoothing windows based on data density to provide better detail in dense regions and smoother curves in sparse regions

Quick Start
-----------

Install Calibre:

.. code-block:: bash

   pip install calibre

Basic usage:

.. code-block:: python

   import numpy as np
   from calibre import NearlyIsotonicCalibrator
   
   # Example data: model predictions and true binary outcomes
   np.random.seed(42)
   y_pred = np.sort(np.random.uniform(0, 1, 1000))
   y_true = np.random.binomial(1, y_pred, 1000)
   
   # Calibrate with nearly isotonic regression
   calibrator = NearlyIsotonicCalibrator(lam=1.0)
   calibrator.fit(y_pred, y_true)
   y_calibrated = calibrator.transform(y_pred)

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
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`