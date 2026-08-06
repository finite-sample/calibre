Performance Benchmarks
======================

.. note::

   This page previously carried a star-rating table comparing the calibrators
   across "calibration error", "granularity preservation", "speed" and
   "robustness". Those ratings had no provenance — no script produced them and
   no measurement backed them — so they have been removed rather than updated.
   A reproducible benchmark with committed results is being built to replace
   them.

What we can currently claim, and where it comes from
----------------------------------------------------

Held out over 30 random datasets (an overconfident logistic model; fit on one
half, scored on the other). Lower Brier is better; ΔBrier is the improvement
over leaving the model uncalibrated.

.. list-table::
   :header-rows: 1

   * - Method
     - Brier
     - ΔBrier
     - ECE
     - Distinct values
   * - Uncalibrated
     - 0.1581
     - —
     - 0.0826
     - 2000
   * - ``IsotonicCalibrator``
     - 0.1515
     - +0.0066
     - 0.0265
     - **56**
   * - ``CenteredIsotonicCalibrator``
     - 0.1511
     - +0.0070
     - 0.0272
     - 1874
   * - ``SplineCalibrator``
     - **0.1509**
     - **+0.0072**
     - 0.0258
     - 1999
   * - ``RelaxedPAVACalibrator(min_slope=1e-5)``
     - 0.1515
     - +0.0066
     - 0.0269
     - 1941

Against plain isotonic on held-out Brier: ``CenteredIsotonicCalibrator`` wins
24/30 seeds, ``SplineCalibrator`` 26/30, ``RelaxedPAVACalibrator`` 28/30.

Two things worth reading honestly off that table. The Brier gains over isotonic
are **small** — the large win is the last column, ~1900 distinct values instead
of 56. And ECE barely moves, because ECE is computed on bins and is largely
blind to the resolution you just recovered; that is a reason to be careful with
ECE, not a reason to prefer isotonic.

Interactive comparison
----------------------

The most comprehensive comparison currently available is the notebook, which
runs end to end when the docs are built:

- :doc:`../notebooks/04_performance_comparison` — systematic comparison across
  methods and miscalibration patterns, with timings.
- :doc:`../notebooks/05_evaluating_calibration` — how to evaluate a calibrator
  without fooling yourself.

To run it yourself:

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   uv sync --all-extras --dev
   uv run jupyter notebook docs/source/notebooks/04_performance_comparison.ipynb
