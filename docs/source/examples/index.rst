Examples and Tutorials
======================

This section provides comprehensive examples and tutorials for using Calibre.

.. toctree::
   :maxdepth: 2

   basic_usage
   advanced_usage
   benchmarks

Interactive Jupyter Notebooks
------------------------------

We provide focused, executable Jupyter notebooks for hands-on learning:

.. toctree::
   :maxdepth: 1
   :caption: Interactive Examples

   ../notebooks/01_getting_started
   ../notebooks/02_validation_and_evaluation
   ../notebooks/03_diagnostics_and_troubleshooting
   ../notebooks/04_performance_comparison
   ../notebooks/05_evaluating_calibration
   ../notebooks/06_multiclass

Notebook Overview
~~~~~~~~~~~~~~~~~

📚 **Getting Started** (:doc:`../notebooks/01_getting_started`)
   - Basic calibration workflow with realistic ML predictions
   - Choosing the right calibrator for your data
   - Visual validation with reliability diagrams
   - Quick start guide for new users

🔍 **Validation and Evaluation** (:doc:`../notebooks/02_validation_and_evaluation`)
   - Comprehensive calibration quality assessment
   - Mathematical property validation (bounds, monotonicity, granularity)
   - Performance across different miscalibration patterns
   - Edge case testing and robustness analysis

🩺 **Diagnostics and Troubleshooting** (:doc:`../notebooks/03_diagnostics_and_troubleshooting`)
   - Plateau diagnostic tools for isotonic regression
   - Distinguishing genuine vs. limited-data flattening
   - Progressive sampling: how granularity changes with sample size
   - Decision framework for method selection

⚡ **Performance Comparison** (:doc:`../notebooks/04_performance_comparison`)
   - Systematic comparison across all calibration methods
   - Performance on overconfident, underconfident, and distorted predictions
   - Computational efficiency and method ranking
   - Guidelines for choosing the optimal method

📐 **Evaluating Calibration** (:doc:`../notebooks/05_evaluating_calibration`)
   - CORP reliability diagrams: no bin count to choose
   - Consistency and confidence bands for uncertainty quantification
   - The MCB / DSC / UNC decomposition — separating fixable miscalibration
     from lost discrimination
   - Why in-sample calibration error is identically zero, and what to use instead

🎯 **Multiclass** (:doc:`../notebooks/06_multiclass`)
   - Two regimes with different winners, measured against known true probabilities
   - ``miscalibration_profile``: telling which regime your data is in
   - Why temperature scaling can never change accuracy, and when that is the problem
   - The within-class reordering that no standard metric reveals

Running the Notebooks
~~~~~~~~~~~~~~~~~~~~~~

To run these notebooks locally:

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   uv sync --all-extras --dev
   jupyter notebook docs/source/notebooks/

Or install required dependencies:

.. code-block:: bash

   pip install matplotlib  # only needed to plot the examples yourself

Additional Documentation Examples
----------------------------------

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~

The :doc:`basic_usage` section covers:

- Simple calibration workflows
- Choosing the right calibration method
- Evaluating calibration quality
- Common use cases and patterns

Advanced Usage Examples
~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`advanced_usage` section includes:

- Multi-class calibration strategies
- Handling imbalanced datasets
- Cross-validation for calibration
- Custom calibration pipelines

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

The :doc:`benchmarks` section provides:

- Comparative analysis of different methods
- Performance on various dataset types
- Computational efficiency comparisons