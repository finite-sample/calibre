Examples and Tutorials
======================

This section provides comprehensive examples and tutorials for using Calibre.

.. toctree::
   :maxdepth: 2

   basic_usage
   advanced_usage
   benchmarks

Available Examples
------------------

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
- Interactive Jupyter notebook examples

Jupyter Notebooks
------------------

Interactive examples are available in the repository's ``examples/`` directory:

- ``benchmark.ipynb``: Comprehensive performance comparison
- ``validation/calibration_validation.ipynb``: Validation examples

To run the notebooks locally:

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   pip install -e ".[dev]"
   jupyter notebook examples/