Installation
============

Requirements
------------

Calibre requires Python 3.12 or later.

Installing from PyPI
---------------------

The easiest way to install Calibre is from PyPI using pip:

.. code-block:: bash

   pip install calibre

This will install Calibre along with all required dependencies:

- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- cvxpy >= 1.2.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

Installing from Source
----------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   pip install -e .

Development Installation
------------------------

For development, install with additional development dependencies:

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   pip install -e ".[dev]"

This includes:

- pytest >= 6.0.0 (testing)
- pytest-cov >= 2.12.0 (coverage)
- black >= 21.5b2 (code formatting)
- isort >= 5.9.0 (import sorting)
- flake8 >= 3.8.0 (linting)

Documentation Dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"

This includes:

- sphinx >= 7.0.0
- sphinx-rtd-theme >= 2.0.0
- sphinx-autodoc-typehints >= 1.24.0
- nbsphinx >= 0.9.0
- myst-parser >= 2.0.0

Verifying Installation
----------------------

To verify that Calibre is installed correctly:

.. code-block:: python

   import calibre
   print(f"Calibre version: {calibre.__version__}")

You should see the version number printed without any errors.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'cvxpy'**

This usually means CVXPY failed to install. Try installing it separately:

.. code-block:: bash

   pip install cvxpy

**Installation fails on macOS**

If you encounter issues on macOS, try using conda instead:

.. code-block:: bash

   conda install -c conda-forge cvxpy
   pip install calibre

**Memory errors during installation**

If you encounter memory errors, try installing with pip's no-cache option:

.. code-block:: bash

   pip install --no-cache-dir calibre

Getting Help
~~~~~~~~~~~~

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/finite-sample/calibre/issues>`_ for similar problems
2. Create a new issue with details about your system and error messages
3. Include the output of ``pip --version`` and ``python --version``