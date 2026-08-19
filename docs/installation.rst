Installation
============

Requirements
------------

Calibre requires Python 3.12 or later.

Installing from PyPI
---------------------

.. code-block:: bash

   pip install calibre

This installs Calibre along with its runtime dependencies:

- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- cvxpy >= 1.2.0

That is the whole list. pandas and matplotlib were runtime dependencies before
0.7.0 and are not any more — nothing in the package imported them.

Plotting
--------

:doc:`api/plots` needs matplotlib, which is an optional extra:

.. code-block:: bash

   pip install 'calibre[plots]'

Everything else in calibre works without it. Importing ``calibre`` never imports
matplotlib, so the core package stays light for anyone who only needs the
numbers.

Installing from Source
----------------------

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   pip install -e .

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   uv sync --all-extras --dev
   uv run pytest

.. important::

   Development dependencies are a :pep:`735` ``[dependency-groups]`` entry, so
   ``pip install -e ".[dev]"`` does **not** work. Use ``uv sync --all-extras
   --dev``.

Tooling is ruff (formatting and linting) plus pytest, pytest-cov and pyright.
black, isort and flake8 were replaced by ruff in 0.4.1.

Documentation Dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

   uv sync --all-extras --dev
   make docs

The docs group pulls sphinx, furo, myst-parser, sphinx-copybutton,
sphinx-autodoc-typehints and nbsphinx, plus matplotlib and pandas, which the
example notebooks use.

Verifying Installation
----------------------

.. code-block:: python

   import calibre

   print(f"Calibre version: {calibre.__version__}")

Troubleshooting
---------------

**ImportError: No module named 'cvxpy'**

CVXPY is a hard runtime dependency because ``nearly_isotonic.py`` imports it at
module level. If it failed to install, try it on its own:

.. code-block:: bash

   pip install cvxpy

On macOS, conda is often easier:

.. code-block:: bash

   conda install -c conda-forge cvxpy
   pip install calibre

**Memory errors during installation**

.. code-block:: bash

   pip install --no-cache-dir calibre

Getting Help
~~~~~~~~~~~~

If you hit an installation problem, check the `GitHub Issues
<https://github.com/finite-sample/calibre/issues>`_ for similar reports, then
open a new one including the output of ``pip --version`` and ``python
--version``.
