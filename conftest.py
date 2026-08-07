"""Pytest configuration shared by the test suite and the collected doctests.

This lives at the repository root rather than in ``tests/`` because
``testpaths`` includes ``calibre`` and ``--doctest-modules`` collects docstrings
from ``calibre/plots/``, which draw.

Deliberately imports nothing but the standard library: setting ``MPLBACKEND``
only has an effect before matplotlib is first imported, so this file must not be
the thing that imports it.
"""

from __future__ import annotations

import os
import sys

import pytest

# Headless, non-interactive, and deterministic. setdefault so an explicit
# MPLBACKEND in the environment still wins.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test or doctest opened.

    matplotlib warns once more than 20 figures are open, and the plotting
    doctests alone open more than that. Guarded on ``sys.modules`` so that tests
    which never touch matplotlib do not import it.

    Yields:
    ------
    None
        Control returns to the test.
    """
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
