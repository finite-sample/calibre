"""Tests for the optional-matplotlib contract.

calibre promises that installing it does not drag matplotlib in. That promise is
easy to break by accident -- one module-level ``import matplotlib`` anywhere
under ``calibre/`` is enough -- and impossible to notice locally, because the dev
environment always has matplotlib. These tests are the guard.
"""

from __future__ import annotations

import builtins
import subprocess
import sys

import pytest

import calibre
import calibre.plots
from calibre.plots._deps import require_matplotlib


def test_importing_calibre_does_not_import_matplotlib():
    """``import calibre`` must not pull matplotlib in.

    Run in a subprocess because this test session has already imported
    matplotlib, so an in-process check on ``sys.modules`` would pass no matter
    what calibre did.
    """
    code = (
        "import sys\n"
        "import calibre\n"
        "bad = sorted(m for m in sys.modules if m.startswith('matplotlib'))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_calibre_plots_does_not_import_matplotlib():
    """Importing the plots package must not import matplotlib either.

    The import happens inside each drawing function, so merely reaching for the
    namespace stays cheap and stays safe on a machine without matplotlib.
    """
    code = (
        "import sys\n"
        "import calibre.plots\n"
        "bad = sorted(m for m in sys.modules if m.startswith('matplotlib'))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_require_matplotlib_returns_the_modules():
    """The happy path hands back matplotlib and pyplot."""
    mpl, plt = require_matplotlib()
    assert mpl.__name__ == "matplotlib"
    assert plt.__name__ == "matplotlib.pyplot"


def test_missing_matplotlib_names_the_install_command(monkeypatch):
    """The error must tell the user exactly what to install.

    Simulating absence rather than skipping: CLAUDE.md requires tests to fail
    rather than skip, and a test that skips when matplotlib is absent would
    never run anywhere.
    """
    real_import = builtins.__import__

    def _no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)
    for module in list(sys.modules):
        if module.startswith("matplotlib"):
            monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(ImportError) as excinfo:
        require_matplotlib()

    message = str(excinfo.value)
    assert "calibre[plots]" in message
    assert "Everything else in calibre works without it." in message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_plots_is_reachable_as_an_attribute_of_calibre():
    """:pep:`562` lookup exposes the subpackage without importing it eagerly."""
    assert calibre.plots is sys.modules["calibre.plots"]


def test_unknown_attribute_still_raises():
    """The lazy hook must not swallow genuine typos."""
    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        _ = calibre.nonexistent


def test_dir_advertises_plots():
    """``dir(calibre)`` should list the lazy attribute."""
    assert "plots" in dir(calibre)
