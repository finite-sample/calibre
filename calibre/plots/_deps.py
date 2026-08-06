"""Lazy access to matplotlib.

matplotlib is an optional dependency. Nothing under ``calibre/`` may import it at
module level, because that would make it a runtime dependency of the whole
package by the back door. Every plotting function calls
:func:`require_matplotlib` as its first statement instead.
"""

from __future__ import annotations

from types import ModuleType

_INSTALL_HINT = (
    "calibre.plots needs matplotlib, which calibre does not install by default.\n"
    "    pip install 'calibre[plots]'\n"
    "    uv add 'calibre[plots]'\n"
    "Everything else in calibre works without it."
)


def require_matplotlib() -> tuple[ModuleType, ModuleType]:
    """Import matplotlib, or raise an error naming the install command.

    Returns
    -------
    tuple of module
        ``(matplotlib, matplotlib.pyplot)``.

    Raises
    ------
    ImportError
        If matplotlib is not installed. The original exception is chained.

    Examples
    --------
    >>> mpl, plt = require_matplotlib()
    >>> mpl.__name__
    'matplotlib'
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    return mpl, plt
