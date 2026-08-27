"""Sphinx configuration — fleet standard via py-canon, plus this repo's extras."""

import doctest as _doctest

from py_canon.sphinx import configure

config = globals()
configure(config)

# MyST tutorials are executed at build time, so rendered results always come from
# the current library. myst-nb parses the Markdown without requiring pandoc.
config["extensions"].remove("myst_parser")  # bundled by myst_nb; listing both errors
config["extensions"] += ["myst_nb", "sphinx.ext.mathjax"]
config["source_suffix"][".md"] = "myst-nb"
nb_execution_mode = "force"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# The tutorials write math as $...$ / $$...$$.
config["myst_enable_extensions"].append("dollarmath")

html_static_path = ["_static"]

# pytest runs the same examples via --doctest-modules, where each docstring sees
# its module's globals; sphinx's doctest builder runs them in an empty namespace.
# Recreate the module context here, and mirror pytest's doctest_optionflags so
# the two runners accept the same expected output.
doctest_default_flags = _doctest.ELLIPSIS | _doctest.NORMALIZE_WHITESPACE
doctest_global_setup = """
import matplotlib
matplotlib.use("Agg")
import numpy as np
from calibre import *
from calibre._core import *
from calibre.diagnostics import *
from calibre.evaluation import *
from calibre.metrics import *
from calibre.multiclass import *
from calibre.plots._style import color_cycle, style_context
from calibre.selection import *
from calibre.utils import *
"""

# The API pages cross-reference the scientific stack's types.
config["intersphinx_mapping"].update(
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "sklearn": ("https://scikit-learn.org/stable/", None),
        "matplotlib": ("https://matplotlib.org/stable/", None),
    }
)
