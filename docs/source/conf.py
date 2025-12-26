# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Calibre'
copyright = '2024, Gaurav Sood'
author = 'Gaurav Sood'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',    # Generate summary tables
    'sphinx.ext.napoleon',       # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.coverage',       # Coverage extension
    'sphinx.ext.mathjax',        # Math support
    'sphinx_autodoc_typehints',  # Type hints support
    'sphinx_copybutton',         # Copy button for code blocks
    'nbsphinx',                  # Jupyter notebook support
    'myst_parser',               # Markdown support
]

# Source file configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc configuration --------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

autosummary_generate = True
autodoc_typehints = 'description'

# -- Napoleon configuration -------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Intersphinx configuration ----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = project
html_static_path = ['_static']

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#336790",
        "color-brand-content": "#336790",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4db8ff",
        "color-brand-content": "#4db8ff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
}

# Custom sidebar templates for furo theme
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Options for nbsphinx ---------------------------------------------------

nbsphinx_execute = 'always'  # Execute notebooks during build to show outputs
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# Notebook execution timeout (in case we change to auto-execute)
nbsphinx_timeout = 120

# Configure nbsphinx to handle both mime types
nbsphinx_custom_formats = {
    'text/x-rst': 'rst',
    'text/restructuredtext': 'rst'
}

# Patch nbconvert to support the expected mime type
import nbconvert
original_get_template_names = nbconvert.RSTExporter.get_template_names

def patched_get_template_names(self):
    # Override to support text/restructuredtext mime type
    if hasattr(self, 'output_mimetype') and self.output_mimetype == 'text/restructuredtext':
        self.output_mimetype = 'text/x-rst'
    return original_get_template_names(self)

nbconvert.RSTExporter.get_template_names = patched_get_template_names


# -- Options for LaTeX output -----------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'printindex': '',
}

latex_documents = [
    ('index', 'calibre.tex', 'Calibre Documentation',
     'Gaurav Sood', 'manual'),
]