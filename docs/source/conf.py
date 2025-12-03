"""Sphinx configuration for IsletScope documentation.

This configuration file sets up the documentation build for Read the Docs.
It enables autodoc and napolean extensions for parsing NumPy‑style docstrings,
and configures the project metadata.  The generated API reference will
automatically include all modules in the :mod:`isletscope` package.
"""

import os
import sys

# Add the project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('../../'))

project = 'IsletScope'
author = 'OpenAI Assistant'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_logo = '_static/logo.png'
html_static_path = ['_static']
