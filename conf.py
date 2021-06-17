# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import datetime


# -- Project information -----------------------------------------------------

project = 'Sigma2/Metacenter documentation'
copyright = f'{datetime.datetime.now().year}, Sigma2/Metacenter'
author = 'Sigma2/Metacenter'

# Logo setup
html_favicon = 'img/sigma2.ico'
html_logo = 'img/logo.png'

html_title = 'Sigma2 documentation'
html_short_title = 'Sigma2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['myst_parser']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'venv', '_book', 'node_modules', 'README.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# this configures where "view page source" (top right on rendered page) points to
html_context = {
    'display_github': True,
    'github_user': 'UNINETTSigma2',
    'github_repo': 'documentation',
    'github_version': 'master/' ,
}

# ignoring these because they are behind vpn/login and linkchecker cannot verify these
# or because they don't really exist
linkcheck_ignore = [
    'https://rt.uninett.no/SelfService',
    r'localhost:\d+',
    r'https://desktop.saga.sigma2.no:\d+',
    r'https://desktop.fram.sigma2.no:\d+',
    r'.*slurmbrowser.*',
    'http://www.linuxconfig.org/Bash_scripting_Tutorial',
    'http://www.wannier.org',
]
