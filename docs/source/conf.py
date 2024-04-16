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
import os
import sys
import re
import yaml

sys.path.insert(0, os.path.abspath("../../"))

import boario


# -- Project information -----------------------------------------------------

project = "BoARIO"
copyright = "2022, Samuel Juhel"
author = "Samuel Juhel"
html_last_updated_fmt = "%b %d, %Y"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
#    'autoapi.sphinx',
]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = False

# automatically generate api references
autosummary_generate = ["boario-api-reference.rst"]

# bibfile
bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["archive.py",
                    "**.ipynb_checkpoints",
                    ]

# The master toctree document.
master_doc = "index"

# copybutton config
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
version = boario.__version__
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_end": ["navbar-icon-links", "last-updated", "theme-switcher", "version-switcher"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "use_edit_page_button": True,
    "show_nav_level": 2,
    "switcher": {
        "json_url": "https://spjuhel.github.io/BoARIO/develop/en/_static/switcher.json",
        "version_match": version,
    }
}

html_context = {
    "default_mode": "light",
    "display_github": True,  # Integrate GitHub
    "github_user": "spjuhel",  # Username
    "github_repo": "BoARIO",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/source/",  # Path in the checkout to the docs root
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The entries in this file are checked regularly for validity via the Github Action
# sited at github.com/bskinn/intersphinx-gist.
# Please feel free to post an issue at that repo if any of these mappings don't work for you,
# or if you're having trouble constructing a mapping for a project not listed here.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pymrio": ("https://pymrio.readthedocs.io/en/latest", None),
}

html_js_files = ["js/custom.js"]

html_css_files = [
    "css/custom.css",
]

autodoc_typehints = "none"
#autodoc_typehints_description_target = ""
add_module_names = False

# Keep members order from source :
autodoc_member_order = "bysource"

# Keep __init__ method
autoclass_content = "class"

# autoapi config
autoapi_modules = {"boario": {"output": "auto", "prune":True, "override":True}}

# mathjax config
mathjax3_config = {
    "chtml": {"displayAlign": "left", "displayIndent": "2em"},
    "loader": {"load": ["[tex]/color"]},
    "tex": {
        "packages": {"[+]": ["color"]},
        "macros": {
            "colvec": ["\\begin{bmatrix}#1 \\\\ \\vdots\\\\ #2 \\end{bmatrix}", 2],
            "rfirmsset": ["\\mathbf{F}^{#1}", 1],
            "mdefentryval": [
                "(#1_{#3#5}( #2 )^{#7#8})_{\\substack{#3 \\in #4 \\\\ #5 \\in #6}}",
                8,
            ],
            "mdefentry": [
                "(#1_{#3#5}^{#7#8})_{\\substack{#3 \\in #4 \\\\ #5 \\in #6}}",
                8,
            ],
            "irowsum": "\\begin{bmatrix} 1 \\\\ \\vdots \\\\ 1 \\end{bmatrix}",
            "isum": "\\underset{s \\times p}{\\begin{bmatrix} 1 & \\cdots & 0 & & 1 & \\cdots & 0\\\\ \\vdots & \\ddots & \\vdots & \\cdots &\\vdots & \\ddots & \\vdots\\\\ 0 & \\cdots & 1 & & 0 & \\cdots & 1 \\end{bmatrix}}",
            "ioorders": "\\mathbf{O}",
            "sectorsset": "\\mathbb{S}",
            "sectorssetsize": "n",
            "regionsset": "\\mathbb{R}",
            "regionssetsize": "m",
            "firmsset": "\\mathbb{F}",
            "firmssetsize": "p",
            "catfdset": "\\mathbb{C}^{\\textrm{fd}}",
            "catfdsetsize": "r",
            "catvaset": "\\mathbb{C}^{\\textrm{va}}",
            "catvasetsize": "q",
            "ioz": "\\mathbf{Z}",
            "ioy": "\\mathbf{Y}",
            "iov": "\\mathbf{V}",
            "iox": "\\mathbf{x}",
            "ioa": "\\mathbf{A}",
            "iok": "\\mathbf{K}",
            "ioava": "\\mathbf{A}_{\\textrm{va}}",
            "ioinv": "\\mathbf{\\Omega}",
            "Damage": "\\mathbf{\\Gamma}",
            "damage": "\\gamma",
        },
    },
}

# build_all_docs = os.environ.get("build_all_docs")
# pages_root = os.environ.get("pages_root", "")

# if build_all_docs is not None:
#   current_language = os.environ.get("current_language")
#   current_version = os.environ.get("current_version")

#   html_context = {
#     'current_language' : current_language,
#     'languages' : [],
#     'current_version' : current_version,
#     'versions' : [],
#   }

#   if (current_version == 'latest'):
#     html_context['languages'].append(['en', pages_root])
#     #html_context['languages'].append(['de', pages_root+'/de'])

#   if (current_language == 'en'):
#     html_context['versions'].append(['latest', pages_root])
#   #if (current_language == 'de'):
#   #  html_context['versions'].append(['latest', pages_root+'/de'])

#   with open("versions.yaml", "r") as yaml_file:
#     docs = yaml.safe_load(yaml_file)

#   for version, details in docs.items():
#     html_context['versions'].append([version, pages_root+'/'+version+'/'+current_language])
