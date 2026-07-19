# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys

sys.path.append('..')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PhaseGen'
year = datetime.datetime.now().year
copyright = f'{year}, Janek Sendrowski'
author = 'Janek Sendrowski'
release = '1.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'autodocsumm',  # per-class method-summary table at the top of each class
    'myst_nb',
    'sphinx_book_theme'
]

# Resolve cross-references to the standalone sfsutils package (the site-frequency-spectrum containers, which PhaseGen
# re-exports) and to standard-library / scientific-stack types in autodoc'd signatures against their published
# documentation, rather than repeating those objects in PhaseGen's own reference.
intersphinx_mapping = {
    'sfsutils': ('https://sfsutils.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# the autosummary class tables on the module pages are written inline (no generated stub pages)
autosummary_generate = False

typehints_use_signature = True
typehints_fully_qualified = False

pygments_style = 'default'

# disable notebook execution
nb_execution_mode = 'off'

# merge consecutive stdout/stderr chunks from one cell into a single output block (a mid-cell flush would
# otherwise split e.g. three prints into two separate output blocks)
nb_merge_streams = True

templates_path = ['_templates']
# 'jupyter_execute' is a myst-nb build artifact; excluding it keeps Sphinx from scanning (and recursively
# re-nesting) it as source, which otherwise floods the build with "not in any toctree" warnings.
exclude_patterns = ['_build', 'jupyter_execute', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
    # autodocsumm: prepend a compact summary table to each documented object -- a class table at the top of every
    # module page and a method table at the top of every class. Limit it to those two sections (``;;``-separated):
    # the Attributes summary just duplicates the per-attribute docs below it.
    'autosummary': True,
    'autosummary-sections': 'Classes;;Methods',
    'autosummary-nosignatures': True  # summary tables list bare names; signatures stay in the detailed docs below
}

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'search_bar_text': 'Search...',
    'repository_url': 'https://github.com/Sendrowski/phasegen',
    'repository_branch': 'master',
    'use_repository_button': True,
    'use_edit_page_button': False,
    'use_issues_button': False
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "logo.png"
html_favicon = "favicon.ico"


def _resolve_sfs_to_sfsutils(app, env, node, contnode):
    """Redirect references to PhaseGen's ``SFS`` (a thin, undocumented subclass of :class:`sfsutils.spectrum.Spectrum`)
    to sfsutils' ``Spectrum`` documentation, so its return-type annotations resolve to a link rather than rendering as
    bare text. The reference keeps its ``SFS`` label (the original ``contnode``) and points at the sfsutils page."""
    from sphinx.ext.intersphinx import missing_reference

    if node.get('reftype') in ('class', 'obj') and node.get('reftarget') in ('SFS', 'phasegen.spectrum.SFS'):
        redirected = node.copy()
        redirected['reftarget'] = 'sfsutils.spectrum.Spectrum'
        return missing_reference(app, env, redirected, contnode)

    return None


def setup(app):
    app.connect('missing-reference', _resolve_sfs_to_sfsutils)
