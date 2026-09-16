# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

sys.path.append('..')

from phasegen import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PhaseGen'
author = 'Janek Sendrowski'
release = __version__
html_show_copyright = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_paramlinks',  # anchors on :param: entries, so arguments are linkable
    'autodocsumm',  # per-class method-summary table at the top of each class
    'myst_nb',
    'sphinx_design',
    'sphinx_book_theme'
]

# Resolve cross-references to the standalone sfsutils package (the site-frequency-spectrum containers, which PhaseGen
# re-exports) and to standard-library / scientific-stack types in autodoc'd signatures against their published
# documentation, rather than repeating those objects in PhaseGen's own reference.
intersphinx_mapping = {
    'sfsutils': ('https://sfsutils.readthedocs.io/en/latest/', None),
    'msprime': ('https://tskit.dev/msprime/docs/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# the autosummary class tables on the module pages are written inline (no generated stub pages)
autosummary_generate = False

typehints_use_signature = True
typehints_use_signature_return = True
typehints_document_rtype = False
typehints_fully_qualified = False

# Render unions as ``X | Y``. The Python inventory lists ``typing.Union`` as a class, which the ``data`` role
# sphinx_autodoc_typehints emits for it cannot resolve.
always_use_bars_union = True

pygments_style = 'default'

# disable notebook execution
nb_execution_mode = 'off'

# merge consecutive stdout/stderr chunks from one cell into a single output block (a mid-cell flush would
# otherwise split e.g. three prints into two separate output blocks)
nb_merge_streams = True

templates_path = ['_templates']
# 'jupyter_execute' is a myst-nb build artifact; excluding it keeps Sphinx from scanning (and recursively
# re-nesting) it as source, which otherwise floods the build with "not in any toctree" warnings.
# 'source' holds the User Guide sources, which docs/split_page.py and docs/merge_notebooks.py turn into the pages.
# 'outputs' holds the outputs of the pages, which docs/extract_outputs.py writes for version control.
exclude_patterns = ['_build', 'jupyter_execute', 'outputs', 'source', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    # Members inherited from ``collections.abc.Mapping`` carry docstring signatures that autodoc misreads as types.
    'inherited-members': 'object,Mapping',
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
    # sphinx-book-theme puts the search in the primary sidebar and clears this in its
    # theme.conf, but pydata only honours that when the key is set here, so it re-adds a
    # second search field to the header. Clear it explicitly.
    'navbar_persistent': [],
    'search_bar_text': 'Search...',
    'repository_url': 'https://github.com/Sendrowski/phasegen',
    'repository_branch': 'master',
    'use_repository_button': True,
    'use_edit_page_button': False,
    'use_issues_button': False,
    'use_download_button': False
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_js_files = ["language-tabs.js"]
html_logo = "logo.png"
html_favicon = "favicon.ico"


# Class references whose target is absent from the published inventories, mapped to the documented page: PhaseGen's
# ``SFS`` is a thin, undocumented subclass of sfsutils' ``Spectrum``, and pandas lists ``DataFrame`` only under its
# public path.
_REDIRECTS = {
    'SFS': 'sfsutils.spectrum.Spectrum',
    'phasegen.spectrum.SFS': 'sfsutils.spectrum.Spectrum',
    'pandas.core.frame.DataFrame': 'pandas.DataFrame',
}


def _resolve_redirects(app, env, node, contnode):
    """Resolve a class reference listed in ``_REDIRECTS`` against its mapped intersphinx target. The reference keeps
    its original label (the ``contnode``)."""
    from sphinx.ext.intersphinx import missing_reference

    if node.get('reftype') in ('class', 'obj') and node.get('reftarget') in _REDIRECTS:
        redirected = node.copy()
        redirected['reftarget'] = _REDIRECTS[node['reftarget']]
        return missing_reference(app, env, redirected, contnode)

    return None


def setup(app):
    app.connect('missing-reference', _resolve_redirects)
