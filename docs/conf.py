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
release = '1.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_nb',
    'sphinx_book_theme'
]

typehints_use_signature = True
typehints_fully_qualified = False

pygments_style = 'default'

# disable notebook execution
nb_execution_mode = 'off'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True
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
