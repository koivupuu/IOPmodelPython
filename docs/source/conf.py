# docs/source/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../../Python'))  # adjust if needed

project = 'IOPmodelPython'
author = 'Miika Koivisto'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

html_theme = 'furo'  # or 'sphinx_rtd_theme'

templates_path = ['_templates']
exclude_patterns = []
