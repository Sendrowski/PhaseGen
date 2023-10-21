"""
Plot comparison of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-03-11"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "resources/configs/test_plot_pdf_const_tree_height.yaml"

    out = "scratch/test_comp.png"

import os

import yaml
from matplotlib import pyplot as plt

from phasegen import Comparison

# load config from file
with open(file, 'r') as f:
    config = yaml.safe_load(f)

s = Comparison(**config['config'])

# plot
stats = config['stats']
for stat in stats:
    prop = list(stats[stat].keys())[0]
    func = stats[stat][prop]
    getattr(getattr(getattr(s, stat), prop), func)(show=False, clear=False, label=stat)

name = os.path.splitext(os.path.basename(file))[0]
plt.title(config['config'], fontsize=10)
plt.legend()

# save plot
plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()
