"""
Plot comparison of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

try:
    testing = False
    file = snakemake.input[0]
    stats = snakemake.params.stats
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "resources/configs/1_epoch_n_2.yaml"
    stats = {'ph': {'tree_height': 'pdf'}, 'ms': {'tree_height': 'pdf'}}
    out = "scratch/test_comp.png"

import os

from matplotlib import pyplot as plt

from phasegen.comparison import Comparison

s = Comparison.from_yaml(file)

# plot
for stat in stats:
    prop = list(stats[stat].keys())[0]
    func = stats[stat][prop]
    getattr(getattr(getattr(s, stat), prop), func).plot(show=False, clear=False, label=stat)

name = os.path.splitext(os.path.basename(file))[0]
plt.title(name, fontsize=10)
plt.legend()

# save plot
plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()
