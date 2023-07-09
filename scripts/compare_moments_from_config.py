"""
Compare moments of msprime and ph.
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
    dist = snakemake.params.dist
    stat = snakemake.params.stat
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "resources/configs/4_epoch_up_down_n_10.yaml"
    dist = 'tree_height'
    stat = 'var'
    out = "scratch/4_epoch_up_down_n_10.json"

import yaml

from phasegen.comparison import Comparison

# load config from file
with open(file, 'r') as f:
    config = yaml.safe_load(f)

comp = Comparison(**config)

s1 = getattr(getattr(comp.ph, dist), stat)
s2 = getattr(getattr(comp.ms, dist), stat)

pass
