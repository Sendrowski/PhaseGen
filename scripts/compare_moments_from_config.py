"""
Compare moments of msprime and phasegen.
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
    name = "1_epoch_migration_disparate_migration_sizes_2_each_n_6"
    file = f"resources/configs/{name}.yaml"
    dist = 'tree_height'
    stat = 'mean'
    out = f"scratch/{name}.json"

import yaml

from phasegen.comparison import Comparison

# load config from file
with open(file, 'r') as f:
    config = yaml.safe_load(f)

comp = Comparison(**config)

s1 = getattr(getattr(comp.ph, dist), stat)
s2 = getattr(getattr(comp.ms, dist), stat)

if hasattr(s1, 'plot'):
    s1 = s1.plot()
    s2 = s2.plot()

pass
