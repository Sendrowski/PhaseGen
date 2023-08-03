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
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    name = "1_epoch_migration_one_each_n_2"
    file = f"resources/configs/{name}.yaml"
    out = f"scratch/{name}.json"

import yaml

from phasegen.comparison import Comparison

# load config from file
with open(file, 'r') as f:
    config = yaml.safe_load(f)

c = Comparison(**config)

# touch msprime stats
c.ms._touch()

# drop computed stats
c.ms._drop()

c.to_file(out)
