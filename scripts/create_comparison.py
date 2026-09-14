"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

import os

try:
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    name = "1_epoch_2_loci_2_pops_n_2_r_1"
    file = f"resources/configs/{name}.yaml"
    out = f"scratch/{name}.json"

import phasegen as pg
from phasegen.comparison import Comparison

# PG_PARALLELIZE=0 runs the msprime simulations serially, which is faster for small state spaces
pg.Settings.parallelize = os.environ.get('PG_PARALLELIZE', '1') != '0'

# the spawned workers re-import this module under a different ``__name__``; without the guard they would re-run the
# simulation from the top and the pool would fail to start
if __name__ == '__main__':
    c = Comparison.from_yaml(file)

    # cache the ground truth (msprime for the top-level stats, the sampler for a nested ``empirical`` sub-spec -- each
    # only if present, so a config may validate against either operand or both)
    c.cache_ground_truth()

    # drop the simulated data of whichever operands were built (the cached stats/surfaces are retained + serialized)
    if 'ms' in c.__dict__:
        c.ms._drop()
    if 'empirical' in c.__dict__:
        c.empirical._drop()

    c.to_file(out)
