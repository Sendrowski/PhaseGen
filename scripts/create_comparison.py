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

# Whether to parallelise the msprime simulation across worker processes. On macOS the workers use the 'spawn' start
# method (each re-imports phasegen), so for small state spaces the per-worker spawn overhead dwarfs the simulation and
# serial is much faster -- set PG_PARALLELIZE=0 to force serial (e.g. when regenerating many small-n fixtures at once).
parallelize = os.environ.get('PG_PARALLELIZE', '1') != '0'

from phasegen.comparison import Comparison

c = Comparison.from_yaml(file)
c.parallelize = parallelize

# cache the msprime ground truth (per-statistic caches + any configured pairwise surface grids)
c.cache_ground_truth()

# drop simulated data
c.ms.drop()

c.to_file(out)

pass
