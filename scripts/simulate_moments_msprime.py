"""
Simulate moments of given population scenario using msprime.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-02-25"

import msprime as ms
import numpy as np
import tskit
from typing import Generator
import JSON

try:
    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
    num_replicates = snakemake.params.num_replicates
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 5  # sample size
    pop_sizes = [0.12, 1, 0.01, 10]
    times = [0, 0.3, 1, 1.4]
    num_replicates = 100000  # number of samples
    out = "scratch/msprime.json"

# configure demography
d = ms.Demography()
d.add_population(initial_size=pop_sizes[0])

for i in range(1, len(pop_sizes)):
    d.add_population_parameters_change(time=times[i], initial_size=pop_sizes[i])

# simulate trees
g: Generator = ms.sim_ancestry(
    samples=n,
    num_replicates=num_replicates,
    demography=d,
    model=ms.StandardCoalescent(),
    ploidy=1
)

ts: tskit.TreeSequence
heights = np.zeros(num_replicates)
total_branch_lengths = np.zeros(num_replicates)
for i, ts in enumerate(g):
    t: tskit.Tree = ts.first()
    total_branch_lengths[i] = t.total_branch_length
    heights[i] = t.time(t.root)

# get moments of tree height
height = dict(
    mu=np.mean(heights),
    var=np.var(heights)
)

# get moments of branch length
total_branch_length = dict(
    mu=np.mean(total_branch_lengths),
    var=np.var(total_branch_lengths)
)

JSON.save(dict((k, globals()[k]) for k in ['n', 'height', 'total_branch_length']), out)
