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
    end_time = snakemake.params.end_time
    num_replicates = snakemake.params.num_replicates
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 5  # sample size
    times = [0, 0.1]
    pop_sizes = [10, 0.1]
    start_time = 0
    end_time = None
    num_replicates = 100000
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
    ploidy=1,
    start_time=start_time,
    end_time=end_time
)

ts: tskit.TreeSequence
heights = np.zeros(num_replicates)
total_branch_lengths = np.zeros(num_replicates)
for i, ts in enumerate(g):
    total_branch_lengths[i] = ts.first().total_branch_length
    heights[i] = ts.max_time

# get moments of tree height
height = dict(
    mu=np.mean(heights),
    var=np.var(heights),
    mu2=np.mean(heights ** 2),
)

if end_time is not None:
    # get moments of time spent in absorbing state
    time_in_absorption = dict(
        mu=np.mean(end_time - heights[heights != end_time]),
        var=np.var(end_time - heights[heights != end_time]),
        mu2=np.mean((end_time - heights[heights != end_time]) ** 2),
    )

# get moments of branch length
total_branch_length = dict(
    mu=np.mean(total_branch_lengths),
    var=np.var(total_branch_lengths),
    mu2=np.mean(total_branch_lengths ** 2),
)

JSON.save(dict((k, globals()[k]) for k in ['n', 'height', 'total_branch_length']), out)

pass
