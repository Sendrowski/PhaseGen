""""
Simulate moments using msprime
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-03-11"

import numpy as np

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
    start_time = snakemake.params.start_time
    end_time = snakemake.params.end_time
    exclude_unfinished = snakemake.params.exclude_unfinished
    exclude_finished = snakemake.params.exclude_finished
    growth_rate = snakemake.params.growth_rate
    N0 = snakemake.params.N0
    alpha = snakemake.params.get('alpha', np.eye(1, n - 1, 0)[0])
    num_replicates = snakemake.params.get('num_replicates', 10000)
    n_threads = snakemake.params.get('n_threads', 100)
    parallelize = snakemake.params.get('parallelize', True)
    dist = snakemake.params.dist
    stat = snakemake.params.stat
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 3  # sample size
    times = [0, 1]
    pop_sizes = [1, 1]
    start_time = 1
    end_time = None
    exclude_unfinished = True
    exclude_finished = False
    growth_rate = None
    N0 = 1
    alpha = np.eye(1, n, 0)[0]
    num_replicates = 100000
    n_threads = 1000
    parallelize = True
    dist = 'sfs'
    stat = 'm2'
    out = "scratch/test_comp.png"

from phasegen import MsprimeCoalescent

ms = MsprimeCoalescent(
    n=n,
    pop_sizes=pop_sizes,
    times=times,
    growth_rate=growth_rate,
    N0=N0,
    start_time=start_time,
    end_time=end_time,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize,
    exclude_unfinished=exclude_unfinished,
    exclude_finished=exclude_finished
)

ms.simulate()

_ = getattr(getattr(ms, dist), stat)

pass
