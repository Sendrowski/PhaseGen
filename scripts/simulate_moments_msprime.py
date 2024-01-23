""""
Simulate moments using msprime
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    times = snakemake.params.times
    pop_sizes = snakemake.params.pop_sizes
    migration_rates = snakemake.params.migration_rates
    start_time = snakemake.params.start_time
    end_time = snakemake.params.end_time
    exclude_unfinished = snakemake.params.exclude_unfinished
    exclude_finished = snakemake.params.exclude_finished
    growth_rate = snakemake.params.growth_rate
    N0 = snakemake.params.N0
    num_replicates = snakemake.params.get('num_replicates', 10000)
    n_threads = snakemake.params.get('n_threads', 100)
    parallelize = snakemake.params.get('parallelize', True)
    dist = snakemake.params.dist
    stat = snakemake.params.stat
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 10  # sample size
    times = dict(pop_0=[0, 1], pop_1=[0, 2])
    pop_sizes = dict(pop_0=[1, 5], pop_1=[2, 0.5])
    migration_rates = {('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}
    start_time = 1
    end_time = None
    exclude_unfinished = True
    exclude_finished = False
    growth_rate = None
    N0 = 1
    num_replicates = 100000
    n_threads = 1000
    parallelize = True
    dist = 'sfs'
    stat = 'mean'
    out = "scratch/test_comp.png"

import phasegen as pg

ms = pg._MsprimeCoalescent(
    n=n,
    demography=pg.PiecewiseConstantDemography(
        pop_sizes=pop_sizes,
        times=times,
        migration_rates=migration_rates
    ),
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

if hasattr(_, 'plot'):
    _.plot(out)

pass
