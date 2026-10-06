"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

import matplotlib.pyplot as plt

from phasegen.comparison import Comparison

try:
    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    migration_rates = snakemake.params.migration_rates
    num_replicates = snakemake.params.get('num_replicates', 10000)
    n_threads = snakemake.params.get('n_threads', 100)
    parallelize = snakemake.params.get('parallelize', True)
    model = snakemake.params.model
    alpha = snakemake.params.alpha
    dist = snakemake.params.dist
    stat = snakemake.params.stat
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = dict(pop_0=3, pop_1=5)  # sample size
    pop_sizes = dict(pop_0={0: .02}, pop_1={0: .02})
    migration_rates = {('pop_0', 'pop_1'): {0: 0.5}, ('pop_1', 'pop_0'): {0: 0.5}}
    num_replicates = 10000
    n_threads = 100
    parallelize = True
    model = 'standard'
    alpha = 1.5
    dist = 'tree_height'
    stat = 'pdf'
    out = "scratch/test_comp.png"

comp = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    migration_rates=migration_rates,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize,
    model=model,
    alpha=alpha
)

ax = plt.gca()
getattr(getattr(comp.ph, dist), stat).plot(ax=ax, show=False, label='phasegen')
getattr(getattr(comp.ms, dist), stat).plot(ax=ax, show=False, label='msprime')
plt.legend()

plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()
