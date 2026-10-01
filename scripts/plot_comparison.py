"""
Plot comparison of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

import numpy as np

try:
    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    num_replicates = snakemake.params.get('num_replicates', 10000)
    n_threads = snakemake.params.get('n_threads', 100)
    parallelize = snakemake.params.get('parallelize', True)
    models = snakemake.params.models
    type = snakemake.params.type
    dist = snakemake.params.dist
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 4  # sample size
    pop_sizes = {'pop_0': {0: 1}}
    num_replicates = 10000
    n_threads = 100
    parallelize = True
    models = ['ph', 'ms']
    type = 'total_branch_length'
    dist = 'pdf'
    out = "scratch/test_comp.png"

from matplotlib import pyplot as plt

from phasegen.comparison import Comparison

comp = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize
)

x = np.linspace(0, 10, 100)
for model in models:
    getattr(getattr(getattr(comp, model), type), dist).plot(t=x, show=False, clear=False, label=model)

plt.legend()

# save plot
plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()
