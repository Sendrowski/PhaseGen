"""
Plot comparison of msprime and phasegen.
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
    file = snakemake.input[0]
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
    alpha = snakemake.params.get('alpha', np.eye(1, n - 1, 0)[0])
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
    times = [0]
    pop_sizes = [1]
    alpha = np.eye(1, n, 0)[0]
    num_replicates = 100000
    n_threads = 100
    parallelize = True
    models = ['ph_const', 'ms']
    type = 'total_branch_length'
    dist = 'plot_pdf'
    out = "scratch/test_comp.png"

from matplotlib import pyplot as plt

from phasegen import Comparison

comp = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    times=times,
    alpha=alpha,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize
)

x = np.linspace(0, 10, 100)
for model in models:
    getattr(getattr(getattr(comp, model), type), dist)(x=x, show=False, clear=False, label=model)

# save plot
plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()

pass
