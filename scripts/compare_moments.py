"""
Compare moments of msprime and ph.
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
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "resources/configs/test_plot_pdf_var_total_branch_length_n_3.yaml"
    n = 5  # sample size
    times = [0, 0.3, 1, 1.4]
    pop_sizes = [1.2, 10, 0.8, 10]
    alpha = np.eye(1, n - 1, 0)[0]
    num_replicates = 100000
    n_threads = 100
    parallelize = True
    out = "scratch/test_comp.png"

from PH import Comparison

s = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    times=times,
    alpha=alpha,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize
)

_ = s.ph.total_branch_length.mean
_ = s.msprime.total_branch_length.mean

pass