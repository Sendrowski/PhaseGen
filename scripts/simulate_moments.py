"""
Simulate moments of given population scenario using phase-type theory
and msprime.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-02-25"

import numpy as np

from PH.comparison import Comparison

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
    num_replicates = snakemake.params.num_replicates
    alpha = snakemake.params.get('alpha', np.eye(1, n - 1, 0)[0])
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 10  # sample size
    times = [0, 0.3, 1, 1.4]
    pop_sizes = [1.2, 10, 0.8, 10]
    #pop_sizes = [1, 0.00000001]
    #times = [0, 1]
    #pop_sizes = [1]
    #times = [0]
    alpha = np.eye(1, n - 1, 0)[0]
    num_replicates = 10000
    out = "scratch/result.json"

s = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    times=times,
    num_replicates=num_replicates,
    alpha=alpha,
)

s.to_file(out)
