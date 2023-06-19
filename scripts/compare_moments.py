"""
Compare moments of msprime and ph.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-03-11"

import numpy as np
from fastdfe import Spectra, Spectrum

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
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
    n = 4  # sample size
    times = [0]
    pop_sizes = [1]
    alpha = np.eye(1, n, 0)[0]
    num_replicates = 100000
    n_threads = 1000
    parallelize = True
    dist = 'sfs'
    stat = 'var'
    out = "scratch/test_comp.png"

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

s1 = getattr(getattr(comp.ph, dist), stat)
#s0 = getattr(getattr(comp.ph_legacy, dist), stat)
s2 = getattr(getattr(comp.ms, dist), stat)

if dist == 'sfs':
    Spectra.from_spectra(dict(ms=Spectrum(s2), ph=Spectrum(s1))).plot()

pass
