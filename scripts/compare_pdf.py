"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

import time

import matplotlib.pyplot as plt
import numpy as np
from fastdfe import Spectra

import phasegen as pg
from phasegen.comparison import Comparison

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    times = snakemake.params.times
    pop_sizes = snakemake.params.pop_sizes
    migration_rates = snakemake.params.migration_rates
    growth_rate = snakemake.params.growth_rate
    N0 = snakemake.params.N0
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
    times = dict(pop_0=[0], pop_1=[0])
    pop_sizes = dict(pop_0=[.02], pop_1=[.02])
    migration_rates = {('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
    """n = 10
    times = [0]
    pop_sizes = [1, 3]
    migration_rates = None"""
    growth_rate = None
    N0 = 1
    num_replicates = 100000
    n_threads = 1000
    parallelize = True
    model = 'standard'
    alpha = 1.5
    dist = 'tree_height'
    stat = 'plot_pdf'
    out = "scratch/test_comp.png"

comp = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    times=times,
    growth_rate=growth_rate,
    migration_rates=migration_rates,
    N0=N0,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize,
    model=model,
    alpha=alpha
)

#ph = getattr(getattr(comp.ph, dist), stat)(ax=plt.gca(), show=False, label='phasegen')
#ms = getattr(getattr(comp.ms, dist), stat)(ax=plt.gca(), show=True, label='msprime')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#getattr(getattr(comp.ms, dist), stat)(ax=ax2, show=False, label='msprime')
getattr(getattr(comp.ph, dist), stat)(ax=ax1, show=True, label='phasegen')

pass
