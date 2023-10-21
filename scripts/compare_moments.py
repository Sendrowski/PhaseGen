"""
Compare moments of msprime and phasegen.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-03-11"

import time

import matplotlib.pyplot as plt
import numpy as np
from fastdfe import Spectra

import phasegen as pg
from phasegen.comparison import Comparison

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    times = snakemake.params.times
    pop_sizes = snakemake.params.pop_sizes
    migration_matrix = snakemake.params.migration_matrix
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
    """n = dict(pop_0=3, pop_1=5)  # sample size
    times = dict(pop_0=[0], pop_1=[0])
    pop_sizes = dict(pop_0=[1], pop_1=[1])
    migration_matrix = {('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}"""
    n = 10
    times = [0]
    pop_sizes = [1]
    migration_matrix = None
    growth_rate = None
    N0 = 1
    num_replicates = 100000
    n_threads = 1000
    parallelize = True
    model = 'beta'
    alpha = 1.5
    dist = 'sfs'
    stat = 'corr'
    out = "scratch/test_comp.png"

comp = Comparison(
    n=n,
    pop_sizes=pop_sizes,
    times=times,
    growth_rate=growth_rate,
    migration_matrix=migration_matrix,
    N0=N0,
    num_replicates=num_replicates,
    n_threads=n_threads,
    parallelize=parallelize,
    model=model,
    alpha=alpha
)

start_time = time.time()
ph = getattr(getattr(comp.ph, dist), stat)
runtime = time.time() - start_time

ms = getattr(getattr(comp.ms, dist), stat)

if isinstance(ph, pg.SFS2):

    _, axs = plt.subplots(ncols=2, subplot_kw={"projection": "3d"}, figsize=(8, 4))

    ph.plot(ax=axs[0], title='ph', show=False)
    ms.plot(ax=axs[1], title='ms')

    abs_max = np.nanmax(np.abs((ms.data - ph.data) / (ms.data + ph.data)))

elif isinstance(ms, pg.SFS):

    Spectra.from_spectra(dict(
        ms=ms,
        ph=ph,
    )).plot()

    abs_max = np.nanmax(np.abs((ms.data - ph.data) / (ms.data + ph.data)))

else:

    Spectra.from_spectra(dict(
        ms=pg.SFS([0, ms, 0]),
        ph=pg.SFS([0, ph, 0]),
    )).plot()

pass
