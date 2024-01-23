"""
Simulate moments of given population scenario using phase-type theory.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-25"

import numpy as np

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
    alpha = snakemake.params.get('alpha', np.eye(1, n, 0)[0])
except NameError:
    # testing
    testing = True
    n = 5  # sample size
    pop_sizes = dict(pop_1=[1, 5])
    times = dict(pop_1=[0, 1])
    alpha = np.eye(1, n, 0)

import phasegen as pg

cd = pg.Coalescent(
    model=pg.StandardCoalescent(),
    n=n,
    demography=pg.PiecewiseConstantDemography(pop_sizes=pop_sizes, times=times)
)

cd.sfs.var.plot()

pass
