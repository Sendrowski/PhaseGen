"""
Simulate moments of given population scenario using phase-type theory.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-02-25"

import numpy as np

try:
    import sys

    # necessary to import dfe module
    sys.path.append('.')

    testing = False
    n = snakemake.params.n
    pop_sizes = snakemake.params.pop_sizes
    times = snakemake.params.times
    alpha = snakemake.params.get('alpha', np.eye(1, n - 1, 0)[0])
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 5  # sample size
    pop_sizes = [1, 0.00000001]
    times = [0, 1]
    alpha = np.eye(1, n - 1, 0)[0]
    out = "scratch/ph.json"

from PH import VariablePopSizeConstantPopSizeCoalescentDistribution, StandardCoalescent, PiecewiseConstantDemography, \
    rewards
from scripts import json_handlers

cd = VariablePopSizeConstantPopSizeCoalescentDistribution(
    model=StandardCoalescent(),
    n=n,
    alpha=alpha,
    demography=PiecewiseConstantDemography(pop_sizes=pop_sizes, times=times)
)

height = dict(
    mu=cd.mean,
    var=cd.var
)

cd = cd.set_reward(rewards.TotalBranchLength())

total_branch_length = dict(
    mu=cd.mean,
    var=cd.var
)

if testing:
    pass
    # cd.plot_cdf(t_max=100)
    # cd.plot_pdf(u_max=100)

JSON.save(dict((k, globals()[k]) for k in ['n', 'height', 'total_branch_length']), out)
