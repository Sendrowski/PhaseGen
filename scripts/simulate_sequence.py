"""
Simulate sequence data from which a 2-SFS is to be obtained.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-02-12"

from typing import Literal

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    mu = snakemake.params.mu
    Ne = snakemake.params.Ne
    length = snakemake.params.length
    folded = snakemake.params.folded
    model = snakemake.params.model
    alpha = snakemake.params.alpha
    n = snakemake.params.n
    recombination_rate = snakemake.params.recombination_rate
    out_data = snakemake.output.data
    out_info = snakemake.output.info
except NameError:
    # testing
    testing = True
    mu = 1e-7
    Ne = 1e4
    length = 1e7
    folded = True
    model: Literal['standard', 'beta'] = 'standard'
    alpha = 1.8
    n = 40
    recombination_rate = 1e-8
    out_data = 'scratch/seqs.csv'
    out_info = 'scratch/info.yaml'

from tqdm import tqdm
import phasegen as pg
import msprime as ms
import tskit
import numpy as np
import pandas as pd
import yaml


def get_mean_tree_height(ts: tskit.TreeSequence) -> float:
    """
    Get the mean tree height from a tree sequence.

    :param ts: tree sequence
    :return: mean tree height
    """
    heights = np.zeros(ts.num_trees)

    for tree in ts.trees():
        heights[tree.index] = tree.time(tree.root)

    return np.mean(heights)


def get_mean_total_branch_length(ts: tskit.TreeSequence) -> float:
    """
    Get the mean total branch length from a tree sequence.

    :param ts: tree sequence
    :return: mean total branch length
    """
    return np.mean([tree.total_branch_length for tree in ts.trees()])


coal = pg.Coalescent(
    model=pg.StandardCoalescent() if model == 'standard' else pg.BetaCoalescent(alpha=alpha, scale_time=False),
    n=n
)

pop_size = Ne if model == 'standard' else Ne / pg.BetaCoalescent(alpha=alpha)._get_timescale(1) ** 2

# simulate trees
ts = ms.sim_ancestry(
    samples=n,
    recombination_rate=recombination_rate,
    sequence_length=length,
    population_size=pop_size,
    ploidy=1,
    model=ms.StandardCoalescent() if model == 'standard' else ms.BetaCoalescent(alpha=alpha)
)

# normalized mean tree height
tree_height = get_mean_tree_height(ts)
total_branch_length = get_mean_total_branch_length(ts)

# simulate mutations
ts = ms.sim_mutations(ts, rate=mu, discrete_genome=True)

# count number of mutations at each site
freq = np.zeros(int(length), dtype=int)
n_repeat = 0

for variant in tqdm(ts.variants()):
    # increment by frequency at site
    freq[int(variant.site.position)] = np.sum(variant.genotypes > 0)

    if len(variant.site.mutations) > 1:
        n_repeat += 1

if folded:
    freq = np.minimum(freq, n - freq).astype(int)

n_sites_mutation = np.sum(freq > 0)

info_dict = dict(
    n_repeat=n_repeat,
    n_sites_mutation=float(n_sites_mutation),
    n_trees=ts.num_trees,
    tree_height=float(tree_height)
)

k = np.full(int(length), n, dtype=int)

# save to dataframe
sites = pd.DataFrame({'k': k, 'freq': freq})

sites.to_csv(out_data, index=False, header=None, sep=' ')

# save info to yaml
with open(out_info, 'w') as f:
    yaml.dump(info_dict, f)
