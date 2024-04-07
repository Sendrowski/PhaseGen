"""
Calculate the 2-SFS from the data provided by Rice et al.
(https://www.biorxiv.org/content/10.1101/461517v1)
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-02-11"

import matplotlib.pyplot as plt

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    file_counts = snakemake.input.counts
    file_4fold = snakemake.input.get('fourfold', '')
    n_proj = snakemake.params.n_proj
    d = snakemake.params.d
    folded = snakemake.params.folded
    filter_4fold = snakemake.params.filter_4fold
    filter_boundaries = snakemake.params.filter_boundaries
    n_subset = snakemake.params.get('n_subset', None)
    chrom = snakemake.params.chrom
    boundaries = snakemake.params.get('boundaries', {})
    sep = snakemake.params.get('sep', ' ')
    out_data = snakemake.output.data
    out_image = snakemake.output.image
except NameError:
    # testing
    testing = True
    chrom = 'simulated'
    n_proj = 20  # sophisticated sampling and keeping track of lineages-specific mutations necessary if down-projecting
    # file_counts = f"results/drosophila/data/{chrom}_{n_proj}.csv"
    # file_counts = f"resources/rice/data/DPGP3/minor_allele_counts/Chr{chrom}.mac.txt.gz"
    file_counts = "results/simulations/data/beta.1.8/mu=3e-06/Ne=10000.0/n=20/L=1000000.0/r=3e-07.txt"
    file_4fold = "resources/rice/data/dmel-4Dsites.txt.gz"
    d = 10
    folded = False
    filter_4fold = False
    filter_boundaries = False
    n_subset = None
    boundaries = {
        '2L': (1e6, 17e6),
        '2R': (6e6, 19e6),
        '3L': (1e6, 17e6),
        '3R': (10e6, 26e6)
    }
    sep = ' '
    out_data = 'scratch/2sfs.txt'
    out_image = 'scratch/2sfs.png'

import numpy as np
import pandas as pd
from tqdm import tqdm

from phasegen import SFS2


def compute(data: pd.DataFrame, n: int = 20, d: int = 100, d_offset: int = 0) -> SFS2:
    """
    Create 2-SFS from allele counts.

    :param data: Dataframe with columns 'pos', 'count' and 'chrom'
    :param n: Number of individuals
    :param d: Distance over which to calculate the 2-SFS
    :param d_offset: Minimum distance between sites
    :return: 2-SFS and SFS
    """
    # initialize SFS
    sfs = np.zeros(n + 1, dtype=int)
    sfs2 = np.zeros((n + 1, n + 1), dtype=int)

    n_sites = len(data)

    # create tqdm progress bar
    pbar = tqdm(total=n_sites)

    # group by chromosome
    data = data.groupby('chrom')

    # iterate over chromosomes
    for chrom in list(data.groups):
        positions = data.get_group(chrom).set_index('pos')

        # iterate over positions
        for k in positions.index:

            i = positions.loc[k]['count']
            sfs[i] += 1

            # iterate over genomic distances
            for dist in range(d_offset + 1, d_offset + d + 1):

                # check if site is defined
                if k + dist in positions.index:
                    j = positions.loc[k + dist]['count']
                    sfs2[i, j] += 1

            # update progress bar
            pbar.update(1)

    pbar.close()

    # normalize counts
    sfs2 = SFS2(sfs2).symmetrize().data / sfs2.sum()
    sfs = sfs / sfs.sum()
    sfs2_neutral = np.outer(sfs, sfs)

    if folded:
        sfs2 = SFS2(sfs2).fold().data
        sfs2_neutral = SFS2(sfs2_neutral).fold().data

    # cov(X, Y) = E[XY] - E[X]E[Y]
    cov = sfs2 - sfs2_neutral

    # var(X) = E[X^2] - E[X]^2
    diag = np.diag(cov)
    var = np.outer(np.sqrt(diag), np.sqrt(diag))

    # corr(X, Y) = cov(X, Y) / sqrt(var(X))sqrt(var(Y))
    corr = cov / var

    return SFS2(corr)


sites = pd.read_csv(file_counts, sep=sep, header=None, names=['n', 'k'])

if filter_boundaries:
    sites = sites.loc[int(boundaries[chrom][0]):int(boundaries[chrom][1])]

if filter_4fold:
    # load sites
    sites_neutral = pd.read_csv(file_4fold, sep="\t", header=None, names=['chr', 'pos'])

    # filter for chromosome
    sites_neutral = sites_neutral[sites_neutral.chr == chrom]

    # convert to zero indexed positions
    sites_neutral.pos = sites_neutral.pos - 1

    # filter for 4-fold sites
    sites = sites[sites.index.isin(sites_neutral.pos)]

if n_subset:
    sites = sites.head(n_subset)

# convert to required input format
sites['pos'] = sites.index
sites['chrom'] = chrom

# filter out site with fewer than n_proj individuals
sites = sites[sites.n >= n_proj]

# subsample hyper-geometrically
# sites['count'] = np.array([hypergeom.rvs(M=n, n=k, N=n_proj) for k, n in zip(sites.k, sites.n)], dtype=int)
sites['count'] = sites.k

# make sure that the number of individuals is ``n_proj`` for all sites
assert (sites.n == n_proj).all()

# calculate spectra
sfs2 = compute(sites, n=n_proj, d=d)

sfs2.plot(show=False, max_abs=1)

plt.savefig(out_image, dpi=300, bbox_inches='tight')

if testing:
    plt.show()

sfs2.to_file(out_data)
