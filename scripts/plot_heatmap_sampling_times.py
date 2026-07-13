"""
Plot the vectorized-sampling runtime for the same scenarios as the exact-computation benchmark
(``plot_heatmap_execution_times.py``), for a side-by-side comparison. The sampler's cost grows with the number
of samples rather than the state space, so it stays roughly flat where the exact computation blows up.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"

import itertools
import time
from typing import Callable

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    testing = False
    out = snakemake.output[0]
except NameError:
    testing = True
    out = "reports/manuscripts/merged/figures/sampling_times.png"

import phasegen as pg

#: Number of trajectories drawn per statistic (the ``to_empirical`` default).
N_SAMPLES = 100_000


def benchmark(callback: Callable) -> float:
    """Time a call in seconds."""
    start = time.time()
    callback()
    return time.time() - start


def plot_heatmap(
        N: np.ndarray,
        D: np.ndarray,
        callback: Callable[[pg.Coalescent], None],
        title: str = "Sampling time",
        ax: plt.Axes = None,
        locus_config=pg.LocusConfig()
) -> plt.Axes:
    """
    Time ``callback`` (a sampling call) over a grid of lineage and deme counts and draw it as a heatmap.

    :param N: Number of lineages
    :param D: Number of demes
    :param callback: Function to benchmark
    :param title: Title of the plot
    :param ax: Axes to plot on
    :param locus_config: Locus configuration
    :return: Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    sizes = {}
    pbar = tqdm(total=len(N) * len(D))

    for (n, d) in itertools.product(N, D):
        pbar.set_description(f"n={n}, d={d}")

        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)}),
            loci=locus_config
        )

        sizes[(n, d)] = benchmark(lambda: callback(coal))

        pbar.update(1)

    data = np.array([[sizes[(n, d)] for d in D] for n in N])

    sns.heatmap(
        data=data,
        ax=ax,
        annot=True,
        fmt=".3f",
        xticklabels=D,
        yticklabels=N,
        cmap='viridis',
        # the same fixed colour scale as the exact-computation figure, so the two are directly comparable
        norm=mpl.colors.LogNorm(vmin=1e-3, vmax=20),
        cbar=False
    )

    ax.set_xlabel("n demes")
    ax.set_ylabel("n lineages")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)
    ax.set_box_aspect(1)


fig, ax = plt.subplots(2, 2, figsize=(9, 8))

# warm start
_ = pg.Coalescent(n={'pop_0': 3, 'pop_1': 0, 'pop_2': 0}).tree_height.to_empirical(1000).mean

plot_heatmap(
    ax=ax[0, 0],
    N=np.arange(2, 13, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.tree_height.to_empirical(N_SAMPLES).mean,
    title="Mean tree height, one locus"
)

plot_heatmap(
    ax=ax[0, 1],
    N=np.arange(2, 11, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.sfs.to_empirical(N_SAMPLES).mean,
    title="Mean SFS, one locus"
)

plot_heatmap(
    ax=ax[1, 0],
    N=np.arange(2, 7, 1),
    D=np.arange(1, 3),
    callback=lambda coal: coal.tree_height.to_empirical(N_SAMPLES).mean,
    title="Mean tree height, two loci",
    locus_config=pg.LocusConfig(2)
)

# mean two-locus SFS (the recombination-aware 2-SFS); single population
plot_heatmap(
    ax=ax[1, 1],
    N=np.arange(2, 7, 1),
    D=np.arange(1, 2),
    callback=lambda coal: coal.sfs2.to_empirical(N_SAMPLES).mean,
    title="Mean two-locus SFS",
    locus_config=pg.LocusConfig(2, recombination_rate=1.0)
)

fig.suptitle(f"Vectorized sampling, {N_SAMPLES:,} samples")
fig.tight_layout(pad=2)

plt.savefig(out)

if testing:
    plt.show()
