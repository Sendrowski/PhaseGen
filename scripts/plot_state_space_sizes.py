"""
Plot the state space size for different number of configurations.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

from typing import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

try:
    import sys

    # necessary to import local module
    sys.path.append('.')
    testing = False
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    out = "scratch/state_space_size.png"

import phasegen as pg
from phasegen.utils import parallelize

cache = {}


def plot_heatmap(
        N: np.ndarray,
        D: np.ndarray,
        state_space: Callable[[pg.Coalescent], pg.state_space.StateSpace],
        locus_config: pg.LocusConfig,
        title: str = "State space size",
        ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot the state space size for different number of configurations.

    :param N: Number of lineages
    :param D: Number of demes
    :param state_space: State space function
    :param locus_config: Locus configuration
    :param title: Title of the plot
    :param ax: Axes to plot on
    :return: Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    def get_state_space(kwargs: tuple[int, int]):
        """
        Get the state space size for a given number of lineages and demes.

        :param kwargs: Tuple of number of lineages and demes
        :return: Number of states
        """
        n, d = kwargs

        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)}),
            loci=locus_config
        )

        return state_space(coal).k

    data = parallelize(get_state_space, [(n, d) for n in N for d in D]).reshape(len(N), len(D))

    cache[title] = data

    # Plot heatmap using Seaborn
    sns.heatmap(
        data=data,
        ax=ax,
        annot=True,  # Show annotations
        fmt="d",  # Format for annotations as integers
        xticklabels=D,
        yticklabels=N,
        cmap='coolwarm',
        cbar=False,
        vmax=1000
    )

    ax.set_xlabel("n demes")
    ax.set_ylabel("n lineages")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)
    ax.set_box_aspect(1)

    return ax


fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))

plot_heatmap(
    ax=ax[0],
    N=np.arange(2, 20, 2),
    D=np.arange(1, 6),
    state_space=lambda coal: coal.lineage_counting_state_space,
    locus_config=pg.LocusConfig(1),
    title="Lineage-counting state space, one locus"
)

plot_heatmap(
    ax=ax[1],
    N=np.arange(2, 8, 1),
    D=np.arange(1, 4),
    state_space=lambda coal: coal.lineage_counting_state_space,
    locus_config=pg.LocusConfig(2),
    title="Lineage-counting state space, two loci"
)

plot_heatmap(
    ax=ax[2],
    N=np.arange(2, 16, 2),
    D=np.arange(1, 4),
    state_space=lambda coal: coal.block_counting_state_space,
    locus_config=pg.LocusConfig(1),
    title="Block-counting state space, one locus"
)

fig.tight_layout()

plt.savefig(out, dpi=400)

if testing:
    plt.show()

pass
