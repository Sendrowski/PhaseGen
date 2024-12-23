"""
Plot the state space size for different number of configurations.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-09"

import itertools
import time
from typing import Callable

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    out = "reports/manuscripts/merged/figures/execution_times.png"

import phasegen as pg


def benchmark(callback: Callable) -> float:
    """
    Benchmark a function.

    :param callback: function to benchmark
    :return: time in seconds
    """
    start = time.time()

    callback()

    end = time.time()

    return end - start


def plot_heatmap(
        N: np.ndarray,
        D: np.ndarray,
        callback: Callable[[pg.Coalescent], None],
        title: str = "State space time",
        ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot the state space size for different number of configurations.

    :param N: Number of lineages
    :param D: Number of demes
    :param callback: Function to benchmark
    :param title: Title of the plot
    :param ax: Axes to plot on
    :return: Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    sizes = {}

    pbar = tqdm(total=len(N) * len(D))

    for (n, d) in itertools.product(N, D):
        pbar.set_description(f"n={n}, d={d}")

        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)})
        )

        sizes[(n, d)] = benchmark(lambda: callback(coal))

        pbar.update(1)

    data = np.array([[sizes[(n, d)] for d in D] for n in N])

    # Plot heatmap using Seaborn
    sns.heatmap(
        data=data,
        ax=ax,
        annot=True,
        fmt=".3f",
        xticklabels=D,
        yticklabels=N,
        cmap='viridis',
        norm=mpl.colors.LogNorm(),
        cbar=False
    )

    ax.set_xlabel("n demes")
    ax.set_ylabel("n lineages")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)
    ax.set_box_aspect(1)


fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# warm start
_ = pg.Coalescent(
    n={'pop_0': 3, 'pop_1': 3, 'pop_2': 3},
).tree_height.mean

plot_heatmap(
    ax=ax[0],
    N=np.arange(2, 13, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.tree_height.mean,
    title="Mean tree height"
)

plot_heatmap(
    ax=ax[1],
    N=np.arange(2, 9, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.sfs.mean,
    title="Mean SFS"
)

fig.tight_layout()

plt.savefig(out)

if testing:
    plt.show()

pass
