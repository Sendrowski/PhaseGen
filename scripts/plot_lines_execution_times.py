"""
Plot the state space size for different number of configurations.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-06-26"

import time
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt, ticker
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
    out = "reports/manuscripts/merged/figures/execution_times_lineplot.png"

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


def plot_line(
        N: np.ndarray,
        stat: Callable[[int], float],
        ax: plt.Axes,
        label: str
):
    """
    Plot the state space size for different number of configurations.

    :param N: Number of lineages
    :param stat: Function to benchmark
    :param ax: Axes to plot on
    :param label: Label for the line
    """
    ax.plot(N, [benchmark(lambda n=n: stat(n)) for n in tqdm(N)], label=label)


N = np.arange(2, 10)

fig, ax = plt.subplots(figsize=(9, 3))

# warm start
_ = pg.Coalescent(
    n={'pop_0': 3, 'pop_1': 3, 'pop_2': 3},
).tree_height.mean

for n_epochs in [1]:
    for stat_label, stat in {
        'tree height': lambda coal: coal.tree_height.mean,
        'SFS': lambda coal: coal.sfs.mean
    }.items():
        # 1 deme
        plot_line(
            ax=ax,
            N=N,
            stat=lambda n: stat(pg.Coalescent(
                n={'pop_0': n},
                demography=pg.Demography(
                    pop_sizes={'pop_0': {i * 0.1: 1 for i in range(n_epochs)}}
                )
            )),
            label=f"1 deme, 1 locus, {stat_label}"
        )

        # 2 demes
        plot_line(
            ax=ax,
            N=N,
            stat=lambda n: stat(pg.Coalescent(
                n={'pop_0': n - 2, 'pop_1': 2},
                demography=pg.Demography(
                    pop_sizes={'pop_0': {i * 0.1: 1 for i in range(n_epochs)}},
                    events=[pg.SymmetricMigrationRateChanges(pops=['pop_0', 'pop_1'], rate=1)]
                )
            )),
            label=f"2 demes, 1 locus, {stat_label}"
        )

        # 3 demes
        plot_line(
            ax=ax,
            N=N,
            stat=lambda n: stat(pg.Coalescent(
                n={'pop_0': n - 2, 'pop_1': 1, 'pop_2': 1},
                demography=pg.Demography(
                    pop_sizes={'pop_0': {i * 0.1: 1 for i in range(n_epochs)}},
                    events=[pg.SymmetricMigrationRateChanges(pops=['pop_0', 'pop_1', 'pop_2'], rate=1)]
                )
            )),
            label=f"3 demes, 1 locus, {stat_label}"
        )

# 2 loci, 1 deme
plot_line(
    ax=ax,
    N=N,
    stat=lambda n: pg.Coalescent(
        n={'pop_0': n},
        demography=pg.Demography(
            pop_sizes={'pop_0': {i * 0.1: 1 for i in range(n_epochs)}}
        ),
        loci=pg.LocusConfig(
            n=2,
            recombination_rate=0.1
        )
    ).tree_height.mean,
    label=f"1 deme, 2 loci, tree height"
)

ax.set_xlabel("number of lineages")
ax.set_ylabel("execution time in seconds")

# log scale on y axis
ax.set_yscale('log')
#ax.yaxis.set_major_formatter(ticker.LogFormatter())
#plt.ticklabel_format(axis='y', style='plain')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.tight_layout()

plt.savefig(out, dpi=400)

if testing:
    plt.show()

pass
