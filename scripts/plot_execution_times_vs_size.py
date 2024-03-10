"""
Plot the execution times against the size of the state space.
"""
__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-03-09"

import itertools
import time
from typing import Callable, List, Tuple

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
    out = "scratch/state_space_time.png"

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


def compute(
        N: np.array,
        D: np.array,
        callback: Callable[[pg.Coalescent], None],
        size: Callable[[pg.Coalescent], pg.StateSpace],
) -> List[Tuple[float, int]]:
    """
    Plot the state space size for different number of configurations.

    :param N: Number of lineages
    :param D: Number of demes
    :param callback: Function to benchmark
    :param size: Function to compute Van Loan's state space size
    :return: List of tuples (time, state space size)
    """
    sizes = []

    pbar = tqdm(total=len(N) * len(D))

    for (n, d) in itertools.product(N, D):
        pbar.set_description(f"n={n}, d={d}")

        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)})
        )

        sizes += [(benchmark(lambda: callback(coal)), size(coal))]

        pbar.update(1)

    return sizes


# warm start
_ = pg.Coalescent(
    n={'pop_0': 9, 'pop_1': 0, 'pop_2': 0},
).tree_height.mean

sizes = []

sizes += compute(
    N=np.arange(2, 13, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.tree_height.mean,
    size=lambda coal: coal.default_state_space.k * 2,
)

sizes += compute(
    N=np.arange(2, 13, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.tree_height.var,
    size=lambda coal: coal.default_state_space.k * 3,
)

sizes += compute(
    N=np.arange(2, 13, 1),
    D=np.arange(1, 4),
    callback=lambda coal: coal.total_branch_length.mean,
    size=lambda coal: coal.default_state_space.k * 2,
)

sizes += compute(
    N=np.arange(2, 8, 1),
    D=np.arange(1, 3),
    callback=lambda coal: coal.moment(1, rewards=(pg.UnfoldedSFSReward(1)),
                                      state_space=coal.block_counting_state_space),
    size=lambda coal: coal.default_state_space.k * 2,
)

sizes += compute(
    N=np.arange(3, 8, 1),
    D=np.arange(1, 3),
    callback=lambda coal: coal.moment(1, rewards=(pg.UnfoldedSFSReward(2)),
                                      state_space=coal.block_counting_state_space),
    size=lambda coal: coal.default_state_space.k * 2,
)

sizes += compute(
    N=np.arange(3, 8, 1),
    D=np.arange(1, 3),
    callback=lambda coal: coal.moment(2, rewards=(pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(1)),
                                      state_space=coal.block_counting_state_space),
    size=lambda coal: coal.default_state_space.k * 3,
)

# scatter plot
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots()

sizes = np.array(sizes)

ax.scatter(
    sizes[:, 1],
    sizes[:, 0],
    alpha=0.5,
    s=10,
)

# remove margin
ax.set_xlabel("Van Loan matrix size")
ax.set_ylabel("Execution time (s)")

fig.tight_layout()

plt.savefig(out)

if testing:
    plt.show()

pass
