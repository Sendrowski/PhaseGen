"""
Simulate potential bug in msprime when simulating genealogies under the beta coalescent.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-02-04"

import multiprocessing as mp

import matplotlib.pyplot as plt
import msprime
import numpy as np
import scipy
import math


def beta(x, y):
    """
    Compute the beta function based on numpy's gamma function.
    """
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)


def compute_beta_timescale(pop_size, alpha, ploidy):
    """
    Compute the generation time for the beta coalescent exactly as done in msprime in verification.py.
    See https://github.com/tskit-dev/msprime/blob/804e0361c4f8b5f5051a9fbf411054ee8be3426a/verification.py#L3447
    """

    if ploidy > 1:
        N = pop_size / 2
        m = 2 + np.exp(
            alpha * np.log(2) + (1 - alpha) * np.log(3) - np.log(alpha - 1)
        )
    else:
        N = pop_size
        m = 1 + np.exp((1 - alpha) * np.log(2) - np.log(alpha - 1))
    ret = np.exp(
        alpha * np.log(m)
        + (alpha - 1) * np.log(N)
        - np.log(alpha)
        - scipy.special.betaln(2 - alpha, alpha)
    )
    return ret


def simulate_tree_height(alpha):
    """
    Simulate genealogies under the beta coalescent and compute the average tree height.
    """

    # simulate genealogies under the beta coalescent
    g = msprime.sim_ancestry(
        samples=2,
        num_replicates=100000,
        model=msprime.BetaCoalescent(alpha=alpha),
        ploidy=1
    )

    return np.mean([ts.first().total_branch_length / 2 for ts in g])


if __name__ == "__main__":
    alphas = np.linspace(1.99, 1.999, 50)

    # simulate average tree heights in parallel
    with mp.Pool() as pool:
        heights_observed = np.array(pool.map(simulate_tree_height, alphas))

    # compute theoretical tree heights
    heights_theoretical = np.array([compute_beta_timescale(1, alpha, 1) for alpha in alphas])

    # compute relative difference between observed and theoretical
    diff_rel = (heights_observed - heights_theoretical) / heights_theoretical

    # plot relative difference against alpha
    plt.figure(figsize=(7, 2.5))
    plt.plot(alphas, diff_rel)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('alpha')
    plt.ylabel('diff_rel')
    plt.title('1e6 replicates')
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()

    pass
