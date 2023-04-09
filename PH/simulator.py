from typing import Generator, List, Callable

import msprime as ms
import multiprocess as mp
import numpy as np
import tskit
from tqdm import tqdm

from .distributions import PiecewiseConstantDemography, StandardCoalescent, VariablePopSizeCoalescent
from scripts import json_handlers


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True
) -> np.ndarray:
    """
    Convenience function that parallelizes the given function
    if specified or executes them sequentially otherwise.
    :param pbar:
    :type pbar:
    :param parallelize:
    :type parallelize:
    :param data:
    :type data:
    :param func:
    :type func: Callable
    :return:
    """

    if parallelize and len(data) > 1:
        # parallelize
        iterator = mp.Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    if pbar:
        iterator = tqdm(iterator, total=len(data))

    return np.array(list(iterator), dtype=object)


def calculate_sfs(tree: tskit.trees.Tree) -> np.ndarray:
    """
    Calculate the SFS of given tree by looking at mutational opportunities.
    :param tree:
    :return:
    """
    sfs = np.zeros(tree.sample_size + 1)
    for u in tree.nodes():
        if u != tree.root:
            t = tree.get_branch_length(u)
            n = tree.get_num_leaves(u)

            sfs[n] += t

    return sfs


class Simulator:
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """

    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List,
            times: np.ndarray | List,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            alpha: np.ndarray | List = None
    ):
        self.n = n
        self.pop_sizes = pop_sizes
        self.times = times
        self.num_replicates = num_replicates
        self.n_threads = n_threads
        self.parallelize = parallelize

        if alpha is None:
            self.alpha = np.eye(1, n - 1, 0)[0]
        else:
            self.alpha = alpha

        self.msprime = {}
        self.ph = {}

    def simulate(self) -> None:
        """
        Simulate moment using both phase-type theory and msprime.
        :return:
        """
        self.simulate_msprime()
        self.simulate_ph()

    def simulate_msprime(self) -> None:
        """
        Simulate moments using msprime.
        :return:
        """
        # configure demography
        d = ms.Demography()
        d.add_population(initial_size=self.pop_sizes[0])

        # add population size change is specified
        for i in range(1, len(self.pop_sizes)):
            d.add_population_parameters_change(time=self.times[i], initial_size=self.pop_sizes[i])

        def simulate(_) -> (np.ndarray, np.ndarray, np.ndarray):
            """
            Simulate statistics.
            :param _:
            :type _:
            :return:
            :rtype:
            """
            # number of replicates for one thread
            num_replicates = self.num_replicates // self.n_threads

            # simulate trees
            g: Generator = ms.sim_ancestry(
                samples=self.n,
                num_replicates=num_replicates,
                demography=d,
                model=ms.StandardCoalescent(),
                ploidy=1
            )

            # initialize variables
            heights = np.zeros(num_replicates)
            total_branch_lengths = np.zeros(num_replicates)
            sfs = np.zeros((num_replicates, self.n + 1))

            # iterate over trees and compute statistics
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):
                t: tskit.Tree = ts.first()
                total_branch_lengths[i] = t.total_branch_length
                heights[i] = t.time(t.root)
                sfs[i] = calculate_sfs(t)

            return np.concatenate([[heights.T], [total_branch_lengths.T], sfs.T])

        res = np.hstack(parallelize(simulate, [None] * self.n_threads, parallelize=self.parallelize))

        # unpack statistics
        heights, total_branch_lengths, sfs = res[0], res[1], res[2:]

        self.msprime = dict(
            # get moments of tree height
            height=dict(
                mu=np.mean(heights),
                var=np.var(heights)
            ),
            # get moments of branch length
            total_branch_length=dict(
                mu=np.mean(total_branch_lengths),
                var=np.var(total_branch_lengths)
            ),
            sfs=np.mean(sfs, axis=1)
        )

    def simulate_ph(self) -> None:
        """
        Simulate moments using phase-type theory.
        :return:
        """
        cd = VariablePopSizeCoalescent(
            model=StandardCoalescent(),
            n=self.n,
            alpha=self.alpha,
            demography=PiecewiseConstantDemography(pop_sizes=self.pop_sizes, times=self.times)
        )

        self.ph = dict(
            height=dict(
                mu=cd.tree_height.mean,
                var=cd.tree_height.var
            ),
            total_branch_length=dict(
                mu=cd.total_branch_length.mean,
                var=cd.total_branch_length.var
            )
        )

    def to_file(self, file: str) -> None:
        """
        Save object to file.
        :param file:
        :return:
        """
        json_handlers.save(self.__dict__, file)
