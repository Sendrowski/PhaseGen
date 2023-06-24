import functools
import logging
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from math import factorial
from typing import Generator, List, Callable, Tuple

import msprime as ms
import numpy as np
import tskit
from matplotlib import pyplot as plt
from multiprocess import Pool
from numpy.linalg import matrix_power
from scipy.linalg import inv, expm, fractional_matrix_power
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from .coalescent_models import StandardCoalescent, CoalescentModel
from .demography import PiecewiseConstantDemography
from .visualization import Visualization

logger = logging.getLogger('phasegen')


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        batch_size: int = 1,
        desc: str = None,
) -> np.ndarray:
    """
    Convenience function that parallelizes the given function
    if specified or executes them sequentially otherwise.

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize
    :param pbar: Whether to show a progress bar
    :param batch_size: Number of units to show in the pbar per function
    :param desc: Description for tqdm progress bar
    :return:
    """

    if parallelize and len(data) > 1:
        # parallelize
        iterator = Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    if pbar:
        iterator = tqdm(iterator, total=len(data), unit_scale=batch_size, desc=desc)

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


def van_loan(B: np.ndarray, S: np.ndarray, tau: float, k: int = 1) -> np.ndarray:
    """
    Use Van Loan's method to evaluate the integral ∫u S(u)B(u)S(u)du for k = 1,
    and accordingly for higher moments.

    :param B: Matrix B
    :param S: Matrix S
    :param tau: Time to integrate over
    :param k: The kth moment to evaluate
    :return: Evaluated integral
    """
    n = B.shape[0]

    # matrix of zeros
    O = np.zeros_like(B)

    # create compound matrix
    A = np.block([[S if i == j else B if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)])

    # compute matrix exponential of A to determine integral
    V = expm(tau * A)

    # return upper right block
    return V[:n, -n:]


class ProbabilityDistribution(ABC):
    @abstractmethod
    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.

        :param alpha:
        :return:
        """
        pass

    @abstractmethod
    @cached_property
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.

        :param alpha:
        :return:
        """
        pass

    def touch(self):
        """
        Touch all cached properties.
        """
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, functools.cached_property):
                getattr(self, attr)

    @abstractmethod
    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t:
        :return:
        """
        pass

    @abstractmethod
    def pdf(self, u) -> float | np.ndarray:
        """
        Density function.

        :param u:
        :return:
        """
        pass

    def plot_cdf(
            self,
            x: np.ndarray = np.linspace(0, 20, 100),
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None
    ) -> plt.axis:
        """
        Plot cumulative distribution function.

        :return:
        """
        return Visualization.plot(
            x=x,
            y=self.cdf(x),
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear
        )

    def plot_pdf(
            self,
            x: np.ndarray = np.linspace(0, 20, 100),
            show=True,
            file: str = None,
            clear: bool = True,
            label: str = None
    ) -> plt.axis:
        """
        Plot density function.

        :return:
        """
        return Visualization.plot(
            x=x,
            y=self.pdf(x),
            xlabel='u',
            ylabel='f(u)',
            label=label,
            file=file,
            show=show,
            clear=clear
        )


class PhaseTypeDistribution(ProbabilityDistribution, ABC):
    pass


class ConstantPopSizeDistribution(PhaseTypeDistribution):
    def __init__(self, cd: 'ConstantPopSizeCoalescent', r: np.ndarray | List):
        self.cd = cd
        self.r = np.array(r)

    @cached_property
    def R(self) -> np.ndarray:
        """
        The rewards matrix.

        :return:
        """
        return np.diag(self.r)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Intensity matrix with rewards.

        :return: Intensity matrix
        """
        return self.R @ self.cd.S

    @cached_property
    def T(self) -> np.ndarray:
        """
        Probability transition matrix.

        :return: Transition matrix
        """
        return expm(self.S)

    @cached_property
    def s(self) -> np.ndarray:
        """
        Exit rate vector.

        :return: Exit rate vector
        """
        return -self.S[:-1, :-1] @ self.cd.e[:-1]

    def nth_moment(self, k: int, alpha: np.ndarray = None, Ne: float = None, r: np.ndarray = None) -> float:
        """
        Get the nth moment.

        :param r: Full reward vector
        :param Ne: The effective population size
        :param k: The order of the moment
        :param alpha: Full initial state vector
        :return: The nth moment
        """
        if alpha is None:
            alpha = self.cd.alpha

        if Ne is None:
            Ne = self.cd.Ne

        if r is None:
            r = self.r

        R = self.cd.invert_reward(np.diag(r[:-1]))

        M = matrix_power(self.cd.U @ R, k)

        # self.Ne ** k is the rescaling due to population size
        return Ne ** k * factorial(k) * alpha[:-1] @ M @ self.cd.e[:-1]

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.

        :param alpha:
        :return:
        """
        return self.nth_moment(k=1, alpha=alpha)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.

        :param alpha:
        :return:
        """
        return self.nth_moment(k=2, alpha=alpha) - self.nth_moment(k=1, alpha=alpha) ** 2

    def cdf(self, t) -> float | np.ndarray:
        """
        Vectorized cumulative distribution function.

        :param t:
        :return:
        """

        def cdf(t: float) -> float:
            """
            Cumulative distribution function.

            :param t:
            :return:
            """
            return 1 - self.cd.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], t / self.cd.Ne) @ self.cd.e[:-1]

        return np.vectorize(cdf)(t)

    def pdf(self, u) -> float | np.ndarray:
        """
        Vectorized density function.

        :param u:
        :return:
        """

        def pdf(u: float) -> float:
            """
            Density function.

            :param u:
            :return:
            """
            return self.cd.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], u) @ self.s / self.cd.Ne

        return np.vectorize(pdf)(u)


class PiecewiseConstantPopSizeDistribution(ConstantPopSizeDistribution):
    """
    Variable population size distribution using IPH.
    """
    #: Threshold for absorption probability under which the absorption probability is considered zero
    absorption_threshold = 1e-10

    def __init__(self, cd: 'PiecewiseConstantPopSizeCoalescent', r: np.ndarray | List):
        super().__init__(cd, r)

        self.cd = cd
        self.r = np.array(r)

    def _nth_moment(self, k: int, alpha: Tuple[float] = None, r: Tuple[float] = None) -> (np.ndarray, np.ndarray):
        """
        Get the nth (non-central) moment per epoch.

        :param k: The kth moment
        :param alpha: Full initial value vector
        :param r: Full reward vector
        :return: Moments and absorption probability per epoch
        """
        if alpha is None:
            alpha = self.cd.alpha
        else:
            alpha = np.array(alpha)

        if r is None:
            R = self.R
        else:
            R = np.diag(r)

        # moments conditional on no absorption in previous epoch
        moments = np.zeros(self.cd.n_epochs)

        # moments conditional on no absorption in current epoch
        moments_no_absorp = np.zeros(self.cd.n_epochs)

        # state probabilities conditional on no absorption in previous epoch
        alphas = np.zeros((self.cd.n_epochs + 1, self.cd.n))

        # state probabilities conditional on no absorption in current epoch
        alphas_no_absorp = np.zeros((self.cd.n_epochs + 1, self.cd.n))

        # initial state probabilities
        alphas[0] = alpha
        alphas_no_absorp[0] = np.concatenate([(alpha[:-1] / alpha[:-1].sum()), [0]])

        # time spent in epochs scaled by Ne
        taus = np.zeros(self.cd.n_epochs)

        # iterate through epochs and compute initial values
        for i in range(self.cd.n_epochs):

            # Ne of current epoch
            Ne = self.cd.pop_sizes[i]

            # all but last epoch
            if i < self.cd.n_epochs - 1:
                # time spent in current epoch scaled by Ne
                taus[i] = (self.cd.times[i + 1] - self.cd.times[i]) / Ne

                # don't recompute transition probabilities if tau is the same as in the previous epoch
                if i == 0 or taus[i - 1] != taus[i]:
                    # calculate transition probabilities for current epoch
                    P = fractional_matrix_power(self.cd.T, taus[i])
            # last epoch
            else:
                # determine tau so that we reach the absorbing state almost surely
                taus[i] = 10
                P = matrix_power(self.cd.T, 10)

                # increase tau until we reach the absorbing state almost surely
                while (np.abs(1 - P[:, -1]) > 1e-14).any():
                    taus[i] *= 10
                    P = matrix_power(P, 10)

            # update alpha for the time spent in the current epoch
            # noinspection all
            alphas[i + 1] = alphas_no_absorp[i] @ P
            alphas_no_absorp[i + 1] = np.concatenate([(alphas[i + 1][:-1] / alphas[i + 1][:-1].sum()), [0]])

            # check if absorption probability is less than threshold
            if np.prod(1 - alphas[:i + 1, -1]) < self.absorption_threshold:
                logger.info(f"Probability of no absorption is less than {self.absorption_threshold} "
                            f"in epoch {i + 1}. Terminating loop.")
                break

            # don't recompute sojourn times if tau is the same as in the previous epoch
            if i == 0 or taus[i - 1] != taus[i]:
                # obtain sojourn matrix for the current epoch using Van Loan's method
                M = van_loan(S=self.cd.S, B=self.cd.invert_reward(R), tau=taus[i], k=k)

                # normalize by transition probabilities
                M[P != 0] /= P[P != 0]

            # calculate moments in this epoch given no absorption in the previous epoch
            # noinspection all
            moments[i] = Ne ** k * factorial(k) * (alphas_no_absorp[i] @ M)[-1]

            # all but last epoch
            if i < self.cd.n_epochs - 1:
                # make sojourn matrix symmetric
                M += M.T - np.diag(np.diag(M))

                # calculate moments given no absorption in this epoch
                moments_no_absorp[i + 1] = Ne ** k * factorial(k) * alphas_no_absorp[i] @ M @ alphas_no_absorp[i + 1]

        # probability of no absorption until the nth epoch
        no_absorption = np.cumprod(1 - alphas[:-1, -1])

        # probability of absorption in the nth epoch conditional on no absorption in previous epochs
        absorption_probs = alphas[1:, -1]

        # probability of absorption in the nth epoch
        total_absorption_probs = no_absorption * absorption_probs

        return moments, moments_no_absorp, total_absorption_probs

    def nth_moment(self, k: int, alpha: np.ndarray = None, r: np.ndarray = None, **kwargs) -> float:
        """
        Get the nth (non-central) moment.

        :param k: The kth moment
        :param alpha: Full initial value vector
        :param r: Full reward vector
        :param kwargs: Additional arguments
        :return: Moment
        """
        if alpha is not None:
            alpha = tuple(alpha)

        if r is not None:
            r = tuple(r)

        # get moments per epoch
        moments, moments_no_absorp, absorption_probs = self._nth_moment(k=k, alpha=alpha, r=r)

        if k == 1:
            moments_total = np.cumsum(moments_no_absorp) + moments

            return (moments_total * absorption_probs).sum()

        if k == 2:
            # get means
            means, _, _ = self._nth_moment(k=1, alpha=alpha, r=r)

            m2_total = moments + np.cumsum(moments_no_absorp) + 2 * np.cumsum(np.sqrt(moments_no_absorp)) * means

            return (m2_total * absorption_probs).sum()

    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Time
        :return: Cumulative probability
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not np.all(self.r[:-1] == 1):
            raise NotImplementedError("CDF not implemented for non-default rewards.")

        def cdf(t: float) -> float:
            # get the cumulative coalescent rate up to time t
            cum = self.cd.demography.get_cum_rate(t)

            return 1 - self.cd.alpha[:-1] @ fractional_matrix_power(self.cd.T[:-1, :-1], cum) @ self.cd.e[:-1]

        return np.vectorize(cdf)(t)

    def pdf(self, u: float | np.ndarray) -> float | np.ndarray:
        """
        Density function.

        :param u: Time
        :return: Density
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not np.all(self.r[:-1] == 1):
            raise NotImplementedError("PDF not implemented for non-default reward.")

        def pdf(u: float) -> float:
            # get the cumulative coalescent rate up to time u
            cum = self.cd.demography.get_cum_rate(u)

            # get current coalescent rate
            rate = self.cd.demography.get_rate(u)

            return self.cd.alpha[:-1] @ fractional_matrix_power(self.cd.T[:-1, :-1], cum) @ self.cd.s * rate

        return np.vectorize(pdf)(u)


class SFSDistribution(PiecewiseConstantPopSizeDistribution):
    """
    Variable population size distribution using IPH.
    """

    def nth_moment(self, k: int, alpha: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Get the nth moment.

        :param k: The order of the moment
        :param alpha: Full initial state vector
        :param kwargs: Additional arguments
        :return: The nth moment
        """
        # if we are calculating the first moment
        # we can easily merge the states of the different tree topologies
        # by taking the average of those states weighted by the probability
        # of the tree topology. This is much faster than the general case
        # as we work with a much smaller state space
        if k == 1:
            probs = self.cd.model.get_sample_config_probs(self.cd.n)

            R = np.zeros((self.cd.n, self.cd.n))

            for state, prob in probs.items():
                R[:, self.cd.n - sum(state)] += prob * np.array(state)
        else:
            # TODO create large state space
            R = None

        sfs = np.zeros(self.cd.n + 1)
        for i, r in enumerate(R[:-1]):
            sfs[i + 1] = super().nth_moment(k=k, r=self.cd.invert_reward(r), alpha=alpha)

        return sfs


class EmpiricalDistribution(ProbabilityDistribution):
    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples:
        """
        self.samples = np.array(samples, dtype=float)

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the mean absorption time.

        :param alpha:
        :return:
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the variance in the absorption time.

        :param alpha:
        :return:
        """
        return np.var(self.samples, axis=0)

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t:
        :return:
        """
        x = np.sort(self.samples)
        y = np.arange(1, len(self.samples) + 1) / len(self.samples)

        return np.interp(t, x, y)

    def pdf(self, u: float | np.ndarray, n_bins: int = 10000, sigma: float = None) -> float | np.ndarray:
        """
        Density function.

        :param sigma:
        :param n_bins:
        :param u:
        :return:
        """
        hist, bin_edges = np.histogram(self.samples, range=(0, max(self.samples)), bins=n_bins, density=True)

        # determine bins for u
        bins = np.minimum(np.sum(bin_edges <= u[:, None], axis=1) - 1, np.full_like(u, n_bins - 1, dtype=int))

        # use proper bins for y values
        y = hist[bins]

        # smooth using gaussian filter
        if sigma is not None:
            y = gaussian_filter1d(y, sigma=sigma)

        return y


class CoalescentDistribution:
    @property
    @abstractmethod
    def tree_height(self) -> ProbabilityDistribution:
        """
        Tree height distribution.

        :return:
        :rtype:
        """
        pass

    @property
    @abstractmethod
    def total_branch_length(self) -> ProbabilityDistribution:
        """
        Total branch length distribution.

        :return:
        :rtype:
        """
        pass


class ConstantPopSizeCoalescent(CoalescentDistribution):
    """
    Coalescent distribution for a constant population size.
    """

    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray | List = None,
            Ne: float | int = 1,
            S_sub: np.ndarray = None,
    ):
        """
        Create object.

        :param n: Number of lineages.
        :param model: Coalescent model.
        :param alpha: Initial state vector.
        :param Ne: Effective population size.
        :param S_sub: Sub-intensity matrix (advanced use, ``model`` will be ignored).
        """
        self.alpha = None

        # coalescent model
        self.model = model

        # sample size
        self.n = n

        # initial conditions
        if alpha is None:
            self.alpha = np.eye(1, self.n, 0)[0]
        elif len(alpha) != self.n:
            raise Exception(f"alpha should be of length n={self.n} but has instead length {len(alpha)}.")
        else:
            self.alpha = np.array(alpha)

        # effective population size
        self.Ne = Ne

        if S_sub is None:
            # obtain sub-intensity matrix
            S_sub = self.model.get_rate_matrix(self.n)

        # obtain exit rate vector
        self.s = -S_sub @ self.e[:-1]

        # obtain full intensity matrix
        self.S = np.block([
            [S_sub, self.s[:, None]],
            [np.zeros(self.n)]
        ])

    @staticmethod
    def pad(x: np.ndarray) -> np.ndarray:
        """
        Pad a matrix with a row and column of zeros or a vector with a zero.

        :param x: The matrix or vector to pad.
        :return: The padded matrix or vector.
        """
        if x.ndim == 1:
            return np.pad(x, (0, 1), mode='constant', constant_values=0)

        return np.pad(x, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    @staticmethod
    def invert_reward(r: np.ndarray) -> np.ndarray:
        """
        Invert the reward matrix or reward vector.

        :param r: The reward matrix or reward vector.
        :return: The inverted reward matrix or reward vector.
        """
        if r.ndim == 2:
            r = r[np.diag_indices(r.shape[0])]

            r[r != 0] = 1 / r[r != 0]

            return np.diag(r)

        r[r != 0] = 1 / r[r != 0]

        return r

    @cached_property
    def e(self) -> np.ndarray:
        """
        Get a vector with ones of size 1.

        :return:
        :rtype:
        """
        return np.ones(self.n)

    @cached_property
    def T(self) -> np.ndarray:
        """
        The transition matrix.

        :return:
        """
        return expm(self.S)

    @cached_property
    def t(self) -> np.ndarray:
        """
        The exit probability vector.

        :return:
        :rtype:
        """
        return 1 - self.T * self.e

    @cached_property
    def U(self) -> np.ndarray:
        """
        The Green matrix (negative inverse of sub-intensity matrix).

        :return:
        """
        return -inv(self.S[:-1, :-1])

    @cached_property
    def T_inv(self) -> np.ndarray:
        """
        Inverse of transition matrix.

        :return:
        """
        return inv(self.T)

    @cached_property
    def tree_height(self) -> ConstantPopSizeDistribution:
        """
        Tree height distribution.

        :return:
        :rtype:
        """
        return ConstantPopSizeDistribution(
            cd=self,
            r=self.pad(np.ones(self.n - 1))
        )

    @cached_property
    def total_branch_length(self) -> ConstantPopSizeDistribution:
        """
        Total branch length distribution.

        :return:
        :rtype:
        """
        return ConstantPopSizeDistribution(
            cd=self,
            r=self.pad(1 / np.arange(2, self.n + 1)[::-1])
        )


class PiecewiseConstantPopSizeCoalescent(ConstantPopSizeCoalescent):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            demography: PiecewiseConstantDemography = None,
            S_sub: np.ndarray = None,
    ):
        """
        Create object.

        :param n: Number of lineages.
        :param model: Coalescent model.
        :param alpha: Initial state vector.
        :param S_sub: Sub-intensity matrix (advanced use, ``model`` will be ignored).
        """
        super().__init__(
            model=model,
            n=n,
            alpha=alpha,
            Ne=1,
            S_sub=S_sub
        )

        self.demography = demography

        if isinstance(demography, PiecewiseConstantDemography):
            self.times = demography.times
            self.pop_sizes = demography.pop_sizes
            self.n_epochs = len(self.times)
        else:
            self.times = np.array([0])
            self.pop_sizes = [self.Ne]
            self.n_epochs = 1

    @cached_property
    def tree_height(self) -> PiecewiseConstantPopSizeDistribution:
        """
        Tree height distribution.

        :return:
        :rtype:
        """
        return PiecewiseConstantPopSizeDistribution(
            cd=self,
            r=self.pad(np.ones(self.n - 1))
        )

    @cached_property
    def total_branch_length(self) -> PiecewiseConstantPopSizeDistribution:
        """
        Total branch length distribution.

        :return:
        :rtype:
        """
        return PiecewiseConstantPopSizeDistribution(
            cd=self,
            r=self.pad(1 / np.arange(2, self.n + 1)[::-1])
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site-frequency spectrum.

        :return:
        :rtype:
        """
        return SFSDistribution(self, r=self.pad(np.ones(self.n - 1)))


class MsprimeCoalescent(CoalescentDistribution):
    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List = None,
            times: np.ndarray | List = None,
            growth_rate: float = None,
            N0: float = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True
    ):
        """
        Simulate data using msprime.

        :param n: Number of Lineages.
        :param pop_sizes: Population sizes
        :param times: Epoch times
        :param growth_rate: Exponential growth rate so that at time ``t`` in the past we have
            ``N0 * exp(- growth_rate * t)``.
        :param N0: Initial population size (only used if growth_rate is specified).
        :param num_replicates: Number of replicates
        :param n_threads: Number of threads
        :param parallelize: Whether to parallelize
        """
        self.sfs_counts = None
        self.total_branch_lengths = None
        self.heights = None

        self.n = n
        self.pop_sizes = pop_sizes
        self.times = times
        self.growth_rate = growth_rate
        self.N0 = N0
        self.num_replicates = num_replicates
        self.n_threads = n_threads
        self.parallelize = parallelize

    @lru_cache
    def simulate(self):
        """
        Simulate data using msprime.
        """
        # configure demography
        d = ms.Demography()

        # add population
        d.add_population(initial_size=self.pop_sizes[0])

        # exponential growth
        if self.growth_rate is not None:
            d.add_population_parameters_change(time=0, initial_size=self.N0, growth_rate=self.growth_rate)

        # piecewise constant
        else:

            # add population size change is specified
            for i in range(1, len(self.pop_sizes)):
                d.add_population_parameters_change(time=self.times[i], initial_size=self.pop_sizes[i])

        # number of replicates for one thread
        num_replicates = self.num_replicates // self.n_threads

        def simulate_batch(_) -> (np.ndarray, np.ndarray, np.ndarray):
            """
            Simulate statistics.

            :param _:
            :type _:
            :return:
            :rtype:
            """
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

        # parallelize and add up results
        res = np.hstack(parallelize(
            func=simulate_batch,
            data=[None] * self.n_threads,
            parallelize=self.parallelize,
            batch_size=num_replicates,
            desc="Simulating trees"
        ))

        # store results
        self.heights, self.total_branch_lengths, self.sfs_counts = res[0], res[1], res[2:]

    def touch(self):
        """
        Touch cached properties.
        """
        self.total_branch_length.touch()
        self.tree_height.touch()
        self.sfs.touch()

    def drop(self):
        """
        Drop simulated data.
        """
        self.heights = None
        self.total_branch_lengths = None
        self.sfs_counts = None

    @cached_property
    def tree_height(self) -> EmpiricalDistribution:
        """
        Tree height distribution.

        :return:
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.heights)

    @cached_property
    def total_branch_length(self) -> EmpiricalDistribution:
        """
        Total branch length distribution.

        :return:
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.total_branch_lengths)

    @cached_property
    def sfs(self) -> EmpiricalDistribution:
        """
        Site-frequency spectrum.

        :return:
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.sfs_counts.T)
