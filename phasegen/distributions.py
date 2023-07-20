import logging
from abc import ABC, abstractmethod
from functools import cached_property, cache
from math import factorial
from typing import Generator, List, Callable, Tuple, Iterable

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
from .demography import PiecewiseConstantDemography, Demography
from .spectrum import SFS, SFS2
from .visualization import Visualization

logger = logging.getLogger('phasegen')


def _parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        batch_size: int = None,
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
    :return: Array of results
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


def _calculate_sfs(tree: tskit.trees.Tree) -> np.ndarray:
    """
    Calculate the SFS of given tree by looking at mutational opportunities.

    :param tree: Tree to calculate SFS for.
    :return: SFS.
    """
    sfs = np.zeros(tree.sample_size + 1)

    for u in tree.nodes():
        t = tree.get_branch_length(u)
        n = tree.get_num_leaves(u)

        sfs[n] += t

    return sfs


def _get_van_loan_matrix(R: Tuple[np.ndarray], S: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Get the block matrix for the given reward matrices and transition matrix.

    :param R: Tuple of length k of reward matrices
    :param S: Matrix S
    :param k: The kth moment to evaluate
    :return: Block matrix
    """
    # matrix of zeros
    O = np.zeros_like(S)

    # create compound matrix
    return np.block([[S if i == j else R[i] if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)])


def _pad(x: np.ndarray) -> np.ndarray:
    """
    Pad a matrix with a row and column of zeros or a vector with a zero.

    :param x: The matrix or vector to pad.
    :return: The padded matrix or vector.
    """
    if x.ndim == 1:
        return np.pad(x, (0, 1), mode='constant', constant_values=0)

    return np.pad(x, ((0, 1), (0, 1)), mode='constant', constant_values=0)


def _invert_reward(r: np.ndarray) -> np.ndarray:
    """
    Invert the reward matrix or reward vector.

    :param r: The reward matrix or reward vector.
    :return: The inverted reward matrix or reward vector.
    """
    r = r.copy()

    if r.ndim == 2:
        r = r[np.diag_indices(r.shape[0])]

        return np.diag(_invert_reward(r))

    r_inv = r.copy().astype(float)
    r_inv[r != 0] = 1 / r_inv[r != 0]

    return r_inv


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    @abstractmethod
    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.

        :param alpha: The initial state distribution.
        :return: The mean absorption time.
        """
        pass

    @abstractmethod
    @cached_property
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.

        :param alpha: The initial state distribution.
        :return: The variance in the absorption time.
        """
        pass

    @abstractmethod
    @cached_property
    def m2(self, alpha: np.ndarray = None) -> float:
        """
        Get the second (non-central) moment of the absorption time.

        :param alpha: The initial state distribution.
        :return: The variance in the absorption time.
        """
        pass

    def touch(self):
        """
        Touch all cached properties.
        """
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, cached_property):
                getattr(self, attr)

    @abstractmethod
    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Time.
        :return: CDF.
        """
        pass

    @abstractmethod
    def pdf(self, u) -> float | np.ndarray:
        """
        Density function.

        :param u: Time.
        :return: Density.
        """
        pass

    def plot_cdf(
            self,
            ax: plt.Axes = None,
            x: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None
    ) -> plt.axes:
        """
        Plot cumulative distribution function.

        :param ax: Axes to plot on.
        :param x: Values to evaluate the CDF at.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :return: Axes.
        """
        if x is None:
            x = np.linspace(0, 20, 100)

        return Visualization.plot(
            ax=ax,
            x=x,
            y=self.cdf(x),
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title='CDF'
        )

    def plot_pdf(
            self,
            ax: plt.Axes = None,
            x: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None
    ) -> plt.axes:
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param x: Values to evaluate the density function at.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :return: Axes.
        """
        if x is None:
            x = np.linspace(0, 20, 100)

        return Visualization.plot(
            ax=ax,
            x=x,
            y=self.pdf(x),
            xlabel='u',
            ylabel='f(u)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title='PDF'
        )


class PhaseTypeDistribution(ProbabilityDistribution, ABC):
    """
    Abstract base class for phase-type distributions.
    """

    @abstractmethod
    def nth_moment(self, k: int, alpha: np.ndarray = None, Ne: float = None, r: np.ndarray = None) -> float | SFS:
        """
        Get the nth moment.

        :param r: Full reward vector
        :param Ne: The effective population size
        :param k: The order of the moment
        :param alpha: Full initial state vector
        :return: The nth moment
        """
        pass

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float | SFS:
        """
        Get the mean absorption time.

        :param alpha: The initial state distribution.
        :return: The mean absorption time.
        """
        return self.nth_moment(k=1, alpha=alpha)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float | SFS:
        """
        Get the variance in the absorption time.

        :param alpha: The initial state distribution.
        :return: The variance in the absorption time.
        """
        return self.nth_moment(k=2, alpha=alpha) - self.nth_moment(k=1, alpha=alpha) ** 2

    @cached_property
    def m2(self, alpha: np.ndarray = None) -> float | SFS:
        """
        Get the (non-central) second moment.

        :param alpha: The initial state distribution.
        :return: The second moment.
        """
        return self.nth_moment(k=2, alpha=alpha)


class ConstantPopSizeDistribution(PhaseTypeDistribution):
    """
    Phase-type distribution for a constant population size coalescent.
    """

    def __init__(self, cd: 'ConstantPopSizeCoalescent', r: np.ndarray | List):
        """
        Initialize the distribution.

        :param cd: Constant population size coalescent.
        :param r: Reward vector.
        """
        self.cd = cd
        self.r = np.array(r)

    @cached_property
    def R(self) -> np.ndarray:
        """
        The reward matrix.
        """
        return np.diag(self.r)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Intensity matrix with rewards.
        """
        return self.R @ self.cd.S

    @cached_property
    def T(self) -> np.ndarray:
        """
        Probability transition matrix.
        """
        return expm(self.S)

    @cached_property
    def s(self) -> np.ndarray:
        """
        Exit rate vector.
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

        R = _invert_reward(np.diag(r[:-1]))

        M = matrix_power(self.cd.U @ R, k)

        # self.Ne ** k is the rescaling due to population size
        return Ne ** k * factorial(k) * alpha[:-1] @ M @ self.cd.e[:-1]

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized cumulative distribution function.

        :param t: Time.
        :return: CDF.
        """

        def cdf(t: float) -> float:
            """
            Cumulative distribution function.

            :param t: Time.
            :return: CDF.
            """
            return 1 - self.cd.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], t / self.cd.Ne) @ self.cd.e[:-1]

        if isinstance(t, Iterable):
            return np.vectorize(cdf)(t)

        return cdf(t)


    def pdf(self, u: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized density function.

        :param u: Time.
        :return: Density.
        """

        def pdf(u: float) -> float:
            """
            Density function.

            :param u: Time.
            :return: Density.
            """
            return self.cd.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], u) @ self.s / self.cd.Ne

        if isinstance(u, Iterable):
            return np.vectorize(pdf)(u)

        return pdf(u)


class PiecewiseConstantPopSizeDistribution(ConstantPopSizeDistribution):
    """
    Phase-type distribution for a piecewise constant population size coalescent.
    """

    def __init__(self, cd: 'PiecewiseConstantPopSizeCoalescent', r: np.ndarray | List):
        """
        Initialize the distribution.

        :param cd: Piecewise constant population size coalescent.
        :param r: Reward vector.
        """
        super().__init__(cd, r)

        self.cd = cd
        self.r = np.array(r)

    @cache
    def _nth_moment(
            self,
            k: int,
            alpha: Tuple[float] = None,
            r: Tuple[Tuple[float]] = None
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Get the nth (non-central) moment per epoch.

        :param k: The kth moment
        :param alpha: Full initial value vector
        :param r: Tuple of length k of full reward vectors
        :return: Moments, moments conditional on no absorption, and absorption probabilities per epoch
        """
        if r is None:
            R = (_invert_reward(self.R),) * k
        else:
            R = tuple(np.diag(_invert_reward(np.array(r_i))) for r_i in r)

        if alpha is None:
            alpha = self.cd.alpha

        # time spent in epochs scaled by Ne
        taus = np.zeros(self.cd.n_epochs)
        taus[:-1] = (self.cd.times[1:] - self.cd.times[:-1])

        # choose a large value for the last epoch
        taus[self.cd.n_epochs - 1] = 10

        # number of lineages
        n = self.cd.n

        # initialize Van Loan matrix holding (rewarded) moments
        M = np.eye(n * (k + 1))

        # iterate through epochs and compute initial values
        for i in range(self.cd.n_epochs):
            # get Van Loan matrix
            A = _get_van_loan_matrix(S=self.cd.S / self.cd.pop_sizes[i], R=R, k=k)

            # get matrix exponential
            B = expm(A * taus[i])

            if i < self.cd.n_epochs - 1:
                # simply multiply by B
                M @= B
            else:
                M_next = M @ B

                # iterate until convergence
                while not np.allclose(M, M_next, rtol=1e-10, atol=1e-16):
                    M = M_next
                    M_next = M @ B

        m = factorial(k) * (alpha @ M[:n, -n:])[-1]

        return m

    def nth_moment(self, k: int, alpha: np.ndarray = None, r: List[np.ndarray] = None, **kwargs) -> float:
        """
        Get the nth (non-central) moment.

        :param k: The kth moment
        :param alpha: Full initial value vector
        :param r: Full reward vectors
        :param kwargs: Additional arguments
        :return: Moment
        """
        if alpha is not None:
            alpha = tuple(alpha)

        if r is not None:
            r = tuple(tuple(r_i) for r_i in r)

        # get moments per epoch
        moment = self._nth_moment(k=k, alpha=alpha, r=r)

        return moment

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
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        def pdf(u: float) -> float:
            # get the cumulative coalescent rate up to time u
            cum = self.cd.demography.get_cum_rate(u)

            # get current coalescent rate
            rate = self.cd.demography.get_rate(u)

            return self.cd.alpha[:-1] @ fractional_matrix_power(self.cd.T[:-1, :-1], cum) @ self.cd.s * rate

        return np.vectorize(pdf)(u)


class SFSDistribution(PiecewiseConstantPopSizeDistribution):
    """
    Site-frequency spectrum distribution.
    """

    @cache
    def nth_moment(self, k: int, alpha: tuple = None, i: int = None) -> SFS | float:
        """
        Get the nth moment.

        :param k: The order of the moment
        :param alpha: Full initial state vector
        :param i: The ith SFS count. Return full SFS if not specified.
        :return: The nth moment
        """
        # When calculating the first moment
        # we can easily merge the states of the different tree topologies
        # by taking the average of those states weighted by the probability
        # of the tree topology. This is much faster than the general case
        # as we work with a much smaller state space.
        if k == 1:
            R, cd = self._sfs_rewards, self.cd
        else:
            R, cd = self._dist

        def get_count(i: int) -> float:
            """
            Get the moment of ith frequency count.

            :param i: The ith frequency count
            :return: The moment
            """
            return cd.tree_height.nth_moment(k=k, r=(_invert_reward(R[i]),) * k, alpha=alpha)

        # return ith count only if specified
        if i is not None:
            return get_count(i)

        # calculate moments in parallel
        moments = _parallelize(
            func=get_count,
            data=list(range(R.shape[0] - 1)),
            desc=f"Calculating moments of order {k}",
            pbar=self.cd.pbar,
            parallelize=self.cd.parallelize
        )

        return SFS([0] + list(moments) + [0])
    
    @cached_property
    def _sfs_rewards(self) -> np.ndarray:
        """
        Get the rewards suitable for calculating the expected SFS using a reduced state
        space where we don't distinguish between different topologies.
        :return: 
        """
        probs = self.cd.model.get_sample_config_probs(self.cd.n)

        R = np.zeros((self.cd.n, self.cd.n))

        for state, prob in probs.items():
            R[:, self.cd.n - sum(state)] += prob * np.array(state)

        return R

    @cached_property
    def _dist(self):
        """
        Get the rewards and coalescent distribution suitable for calculating
        higher SFS moments. In order to distinguish between different topologies
        we require a larger state space.

        :return: The reward matrix and the coalescent distribution
        """
        # obtain intensity matrix for state space that distinguishes between
        # the different tree topologies
        S, R = self.cd.model.get_rate_matrix_infinite_alleles(self.cd.n)

        # get coalescent distribution
        cd = PiecewiseConstantPopSizeCoalescent(
            n=S.shape[0],
            S_sub=S[:-1, :-1],
            demography=self.cd.demography
        )

        return R.T, cd

    @cached_property
    def cov(self) -> SFS2:
        """
        If no arguments are given, get the 2-SFS, i.e. the covariance matrix of the site-frequencies.

        :param alpha: Full initial state vector
        :return: A 2-SFS or a single covariance
        """
        R, cd = self._dist

        def get(args: Tuple[int, int]) -> float:
            """
            Calculate the expectation of the product between the ith and jth site-frequency.

            :param args: Tuple of i and j
            :return: Covariance
            """
            i, j = args

            return cd.tree_height.nth_moment(
                k=2,
                r=(_invert_reward(R[i]), _invert_reward(R[j]))
            )

        # create list of arguments for each combination of i, j
        indices = [(i, j) for i in range(R.shape[0] - 1) for j in range(R.shape[0] - 1)]

        # get sfs using parallelized function
        sfs_results = _parallelize(
            func=get,
            data=indices,
            desc="Calculating covariance",
            pbar=self.cd.pbar,
            parallelize=self.cd.parallelize
        )

        # re-structure the results to a matrix form
        sfs = np.zeros((self.cd.n + 1, self.cd.n + 1))
        for ((i, j), result) in zip(indices, sfs_results):
            sfs[i + 1, j + 1] = result

        # get matrix of marginal second moments
        m2 = np.outer(self.mean.data, self.mean.data)

        # calculate covariances
        cov = (sfs + sfs.T) / 2 - m2

        return SFS2(cov)

    @cache
    def get_cov(self, i: int, j: int) -> float:
        """
        Get the covariance between the ith and jth site-frequency.

        :param i: The ith site-frequency
        :param j: The jth site-frequency
        :return:
        """
        R, cd = self._dist

        return cd.tree_height.nth_moment(
            k=2,
            r=(_invert_reward(R[i]), _invert_reward(R[j]))
        )

    @cached_property
    def corr(self) -> SFS2:
        """
        If no arguments are given, get the 2-SFS, i.e. the correlation matrix of the site-frequencies.
        If i and j are given, get the correlation coefficient between the ith and jth site-frequency.

        :return: The correlation coefficient matrix
        """
        # get standard deviations
        std = np.sqrt(self.var.data)

        return SFS2(self.cov.data / np.outer(std, std))

    @cache
    def get_corr(self, i: int, j: int) -> float:
        """
        Get the correlation coefficient between the ith and jth site-frequency.

        :param i: The ith site-frequency
        :param j: The jth site-frequency
        :return: Correlation coefficient
        """
        std_i = np.sqrt(self.nth_moment(k=2, i=i))
        std_j = np.sqrt(self.nth_moment(k=2, i=j))

        return self.get_cov(i, j) / (std_i * std_j)


class EmpiricalDistribution(ProbabilityDistribution):
    """
    Probability distribution based on realisations.
    """

    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples: Samples.
        """
        self.samples = np.array(samples, dtype=float)

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the mean absorption time.

        :param alpha: Initial state.
        :return: Mean absorption time.
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the variance in the absorption time.

        :param alpha: Initial state.
        :return: Variance in the absorption time.
        """
        return np.var(self.samples, axis=0)

    @cached_property
    def m2(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the second moment.

        :param alpha: Initial state.
        :return: Second moment.
        """
        return np.mean(self.samples ** 2, axis=0)

    @cached_property
    def cov(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the covariance matrix.

        :param alpha: Initial state.
        :return: Second moment.
        """
        return np.nan_to_num(np.cov(self.samples, rowvar=False))

    @cached_property
    def corr(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the correlation matrix.

        :param alpha: Initial state.
        :return: Second moment.
        """
        return np.nan_to_num(np.corrcoef(self.samples, rowvar=False))

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Time.
        :return: Cumulative probability.
        """
        x = np.sort(self.samples)
        y = np.arange(1, len(self.samples) + 1) / len(self.samples)

        return np.interp(t, x, y)

    def pdf(self, u: float | np.ndarray, n_bins: int = 10000, sigma: float = None) -> float | np.ndarray:
        """
        Density function.

        :param sigma: Sigma for gaussian filter.
        :param n_bins: Number of bins.
        :param u: Time.
        :return: Density.
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


class EmpiricalSFSDistribution(EmpiricalDistribution):
    """
    SFS Probability distribution based on realisations.
    """

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> SFS:
        """
        Get the mean absorption time.

        :param alpha: Initial state.
        :return: Mean absorption time.
        """
        return SFS(super().mean)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> SFS:
        """
        Get the variance in the absorption time.

        :param alpha: Initial state.
        :return: Variance in the absorption time.
        """
        return SFS(super().var)

    @cached_property
    def m2(self, alpha: np.ndarray = None) -> SFS:
        """
        Get the second moment.

        :param alpha: Initial state.
        :return: Second moment.
        """
        return SFS(super().m2)

    @cached_property
    def cov(self, alpha: np.ndarray = None) -> SFS2:
        """
        Get the covariance matrix.

        :param alpha: Initial state.
        :return: Second moment.
        """
        return SFS2(np.nan_to_num(np.cov(self.samples, rowvar=False)))

    @cached_property
    def corr(self, alpha: np.ndarray = None) -> SFS2:
        """
        Get the correlation matrix.

        :param alpha: Initial state.
        :return: Second moment.
        """
        return SFS2(np.nan_to_num(np.corrcoef(self.samples, rowvar=False)))


class Coalescent:

    demography: Demography

    @property
    @abstractmethod
    def tree_height(self) -> ProbabilityDistribution:
        """
        Tree height distribution.
        """
        pass

    @property
    @abstractmethod
    def total_branch_length(self) -> ProbabilityDistribution:
        """
        Total branch length distribution.
        """
        pass

    @property
    @abstractmethod
    def sfs(self) -> SFSDistribution:
        """
        Site frequency spectrum distribution.
        """
        pass


class ConstantPopSizeCoalescent(Coalescent):
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
            pbar: bool = True,
            parallelize: bool = True
    ):
        """
        Create object.

        :param n: Number of lineages.
        :param model: Coalescent model.
        :param alpha: Initial state vector.
        :param Ne: Effective population size.
        :param S_sub: Sub-intensity matrix (advanced use, ``model`` will be ignored).
        :param pbar: Whether to show a progress bar
        :param parallelize: Whether to parallelize computations.
        """
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

        #: Whether to show a progress bar
        self.pbar: bool = pbar

        #: Whether to parallelize computions
        self.parallelize: bool = parallelize

        if S_sub is None:
            # obtain sub-intensity matrix
            S_sub = self.model.get_rate_matrix(self.n)

        # exit rate vector
        self.s = -S_sub @ self.e[:-1]

        # obtain full intensity matrix
        self.S = np.block([
            [S_sub, self.s[:, None]],
            [np.zeros(self.n)]
        ])

    @cached_property
    def e(self) -> np.ndarray:
        """
        Vector with ones of size ``n``.
        """
        return np.ones(self.n)

    @cached_property
    def T(self) -> np.ndarray:
        """
        Transition matrix.
        """
        return expm(self.S)

    @cached_property
    def t(self) -> np.ndarray:
        """
        Exit probability vector.
        """
        return 1 - self.T @ self.e

    @cached_property
    def U(self) -> np.ndarray:
        """
        Green matrix (negative inverse of sub-intensity matrix).
        """
        return -inv(self.S[:-1, :-1])

    @cached_property
    def T_inv(self) -> np.ndarray:
        """
        Inverse of transition matrix.
        """
        return inv(self.T)

    @cached_property
    def tree_height(self) -> ConstantPopSizeDistribution:
        """
        Tree height distribution.
        """
        return ConstantPopSizeDistribution(
            cd=self,
            r=_pad(np.ones(self.n - 1))
        )

    @cached_property
    def total_branch_length(self) -> ConstantPopSizeDistribution:
        """
        Total branch length distribution.
        """
        return ConstantPopSizeDistribution(
            cd=self,
            r=_pad(1 / np.arange(2, self.n + 1)[::-1])
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site frequency spectrum distribution.

        TODO implement or scrap this class.
        """
        raise NotImplementedError("Not implemented.")


class PiecewiseConstantPopSizeCoalescent(ConstantPopSizeCoalescent):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            demography: PiecewiseConstantDemography = None,
            S_sub: np.ndarray = None,
            pbar: bool = True,
            parallelize: bool = True
    ):
        """
        Create object.

        :param n: Number of lineages.
        :param model: Coalescent model.
        :param alpha: Initial state vector.
        :param S_sub: Sub-intensity matrix (advanced use, ``model`` will be ignored).
        :param pbar: Whether to show a progress bar
        :param parallelize: Whether to parallelize computations
        """
        super().__init__(
            model=model,
            n=n,
            alpha=alpha,
            Ne=1,
            S_sub=S_sub,
            pbar=pbar,
            parallelize=parallelize
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
        """
        return PiecewiseConstantPopSizeDistribution(
            cd=self,
            r=_pad(np.ones(self.n - 1))
        )

    @cached_property
    def total_branch_length(self) -> PiecewiseConstantPopSizeDistribution:
        """
        Total branch length distribution.
        """
        return PiecewiseConstantPopSizeDistribution(
            cd=self,
            r=_pad(1 / np.arange(2, self.n + 1)[::-1])
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site frequency spectrum distribution.
        """
        return SFSDistribution(self, r=_pad(np.ones(self.n - 1)))


class MsprimeCoalescent(Coalescent):
    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List = None,
            times: np.ndarray | List = None,
            start_time: float = None,
            end_time: float = None,
            exclude_unfinished: bool = True,
            exclude_finished: bool = False,
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
        :param start_time: Time when to start the simulation
        :param end_time: Time when to end the simulation
        :param exclude_unfinished: Whether to exclude unfinished trees when calculating the statistics
        :param exclude_unfinished: Whether to exclude finished trees when calculating the statistics
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
        self.start_time = start_time
        self.end_time = end_time
        self.exclude_unfinished = exclude_unfinished
        self.exclude_finished = exclude_finished
        self.growth_rate = growth_rate
        self.N0 = N0
        self.num_replicates = num_replicates
        self.n_threads = n_threads
        self.parallelize = parallelize

        self.p_accepted = 0

    @cache
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
            :return:
            """
            # simulate trees
            g: Generator = ms.sim_ancestry(
                samples=self.n,
                num_replicates=num_replicates,
                demography=d,
                model=ms.StandardCoalescent(),
                ploidy=1,
                end_time=self.end_time
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
                heights[i] = np.sum([t.time(r) for r in t.roots])
                sfs[i] = _calculate_sfs(t)

            return np.concatenate([[heights.T], [total_branch_lengths.T], sfs.T])

        # parallelize and add up results
        res = np.hstack(_parallelize(
            func=simulate_batch,
            data=[None] * self.n_threads,
            parallelize=self.parallelize,
            batch_size=num_replicates,
            desc="Simulating trees"
        ))

        if self.exclude_unfinished:

            if self.end_time is not None:
                res = res[:, res[0] <= self.end_time]

        if self.exclude_finished:

            if self.end_time is not None:
                res = res[:, res[0] >= self.end_time]

        if self.start_time is not None:
            res = res[:, res[0] >= self.start_time]

        self.p_accepted = res.shape[1] / self.num_replicates

        # store results
        self.heights, self.total_branch_lengths, self.sfs_counts = res[0], res[1], res[2:]

    def _touch(self):
        """
        Touch cached properties.
        """
        self.total_branch_length.touch()
        self.tree_height.touch()
        self.sfs.touch()

    def _drop(self):
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
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.heights)

    @cached_property
    def total_branch_length(self) -> EmpiricalDistribution:
        """
        Total branch length distribution.
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.total_branch_lengths)

    @cached_property
    def sfs(self) -> EmpiricalDistribution:
        """
        Site frequency spectrum distribution.
        """
        self.simulate()

        return EmpiricalSFSDistribution(samples=self.sfs_counts.T)