import logging
from abc import ABC, abstractmethod
from functools import cached_property, cache
from itertools import chain
from math import factorial
from typing import Generator, List, Callable, Tuple, Iterable, Dict, Iterator

import msprime as ms
import numpy as np
import tskit
from matplotlib import pyplot as plt
from msprime import AncestryModel
from multiprocess import Pool
from numpy.linalg import matrix_power
from numpy.polynomial.hermite_e import HermiteE
from scipy import special
from scipy.linalg import expm, fractional_matrix_power
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from tqdm import tqdm

from .coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from .demography import Demography, TimeHomogeneousDemography
from .population import PopConfig
from .rewards import Reward, TreeHeightReward, TotalBranchLengthReward, SFSReward
from .spectrum import SFS, SFS2
from .state_space import BlockCountingStateSpace, DefaultStateSpace, StateSpace
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
    Parallelize given function or execute sequentially.

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


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    @abstractmethod
    @cached_property
    def mean(self) -> float:
        """
        Get the mean absorption time.

        :return: The mean absorption time.
        """
        pass

    @abstractmethod
    @cached_property
    def var(self) -> float:
        """
        Get the variance in the absorption time.

        :return: The variance in the absorption time.
        """
        pass

    @abstractmethod
    @cached_property
    def m2(self) -> float:
        """
        Get the second (non-central) moment of the absorption time.

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

        :param t: Value or values to evaluate the CDF at.
        :return: CDF.
        """
        pass

    @abstractmethod
    def pdf(self, u) -> float | np.ndarray:
        """
        Density function.

        :param u: Value or values to evaluate the density function at.
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
            label: str = None,
            title: str = 'CDF'
    ) -> plt.axes:
        """
        Plot cumulative distribution function.

        :param ax: Axes to plot on.
        :param x: Values to evaluate the CDF at.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
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
            title=title
        )

    def plot_pdf(
            self,
            ax: plt.Axes = None,
            x: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'PDF'
    ) -> plt.axes:
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param x: Values to evaluate the density function at.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
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
            title=title
        )


class PhaseTypeDistribution(ProbabilityDistribution, ABC):
    """
    Abstract base class for phase-type distributions.
    """

    @staticmethod
    def _get_van_loan_matrix(R: List[np.ndarray], S: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Get the block matrix for the given reward matrices and transition matrix.

        :param R: List of length k of reward matrices
        :param S: Intensity matrix
        :param k: The kth moment to evaluate
        :return: Van Loan matrix which is a block matrix of size (k + 1) * (k + 1)
        """
        # matrix of zeros
        O = np.zeros_like(S)

        # create compound matrix
        return np.block([[S if i == j else R[i] if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)])

    @abstractmethod
    def moment(self, k: int) -> float | SFS:
        """
        Get the nth moment.

        :param k: The order of the moment
        :return: The nth moment
        """
        pass

    @cached_property
    def mean(self) -> float | SFS:
        """
        Get the mean absorption time.

        :return: The mean absorption time.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> float | SFS:
        """
        Get the variance in the absorption time.

        :return: The variance in the absorption time.
        """
        return self.moment(k=2) - self.moment(k=1) ** 2

    @cached_property
    def m2(self) -> float | SFS:
        """
        Get the (non-central) second moment.

        :return: The second moment.
        """
        return self.moment(k=2)


class TimeHomogeneousDistribution(PhaseTypeDistribution):
    """
    Phase-type distribution for a time-homogeneous model.
    """

    def __init__(
            self,
            pop_config: PopConfig,
            state_space: DefaultStateSpace,
            reward: Reward = TreeHeightReward(),
            demography: TimeHomogeneousDemography = TimeHomogeneousDemography()
    ):
        """
        Initialize the distribution.

        :param pop_config: The population configuration.
        :param state_space: The state space.
        :param reward: The reward.
        :param demography: Time-homogeneous demography.
        """
        # raise error if not time-homogeneous
        if not isinstance(demography, TimeHomogeneousDemography):
            raise NotImplementedError('TimeHomogeneousDistribution only supports time-homogeneous demographies.')

        #: The reward.
        self.reward: Reward = reward

        #: The population configuration.
        self.pop_config: PopConfig = pop_config

        #: The state space.
        self.state_space: DefaultStateSpace = state_space

        #: The demography.
        self.demography: TimeHomogeneousDemography = demography

    @cached_property
    def S(self) -> np.ndarray:
        """
        Intensity matrix with rewards.
        """
        return self.reward.get(self.state_space) @ self.state_space.S

    @cached_property
    def T(self) -> np.ndarray:
        """
        Probability transition matrix with rewards.
        """
        return expm(self.S)

    @cached_property
    def s(self) -> np.ndarray:
        """
        Exit rate vector.
        """
        return -self.S[:-1, :-1] @ self.state_space.e[:-1]

    @cached_property
    def e(self) -> np.ndarray:
        """
        Vector with ones of size ``n``.
        """
        return self.state_space.e

    @cached_property
    def alpha(self) -> np.ndarray:
        """
        Initial state vector.
        """
        return self.pop_config.get_initial_states(self.state_space)

    def moment(self, k: int, reward: Reward = None) -> float:
        """
        Get the nth moment.

        TODO deprecate this

        :param k: The order of the moment
        :param reward: The reward.
        :return: The nth moment
        """
        if reward is None:
            reward = self.reward

        R = reward.get(state_space=self.state_space)[:-1, :-1]

        M = matrix_power(self.state_space.U @ R, k)

        return factorial(k) * self.alpha[:-1] @ M @ self.e[:-1]

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized cumulative distribution function.
        TODO doesn't work for total branch length

        :param t: Value or values to evaluate the CDF at.
        :return: CDF.
        """

        def cdf(t: float) -> float:
            """
            Cumulative distribution function.

            :param t: Value to evaluate the CDF at.
            :return: CDF.
            """
            return 1 - self.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], t) @ self.e[:-1]

        if isinstance(t, Iterable):
            return np.vectorize(cdf)(t)

        return cdf(t)

    def pdf(self, u: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized density function.
        TODO doesn't work for total branch length

        :param u: Value or values to evaluate the density function at.
        :return: Density.
        """

        def pdf(u: float) -> float:
            """
            Density function.

            :param u: Value to evaluate the density function at.
            :return: Density.
            """
            return self.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], u) @ self.s

        if isinstance(u, Iterable):
            return np.vectorize(pdf)(u)

        return pdf(u)


class PiecewiseTimeHomogeneousDistribution(PhaseTypeDistribution):
    """
    Phase-type distribution for a piecewise constant population size coalescent.
    """

    #: Maximum number iterations in the last epoch.
    max_iter = 100

    def __init__(
            self,
            pop_config: PopConfig,
            state_space: StateSpace,
            demography: Demography = TimeHomogeneousDemography(),
            reward: Reward = TreeHeightReward(),
    ):
        """
        Initialize the distribution.

        :param pop_config: The population configuration.
        :param state_space: The state space.
        :param demography: The demography.
        :param reward: The reward.
        """
        #: Population configuration
        self.pop_config: PopConfig = pop_config
        
        #: Reward
        self.reward: Reward = reward
        
        #: State space
        self.state_space: StateSpace = state_space

        #: Demography
        self.demography: Demography = demography

    @cached_property
    def alpha(self) -> np.ndarray:
        """
        Initial state vector.
        """
        return self.pop_config.get_initial_states(self.state_space)

    @cache
    def moment(
            self,
            k: int,
            rewards: Tuple[Reward] = None
    ) -> float:
        """
        Get the nth (non-central) moment.

        :param k: The kth moment
        :param rewards: The rewards
        :return: The kth moment
        """
        # use default reward if not specified
        if rewards is None:
            rewards = [self.reward] * k

        # get reward matrix
        R = [r.get(state_space=self.state_space) for r in rewards]

        # number of states
        n_states = self.state_space.k

        # initialize Van Loan matrix holding (rewarded) moments
        M = np.eye(n_states * (k + 1))

        # get time and population size, and migration rate generators
        times: Iterator[float] = self.demography.times
        pop_sizes: Dict[str, Iterator[float]] = self.demography.pop_sizes
        migration_rates: Dict[Tuple[str, str], Iterator[float]] = self.demography.migration_rates

        # previous time
        time_prev: float = next(times)

        # iterate through epochs and compute initial values
        for i, time in enumerate(chain(times, [np.inf])):

            # get population sizes for this epoch
            pop_size = dict((p, next(pop_sizes[p])) for p in pop_sizes)

            # get migration rates for this epoch
            migration_rate = dict((p, next(migration_rates[p])) for p in migration_rates)

            # get state space for this epoch
            self.state_space.update_demography(
                demography=TimeHomogeneousDemography(
                    pop_sizes=pop_size,
                    migration_rates=migration_rate
                )
            )

            # get Van Loan matrix
            A = self._get_van_loan_matrix(S=self.state_space.S, R=R, k=k)

            # if not in the last epoch
            if time < np.inf:

                tau = time - time_prev
                B = expm(A * tau)

                # simply multiply by B
                M @= B

                # noinspection all
                time_prev = time
            else:
                # if in the last epoch, we need to iterate until convergence
                max_pop_size = np.max(list(pop_size.values()))

                # determine tau
                tau = max_pop_size * 1000

                B = expm(A * tau)
                M_next = M @ B

                i = 0
                # iterate until convergence
                while not np.allclose(M, M_next, rtol=1e-10, atol=1e-16):

                    if i > self.max_iter:
                        # TODO what to do if we have demes that are unreachable and that start without lineages?
                        raise RuntimeError(f"No convergence after {int(tau * self.max_iter)} time steps. "
                                           f"Check if the demography is well defined and consider increasing "
                                           f"the maximum number of iterations (max_iter) if necessary.")

                    M = M_next
                    M_next = M @ B

                    i += 1

        # compute moment
        m = factorial(k) * self.alpha @ M[:n_states, -n_states:] @ self.state_space.e

        return m

    '''
    def cumulant(self, k: int, rewards: Tuple[Reward] = None) -> float:
        """
        Get the nth cumulant.

        :param k: The order of the cumulant
        :param rewards: The rewards
        :return: The nth cumulant
        """
        return cum2mc([self.moment(k=k, rewards=rewards) for k in range(1, k + 1)])[-1]
    '''

    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.

        TODO extrapolate from moments and deprecate get_rate and get_cum_rate

        :param t: Value or values to evaluate the CDF at.
        :return: Cumulative probability
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not isinstance(self.reward, TreeHeightReward):
            raise NotImplementedError("CDF not implemented for non-default rewards.")

        # raise error if multiple populations
        if self.demography.n_pops > 1:
            raise NotImplementedError("CDF not implemented for multiple populations.")

        # get the transition matrix for the standard coalescent
        self.state_space.update_demography(
            demography=TimeHomogeneousDemography()
        )

        def cdf(t: float) -> float:
            """
            Cumulative distribution function.

            :param t: Time.
            :return: Cumulative probability.
            """
            # get the cumulative coalescent rate up to time t
            cum = self.demography.get_cum_rate(t)[self.demography.pop_names[0]]

            return 1 - self.alpha[:-1] @ fractional_matrix_power(self.state_space.T[:-1, :-1],
                                                                 cum) @ self.state_space.e[:-1]

        return np.vectorize(cdf)(t)

    def pdf(self, u: float | np.ndarray, n_moments: int = 11) -> float | np.ndarray:
        """
        Density function.

        TODO extrapolate from moments and deprecate get_rate and get_cum_rate

        :param u: Value or values to evaluate the density function at.
        :param n_moments: Number of moments to use when approximating the distribution using Hermite polynomials.
        :return: Density
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if isinstance(self.reward, TreeHeightReward) and self.demography.n_pops == 1:
            # get the transition matrix for the standard coalescent
            self.state_space.update_demography(
                demography=TimeHomogeneousDemography()
            )

            def pdf(u: float) -> float:
                """
                Density function.

                :param u: Time.
                :return: Density.
                """
                # get the cumulative coalescent rate up to time u
                cum = self.demography.get_cum_rate(u)[self.demography.pop_names[0]]

                # get current coalescent rate
                rate = self.demography.get_rate(u)[self.demography.pop_names[0]]

                return self.alpha[:-1] @ fractional_matrix_power(self.state_space.T[:-1, :-1],
                                                                 cum) @ self.state_space.s * rate

            return np.vectorize(pdf)(u)

        raise NotImplementedError("PDF not implemented for non-default rewards or multiple populations.")

        # get moments
        moments = np.array([self.moment(k=k) for k in range(1, n_moments + 1)])

        return _GramCharlierExpansion.pdf(u, moments)


class SFSDistribution(PhaseTypeDistribution):
    """
    Site-frequency spectrum distribution.
    """

    def __init__(
            self,
            pop_config: PopConfig,
            state_space: BlockCountingStateSpace,
            demography: Demography,
            pbar: bool = False,
            parallelize: bool = False
    ):
        """
        Initialize the distribution.

        :param pop_config: The population configuration.
        :param state_space: Block counting state space.
        :param demography: The demography.
        :param pbar: Whether to show a progress bar.
        :param parallelize: Use parallelization.
        """
        #: Population configuration
        self.pop_config: PopConfig = pop_config

        #: State space for the block counting process
        self.state_space: BlockCountingStateSpace = state_space

        #: Demography
        self.demography: Demography = demography

        #: Whether to show a progress bar
        self.pbar: bool = pbar

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

    @cache
    def moment(self, k: int, i: int = None) -> SFS | float:
        """
        Get the nth moment.

        :param k: The order of the moment
        :param i: The ith SFS count. Return full SFS if not specified.
        :return: The nth moment
        """

        def get_count(i: int) -> float:
            """
            Get the moment of ith frequency count.

            :param i: The ith frequency count
            :return: The moment
            """
            d = PiecewiseTimeHomogeneousDistribution(
                pop_config=self.pop_config,
                reward=SFSReward(i),
                state_space=self.state_space,
                demography=self.demography
            )

            return d.moment(
                k=k
            )

        # return ith count only if specified
        if i is not None:
            return get_count(i)

        # calculate moments in parallel
        moments = _parallelize(
            func=get_count,
            data=np.arange(self.pop_config.n - 1),
            desc=f"Calculating moments of order {k}",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        return SFS([0] + list(moments) + [0])

    @cached_property
    def cov(self) -> SFS2:
        """
        If no arguments are given, get the 2-SFS, i.e. the covariance matrix of the site-frequencies.

        :return: A 2-SFS or a single covariance value.
        """
        # create list of arguments for each combination of i, j
        indices = [(i, j) for i in range(self.pop_config.n - 1) for j in range(self.pop_config.n - 1)]

        # get sfs using parallelized function
        sfs_results = _parallelize(
            func=lambda args: self.get_cov(*args),
            data=indices,
            desc="Calculating covariance",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        # re-structure the results to a matrix form
        sfs = np.zeros((self.pop_config.n + 1, self.pop_config.n + 1))
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
        d = PiecewiseTimeHomogeneousDistribution(
            pop_config=self.pop_config,
            state_space=self.state_space,
            demography=self.demography
        )

        return d.moment(
            k=2,
            rewards=(SFSReward(i), SFSReward(j))
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

        sfs = SFS2(self.cov.data / np.outer(std, std))

        # replace NaNs with zeros
        sfs.data[np.isnan(sfs.data)] = 0

        return sfs

    @cache
    def get_corr(self, i: int, j: int) -> float:
        """
        Get the correlation coefficient between the ith and jth site-frequency.

        :param i: The ith site-frequency
        :param j: The jth site-frequency
        :return: Correlation coefficient
        """
        std_i = np.sqrt(self.moment(k=2, i=i))
        std_j = np.sqrt(self.moment(k=2, i=j))

        return self.get_cov(i, j) / (std_i * std_j)

    def pdf(self, u: float | np.ndarray) -> float | np.ndarray:
        """
        Density function.
        TODO implement this

        :param u: Time or time points
        :return: Density.
        """
        raise NotImplementedError("PDF not implemented for site-frequency spectrum distribution.")

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Cumulative distribution function.
        TODO implement this

        :param t: Time or time points.
        :return: Cumulative probability.
        """
        raise NotImplementedError("CDF not implemented for site-frequency spectrum distribution.")


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
    def mean(self) -> float | np.ndarray:
        """
        Get the mean absorption time.

        :return: Mean absorption time.
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self) -> float | np.ndarray:
        """
        Get the variance in the absorption time.

        :return: Variance in the absorption time.
        """
        return np.var(self.samples, axis=0)

    @cached_property
    def m2(self) -> float | np.ndarray:
        """
        Get the second moment.

        :return: Second moment.
        """
        return np.mean(self.samples ** 2, axis=0)

    @cached_property
    def cov(self) -> float | np.ndarray:
        """
        Get the covariance matrix.

        :return: Second moment.
        """
        return np.nan_to_num(np.cov(self.samples, rowvar=False))

    @cached_property
    def corr(self) -> float | np.ndarray:
        """
        Get the correlation matrix.

        :return: Second moment.
        """
        return np.nan_to_num(np.corrcoef(self.samples, rowvar=False))

    def moment(self, k: int) -> float | np.ndarray:
        """
        Get the nth moment.

        :param k: The order of the moment
        :return: The nth moment
        """
        return np.mean(self.samples ** k, axis=0)

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
    SFS probability distribution based on realisations.
    """

    @cached_property
    def mean(self) -> SFS:
        """
        Get the mean absorption time.

        :return: Mean absorption time.
        """
        return SFS(super().mean)

    @cached_property
    def var(self) -> SFS:
        """
        Get the variance in the absorption time.

        :return: Variance in the absorption time.
        """
        return SFS(super().var)

    @cached_property
    def m2(self) -> SFS:
        """
        Get the second moment.

        :return: Second moment.
        """
        return SFS(super().m2)

    @cached_property
    def cov(self) -> SFS2:
        """
        Get the covariance matrix.

        :return: Second moment.
        """
        return SFS2(np.nan_to_num(np.cov(self.samples, rowvar=False)))

    @cached_property
    def corr(self) -> SFS2:
        """
        Get the correlation matrix.

        :return: Second moment.
        """
        return SFS2(np.nan_to_num(np.corrcoef(self.samples, rowvar=False)))


class Coalescent(ABC):
    """
    Coalescent distribution.
    This class provides probability distributions for the tree height, total branch length and site frequency spectrum.
    """
    #: Demography
    demography: Demography

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | np.ndarray | PopConfig,
            model: CoalescentModel = StandardCoalescent()
    ):
        """
        Create object.

        :param n: n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`PopulationConfig` object can be passed.
        :param model: Coalescent model.
        """
        if not isinstance(n, PopConfig):
            #: Population configuration
            self.pop_config: PopConfig = PopConfig(n)
        else:
            #: Population configuration
            self.pop_config: PopConfig = n

        #: Coalescent model
        self.model: CoalescentModel = model

    @property
    @abstractmethod
    def tree_height(self) -> PhaseTypeDistribution:
        """
        Tree height distribution.
        """
        pass

    @property
    @abstractmethod
    def total_branch_length(self) -> PhaseTypeDistribution:
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


class TimeHomogeneousCoalescent(Coalescent):
    """
    Coalescent distribution for a constant population size.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | np.ndarray | PopConfig,
            model: CoalescentModel = StandardCoalescent(),
            demography: TimeHomogeneousDemography = TimeHomogeneousDemography(),
            pbar: bool = True,
            parallelize: bool = True
    ):
        """
        Create object.

        :param n: n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`PopulationConfig` object can be passed.
        :param model: Coalescent model.
        :param demography: Time-homogeneous demography.
        :param pbar: Whether to show a progress bar
        :param parallelize: Whether to parallelize computations.
        """
        super().__init__(n=n, model=model)

        #: Time-homogeneous demography
        self.demography: TimeHomogeneousDemography = demography

        # check if the demography and population configuration have the same population names
        if set(self.demography.pop_names) != set(self.pop_config.pop_names):
            raise ValueError("Population names in population configuration and demography do not match. ")

        #: Whether to show a progress bar
        self.pbar: bool = pbar

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

    @cached_property
    def _state_space(self) -> DefaultStateSpace:
        """
        The default state space.
        """
        return DefaultStateSpace(
            pop_config=self.pop_config,
            model=self.model,
            demography=self.demography
        )

    @cached_property
    def _state_space_BCP(self) -> BlockCountingStateSpace:
        """
        The block counting state space.
        """
        return BlockCountingStateSpace(
            pop_config=self.pop_config,
            model=self.model,
            demography=self.demography
        )

    @cached_property
    def tree_height(self) -> TimeHomogeneousDistribution:
        """
        Tree height distribution.
        """
        return TimeHomogeneousDistribution(
            pop_config=self.pop_config,
            reward=TreeHeightReward(),
            state_space=self._state_space,
            demography=self.demography
        )

    @cached_property
    def total_branch_length(self) -> TimeHomogeneousDistribution:
        """
        Total branch length distribution.
        """
        return TimeHomogeneousDistribution(
            pop_config=self.pop_config,
            reward=TotalBranchLengthReward(),
            state_space=self._state_space,
            demography=self.demography
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site frequency spectrum distribution.
        """
        return SFSDistribution(
            pop_config=self.pop_config,
            state_space=self._state_space_BCP,
            demography=self.demography
        )


class PiecewiseTimeHomogeneousCoalescent(TimeHomogeneousCoalescent):
    """
    Coalescent distribution for the piecewise time-homogeneous coalescent.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | np.ndarray | PopConfig,
            model: CoalescentModel = StandardCoalescent(),
            demography: Demography = None,
            pbar: bool = True,
            parallelize: bool = True
    ):
        """
        Create object.

        :param :param n: n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`PopulationConfig` object can be passed.
        :param model: Coalescent model.
        :param pbar: Whether to show a progress bar
        :param parallelize: Whether to parallelize computations
        """
        # get population sizes for first epoch
        pop_size = dict((p, next(demography.pop_sizes[p])) for p in demography.pop_names)

        # get migration rates for first epoch
        migration_rates = dict(((p1, p2), next(demography.migration_rates[(p1, p2)]))
                               for p1, p2 in demography.migration_rates)

        super().__init__(
            model=model,
            n=n,
            demography=TimeHomogeneousDemography(
                pop_sizes=pop_size,
                migration_rates=migration_rates
            ),
            pbar=pbar,
            parallelize=parallelize
        )

        #: Demography, possibly piecewise time-homogeneous
        self._demography: Demography = demography

    @cached_property
    def tree_height(self) -> PiecewiseTimeHomogeneousDistribution:
        """
        Tree height distribution.
        """
        return PiecewiseTimeHomogeneousDistribution(
            pop_config=self.pop_config,
            reward=TreeHeightReward(),
            state_space=self._state_space,
            demography=self._demography
        )

    @cached_property
    def total_branch_length(self) -> PiecewiseTimeHomogeneousDistribution:
        """
        Total branch length distribution.
        """
        return PiecewiseTimeHomogeneousDistribution(
            pop_config=self.pop_config,
            reward=TotalBranchLengthReward(),
            state_space=self._state_space,
            demography=self._demography
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site frequency spectrum distribution.
        """
        return SFSDistribution(
            pop_config=self.pop_config,
            state_space=self._state_space_BCP,
            demography=self._demography
        )


class MsprimeCoalescent(Coalescent):
    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | np.ndarray | PopConfig,
            demography: Demography = None,
            model: CoalescentModel = StandardCoalescent(),
            migration_rates: Dict[Tuple[str, str], float] = None,
            start_time: float = None,
            end_time: float = None,
            exclude_unfinished: bool = True,
            exclude_finished: bool = False,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True
    ):
        """
        Simulate data using msprime.

        :param n: Number of Lineages.
        :param demography: Demography
        :param model: Coalescent model
        :param migration_rates: Migration matrix
        :param start_time: Time when to start the simulation
        :param end_time: Time when to end the simulation
        :param exclude_unfinished: Whether to exclude unfinished trees when calculating the statistics
        :param exclude_unfinished: Whether to exclude finished trees when calculating the statistics
        :param num_replicates: Number of replicates
        :param n_threads: Number of threads
        :param parallelize: Whether to parallelize
        """
        super().__init__(n=n, model=model)

        self.sfs_counts: np.ndarray | None = None
        self.total_branch_lengths: np.ndarray | None = None
        self.heights: np.ndarray | None = None

        self.demography: Demography = demography
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.exclude_unfinished: bool = exclude_unfinished
        self.exclude_finished: bool = exclude_finished
        self.num_replicates: int = num_replicates
        self.n_threads: int = n_threads
        self.parallelize: bool = parallelize
        self.migration_rates: Dict[Tuple[str, str], float] = migration_rates

        self.p_accepted: int = 0

    @staticmethod
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

    def get_coalescent_model(self) -> AncestryModel:
        """
        Get the coalescent model.

        :return: msprime coalescent model.
        """
        if isinstance(self.model, StandardCoalescent):
            return ms.StandardCoalescent()

        if isinstance(self.model, BetaCoalescent):
            return ms.BetaCoalescent(alpha=self.model.alpha)

        if isinstance(self.model, DiracCoalescent):
            return ms.DiracCoalescent(psi=self.model.psi, c=self.model.c)

    @cache
    def simulate(self):
        """
        Simulate data using msprime.
        Tes
        """
        # number of replicates for one thread
        num_replicates = self.num_replicates // self.n_threads

        demography = self.demography.to_msprime()

        def simulate_batch(_) -> (np.ndarray, np.ndarray, np.ndarray):
            """
            Simulate statistics.

            :param _:
            :return:
            """
            # simulate trees
            g: Generator = ms.sim_ancestry(
                samples=self.pop_config.lineage_dict,
                num_replicates=num_replicates,
                demography=demography,
                model=self.get_coalescent_model(),
                ploidy=1,
                end_time=self.end_time
            )

            # initialize variables
            heights = np.zeros(num_replicates)
            total_branch_lengths = np.zeros(num_replicates)
            sfs = np.zeros((num_replicates, self.pop_config.n + 1))

            # iterate over trees and compute statistics
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):
                t: tskit.Tree = ts.first()
                total_branch_lengths[i] = t.total_branch_length
                heights[i] = np.sum([t.time(r) for r in t.roots])
                sfs[i] = self._calculate_sfs(t)

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


class _GramCharlierExpansion:
    """
    Probability density function approximated from its moments using Hermite polynomials.

    .. note::
        Does not work reliably at all. We get negative values for the density function which seems to be a problem
        with the method in general. Trying some other statsmodels implementation provides similarly cumbersome results.
    """

    @classmethod
    def pdf(cls, x: np.ndarray, moments: np.ndarray[float], mu: float = 0, sigma: float = 1) -> np.ndarray:
        """
        Approximate the distribution using its moments and Hermite polynomials.

        :param x: Array of standard normal values.
        :param moments: List of moments of the distribution.
        :param mu: Mean of the normal distribution.
        :param sigma: Standard deviation of the normal distribution.
        :return: Approximated probability density function values.
        """
        y = np.ones_like(x)

        # d = [cls.d(n, moments, mu, sigma) for n in range(0, len(moments) + 1)]

        for n in range(3, len(moments) + 1):
            y += cls.d(n, moments, mu, sigma) * cls.H(n, (x - mu) / sigma)

        return norm.pdf(x, loc=mu, scale=sigma) * y

    @classmethod
    def H(cls, n: int, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Probabilist's Hermite polynomial of order n.

        :param n: Order of the polynomial.
        :param x: Values to evaluate the polynomial at.
        :return: Hermite polynomial of order n evaluated at x.
        """
        y = np.zeros_like(x)

        for j in range(int(n / 2) + 1):
            y += (((-1) ** j * factorial(n) * x ** (n - 2 * j) / (2 ** j * factorial(n - 2 * j) * factorial(j)))
                  .astype(float))

        return y

    @classmethod
    def d(cls, n: int, moments: np.ndarray[float], mu: float, sigma: float) -> float:
        """
        Coefficient of the Hermite series.

        :param n: Order of the coefficient.
        :param moments: Moments of the distribution.
        :param mu: Mean of the distribution.
        :param sigma: Standard deviation of the distribution.
        :return: Coefficients of the Hermite polynomial of order n.
        """
        y = np.zeros(int(n / 2) + 1)

        for j in range(len(y)):
            y[j] = (-1) ** j / 2 ** j / factorial(j) * cls.E(n - 2 * j, moments, mu, sigma)

        return y.sum()

    @classmethod
    def E(cls, n: int, moments: np.ndarray[float], mu: float, sigma: float) -> float:
        """
        Expectation of the nth standard normal moment.

        :param n: Order of the polynomial.
        :param moments: Moments of the distribution.
        :param mu: Mean of the distribution.
        :param sigma: Standard deviation of the distribution.
        :return: Expectation of the Hermite polynomial of order n.
        """
        # add zeroth moment
        all_moments = np.concatenate((np.array([1]), moments))

        # return all_moments[n] / factorial(n)

        y = np.zeros(n + 1)

        for k in range(len(y)):
            y[k] = (-1) ** k * mu ** (n - k) * all_moments[k] / factorial(n - k) / factorial(k)

        return y.sum() / sigma ** n


class _EdgeworthExpansion:
    """
    Probability density function approximated from its moments using Edgeworth expansion.

    Adapted from statsmodels.distributions.edgeworth.

    .. note::
        Only works well for values for which the normal distribution is not close to zero, so it is unsuitable for
        approximating long-tailed distributions.
    """

    @staticmethod
    def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
        """
        Standard normal probability density function.

        :param x: Value or values to evaluate the PDF at.
        :return: PDF.
        """
        return np.exp(-x ** 2 / 2.0) / np.sqrt(2 * np.pi)

    @staticmethod
    def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
        """
        Standard normal cumulative distribution function.

        :param x: Value or values to evaluate the CDF at.
        :return: CDF.
        """
        return special.ndtr(x)

    def __init__(self, cum: List[float]):
        """
        Initialize object.

        :param cum: Cumulants.
        """

        self._logger = logger.getChild(self.__class__.__name__)

        self._coef, self._mu, self._sigma = self._compute_coefficients(cum)

        self._herm_pdf = HermiteE(self._coef)

        if self._coef.size > 2:
            self._herm_cdf = HermiteE(-self._coef[1:])
        else:
            self._herm_cdf = lambda x: 0

        # warn if pdf(x) < 0 for some values of x within 4 sigma
        r = np.real_if_close(self._herm_pdf.roots())
        r = (r - self._mu) / self._sigma

        if r[(np.imag(r) == 0) & (np.abs(r) < 4)].any():
            self._logger.warning(f'PDF has zeros at {r}')

    def _pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Probability density function.

        :param x: Value or values to evaluate the PDF at.
        :return: PDF.
        """
        y = (x - self._mu) / self._sigma

        return self._herm_pdf(y) * self._norm_pdf(y) / self._sigma

    def _cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Cumulative distribution function.

        :param x: Value or values to evaluate the CDF at.
        :return: CDF.
        """
        y = (x - self._mu) / self._sigma

        return self._norm_cdf(y) + self._herm_cdf(y) * self._norm_pdf(y)

    def _compute_coefficients(self, cum: List[float]) -> (np.ndarray, float, float):
        """
        Compute coefficients of the Edgeworth expansion for the PDF.

        :param cum: Cumulants.
        :return: Coefficients, mean and standard deviation.
        """
        # scale cumulants by \sigma
        mu, sigma = cum[0], np.sqrt(cum[1])
        lam = np.asarray(cum)
        for j, l in enumerate(lam):
            lam[j] /= cum[1] ** j

        coef = np.zeros(lam.size * 3 - 5)
        coef[0] = 1

        for s in range(lam.size - 2):
            for p in self._generate_partitions(s + 1):
                term = sigma ** (s + 1)

                for (m, k) in p:
                    term *= np.power(lam[m + 1] / factorial(m + 2), k) / factorial(k)

                r = sum(k for (m, k) in p)
                coef[s + 1 + 2 * r] += term

        return coef, mu, sigma

    @classmethod
    def cumulant_from_moments(cls, moments: List[float], n: int) -> float:
        """
        Compute n-th cumulant from moments.

        :param moments: The moments, raw or central
        :param n: The order of the cumulant to compute
        :return: The cumulant
        """
        kappa = 0

        for p in cls._generate_partitions(n):
            r = sum(k for (m, k) in p)
            term = (-1) ** (r - 1) * factorial(r - 1)

            for (m, k) in p:
                term *= np.power(moments[m - 1] / factorial(m), k) / factorial(k)

            kappa += term

        kappa *= factorial(n)

        return kappa

    @classmethod
    def cumulants_from_moments(cls, moments: List[float]) -> List[float]:
        """
        Compute cumulants from moments.

        TODO use mnc2cum implementation which provides more accurate results

        :param moments: The moments, raw or central
        :return: The cumulants
        """
        return [cls.cumulant_from_moments(moments, k) for k in range(1, len(moments) + 1)]

    @classmethod
    def _generate_partitions(cls, n: int):
        """
        Generate partitions of an integer.

        :param n: Integer to partition.
        :return: Partitions formatted as a list of tuples where the first element is the partition and the second
            element is the number of times the partition is repeated.
        """
        x = BlockCountingStateSpace._find_sample_configs(n, n)

        for v in x:

            m = []
            for i in range(0, n):
                if v[i] > 0:
                    m.append((i + 1, v[i]))

            yield m
