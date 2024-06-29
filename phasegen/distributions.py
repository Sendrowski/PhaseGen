"""
Probability distributions.
"""

import copy
import functools
import itertools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property, cache
from math import factorial
from typing import Generator, List, Callable, Tuple, Dict, Collection, Iterable, Iterator, Optional, Sequence, Set, Type

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from .demography import Demography, PopSizeChanges
from .expm import Backend
from .lineage import LineageConfig
from .locus import LocusConfig
from .rewards import Reward, TreeHeightReward, TotalBranchLengthReward, UnfoldedSFSReward, DemeReward, UnitReward, \
    LocusReward, CombinedReward, FoldedSFSReward, SFSReward
from .serialization import Serializable
from .spectrum import SFS, SFS2
from .state_space import BlockCountingStateSpace, LineageCountingStateSpace, StateSpace
from .utils import parallelize, multiset_permutations

expm = Backend.expm

logger = logging.getLogger('phasegen')


def _make_hashable(func: Callable) -> Callable:
    """
    Decorator that makes a function hashable by converting non-hashable arguments to hashable ones.
    """

    @functools.wraps(func)
    def wrapper(self, *args: tuple, **kwargs: dict):
        """
        Wrapper function.

        :param self: Self.
        :return: The result of the function.
        """
        args = list(args)

        for i, arg in enumerate(args):
            if isinstance(arg, (list, np.ndarray)):
                args[i] = tuple(arg)

        for key, value in kwargs.items():
            if isinstance(value, (list, np.ndarray)):
                kwargs[key] = tuple(value)

        return func(self, *args, **kwargs)

    return wrapper


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    def __init__(self):
        """
        Create object.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

    def touch(self, **kwargs):
        """
        Touch all cached properties.

        :param kwargs: Additional keyword arguments.
        """
        for cls in self.__class__.__mro__:
            for attr, value in cls.__dict__.items():
                if isinstance(value, cached_property):
                    getattr(self, attr)


class MomentAwareDistribution(ProbabilityDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    @abstractmethod
    @cached_property
    def mean(self) -> float:
        """
        First moment / mean.
        """
        pass

    @abstractmethod
    @cached_property
    def var(self) -> float:
        """
        Second central moment / variance.
        """
        pass

    @abstractmethod
    @cached_property
    def m2(self) -> float:
        """
        Second (non-central) moment.
        """
        pass


class MarginalDistributions(Mapping, ABC):
    """
    Base class for marginal distributions.
    """

    @abstractmethod
    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix.
        """
        pass

    @abstractmethod
    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix.
        """
        pass

    @abstractmethod
    def get_cov(self, d1, d2) -> float:
        """
        Get the covariance between two marginal distributions.

        :param d1: The index of the first marginal distribution.
        :param d2: The index of the second marginal distribution.
        :return: covariance
        """
        pass

    @abstractmethod
    def get_corr(self, d1, d2) -> float:
        """
        Get the correlation coefficient between two marginal distributions.

        :param d1: The index of the first marginal distribution.
        :param d2: The index of the second marginal distribution.
        :return: correlation coefficient
        """
        pass


class MarginalLocusDistributions(MarginalDistributions):
    """
    Marginal locus distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution'):
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item):
        """
        Get the distribution for the given locus.

        :param item: Deme name.
        :return: Distribution.
        """
        return self.loci[item]

    def __iter__(self) -> Iterator:
        """
        Iterate over distributions.

        :return: Iterator.
        """
        return iter(self.loci)

    def __len__(self) -> int:
        """
        Get the number of distributions.

        :return: Number of distributions.
        """
        return len(self.loci)

    @cached_property
    def loci(self) -> 'MarginalLocusDistributions':
        """
        Distributions marginalized over loci.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        loci = {}
        for locus in range(self.dist.locus_config.n):
            loci[locus] = cls(
                state_space=self.dist.state_space,
                tree_height=self.dist.tree_height,
                demography=self.dist.demography,
                reward=CombinedReward([self.dist.reward, LocusReward(locus)])
            )

        return loci

    def get_cov(self, locus1: int, locus2: int) -> float:
        """
        Get the covariance between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The covariance.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        return self.dist.moment(
            k=2,
            rewards=(
                CombinedReward([self.dist.reward, LocusReward(locus1)]),
                CombinedReward([self.dist.reward, LocusReward(locus2)])
            ),
            center=True
        )

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_cov(i, j) for i in range(n_loci)] for j in range(n_loci)])

    def get_corr(self, locus1: int, locus2: int) -> float:
        """
        Get the correlation coefficient between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The correlation coefficient.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        return self.get_cov(locus1, locus2) / (self.loci[locus1].std * self.loci[locus2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_corr(i, j) for i in range(n_loci)] for j in range(n_loci)])


class MarginalDemeDistributions(MarginalDistributions):
    """
    Marginal deme distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution'):
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item):
        """
        Get the distribution for the given deme.

        :param item: Deme name.
        :return: Distribution.
        """
        return self.demes[item]

    def __iter__(self) -> Iterator:
        """
        Iterate over distributions.

        :return: Iterator.
        """
        return iter(self.demes)

    def __len__(self) -> int:
        """
        Get the number of distributions.

        :return: Number of distributions.
        """
        return len(self.demes)

    @cached_property
    def demes(self) -> 'MarginalDemeDistributions':
        """
        Distributions marginalized over demes.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        demes = {}
        for pop in self.dist.lineage_config.pop_names:
            demes[pop] = cls(
                state_space=self.dist.state_space,
                tree_height=self.dist.tree_height,
                demography=self.dist.demography,
                reward=CombinedReward([self.dist.reward, DemeReward(pop)])
            )

        return demes

    def get_cov(self, pop1: str, pop2: str) -> float:
        """
        Get the covariance between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The covariance.
        """
        if pop1 not in self.dist.lineage_config.pop_names or pop2 not in self.dist.lineage_config.pop_names:
            raise ValueError(f"Population {pop1} or {pop2} does not exist.")

        return self.dist.moment(
            k=2,
            rewards=(
                CombinedReward([self.dist.reward, DemeReward(pop1)]),
                CombinedReward([self.dist.reward, DemeReward(pop2)])
            ),
            center=True
        )

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix across demes.
        """
        pops = self.dist.lineage_config.pop_names

        return np.array([[self.get_cov(p1, p2) for p1 in pops] for p2 in pops])

    def get_corr(self, pop1: str, pop2: str) -> float:
        """
        Get the correlation coefficient between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The correlation coefficient.
        """
        return self.get_cov(pop1, pop2) / (self.demes[pop1].std * self.demes[pop2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across demes.
        """
        pops = self.dist.lineage_config.pop_names

        return np.array([[self.get_corr(p1, p2) for p1 in pops] for p2 in pops])


class DensityAwareDistribution(MomentAwareDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments and densities can be calculated.
    """

    @abstractmethod
    def cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Value or values to evaluate the CDF at.
        :return: CDF.
        """
        pass

    @abstractmethod
    def quantile(self, q: float) -> float:
        """
        Get the qth quantile.
        """
        pass

    @abstractmethod
    def pdf(self, t: float | Sequence[float], **kwargs) -> float | np.ndarray:
        """
        Density function.

        :param t: Value or values to evaluate the density function at.
        :param kwargs: Additional keyword arguments.
        :return: Density.
        """
        pass

    def plot_cdf(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height CDF'
    ) -> 'plt.Axes':
        """
        Plot cumulative distribution function.

        :param ax: Axes to plot on.
        :param t: Values to evaluate the CDF at. By default, 200 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from .visualization import Visualization

        if t is None:
            t = np.linspace(0, self.quantile(0.99), 200)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self.cdf(t),
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
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height PDF',
            dx: float = None
    ) -> 'plt.Axes':
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param t: Values to evaluate the density function at.
            By default, 200 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :param dx: Step size for numerical differentiation. By default, the 99th percentile divided by 1e10.
        :return: Axes.
        """
        from .visualization import Visualization

        if dx is None:
            dx = self.quantile(0.99) / 1e10

        if t is None:
            t = np.linspace(0, self.quantile(0.99), 200)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self.pdf(t, dx=dx),
            xlabel='t',
            ylabel='f(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )


class PhaseTypeDistribution(MomentAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process.
    """

    def __init__(
            self,
            state_space: StateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography = None,
            reward: Reward = None,
            regularize: bool = True
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward. By default, the tree height reward.
        :param regularize: Whether to regularize the intensity matrix for numerical stability.
        """
        if demography is None:
            demography = Demography()

        if reward is None:
            reward = TreeHeightReward()

        super().__init__()

        #: Population configuration
        self.lineage_config: LineageConfig = state_space.lineage_config

        #: Locus configuration
        self.locus_config: LocusConfig = state_space.locus_config

        #: Reward
        self.reward: Reward = reward

        #: Whether to regularize the intensity matrix for numerical stability
        self.regularize: bool = regularize

        #: State space
        self.state_space: StateSpace = state_space

        #: Demography
        self.demography: Demography = demography

        #: Tree height distribution
        self.tree_height: TreeHeightDistribution = tree_height

    @staticmethod
    def _get_van_loan_matrix(R: List[np.ndarray], S: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Get the block matrix for the given reward matrices and transition matrix.

        :param R: List of length k of reward matrices
        :param S: Intensity matrix
        :param k: The order of the moment.
        :return: Van Loan matrix which is a block matrix of size (k + 1) * (k + 1)
        """
        # matrix of zeros
        O = np.zeros_like(S)

        # create compound matrix
        return np.block([[S if i == j else R[i] if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)])

    @cached_property
    def mean(self) -> float | SFS:
        """
        First moment / mean.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> float | SFS:
        """
        Second central moment / variance.
        """
        return self.moment(k=2, center=True)

    @cached_property
    def std(self) -> float | SFS:
        """
        Standard deviation.
        """
        return self.var ** 0.5

    @cached_property
    def m2(self) -> float | SFS:
        """
        Second (non-central) moment.
        """
        return self.moment(k=2, center=False)

    @cached_property
    def demes(self) -> MarginalDemeDistributions:
        """
        Marginal distributions over each deme.
        """
        return MarginalDemeDistributions(self)

    @cached_property
    def loci(self) -> MarginalLocusDistributions:
        """
        Marginal distributions over each locus.
        """
        return MarginalLocusDistributions(self)

    @_make_hashable
    @cache
    def moment(
            self,
            k: int,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        Get the kth (non-central) (cross-)moment.

        :param k: The order of the moment.
        :param rewards: Iterable of k rewards. By default, the reward of the underlying distribution.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The kth moment
        """
        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            end_time = self.tree_height.t_max

        if start_time > 0:
            m_start, m_end = PhaseTypeDistribution.accumulate(
                self,
                k=k,
                end_times=[start_time, end_time],
                rewards=rewards,
                center=center,
                permute=permute
            )

            m = float(m_end - m_start)
        else:
            m = float(PhaseTypeDistribution.accumulate(
                self,
                k=k,
                end_times=[end_time],
                rewards=rewards,
                center=center,
                permute=permute
            )[0])

        if np.isnan(m):
            raise ValueError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return m

    def _get_regularization_factor(self, S: np.ndarray) -> float:
        """
        Get the regularization factor for the given intensity matrix. We
        multiply the intensity matrix by this factor to improve numerical
        stability when computing the matrix exponential of the Van Loan matrix.
        If regularization is disabled, this factor is 1.

        :param S: Intensity matrix.
        :return: Regularization factor.
        """
        if not self.regularize:
            return 1.0

        # obtain positive rates
        rates = S[S > 0]

        # rewards in the Van Loan matrix are of order 1
        return 10 ** - np.log10(rates).mean()

    def _check_numerical_stability(self, S: np.ndarray, epoch: int):
        """
        Warn about potential numerical instability with very small or very large rates.

        :param S: (Regularized) intensity matrix.
        :param epoch: Epoch number.
        """
        rates = S[S > 0]

        if rates.min() / rates.max() < 1e-10:
            self._logger.warning(
                f"Intensity matrix in epoch {epoch} contains rates that differ by more than 10 orders of magnitude: "
                f"min: {rates.min()}, max: {rates.max()}. "
                f"This may potentially lead to numerical instability, despite matrix regularization."
            )

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moment at different end times.

        :param k: The order of the moment.
        :param end_times: List of ends times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The moment accumulated at the specified times or time.
        """
        k = int(k)

        if rewards is None:
            rewards = [self.reward] * k

        if k != len(rewards):
            raise ValueError(f"Number of specified rewards for moment of order {k} must be {k}.")

        if k == 0:
            return np.ones_like(list(end_times))

        # center moments around the mean
        if center and k > 1:

            components = []

            # first order moments
            means = [
                PhaseTypeDistribution.accumulate(
                    self,
                    k=1,
                    rewards=(rewards[i],),
                    end_times=end_times
                ) for i in range(k)
            ]

            for i in range(k + 1):
                # iterate over all possible subsets of rewards of size i
                for indices in itertools.combinations(range(k), i):
                    # joint moment
                    mu_i = PhaseTypeDistribution.accumulate(
                        self,
                        k=i,
                        rewards=tuple(rewards[j] for j in indices),
                        end_times=end_times,
                        center=False,
                        permute=permute
                    )

                    # product of means of remaining rewards
                    mu1 = np.prod([means[j] for j in range(k) if j not in indices], axis=0)

                    components += [(-1) ** (k - i) * mu_i * mu1]

            return np.sum(components, axis=0)

        if permute:
            # get all possible permutations of rewards
            permutations = list(itertools.permutations(rewards))

            # compute average over all permutations
            return np.sum([self._accumulate(k, tuple(end_times), r) for r in permutations], axis=0) / len(permutations)

        return self._accumulate(k, tuple(end_times), rewards)

    @_make_hashable
    @cache
    def _accumulate(
            self,
            k: int,
            end_times: Sequence[float],
            rewards: Sequence[Reward] = None
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) conditioned moment at different end times.

        :param k: The order of the moment.
        :param end_times: Sequence of ends times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :return: The moment accumulated at the specified times or time.
        """
        end_times = np.array(end_times)

        # check for negative values
        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

        # use default reward if not specified
        if rewards is None:
            rewards = (self.reward,) * k
        else:
            if len(rewards) != k:
                raise ValueError(f"Number of rewards must be {k}.")

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(end_times)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get state space for the first epoch
        self.state_space.update_epoch(epoch)

        # number of states
        n_states = self.state_space.k

        # initialize block matrix holding (rewarded) moments
        Q = np.eye(n_states * (k + 1))
        u_prev = 0

        # initialize probabilities
        moments = np.zeros_like(t_sorted, dtype=float)

        # regularization parameter
        lamb = self._get_regularization_factor(self.state_space.S)

        # regularized intensity matrix
        S = self.state_space.S * lamb

        # check numerical stability
        self._check_numerical_stability(S, 0)

        # get reward matrix
        R = [np.diag(r._get(state_space=self.state_space)) for r in rewards]

        # get Van Loan matrix
        V = self._get_van_loan_matrix(S=S, R=R, k=k)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                # update transition matrix with remaining time in current epoch
                Q @= expm(V * (epoch.end_time - u_prev) / lamb)

                # fetch and update for next epoch
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

                # compute Van Loan matrix for next epoch using regularized intensity matrix
                S = self.state_space.S * lamb
                self._check_numerical_stability(S, 0)
                V = self._get_van_loan_matrix(S=S, R=R, k=k)

            # update with remaining time in current epoch
            Q @= expm(V * (u - u_prev) / lamb)

            alpha = self.state_space.alpha
            e = self.state_space.e
            moments[i] = factorial(k) * lamb ** k * alpha @ Q[:n_states, -n_states:] @ e

            u_prev = u

        # sort probabilities back to original order
        moments = moments[np.argsort(end_times)]

        if np.isnan(moments).any():
            self._logger.warning(
                "NaN values encountered when computing moments. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot accumulation of (non-central) moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and the
            99th percentile.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        k = int(k)

        from .visualization import Visualization

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = f"Moment accumulation ({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})"

        y = self.accumulate(k, end_times, rewards, center, permute)

        Visualization.plot(
            ax=ax,
            x=end_times,
            y=y,
            xlabel='t',
            ylabel='moment',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )


class TreeHeightDistribution(PhaseTypeDistribution, DensityAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process that allows the computation of the
    density function. This is currently only possible with default rewards.
    """
    #: Maximum number of epochs to consider when determining time to almost sure absorption.
    max_epochs: int = 10000

    #: Maximum number of time we double the end time when determining time to almost sure absorption.
    max_iter: int = 20

    #: Probability of almost sure absorption.
    p_absorption: float = 1 - 1e-15

    def __init__(
            self,
            state_space: LineageCountingStateSpace,
            demography: Demography = None,
            start_time: float = 0,
            end_time: float = None,
            regularize: bool = True
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param demography: The demography.
        :param start_time: Time when to start accumulating moments.
        :param end_time: Time when to end accumulation of moments. By default, the time until almost sure absorption.
        :param regularize: Whether to regularize the intensity matrix for numerical stability.
        """
        if start_time < 0:
            raise ValueError("Start time must be greater than or equal to 0.")

        if end_time is not None and end_time < 0:
            raise ValueError("End time must be greater than or equal to 0.")

        if end_time is not None and end_time < start_time:
            raise ValueError("End time must be greater than equal start time.")

        super().__init__(
            state_space=state_space,
            tree_height=self,
            demography=demography,
            reward=TreeHeightReward(),
            regularize=regularize
        )

        #: State space
        self.state_space: LineageCountingStateSpace = state_space

        #: Start time
        self.start_time: float = start_time

        #: End time
        self.end_time: float | None = end_time

    def cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Value or values to evaluate the CDF at.
        :return: Cumulative probability
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not isinstance(self.reward, TreeHeightReward):
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        # assume scalar if not array
        if not isinstance(t, Iterable):
            return self.cdf(np.array([t]))[0]

        # check for negative values
        if np.any(t < 0):
            raise ValueError("Negative values are not allowed.")

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(t).astype(float)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get the transition matrix for the first epoch
        self.state_space.update_epoch(epoch)

        # initialize transition matrix
        T = np.eye(self.state_space.k)
        u_prev = 0

        # initialize probabilities
        probs = np.zeros_like(t_sorted)

        # take reward vector as exit vector
        e = self.reward._get(self.state_space)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                self._check_numerical_stability(self.state_space.S, i_epoch)

                # update transition matrix with remaining time in current epoch
                T @= expm(self.state_space.S * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

            self._check_numerical_stability(self.state_space.S, i_epoch)

            # update transition matrix with remaining time in current epoch
            T @= expm(self.state_space.S * (u - u_prev))

            probs[i] = 1 - self.state_space.alpha @ T @ e

            u_prev = u

        # sort probabilities back to original order
        probs = probs[np.argsort(t)]

        if np.isnan(probs).any():
            self._logger.critical(
                "NaN values in CDF. This is likely due to an ill-conditioned rate matrix."
            )

        return probs

    def _update(
            self,
            u: float,
            u_prev: float,
            T: np.ndarray,
            epoch: 'Epoch'
    ) -> Tuple[float, np.ndarray, 'Epoch']:
        """
        Update transition matrix and time.

        :param u: Time to update to.
        :param u_prev: Previous time.
        :param T: Transition matrix.
        :param epoch: Current epoch.
        :return: Updated time, transition matrix, and epoch.
        """
        self.state_space.update_epoch(epoch)

        while u > epoch.end_time:

            # update transition matrix with remaining time in current epoch
            tau = epoch.end_time - u_prev
            T = T @ expm(self.state_space.S * tau)
            u_prev = epoch.end_time

            # fetch and update for next epoch
            epoch = self.demography.get_epoch(epoch.end_time)
            self.state_space.update_epoch(epoch)
        else:
            # update transition matrix
            T = T @ expm(self.state_space.S * (u - u_prev))

        return u, T, epoch

    @cached_property
    def _e(self) -> np.ndarray:
        """
        Exit vector.
        """
        return self.reward._get(self.state_space)

    def _cum(self, T: np.ndarray) -> float:
        """
        Get cumulative probability for given transition matrix.

        :param T: Transition matrix.
        :return: Cumulative probability.
        """
        return float(1 - self.state_space.alpha @ T @ self._e)

    @cache
    def quantile(
            self,
            q: float,
            expansion_factor: float = 2,
            precision: float = 1e-5,
            max_iter: int = 1000
    ):
        """
        Find the specified quantile of a CDF using an adaptive bisection method.

        :param q: The desired quantile (between 0 and 1).
        :param expansion_factor: Factor by which to multiply the upper bound that does not yet contain the quantile.
        :param precision: Precision for quantile, i.e. ``F(b) - F(a) < precision``.
        :param max_iter: Maximum number of iterations.
        :return: The quantile.
        """
        if q < 0 or q > 1:
            raise ValueError("Specified quantile must be between 0 and 1.")

        if expansion_factor <= 1:
            raise ValueError("Expansion factor must be greater than 1.")

        # initialize bounds
        a, b = 0, 1

        T_a = np.eye(self.state_space.k)
        epoch_a, epoch_b = self.demography.get_epoch(0), self.demography.get_epoch(0)
        b, T_b, epoch_b = self._update(b, a, T_a, epoch_b)

        i = 0

        # expand lower bound until it contains the quantile
        while self._cum(T_b) < q and i < max_iter:
            b, T_b, epoch_b = self._update(b * expansion_factor, b, T_b, epoch_b)

            i += 1

        # use bisection method within the determined bounds
        while self._cum(T_b) - self._cum(T_a) > precision and i < max_iter:
            m, T_m, epoch_m = self._update((a + b) / 2, a, T_a, epoch_a)

            if self._cum(T_m) < q:
                a, T_a, epoch_a = m, T_m, epoch_m
            else:
                b, T_b, epoch_b = m, T_m, epoch_m

            i += 1

        # warn if maximum number of iterations reached
        if i - 1 == max_iter:
            raise RuntimeError("Maximum number of iterations reached when determining quantile.")

        return (a + b) / 2

    def pdf(self, t: float | Sequence[float], dx: float = None) -> float | np.ndarray:
        """
        Density function. We use numerical differentiation of the CDF to calculate the density. This provides good
        results as the CDF is exact and continuous.

        :param t: Value or values to evaluate the density function at.
        :param dx: Step size for numerical differentiation. By default, the 99th percentile divided by 1e10.
        :return: Density
        """
        if dx is None:
            dx = self.quantile(0.99) / 1e10

        if isinstance(t, Iterable):
            t = np.array(t)

        # determine (non-negative) evaluation points
        x1 = np.max([t - dx / 2, np.zeros_like(t)], axis=0)
        x2 = x1 + dx

        return (self.cdf(x2) - self.cdf(x1)) / dx

    @cached_property
    def t_max(self) -> float:
        """
        Time until which computations are performed. This is either the end time specified when initializing
        the distribution or the time until almost sure absorption.
        """
        if self.end_time is not None:
            return self.end_time

        t_abs = self._get_absorption_time()

        if t_abs < self.start_time:
            raise ValueError(
                f"Determined time of almost sure absorption ({t_abs:.1f}) "
                f"is smaller than start time ({self.start_time:.1f}). "
                "The start time may be too large or the demography not well-defined."
            )

        return t_abs

    def _get_absorption_time(self) -> float:
        """
        Get a time estimate for when we have reached absorption almost surely.
        We base this computation on the transition matrix rather than the moments, because here
        we have a good idea about how likely absorption, and can warn the user if necessary.
        Stopping the computation when no more rewards are accumulated is not a good idea, as this
        can happen before almost sure absorption (exponential runaway growth, temporary isolation in different demes).
        """
        i = 0
        T = np.eye(self.state_space.k)
        epoch = self.demography.get_epoch(0)
        t = 2 ** int(np.log2(np.mean(list(epoch.pop_sizes.values()))))
        expansion_factor = 2

        t, T, epoch = self._update(t, 0, T, epoch)
        p = self._cum(T)

        # multiple time by expansion_factor until we reach p_absorption
        while p < self.p_absorption and i < self.max_iter:
            t, T, epoch = self._update(t * expansion_factor, t, T, epoch)
            p = self._cum(T)

            if np.isnan(p):
                self._logger.critical(
                    "Could not reliably find time of almost sure absorption "
                    "as probability of absorption is NaN. "
                    "This is likely due to an ill-conditioned rate matrix. "
                    f"Using time {t:.1f}. "
                )

            i += 1

        if i - 1 == self.max_iter:
            self._logger.warning(
                "Could not reliably find time of almost sure absorption after maximum number of iterations. "
                f"Using time {t:.1f} with probability of absorption 1 - {1 - p:.1e}. "
                "This could be due to numerical imprecision, unreachable states or very large or small "
                "absorption times. You can set the end time manually (see `Coalescent.end_time`) or increase "
                "the maximum number of iterations (`TreeHeightDistribution.max_iter`)."
            )

        return t


class SFSDistribution(PhaseTypeDistribution, ABC):
    """
    Base class for site-frequency spectrum distributions.
    """

    def __init__(
            self,
            state_space: BlockCountingStateSpace,
            tree_height: TreeHeightDistribution,
            demography: Demography,
            pbar: bool = False,
            parallelize: bool = False,
            reward: Reward = None,
            regularize: bool = True
    ):
        """
        Initialize the distribution.

        :param state_space: Block-counting state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param pbar: Whether to show a progress bar.
        :param parallelize: Use parallelization.
        :param reward: The reward to multiply the SFS reward with. By default, the unit reward is used, which
            has no effect.
        :param regularize: Whether to regularize the intensity matrix for numerical stability.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward,
            regularize=regularize
        )

        #: Whether to show a progress bar.
        self.pbar: bool = pbar

        #: Whether to parallelize computations.
        self.parallelize: bool = parallelize

        #: Generated probability mass by iterator returned from :meth:`get_mutation_configs`.
        self.generated_mass = 0

    @abstractmethod
    def _get_sfs_reward(self, i: int) -> SFSReward:
        """
        Get the reward for the ith site-frequency count.

        :param i: The ith site-frequency count.
        :return: The reward.
        """
        pass

    @abstractmethod
    def _get_indices(self) -> np.ndarray:
        """
        Get the indices for the site-frequency spectrum.

        :return: The indices.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_configs(n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Get all possible mutational configurations for a given number of mutations.

        :param n: The number of lineages.
        :param k: The number of mutations.
        :return: An iterator over all possible mutational configurations.
        """
        pass

    @_make_hashable
    @cache
    def moment(
            self,
            k: int,
            rewards: Sequence[SFSReward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> SFS:
        """
        Get the kth moments of the site-frequency spectrum.

        :param k: The order of the moment
        :param rewards: Sequence of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: A site-frequency spectrum of kth order moments.
        """
        if rewards is None:
            rewards = (self.reward,) * k

        # optionally parallelize the moment computation of the SFS bins
        moments = parallelize(
            func=lambda x: self._moment(*x),
            data=[[k, i, rewards, start_time, end_time, center, permute] for i in self._get_indices()],
            desc=f"Calculating {k}-moments",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        return SFS([0] + list(moments) + [0] * (self.lineage_config.n - len(moments)))

    def _moment(
            self,
            k: int,
            i: int,
            rewards: Sequence[SFSReward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        Get the kth moment for the ith site-frequency count.

        :param k: The order of the moment
        :param i: The ith site-frequency count
        :param rewards: Sequence of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The kth SFS (cross)-moment at the ith site-frequency count
        """
        return PhaseTypeDistribution.moment(
            self,
            k=k,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards]),
            start_time=start_time,
            end_time=end_time,
            center=center,
            permute=permute
        )

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moments for site-frequency spectrum at different end times.

        :param k: The order of the moment.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: Array of moments accumulated at the specified times, one for each site-frequency count.
        """
        k = int(k)
        indices = self._get_indices()
        end_times = np.array(list(end_times))

        accumulation = parallelize(
            func=lambda x: self.get_accumulation(*x),
            data=[[k, i, end_times, rewards] for i in indices],
            desc=f"Calculating accumulation of {k}-moments",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        # pad with zeros
        return np.concatenate([
            np.zeros((1, len(end_times))),
            accumulation,
            np.zeros((self.lineage_config.n - len(indices), len(end_times)))
        ])

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot accumulation of (non-central) SFS moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and
            the 99th percentile.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from .visualization import Visualization

        k = int(k)

        if ax is None:
            ax = plt.gca()

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = (f"SFS Moment accumulation "
                     f"({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})")

        # get accumulation of moments
        accumulation = self.accumulate(k, end_times, rewards, center, permute)

        for i, acc in zip(self._get_indices(), accumulation[1: -1]):
            Visualization.plot(
                ax=ax,
                x=end_times,
                y=acc,
                xlabel='t',
                ylabel='moment',
                label=f'{i}',
                file=file,
                show=i == self._get_indices()[-1] and show,
                clear=clear,
                title=title
            )

        return ax

    def get_accumulation(
            self,
            k: int,
            i: int,
            end_times: Iterable[float] | float,
            rewards: Sequence[SFSReward] = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray | float:
        """
        Get accumulation of moments for the ith site-frequency count.

        :param k: The order of the moment
        :param i: The ith site-frequency count.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Sequence of k rewards.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The kth SFS (cross)-moment accumulations at the ith site-frequency count
        """
        if rewards is None:
            rewards = [self.reward] * k

        return super().accumulate(
            k=k,
            end_times=end_times,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards]),
            center=center,
            permute=permute
        )

    @cached_property
    def cov(self) -> SFS2:
        """
        Covariance matrix across site-frequency counts.
        """
        # create list of arguments for each combination of i, j
        indices = [(i, j) for i in self._get_indices() for j in self._get_indices()]

        # get sfs using parallelized function
        sfs_results = parallelize(
            func=lambda x: (
                PhaseTypeDistribution.moment(self, k=2, permute=False, center=False, rewards=(
                    CombinedReward([self.reward, self._get_sfs_reward(x[0])]),
                    CombinedReward([self.reward, self._get_sfs_reward(x[1])])
                ))
            ),
            data=indices,
            desc="Calculating covariance",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        # re-structure the results to a matrix form
        sfs = np.zeros((self.lineage_config.n + 1, self.lineage_config.n + 1))
        for ((i, j), result) in zip(indices, sfs_results):
            sfs[i, j] = result

        # get matrix of marginal moments
        m2 = np.outer(self.mean.data, self.mean.data)

        # calculate covariances
        cov = (sfs + sfs.T) / 2 - m2

        return SFS2(cov)

    def get_cov(self, i: int, j: int) -> float:
        """
        Get the covariance between the ith and jth site-frequency.

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: covariance
        """
        if i in (0, self.lineage_config.n) or j in (0, self.lineage_config.n):
            return 0

        return super().moment(
            k=2,
            rewards=(
                CombinedReward([self.reward, self._get_sfs_reward(i)]),
                CombinedReward([self.reward, self._get_sfs_reward(j)])
            ),
            center=True
        )

    @cached_property
    def corr(self) -> SFS2:
        """
        Correlation matrix across site-frequency counts.
        """
        # get standard deviations
        std = np.sqrt(self.var.data)

        sfs = SFS2(self.cov.data / np.outer(std, std))

        # replace NaNs with zeros
        sfs.data[np.isnan(sfs.data)] = 0

        return sfs

    def get_corr(self, i: int, j: int) -> float:
        """
        Get the correlation coefficient between the ith and jth site-frequency.

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: Correlation coefficient
        """
        if i in (0, self.lineage_config.n) or j in (0, self.lineage_config.n):
            return 0

        return self.get_cov(i, j) / (np.sqrt(self.get_cov(i, i)) * np.sqrt(self.get_cov(j, j)))

    @cache
    def _get_P(self, n: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get transition matrix for mutational configuration probabilities.

        :param n: The number of frequency bins.
        :param theta: The mutation rate.
        :return: Transition matrix and exit vector.
        """
        # get non-absorbing states
        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)

        e = self.state_space.e[non_absorbing]
        R = np.array([self._get_sfs_reward(i)._get(self.state_space) for i in range(1, n + 1)])[:, non_absorbing]
        r_total = R.T @ np.ones(n)

        S = self.state_space.S[non_absorbing, :][:, non_absorbing]
        I = np.eye(S.shape[0])

        P_total = np.linalg.inv(I - np.diag(1 / r_total) / theta @ S)
        p_total = (I - P_total) @ e
        P = np.array([P_total @ np.diag(R[i] / r_total) for i in range(n)])

        return P, p_total

    def get_mutation_config(self, config: Sequence[int], theta: float) -> float:
        """
        Get the probabilities of observing the given mutational configurations according to the infinite sites model.

        .. note::
            This currently only works for a single epoch, i.e. a time-homogeneous demography, and recombination is not
            supported.

        :param config: The mutational configuration. A sequence of integers of length n - 1 for unfolded configurations
            and n // 2 for folded configurations, where n is the number of
            lineages. Each element in the sequence is an integer representing the number of mutations
            at each frequency count starting from 1. For example, the unfolded configuration [2, 1, 0] represents two
            singleton, one doubleton and zero tripleton mutations for a sample size of 4 lineages. Similarly, the
            folded configuration [2, 1] represents two singleton or tripleton and one doubleton mutation for the same
            number of lineages.
        :param theta: The mutation rate.
        :return: The probability of observing the given mutational configuration.
        """
        # raise not implemented error if more than one epoch
        if self.demography.has_n_epochs(2):
            raise NotImplementedError("Sampling not implemented for more than one epoch.")

        # make sure theta is non-negative
        if theta < 0:
            raise ValueError("Theta must be greater than or equal to 0.")

        # number of frequency bins
        n = len(self._get_configs(self.lineage_config.n, 0)[0])

        if len(config) != n:
            raise ValueError(
                "The length of the configuration must be equal to the number of frequency bins. "
                f"Expected {n}, got {len(config)}."
            )

        # explicitly convert to tuple of integers
        config = tuple(int(c) for c in config)

        # handle special case when theta = 0
        if theta == 0:
            if sum(config) == 0:
                return 1

            return 0

        # get non-absorbing states
        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)

        # number of non-absorbing states
        k = non_absorbing.sum()

        alpha = self.state_space.alpha[non_absorbing]

        P, p_total = self._get_P(n, theta)

        q = list(itertools.chain(*[[i + 1] * j for i, j in enumerate(config)]))

        # iterate over permutations of q
        Q = np.zeros((k, k))
        for p in multiset_permutations(q):
            U = np.eye(k)

            for i in p:
                U @= P[i - 1]

            Q += U

        p = alpha @ Q @ p_total

        return p

    def get_mutation_configs(self, theta: float) -> Iterator[Tuple[Tuple[float, ...], float]]:
        """
        An iterator over the probabilities of observing mutational configurations according to the infinite sites model.
        The order of the mutational configurations generated ascends in the number of mutations observed.
        See :meth:`get_mutation_config` for more information on mutational configurations.

        .. note::
            This currently only works for a single epoch, i.e. a time-homogeneous demography, and recombination is not
            supported. Also note that the number of configurations is infinite, so this iterator will never stop.
            However, depending on the mutation rate, the probability of observing configurations of higher mutation
            counts will decrease over time. You can keep track of the generated probability mass by checking the
            :attr:`~.generated_mass` attribute, which is reset every time this method is called.
            A good approach is thus to keep generating configurations until the generated mass is above a certain
            threshold. More complex demographic models and larger sample sizes increase the number of configurations
            and higher mutation rates, the number of generated configurations necessary to reach a certain mass.

        Code example:

        ::

            coal = pg.Coalescent(n=5)

            it = coal.sfs.get_mutation_configs(theta=1)

            # continue until generated mass is above 0.8
            samples = list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))

        :param theta: The mutation rate.
        :return: An iterator over the probabilities of observing mutational configurations.
        """
        # reset generated mass
        self.generated_mass = 0

        # iterate over number of mutations
        i = 0
        while True:
            # iterate over configurations
            for config in self._get_configs(self.lineage_config.n, i):
                p = self.get_mutation_config(config=config, theta=theta)
                self.generated_mass += p
                yield config, p

            # increase counter for number of mutations
            i += 1


class UnfoldedSFSDistribution(SFSDistribution):
    """
    Unfolded site-frequency spectrum distribution.
    """

    def _get_sfs_reward(self, i: int) -> UnfoldedSFSReward:
        """
        Get the reward for the ith site-frequency count.

        :param i: The ith site-frequency count.
        :return: The reward.
        """
        return UnfoldedSFSReward(i)

    def _get_indices(self) -> np.ndarray:
        """
        Get the indices for the site-frequency spectrum.

        :return: The indices.
        """
        return np.arange(1, self.lineage_config.n)

    @staticmethod
    def _get_configs(n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Get all possible mutational configurations for a given number of mutations.

        :param n: The number of lineages.
        :param k: The number of mutations.
        :return: An iterator over all possible mutational configurations.
        """
        return StateSpace._get_partitions(n=k, k=n - 1)


class FoldedSFSDistribution(SFSDistribution):
    """
    Folded site-frequency spectrum distribution.
    """

    def _get_sfs_reward(self, i: int) -> FoldedSFSReward:
        """
        Get the reward for the ith site-frequency count.

        :param i: The ith site-frequency count.
        :return: The reward.
        """
        return FoldedSFSReward(i)

    def _get_indices(self) -> np.ndarray:
        """
        Get the indices for the site-frequency spectrum.

        :return: The indices.
        """
        return np.arange(1, self.lineage_config.n // 2 + 1)

    @staticmethod
    def _get_configs(n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Get all possible mutational configurations for a given number of mutations.

        :param n: The number of lineages.
        :param k: The number of mutations.
        :return: An iterator over all possible mutational configurations.
        """
        return StateSpace._get_partitions(n=k, k=n // 2)

    def _unfold(self, config: Sequence[int]) -> Set[Tuple[int, ...]]:
        """
        Unfold a folded configuration into all possible unfolded configurations.

        :param config: The folded configuration. A sequence of integers of length n // 2 where n is the number of
            lineages.
        :return: The unfolded configurations.
        """
        n = self.lineage_config.n

        if n // 2 != len(config):
            raise ValueError("The length of the configuration must equal n // 2 where n is the number of lineages.")

        if n % 2 == 1:
            lower_counts = [range(i + 1) for i in config]
            i_center = len(config)
        else:
            lower_counts = [range(i + 1) for i in config[:-1]] + [[config[-1]]]
            i_center = len(config) - 1

        unfolded = []
        # iterate over unfolded configurations
        for lower in itertools.product(*lower_counts):
            # get higher counts
            higher = (np.array(config) - np.array(lower))[:i_center][::-1]

            unfolded += [list(lower) + list(higher)]

        return set(tuple(u) for u in unfolded)


class EmpiricalDistribution(DensityAwareDistribution):  # pragma: no cover
    """
    Probability distribution based on realisations.
    """

    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples: 1-D array of samples.
        """
        super().__init__()

        self._cache = None

        #: Samples
        self.samples = np.array(samples, dtype=float)

    def touch(self, t: np.ndarray):
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super().touch()

        self._cache = dict(
            t=t,
            cdf=self.cdf(t),
            pdf=self.pdf(t)
        )

    def drop(self):
        """
        Drop simulated samples.
        """
        self.samples = None

    @cached_property
    def mean(self) -> float | np.ndarray:
        """
        First moment / mean.
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self) -> float | np.ndarray:
        """
        Second central moment / variance.
        """
        return np.var(self.samples, axis=0)

    @cached_property
    def m2(self) -> float | np.ndarray:
        """
        Second non-central moment.
        """
        return np.mean(self.samples ** 2, axis=0)

    @cached_property
    def m3(self) -> float | np.ndarray:
        """
        Third non-central moment.
        """
        return np.mean(self.samples ** 3, axis=0)

    @cached_property
    def m4(self) -> float | np.ndarray:
        """
        Fourth non-central moment.
        """
        return np.mean(self.samples ** 4, axis=0)

    @cached_property
    def cov(self) -> float | np.ndarray:
        """
        Covariance matrix.
        """
        return np.nan_to_num(np.cov(self.samples, rowvar=False))

    @cached_property
    def corr(self) -> float | np.ndarray:
        """
        Correlation matrix.
        """
        return np.nan_to_num(np.corrcoef(self.samples, rowvar=False))

    def moment(self, k: int) -> float | np.ndarray:
        """
        Get the kth moment.

        :param k: The order of the moment
        :return: The kth moment
        """
        return np.mean(self.samples ** k, axis=0)

    def cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Time.
        :return: Cumulative probability.
        """
        x = np.sort(self.samples)
        y = np.arange(1, len(self.samples) + 1) / len(self.samples)

        if x.ndim == 1:
            return np.interp(t, x, y)

        if x.ndim == 2:
            return np.array([np.interp(t, x_, y) for x_ in x.T])

        raise ValueError("Samples must be 1 or 2 dimensional.")

    def quantile(self, q: float) -> float:
        """
        Get the qth quantile.

        :param q: Quantile.
        :return: Quantile.
        """
        return np.quantile(self.samples, q=q)

    def pdf(
            self,
            t: float | np.ndarray,
            n_bins: int = 10000,
            sigma: float = None,
            samples: np.ndarray = None
    ) -> float | np.ndarray:
        """
        Density function.

        :param sigma: Sigma for Gaussian filter.
        :param n_bins: Number of bins.
        :param t: Time.
        :param samples: Samples.
        :return: Density.
        """
        samples = self.samples if samples is None else samples

        if samples.ndim == 2:
            return np.array([self.pdf(t, n_bins=n_bins, sigma=sigma, samples=s) for s in samples.T])

        hist, bin_edges = np.histogram(samples, range=(0, max(samples)), bins=n_bins, density=True)

        # determine bins for u
        bins = np.minimum(np.sum(bin_edges <= t[:, None], axis=1) - 1, np.full_like(t, n_bins - 1, dtype=int))

        # use proper bins for y values
        y = hist[bins]

        # smooth using gaussian filter
        if sigma is not None:
            y = gaussian_filter1d(y, sigma=sigma)

        return y


class EmpiricalSFSDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    SFS probability distribution based on realisations.
    """

    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples: 2-D array of samples.
        """
        super().__init__(samples)

    @cached_property
    def mean(self) -> SFS:
        """
        First moment / mean.
        """
        return SFS(super().mean)

    @cached_property
    def var(self) -> SFS:
        """
        Second central moment / variance.
        """
        return SFS(super().var)

    @cached_property
    def m2(self) -> SFS:
        """
        Second non-central moment.
        """
        return SFS(super().m2)

    @cached_property
    def cov(self) -> SFS2:
        """
        Covariance matrix.
        """
        return SFS2(np.nan_to_num(np.cov(self.samples, rowvar=False)))

    @cached_property
    def corr(self) -> SFS2:
        """
        Correlation matrix.
        """
        return SFS2(np.nan_to_num(np.corrcoef(self.samples, rowvar=False)))


class DictContainer(dict):  # pragma: no cover
    """
    Dictionary container.
    """
    pass


class EmpiricalPhaseTypeDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    Phase-type distribution based on realisations.
    """

    def __init__(
            self,
            samples: np.ndarray | list,
            pops: List[str],
            locus_agg: Callable = lambda x: x.sum(axis=0)
    ):
        """
        Create object.

        :param samples: 3-D array of samples.
        :param pops: List of population names.
        :param locus_agg: Aggregation function for loci.
        """
        over_loci = locus_agg(samples).astype(float)
        over_demes = samples.sum(axis=1).astype(float)

        super().__init__(over_loci.sum(axis=0))

        #: Population names
        self.pops = pops

        #: Samples by deme and locus
        self._samples = samples

        #: Covariance matrix for the demes
        self.pops_cov: np.ndarray = np.cov(over_loci)

        #: Correlation matrix for the demes
        self.pops_corr: np.ndarray = np.corrcoef(over_loci)

        #: Covariance matrix for the loci
        self.loci_corr: np.ndarray = np.corrcoef(over_demes)

        #: Correlation matrix for the loci
        self.loci_cov: np.ndarray = np.cov(over_demes)

    def touch(self, t: np.ndarray):
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super().touch(t)

        [d.touch(t) for d in self.demes.values()]
        [l.touch(t) for l in self.loci.values()]

    def drop(self):
        """
        Drop simulated samples.
        """
        super().drop()

        self._samples = None

        [d.drop() for d in self.demes.values()]
        [l.drop() for l in self.loci.values()]

    @cached_property
    def demes(self) -> Dict[str, EmpiricalDistribution]:
        """
        Get the distribution for each deme.

        :return: Dictionary of distributions.
        """
        demes = DictContainer(
            {pop: EmpiricalDistribution(self._samples.sum(axis=0)[i]) for i, pop in enumerate(self.pops)}
        )

        # TODO this is the covariance in the tree height but phasegen
        #  provides the covariance in the number of lineages per deme
        demes.cov = self.pops_cov
        demes.corr = self.pops_corr

        return demes

    @cached_property
    def loci(self) -> Dict[int, EmpiricalDistribution]:
        """
        Get the distribution for each locus.

        :return: Dictionary of distributions.
        """
        loci = DictContainer(
            {i: EmpiricalDistribution(self._samples[i].sum(axis=0)) for i in range(self._samples.shape[0])}
        )

        loci.cov = self.loci_cov
        loci.corr = self.loci_corr

        return loci


class EmpiricalPhaseTypeSFSDistribution(EmpiricalPhaseTypeDistribution):  # pragma: no cover
    """
    SFS phase-type distribution based on realisations.
    """

    def __init__(
            self,
            branch_lengths: np.ndarray,
            mutations: np.ndarray,
            pops: List[str],
            sfs_dist: Type[SFSDistribution],
            locus_agg: Callable = lambda x: x.sum(axis=0),
    ):
        """
        Create object.

        :param branch_lengths: 4-D array of branch length samples.
        :param mutations: 4-D array of mutation counts.
        :param pops: List of population names.
        :param sfs_dist: SFS distribution class.
        :param locus_agg: Aggregation function for loci.
        """
        over_loci = locus_agg(branch_lengths).astype(float)

        EmpiricalDistribution.__init__(self, over_loci.sum(axis=0))

        #: Population names
        self.pops = pops

        # : Number of lineages
        self.n = branch_lengths.shape[-1] - 1

        #: SFS distribution class
        self._sfs_dist = sfs_dist

        #: Branch length samples by deme and locus
        self._samples = branch_lengths

        #: Mutation counts by deme and locus
        self._mutations = mutations

        #: Correlation matrix for the loci
        self.pops_corr = self._get_stat_pops(over_loci, np.corrcoef)

        #: Covariance matrix for the demes
        self.pops_cov: np.ndarray = self._get_stat_pops(over_loci, np.cov)

        #: Correlation matrix for the loci
        self.loci_corr: np.ndarray = None

        #: Covariance matrix for the loci
        self.loci_cov: np.ndarray = None

        #: Generated probability mass by iterator returned from :meth:`get_mutation_configs`.
        self.generated_mass = 0

    def drop(self):
        """
        Drop simulated samples.
        """
        super().drop()

        self._mutations = None

    @staticmethod
    def _get_stat_pops(samples: np.ndarray, callback: Callable) -> np.ndarray:
        """
        Get the covariance matrix for the demes.

        :param callback: Callback function to apply to the samples.
        :return: Covariance matrix.
        """
        stats = np.zeros((samples.shape[0], samples.shape[0], samples.shape[2], samples.shape[2]))

        for i, j in itertools.product(range(1, samples.shape[2] - 1), range(1, samples.shape[2] - 1)):
            stats[:, :, i, j] = callback(samples[:, :, i])

        return stats

    @cached_property
    def demes(self) -> Dict[str, EmpiricalDistribution]:
        """
        Get the distribution for each deme.

        :return: Dictionary of distributions.
        """
        return {pop: EmpiricalSFSDistribution(self._samples.sum(axis=0)[i]) for i, pop in enumerate(self.pops)}

    @cached_property
    def mutation_configs(self) -> Dict[Tuple[float, ...], float]:
        """
        Get a dictionary of all mutation configurations and their probabilities.

        :return: Dictionary of distributions.
        """
        configs = defaultdict(lambda: 0)

        for config in self._mutations[0, 0]:
            configs[tuple(config)] += 1 / self._mutations.shape[2]

        return configs

    def get_mutation_config(self, config: Sequence[int]) -> float:
        """
        Get the probability of observing the given mutational configuration.

        :param config: The mutational configuration.
        :return: The probability of observing the given mutational configuration.
        """
        return self.mutation_configs[tuple(config)]

    def get_mutation_configs(self) -> Iterator[Tuple[Tuple[float, ...], float]]:
        """
        An iterator over the probabilities of observing mutational configurations according to the infinite sites model.
        The order of the mutational configurations generated ascends in the number of mutations observed.

        :return: An iterator over the probabilities of observing mutational configurations.
        """
        # reset generated mass
        self.generated_mass = 0

        # iterate over number of mutations
        i = 0
        while True:
            # iterate over configurations
            for config in self._sfs_dist._get_configs(self.n, i):
                p = self.get_mutation_config(config=config)
                self.generated_mass += p
                yield config, p

            # increase counter for number of mutations
            i += 1


class AbstractCoalescent(ABC):
    """
    Abstract base class for coalescent distributions. This class provides probability distributions for the
    tree height, total branch length and site frequency spectrum.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | LineageConfig,
            model: CoalescentModel = None,
            demography: Demography = None,
            loci: int | LocusConfig = 1,
            recombination_rate: float = None,
            end_time: float = None
    ):
        """
        Create object.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`~phasegen.lineage.LineageConfig` object can be passed.
        :param model: Coalescent model. By default, the standard coalescent is used.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate.
        :param demography: Demography.
        :param end_time: Time when to end the computation. If ``None``, the end time is end time is taken to be the
            time of almost sure absorption. Note that unnecessarily large end times can lead to numerical errors.
        """
        self._logger = logger.getChild(self.__class__.__name__)

        if model is None:
            model = StandardCoalescent()

        if not isinstance(n, LineageConfig):
            #: Population configuration
            self.lineage_config: LineageConfig = LineageConfig(n)
        else:
            #: Population configuration
            self.lineage_config: LineageConfig = n

        if demography is None:
            demography = Demography(pop_sizes={p: 1 for p in self.lineage_config.pop_names})

        # set up locus configuration
        if isinstance(loci, int):
            #: Locus configuration
            self.locus_config: LocusConfig = LocusConfig(
                n=loci,
                recombination_rate=recombination_rate if recombination_rate is not None else 0
            )
        else:
            #: Locus configuration
            self.locus_config: LocusConfig = loci

            if recombination_rate is not None:
                self.locus_config.recombination_rate = recombination_rate

        # population names present in the population configuration but not in the demography
        initial_sizes = {p: {0: 1} for p in self.lineage_config.pop_names if p not in demography.pop_names}

        # add missing population sizes to demography
        if len(initial_sizes) > 0:
            demography.add_event(
                PopSizeChanges(initial_sizes)
            )

        # add zero lineage counts to lineage configuration for populations only present in the demography
        unspecified_lineages = set(demography.pop_names) - set(self.lineage_config.pop_names)
        self.lineage_config = LineageConfig(self.lineage_config.lineage_dict | {p: 0 for p in unspecified_lineages})

        #: Coalescent model
        self.model: CoalescentModel = model

        #: Demography
        self.demography: Demography = demography

        #: End time
        self.end_time: float = end_time

    @property
    @abstractmethod
    def tree_height(self) -> DensityAwareDistribution:
        """
        Tree height distribution.
        """
        pass

    @property
    @abstractmethod
    def total_branch_length(self) -> MomentAwareDistribution:
        """
        Total branch length distribution.
        """
        pass

    @property
    @abstractmethod
    def sfs(self) -> MomentAwareDistribution:
        """
        Unfolded site-frequency spectrum distribution.
        """
        pass

    @property
    @abstractmethod
    def fsfs(self) -> MomentAwareDistribution:
        """
        Folded site-frequency spectrum distribution.
        """
        pass


class Coalescent(AbstractCoalescent, Serializable):
    """
    Coalescent distribution.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | LineageConfig,
            model: CoalescentModel = None,
            demography: Demography = None,
            loci: int | LocusConfig = 1,
            recombination_rate: float = None,
            pbar: bool = False,
            parallelize: bool = True,
            start_time: float = 0,
            end_time: float = None,
            regularize: bool = True
    ):
        """
        Create object.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
            or dictionary with population names as keys and number of lineages as values for multiple populations.
             Alternatively, a :class:`~phasegen.lineage.LineageConfig` object can be passed.
        :param model: Coalescent model. Default is the standard coalescent.
        :param demography: Demography.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate.
        :param pbar: Whether to show a progress bar.
        :param parallelize: Whether to parallelize computations.
        :param start_time: Time when to start accumulating moments. By default, this is 0.
        :param end_time: Time when to end the accumulating moments. If ``None``, the end time is taken to
            be the time of almost sure absorption. Note that unnecessarily large end times can lead to numerical errors.
        :param regularize: Whether to regularize the intensity matrix for numerical stability.
        """
        super().__init__(
            n=n,
            model=model,
            loci=loci,
            recombination_rate=recombination_rate,
            demography=demography,
            end_time=end_time
        )

        #: Time when to start accumulating moments
        self.start_time: float = start_time

        #: Whether to show a progress bar
        self.pbar: bool = pbar

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

        #: Whether to regularize the intensity matrix for numerical stability
        self.regularize: bool = regularize

    @cached_property
    def lineage_counting_state_space(self) -> LineageCountingStateSpace:
        """
        The lineage-counting state space.
        """
        return LineageCountingStateSpace(
            lineage_config=self.lineage_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.demography.get_epoch(0)
        )

    @cached_property
    def block_counting_state_space(self) -> BlockCountingStateSpace:
        """
        The block-counting state space.
        """
        return BlockCountingStateSpace(
            lineage_config=self.lineage_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.demography.get_epoch(0)
        )

    @cached_property
    def tree_height(self) -> TreeHeightDistribution:
        """
        Tree height distribution.
        """
        return TreeHeightDistribution(
            state_space=self.lineage_counting_state_space,
            demography=self.demography,
            start_time=self.start_time,
            end_time=self.end_time,
            regularize=self.regularize
        )

    @cached_property
    def total_branch_length(self) -> PhaseTypeDistribution:
        """
        Total branch length distribution.
        """
        return PhaseTypeDistribution(
            reward=TotalBranchLengthReward(),
            tree_height=self.tree_height,
            state_space=self.lineage_counting_state_space,
            demography=self.demography,
            regularize=self.regularize
        )

    @cached_property
    def sfs(self) -> UnfoldedSFSDistribution:
        """
        Unfolded site-frequency spectrum distribution.
        """
        return UnfoldedSFSDistribution(
            state_space=self.block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography,
            regularize=self.regularize
        )

    @cached_property
    def fsfs(self) -> FoldedSFSDistribution:
        """
        Folded site-frequency spectrum distribution.
        """
        return FoldedSFSDistribution(
            state_space=self.block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography,
            regularize=self.regularize
        )

    def _get_dist(self, k: int, rewards: Iterable[Reward] = None) -> PhaseTypeDistribution:
        """
        Get the kth-order phase-type distribution with state space inferred from the rewards.
        The returned phase-type distribution is configured with the unit reward.

        :param k: Order of the moment.
        :param rewards: Sequence of k rewards. By default, tree height rewards are used.
        :return: Distribution.
        """
        if rewards is None:
            rewards = [TreeHeightReward()] * k

        if Reward.support(LineageCountingStateSpace, rewards):
            state_space = self.lineage_counting_state_space
        else:
            state_space = self.block_counting_state_space

        return PhaseTypeDistribution(
            reward=UnitReward(),
            tree_height=self.tree_height,
            state_space=state_space,
            demography=self.demography,
            regularize=self.regularize
        )

    @_make_hashable
    @cache
    def moment(
            self,
            k: int = 1,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        Get the kth (non-central) moment using the specified rewards and state space.

        :param k: The order of the moment
        :param rewards: Sequence of k rewards. By default, tree height rewards are used.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: The kth moment
        """
        return self._get_dist(k, rewards).moment(
            k=k,
            rewards=rewards,
            start_time=start_time,
            end_time=end_time,
            center=center,
            permute=permute
        )

    def _raw_moment(
            self,
            k: int,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None
    ) -> float:
        """
        Get the kth raw moment using the specified rewards and state space.

        :param k: The order of the moment
        :param rewards: Sequence of k rewards. By default, tree height rewards are used.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :return: The kth raw moment
        """
        return self.moment(
            k=k,
            rewards=rewards,
            start_time=start_time,
            end_time=end_time,
            center=False,
            permute=False
        )

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        """
        Accumulate moments at different times.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and
            the 99th percentile.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: Accumulation of moments.
        """
        return self._get_dist(k, rewards).accumulate(
            k=k,
            end_times=end_times,
            rewards=rewards,
            center=center,
            permute=permute
        )

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = False,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot the accumulation of moments.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and
            the 99th percentile.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        self._get_dist(k, rewards).plot_accumulation(
            k=k,
            end_times=end_times,
            rewards=rewards,
            center=center,
            permute=permute,
            ax=ax,
            show=show,
            file=file,
            clear=clear,
            label=label,
            title=title
        )

    def drop_cache(self):
        """
        Drop state space cache.
        """
        self.lineage_counting_state_space.drop_cache()
        self.block_counting_state_space.drop_cache()

    def __setstate__(self, state: dict):
        """
        Restore the state of the object from a serialized state.

        :param state: State.
        """
        self.__dict__.update(state)

    def __getstate__(self) -> dict:
        """
        Get the state of the object for serialization.

        :return: State.
        """
        # create deep copy of object without causing infinite recursion
        other = copy.deepcopy(self.__dict__)

        if 'lineage_counting_state_space' in other:
            other['lineage_counting_state_space'].drop_cache()

        if 'block_counting_state_space' in other:
            other['block_counting_state_space'].drop_cache()

        return other

    def to_json(self) -> str:
        """
        Serialize to JSON. Drop cache before serializing.

        :return: JSON string.
        """
        # copy object to avoid modifying the original
        other = copy.deepcopy(self)

        # drop cache
        other.drop_cache()

        return super(self.__class__, other).to_json()

    def _to_msprime(
            self,
            num_replicates: int = 10000,
            n_threads: int = 10,
            parallelize: bool = True,
            record_migration: bool = False,
            simulate_mutations: bool = False,
            mutation_rate: float = None,
            seed: int = None
    ) -> 'MsprimeCoalescent':
        """
        Convert to msprime coalescent.

        :param num_replicates: Number of replicates.
        :param n_threads: Number of threads.
        :param parallelize: Whether to parallelize.
        :param record_migration: Whether to record migrations which is necessary to calculate statistics per deme.
        :param simulate_mutations: Whether to simulate mutations.
        :param mutation_rate: Mutation rate.
        :return: msprime coalescent.
        """
        if self.start_time != 0:
            self._logger.warning("Non-zero start times are not supported by MsprimeCoalescent.")

        return MsprimeCoalescent(
            n=self.lineage_config,
            demography=self.demography,
            model=self.model,
            loci=self.locus_config,
            recombination_rate=self.locus_config.recombination_rate,
            mutation_rate=mutation_rate,
            end_time=self.end_time,
            num_replicates=num_replicates,
            n_threads=n_threads,
            parallelize=parallelize,
            record_migration=record_migration,
            simulate_mutations=simulate_mutations,
            seed=seed
        )


class MsprimeCoalescent(AbstractCoalescent):  # pragma: no cover
    """
    Empirical coalescent distribution based on msprime simulations.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | LineageConfig,
            demography: Demography = None,
            model: CoalescentModel = StandardCoalescent(),
            loci: int | LocusConfig = 1,
            recombination_rate: float = None,
            mutation_rate: float = None,
            end_time: float = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            record_migration: bool = False,
            simulate_mutations: bool = False,
            seed: int = None
    ):
        """
        Simulate data using msprime.

        :param n: Number of Lineages.
        :param demography: Demography.
        :param model: Coalescent model.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate.
        :param mutation_rate: Mutation rate.
        :param end_time: Time when to end the simulation.
        :param num_replicates: Number of replicates.
        :param n_threads: Number of threads.
        :param parallelize: Whether to parallelize.
        :param record_migration: Whether to record migrations which is necessary to calculate statistics per deme.
        :param simulate_mutations: Whether to simulate mutations.
        :param seed: Random seed.
        """
        super().__init__(
            n=n,
            model=model,
            loci=loci,
            recombination_rate=recombination_rate,
            demography=demography,
            end_time=end_time
        )

        if mutation_rate is not None and not simulate_mutations:
            self._logger.warning("Mutation rate is set but mutations are not simulated.")

        #: Site frequency spectrum counts per locus, deme and replicate.
        self.sfs_lengths: np.ndarray | None = None

        #: Total branch lengths per locus, deme and replicate.
        self.total_branch_lengths: np.ndarray | None = None

        #: Tree heights per locus, deme and replicate.
        self.heights: np.ndarray | None = None

        #: Mutations per locus, deme and replicate.
        self.mutations: np.ndarray | None = None

        #: Number of replicates.
        self.num_replicates: int = num_replicates

        #: Mutation rate.
        self.mutation_rate: float = mutation_rate

        #: Number of threads.
        self.n_threads: int = n_threads

        #: Whether to parallelize computations.
        self.parallelize: bool = parallelize

        #: Whether to record migrations.
        self.record_migration: bool = record_migration

        #: Whether to simulate mutations.
        self.simulate_mutations: bool = simulate_mutations

        #: Random seed.
        self.seed: int = seed

    def get_coalescent_model(self) -> 'msprime.AncestryModel':
        """
        Get the coalescent model.

        :return: msprime coalescent model.
        """
        import msprime as ms

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
        """
        # number of replicates for one thread
        num_replicates = self.num_replicates // self.n_threads
        samples = self.lineage_config.lineage_dict
        demography = self.demography.to_msprime()
        model = self.get_coalescent_model()
        end_time = self.end_time
        n_pops = self.demography.n_pops
        sample_size = self.lineage_config.n

        def simulate_batch(seed: Optional[int]) -> (np.ndarray, np.ndarray, np.ndarray):
            """
            Simulate statistics.

            :param seed: Random seed.
            :return: Statistics.
            """
            import msprime as ms
            import tskit

            # simulate trees
            g: Generator = ms.sim_ancestry(
                sequence_length=self.locus_config.n,
                recombination_rate=self.locus_config.recombination_rate,
                samples=samples,
                num_replicates=num_replicates,
                record_migrations=self.record_migration,
                demography=demography,
                model=model,
                ploidy=1,
                end_time=end_time,
                random_seed=seed
            )

            # initialize variables
            heights = np.zeros((self.locus_config.n, n_pops, num_replicates), dtype=float)
            total_branch_lengths = np.zeros((self.locus_config.n, n_pops, num_replicates), dtype=float)
            sfs = np.zeros((self.locus_config.n, n_pops, num_replicates, sample_size + 1), dtype=float)
            mutations = np.zeros((self.locus_config.n, n_pops, num_replicates, sample_size + 1), dtype=int)

            # iterate over trees and compute statistics
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):

                tree: tskit.Tree
                for j, tree in enumerate(self._expand_trees(ts)):

                    # TODO record_migration only appears to work for relatively simple scenarios
                    if self.record_migration:

                        lineages = np.array(list(samples.values()))
                        t_coal = ts.tables.nodes.time[sample_size:]
                        node = sample_size - 1
                        t_migration = ts.migrations_time
                        i_migration = 0
                        time = 0

                        # population state per leave
                        pop_states = {n: tree.population(n) for n in range(sample_size)}

                        # iterate over coalescence events
                        for coal_time in t_coal:

                            # iterate over migration events within this coalescence event
                            while i_migration < len(t_migration) and time < t_migration[i_migration] <= coal_time:
                                delta = t_migration[i_migration] - time

                                # update statistics
                                heights[j, :, i] += delta * lineages / sum(lineages)
                                total_branch_lengths[j, :, i] += delta * lineages

                                for n, pop in pop_states.items():
                                    sfs[j, pop, i, tree.get_num_leaves(n)] += delta

                                # update lineages with migrations
                                lineages[ts.migrations_source[i_migration]] -= 1
                                lineages[ts.migrations_dest[i_migration]] += 1
                                pop_states[ts.migrations_node[i_migration]] = ts.migrations_dest[i_migration]

                                i_migration += 1
                                time += delta

                            # remaining time to next coalescence event
                            delta = coal_time - time

                            # update statistics
                            heights[j, :, i] += delta * lineages / sum(lineages)
                            total_branch_lengths[j, :, i] += delta * lineages

                            for n, pop in pop_states.items():
                                sfs[j, pop, i, tree.get_num_leaves(n)] += delta

                            # reduce by number of coalesced lineages
                            lineages[tree.population(node + 1)] -= len(tree.get_children(node + 1)) - 1

                            # delete children from pop_states
                            [pop_states.__delitem__(n) for n in tree.get_children(node + 1)]

                            # add parent to pop_states
                            pop_states[node + 1] = tree.population(node + 1)

                            time += delta
                            node += 1

                    else:

                        heights[j, 0, i] = tree.time(tree.roots[0])
                        total_branch_lengths[j, 0, i] = tree.total_branch_length

                        for node in tree.nodes():
                            t = tree.get_branch_length(node)
                            n = tree.get_num_leaves(node)

                            sfs[j, 0, i, n] += t

                # simulate mutations if specified
                if self.simulate_mutations:

                    mts = ms.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)
                    tree = next(mts.trees())

                    for node in mts.mutations_node:
                        mutations[0, 0, i, tree.get_num_leaves(node)] += 1

            return np.concatenate([[heights.T], [total_branch_lengths.T], sfs.T, mutations.T])

        # parallelize and add up results
        res = np.hstack(parallelize(
            func=simulate_batch,
            data=[self.seed + i if self.seed is not None else None for i in range(self.n_threads)],
            parallelize=self.parallelize,
            batch_size=num_replicates,
            desc="Simulating trees"
        ))

        # store results
        self.heights = res[0].T
        self.total_branch_lengths = res[1].T
        self.sfs_lengths = res[2:sample_size + 3].T
        self.mutations = res[sample_size + 3:].T.astype(int)

    @staticmethod
    def _expand_trees(ts: 'tskit.TreeSequence') -> Iterator['tskit.Tree']:
        """
        Expand tree sequence to `n` trees where `n` is the number of loci.

        :param ts: Tree sequence.
        :return: List of trees.
        """
        for tree in ts.trees():
            for _ in range(int(tree.length)):
                yield tree

    def _get_cached_times(self) -> np.ndarray:
        """
        Get cached times.

        """
        t_max = self.heights.sum(axis=1).max()

        return np.linspace(0, t_max, 100)

    def touch(self, **kwargs):
        """
        Touch cached properties.

        :param kwargs: Additional keyword arguments.
        """
        self.simulate()

        t = self._get_cached_times()

        self.tree_height.touch(t)
        self.total_tree_height.touch(t)
        self.total_branch_length.touch(t)
        self.sfs.touch(t)
        self.fsfs.touch(t)

    def drop(self):
        """
        Drop simulated data.
        """
        self.heights = None
        self.total_branch_lengths = None
        self.sfs_lengths = None
        self.mutations = None

        self.tree_height.drop()
        self.total_tree_height.drop()
        self.total_branch_length.drop()
        self.sfs.drop()
        self.fsfs.drop()

        # caused problems when serializing
        self.demography = None

    @cached_property
    def tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """
        Tree height distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(
            self.heights,
            pops=self.demography.pop_names,
            locus_agg=lambda x: x.max(axis=0)
        )

    @cached_property
    def total_tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """
        Total tree height distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(self.heights, pops=self.demography.pop_names)

    @cached_property
    def total_branch_length(self) -> EmpiricalPhaseTypeDistribution:
        """
        Total branch length distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(self.total_branch_lengths, pops=self.demography.pop_names)

    @cached_property
    def sfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Unfolded site-frequency spectrum distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=self.sfs_lengths,
            mutations=self.mutations.T[1:-1].T,
            pops=self.demography.pop_names,
            sfs_dist=UnfoldedSFSDistribution
        )

    @cached_property
    def fsfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Folded site-frequency spectrum distribution.
        """
        self.simulate()

        mid = (self.lineage_config.n + 1) // 2

        # fold SFS branch lengths
        lengths = self.sfs_lengths.copy().T
        lengths[:mid] += lengths[-mid:][::-1]
        lengths[-mid:] = 0

        # fold SFS mutations
        mutations = self.mutations.copy().T
        mutations[:mid] += mutations[-mid:][::-1]
        mutations = mutations[1:self.lineage_config.n // 2 + 1]

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=lengths.T,
            mutations=mutations.T,
            pops=self.demography.pop_names,
            sfs_dist=FoldedSFSDistribution
        )

    def to_phasegen(self) -> Coalescent:
        """
        Convert to native phasegen coalescent.

        :return: phasegen coalescent.
        """
        return Coalescent(
            n=self.lineage_config,
            model=self.model,
            demography=self.demography,
            loci=self.locus_config,
            recombination_rate=self.locus_config.recombination_rate,
            end_time=self.end_time
        )
