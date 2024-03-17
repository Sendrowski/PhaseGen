"""
Probability distributions.
"""

import copy
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property, cache
from math import factorial
from typing import Generator, List, Callable, Tuple, Dict, Collection, Iterable, Iterator

import numpy as np
import tskit
from matplotlib import pyplot as plt
from numpy.polynomial.hermite_e import HermiteE
from scipy import special
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

from .coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from .demography import Demography, Epoch, PopSizeChanges
from .lineage import LineageConfig
from .locus import LocusConfig
from .rewards import Reward, TreeHeightReward, TotalBranchLengthReward, UnfoldedSFSReward, DemeReward, UnitReward, \
    LocusReward, CombinedReward, FoldedSFSReward, SFSReward
from .serialization import Serializable
from .spectrum import SFS, SFS2
from .state_space import BlockCountingStateSpace, DefaultStateSpace, StateSpace
from .utils import expm, parallelize
from .visualization import Visualization

logger = logging.getLogger('phasegen')


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

    def touch(self):
        """
        Touch all cached properties.
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


class MarginalDistributions(Mapping, ABC):
    """
    Base class for marginal distributions.

    TODO remove marginal properties from sub-distributions
    """

    @abstractmethod
    @cached_property
    def cov(self) -> np.ndarray:
        """
        Get the covariance matrix.

        :return: covariance matrix
        """
        pass

    @abstractmethod
    @cached_property
    def corr(self) -> np.ndarray:
        """
        Get the correlation matrix.

        :return: correlation matrix
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
    def loci(self) -> Dict[int, 'PhaseTypeDistribution']:
        """
        Get the distribution for each locus.

        :return: List of distributions.
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
        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        m2 = self.dist.moment(k=2, rewards=(
            CombinedReward([self.dist.reward, LocusReward(locus1)]),
            CombinedReward([self.dist.reward, LocusReward(locus2)])
        ))

        return m2 - self.loci[locus1].mean * self.loci[locus2].mean

    def get_corr(self, locus1: int, locus2: int) -> float:
        """
        Get the correlation coefficient between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The correlation coefficient.
        """
        return self.get_cov(locus1, locus2) / (self.loci[locus1].std * self.loci[locus2].std)

    @cached_property
    def cov(self) -> np.ndarray:
        """
        The covariance matrix for the loci under the distribution in question.

        :return: covariance matrix
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_cov(i, j) for i in range(n_loci)] for j in range(n_loci)])

    @cached_property
    def corr(self) -> np.ndarray:
        """
        The correlation matrix for the loci under the distribution in question.

        :return: correlation matrix
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
    def demes(self) -> Dict[str, 'PhaseTypeDistribution']:
        """
        Get the distribution for each deme.

        :return: List of distributions.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        demes = {}
        for pop in self.dist.pop_config.pop_names:
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
        if pop1 not in self.dist.pop_config.pop_names or pop2 not in self.dist.pop_config.pop_names:
            raise ValueError(f"Population {pop1} or {pop2} does not exist.")

        m2 = self.dist.moment(k=2, rewards=(
            CombinedReward([self.dist.reward, DemeReward(pop1)]),
            CombinedReward([self.dist.reward, DemeReward(pop2)])
        ))

        return m2 - self.demes[pop1].mean * self.demes[pop2].mean

    def get_corr(self, pop1: str, pop2: str) -> float:
        """
        Get the correlation coefficient between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The correlation coefficient.
        """
        return self.get_cov(pop1, pop2) / (self.demes[pop1].std * self.demes[pop2].std)

    @cached_property
    def cov(self) -> np.ndarray:
        """
        The covariance matrix for the demes under the distribution in question.

        :return: covariance matrix
        """
        pops = self.dist.pop_config.pop_names

        return np.array([[self.get_cov(p1, p2) for p1 in pops] for p2 in pops])

    @cached_property
    def corr(self) -> np.ndarray:
        """
        The correlation matrix for the demes under the distribution in question.

        :return: correlation matrix
        """
        pops = self.dist.pop_config.pop_names

        return np.array([[self.get_corr(p1, p2) for p1 in pops] for p2 in pops])


class DensityAwareDistribution(MomentAwareDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments and densities can be calculated.
    """

    @abstractmethod
    def cdf(self, t) -> float | np.ndarray:
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
    def pdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Density function.

        :param t: Value or values to evaluate the density function at.
        :return: Density.
        """
        pass

    def plot_cdf(
            self,
            ax: plt.Axes = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height CDF'
    ) -> plt.axes:
        """
        Plot cumulative distribution function.

        :param ax: Axes to plot on.
        :param t: Values to evaluate the CDF at. By default, 100 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        if t is None:
            t = np.linspace(0, self.quantile(0.99), 100)

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
            ax: plt.Axes = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height PDF'
    ) -> plt.axes:
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param t: Values to evaluate the density function at.
            By default, 100 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        if t is None:
            t = np.linspace(0, self.quantile(0.99), 100)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self.pdf(t),
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
    #: Number of decimals to round moments to.
    n_decimals: int = 12

    def __init__(
            self,
            state_space: StateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography = None,
            reward: Reward = None
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward. By default, the tree height reward.
        """
        if demography is None:
            demography = Demography()

        if reward is None:
            reward = TreeHeightReward()

        super().__init__()

        #: Population configuration
        self.pop_config: LineageConfig = state_space.pop_config

        #: Locus configuration
        self.locus_config: LocusConfig = state_space.locus_config

        #: Reward
        self.reward: Reward = reward

        #: State space
        self.state_space: StateSpace = state_space

        #: Demography
        self.demography: Demography = demography

        #: Tree height distribution
        self.tree_height = tree_height

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
    def std(self) -> float | SFS:
        """
        Get the standard deviation in the absorption time.

        :return: The standard deviation in the absorption time.
        """
        return self.var ** 0.5

    @cached_property
    def m2(self) -> float | SFS:
        """
        Get the (non-central) second moment.

        :return: The second moment.
        """
        return self.moment(k=2)

    @cached_property
    def demes(self) -> MarginalDemeDistributions:
        """
        Get marginal distributions for each deme.

        :return: Marginal distributions.
        """
        return MarginalDemeDistributions(self)

    @cached_property
    def loci(self) -> MarginalLocusDistributions:
        """
        Get marginal distributions for each locus.

        :return: Marginal distributions.
        """
        return MarginalLocusDistributions(self)

    @cache
    def moment_deprecated(
            self,
            k: int,
            rewards: Tuple[Reward, ...] = None,
            start_time: float = None,
            end_time: float = None
    ) -> float:
        """
        Get the kth (non-central) moment.

        TODO remove later

        :param k: The order of the moment.
        :param rewards: Tuple of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :return: The kth moment
        """
        # use default reward if not specified
        if rewards is None:
            rewards = (self.reward,) * k
        else:
            if len(rewards) != k:
                raise ValueError(f"Number of rewards must be {k}.")

        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            end_time = self.tree_height.t_max

        # get reward matrix
        R = [np.diag(r.get(state_space=self.state_space)) for r in rewards]

        # number of states
        n_states = self.state_space.k

        # initialize Van Loan matrix holding (rewarded) moments
        Q = np.eye(n_states * (k + 1))

        # iterate through epochs and compute initial values
        for i, epoch in enumerate(self.demography.epochs):

            # get state space for this epoch
            self.state_space.update_epoch(epoch)

            # get Van Loan matrix
            V = self._get_van_loan_matrix(S=self.state_space.S, R=R, k=k)

            # compute tau
            tau = min(epoch.end_time, end_time) - epoch.start_time

            # compute matrix exponential
            Q_curr = expm(V * tau)

            # update by accumulated reward
            Q = Q @ Q_curr

            # break if we have reached the end time
            if epoch.end_time >= end_time:
                break

        # calculate moment
        m = factorial(k) * self.state_space.alpha @ Q[:n_states, -n_states:] @ self.state_space.e

        # subtract accumulated moment up to start time if greater than 0
        if start_time > 0:
            # call on PhaseTypeDistribution to make possible SFS moment calculation
            m -= PhaseTypeDistribution.moment(
                self,
                k=k,
                rewards=rewards,
                start_time=0,
                end_time=self.tree_height.start_time
            )

        # TODO round to avoid numerical errors?
        # return np.round(m, self.n_decimals)
        return m

    @cache
    def moment(
            self,
            k: int,
            rewards: Tuple[Reward, ...] = None,
            start_time: float = None,
            end_time: float = None
    ) -> float:
        """
        Get the kth (non-central) moment.

        :param k: The order of the moment.
        :param rewards: Tuple of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :return: The kth moment
        """
        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            end_time = self.tree_height.t_max

        if start_time > 0:
            m_start, m_end = PhaseTypeDistribution.accumulation(self, k, [start_time, end_time], rewards)
            return float(m_end - m_start)

        return float(PhaseTypeDistribution.accumulation(self, k, [end_time], rewards)[0])

    def accumulation(
            self,
            k: int,
            end_times: Iterable[float] | float,
            rewards: Tuple[Reward, ...] = None
    ) -> np.ndarray | float:
        """
        Evaluate the kth (non-central) moment at different end times.

        :param k: The order of the moment.
        :param end_times: List of ends times or end time when to evaluate the moment.
        :param rewards: Tuple of k rewards. By default, the default reward of the underlying distribution.
        :return: The moment accumulated at the specified times or time.
        """
        # handle scalar input
        if not isinstance(end_times, Iterable):
            return self.accumulation(k, [end_times], rewards)[0]

        end_times = np.array(list(end_times))

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

        epochs = self.demography.epochs
        epoch: Epoch = next(epochs)

        # get the transition matrix for the first epoch
        self.state_space.update_epoch(epoch)

        # number of states
        n_states = self.state_space.k

        # initialize block matrix holding (rewarded) moments
        Q = np.eye(n_states * (k + 1))
        u_prev = 0

        # initialize probabilities
        moments = np.zeros_like(t_sorted, dtype=float)

        # get reward matrix
        R = [np.diag(r.get(state_space=self.state_space)) for r in rewards]

        # get Van Loan matrix
        V = self._get_van_loan_matrix(S=self.state_space.S, R=R, k=k)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                # update transition matrix with remaining time in current epoch
                Q @= expm(V * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                epoch = next(epochs)
                self.state_space.update_epoch(epoch)
                V = self._get_van_loan_matrix(S=self.state_space.S, R=R, k=k)

            # update with remaining time in current epoch
            Q @= expm(V * (u - u_prev))

            moments[i] = factorial(k) * self.state_space.alpha @ Q[:n_states, -n_states:] @ self.state_space.e

            u_prev = u

        # sort probabilities back to original order
        moments = moments[np.argsort(end_times)]

        return moments

    def plot_accumulation(
            self,
            k: int,
            end_times: Iterable[float] = None,
            rewards: Tuple[Reward, ...] = None,
            ax: plt.Axes = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> plt.axes:
        """
        Plot accumulation of (non-central) moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 100 evenly spaced values between 0 and the
            99th percentile.
        :param rewards: Tuple of k rewards. By default, the default reward of the underlying distribution.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 100)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = f"Moment accumulation ({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})"

        Visualization.plot(
            ax=ax,
            x=end_times,
            y=self.accumulation(k, end_times, rewards),
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

    #: Maximum number of iterations when determining time to almost sure absorption in the last epoch.
    max_iter: int = 30

    #: Probability of almost sure absorption.
    p_absorption: float = 1 - 1e-15

    def __init__(
            self,
            state_space: DefaultStateSpace,
            demography: Demography = None,
            start_time: float = 0,
            end_time: float = None
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param demography: The demography.
        :param start_time: Time when to start accumulating moments.
        :param end_time: Time when to end accumulation of moments. By default, the time until almost sure absorption.
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
            reward=TreeHeightReward()
        )

        #: State space
        self.state_space: DefaultStateSpace = state_space

        #: Start time
        self.start_time: float = start_time

        #: End time
        self.end_time: float | None = end_time

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
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
        t_sorted: Collection[float] = np.sort(t)

        epochs = self.demography.epochs
        epoch: Epoch = next(epochs)

        # get the transition matrix for the first epoch
        self.state_space.update_epoch(epoch)

        # initialize transition matrix
        T = np.eye(self.state_space.k)
        u_prev = 0

        # initialize probabilities
        probs = np.zeros_like(t_sorted)

        # take reward vector as exit vector
        e = self.reward.get(self.state_space)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                # update transition matrix with remaining time in current epoch
                T @= expm(self.state_space.S * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                epoch = next(epochs)
                self.state_space.update_epoch(epoch)

            # update transition matrix with remaining time in current epoch
            T @= expm(self.state_space.S * (u - u_prev))

            probs[i] = 1 - self.state_space.alpha @ T @ e

            u_prev = u

        # sort probabilities back to original order
        probs = probs[np.argsort(t)]

        return probs

    @cache
    def quantile(
            self,
            q: float,
            expansion_factor: float = 2,
            tol: float = 1e-5,
            max_iter: int = 1000
    ):
        """
        Find the specified quantile of a CDF using an adaptive bisection method.

        TODO this can be optimized

        :param q: The desired quantile (between 0 and 1).
        :param expansion_factor: Factor by which to expand the upper bound that does not yet contain the quantile.
        :param tol: The tolerance for convergence.
        :param max_iter: Maximum number of iterations for the bisection method.
        :return: The approximate x value for the nth quantile.
        """
        if q < 0 or q > 1:
            raise ValueError("Specified quantile must be between 0 and 1.")

        # initialize bounds
        a, b = 0, 1

        # expand upper bound until it contains the quantile
        while self.cdf(b) < q and max_iter > 0:
            b *= expansion_factor
            max_iter -= 1

        # use bisection method within the determined bounds
        while (b - a) > tol and max_iter > 0:
            m = (a + b) / 2
            if self.cdf(m) < q:
                a = m
            else:
                b = m
            max_iter -= 1

        # warn if maximum number of iterations reached
        if max_iter == 0:
            raise RuntimeError("Maximum number of iterations reached when determining quantile.")

        return (a + b) / 2

    def pdf(self, t: float | np.ndarray, dx: float = 1e-10) -> float | np.ndarray:
        """
        Density function. We use numerical differentiation of the CDF to calculate the density. This provides good
        results as the CDF is exact and continuous.

        :param t: Value or values to evaluate the density function at.
        :param dx: Step size for numerical differentiation.
        :return: Density
        """
        # determine (non-negative) evaluation points
        x1 = np.max([t - dx / 2, np.zeros_like(t)], axis=0)
        x2 = x1 + dx

        return (self.cdf(x2) - self.cdf(x1)) / dx

    @cached_property
    def t_max(self) -> float:
        """
        Get the time until which computations are performed. This is either the end time specified when initializing
        the distribution or the time until almost sure absorption.

        :return: The maximum time.
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
        Stopping the simulation when no more rewards are accumulated is not a good idea, as this
        can happen before almost sure absorption (exponential runaway growth, temporary deadlock in different demes).

        TODO clean up
        """
        # initialize transition matrix
        T_curr = np.eye(self.state_space.k)

        # take reward vector as exit vector
        e = self.reward.get(self.state_space)

        # current time and probability of absorption
        t, p = 0, 0

        for i, epoch in enumerate(self.demography.epochs):
            # update state space
            self.state_space.update_epoch(epoch)

            if i > self.max_epochs:
                return self._warn_convergence(t, p)

            if epoch.tau < np.inf:
                tau = epoch.tau

                # update transition matrix
                T_curr = expm(self.state_space.S * tau) @ T_curr

                # calculate probability of absorption
                p = 1 - self.state_space.alpha @ T_curr @ e

                t += tau

                if p >= self.p_absorption:
                    return t
            else:
                break

        tau = 1
        T_tau = expm(self.state_space.S)

        # in the last epoch, we increase tau exponentially
        for i in range(self.max_iter):
            # update transition matrix
            T_curr = T_tau @ T_curr

            # calculate probability of absorption
            p_next = 1 - self.state_space.alpha @ T_curr @ e

            # break if p_next is not increasing anymore
            if p_next < p:
                return self._warn_convergence(t, p)

            t += tau
            tau *= 2
            T_tau = T_tau @ T_tau
            p = p_next

            if p >= self.p_absorption:
                return t

        return self._warn_convergence(t, p)

    def _warn_convergence(self, t: float, p: float) -> float:
        """
        Warn the user if the probability of absorption is not close to 1.

        :param t: The time.
        :param p: The probability of absorption.
        :return: The time.
        """
        self._logger.warning(
            "Could not reliably find time of almost sure absorption. "
            f"Using time {t:.1f} with probability of absorption {p:.1e}. "
            "If p is close to 1, this is likely due to numerical imprecision. "
            "If p is not close to 1, this could be due to unreachable states or very large or small absorption times. "
            "You can also set the end time manually (see :attr:`Coalescent.end_time`)."
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
            reward: Reward = None
    ):
        """
        Initialize the distribution.

        :param state_space: Block counting state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param pbar: Whether to show a progress bar.
        :param parallelize: Use parallelization.
        :param reward: The reward to multiply the SFS reward with. By default, the unit reward is used, which
            has no effect.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward,
        )

        #: Whether to show a progress bar
        self.pbar: bool = pbar

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

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

    @cache
    def moment(
            self,
            k: int,
            rewards: Tuple[SFSReward, ...] = None,
            start_time: float = None,
            end_time: float = None
    ) -> SFS:
        """
        Get the kth moment of the site-frequency spectrum.

        :param k: The order of the moment
        :param rewards: Tuple of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :return: The kth SFS moments
        """
        if rewards is None:
            rewards = (self.reward,) * k

        # optionally parallelize the moment computation of the SFS bins
        moments = parallelize(
            func=lambda x: self.get_moment(*x),
            data=[[k, i, rewards, start_time, end_time] for i in self._get_indices()],
            desc=f"Calculating moments of order {k}",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        return SFS([0] + list(moments) + [0] * (self.pop_config.n - len(moments)))

    def accumulation(
            self,
            k: int,
            end_times: Iterable[float] | float,
            rewards: Tuple[Reward, ...] = None
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moments at different end times.

        :param k: The order of the moment.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Tuple of k rewards. By default, the default reward of the underlying distribution.
        :return: Array of moments accumulated at the specified times, one for each site-frequency count.
        """
        if not isinstance(end_times, Iterable):
            return self.accumulation(k, [end_times], rewards)[:, 0]

        indices = self._get_indices()
        end_times = np.array(list(end_times))

        accumulation = parallelize(
            func=lambda x: self.get_accumulation(*x),
            data=[[k, i, end_times, rewards] for i in indices],
            desc=f"Calculating accumulation of moments of order {k}",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        # pad with zeros
        return np.concatenate([
            np.zeros((1, len(end_times))),
            accumulation,
            np.zeros((self.pop_config.n - len(indices), len(end_times)))
        ])

    def plot_accumulation(
            self,
            k: int,
            end_times: Iterable[float] = None,
            rewards: Tuple[Reward, ...] = None,
            ax: plt.Axes = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> plt.axes:
        """
        Plot accumulation of (non-central) SFS moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 100 evenly spaced values between 0 and
            the 99th percentile.
        :param rewards: Tuple of k rewards. By default, the default reward of the underlying distribution.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        if ax is None:
            ax = plt.gca()

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 100)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = f"SFS Moment accumulation ({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})"

        # get accumulation of moments
        accumulation = self.accumulation(k, end_times, rewards)

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

    def get_moment(
            self,
            k: int,
            i: int,
            rewards: Tuple[SFSReward, ...] = None,
            start_time: float = None,
            end_time: float = None
    ) -> float:
        """
        Get the nth moment for the ith site-frequency count.

        :param k: The order of the moment
        :param i: The ith site-frequency count
        :param rewards: Tuple of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :return: The kth SFS (cross)-moment at the ith site-frequency count
        """
        if rewards is None:
            rewards = (self.reward,) * k

        return super().moment(
            k=k,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards]),
            start_time=start_time,
            end_time=end_time
        )

    def get_accumulation(
            self,
            k: int,
            i: int,
            end_times: Iterable[float] | float,
            rewards: Tuple[SFSReward, ...] = None
    ) -> np.ndarray | float:
        """
        Get accumulation of moments for the ith site-frequency count.

        :param k: The order of the moment
        :param i: The ith site-frequency count.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Tuple of k rewards.
        :return: The kth SFS (cross)-moment accumulations at the ith site-frequency count
        """
        if rewards is None:
            rewards = [self.reward] * k

        return super().accumulation(
            k=k,
            end_times=end_times,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards])
        )

    @cached_property
    def cov(self) -> SFS2:
        """
        The 2-SFS, i.e. the covariance matrix of the site-frequencies.
        """
        # create list of arguments for each combination of i, j
        indices = [(i, j) for i in self._get_indices() for j in self._get_indices()]

        # get sfs using parallelized function
        sfs_results = parallelize(
            func=lambda x: self.get_cov(*x),
            data=indices,
            desc="Calculating covariance",
            pbar=self.pbar,
            parallelize=self.parallelize
        )

        # re-structure the results to a matrix form
        sfs = np.zeros((self.pop_config.n + 1, self.pop_config.n + 1))
        for ((i, j), result) in zip(indices, sfs_results):
            sfs[i, j] = result

        # get matrix of marginal second moments
        m2 = np.outer(self.mean.data, self.mean.data)

        # calculate covariances
        cov = (sfs + sfs.T) / 2 - m2

        return SFS2(cov)

    @cache
    def get_cov(self, i: int, j: int) -> float:
        """
        Get the covariance between the ith and jth site-frequency.

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: covariance
        """
        return super().moment(
            k=2,
            rewards=(
                CombinedReward([self.reward, self._get_sfs_reward(i)]),
                CombinedReward([self.reward, self._get_sfs_reward(j)])
            )
        )

    @cached_property
    def corr(self) -> SFS2:
        """
        The correlation coefficient matrix of the site-frequencies.

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

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: Correlation coefficient
        """
        std_i = np.sqrt(super().moment(k=2, rewards=(CombinedReward([self.reward, self._get_sfs_reward(i)]),) * 2))
        std_j = np.sqrt(super().moment(k=2, rewards=(CombinedReward([self.reward, self._get_sfs_reward(j)]),) * 2))

        return self.get_cov(i, j) / (std_i * std_j)


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
        return np.arange(1, self.pop_config.n)


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
        return np.arange(1, self.pop_config.n // 2 + 1)


class EmpiricalDistribution(DensityAwareDistribution):
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
        Drop all cached properties.
        """
        self.samples = None

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
        Get the kth moment.

        :param k: The order of the moment
        :return: The kth moment
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


class EmpiricalSFSDistribution(EmpiricalDistribution):
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


class DictContainer(dict):
    """
    Dictionary container.
    """
    pass


class EmpiricalPhaseTypeDistribution(EmpiricalDistribution):
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
        Drop all cached properties.
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


class EmpiricalPhaseTypeSFSDistribution(EmpiricalPhaseTypeDistribution):
    """
    SFS phase-type distribution based on realisations.
    """

    def __init__(
            self,
            samples: np.ndarray | list, 
            pops: List[str],
            locus_agg: Callable = lambda x: x.sum(axis=0)
    ):
        """
        Create object.

        :param samples: 4-D array of samples.
        :param pops: List of population names.
        :param locus_agg: Aggregation function for loci.
        """
        over_loci = locus_agg(samples).astype(float)

        EmpiricalDistribution.__init__(self, over_loci.sum(axis=0))

        #: Population names
        self.pops = pops

        #: Samples by deme and locus
        self._samples = samples

        #: Correlation matrix for the loci
        self.pops_corr = self._get_stat_pops(over_loci, np.corrcoef)

        #: Covariance matrix for the demes
        self.pops_cov: np.ndarray = self._get_stat_pops(over_loci, np.cov)

        self.loci_corr = None

        self.loci_cov = None

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

        :param n: n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`PopulationConfig` object can be passed.
        :param model: Coalescent model. By default, the standard coalescent is used.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate.
        :param demography: Demography.
        :param end_time: Time when to end the computation. If `None`, the end time is end time is taken to be the
            time of almost sure absorption. Note that unnecessarily large end times can lead to numerical errors.
        """
        self._logger = logger.getChild(self.__class__.__name__)

        if model is None:
            model = StandardCoalescent()

        if demography is None:
            demography = Demography()

        if not isinstance(n, LineageConfig):
            #: Population configuration
            self.pop_config: LineageConfig = LineageConfig(n)
        else:
            #: Population configuration
            self.pop_config: LineageConfig = n

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
        Site frequency spectrum distribution.
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
            parallelize: bool = False,
            start_time: float = 0,
            end_time: float = None
    ):
        """
        Create object.

        :param :param n: n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`PopulationConfig` object can be passed.
        :param model: Coalescent model. Default is the standard coalescent.
        :param demography: Demography.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate.
        :param pbar: Whether to show a progress bar.
        :param parallelize: Whether to parallelize computations.
        :param start_time: Time when to start accumulating moments. By default, this is 0.
        :param end_time: Time when to end the accumulating moments. If `None`, the end time is taken to
            be the time of almost sure absorption. Note that unnecessarily large end times can lead to numerical errors.
        """
        super().__init__(
            n=n,
            model=model,
            loci=loci,
            recombination_rate=recombination_rate,
            demography=demography,
            end_time=end_time
        )

        # population names present in the population configuration but not in the demography
        initial_sizes = {p: {0: 1} for p in self.pop_config.pop_names if p not in self.demography.pop_names}

        # add missing population sizes to demography
        if len(initial_sizes) > 0:
            self.demography.add_event(
                PopSizeChanges(initial_sizes)
            )

        #: Time when to start accumulating moments
        self.start_time: float = start_time

        #: Whether to show a progress bar
        self.pbar: bool = pbar

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

    @cached_property
    def default_state_space(self) -> DefaultStateSpace:
        """
        The default state space.
        """
        return DefaultStateSpace(
            pop_config=self.pop_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.demography.get_epochs(0)
        )

    @cached_property
    def block_counting_state_space(self) -> BlockCountingStateSpace:
        """
        The block counting state space.
        """
        return BlockCountingStateSpace(
            pop_config=self.pop_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.demography.get_epochs(0)
        )

    @cached_property
    def tree_height(self) -> TreeHeightDistribution:
        """
        Tree height distribution.
        """
        return TreeHeightDistribution(
            state_space=self.default_state_space,
            demography=self.demography,
            start_time=self.start_time,
            end_time=self.end_time
        )

    @cached_property
    def total_branch_length(self) -> PhaseTypeDistribution:
        """
        Total branch length distribution.
        """
        return PhaseTypeDistribution(
            reward=TotalBranchLengthReward(),
            tree_height=self.tree_height,
            state_space=self.default_state_space,
            demography=self.demography
        )

    @cached_property
    def sfs(self) -> UnfoldedSFSDistribution:
        """
        Site frequency spectrum distribution.
        """
        return UnfoldedSFSDistribution(
            state_space=self.block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography
        )

    @cached_property
    def fsfs(self) -> FoldedSFSDistribution:
        """
        Folded site frequency spectrum distribution.
        """
        return FoldedSFSDistribution(
            state_space=self.block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography
        )

    def moment(
            self,
            k: int = 1,
            rewards: Tuple[Reward, ...] = None,
            state_space: StateSpace = None
    ) -> float:
        """
        Get the kth (non-central) moment using the specified rewards and state space.
        TODO infer state space type from rewards

        :param k: The order of the moment
        :param rewards: Tuple of k rewards. By default, tree height rewards are used.
        :param state_space: State space to use. By default, the default state space is used.
        :return: The kth moment
        """
        dist = PhaseTypeDistribution(
            reward=TreeHeightReward() if rewards is None else UnitReward(),
            tree_height=self.tree_height,
            state_space=self.default_state_space if state_space is None else state_space,
            demography=self.demography
        )

        return dist.moment(k, rewards)

    def drop_cache(self):
        """
        Drop state space cache.
        """
        self.default_state_space.drop_cache()
        self.block_counting_state_space.drop_cache()

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
            record_migration: bool = False
    ) -> 'MsprimeCoalescent':
        """
        Convert to msprime coalescent.

        :param num_replicates: Number of replicates.
        :param n_threads: Number of threads.
        :param parallelize: Whether to parallelize.
        :param record_migration: Whether to record migrations which is necessary to calculate statistics per deme.
        :return: msprime coalescent.
        """
        if self.start_time != 0:
            self._logger.warning("Non-zero start times are not supported by MsprimeCoalescent.")

        return MsprimeCoalescent(
            n=self.pop_config,
            demography=self.demography,
            model=self.model,
            loci=self.locus_config,
            recombination_rate=self.locus_config.recombination_rate,
            end_time=self.end_time,
            num_replicates=num_replicates,
            n_threads=n_threads,
            parallelize=parallelize,
            record_migration=record_migration
        )


class MsprimeCoalescent(AbstractCoalescent):
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
            end_time: float = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            record_migration: bool = False
    ):
        """
        Simulate data using msprime.

        :param n: Number of Lineages.
        :param demography: Demography.
        :param model: Coalescent model.
        :param loci: Number of loci or locus configuration.
        :param end_time: Time when to end the simulation.
        :param num_replicates: Number of replicates.
        :param n_threads: Number of threads.
        :param parallelize: Whether to parallelize.
        :param record_migration: Whether to record migrations which is necessary to calculate statistics per deme.
        """
        super().__init__(
            n=n,
            model=model,
            loci=loci,
            recombination_rate=recombination_rate,
            demography=demography,
            end_time=end_time
        )

        self.sfs_counts: np.ndarray | None = None
        self.total_branch_lengths: np.ndarray | None = None
        self.heights: np.ndarray | None = None

        self.num_replicates: int = num_replicates
        self.n_threads: int = n_threads
        self.parallelize: bool = parallelize
        self.record_migration: bool = record_migration

        self.p_accepted: int = 0

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
        samples = self.pop_config.lineage_dict
        demography = self.demography.to_msprime()
        model = self.get_coalescent_model()
        end_time = self.end_time
        n_pops = self.demography.n_pops
        sample_size = self.pop_config.n

        def simulate_batch(_) -> (np.ndarray, np.ndarray, np.ndarray):
            """
            Simulate statistics.

            :param _:
            :return: Statistics.
            """
            import msprime as ms

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
                end_time=end_time
            )

            # initialize variables
            heights = np.zeros((self.locus_config.n, n_pops, num_replicates), dtype=float)
            total_branch_lengths = np.zeros((self.locus_config.n, n_pops, num_replicates), dtype=float)
            sfs = np.zeros((self.locus_config.n, n_pops, num_replicates, sample_size + 1), dtype=float)

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
        self.heights, self.total_branch_lengths, self.sfs_counts = res[0].T, res[1].T, res[2:].T

    @staticmethod
    def _expand_trees(ts: tskit.TreeSequence) -> Iterator[tskit.Tree]:
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

    def touch(self):
        """
        Touch cached properties.
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
        self.sfs_counts = None

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
        Site frequency spectrum distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeSFSDistribution(self.sfs_counts, pops=self.demography.pop_names)

    @cached_property
    def fsfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Folded site frequency spectrum distribution.
        """
        self.simulate()

        mid = (self.pop_config.n + 1) // 2

        counts = self.sfs_counts.copy().T

        counts[:mid] += counts[-mid:][::-1]
        counts[-mid:] = 0

        return EmpiricalPhaseTypeSFSDistribution(counts.T, pops=self.demography.pop_names)

    def to_phasegen(self) -> Coalescent:
        """
        Convert to native phasegen coalescent.

        :return: phasegen coalescent.
        """
        return Coalescent(
            n=self.pop_config,
            model=self.model,
            demography=self.demography,
            loci=self.locus_config,
            recombination_rate=self.locus_config.recombination_rate,
            end_time=self.end_time
        )


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

        The mnc2cum implementation which provides more accurate results

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
