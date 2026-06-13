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
from ..caching import cached_property, cache
from math import factorial
from typing import Generator, List, Callable, Tuple, Dict, Collection, Iterable, Iterator, Optional, Sequence, Set, \
    Type, Union

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from ..coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from ..demography import Demography, PopSizeChanges
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalBranchLengthReward, UnfoldedSFSReward, DemeReward, UnitReward, \
    LocusReward, CombinedReward, FoldedSFSReward, SFSReward, CustomReward, JointSFSReward, TwoLocusSFSReward
from ..serialization import Serializable
from ..settings import Settings
from ..spectrum import SFS, SFS2, JointSFS, TwoLocusSFS
from ..state_space import BlockCountingStateSpace, LineageCountingStateSpace, StateSpace, JointBlockCountingStateSpace, \
    TwoLocusBlockCountingStateSpace
from ..utils import parallelize, multiset_permutations

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

    def touch(self, **kwargs: dict):
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
    def pdf(self, t: float | Sequence[float], **kwargs: dict) -> float | np.ndarray:
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
        from ..visualization import Visualization

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
        from ..visualization import Visualization

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
        self.lineage_config: LineageConfig = state_space.lineage_config

        #: Locus configuration
        self.locus_config: LocusConfig = state_space.locus_config

        #: Reward
        self.reward: Reward = reward

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

    @staticmethod
    def _get_van_loan_matrix_sparse(R: List[np.ndarray], S: 'sp.spmatrix', k: int = 1) -> 'sp.spmatrix':
        """
        Sparse, block-bidiagonal Van Loan matrix: the (sparse) intensity matrix ``S`` on the diagonal and the
        (diagonal) reward matrices on the super-diagonal. Built directly as a sparse matrix to avoid materializing
        the dense ``(k + 1) * n`` block matrix.

        :param R: List of length k of reward vectors (the diagonals of the reward matrices).
        :param S: Sparse intensity matrix.
        :param k: The order of the moment.
        :return: Sparse Van Loan matrix of size ``(k + 1) * (k + 1)`` blocks.
        """
        blocks = [[None] * (k + 1) for _ in range(k + 1)]
        for i in range(k + 1):
            blocks[i][i] = S
            if i < k:
                blocks[i][i + 1] = sp.diags(R[i])

        return sp.bmat(blocks, format='csr')

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
            # evaluate the moment to absorption: signal the closed-form path with an infinite end time when it
            # applies (no explicit end time, accumulation from 0, and absorption certain in the last epoch), but not
            # when flattening applies (which takes precedence and delegates to the smaller lineage-counting space),
            # otherwise use the estimated absorption time
            if (
                    Settings.closed_form_last_epoch and
                    not self._flattening_applies(k) and
                    start_time == 0 and
                    self.tree_height.end_time is None and
                    self._absorption_certain_in_last_epoch()
            ):
                end_time = np.inf
            else:
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

    @staticmethod
    def _get_regularization_factor(S: np.ndarray) -> float:
        """
        Get the regularization factor for the given intensity matrix. We
        multiply the intensity matrix by this factor to improve numerical
        stability when computing the matrix exponential of the Van Loan matrix.
        If regularization is disabled, this factor is 1.

        :param S: Intensity matrix.
        :return: Regularization factor.
        """
        if not Settings.regularize:
            return 1.0

        # obtain positive rates (for a sparse matrix, the positive stored entries)
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        # rewards in the Van Loan matrix are of order 1
        return 10 ** - np.log10(rates).mean()

    def _check_demography_conditioning(self):
        """
        Fail fast on extreme demographies whose population sizes or migration rates differ by more than ~double
        precision. Such demographies make the moment computation numerically unreliable, whether via the
        matrix-exponential absorption-time estimate (where scipy's ``expm`` one-norm power iteration becomes
        intermittently prohibitively slow) or the closed-form transient solve (where the rate matrix is
        ill-conditioned). Detected up front from the demography (not the rate matrix, whose range can also be
        widened by the coalescent model, e.g. multiple-merger models).

        :raises ValueError: if the population sizes and migration rates differ by a factor of more than ``1e16``.
        """
        epoch = self.demography.get_epoch(0)

        # coalescence rates scale as 1 / pop_size, migration enters at its own rate
        scales = [1 / v for v in epoch.pop_sizes.values() if v > 0]
        scales += [v for v in epoch.migration_rates.values() if v > 0]
        ratio = max(scales) / min(scales) if scales else 1

        if ratio > 1e16:
            raise ValueError(
                "The demography is too ill-conditioned to reliably compute the time of almost sure absorption: its "
                f"population sizes and migration rates differ by a factor of {ratio:.1e}. Use less extreme "
                "parameters, or set the end time manually (see ``Coalescent.end_time``)."
            )

    def _check_numerical_stability(self, S: np.ndarray, epoch: int):
        """
        Warn about potential numerical instability with very small or very large rates.

        :param S: (Regularized) intensity matrix.
        :param epoch: Epoch number.
        """
        # positive (off-diagonal) rates; for a sparse matrix these are the positive stored entries
        rates = S.data[S.data > 0] if sp.issparse(S) else S[S > 0]

        if rates.min() / rates.max() < 1e-10:
            self._logger.warning(
                f"Intensity matrix in epoch {epoch} contains rates that differ by more than 10 orders of magnitude: "
                f"min: {rates.min()}, max: {rates.max()}. "
                f"This may lead to numerical instability, despite matrix regularization."
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
        Evaluate the kth moment at different end times.

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
            self._logger.debug("accumulate (k=%d): centering (subtracting lower-order moment products)", k)

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
    def _accumulate_flattened(
            self,
            k: int,
            end_times: Sequence[float],
            rewards: Sequence[Reward] = None
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moment at different end times using the lineage counting state space.

        :param k: The order of the moment.
        :param end_times: Sequence of end times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :return: The moment accumulated at the specified times or time.
        :raises ValueError: If the state space is not a BlockCountingStateSpace, or if k is not 1, or if there are
            multiple populations or loci or if the coalescent model is not the standard coalescent.
        """

        if not isinstance(self.state_space, BlockCountingStateSpace):
            raise ValueError("Flattened accumulation is only supported for BlockCountingStateSpace.")

        if k != 1:
            raise ValueError("Flattened accumulation is only supported for k = 1.")

        if self.lineage_config.n_pops != 1 or self.locus_config.n != 1:
            raise ValueError("Flattened accumulation is only supported for a single population and a single locus.")

        if not isinstance(self.state_space.model, StandardCoalescent):
            raise ValueError("Flattened accumulation is only supported for standard coalescent.")

        reward = rewards[0] if rewards else self.reward
        r = reward._get(self.state_space)

        probs = self.state_space._state_probs

        # sum up weights for each state based on the number of lineages
        n = self.lineage_config.n
        weights = np.zeros(n)
        for i, s in enumerate(self.state_space.states):
            weights[n - s.lineages.sum()] += probs[i] * r[i]

        # Create a custom reward that returns the weights.
        weighted_reward = CustomReward(lambda _: weights)

        self._logger.debug(
            "flattening block-counting state space (%d states) onto the lineage-counting state space (%d states)",
            len(self.state_space.states), self.tree_height.state_space.k
        )

        return self.tree_height._accumulate(k=k, end_times=end_times, rewards=(weighted_reward,))

    @_make_hashable
    @cache
    def _accumulate(
            self,
            k: int,
            end_times: Sequence[float],
            rewards: Sequence[Reward] = None
    ) -> np.ndarray:
        """
        Evaluate the kth (non-central) moment at different end times.

        :param k: The order of the moment.
        :param end_times: Sequence of ends times or end time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :return: The moment accumulated at the specified times or time.
        """
        # use default reward if not specified
        if rewards is None:
            rewards = (self.reward,) * k
        elif len(rewards) != k:
            raise ValueError(f"Number of rewards must be {k}.")

        end_times = np.array(end_times)

        # flattening takes precedence over the closed form (it shrinks the state space, which dominates the cost)
        if self._flattening_applies(k):
            self._logger.debug("accumulate (k=%d): flattened block-counting", k)
            return self._accumulate_flattened(k, end_times, rewards)

        # closed-form evaluation of the moment to absorption (signalled by an infinite end time): the final
        # unbounded epoch is solved directly instead of exponentiating over the estimated absorption time
        if Settings.closed_form_last_epoch and end_times.size == 1 and np.isinf(end_times.flat[0]):
            self._logger.debug("accumulate (k=%d): closed-form last epoch", k)
            return np.array([self._accumulate_closed_form(k, rewards)])

        # check for negative values
        if np.any(end_times < 0):
            raise ValueError("Negative end times are not allowed.")

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(end_times)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get state space for the first epoch
        self.state_space.update_epoch(epoch)

        # number of states
        n_states = self.state_space.k

        # for large (sparse) state spaces, compute the moment via the action of the matrix exponential on a vector
        # (threading through the epochs) instead of forming the dense Van Loan propagator
        if (k + 1) * n_states >= Settings.expm_action_min_dim:
            self._logger.debug(
                "accumulate (k=%d): sparse matrix-exponential action (Van Loan dim %d >= %d)",
                k, (k + 1) * n_states, Settings.expm_action_min_dim
            )
            return self._accumulate_action(k, end_times, t_sorted, rewards)

        self._logger.debug("accumulate (k=%d): dense Van Loan matrix exponential (dim %d)", k, (k + 1) * n_states)

        # initialize block matrix holding (rewarded) moments
        Q = np.eye(n_states * (k + 1))
        u_prev = 0

        # initialize probabilities
        moments = np.zeros_like(t_sorted, dtype=float)

        # regularization parameter
        lamb = self._get_regularization_factor(self.state_space.S)

        # regularized intensity matrix
        S = self._dense_rate_matrix() * lamb

        # check numerical stability
        self._check_numerical_stability(S, 0)

        # get reward matrix
        R = [np.diag(r._get(state_space=self.state_space)) for r in rewards]

        # get Van Loan matrix
        V = self._get_van_loan_matrix(S=S, R=R, k=k)

        # The Van Loan exponential is evaluated over the absorption time, which scales with Ne (the doubling search
        # in ``_get_absorption_time`` deliberately spans many orders of magnitude). For a large time the dense
        # ``expm`` can transiently over/underflow inside scipy's scaling-squaring on some BLAS builds, even though
        # the regularized result (corrected by ``lamb ** k``) is finite. The benign intermediate over/divide/invalid
        # is silenced here and the *output* is checked for finiteness below, so a genuine blow-up still surfaces.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
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
                    S = self._dense_rate_matrix() * lamb
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

        # the suppressed intermediate over/underflow must not have corrupted the (finite) result
        if not np.isfinite(moments).all():
            self._logger.warning(
                "Non-finite values encountered when computing moments. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

    def _accumulate_action(
            self,
            k: int,
            end_times: np.ndarray,
            t_sorted: np.ndarray,
            rewards: Sequence[Reward]
    ) -> np.ndarray:
        """
        Sparse-action variant of :meth:`_accumulate` for large state spaces. Instead of forming the dense Van Loan
        propagator ``Q = prod_i exp(V_i tau_i)`` and reading off ``alpha @ Q[:n, -n:] @ e``, this threads the vector
        ``w = alpha_ext`` through the epochs via the action of the matrix exponential on the (sparse) Van Loan
        matrix (``scipy.sparse.linalg.expm_multiply``), reading off ``w @ e_ext`` at each end time. This is exact
        (a product applied to a vector is a sequence of matrix-vector actions) and exploits the rate matrix sparsity.

        :param k: The order of the moment.
        :param end_times: The (unsorted) end times, used to restore the original order.
        :param t_sorted: The sorted end times.
        :param rewards: Sequence of k rewards.
        :return: The moment accumulated at the specified times.
        """
        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)
        self.state_space.update_epoch(epoch)

        n = self.state_space.k
        lamb = self._get_regularization_factor(self.state_space.S)

        def transposed_van_loan() -> 'sp.spmatrix':
            """Transposed sparse Van Loan matrix for the current epoch (transposed for the left vector action)."""
            S = self.state_space.S * lamb
            self._check_numerical_stability(S, i_epoch)
            r_vecs = [np.asarray(r._get(state_space=self.state_space), dtype=float) for r in rewards]
            return self._get_van_loan_matrix_sparse(R=r_vecs, S=sp.csr_matrix(S), k=k).T.tocsr()

        Vt = transposed_van_loan()

        # w = alpha_ext (alpha in the first block); e_ext = e in the last block, so w @ Q @ e_ext = alpha @ Q[:n,-n:] @ e
        w = np.zeros((k + 1) * n)
        w[:n] = self.state_space.alpha
        e_ext = np.zeros((k + 1) * n)
        e_ext[-n:] = self.state_space.e

        moments = np.zeros_like(t_sorted, dtype=float)
        u_prev = 0.0

        for i, u in enumerate(t_sorted):

            # advance through whole epochs between u_prev and u
            while u > epoch.end_time:
                w = Backend.expm_multiply(Vt * ((epoch.end_time - u_prev) / lamb), w)
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)
                Vt = transposed_van_loan()

            # remaining time in the current epoch
            w = Backend.expm_multiply(Vt * ((u - u_prev) / lamb), w)
            moments[i] = factorial(k) * lamb ** k * float(w @ e_ext)
            u_prev = u

        moments = moments[np.argsort(end_times)]

        if np.isnan(moments).any():
            self._logger.warning(
                "NaN values encountered when computing moments via the matrix-exponential action. "
                f"Epoch: {i_epoch} at time: {epoch.start_time}. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return moments

    def _get_epochs_until_unbounded(self) -> List['Epoch']:
        """
        Materialize the demographic epochs up to and including the final, unbounded epoch (``end_time == inf``).

        :return: List of epochs, the last of which is unbounded.
        """
        epochs = []
        for epoch in self.demography.epochs:
            epochs.append(epoch)
            if epoch.end_time == np.inf:
                break
        return epochs

    def _absorption_certain_in_last_epoch(self) -> bool:
        """
        Structural check on the final (unbounded) epoch: whether every transient state can reach an absorbing
        state, i.e. the transient sub-generator is non-singular and the moment-to-absorption can be evaluated in
        closed form. When this is ``False`` (e.g. disconnected demes or a migration barrier in the last epoch) the
        moment may still be finite if absorption occurs in earlier epochs, so callers fall back to the
        matrix-exponential path rather than relying on the closed form.

        :return: Whether absorption is certain from every transient state of the last epoch.
        """
        # the result depends only on the (fixed) last-epoch structure, so memoize it: the closed form queries this
        # once per moment, and an SFS/jSFS evaluates many bins, so recomputing the reachability each time dominated.
        if getattr(self, '_absorption_certain_cache', None) is not None:
            return self._absorption_certain_cache

        self.state_space.update_epoch(self._get_epochs_until_unbounded()[-1])
        absorbing, reach = self._reaches_absorption()

        self._absorption_certain_cache = bool(reach[~absorbing].all())
        return self._absorption_certain_cache

    def _reaches_absorption(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward reachability over the *current* epoch's rate graph: which states can reach an absorbing state.
        A state can reach absorption iff it is absorbing or has an outgoing edge (rate ``S[i, j] > 0``) to a state
        that can; this is propagated backwards with a sparse adjacency, so each pass is O(nnz). Used both to decide
        whether the closed form applies (:meth:`_absorption_certain_in_last_epoch`) and to guard against demographies
        that never absorb (:meth:`_get_absorption_time`).

        :return: ``(absorbing, reach)`` boolean masks over the states; ``reach`` includes the absorbing states.
        """
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])

        S = self.state_space.S
        if sp.issparse(S):
            adj = S.tocsr(copy=True)
            adj.setdiag(0)
            adj.eliminate_zeros()
        else:
            adj = sp.csr_matrix((S - np.diag(np.diag(S))) > 0)

        reach = absorbing.copy()
        while True:
            nxt = absorbing | (adj @ reach > 0)
            if np.array_equal(nxt, reach):
                break
            reach = nxt

        return absorbing, reach

    def _assert_absorbs(self, T: np.ndarray):
        """
        Raise if the demography can never absorb. Distinguishes a *structural* barrier (an isolated deme or a
        one-way/blocked migration in the final, unbounded epoch, leaving lineages that can never coalesce) from a
        merely slow or numerically imprecise computation. ``T`` is the transition matrix integrated to a large time,
        so ``alpha @ T`` is the occupation distribution there and its support is exactly the mass still in play; the
        final epoch's rate graph (``state_space`` is expected to be updated to it) tells us which states can
        structurally reach absorption. Residual mass parked on states that cannot is permanent. Shared by the
        absorption-time and quantile searches, both of which otherwise silently run to their iteration ceiling.

        :param T: Transition matrix integrated from time 0 to a large time in the final, unbounded epoch.
        :raises ValueError: if a non-negligible fraction of the mass can never reach a common ancestor.
        """
        _, reach = self._reaches_absorption()
        stuck = float((self.state_space.alpha @ T)[~reach].sum())

        if stuck > 1e-8:
            raise ValueError(
                f"The demography does not absorb: a fraction {stuck:.2e} of the probability mass remains on "
                "states that can never reach a common ancestor, so there is no almost-sure absorption time. "
                "This typically means a deme is isolated or migration is one-way/blocked in the final "
                "(unbounded) epoch, leaving lineages that can never coalesce. Check the migration structure "
                "of the last epoch."
            )

    def _accumulate_closed_form(self, k: int, rewards: Sequence[Reward]) -> float:
        """
        Evaluate the kth (non-central) moment accumulated until absorption, evaluating the final unbounded epoch in
        closed form. The final epoch's contribution to ``t -> inf`` is the limit ``z = lim_t exp(V t) e_ext`` of the
        Van Loan propagator, whose transient part is the back-substitution ``nu_j = (-T)^{-1} R_j nu_{j+1}`` (with
        ``nu_k`` the exit vector) and whose absorbing part is the exit vector in the last block. The preceding finite
        epochs are applied to ``z`` via the (well-conditioned, finite-interval) matrix exponential of the full Van
        Loan matrix. This is for a single reward ordering; permutation averaging is handled by :meth:`accumulate`.

        :param k: The order of the moment.
        :param rewards: Sequence of k rewards (a single ordering).
        :return: The kth moment accumulated until absorption.
        """
        self._check_demography_conditioning()

        epochs = self._get_epochs_until_unbounded()
        n = self.state_space.k

        # --- final, unbounded epoch: limit vector z ---
        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, len(epochs) - 1)
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        idx_t = np.where(~absorbing)[0]
        idx_a = np.where(absorbing)[0]
        e = np.asarray(self.state_space.e)

        # The closed form factors the transient sub-generator ``T`` (size = number of transient states), whose
        # dense-LU vs sparse-LU crossover sits at ``Settings.closed_form_sparse_min_states`` transient states. This is a different
        # quantity from the Van Loan dimension that governs the matrix-exponential path (:attr:`expm_action_min_dim`):
        # the LU only ever sees ``T``, independent of the moment order, so the threshold is on ``len(idx_t)`` alone.
        use_action = len(idx_t) >= Settings.closed_form_sparse_min_states

        # transient sub-generator and its (sparse or dense) factorization, reused across the back-substitution
        T = self._transient_block(idx_t, sparse=use_action)
        if use_action:
            self._logger.debug(
                "closed form (k=%d): sparse LU (splu) of T (n_t=%d >= %d), %d finite epoch(s)",
                k, len(idx_t), Settings.closed_form_sparse_min_states, len(epochs) - 1
            )
            solver = spla.splu(sp.csc_matrix(-T))
            solve = solver.solve
        else:
            self._logger.debug(
                "closed form (k=%d): dense LU of T (n_t=%d), %d finite epoch(s)", k, len(idx_t), len(epochs) - 1
            )
            lu = sla.lu_factor(-T)
            solve = lambda b: sla.lu_solve(lu, b)

        # reward diagonals restricted to the transient states (the off-diagonal Van Loan reward blocks are diagonal)
        r_t = [np.asarray(r._get(self.state_space), dtype=float)[idx_t] for r in rewards]

        nu = [None] * (k + 1)
        nu[k] = e[idx_t]
        for j in range(k - 1, -1, -1):
            nu[j] = solve(r_t[j] * nu[j + 1])

        z = np.zeros((k + 1) * n)
        for j in range(k + 1):
            z[j * n + idx_t] = nu[j]
        z[k * n + idx_a] = e[idx_a]

        # --- preceding finite epochs, backward, via the (sparse or dense) full Van Loan matrix exponential ---
        for i_epoch, epoch in reversed(list(enumerate(epochs[:-1]))):
            self.state_space.update_epoch(epoch)
            S = self.state_space.S
            self._check_numerical_stability(S, i_epoch)
            tau = epoch.end_time - epoch.start_time

            if use_action:
                r_vecs = [np.asarray(r._get(self.state_space), dtype=float) for r in rewards]
                S_csr = S.tocsr() if sp.issparse(S) else sp.csr_matrix(np.asarray(S))
                V = self._get_van_loan_matrix_sparse(R=r_vecs, S=S_csr, k=k)
                z = Backend.expm_multiply(V * tau, z)
            else:
                S_dense = np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)
                R = [np.diag(r._get(self.state_space)) for r in rewards]
                V = self._get_van_loan_matrix(S=S_dense, R=R, k=k)
                z = expm(V * tau) @ z

        alpha_ext = np.zeros((k + 1) * n)
        alpha_ext[:n] = self.state_space.alpha

        return factorial(k) * float(alpha_ext @ z)

    def _flattening_applies(self, k: int) -> bool:
        """
        Whether the block-counting state space can be flattened to the (much smaller) lineage-counting state space
        for this moment: the first moment of the standard coalescent on a single population and a single locus. When
        it applies it takes precedence over the closed form / batched occupation, because reducing the state space
        (e.g. thousands of block states to ``n`` lineage states) dominates the per-solve cost.
        """
        return (
                Settings.flatten_block_counting and
                k == 1 and
                isinstance(self.state_space, BlockCountingStateSpace) and
                isinstance(self.state_space.model, StandardCoalescent) and
                self.lineage_config.n_pops == 1 and
                self.locus_config.n == 1
        )

    def _transient_block(self, idx_t: np.ndarray, sparse: bool = False):
        """
        The transient sub-generator ``T = S[idx_t, idx_t]`` extracted from the (dense or sparse) rate matrix,
        returned as a dense array (default) or a sparse CSC matrix (``sparse=True``, for the large-state-space LU /
        exp-action paths, which never materialise the dense block).
        """
        S = self.state_space.S
        if sp.issparse(S):
            sub = S[idx_t][:, idx_t]
            return sub.tocsc() if sparse else np.asarray(sub.todense())
        sub = np.asarray(S)[np.ix_(idx_t, idx_t)]
        return sp.csc_matrix(sub) if sparse else sub

    def _dense_rate_matrix(self) -> np.ndarray:
        """
        The full rate matrix as a dense array (densifying if it is stored sparse). Used by the dense moment paths,
        which are only taken for state spaces small enough that a dense matrix is cheap.
        """
        S = self.state_space.S
        return np.asarray(S.todense()) if sp.issparse(S) else np.asarray(S)

    def _occupation_times(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Expected total time spent in each transient state until absorption. This is the bin-independent quantity
        shared by every bin of a *mean* spectrum: the mean of a reward ``r`` is simply ``occupation . r``, so a whole
        SFS / joint SFS mean is one contraction ``occupation @ R`` over the stacked bin rewards instead of a separate
        solve per bin. Finite epochs contribute ``p_i (exp(S_i tau_i) - I) S_i^{-1}`` (entered with distribution
        ``p_i``); the final unbounded epoch contributes ``p (-T)^{-1}``.

        :return: ``(occupation, idx_t)`` with the occupation times over the transient states ``idx_t`` of the final
            epoch, or ``None`` if absorption is not almost sure (callers then fall back to per-bin evaluation).
        """
        if not self._absorption_certain_in_last_epoch():
            return None

        epochs = self._get_epochs_until_unbounded()

        self.state_space.update_epoch(epochs[-1])
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        idx_t = np.where(~absorbing)[0]
        nt = len(idx_t)
        use_action = nt >= Settings.closed_form_sparse_min_states

        p = np.asarray(self.state_space.alpha)[idx_t].astype(float)
        m = np.zeros(nt)

        self._logger.debug(
            "occupation times (batched mean): %s factorization, n_t=%d, %d finite epoch(s)",
            "sparse" if use_action else "dense", nt, len(epochs) - 1
        )

        # finite epochs: accumulate the within-epoch occupation and propagate the entry distribution. The occupation
        # integral ``A = int_0^tau exp(S t) dt`` is read off the augmented (Van Loan) generator ``[[S, I], [0, 0]]``,
        # which is robust even when the finite-epoch block ``S`` is singular (e.g. a migration barrier), unlike
        # ``(exp(S tau) - I) S^-1``. Only the row-action ``[p, 0] exp(aug tau) = [p exp(S tau), p A]`` is needed (the
        # propagated entry distribution and the occupation increment ``p A`` at once), so for large state spaces apply
        # the sparse matrix-exponential action instead of forming the dense ``2 nt x 2 nt`` exponential.
        for epoch in epochs[:-1]:
            self.state_space.update_epoch(epoch)
            self._check_numerical_stability(self.state_space.S, 0)
            S = self._transient_block(idx_t, sparse=use_action)
            tau = epoch.end_time - epoch.start_time
            if use_action:
                aug = sp.bmat([
                    [sp.csc_matrix(S), sp.identity(nt, format='csc')],
                    [None, sp.csc_matrix((nt, nt))]
                ], format='csc')
                # [p, 0] exp(aug tau) = (exp((aug tau)^T) [p; 0])^T, so apply the action to the transposed generator
                w = spla.expm_multiply((aug * tau).T.tocsc(), np.concatenate([p, np.zeros(nt)]))
                m += w[nt:]
                p = w[:nt]
            else:
                aug = np.zeros((2 * nt, 2 * nt))
                aug[:nt, :nt] = S
                aug[:nt, nt:] = np.eye(nt)
                exp_aug = expm(aug * tau)
                m += p @ exp_aug[:nt, nt:]
                p = p @ exp_aug[:nt, :nt]

        # final unbounded epoch: occupation = p (-T)^{-1}, i.e. solve (-T)^T x = p
        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, len(epochs) - 1)
        neg_t = -self._transient_block(idx_t, sparse=use_action)
        if use_action:
            m += spla.splu(sp.csc_matrix(neg_t.T)).solve(p)
        else:
            m += sla.solve(neg_t.T, p)

        return m, idx_t

    def _two_point_occupation(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Two-point occupation matrix ``K_{a,b} = int_{s<u} P(X_s = a, X_u = b) ds du`` — the bin-independent quantity
        shared by every *pair* of a second-moment spectrum: the uncentered cross-moment of two rewards ``r, r'`` is
        ``r^T (K + K^T) r'``, so the whole 2-SFS covariance is a single contraction over the stacked bin rewards.

        Restricted to a **single (unbounded) epoch**, where it is the exact closed form ``K = diag(m) (-T)^{-1}``
        (``m = alpha (-T)^{-1}`` the occupation times) and needs no numerical integration. The multi-epoch version
        requires integrating the ``O(n_states^2)`` matrix ODE ``dJ/du = J S + diag(f(u))``, whose explicit
        integrator degenerates (very many tiny steps) on stiff demographies, so it is deliberately not used: the
        caller falls back to the per-pair matrix-exponential path instead.

        :return: ``(K, idx_t)`` over the transient states, or ``None`` when not applicable (caller falls back).
        """
        if not (Settings.closed_form_last_epoch and self.tree_height.end_time is None):
            return None

        epochs = self._get_epochs_until_unbounded()

        # only the single-epoch closed form is used; the multi-epoch ODE is stiffness-fragile (see docstring)
        if len(epochs) > 1:
            self._logger.debug(
                "two-point occupation: %d epochs; using per-pair matrix-exponential (multi-epoch closed form "
                "disabled)", len(epochs)
            )
            return None

        if not self._absorption_certain_in_last_epoch():
            return None

        self.state_space.update_epoch(epochs[-1])
        self._check_numerical_stability(self.state_space.S, 0)
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        idx_t = np.where(~absorbing)[0]

        neg_t_inv = sla.inv(-self._transient_block(idx_t))
        m = np.asarray(self.state_space.alpha)[idx_t].astype(float) @ neg_t_inv

        self._logger.debug("two-point occupation: single-epoch closed form diag(m)(-T)^-1 (n_t=%d)", len(idx_t))

        return np.diag(m) @ neg_t_inv, idx_t

    def _sample(
            self,
            n_samples: int,
            rewards: Sequence[Reward] = None,
            record_visits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate samples from the mean reward distribution by simulating trajectories.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is the tree height reward.
        :param record_visits: Whether to record which states were visited during the sampling.
        :return: Array of sampled rewards of size (n_samples, len(rewards)),
                 and optionally an array of probabilities of visiting each state.
        """
        if rewards is None:
            rewards = [self.reward]

        n_rewards = len(rewards)
        samples = np.zeros((n_samples, n_rewards))
        absorbing = np.array([s.is_absorbing() for s in self.state_space.states])
        alpha = self.state_space.alpha
        states_visited = np.zeros_like(alpha)
        R = np.array([r._get(self.state_space) for r in rewards])

        # iterate over samples
        for i in tqdm(range(n_samples), disable=not Settings.use_pbar):
            mass = np.zeros(n_rewards)
            t = 0
            rate = 0
            state = np.random.choice(len(alpha), p=alpha)
            epochs = self.demography.epochs
            traj_probs = [] if record_visits else None

            try:
                # find first non-zero rate epoch
                while rate == 0:
                    epoch = next(epochs)
                    self.state_space.update_epoch(epoch)
                    rate = -self.state_space.S[state, state]

                # sample next time step
                dt = np.random.exponential(1 / rate)

                # iterate over transitions
                while True:

                    # iterate over epochs
                    while t + dt >= epoch.end_time:
                        # reward until epoch boundary
                        mass += R[:, state] * (epoch.end_time - t)
                        dt -= (epoch.end_time - t)
                        t = epoch.end_time

                        # advance epoch
                        epoch = next(epochs)
                        self.state_space.update_epoch(epoch)

                        new_rate = -self.state_space.S[state, state]
                        if new_rate == 0:
                            t = epoch.end_time
                            continue

                        # rescale remaining time
                        dt *= rate / new_rate
                        rate = new_rate

                    # step completes in current epoch
                    mass += R[:, state] * dt
                    t += dt

                    # sample next state
                    probs = self.state_space.S[state].copy()
                    probs[state] = 0
                    state = np.random.choice(len(probs), p=probs / rate)

                    states_visited[state] += 1

                    if absorbing[state]:
                        raise StopIteration

                    rate = -self.state_space.S[state, state]

                    # if rate is zero, we skip to the next epoch
                    if rate == 0:
                        t = epoch.end_time
                        continue

                    # sample next time step
                    dt = np.random.exponential(1 / rate)

            except StopIteration:
                pass

            samples[i] = mass

        # normalize states visited
        states_visited /= n_samples

        return (samples, states_visited) if record_visits else samples

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

        from ..visualization import Visualization

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
                T @= expm(self._dense_rate_matrix() * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

            self._check_numerical_stability(self.state_space.S, i_epoch)

            # update transition matrix with remaining time in current epoch
            T @= expm(self._dense_rate_matrix() * (u - u_prev))

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
            T = T @ expm(self._dense_rate_matrix() * tau)
            u_prev = epoch.end_time

            # fetch and update for next epoch
            epoch = self.demography.get_epoch(epoch.end_time)
            self.state_space.update_epoch(epoch)
        else:
            # update transition matrix
            T = T @ expm(self._dense_rate_matrix() * (u - u_prev))

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

        # finite upper bound for the search: the time of almost-sure absorption (any quantile q < 1 lies below it).
        # This also guards against a demography that never absorbs — ``_get_absorption_time`` raises in that case —
        # and keeps the expansion below from doubling ``b`` to an overflow-inducing ceiling. A user-supplied end
        # time bounds the (necessarily proper) distribution instead.
        b_max = self.end_time if self.end_time is not None else self._get_absorption_time()

        # initialize bounds
        a, b = 0, 1

        T_a = np.eye(self.state_space.k)
        epoch_a, epoch_b = self.demography.get_epoch(0), self.demography.get_epoch(0)
        b, T_b, epoch_b = self._update(min(b, b_max), a, T_a, epoch_b)

        i = 0

        # expand the upper bound until its CDF reaches q (bounded by the absorption time, so it always terminates)
        while self._cum(T_b) < q and b < b_max and i < max_iter:
            b, T_b, epoch_b = self._update(min(b * expansion_factor, b_max), b, T_b, epoch_b)

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
        we have a good idea about how likely absorption is, and can warn the user if necessary.
        Stopping the computation when no more rewards are accumulated is not a good idea, as this
        can happen before almost sure absorption (exponential runaway growth, temporary isolation in different demes).
        """
        i = 0
        T = np.eye(self.state_space.k)
        epoch = self.demography.get_epoch(0)

        self._check_demography_conditioning()

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

        # if absorption was not reached, fail loudly for a demography that *never* absorbs rather than returning the
        # doubling ceiling (see :meth:`_assert_absorbs`).
        if p < self.p_absorption and not np.isnan(p):
            self._assert_absorbs(T)

        if i - 1 == self.max_iter:
            self._logger.warning(
                "Could not reliably find time of almost sure absorption after maximum number of iterations. "
                f"Using time {t:.1f} with probability of absorption 1 - {1 - p:.1e}. "
                "This could be due to numerical imprecision, unreachable states or very large or small "
                "absorption times. You can set the end time manually (see `Coalescent.end_time`) or increase "
                "the maximum number of iterations (`TreeHeightDistribution.max_iter`)."
            )

        return t

    def _empirical_cdf(self, n_samples: int, reward: Reward = None, t: float | Sequence[float] = None) -> np.ndarray:
        """
        Generate an empirical cumulative distribution function (CDF) by sampling from the distribution.

        :param n_samples: Number of samples to generate.
        :param reward: Reward function to use for sampling. If not specified,
            the default reward of the distribution is used.
        :param t: Values at which to evaluate the CDF. Default to 100 evenly spaced values
            between 0 and the 99th percentile.
        :return: Sorted array of sampled total rewards.
        """
        if t is None:
            t = np.linspace(0, self.tree_height.quantile(0.99), 100)

        samples = self._sample(n_samples, reward).reshape(n_samples)

        x = np.sort(samples)
        y = np.arange(1, n_samples + 1) / n_samples

        if x.ndim == 1:
            return np.interp(t, x, y)

    def _plot_empirical_cdf(
            self,
            n_samples: int = 1000,
            reward: Reward = None,
            t: float | Sequence[float] = None,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Empirical CDF'
    ) -> 'plt.Axes':
        """
        Plot the empirical cumulative distribution function (CDF).

        :param n_samples: Number of samples to generate.
        :param reward: Reward function to use for sampling. If not specified,
            the default reward of the distribution is used.
        :param t: Values at which to evaluate the CDF. Default to 100 evenly spaced values
            between 0 and the 99th percentile.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if t is None:
            t = np.linspace(0, self.tree_height.quantile(0.99), 100)

        y = self._empirical_cdf(n_samples, reward, t)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=y,
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )


class SFSDistribution(PhaseTypeDistribution, ABC):
    """
    Base class for site-frequency spectrum distributions.
    """

    def __init__(
            self,
            state_space: BlockCountingStateSpace,
            tree_height: TreeHeightDistribution,
            demography: Demography,
            reward: Reward = None
    ):
        """
        Initialize the distribution.

        :param state_space: Block-counting state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward to multiply the SFS reward with. By default, the unit reward is used, which
            has no effect.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward
        )

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

        # batched mean: every bin's mean is ``occupation . r_bin`` with the same occupation-time vector, so the whole
        # spectrum is one contraction instead of a per-bin solve. This is the closed form's spectrum path (it shares
        # the transient solve across bins); only for the plain mean to absorption (k=1, default reward, no custom
        # accumulation window) and when flattening does not apply (flattening reduces the state space and wins).
        # Other cases fall through to the per-bin path.
        if (
                Settings.closed_form_last_epoch and
                not self._flattening_applies(k) and
                k == 1 and
                start_time is None and
                end_time is None and
                self.tree_height.end_time is None and
                rewards == (self.reward,)
        ):
            occupation = self._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
                base = np.asarray(self.reward._get(self.state_space), dtype=float)
                R = np.column_stack([
                    (base * np.asarray(self._get_sfs_reward(i)._get(self.state_space), dtype=float))[idx_t]
                    for i in self._get_indices()
                ])
                moments = m @ R
                return SFS([0] + list(moments) + [0] * (self.lineage_config.n - len(moments)))

        # moment of each SFS bin (serial; performance-critical paths use the batched closed form above)
        moments = np.array([
            self._moment(k, i, rewards, start_time, end_time, center, permute)
            for i in self._get_indices()
        ])

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

        accumulation = np.array([
            self.get_accumulation(k, i, end_times, rewards)
            for i in indices
        ])

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
        from ..visualization import Visualization

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

    def _cov_batched(self) -> Optional[SFS2]:
        """
        Batched 2-SFS: all ``O(n^2)`` bin pairs share one two-point occupation operator ``K`` (see
        :meth:`_two_point_occupation`), so the whole covariance is ``cov = R^T (K + K^T) R - outer(mean)`` via a
        single contraction over the stacked bin rewards instead of a cross-moment per pair.

        :return: The covariance, or ``None`` when not applicable (closed form disabled, explicit end time, or
            absorption not almost sure) so the caller falls back to the per-pair path.
        """
        if not Settings.closed_form_last_epoch:
            return None

        two_point = self._two_point_occupation()
        if two_point is None:
            return None

        K, idx_t = two_point
        ss = self.state_space
        base = np.asarray(self.reward._get(ss), dtype=float)
        indices = self._get_indices()
        R = np.column_stack([
            (base * np.asarray(self._get_sfs_reward(i)._get(ss), dtype=float))[idx_t] for i in indices
        ])

        sfs_matrix = R.T @ K @ R                       # R^T K R (one ordering)
        self._logger.debug("sfs.cov: centering with the outer product of bin means")
        mean = np.asarray(self.mean.data)[indices]
        cov = (sfs_matrix + sfs_matrix.T) - np.outer(mean, mean)

        out = np.zeros((self.lineage_config.n + 1, self.lineage_config.n + 1))
        for a, ia in enumerate(indices):
            out[ia, indices] = cov[a]
        return SFS2(out)

    @cached_property
    def cov(self) -> SFS2:
        """
        Covariance matrix across site-frequency counts.
        """
        batched = self._cov_batched()
        if batched is not None:
            self._logger.debug("sfs.cov: batched (shared two-point occupation)")
            return batched

        # create list of arguments for each combination of i, j
        indices = [(i, j) for i in self._get_indices() for j in self._get_indices()]

        self._logger.debug("sfs.cov: per-pair matrix exponential over %d bin pairs", len(indices))

        # cross-moment of each bin pair (serial)
        sfs_results = [
            PhaseTypeDistribution.moment(self, k=2, permute=False, center=False, rewards=(
                CombinedReward([self.reward, self._get_sfs_reward(i)]),
                CombinedReward([self.reward, self._get_sfs_reward(j)])
            ))
            for i, j in indices
        ]

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

        # monomorphic bins have zero variance; the resulting NaNs from dividing by a zero std are expected and
        # replaced with zeros below, so silence the benign divide warning at the source.
        with np.errstate(divide='ignore', invalid='ignore'):
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
            threshold. More complex demographic models, larger sample sizes, and higher mutation rates all increase
            the number of generated configurations necessary to reach a certain mass.

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


class TajimaSFSMixin:
    """
    Mixin providing the branch-length diversity estimators and Tajima's :math:`D` from the site-frequency
    spectrum mean and covariance. Shared by the analytical :class:`UnfoldedSFSDistribution` and the
    simulation-based empirical SFS distribution, so the same statistics can be computed from either source.
    Subclasses supply :meth:`_tajima_n`, :meth:`_tajima_mean` and :meth:`_tajima_cov`.
    """

    def _tajima_n(self) -> int:
        """Number of lineages."""
        raise NotImplementedError

    def _tajima_mean(self) -> np.ndarray:
        """Mean branch length per polymorphic SFS bin (``i = 1 .. n-1``)."""
        raise NotImplementedError

    def _tajima_cov(self) -> np.ndarray:
        """Covariance of the polymorphic SFS bins (``i, j = 1 .. n-1``)."""
        raise NotImplementedError

    @cached_property
    def _tajima_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-bin weights for the two diversity estimators: pairwise diversity ``pi`` and Watterson's ``theta_W``."""
        n = self._tajima_n()
        i = np.arange(1, n)
        w_pi = 2 * i * (n - i) / (n * (n - 1))
        w_w = np.full(n - 1, 1 / np.sum(1 / i))

        return w_pi, w_w

    @cached_property
    def theta_pi(self) -> float:
        r"""
        Mean pairwise diversity :math:`\pi = \sum_i \frac{2 i (n - i)}{n (n - 1)} \mathbb{E}[L_i]`, the branch-length
        estimator of :math:`\theta` based on the expected number of pairwise differences.
        """
        w_pi, _ = self._tajima_weights

        return float(w_pi @ self._tajima_mean())

    @cached_property
    def theta_w(self) -> float:
        r"""
        Watterson's estimator :math:`\theta_W = L_\text{total} / a_n` with :math:`a_n = \sum_{k=1}^{n-1} 1/k`, the
        branch-length estimator of :math:`\theta` based on the total branch length.
        """
        _, w_w = self._tajima_weights

        return float(w_w @ self._tajima_mean())

    @cached_property
    def tajimas_d(self) -> float:
        r"""
        Tajima's :math:`D` in branch form: :math:`D = (\pi - \theta_W) / \sqrt{c^\top \, \mathrm{Cov}[L] \, c}`
        with weights :math:`c_i = \frac{2 i (n - i)}{n (n - 1)} - 1/a_n`. It is ``0`` under the standard neutral
        constant-size model, negative under population growth (excess of low-frequency variants) and positive under
        contraction. The normalization uses the branch-length covariance rather than the mutation-based variance of
        the classical sample estimator.
        """
        w_pi, w_w = self._tajima_weights
        c = w_pi - w_w

        num = c @ self._tajima_mean()
        var = c @ self._tajima_cov() @ c

        if var <= 0:
            return 0.0

        return float(num / np.sqrt(var))


class UnfoldedSFSDistribution(SFSDistribution, TajimaSFSMixin):
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

    def _tajima_n(self) -> int:
        return self.lineage_config.n

    def _tajima_mean(self) -> np.ndarray:
        n = self.lineage_config.n
        return np.asarray(self.mean.data)[1:n]

    def _tajima_cov(self) -> np.ndarray:
        n = self.lineage_config.n
        return np.asarray(self.cov.data)[1:n, 1:n]

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


class JointSFSDistribution(PhaseTypeDistribution):
    """
    Joint (multi-population) site-frequency spectrum distribution.

    Moments are returned as a multi-dimensional array of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, where ``n_p`` is the
    sample size of population ``p``. The entry at index ``(k_0, ..., k_{P-1})`` is the moment for branches subtending
    exactly ``k_p`` samples from population ``p``. The monomorphic bins (the all-zero and the full
    ``(n_0,...,n_{P-1})`` configuration) are zero by convention.
    """

    def __init__(
            self,
            state_space: JointBlockCountingStateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography,
            reward: Reward = None
    ):
        """
        Initialize the distribution.

        :param state_space: Joint block-counting state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward to multiply the joint SFS reward with. By default, the unit reward is used, which
            has no effect.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward
        )

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the joint SFS array, ``(n_0 + 1, ..., n_{P-1} + 1)``.
        """
        return tuple(int(n_p) + 1 for n_p in self.lineage_config.lineages)

    def _get_configs(self) -> List[Tuple[int, ...]]:
        """
        Get the descendant vectors corresponding to (polymorphic) joint SFS bins, i.e. all block configurations
        except the full-sample configuration (which corresponds to the monomorphic, fixed sites).

        :return: List of descendant vectors.
        """
        full = tuple(int(n_p) for n_p in self.lineage_config.lineages)

        return [c for c in self.state_space.block_configs if c != full]

    def moment(
            self,
            k: int,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        """
        Get the kth moments of the joint site-frequency spectrum.

        :param k: The order of the moment.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :return: An array of shape :attr:`shape` holding the kth moment of each joint SFS bin.
        """
        # batched mean: all joint bins share one occupation-time vector, so the whole joint SFS mean is a single
        # contraction over the stacked bin rewards (closed form's spectrum path). Only for the plain mean to
        # absorption; other cases fall through to the per-bin accumulation.
        if (
                Settings.closed_form_last_epoch and
                int(k) == 1 and
                start_time is None and
                end_time is None and
                self.tree_height.end_time is None
        ):
            occupation = self._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
                base = np.asarray(self.reward._get(self.state_space), dtype=float)
                configs = self._get_configs()
                R = np.column_stack([
                    (base * np.asarray(JointSFSReward(config)._get(self.state_space), dtype=float))[idx_t]
                    for config in configs
                ])
                values = m @ R
                out = np.zeros(self.shape)
                for config, value in zip(configs, values):
                    out[config] = value
                return JointSFS(out, pop_names=self.lineage_config.pop_names)

        # like the base distribution, a moment is the accumulation over the [start_time, end_time] window
        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            # evaluate the moment to absorption: signal the closed-form path with an infinite end time when it
            # applies (no explicit end time, accumulation from 0, and absorption certain in the last epoch), but not
            # when flattening applies (which takes precedence and delegates to the smaller lineage-counting space),
            # otherwise use the estimated absorption time
            if (
                    Settings.closed_form_last_epoch and
                    not self._flattening_applies(k) and
                    start_time == 0 and
                    self.tree_height.end_time is None and
                    self._absorption_certain_in_last_epoch()
            ):
                end_time = np.inf
            else:
                end_time = self.tree_height.t_max

        if start_time > 0:
            acc = self.accumulate(k, [start_time, end_time], center=center, permute=permute)
            out = acc[..., 1] - acc[..., 0]
        else:
            out = self.accumulate(k, [end_time], center=center, permute=permute)[..., 0]

        if np.isnan(out).any():
            raise ValueError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return JointSFS(out, pop_names=self.lineage_config.pop_names)

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        """
        Evaluate the kth moments of the joint site-frequency spectrum at different end times.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moments.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :return: Array of shape :attr:`shape` ``+ (len(end_times),)`` with each bin's kth moment over time.
        """
        k = int(k)
        configs = self._get_configs()
        end_times = np.array(list(end_times))

        accumulation = np.array([
            PhaseTypeDistribution.accumulate(
                self,
                k=k,
                end_times=end_times,
                rewards=tuple(CombinedReward([self.reward, JointSFSReward(config)]) for _ in range(k)),
                center=center,
                permute=permute
            )
            for config in configs
        ])

        out = np.zeros(self.shape + (len(end_times),))
        for config, acc in zip(configs, accumulation):
            out[config] = acc

        return out

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot accumulation of joint SFS moments over time, one curve per (polymorphic) bin.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. Defaults to 200 points up to the 99th percentile.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from ..visualization import Visualization

        k = int(k)

        if ax is None:
            ax = plt.gca()

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

        end_times = np.asarray(list(end_times))

        if title is None:
            title = f"Joint SFS moment accumulation (order {k})"

        configs = self._get_configs()
        accumulation = self.accumulate(k, end_times, center=center, permute=permute)

        for i, config in enumerate(configs):
            Visualization.plot(
                ax=ax,
                x=end_times,
                y=accumulation[config],
                xlabel='t',
                ylabel='moment',
                label=str(config),
                file=file,
                show=(i == len(configs) - 1) and show,
                clear=clear,
                title=title
            )

        return ax

    @cached_property
    def mean(self) -> JointSFS:
        """
        Mean of the joint site-frequency spectrum, i.e. the expected branch length subtending each descendant
        configuration.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> JointSFS:
        """
        Variance of the joint site-frequency spectrum.
        """
        batched = self._cov_batched
        if batched is not None:
            configs, cov = batched
            out = np.zeros(self.shape)
            for a, config in enumerate(configs):
                out[config] = cov[a, a]
            return JointSFS(out, pop_names=self.lineage_config.pop_names)

        return self.moment(k=2, center=True)

    def get_cov(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> float:
        """
        Get the covariance between the branch lengths subtending two descendant configurations.

        :param config_a: First descendant configuration.
        :param config_b: Second descendant configuration.
        :return: The covariance.
        """
        return PhaseTypeDistribution.moment(
            self,
            k=2,
            center=True,
            rewards=tuple(CombinedReward([self.reward, JointSFSReward(c)]) for c in (config_a, config_b))
        )

    @cached_property
    def _cov_batched(self) -> Optional[Tuple[List[Tuple[int, ...]], np.ndarray]]:
        """
        Batched joint-SFS covariance: all ``O(n^{2P})`` bin pairs share one two-point occupation operator ``K``
        (see :meth:`_two_point_occupation`), so the whole covariance is ``cov = R^T (K + K^T) R - outer(mean)`` via a
        single contraction over the stacked bin rewards instead of a cross-moment per pair. Cached so that
        :attr:`cov` and :attr:`var` share the single (potentially expensive) ``K`` solve.

        :return: ``(configs, cov)`` with ``cov`` the bins-by-bins covariance over the polymorphic ``configs``, or
            ``None`` when not applicable (closed form disabled, explicit end time, or absorption not almost sure) so
            callers fall back.
        """
        if not Settings.closed_form_last_epoch:
            return None

        two_point = self._two_point_occupation()
        if two_point is None:
            return None

        K, idx_t = two_point
        ss = self.state_space
        base = np.asarray(self.reward._get(ss), dtype=float)
        configs = self._get_configs()
        R = np.column_stack([
            (base * np.asarray(JointSFSReward(config)._get(ss), dtype=float))[idx_t] for config in configs
        ])

        sfs_matrix = R.T @ K @ R                       # R^T K R (one ordering)
        self._logger.debug("jsfs.cov: centering with the outer product of bin means")
        mean = np.array([self.mean.data[config] for config in configs])
        cov = (sfs_matrix + sfs_matrix.T) - np.outer(mean, mean)

        return configs, cov

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance between the branch lengths of all pairs of (polymorphic) joint SFS bins. Returned as an array of
        shape :attr:`shape` ``+`` :attr:`shape`, where ``cov[a_0, ..., a_{P-1}, b_0, ..., b_{P-1}]`` is the covariance
        between bins ``(a_0, ..., a_{P-1})`` and ``(b_0, ..., b_{P-1})``.
        """
        batched = self._cov_batched
        if batched is not None:
            self._logger.debug("jsfs.cov: batched (shared two-point occupation)")
            configs, cov = batched
            out = np.zeros(self.shape + self.shape)
            for a, config_a in enumerate(configs):
                for b, config_b in enumerate(configs):
                    out[tuple(config_a) + tuple(config_b)] = cov[a, b]
            return out

        configs = self._get_configs()
        pairs = [(a, b) for a in configs for b in configs]

        self._logger.debug("jsfs.cov: per-pair matrix exponential over %d config pairs", len(pairs))

        results = [self.get_cov(a, b) for a, b in pairs]

        out = np.zeros(self.shape + self.shape)
        for (a, b), result in zip(pairs, results):
            out[tuple(a) + tuple(b)] = result

        return out


class TwoLocusSFSDistribution(PhaseTypeDistribution):
    """
    Two-locus site-frequency spectrum under recombination. Entry ``(i, j)`` of the (symmetrized) mean is
    ``E[L^0_i · L^1_j]`` — the expected product of the branch length subtending ``i`` samples at locus 0 and ``j``
    samples at locus 1 — computed as a second cross-moment of two per-locus SFS rewards on the two-locus
    block-counting state space. It reduces to ``Coalescent.sfs.cov`` (plus the outer product of the marginal means)
    as ``r → 0`` and to the outer product of the marginal SFS as ``r → ∞``.
    """

    def __init__(
            self,
            state_space: TwoLocusBlockCountingStateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography,
            reward: Reward = None
    ):
        """
        Initialize the distribution.

        :param state_space: Two-locus block-counting state space.
        :param tree_height: The (two-locus) tree height distribution, whose absorption time is when both loci have
            reached their MRCA.
        :param demography: The demography.
        :param reward: An optional reward to multiply the per-locus SFS rewards with. By default the unit reward.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(state_space=state_space, tree_height=tree_height, demography=demography, reward=reward)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the two-locus SFS array, ``(n + 1, n + 1)`` (one axis per locus).
        """
        n = int(self.lineage_config.n)
        return n + 1, n + 1

    def _get_indices(self) -> List[int]:
        """
        Polymorphic SFS bins ``1, ..., n - 1`` (the monomorphic ``0`` and ``n`` bins carry no information).
        """
        return list(range(1, self.lineage_config.n))

    @cached_property
    def mean(self) -> TwoLocusSFS:
        """
        Mean two-locus SFS, ``E[L^0_i · L^1_j]`` for all polymorphic bins, symmetrized over the two loci.
        """
        n = self.lineage_config.n
        indices = [(i, j) for i in self._get_indices() for j in self._get_indices()]

        results = [
            PhaseTypeDistribution.moment(
                self, k=2, permute=False, center=False,
                rewards=(
                    CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),
                    CombinedReward([self.reward, TwoLocusSFSReward(1, j)])
                )
            )
            for i, j in indices
        ]

        out = np.zeros((n + 1, n + 1))
        for (i, j), result in zip(indices, results):
            out[i, j] = result

        # symmetrize over the two (exchangeable) loci, as for the single-locus SFS covariance
        return TwoLocusSFS((out + out.T) / 2)

    @cached_property
    def corr(self) -> TwoLocusSFS:
        """
        Pearson correlation between the locus-0 and locus-1 branch lengths,
        ``Corr(L^0_i, L^1_j) = (E[L^0_i L^1_j] - E[L^0_i] E[L^1_j]) / (sd(L^0_i) sd(L^1_j))``, for all polymorphic
        bins ``(i, j)``. This is the centered, scale-free companion to :attr:`mean` (which is the *uncentered*
        cross-moment ``E[L^0_i L^1_j]`` and therefore tends to the outer product of the marginal SFS means as the
        loci decouple). It is ``0`` as ``r → ∞`` (independent loci) and reduces to the single-locus SFS correlation
        as ``r → 0`` (fully linked). The per-locus means and variances are the marginals of the two-locus space and
        coincide for the two exchangeable loci.
        """
        indices = self._get_indices()
        n = self.lineage_config.n

        # marginal locus-0 mean and variance per bin (identical for locus 1 by exchangeability, and independent of r)
        mean = {
            i: PhaseTypeDistribution.moment(
                self, k=1, center=False,
                rewards=(CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),)
            )
            for i in indices
        }
        var = {
            i: PhaseTypeDistribution.moment(
                self, k=2, center=True,
                rewards=(CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),) * 2
            )
            for i in indices
        }

        cross = self.mean.data
        out = np.zeros((n + 1, n + 1))
        for i in indices:
            for j in indices:
                denom = np.sqrt(var[i] * var[j])
                if denom > 0:
                    out[i, j] = (cross[i, j] - mean[i] * mean[j]) / denom

        return TwoLocusSFS(out)


class EmpiricalJointSFSDistribution:  # pragma: no cover
    """
    Empirical (msprime-based) joint site-frequency spectrum, exposing the same ``mean``/``var``/``m2``/``m3``
    interface as :class:`JointSFSDistribution` so that the two can be compared by
    :class:`~phasegen.comparison.Comparison`. The moments are pre-computed arrays (so the object can be serialized
    as cached ground truth).
    """

    def __init__(self, moments: np.ndarray):
        """
        Initialize the distribution.

        :param moments: Per-configuration (non-central) moments of orders ``1, 2, ...``, stacked along the first
            axis, i.e. an array of shape ``(max_order, n_0 + 1, ..., n_{P-1} + 1)``.
        """
        #: Non-central moments per descendant configuration, indexed by order minus one.
        self._moments: np.ndarray = np.asarray(moments)

    @property
    def mean(self) -> JointSFS:
        """
        Mean of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[0])

    @property
    def m2(self) -> JointSFS:
        """
        Second (non-central) moment of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[1])

    @property
    def m3(self) -> JointSFS:
        """
        Third (non-central) moment of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[2])

    @property
    def var(self) -> JointSFS:
        """
        Variance of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[1] - self._moments[0] ** 2)

    @property
    def data(self) -> np.ndarray:
        """
        The mean joint site-frequency spectrum array.
        """
        return self._moments[0]


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
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.cov(self.samples, rowvar=False))

    @cached_property
    def corr(self) -> float | np.ndarray:
        """
        Correlation matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
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
            samples: np.ndarray = None,
            **kwargs: dict
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
        with np.errstate(divide='ignore', invalid='ignore'):
            return SFS2(np.nan_to_num(np.cov(self.samples, rowvar=False)))

    @cached_property
    def corr(self) -> SFS2:
        """
        Correlation matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
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

        # zero-variance demes/loci make corrcoef divide by zero; the resulting NaNs are expected here, so
        # silence the benign warning
        with np.errstate(divide='ignore', invalid='ignore'):
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


class EmpiricalPhaseTypeSFSDistribution(EmpiricalPhaseTypeDistribution, TajimaSFSMixin):  # pragma: no cover
    """
    SFS phase-type distribution based on realisations.
    """

    def _tajima_n(self) -> int:
        # derive n from the (serialized) mean vector so this works on fixtures restored without ``n``
        return len(np.asarray(self.mean)) - 1

    def _tajima_mean(self) -> np.ndarray:
        n = self._tajima_n()
        return np.asarray(self.mean)[1:n]

    def _tajima_cov(self) -> np.ndarray:
        n = self._tajima_n()
        return np.asarray(self.cov)[1:n, 1:n]

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

        # bins with no variance (e.g. always-zero monomorphic counts) make np.corrcoef divide by a zero standard
        # deviation; the resulting NaNs are expected here, so silence the benign warning rather than emit it.
        with np.errstate(divide='ignore', invalid='ignore'):
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
        :param end_time: Time when to end the computation. If ``None``, the end time is taken to be the
            time of almost sure absorption. Note that unnecessarily large end times can lead to numerical errors.
        """
        self._logger = logger.getChild(self.__class__.__name__)

        # set up default coalescent model
        if model is None:
            model = StandardCoalescent()

        if not isinstance(n, LineageConfig):
            #: Population configuration
            self.lineage_config: LineageConfig = LineageConfig(n)
        else:
            #: Population configuration
            self.lineage_config: LineageConfig = n

        # set up demography
        if demography is None:
            demography = Demography(pop_sizes={p: 1 for p in self.lineage_config.pop_names})

        # set up locus configuration (accept a numeric number of loci, including the float that reticulate passes
        # from R, or a LocusConfig)
        if isinstance(loci, (int, float)):
            #: Locus configuration
            self.locus_config: LocusConfig = LocusConfig(
                n=int(loci),
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

            # warn if population names are present in the population configuration but not in the demography
            self._logger.warning(
                f"The following population names are present in the population configuration but not "
                f"in the demography: {list(initial_sizes.keys())}. "
                f"Adding these populations with population size of 1."
            )

        # determine population names that are present in the demography but not in the population configuration
        unspecified_lineages = set(demography.pop_names) - set(self.lineage_config.pop_names)

        # warn if population names are present in the demography but not in the population configuration
        if len(unspecified_lineages) > 0:
            self._logger.warning(
                f"The following population names are present in the demography but not "
                f"in the population configuration: {list(unspecified_lineages)}. "
                f"Adding these populations with 0 lineages."
            )

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
            start_time: float = 0,
            end_time: float = None,
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
        :param start_time: Time when to start accumulating moments. By default, this is 0.
        :param end_time: Time when to end the accumulating moments. If ``None``, the end time is taken to
            be the time of almost sure absorption. Note that unnecessarily long end times can lead to numerical errors.
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
    def joint_block_counting_state_space(self) -> JointBlockCountingStateSpace:
        """
        The joint block-counting state space (tracks the deme-of-origin composition of each lineage).
        """
        return JointBlockCountingStateSpace(
            lineage_config=self.lineage_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.demography.get_epoch(0)
        )

    @cached_property
    def tree_height(self) -> TreeHeightDistribution:
        """
        Tree height distribution, i.e. the time to the most recent common ancestor. With multiple loci this is the
        time until *all* loci have reached their MRCA (absorption of the two-locus ancestral process), so it equals
        the single-locus height when fully linked (``r = 0``) and grows towards the maximum of the per-locus heights
        as the loci decouple (``r -> inf``).
        """
        return TreeHeightDistribution(
            state_space=self.lineage_counting_state_space,
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
            state_space=self.lineage_counting_state_space,
            demography=self.demography
        )

    def _require_single_locus(self, name: str):
        """
        Raise a clear error if more than one locus is configured for a single-locus SFS statistic.

        :param name: Name of the statistic, used in the error message.
        :raises ValueError: if more than one locus is configured.
        """
        if self.locus_config.n != 1:
            raise ValueError(
                f"`{name}` is the single-locus site-frequency spectrum and is defined for one locus only "
                f"(got {self.locus_config.n}). For two loci under recombination use `sfs2` (the two-locus SFS); "
                f"the single-locus marginal is recombination-invariant, so drop the extra locus to obtain it."
            )

    @cached_property
    def sfs(self) -> UnfoldedSFSDistribution:
        """
        Unfolded site-frequency spectrum distribution. Defined for a single locus; for two loci under recombination
        use :meth:`sfs2`.
        """
        self._require_single_locus('sfs')

        return UnfoldedSFSDistribution(
            state_space=self.block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography
        )

    @cached_property
    def fsfs(self) -> FoldedSFSDistribution:
        """
        Folded site-frequency spectrum distribution. Defined for a single locus; for two loci under recombination
        use :meth:`sfs2`.
        """
        self._require_single_locus('fsfs')

        return FoldedSFSDistribution(
            state_space=self.block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography
        )

    @cached_property
    def jsfs(self) -> JointSFSDistribution:
        """
        Joint (multi-population) site-frequency spectrum distribution. Moments are returned as a multi-dimensional
        array of shape ``(n_0 + 1, ..., n_{P-1} + 1)``.

        .. note::
            The joint state space grows combinatorially with the per-population sample sizes, so this is only
            practical for small samples.

        :raises ValueError: If fewer than two populations are configured (the joint SFS is across populations; use
            :attr:`sfs` for a single population).
        """
        if self.lineage_config.n_pops < 2:
            raise ValueError(
                f"The joint SFS requires at least two populations, but {self.lineage_config.n_pops} is configured. "
                f"Use `sfs` for a single-population site-frequency spectrum."
            )

        return JointSFSDistribution(
            state_space=self.joint_block_counting_state_space,
            tree_height=self.tree_height,
            demography=self.demography
        )

    @cached_property
    def two_locus_block_counting_state_space(self) -> TwoLocusBlockCountingStateSpace:
        """
        The two-locus block-counting state space (tracks each lineage's descendant counts at both loci and the
        recombination/linkage history). Requires exactly two loci and a single population.
        """
        return TwoLocusBlockCountingStateSpace(
            lineage_config=self.lineage_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.demography.get_epoch(0)
        )

    @cached_property
    def _two_locus_tree_height(self) -> TreeHeightDistribution:
        """
        Tree height of the two-locus process, absorbed once *both* loci have reached their MRCA.
        """
        return TreeHeightDistribution(
            state_space=self.two_locus_block_counting_state_space,
            demography=self.demography,
            start_time=self.start_time,
            end_time=self.end_time
        )

    @cached_property
    def sfs2(self) -> TwoLocusSFSDistribution:
        """
        Two-locus site-frequency spectrum under recombination, returned as a :class:`~phasegen.spectrum.TwoLocusSFS`.
        Requires exactly two loci (``loci=2``) and a single population.

        .. note::
            The two-locus state space grows quickly with the sample size, so this is only practical for small ``n``.
        """
        return TwoLocusSFSDistribution(
            state_space=self.two_locus_block_counting_state_space,
            tree_height=self._two_locus_tree_height,
            demography=self.demography
        )

    @cached_property
    def fst(self) -> float:
        r"""
        Hudson's fixation index :math:`F_{ST} = 1 - \mathbb{E}[T_S] / \mathbb{E}[T_B]`, based on pairwise
        coalescence times: :math:`T_S` is the coalescence time of two lineages sampled within the same population
        (averaged over populations) and :math:`T_B` of two lineages from different populations (averaged over
        population pairs). Requires at least two populations.

        Since :math:`F_{ST}` is a pairwise, single-locus quantity, it is computed from two-lineage sub-coalescents
        under the same (possibly time-varying, migrating) demography and coalescent model, and so does not depend on
        the configured sample sizes or number of loci.

        :return: Hudson's :math:`F_{ST}`.
        :raises ValueError: if fewer than two populations are configured.
        """
        pops = self.demography.pop_names

        if len(pops) < 2:
            raise ValueError(f"F_ST requires at least two populations (got {len(pops)}).")

        # within-population pairwise times (both lineages in the same population)
        t_within = [self._pairwise_coalescence_time(q, q) for q in pops]

        # between-population pairwise times (one lineage in each of two distinct populations)
        t_between = [
            self._pairwise_coalescence_time(a, b)
            for i, a in enumerate(pops) for b in pops[i + 1:]
        ]

        return float(1 - np.mean(t_within) / np.mean(t_between))

    def _pairwise_coalescence_time(self, pop_i: str, pop_j: str) -> float:
        """
        Expected coalescence time of two lineages, one sampled in ``pop_i`` and one in ``pop_j`` (or both in the same
        population when ``pop_i == pop_j``), under this demography and coalescent model. Computed from a two-lineage
        sub-coalescent, so it is independent of the configured sample sizes and number of loci.

        :param pop_i: Name of the first population.
        :param pop_j: Name of the second population.
        :return: Expected pairwise coalescence time ``T_{ij}``.
        """
        pops = self.demography.pop_names

        for p in (pop_i, pop_j):
            if p not in pops:
                raise ValueError(f"Unknown population {p!r}; available: {pops}.")

        if pop_i == pop_j:
            counts = {p: (2 if p == pop_i else 0) for p in pops}
        else:
            counts = {p: (1 if p in (pop_i, pop_j) else 0) for p in pops}

        return Coalescent(
            n=counts,
            demography=self.demography,
            model=self.model,
            end_time=self.end_time
        ).tree_height.mean

    def f2(self, pop_0: str, pop_1: str) -> float:
        r"""
        Patterson's :math:`f_2(A, B) = \mathbb{E}[(p_A - p_B)^2]`, the branch (coalescence-time) version
        :math:`f_2 = 2 T_{AB} - T_{AA} - T_{BB}` in terms of pairwise coalescence times (matching ``tskit``'s
        branch-mode ``f2``). Measures the amount of drift separating the two populations.

        :param pop_0: Name of population ``A``.
        :param pop_1: Name of population ``B``.
        :return: :math:`f_2(A, B)`.
        """
        t = self._pairwise_coalescence_time
        return float(2 * t(pop_0, pop_1) - t(pop_0, pop_0) - t(pop_1, pop_1))

    def f3(self, pop_target: str, pop_0: str, pop_1: str) -> float:
        r"""
        Patterson's :math:`f_3(C; A, B) = \mathbb{E}[(p_C - p_A)(p_C - p_B)]`, in branch (coalescence-time) form
        :math:`f_3 = T_{CA} + T_{CB} - T_{AB} - T_{CC}` (matching ``tskit``'s branch-mode ``f3``). A significantly
        negative value is evidence that the target population ``C`` is admixed between ``A`` and ``B``.

        :param pop_target: Name of the (potentially admixed) target population ``C``.
        :param pop_0: Name of source population ``A``.
        :param pop_1: Name of source population ``B``.
        :return: :math:`f_3(C; A, B)`.
        """
        t = self._pairwise_coalescence_time
        return float(t(pop_target, pop_0) + t(pop_target, pop_1) - t(pop_0, pop_1) - t(pop_target, pop_target))

    def f4(self, pop_0: str, pop_1: str, pop_2: str, pop_3: str) -> float:
        r"""
        Patterson's :math:`f_4(A, B; C, D) = \mathbb{E}[(p_A - p_B)(p_C - p_D)]`, in branch (coalescence-time) form
        :math:`f_4 = T_{AD} + T_{BC} - T_{AC} - T_{BD}` (matching ``tskit``'s branch-mode ``f4``). Used to test
        treeness and detect gene flow between the two population pairs.

        :param pop_0: Name of population ``A``.
        :param pop_1: Name of population ``B``.
        :param pop_2: Name of population ``C``.
        :param pop_3: Name of population ``D``.
        :return: :math:`f_4(A, B; C, D)`.
        """
        t = self._pairwise_coalescence_time
        return float(t(pop_0, pop_3) + t(pop_1, pop_2) - t(pop_0, pop_2) - t(pop_1, pop_3))

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

        # only route to the (expensive) joint state space when a reward requires it; then all rewards must support it
        if Reward.requires_joint_state_space(rewards):
            if not Reward.support(JointBlockCountingStateSpace, rewards):
                raise ValueError(
                    "The given rewards are not jointly compatible with any single state space: "
                    f"{[r.__class__.__name__ for r in rewards]}. A joint-SFS reward can only be combined with "
                    "rewards that also support the joint state space."
                )
            state_space = self.joint_block_counting_state_space
        elif Reward.support(LineageCountingStateSpace, rewards):
            state_space = self.lineage_counting_state_space
        else:
            state_space = self.block_counting_state_space

        return PhaseTypeDistribution(
            reward=UnitReward(),
            tree_height=self.tree_height,
            state_space=state_space,
            demography=self.demography
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

    def _sample(
            self,
            n_samples: int,
            rewards: Sequence[Reward] = None,
            record_visits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate samples from the mean reward distribution by simulating trajectories.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is the tree height reward.
        :param record_visits: Whether to record which states were visited during the sampling.
        :return: Array of sampled rewards of size (n_samples, len(rewards)),
                 and optionally an array of probabilities of visiting each state.
        """
        return self._get_dist(k=1, rewards=rewards)._sample(
            n_samples=n_samples,
            rewards=rewards,
            record_visits=record_visits
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

    def to_msprime(
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
        :param seed: Random seed.
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


class EmpiricalTwoLocusSFSDistribution:  # pragma: no cover
    """
    Empirical (msprime-based) two-locus SFS, exposing the same ``mean`` interface as
    :class:`TwoLocusSFSDistribution` (a :class:`~phasegen.spectrum.TwoLocusSFS`) so the two can be compared by
    :class:`~phasegen.comparison.Comparison`.
    """

    def __init__(self, mean: np.ndarray):
        """
        :param mean: The simulated mean two-locus SFS array.
        """
        self._mean = np.asarray(mean)

    @property
    def mean(self) -> TwoLocusSFS:
        """Mean two-locus SFS."""
        return TwoLocusSFS(self._mean)


class MsprimeCoalescent(AbstractCoalescent):
    """
    Empirical coalescent distribution based on `msprime` simulations.
    This is used for testing purposes. Note that the results are stochastic.
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

        #: Joint SFS (non-central) moments per descendant configuration, of orders 1, ..., ``_jsfs_max_order``.
        self.jsfs_moments: np.ndarray | None = None

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

        # joint SFS is accumulated from the same trees, but only for multi-population, single-locus scenarios where
        # it is meaningful (the descendant configuration is by deme of origin)
        compute_jsfs = self.lineage_config.n_pops > 1 and self.locus_config.n == 1
        jsfs_max_order = self._jsfs_max_order
        jsfs_shape = tuple(int(s) + 1 for s in self.lineage_config.lineages)
        name_to_index = {name: i for i, name in enumerate(self.demography.pop_names)}
        n_total = num_replicates * self.n_threads

        def simulate_batch(seed: Optional[int]) -> dict:
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

            # joint SFS moment accumulator (non-central moments of orders 1, ..., jsfs_max_order)
            jsfs_acc = np.zeros((jsfs_max_order,) + jsfs_shape)

            # iterate over trees and compute statistics
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):

                # map each sample to the index of its sampling population (deme of origin) for the joint SFS
                if compute_jsfs:
                    pop_of_leaf = {
                        u: name_to_index[ts.population(ts.node(u).population).metadata['name']]
                        for u in ts.samples()
                    }

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

                    # accumulate the joint SFS from the same tree (single locus only)
                    if compute_jsfs and j == 0:
                        jsfs_rep = np.zeros(jsfs_shape)

                        for node in tree.nodes():

                            # the root subtends all samples (monomorphic) and is skipped
                            if tree.parent(node) == -1:
                                continue

                            # count descendant samples by population (deme of origin)
                            vec = [0] * len(jsfs_shape)
                            for leaf in tree.leaves(node):
                                vec[pop_of_leaf[leaf]] += 1

                            if sum(vec) > 0:
                                jsfs_rep[tuple(vec)] += tree.get_branch_length(node)

                        for order in range(jsfs_max_order):
                            jsfs_acc[order] += jsfs_rep ** (order + 1)

                # simulate mutations if specified
                if self.simulate_mutations:

                    mts = ms.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)
                    tree = next(mts.trees())

                    for node in mts.mutations_node:
                        mutations[0, 0, i, tree.get_num_leaves(node)] += 1

            return dict(
                main=np.concatenate([[heights.T], [total_branch_lengths.T], sfs.T, mutations.T]),
                jsfs=jsfs_acc
            )

        # parallelize over threads
        batches = parallelize(
            func=simulate_batch,
            data=[self.seed + i if self.seed is not None else None for i in range(self.n_threads)],
            parallelize=self.parallelize,
            batch_size=num_replicates,
            desc="Simulating trees",
            dtype=object
        )

        # combine the per-replicate statistics across threads
        res = np.hstack([b['main'] for b in batches])

        # store results
        self.heights = res[0].T
        self.total_branch_lengths = res[1].T
        self.sfs_lengths = res[2:sample_size + 3].T
        self.mutations = res[sample_size + 3:].T.astype(int)

        # combine the joint SFS moments (summed over replicates) across threads and normalize to moments
        self.jsfs_moments = np.sum([b['jsfs'] for b in batches], axis=0) / n_total if compute_jsfs else None

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

    def touch(self, **kwargs: dict):
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

        # cache the joint SFS distribution (its moments were already accumulated by simulate() above) for
        # multi-population, single-locus scenarios, so it is serialized along with the comparison
        if self.lineage_config.n_pops > 1 and self.locus_config.n == 1:
            # noinspection PyStatementEffect
            self.jsfs

    def drop(self):
        """
        Drop simulated data.
        """
        self.heights = None
        self.total_branch_lengths = None
        self.sfs_lengths = None
        self.mutations = None

        # the moments are retained by the cached jsfs distribution (referenced before drop), so this only removes
        # the duplicate reference held on the coalescent
        self.jsfs_moments = None

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

    #: Highest moment order computed for the empirical joint SFS ground truth.
    _jsfs_max_order: int = 3

    @cached_property
    def jsfs(self) -> 'EmpiricalJointSFSDistribution':
        """
        Joint (multi-population) site-frequency spectrum ground truth, accumulated from the same simulated trees as
        the other statistics (see :meth:`simulate`). Returns an :class:`EmpiricalJointSFSDistribution` exposing
        ``mean``, ``m2``, ``m3`` and ``var`` as arrays of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, matching
        :class:`JointSFSDistribution`. The descendant configuration of a branch is the number of its sample
        descendants from each population (its deme of origin). Only available for multi-population, single-locus
        scenarios.
        """
        self.simulate()

        if self.jsfs_moments is None:
            raise NotImplementedError(
                "The joint SFS is only available for multi-population, single-locus scenarios."
            )

        return EmpiricalJointSFSDistribution(moments=self.jsfs_moments)

    @cached_property
    def sfs2(self) -> 'EmpiricalTwoLocusSFSDistribution':
        """
        Two-locus SFS ground truth, simulated with msprime: two sites at recombination distance ``r`` (the two loci),
        the per-bin branch-length cross product averaged over replicates. Only available for two-locus, single-locus-
        sample scenarios. Returns an :class:`EmpiricalTwoLocusSFSDistribution` exposing ``mean`` as a
        :class:`~phasegen.spectrum.TwoLocusSFS`, matching :class:`TwoLocusSFSDistribution`.
        """
        import msprime as ms

        if self.locus_config.n != 2:
            raise NotImplementedError("The two-locus SFS is only available for two-locus scenarios.")

        n = self.lineage_config.n
        demography = self.demography.to_msprime()
        model = self.get_coalescent_model()

        out = np.zeros((n + 1, n + 1))
        for ts in ms.sim_ancestry(
                samples=self.lineage_config.lineage_dict,
                sequence_length=2,
                recombination_rate=self.locus_config.recombination_rate,
                demography=demography,
                model=model,
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self.seed,
        ):
            t0, t1 = ts.at(0.5), ts.at(1.5)
            left = np.zeros(n + 1)
            right = np.zeros(n + 1)
            for nd in t0.nodes():
                if t0.parent(nd) != -1:
                    left[t0.num_samples(nd)] += t0.branch_length(nd)
            for nd in t1.nodes():
                if t1.parent(nd) != -1:
                    right[t1.num_samples(nd)] += t1.branch_length(nd)
            out += np.outer(left, right)

        return EmpiricalTwoLocusSFSDistribution(out / self.num_replicates)

    @cached_property
    def fst(self) -> float:
        r"""
        Hudson's :math:`F_{ST}` ground truth, simulated with msprime: ``1 - mean within-population branch diversity /
        mean between-population branch divergence``, averaged over replicate trees. Requires at least two populations,
        each with at least two sampled lineages. Matches :meth:`Coalescent.fst`.
        """
        import msprime as ms

        pops = self.demography.pop_names

        if len(pops) < 2:
            raise ValueError(f"F_ST requires at least two populations (got {len(pops)}).")

        within = np.zeros(self.num_replicates)
        between = np.zeros(self.num_replicates)

        for k, ts in enumerate(ms.sim_ancestry(
                samples=self.lineage_config.lineage_dict,
                sequence_length=1,
                demography=self.demography.to_msprime(),
                model=self.get_coalescent_model(),
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self.seed,
        )):
            sample_sets = [ts.samples(population=i) for i in range(len(pops))]

            # within-population diversity (only populations with at least two samples are informative)
            w = [ts.diversity(s, mode='branch') for s in sample_sets if len(s) >= 2]
            # between-population divergence over distinct population pairs
            b = [ts.divergence([sample_sets[i], sample_sets[j]], mode='branch')
                 for i in range(len(pops)) for j in range(i + 1, len(pops))
                 if len(sample_sets[i]) and len(sample_sets[j])]

            within[k] = np.mean(w)
            between[k] = np.mean(b)

        return float(1 - within.mean() / between.mean())

    def _branch_f_statistic(self, kind: str, pops: List[str]) -> float:
        """
        msprime branch-mode Patterson f-statistic ground truth (``f2``/``f3``/``f4``) over the given populations,
        averaged over replicate trees (tskit branch mode uses the same 2x pairwise-coalescence convention as the
        analytical :class:`Coalescent` f-statistics).
        """
        import msprime as ms

        names = self.demography.pop_names
        for pop in pops:
            if pop not in names:
                raise ValueError(f"Unknown population '{pop}'. Available populations: {names}.")
        idx = [names.index(pop) for pop in pops]

        values = np.zeros(self.num_replicates)

        for k, ts in enumerate(ms.sim_ancestry(
                samples=self.lineage_config.lineage_dict,
                sequence_length=1,
                demography=self.demography.to_msprime(),
                model=self.get_coalescent_model(),
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self.seed,
        )):
            sample_sets = [ts.samples(population=i) for i in idx]
            values[k] = getattr(ts, kind)(sample_sets, mode='branch')

        return float(values.mean())

    def f2(self, pop_0: str, pop_1: str) -> float:
        """msprime branch-mode ``f2`` ground truth. Matches :meth:`Coalescent.f2`."""
        return self._branch_f_statistic('f2', [pop_0, pop_1])

    def f3(self, pop_target: str, pop_0: str, pop_1: str) -> float:
        """msprime branch-mode ``f3`` ground truth. Matches :meth:`Coalescent.f3`."""
        return self._branch_f_statistic('f3', [pop_target, pop_0, pop_1])

    def f4(self, pop_0: str, pop_1: str, pop_2: str, pop_3: str) -> float:
        """msprime branch-mode ``f4`` ground truth. Matches :meth:`Coalescent.f4`."""
        return self._branch_f_statistic('f4', [pop_0, pop_1, pop_2, pop_3])

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
