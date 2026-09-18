"""The Coalescent facade and its abstract base."""

import copy
import logging
from abc import ABC, abstractmethod
from ..caching import cached_property, cache
from typing import List, Tuple, Dict, Iterable, Sequence, Union, TYPE_CHECKING
import numpy as np
from ..coalescent_models import StandardCoalescent, CoalescentModel
from ..demography import Demography, PopSizeChanges
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalBranchLengthReward
from ..serialization import Serializable
from ..state_space import StateSpace, BlockCountingStateSpace, LineageCountingStateSpace, JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace

from ._common import _make_hashable, _validate_order
from .base import DensityAwareDistribution, MomentAwareDistribution
from .phase_type import PhaseTypeDistribution, TreeHeightDistribution, TotalBranchLengthDistribution
from .spectra import FoldedSFSDistribution, JointSFSDistribution, TwoLocusSFSDistribution, UnfoldedSFSDistribution

if TYPE_CHECKING:
    from .reward import RewardDistribution, JointRewardDistribution
    from matplotlib import pyplot as plt
    from .empirical import MsprimeCoalescent, SampledCoalescent

expm = Backend.expm
logger = logging.getLogger('phasegen')


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
    ) -> None:
        """
        Create object.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values. Alternatively, a
            :class:`~phasegen.lineage.LineageConfig` object can be passed.
        :param model: Coalescent model. By default, the standard coalescent is used.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate. If given, it overrides the rate of ``loci``.
        :param demography: Demography.
        :param end_time: Time when to end the computation. If ``None``, the end time is taken to be the
            time of almost sure absorption. Note that unnecessarily large end times can lead to numerical errors.
        :raises ValueError: If the number of unlinked lineages exceeds the number of lineages.
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
        else:
            # copy so filling in missing populations never mutates the caller-supplied demography
            demography = copy.deepcopy(demography)

        # accept a number of loci (including the float that reticulate passes from R) or a locus configuration
        if not isinstance(loci, LocusConfig):
            loci = LocusConfig(n=loci)

        # a new, validated locus configuration, so the caller-supplied one is never mutated
        #: Locus configuration
        self.locus_config: LocusConfig = LocusConfig(
            n=loci.n,
            n_unlinked=loci.n_unlinked,
            recombination_rate=loci.recombination_rate if recombination_rate is None else recombination_rate
        )

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

        # population names present in the demography but not in the population configuration, in demography order
        unspecified_lineages = [p for p in demography.pop_names if p not in self.lineage_config.pop_names]

        # warn if population names are present in the demography but not in the population configuration
        if len(unspecified_lineages) > 0:
            self._logger.warning(
                f"The following population names are present in the demography but not "
                f"in the population configuration: {list(unspecified_lineages)}. "
                f"Adding these populations with 0 lineages."
            )

        self.lineage_config = LineageConfig(self.lineage_config.lineage_dict | {p: 0 for p in unspecified_lineages})

        if self.locus_config.n_unlinked > self.lineage_config.n:
            raise ValueError(
                f"The number of unlinked lineages ({self.locus_config.n_unlinked}) must not exceed the number of "
                f"lineages ({self.lineage_config.n})."
            )

        #: Coalescent model
        self.model: CoalescentModel = model

        #: Demography
        self.demography: Demography = demography

        #: End time
        self.end_time: float = end_time

    @property
    def n(self) -> int:
        """Total number of sampled lineages across all populations."""
        return self.lineage_config.n

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
    ) -> None:
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
        r"""
        Tree height distribution, i.e. the time to the most recent common ancestor. This is the phase-type absorption
        time :math:`\tau` of the underlying Markov jump process, with the notation of
        :class:`~phasegen.distributions.PhaseTypeDistribution`, equivalently the reward accumulated under the reward
        :math:`r(x) = \mathbb{1}\{x \notin B\}`. With multiple loci this is the time until every locus has reached its
        MRCA (absorption of the two-locus ancestral process), so it equals the single-locus height when fully linked
        (:math:`\rho = 0`, with :math:`\rho` the recombination rate) and grows towards the maximum of the per-locus
        heights as the loci decouple (:math:`\rho \to \infty`).
        """
        return TreeHeightDistribution(
            state_space=self.lineage_counting_state_space,
            demography=self.demography,
            start_time=self.start_time,
            end_time=self.end_time
        )

    @cached_property
    def total_branch_length(self) -> TotalBranchLengthDistribution:
        r"""
        Total branch length distribution: the sum of all branch lengths of the coalescent tree, i.e. the accumulated
        reward :math:`\int_0^{\tau} r_{\text{length}}(X_s)\,\mathrm{d}s` under the reward
        :math:`r_{\text{length}}(i) = (\text{number of lineages in } i)`.
        """
        return TotalBranchLengthDistribution(
            tree_height=self.tree_height,
            state_space=self.lineage_counting_state_space,
            demography=self.demography
        )

    def _require_single_locus(self, name: str) -> None:
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
        r"""
        Unfolded site-frequency spectrum distribution. Bin :math:`j` is the accumulated length of all branches
        subtending exactly :math:`j` of the :math:`n` samples, the reward :math:`r_j(x) = a_j(x)` counting the lineages
        in state :math:`x` that subtend :math:`j` samples. It is defined for a single locus. For two loci under
        recombination, use :attr:`sfs2`.
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
        Folded site-frequency spectrum distribution. It is defined for a single locus. For two loci under
        recombination, use :attr:`sfs2`.
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

        :raises ValueError: If fewer than two populations are configured. For a single population, use :attr:`sfs`.
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
        Two-locus site-frequency spectrum distribution under recombination, whose moments are
        :class:`~sfsutils.spectrum.TwoLocusSFS` objects. Requires exactly two loci (``loci=2``) and a single
        population.

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
        Hudson's fixation index

        .. math::

            F_{ST} = 1 - \frac{\overline{\mathbb{E}[T_{PP}]}}{\overline{\mathbb{E}[T_{PP'}]}},

        where :math:`T_{PP'}` is the coalescence time of one lineage sampled in population :math:`P` and one in
        population :math:`P'`, the numerator averages :math:`\mathbb{E}[T_{PP}]` over all populations and the
        denominator averages :math:`\mathbb{E}[T_{PP'}]` over all unordered pairs :math:`P \ne P'`. Each expectation
        is the mean tree height of a two-lineage coalescent with the same demography and coalescent model, so the
        result does not depend on the configured sample sizes or number of loci.

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
            start_time=self.start_time,
            end_time=self.end_time
        ).tree_height.mean

    def f2(self, pop_0: str, pop_1: str) -> float:
        r"""
        Branch form of Patterson's :math:`f_2(A, B) = \mathbb{E}[(p_A - p_B)^2]`, where :math:`p_A` and :math:`p_B`
        are the allele frequencies in populations :math:`A` and :math:`B`,

        .. math::

            f_2(A, B) = 2\, \mathbb{E}[T_{AB}] - \mathbb{E}[T_{AA}] - \mathbb{E}[T_{BB}],

        with :math:`T_{XY}` the coalescence time of one lineage sampled in population :math:`X` and one in
        population :math:`Y`, matching the branch mode of ``tskit``. It measures the drift separating the two
        populations.

        :param pop_0: Name of population ``A``.
        :param pop_1: Name of population ``B``.
        :return: :math:`f_2(A, B)`.
        """
        t = self._pairwise_coalescence_time
        return float(2 * t(pop_0, pop_1) - t(pop_0, pop_0) - t(pop_1, pop_1))

    def f3(self, pop_target: str, pop_0: str, pop_1: str) -> float:
        r"""
        Branch form of Patterson's :math:`f_3(C; A, B) = \mathbb{E}[(p_C - p_A)(p_C - p_B)]`, with allele
        frequencies and pairwise coalescence times :math:`T_{XY}` as in :meth:`Coalescent.f2()
        <phasegen.distributions.Coalescent.f2>`,

        .. math::

            f_3(C; A, B) = \mathbb{E}[T_{CA}] + \mathbb{E}[T_{CB}] - \mathbb{E}[T_{AB}] - \mathbb{E}[T_{CC}],

        matching the branch mode of ``tskit``. A negative value indicates that the target population :math:`C` is
        admixed between :math:`A` and :math:`B`.

        :param pop_target: Name of the (potentially admixed) target population ``C``.
        :param pop_0: Name of source population ``A``.
        :param pop_1: Name of source population ``B``.
        :return: :math:`f_3(C; A, B)`.
        """
        t = self._pairwise_coalescence_time
        return float(t(pop_target, pop_0) + t(pop_target, pop_1) - t(pop_0, pop_1) - t(pop_target, pop_target))

    def f4(self, pop_0: str, pop_1: str, pop_2: str, pop_3: str) -> float:
        r"""
        Branch form of Patterson's :math:`f_4(A, B; C, D) = \mathbb{E}[(p_A - p_B)(p_C - p_D)]`, with allele
        frequencies and pairwise coalescence times :math:`T_{XY}` as in :meth:`Coalescent.f2()
        <phasegen.distributions.Coalescent.f2>`,

        .. math::

            f_4(A, B; C, D) = \mathbb{E}[T_{AD}] + \mathbb{E}[T_{BC}] - \mathbb{E}[T_{AC}] - \mathbb{E}[T_{BD}],

        matching the branch mode of ``tskit``. It tests treeness and detects gene flow between the two population
        pairs.

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
        The returned phase-type distribution is configured with the first (default: tree-height) reward.

        :param k: Order of the moment.
        :param rewards: Sequence of k rewards. By default, tree height rewards are used.
        :return: Distribution.
        :raises ValueError: if a single :class:`~phasegen.rewards.Reward` is passed instead of a sequence.
        """
        if isinstance(rewards, Reward):
            raise ValueError(
                f"rewards must be a sequence of {k} rewards, but a single {Reward.__name__} instance was given. "
                f"Wrap it in a list, e.g. rewards=[reward]."
            )

        if rewards is None:
            rewards = [TreeHeightReward()] * k

        return PhaseTypeDistribution(
            reward=rewards[0],
            tree_height=self.tree_height,
            state_space=self._select_state_space(rewards),
            demography=self.demography
        )

    def _select_state_space(self, rewards: Iterable[Reward]) -> StateSpace:
        """
        Select the smallest state space jointly compatible with the given rewards -- the reward-compatibility wiring
        shared by :meth:`moment`, :meth:`accumulate`, :meth:`distribution` and :meth:`joint_distribution` (all via
        :meth:`_get_dist`). The (expensive) joint block-counting space is used only when a reward requires it (then
        every reward must also support it). Otherwise the lineage-counting space is used if all rewards support it,
        then the two-locus block-counting space if all rewards support it, else the block-counting space.

        :param rewards: The rewards to be accumulated jointly.
        :return: The state space supporting all the rewards.
        :raises ValueError: if the rewards are not jointly compatible with any single state space.
        """
        if Reward.requires_joint_state_space(rewards):
            if not Reward.support(JointBlockCountingStateSpace, rewards):
                raise ValueError(
                    "The given rewards are not jointly compatible with any single state space: "
                    f"{[r.__class__.__name__ for r in rewards]}. A joint-SFS reward can only be combined with "
                    "rewards that also support the joint state space."
                )
            return self.joint_block_counting_state_space

        if Reward.support(LineageCountingStateSpace, rewards):
            return self.lineage_counting_state_space

        if Reward.support(TwoLocusBlockCountingStateSpace, rewards):
            return self.two_locus_block_counting_state_space

        return self.block_counting_state_space

    @_make_hashable
    @cache
    def distribution(self, reward: Reward = None) -> 'RewardDistribution':
        r"""
        The distribution of the accumulated reward :math:`R`, as a
        :class:`~phasegen.distributions.RewardDistribution` whose evaluation is described there. The state space is
        the smallest one that supports the reward, and the result is cached per reward. For the default tree-height
        reward it describes the same law as :attr:`Coalescent.tree_height
        <phasegen.distributions.Coalescent.tree_height>`, evaluated by transform inversion.

        :param reward: The reward whose accumulation is distributed. Defaults to the tree-height reward.
        :return: The 1D accumulated-reward distribution.
        """
        reward = TreeHeightReward() if reward is None else reward
        return self._get_dist(k=1, rewards=[reward]).distribution(reward)

    @_make_hashable
    @cache
    def joint_distribution(self, reward_a: Reward, reward_b: Reward) -> 'JointRewardDistribution':
        """
        Joint distribution of two accumulated rewards, as a :class:`~phasegen.distributions.JointRewardDistribution`,
        on the smallest state space supporting both rewards and cached per pair of rewards.

        :param reward_a: The first reward.
        :param reward_b: The second reward.
        :return: The joint distribution.

        .. versionadded:: 2.0
        """
        return self._get_dist(k=2, rewards=[reward_a, reward_b]).joint_distribution(reward_a, reward_b)

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
        r"""
        The :math:`k`-th moment of the accumulated rewards, evaluated on the smallest state space supporting all
        ``rewards`` as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the tree-height reward for each factor.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the coalescent.
        :param end_time: The end time :math:`t_\mathrm{end}`. By default, the end time of the coalescent, or absorption.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: The :math:`k`-th moment.
        :raises ValueError: if ``k`` is not integral or is smaller than one.
        """
        k = _validate_order(k)

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
        r"""
        The :math:`k`-th moment accumulated up to each end time :math:`t_\mathrm{end}` in ``end_times``, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t_\mathrm{end}` at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the tree-height reward for each factor.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: The moment at each end time.
        :raises ValueError: if ``k`` is not integral or is smaller than one.
        """
        k = _validate_order(k)

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
        :param end_times: Times when to evaluate the moment. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        :raises ValueError: if ``k`` is not integral or is smaller than one.
        """
        k = _validate_order(k)

        return self._get_dist(k, rewards).plot_accumulation(
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

    #: The state-space caches to drop, one per lazily built (cached-property) state space
    _STATE_SPACE_CACHES = (
        'lineage_counting_state_space',
        'block_counting_state_space',
        'joint_block_counting_state_space',
        'two_locus_block_counting_state_space',
    )

    def drop_cache(self) -> None:
        """
        Drop the cache of every state space that has been built. Spaces that have not been built are left unbuilt.
        """
        for name in self._STATE_SPACE_CACHES:
            if name in self.__dict__:
                self.__dict__[name].drop_cache()

    def __setstate__(self, state: dict) -> None:
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

        for name in self._STATE_SPACE_CACHES:
            if name in other:
                other[name].drop_cache()

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
        :param parallelize: Whether to parallelize. ``Settings.parallelize = False`` overrides it.
        :param record_migration: Whether to record migrations which is necessary to calculate statistics per deme.
        :param simulate_mutations: Whether to simulate mutations.
        :param mutation_rate: Mutation rate.
        :param seed: Random seed.
        :return: msprime coalescent.
        """
        if self.start_time != 0:
            self._logger.warning("Non-zero start times are not supported by MsprimeCoalescent.")

        from .empirical import MsprimeCoalescent
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

    def to_empirical(self, n_samples: int = 100000, seed: int = None) -> 'SampledCoalescent':
        """
        Estimate every statistic by simulation, see :class:`~phasegen.distributions.SampledCoalescent`.

        :param n_samples: Number of trajectories to sample per statistic.
        :param seed: Integer seed, ``None`` for fresh entropy.
        :return: The sampled coalescent.

        .. versionadded:: 2.0
        """
        from .empirical import SampledCoalescent
        return SampledCoalescent(coalescent=self, n_samples=n_samples, seed=seed)

