"""The Coalescent facade and its abstract base."""

import copy
import logging
from abc import ABC, abstractmethod
from ..caching import cached_property, cache
from typing import List, Tuple, Dict, Iterable, Sequence, Union
import numpy as np
from ..coalescent_models import StandardCoalescent, CoalescentModel
from ..demography import Demography, PopSizeChanges
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalBranchLengthReward, UnitReward
from ..serialization import Serializable
from ..spectrum import SFS, TwoLocusSFS
from ..state_space import BlockCountingStateSpace, LineageCountingStateSpace, JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace
from ..utils import parallelize

from ._common import _make_hashable
from .base import DensityAwareDistribution, MomentAwareDistribution
from .phase_type import PhaseTypeDistribution, TreeHeightDistribution
from .spectra import FoldedSFSDistribution, JointSFSDistribution, TwoLocusSFSDistribution, UnfoldedSFSDistribution

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

