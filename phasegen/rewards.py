"""
Rewards assign weights to states in the state space. They are used behind the scenes
to calculate different coalescent tree statistics but can also be passed directly to
:meth:`~phasegen.distributions.Coalescent.moment`,
:meth:`~phasegen.distributions.Coalescent.accumulate`, or
:meth:`~phasegen.distributions.Coalescent.distribution` to accumulate them into more
complex statistics and their distributions.
"""

from abc import abstractmethod, ABC
from typing import List, Callable, Tuple, Iterable, Type

import numpy as np

from .state_space import StateSpace, LineageCountingStateSpace, BlockCountingStateSpace, \
    JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace


class Reward(ABC):
    """
    Base class for reward generation.
    """

    #: Whether :meth:`_get_parts` resolves the locus and deme the rewarded lineages reside in, rather than splitting
    #: the reward in proportion to the lineage counts.
    _resolves_residence: bool = False

    @abstractmethod
    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.
        
        :param state_space: state space
        :return: reward vector
        """
        pass

    def _get_parts(self, state_space: StateSpace) -> np.ndarray:
        r"""
        The reward resolved by locus and deme, of shape ``(n_states, n_loci, n_demes)``, entry :math:`(i, l, d)`
        being the share of the reward of state :math:`i` carried by locus :math:`l` and deme :math:`d`, as summed by
        :class:`RestrictedReward`. By default the reward is split in proportion to the lineages at each locus and
        deme, counting only the loci that carry more than one lineage, and vanishes on the absorbing states.

        :param state_space: state space
        :return: reward parts
        """
        # lineages per locus and deme, of the loci that have not yet reached their MRCA
        weights = state_space.lineages.sum(axis=3) * (state_space.lineages.sum(axis=(2, 3)) > 1)[:, :, None]

        total = weights.sum(axis=(1, 2))
        shares = weights / np.where(total > 0, total, 1)[:, None, None]

        rewards = np.asarray(self._get(state_space), dtype=float) * ~state_space.absorbing

        return rewards[:, None, None] * shares

    def __hash__(self) -> int:
        """
        Get the hash for the reward.

        :return: hash.
        """
        # hash the class name as this class is stateless.
        return hash(self.__class__.__name__)

    def __eq__(self, other: 'Reward') -> bool:
        """
        Check if the rewards are equal.

        :param other: other reward.
        :return: True if the rewards are equal, False otherwise.
        """
        return self.__class__ == other.__class__ and hash(self) == hash(other)

    def prod(self, *rewards: 'Reward') -> 'ProductReward':
        """
        Product of this reward with other rewards.

        :param rewards: Rewards to take the product with.
        :return: Product of the rewards.
        """
        return ProductReward([self] + list(rewards))

    def sum(self, *rewards: 'Reward') -> 'SumReward':
        """
        Sum of this reward with other rewards.

        :param rewards: Rewards to take the sum with.
        :return: Sum of the rewards.
        """
        return SumReward([self] + list(rewards))

    def supports(self, state_space: Type[StateSpace]) -> bool:
        """
        Check if the reward supports the given state space.

        :param state_space: state space
        :return: True if the reward supports the state space, False otherwise
        """
        if state_space is LineageCountingStateSpace:
            return isinstance(self, LineageCountingReward)

        if state_space is BlockCountingStateSpace:
            return isinstance(self, BlockCountingReward)

        if state_space is JointBlockCountingStateSpace:
            return isinstance(self, JointBlockCountingReward)

        if state_space is TwoLocusBlockCountingStateSpace:
            return isinstance(self, TwoLocusBlockCountingReward)

    @staticmethod
    def support(state_space: Type[StateSpace], rewards: Iterable['Reward']) -> bool:
        """
        Check if the rewards support the given state space.

        :param state_space: state space
        :param rewards: rewards
        :return: True if the rewards support the state space, False otherwise
        """
        return all([reward.supports(state_space) for reward in rewards])

    @staticmethod
    def _check_accumulable(state_space: StateSpace, rewards: Iterable['Reward']) -> None:
        r"""
        Check that the rewards vanish on the absorbing states of the state space. The reward accumulated up to
        absorption is determined by the transient entries alone, so a reward with mass on an absorbing state is
        rejected.

        :param state_space: The state space the rewards are resolved against.
        :param rewards: The rewards to accumulate.
        :raises ValueError: If a reward is non-zero on an absorbing state.
        """
        for reward in rewards:
            values = np.asarray(reward._get(state_space), dtype=float)[state_space.absorbing]

            if np.any(values != 0):
                raise ValueError(
                    f"Reward {reward.__class__.__name__} is non-zero on {int(np.sum(values != 0))} of the "
                    f"{len(values)} absorbing states, so it does not define an accumulation until absorption. "
                    f"Combine it with a reward that vanishes on the absorbing states, such as "
                    f"{TreeHeightReward.__name__} or {TotalBranchLengthReward.__name__}."
                )

    @staticmethod
    def requires_joint_state_space(rewards: Iterable['Reward']) -> bool:
        """
        Check whether any (possibly nested) reward can only be evaluated on the joint block-counting state space,
        i.e. supports it but neither the lineage- nor the block-counting state space (e.g. :class:`JointSFSReward`).

        :param rewards: rewards
        :return: True if some reward requires the joint state space
        """
        for reward in rewards:
            # recurse into composite rewards
            if isinstance(reward, CompositeReward):
                if Reward.requires_joint_state_space(reward.rewards):
                    return True
            elif (
                    reward.supports(JointBlockCountingStateSpace)
                    and not reward.supports(BlockCountingStateSpace)
                    and not reward.supports(LineageCountingStateSpace)
            ):
                return True

        return False


class LineageCountingReward(Reward, ABC):
    """
    Base class for rewards that count lineages. Such rewards are compatible with
    :class:`~phasegen.state_space.LineageCountingStateSpace`.
    """
    pass


class BlockCountingReward(Reward, ABC):
    """
    Base class for rewards that count blocks. Such rewards are compatible with
    :class:`~phasegen.state_space.BlockCountingStateSpace`.
    """
    pass


class JointBlockCountingReward(Reward, ABC):
    """
    Base class for rewards that are compatible with
    :class:`~phasegen.state_space.JointBlockCountingStateSpace`.
    """
    pass


class TwoLocusBlockCountingReward(Reward, ABC):
    """
    Base class for rewards that are compatible with
    :class:`~phasegen.state_space.TwoLocusBlockCountingStateSpace`. It is not a :class:`JointBlockCountingReward`.
    Although the two-locus state space subclasses the joint one, its block axis encodes loci and not populations, so
    joint-SFS rewards do not evaluate on it, and two-locus rewards do not evaluate on the joint state space.
    """
    pass


class JointSFSReward(JointBlockCountingReward):
    r"""
    Reward for a single bin of the joint (multi-population) site-frequency spectrum. The bin is identified by a
    descendant vector :math:`\mathbf{c} = (k_0, \dots, k_{P-1})`, and the reward of a state :math:`i` is the number
    of lineages (across all demes of residence and loci) whose descendant vector equals :math:`\mathbf{c}`,
    :math:`r(i) = \#\{\text{lineages in } i \text{ with descendant vector } \mathbf{c}\}`.
    """

    _resolves_residence = True

    def __init__(self, config: Tuple[int, ...]) -> None:
        """
        Initialize the reward.

        :param config: The descendant vector identifying the joint SFS bin, i.e. the number of descendants subtended
            from each population.
        """
        self.config: Tuple[int, ...] = tuple(int(c) for c in config)

    def _get(self, state_space: JointBlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        # the two-locus space subclasses the joint one but its block axis encodes loci, not populations, so reject it
        if isinstance(state_space, JointBlockCountingStateSpace) and not isinstance(
                state_space, TwoLocusBlockCountingStateSpace):
            # sum over demes and loci, and select the block corresponding to the descendant vector
            index = state_space.block_index[self.config]
            return state_space.lineages[:, :, :, index].sum(axis=(1, 2))

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def _get_parts(self, state_space: JointBlockCountingStateSpace) -> np.ndarray:
        r"""
        The lineages with this descendant vector resolved by the locus and deme they reside in,
        :math:`r_{l,d}(i) = \#\{\text{lineages of } i \text{ at locus } l \text{ in deme } d \text{ with descendant
        vector } \mathbf{c}\}`.

        :param state_space: state space
        :return: reward parts
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, JointBlockCountingStateSpace) and not isinstance(
                state_space, TwoLocusBlockCountingStateSpace):
            index = state_space.block_index[self.config]

            parts = state_space.lineages[:, :, :, index].astype(float)

            return parts * ~state_space.absorbing[:, None, None]

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the descendant vector.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.config))


class TwoLocusSFSReward(TwoLocusBlockCountingReward):
    r"""
    Reward for one bin of the marginal site-frequency spectrum at a single locus in the two-locus block-counting
    state space. The reward of a state :math:`i` is the number of lineages that subtend exactly ``count`` samples at
    the given ``locus`` (i.e. whose two-locus descendant vector has component ``locus`` equal to ``count``),
    regardless of how many they subtend at the other locus. The two-locus SFS is obtained as the cross-moment
    :math:`\mathbb{E}[R_a R_b]` of two such rewards, one per locus.
    """

    _resolves_residence = True

    def __init__(self, locus: int, count: int) -> None:
        """
        Initialize the reward.

        :param locus: The locus index (0 or 1).
        :param count: The number of subtended samples at ``locus`` identifying the SFS bin.
        """
        self.locus: int = int(locus)
        self.count: int = int(count)

    def _get(self, state_space: 'TwoLocusBlockCountingStateSpace') -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, TwoLocusBlockCountingStateSpace):
            # select the blocks whose descendant count at this locus equals ``count`` and sum over them
            mask = state_space.block_vectors[:, self.locus] == self.count
            return state_space.lineages[:, :, :, mask].sum(axis=(1, 2, 3))

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def _get_parts(self, state_space: 'TwoLocusBlockCountingStateSpace') -> np.ndarray:
        r"""
        The lineages counted by this bin resolved by the deme they reside in, :math:`r_{l,d}(i)` being the number of
        lineages of state :math:`i` in deme :math:`d` whose descendant count at ``locus`` equals ``count``.

        :param state_space: state space
        :return: reward parts
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, TwoLocusBlockCountingStateSpace):
            mask = state_space.block_vectors[:, self.locus] == self.count

            parts = state_space.lineages[:, :, :, mask].sum(axis=3).astype(float)

            return parts * ~state_space.absorbing[:, None, None]

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name, locus and count.

        :return: hash
        """
        return hash((self.__class__.__name__, self.locus, self.count))


class _LocusHeightReward(Reward, ABC):
    """
    Base class for rewards whose part at a locus is the height of that locus.

    :meta private:
    """

    def _get_parts(self, state_space: StateSpace) -> np.ndarray:
        r"""
        The unit reward of a segregating locus split across the demes in proportion to the lineages residing in them,
        :math:`r_{l,d}(i) = \mathbb{1}\{n_l(i) > 1\} n_{l,d}(i) / n_l(i)`, where :math:`n_{l,d}(i)` is the number of
        lineages of state :math:`i` at locus :math:`l` in deme :math:`d` and :math:`n_l(i)` their sum over demes.

        :param state_space: state space
        :return: reward parts
        """
        lineages = state_space.lineages.sum(axis=3)
        per_locus = lineages.sum(axis=2)

        shares = lineages / np.where(per_locus > 0, per_locus, 1)[:, :, None]

        return shares * (per_locus > 1)[:, :, None] * ~state_space.absorbing[:, None, None]


class TreeHeightReward(_LocusHeightReward, LineageCountingReward, BlockCountingReward, JointBlockCountingReward):
    r"""
    Reward for tree height: unit reward on transient states and zero on the absorbing set :math:`B`,
    :math:`r_\text{height}(i) = \mathbb{1}\{i \notin B\}`, so the accumulated reward is the time to absorption. Note
    that when using multiple loci, this will provide the height of the locus with the highest tree.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        # the two-locus state space is absorbed once both loci have reached their MRCA, so it is non-absorbing while
        # either locus still has more than one ancestral lineage (must precede the JointBlockCountingStateSpace
        # branch, which it subclasses)
        if isinstance(state_space, TwoLocusBlockCountingStateSpace):
            carrying = np.stack(
                [state_space.lineages[:, :, :, state_space.block_vectors[:, locus] > 0].sum(axis=(1, 2, 3))
                 for locus in range(2)],
                axis=1
            )
            return np.any(carrying > 1, axis=1).astype(int)

        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace, JointBlockCountingStateSpace)):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return np.any(state_space.lineages.sum(axis=(2, 3)) > 1, axis=1).astype(int)

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class TotalTreeHeightReward(_LocusHeightReward, LineageCountingReward, BlockCountingReward):
    r"""
    Reward based on tree height, unit reward per non-absorbing locus,
    :math:`r(i) = \sum_l \mathbb{1}\{\text{locus } l \text{ has } > 1 \text{ lineage in } i\}`. When using multiple
    loci, this provides the sum of the tree heights over all loci, regardless of whether they are linked or not.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace)):
            # sum over demes and blocks to obtain the number of ancestral lineages per locus
            loci = state_space.lineages.sum(axis=(2, 3))

            # a reward of 1 per locus that is still non-absorbing (more than one lineage), summed over loci
            return np.sum([(loci[:, i] > 1).astype(int) for i in range(state_space.locus_config.n)], axis=0)

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class TotalBranchLengthReward(LineageCountingReward, BlockCountingReward, JointBlockCountingReward):
    r"""
    Reward for total branch length: the lineage count of a state,
    :math:`r_\text{length}(i) = (\#\text{ lineages in } i)` on transient states (zero on the absorbing set), so the
    accumulated reward sums each lineage's duration. When using multiple loci, this provides the sum of the total
    branch lengths over all loci, regardless of whether they are linked or not. Note that due to inherent limitation
    to rewards, we cannot determine the total branch length of the tree with the largest total branch length as done
    in :class:`TreeHeightReward`.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        # the two-locus state space collapses the locus axis into the block vector, so the per-locus lineage counts
        # below do not exist there (must precede the JointBlockCountingStateSpace branch, which it subclasses)
        if isinstance(state_space, TwoLocusBlockCountingStateSpace):
            raise NotImplementedError(
                f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
            )

        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace, JointBlockCountingStateSpace)):
            # sum over demes and blocks
            loci = state_space.lineages.sum(axis=(2, 3))

            # number of loci
            n_loci = state_space.locus_config.n

            # multiply by number of lineages for each locus for which we have more than one lineage
            weights = np.sum([loci[:, i] * (loci[:, i] > 1).astype(int) for i in range(n_loci)], axis=0)

            return weights

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class SFSReward(BlockCountingReward, ABC):
    """
    Base class for site frequency spectrum (SFS) rewards.

    :meta private:
    """

    _resolves_residence = True

    def __init__(self, index: int) -> None:
        """
        Initialize the reward.

        :param index: The index of the SFS bin to use, starting from 1.
        """
        self.index = int(index)

    @abstractmethod
    def _block_sizes(self, n: int) -> List[int]:
        """
        The block sizes, i.e. the numbers of subtended samples, that this bin counts. The bin index is an entry
        ``0, ..., n`` of the spectrum, and the entries without polymorphic blocks count none.

        :param n: The number of lineages.
        :return: The distinct block sizes, empty for an entry without polymorphic blocks.
        :raises ValueError: if the index lies outside ``0, ..., n``.
        """
        pass

    def _check_index(self, n: int) -> None:
        """
        Check that the bin index is an entry ``0, ..., n`` of the spectrum.

        :param n: The number of lineages.
        :raises ValueError: if the index is out of range.
        """
        if not 0 <= self.index <= n:
            raise ValueError(
                f"{self.__class__.__name__} index must lie in 0, ..., {n} for {n} lineages, got {self.index}."
            )

    def _get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        :raises ValueError: if the index lies outside ``0, ..., n``
        """
        if isinstance(state_space, BlockCountingStateSpace):
            blocks = np.array(self._block_sizes(state_space.lineage_config.n), dtype=int) - 1

            # sum over demes, loci and the selected blocks
            return state_space.lineages[:, :, :, blocks].sum(axis=(1, 2, 3))

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def _get_parts(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        r"""
        The blocks counted by this bin resolved by the locus and deme they reside in,
        :math:`r_{l,d}(i) = \sum_{b \in B} a^{(l,d)}_b(i)`, where :math:`B` are the block sizes of the bin and
        :math:`a^{(l,d)}_b(i)` is the number of blocks of size :math:`b` in state :math:`i` at locus :math:`l` in
        deme :math:`d`.

        :param state_space: state space
        :return: reward parts
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, BlockCountingStateSpace):
            blocks = np.array(self._block_sizes(state_space.lineage_config.n), dtype=int) - 1

            parts = state_space.lineages[:, :, :, blocks].sum(axis=3).astype(float)

            return parts * ~state_space.absorbing[:, None, None]

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.index))


class UnfoldedSFSReward(SFSReward, BlockCountingReward):
    r"""
    Reward for one bin of the unfolded site-frequency spectrum: the count of branches subtending exactly ``index``
    samples, :math:`r_{\text{SFS},k}(i) = a_k(i)` with :math:`k = \text{index}` and :math:`a_k(i)` the number of
    :math:`k`-subtending blocks in state :math:`i`. The index lies in :math:`0, \dots, n`, and the monomorphic
    bins :math:`0` and :math:`n` have zero reward.
    """

    def _block_sizes(self, n: int) -> List[int]:
        """
        The block size that this bin counts.

        :param n: The number of lineages.
        :return: The block size ``index``, or none for the monomorphic bins ``0`` and ``n``.
        :raises ValueError: if the index lies outside ``0, ..., n``.
        """
        self._check_index(n)

        return [self.index] if 0 < self.index < n else []


class FoldedSFSReward(SFSReward, BlockCountingReward):
    r"""
    Reward for one bin of the folded site-frequency spectrum: the count of branches subtending ``index`` or
    :math:`n - \text{index}` samples, :math:`r(i) = a_\text{index}(i) + a_{n-\text{index}}(i)` (the two mirror
    classes summed, and a single class when they coincide). The index lies in :math:`0, \dots, n`, and the bins
    :math:`0` and :math:`\lfloor n/2 \rfloor + 1, \dots, n` have zero reward.
    """

    def _block_sizes(self, n: int) -> List[int]:
        """
        The block sizes that this bin counts.

        :param n: The number of lineages.
        :return: The block sizes ``index`` and ``n - index``, ``index`` alone when they coincide, or none for the bins
            ``0`` and ``n // 2 + 1, ..., n``.
        :raises ValueError: if the index lies outside ``0, ..., n``.
        """
        self._check_index(n)

        if not 0 < self.index <= n // 2:
            return []

        if self.index == n - self.index:
            return [self.index]

        return [self.index, n - self.index]


class StateReward(Reward):
    """
    Reward for a specific state in the state space. This is useful for debugging or testing purposes.
    """

    def __init__(self, state: int) -> None:
        """
        Initialize the reward.

        :param state: The state index to reward.
        """
        self.state: int = state

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return (np.arange(state_space.k) == self.state).astype(int)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the state index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.state))


class LineageReward(LineageCountingReward, JointBlockCountingReward):
    """
    Reward for a specific number of lineages present across all demes and loci.
    It tracks, for example, the individual coalescence times.
    """

    def __init__(self, n: int) -> None:
        """
        Initialize the reward.

        :param n: The number of lineages to reward. Must be at least 2.
        """
        if n < 2:
            raise ValueError('Number of lineages must be at least 2.')

        self.n: int = int(n)

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace, JointBlockCountingStateSpace)):
            return (state_space.lineages.sum(axis=(1, 2, 3)) == self.n).astype(int)

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the lineage index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.n))


class DemeReward(LineageCountingReward, BlockCountingReward, JointBlockCountingReward):
    r"""
    Reward the fraction of lineages residing in a specific deme,
    :math:`r(i) = (\#\text{ lineages of } i \text{ in the deme}) / (\#\text{ lineages in } i)`. Combining this reward
    with another reward through :class:`CombinedReward` restricts that reward to the deme, locus by locus, as
    :class:`RestrictedReward` does, and a :class:`SumReward` of several deme rewards restricts it to the union of
    those demes.
    """

    def __init__(self, pop: str) -> None:
        """
        Initialize the reward.

        :param pop: The population id to use.
        """
        self.pop: str = pop

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace, JointBlockCountingStateSpace)):
            # the deme axis of the state space follows the lineage configuration
            pop_index: int = state_space.lineage_config.pop_names.index(self.pop)

            # fraction of total lineages in the population
            fraction = (state_space.lineages.sum(axis=(1, 3))[:, pop_index] / state_space.lineages.sum(axis=(1, 2, 3)))

            return fraction

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the population name.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.pop))


class LocusReward(LineageCountingReward):
    """
    Reward states in which the given locus is still segregating (an indicator that the locus holds more than one
    lineage). Combining this reward with another reward through :class:`CombinedReward` restricts that reward to the
    locus, as :class:`RestrictedReward` does.
    """

    def __init__(self, locus: int) -> None:
        """
        Initialize the reward.

        :param locus: The locus index to use.
        """
        self.locus: int = int(locus)

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, LineageCountingStateSpace):
            return (state_space.lineages.sum(axis=(2, 3))[:, self.locus] > 1).astype(int)

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the population name.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.locus))


class UnitReward(LineageCountingReward, BlockCountingReward, JointBlockCountingReward):
    r"""
    Reward all states with 1 (including absorbing states), :math:`r(i) = 1` for every state :math:`i`.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.ones(state_space.k)


class BlockCountingUnitReward(BlockCountingReward):
    """
    Reward all states with 1 (including absorbing states), and only support block-counting state spaces.
    Passing it forces the block-counting state space.
    """

    def _get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.ones(state_space.k)


class CompositeReward(Reward, ABC):
    """
    Base class for composite rewards.

    :meta private:
    """

    def __init__(self, rewards: List[Reward]) -> None:
        """
        Initialize the composite reward.

        :param rewards: Rewards to composite
        """
        self.rewards: List[Reward] = rewards

    def supports(self, state_space: Type[StateSpace]) -> bool:
        """
        Check if the reward supports the given state space.

        :param state_space: state space
        :return: True if the reward supports the state space, False otherwise
        """
        return all([reward.supports(state_space) for reward in self.rewards])

    @property
    def _resolves_residence(self) -> bool:
        """Whether a member resolves the residence of the rewarded lineages, so that :meth:`_get_parts` does."""
        return any(r._resolves_residence for r in self.rewards)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the hashes of the two rewards.

        :return: hash
        """
        return hash(self.__class__.__name__ + str([hash(reward) for reward in self.rewards]))


class ProductReward(CompositeReward):
    r"""
    The elementwise product of multiple rewards, :math:`r(i) = \prod_j r_j(i)`.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.prod([r._get(state_space) for r in self.rewards], axis=0)

    @property
    def _resolves_residence(self) -> bool:
        """Whether exactly one factor resolves the residence of the rewarded lineages, see :meth:`_get_parts`."""
        return sum(r._resolves_residence for r in self.rewards) == 1

    def _get_parts(self, state_space: StateSpace) -> np.ndarray:
        r"""
        The parts of the single factor that resolves the residence of the rewarded lineages, multiplied by the values
        of the remaining factors, :math:`r_{l,d}(i) = r^{(m)}_{l,d}(i) \prod_{j \neq m} r_j(i)` for the resolving
        factor :math:`m`. Without such a factor, or with several of them, the residence is not determined by the
        factors and the default split applies.

        :param state_space: state space
        :return: reward parts
        """
        resolving = [i for i, r in enumerate(self.rewards) if r._resolves_residence]

        if len(resolving) != 1:
            return super()._get_parts(state_space)

        others = [np.asarray(r._get(state_space), dtype=float)
                  for i, r in enumerate(self.rewards) if i != resolving[0]]

        parts = self.rewards[resolving[0]]._get_parts(state_space)

        if not others:
            return parts

        return parts * np.prod(others, axis=0)[:, None, None]


class SumReward(CompositeReward):
    r"""
    The elementwise sum of multiple rewards, :math:`r(i) = \sum_j r_j(i)`.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.sum([r._get(state_space) for r in self.rewards], axis=0)

    def _get_parts(self, state_space: StateSpace) -> np.ndarray:
        """
        The sum of the parts of the members.

        :param state_space: state space
        :return: reward parts
        """
        return np.sum([r._get_parts(state_space) for r in self.rewards], axis=0)


class RestrictedReward(CompositeReward):
    r"""
    A reward restricted to a locus, to a deme, or to both. The parts :math:`r_{l,d}` into which the reward resolves
    by locus :math:`l` and deme :math:`d` are summed over the other index, so that the restriction to locus
    :math:`l` is :math:`\sum_d r_{l,d}` and the restriction to deme :math:`d` is :math:`\sum_l r_{l,d}`.
    """

    def __init__(self, reward: Reward, locus: int = None, pop: str = None) -> None:
        """
        Initialize the reward.

        :param reward: The reward to restrict.
        :param locus: The locus index to restrict to, ``None`` for no restriction by locus.
        :param pop: The population id to restrict to, ``None`` for no restriction by deme.
        """
        super().__init__([reward])

        self.locus: int = None if locus is None else int(locus)
        self.pop: str = pop

    @property
    def _resolves_residence(self) -> bool:
        """Whether :meth:`_get_parts` resolves the residence, which it does, its parts vanishing outside the
        restricted locus and deme."""
        return True

    def _get_parts(self, state_space: StateSpace) -> np.ndarray:
        """
        The parts of the wrapped reward, those outside the restricted locus and deme set to zero.

        :param state_space: state space
        :return: reward parts
        :raises ValueError: if the locus or the population does not exist
        :raises NotImplementedError: if the state space does not resolve the loci, or if the parts of the wrapped
            reward do not sum to it over the loci
        """
        parts = np.array(self.rewards[0]._get_parts(state_space), dtype=float)

        if self.locus is None:
            values = np.asarray(self.rewards[0]._get(state_space), dtype=float) * ~state_space.absorbing

            if not np.allclose(parts.sum(axis=(1, 2)), values):
                raise NotImplementedError(
                    f"{self.rewards[0].__class__.__name__} does not decompose additively over the "
                    f"{state_space.locus_config.n} loci, so summing its parts over the loci does not recover it and "
                    f"the restriction is ill-posed. Restrict to a single locus as well, or use a reward that is "
                    f"additive over loci such as {TotalTreeHeightReward.__name__} or "
                    f"{TotalBranchLengthReward.__name__}."
                )
        else:
            if parts.shape[1] != state_space.locus_config.n:
                raise NotImplementedError(
                    f'State space {state_space.__class__.__name__} does not resolve loci, so the reward '
                    f'{self.rewards[0].__class__.__name__} cannot be restricted to locus {self.locus}.'
                )

            if not 0 <= self.locus < parts.shape[1]:
                raise ValueError(f"Locus {self.locus} does not exist.")

            parts[:, np.arange(parts.shape[1]) != self.locus, :] = 0

        if self.pop is not None:
            if self.pop not in state_space.lineage_config.pop_names:
                raise ValueError(f"Population {self.pop} does not exist.")

            # the deme axis of the state space follows the lineage configuration
            pop_index = state_space.lineage_config.pop_names.index(self.pop)

            parts[:, :, np.arange(parts.shape[2]) != pop_index] = 0

        return parts

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return self._get_parts(state_space).sum(axis=(1, 2))

    def supports(self, state_space: Type[StateSpace]) -> bool:
        """
        Check if the reward supports the given state space. A restriction by locus requires a state space whose
        locus axis resolves the loci.

        :param state_space: state space
        :return: True if the reward supports the state space, False otherwise
        """
        if self.locus is not None and state_space is not LineageCountingStateSpace:
            return False

        return super().supports(state_space)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name, the wrapped reward, the locus and the population name.

        :return: hash
        """
        return hash((self.__class__.__name__, hash(self.rewards[0]), self.locus, self.pop))


class CombinedReward(ProductReward):
    """
    The product of several rewards, in which a :class:`DemeReward`, a :class:`LocusReward` or a
    :class:`RestrictedReward` member restricts the product of the remaining members as :class:`RestrictedReward`
    does, the restriction of a :class:`RestrictedReward` member acting on its wrapped reward together with them. A
    :class:`SumReward` of :class:`DemeReward` members, or of :class:`LocusReward` members, restricts to the union of
    those demes or loci, as the sum of the single restrictions.
    """

    def __init__(self, rewards: List[Reward]) -> None:
        """
        Initialize the combined reward.

        :param rewards: Rewards to combine
        """
        loci, pops, rest = [], [], []

        # unions of loci or demes, one list per SumReward member made of LocusReward or DemeReward members only
        loci_unions, pop_unions = [], []

        for reward in rewards:
            if isinstance(reward, LocusReward):
                loci.append(reward.locus)
            elif isinstance(reward, DemeReward):
                pops.append(reward.pop)
            elif isinstance(reward, SumReward) and reward.rewards and all(
                    isinstance(r, LocusReward) for r in reward.rewards):
                loci_unions.append([r.locus for r in reward.rewards])
            elif isinstance(reward, SumReward) and reward.rewards and all(
                    isinstance(r, DemeReward) for r in reward.rewards):
                pop_unions.append([r.pop for r in reward.rewards])
            elif isinstance(reward, RestrictedReward):
                loci += [] if reward.locus is None else [reward.locus]
                pops += [] if reward.pop is None else [reward.pop]
                rest.append(reward.rewards[0])
            else:
                rest.append(reward)

        if not loci and not pops and not loci_unions and not pop_unions:
            # copy so we never mutate (or alias) the caller's list
            super().__init__(list(rewards))
            return

        if not rest:
            combined = UnitReward()
        else:
            combined = rest[0] if len(rest) == 1 else ProductReward(rest)

        # the loci first, so that the deme restriction of a reward that is not additive over loci stays well posed
        for locus in loci:
            combined = RestrictedReward(combined, locus=locus)

        for union in loci_unions:
            combined = SumReward([RestrictedReward(combined, locus=locus) for locus in union])

        for pop in pops:
            combined = RestrictedReward(combined, pop=pop)

        for union in pop_unions:
            combined = SumReward([RestrictedReward(combined, pop=pop) for pop in union])

        super().__init__([combined])


class CustomReward(Reward):
    """
    Custom reward based on a user-defined function.
    """

    def __init__(
            self,
            func: Callable[[StateSpace], np.ndarray],
            supports: Callable[[Type[StateSpace]], bool] = lambda _: True
    ) -> None:
        """
        Initialize the custom reward.

        :param func: The function to use to calculate the reward vector.
        :param supports: The function to use to check if the reward supports the state space.
        """
        #: The function to calculate the reward vector
        self.func: Callable[[StateSpace], np.ndarray] = func

        #: The function to check if the reward supports the state space
        self._supports: Callable[[Type[StateSpace]], bool] = supports

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return self.func(state_space)

    def supports(self, state_space: Type[StateSpace]) -> bool:
        """
        Check if the reward supports the given state space.

        :param state_space: state space
        :return: True if the reward supports the state space, False otherwise
        """
        return self._supports(state_space)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the identities of the function and of the support predicate.

        :return: hash
        """
        return hash((self.__class__.__name__, id(self.func), id(self._supports)))
