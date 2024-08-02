"""
Reward generation.
"""

from abc import abstractmethod, ABC
from typing import List, Callable, Tuple, Dict, Iterable, Type

import numpy as np

from .state_space import StateSpace, LineageCountingStateSpace, BlockCountingStateSpace


class Reward(ABC):
    """
    Base class for reward generation.
    """

    @abstractmethod
    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.
        
        :param state_space: state space
        :return: reward vector
        """
        pass

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

    @staticmethod
    def support(state_space: Type[StateSpace], rewards: Iterable['Reward']) -> bool:
        """
        Check if the rewards support the given state space.

        :param state_space: state space
        :param rewards: rewards
        :return: True if the rewards support the state space, False otherwise
        """
        return all([reward.supports(state_space) for reward in rewards])


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


class TreeHeightReward(LineageCountingReward, BlockCountingReward):
    """
    Reward for tree height. Note that when using multiple loci, this will provide the
    height of the locus with the highest tree.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace)):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return np.any(state_space.lineages.sum(axis=(2, 3)) > 1, axis=1).astype(int)

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class TotalTreeHeightReward(LineageCountingReward, BlockCountingReward):
    """
    Reward based on tree height. When using multiple loci, this will provide the sum of the tree 
    heights over all loci, regardless of whether they are linked or not.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace)):
            return np.sum([LocusReward(i)._get(state_space) for i in range(state_space.locus_config.n)], axis=0)

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class TotalBranchLengthReward(LineageCountingReward, BlockCountingReward):
    """
    Reward for total branch length. When using multiple loci, this will provide the sum of the
    total branch lengths over all loci, regardless of whether they are linked or not. Note that due to
    inherent limitation to rewards, we cannot determine the total branch length of the tree with
    the largest total branch length as done in :class:`TreeHeightReward`.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace)):
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

    def __init__(self, index: int):
        """
        Initialize the reward.

        :param index: The index of the SFS bin to use, starting from 1.
        """
        self.index = int(index)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.index))


class UnfoldedSFSReward(SFSReward, BlockCountingReward):
    """
    Reward for unfolded site frequency spectrum (SFS).
    """

    def _get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, BlockCountingStateSpace):
            # sum over demes and loci, and select block
            return state_space.lineages[:, :, :, self.index - 1].sum(axis=(1, 2))

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class FoldedSFSReward(SFSReward, BlockCountingReward):
    """
    Reward for folded site frequency spectrum (SFS).
    """

    def _get_indices(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the indices of the blocks that make up the folded SFS bin.

        :param state_space: state space
        :return: indices
        """
        if self.index == state_space.lineage_config.n - self.index:
            return np.array([self.index - 1])

        return np.array([self.index - 1, state_space.lineage_config.n - self.index - 1])

    def _get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, BlockCountingStateSpace):
            blocks = self._get_indices(state_space)

            # sum over demes and loci, and select block
            return state_space.lineages[:, :, :, blocks].sum(axis=(1, 2, 3))

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class LineageReward(LineageCountingReward):
    """
    Reward for a specific number of lineages present in across all demes and loci.
    This reward can be used to, for example, track the individual coalescent times.
    """

    def __init__(self, n: int):
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
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace)):
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


class DemeReward(LineageCountingReward, BlockCountingReward):
    """
    Reward fraction of lineages in a specific deme. Taking the product of this reward with another reward
    will result in a reward that only considers the specified deme. Use :class:`SumReward` to marginalize over
    several demes.
    """

    def __init__(self, pop: str):
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
        if isinstance(state_space, (LineageCountingStateSpace, BlockCountingStateSpace)):
            # get the index of the population
            pop_index: int = state_space.epoch.pop_names.index(self.pop)

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
    Reward fraction of lineages in a specific locus. Taking the product of this reward with another reward
    will result in a reward that only considers the specified locus.
    """

    def __init__(self, locus: int):
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


class UnitReward(LineageCountingReward, BlockCountingReward):
    """
    Reward all states with 1 (including absorbing states).
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
    This reward can be used to force usage of the block-counting state space.
    """

    def _get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.ones(state_space.k)


class TotalBranchLengthLocusReward(LocusReward, LineageCountingReward):
    """
    Reward for total branch length per locus. This is needed as the taking the product of
    :class:`TotalBranchLengthReward` and :class:`LocusReward` will not work as expected (see
    :class:`CombinedReward`).
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, LineageCountingStateSpace):
            # number of branches for focal locus
            n_branches = state_space.lineages.sum(axis=(2, 3))[:, self.locus]

            # no reward for loci with less than two branches
            n_branches[n_branches < 2] = 0

            return n_branches

        raise NotImplementedError(
            f'Unsupported state space type for reward {self.__class__.__name__}: {state_space.__class__.__name__}'
        )


class CompositeReward(Reward, ABC):
    """
    Base class for composite rewards.

    :meta private:
    """

    def __init__(self, rewards: List[Reward]):
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

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the hashes of the two rewards.

        :return: hash
        """
        return hash(self.__class__.__name__ + str([hash(reward) for reward in self.rewards]))


class ProductReward(CompositeReward):
    """
    The product of multiple rewards.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.prod([r._get(state_space) for r in self.rewards], axis=0)


class SumReward(CompositeReward):
    """
    The sum of multiple rewards.
    """

    def _get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.sum([r._get(state_space) for r in self.rewards], axis=0)


class CombinedReward(ProductReward):
    """
    Class extending ProductReward to allow for more intuitive combination of rewards.
    """
    #: Dictionary of reward combinations
    combinations: Dict[Tuple[Reward, Reward], Callable[[Reward, Reward], Reward]] = {
        (TotalBranchLengthReward, LocusReward): lambda r1, r2: TotalBranchLengthLocusReward(r2.locus)
    }

    def __init__(self, rewards: List[Reward]):
        """
        Initialize the combined reward.

        :param rewards: Rewards to combine
        """
        # replace rewards with combined rewards if possible
        for (c1, c2), comb in CombinedReward.combinations.items():
            # keep looping until we have no rewards to combine
            while any([isinstance(r, c1) for r in rewards]) and any([isinstance(r, c2) for r in rewards]):
                # get first occurrence of r1 and r2
                r1 = next(r for r in rewards if isinstance(r, c1))
                r2 = next(r for r in rewards if isinstance(r, c2))

                # remove first occurrence of r1 and r2
                rewards.remove(r1)
                rewards.remove(r2)

                # add combined reward
                rewards.append(comb(r1, r2))

        super().__init__(rewards)


class CustomReward(Reward):
    """
    Custom reward based on a user-defined function.
    """

    def __init__(
            self,
            func: Callable[[StateSpace], np.ndarray],
            supports: Callable[[Type[StateSpace]], bool] = lambda _: True
    ):
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
        Calculate the hash of the class name and the index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.func))
