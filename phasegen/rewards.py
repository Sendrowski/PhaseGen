from abc import abstractmethod, ABC
from typing import List, Callable

import numpy as np

from .state_space import StateSpace, DefaultStateSpace, BlockCountingStateSpace


class Reward(ABC):
    """
    Base class for reward generation.
    """

    @abstractmethod
    def get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.
        
        :param state_space: state space
        :return: reward vector
        """
        pass

    def __hash__(self) -> int:
        """
        Hash the class name as this class is stateless.
        
        :return: hash 
        """
        return hash(self.__class__.__name__)

    def prod(self, *rewards: 'Reward') -> 'Reward':
        """
        Union of two rewards.

        :param rewards: Rewards to union
        :return: Union of the rewards
        """
        return ProductReward([self] + list(rewards))


class DefaultReward(Reward, ABC):
    """
    Default reward where all non-absorbing states have a reward of 1.
    """
    pass


class NonDefaultReward(Reward, ABC):
    """
    Non-default reward where not all non-absorbing states have a reward of 1.
    """
    pass


class TreeHeightReward(DefaultReward):
    """
    Reward based on tree height. Note that when using multiple loci, this will provide the
    height of the locus with the highest tree.
    """

    def get(self, state_space: DefaultStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return np.any(state_space.states.sum(axis=(2, 3)) > 1, axis=1).astype(int)

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')


class TotalTreeHeightReward(TreeHeightReward):
    """
    Reward based on tree height. When using multiple loci, this will provide the sum of the
    heights of all loci, regardless of whether they are linked or not.
    """

    def get(self, state_space: DefaultStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return state_space.states.sum(axis=(1, 2, 3)) * super().get(state_space)

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')


class TotalBranchLengthReward(TreeHeightReward):
    """
    Reward based on total branch length. When using multiple loci, this will provide the sum of the
    total branch lengths of all loci, regardless of whether they are linked or not. Note that due to
    inherent limitation to rewards, we cannot determine the total branch length of the tree with
    the largest total branch length as done in :class:`TreeHeightReward`.
    """

    def get(self, state_space: DefaultStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # get total number of lineages per state and multiply by tree height reward
            return state_space.states.sum(axis=(1, 2, 3)) * super().get(state_space)

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')


class SFSReward(NonDefaultReward):
    """
    Reward based on Site Frequency Spectrum.
    """

    def __init__(self, index: int = None):
        """
        Initialize the reward.

        :param index: The index of the SFS bin to use.
        """
        self.index = index

    def get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, BlockCountingStateSpace):
            # sum over demes and average over loci
            return state_space.states[:, :, :, self.index].sum(axis=2).mean(axis=1)

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.index))


class DemeReward(NonDefaultReward):
    """
    Reward based on fraction of lineages in a specific deme. Taking the product of this reward with another reward
    will result in a reward that only considers the specified deme.
    """

    def __init__(self, pop: str):
        """
        Initialize the reward.

        :param pop: The population id to use.
        """
        self.pop: str = pop

    def get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # get the index of the population
            pop_index: int = state_space.epoch.pop_names.index(self.pop)

            # fraction of total lineages in the population
            fraction = (state_space.states.sum(axis=(1, 3))[:, pop_index] / state_space.states.sum(axis=(1, 2, 3)))

            return fraction

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the population name.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.pop))


class LocusReward(NonDefaultReward):
    """
    Reward based on fraction of lineages in a specific locus. Taking the product of this reward with another reward
    will result in a reward that only considers the specified locus.
    """

    def __init__(self, locus: int):
        """
        Initialize the reward.

        :param locus: The locus index to use.
        """
        self.locus: int = locus

    def get(self, state_space: BlockCountingStateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # fraction of total lineages in the population
            fraction = (state_space.states.sum(axis=(1, 3))[:, :, self.locus] / state_space.states.sum(axis=(1, 2, 3)))

            return fraction

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the population name.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.locus))


class UnitReward(DefaultReward):
    """
    Rewards all states with 1.
    """

    def get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.ones(state_space.k)


class CompositeReward(Reward, ABC):
    """
    Base class for composite rewards.
    """

    def __init__(self, rewards: List[Reward]):
        """
        Initialize the composite reward.

        :param rewards: Rewards to composite
        """
        self.rewards: List[Reward] = rewards

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

    def get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return np.prod([r.get(state_space) for r in self.rewards], axis=0)


class CustomReward(Reward):
    """
    Custom reward based on a user-defined function.
    """

    def __init__(self, func: Callable[[StateSpace], np.ndarray]):
        """
        Initialize the custom reward.

        :param func: The function to use to calculate the reward vector.
        """
        self.func: Callable[[StateSpace], np.ndarray] = func

    def get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward vector.

        :param state_space: state space
        :return: reward vector
        """
        return self.func(state_space)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.func))
