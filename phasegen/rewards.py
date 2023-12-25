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
        Get the reward matrix.
        
        :param state_space: state space
        :return: reward matrix
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
    Reward based on tree height.
    """

    def get(self, state_space: DefaultStateSpace) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return np.diag((state_space.states.sum(axis=(1, 2)) > 1).astype(int))

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')


class TotalBranchLengthReward(NonDefaultReward):
    """
    Reward based on total branch length.
    """

    def get(self, state_space: DefaultStateSpace) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # get total number of lineages per state
            lineages = state_space.states.sum(axis=(1, 2))

            # for fewer than 2 lineages, we have an absorbing state which has a reward of 0
            lineages[lineages < 2] = 0

            return np.diag(lineages)

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
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, BlockCountingStateSpace):
            return np.diag(state_space.states[:, :, self.index].sum(axis=1))

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
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, (DefaultStateSpace, BlockCountingStateSpace)):
            # get the index of the population
            pop_index: int = state_space.epoch.pop_names.index(self.pop)

            # fraction of total lineages in the population
            fraction = (state_space.states.sum(axis=2)[:, pop_index] / state_space.states.sum(axis=(1, 2)))

            return np.diag(fraction)

        raise NotImplementedError(f'Unsupported state space type: {type(state_space)}')

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the population name.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.pop))


class UnitReward(DefaultReward):
    """
    Rewards all states with 1.
    """

    def get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        """
        return np.diag(np.ones(state_space.k))


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
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        """
        return np.prod([r.get(state_space) for r in self.rewards], axis=0)


class CustomReward(Reward):
    """
    Custom reward based on a user-defined function.
    """

    def __init__(self, func: Callable[[StateSpace], np.ndarray]):
        """
        Initialize the custom reward.

        :param func: The function to use to calculate the reward.
        """
        self.func: Callable[[StateSpace], np.ndarray] = func

    def get(self, state_space: StateSpace) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :return: reward matrix
        """
        return self.func(state_space)

    def __hash__(self) -> int:
        """
        Calculate the hash of the class name and the index.

        :return: hash
        """
        return hash(self.__class__.__name__ + str(self.func))
