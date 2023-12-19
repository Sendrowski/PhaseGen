from abc import abstractmethod, ABC

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
        if isinstance(state_space, DefaultStateSpace):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return np.diag((np.dot(state_space.states, np.arange(1, state_space.m + 1)).sum(axis=1) > 1).astype(int))

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
        if isinstance(state_space, DefaultStateSpace):
            # get total number of lineages per state
            lineages = np.dot(state_space.states, np.arange(1, state_space.m + 1)).sum(axis=1)

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

        :param index: The index of the SFS to use.
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
