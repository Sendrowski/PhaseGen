from abc import abstractmethod, ABC

import numpy as np

from .state_space import StateSpace, DefaultStateSpace, BlockCountingStateSpace


class Reward(ABC):
    """
    Base class for reward generation.
    """

    @abstractmethod
    def get(self, state_space: StateSpace, invert: bool = True) -> np.ndarray:
        """
        Get the reward matrix.
        
        :param state_space: state space
        :param invert: Whether to invert the reward matrix.
        :return: reward matrix
        """
        pass

    @staticmethod
    def _pad(x: np.ndarray) -> np.ndarray:
        """
        Pad a matrix with a row and column of zeros or a vector with a zero.

        :param x: The matrix or vector to pad.
        :return: The padded matrix or vector.
        """
        if x.ndim == 1:
            return np.pad(x, (0, 1), mode='constant', constant_values=0)

        return np.pad(x, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    @staticmethod
    def _invert(r: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Invert the reward matrix or reward vector.

        :param r: The reward matrix or reward vector.
        :param invert: Whether to invert the reward matrix or reward vector.
        :return: The invert reward matrix or reward vector.
        """
        if not invert:
            return r
        
        r = r.copy()

        if r.ndim == 2:
            r = r[np.diag_indices(r.shape[0])]

            return np.diag(Reward._invert(r))

        r_inv = r.copy().astype(float)
        r_inv[r != 0] = 1 / r_inv[r != 0]

        return r_inv

    def __hash__(self) -> int:
        """
        Hash the class name as this class is stateless.
        
        :return: hash 
        """
        return hash(self.__class__.__name__)


class TreeHeightReward(Reward):
    """
    Reward based on tree height.
    """

    def get(self, state_space: StateSpace, invert: bool = False) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :param invert: Whether to invert the reward matrix.
        :return: reward matrix
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, DefaultStateSpace):
            # a reward of 1 for non-absorbing states and 0 for absorbing states
            return np.diag((np.dot(state_space.states, np.arange(1, state_space.m + 1)).sum(axis=1) > 1).astype(int))

        if isinstance(state_space, BlockCountingStateSpace):
            raise NotImplementedError('Tree height reward not implemented for block counting state space')
        
        raise NotImplementedError(f'Unknown state space type: {type(state_space)}')


class TotalBranchLengthReward(Reward):
    """
    Reward based on total branch length.
    """

    def get(self, state_space: StateSpace, invert: bool = False) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :param invert: Whether to invert the reward matrix.
        :return: reward matrix
        :raises: NotImplementedError if the state space is not supported
        """
        if isinstance(state_space, DefaultStateSpace):

            # get total number of lineages per state
            lineages = np.dot(state_space.states, np.arange(1, state_space.m + 1)).sum(axis=1)

            # if we have fewer than 2 lineages, we have an absorbing state
            # which has a reward of 0
            lineages[lineages < 2] = 0

            return np.diag(lineages)

        if isinstance(state_space, BlockCountingStateSpace):
            raise NotImplementedError('Total branch length reward not implemented for block counting state space')

        raise NotImplementedError(f'Unknown state space type: {type(state_space)}')


class SFSReward(Reward):
    """
    Reward based on Site Frequency Spectrum.
    """

    def __init__(self, index: int = None):
        """
        Initialize the reward.

        :param index: The index of the SFS to use.
        """
        self.index = index

    def get(self, state_space: StateSpace, invert: bool = False) -> np.ndarray:
        """
        Get the reward matrix.

        :param state_space: state space
        :param invert: Whether to invert the reward matrix.
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
