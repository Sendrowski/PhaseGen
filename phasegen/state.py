import numpy as np


class State:
    """
    State utility class.
    """
    #: Axis for loci.
    LOCUS = 0

    #: Axis for demes.
    DEME = 1

    #: Axis for lineage blocks.
    BLOCK = 2

    @staticmethod
    def is_absorbing(state: np.ndarray) -> bool:
        """
        Whether a state is absorbing.

        :param state: State array.
        :return: Whether the state is absorbing.
        """
        return np.all(np.sum(state * np.arange(1, state.shape[2] + 1)[::-1], axis=(1, 2)) == 1)
