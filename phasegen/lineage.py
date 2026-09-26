"""
Lineage configuration
"""

import logging
import numbers
from typing import Dict, List, Iterable

import numpy as np

logger = logging.getLogger('phasegen')


class LineageConfig:
    """
    Class to hold the configuration for the number of lineages.
    """

    def __init__(self, n: int | Dict[str, int] | List[int] | np.ndarray) -> None:
        """
        Initialize the population configuration.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
            or dictionary with population names as keys and number of lineages as values for multiple populations.
            By default, the populations are named 'pop_0', 'pop_1', etc.
        :raises TypeError: If a lineage count is not a number.
        :raises ValueError: If a lineage count is negative or not integral, or fewer than two lineages are given.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if isinstance(n, dict):
            # we have a dictionary
            n_lineages = {k: self._to_count(v) for k, v in n.items()}

        elif isinstance(n, Iterable):
            # we have an iterable
            n_lineages = {f"pop_{i}": self._to_count(n) for i, n in enumerate(n)}

        else:
            # assume we have a scalar
            n_lineages = dict(pop_0=self._to_count(n))

        #: Number of lineages per deme.
        self.lineages: np.ndarray = np.array(list(n_lineages.values()))

        #: Total number of lineages.
        self.n: int = sum(list(n_lineages.values()))

        if self.n < 2:
            raise ValueError("Number of lineages must be at least 2.")

        #: Number of populations.
        self.n_pops: int = len(n_lineages)

        #: Names of populations.
        self.pop_names: List[str] = list(n_lineages.keys())

    @staticmethod
    def _to_count(value: int | float) -> int:
        """
        Convert a lineage count to an integer.

        :param value: The lineage count, a non-negative integral number.
        :return: The lineage count as an integer.
        :raises TypeError: If the value is not a number.
        :raises ValueError: If the value is negative or not integral.
        """
        if not isinstance(value, numbers.Real):
            raise TypeError(f"Lineage counts must be numbers, got {type(value).__name__}.")

        if value < 0 or not float(value).is_integer():
            raise ValueError(f"Lineage counts must be non-negative integers, got {value}.")

        return int(value)

    @property
    def lineage_dict(self) -> Dict[str, int]:
        """
        Get a dictionary with the number of lineages per population.

        :return: Number of lineages per population.
        """
        return dict(zip(self.pop_names, self.lineages))

    def _get_initial_states(self, s: 'StateSpace') -> np.ndarray:
        r"""
        Get the unnormalized population factor of :attr:`StateSpace.alpha <phasegen.state_space.StateSpace.alpha>`,
        the indicator of the states whose per-deme lineage counts match this configuration.

        :param s: State space
        :return: Initial state vector
        """
        # determine the states that correspond to the population configuration
        # it is enough here to focus on the first lineage class
        return (s.lineages[:, :, :, 0] == self.lineages).all(axis=(1, 2)).astype(int)

    def __eq__(self, other) -> bool:
        """
        Check if two lineage configurations are equal, including the order of the populations, which fixes the deme
        axis of a state space and the demes the initially unlinked lineages are taken from.

        :param other: Other lineage configuration
        :return: Whether the two lineage configurations are equal
        """
        return list(self.lineage_dict.items()) == list(other.lineage_dict.items())
