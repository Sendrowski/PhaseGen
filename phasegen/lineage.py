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

    The following example samples two lineages in ``pop_0`` and one in ``pop_1``.

    ::

        coal = pg.Coalescent(n=pg.LineageConfig({'pop_0': 2, 'pop_1': 1}), demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
        ))
    """

    def __init__(self, n: int | Dict[str, int] | List[int] | np.ndarray) -> None:
        """
        Initialize the population configuration.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
            or dictionary with population names as keys and number of lineages as values for multiple populations.
            By default, the populations are named 'pop_0', 'pop_1', etc.
        :raises TypeError: If a lineage count is not a number or a population name is not a string.
        :raises ValueError: If a lineage count is negative or not integral, or fewer than two lineages are given.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if isinstance(n, dict):
            # we have a dictionary
            names = [k for k in n if not isinstance(k, str)]

            if names:
                raise TypeError(f"Population names must be strings, got {names}.")

            n_lineages = {k: self._to_count(v) for k, v in n.items()}

        elif isinstance(n, Iterable) and not (isinstance(n, np.ndarray) and n.ndim == 0):
            # we have an iterable
            n_lineages = {f"pop_{i}": self._to_count(n) for i, n in enumerate(n)}

        else:
            # assume we have a scalar, possibly a 0-d array
            n_lineages = dict(pop_0=self._to_count(n.item() if isinstance(n, np.ndarray) else n))

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
