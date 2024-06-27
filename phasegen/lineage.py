"""
Lineage configuration
"""

import logging
from typing import Dict, List, Iterable

import numpy as np

logger = logging.getLogger('phasegen')


class LineageConfig:
    """
    Class to hold the configuration for the number of lineages.
    """

    def __init__(self, n: int | Dict[str, int] | List[int] | np.ndarray):
        """
        Initialize the population configuration.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
            or dictionary with population names as keys and number of lineages as values for multiple populations.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if isinstance(n, dict):
            # we have a dictionary
            n_lineages = {k: int(v) for k, v in n.items()}

        elif isinstance(n, Iterable):
            # we have an iterable
            n_lineages = {f"pop_{i}": int(n) for i, n in enumerate(n)}

        else:
            # assume we have a scalar
            n_lineages = dict(pop_0=int(n))

        #: Number of lineages per deme.
        self.lineages: np.ndarray = np.array(list(n_lineages.values()))

        #: Total number of lineages.
        self.n: int = sum(list(n_lineages.values()))

        #: Number of populations.
        self.n_pops: int = len(n_lineages)

        #: Names of populations.
        self.pop_names: List[str] = list(n_lineages.keys())

    @property
    def lineage_dict(self) -> Dict[str, int]:
        """
        Get a dictionary with the number of lineages per population.

        :return: Number of lineages per population.
        """
        return dict(zip(self.pop_names, self.lineages))

    def _get_initial_states(self, s: 'StateSpace') -> np.ndarray:
        """
        Get initial state vector for the population configuration.

        :param s: State space
        :return: Initial state vector
        """
        # determine the states that correspond to the population configuration
        # it is enough here to focus on the first lineage class
        return (s.lineages[:, :, :, 0] == self.lineages).all(axis=(1, 2)).astype(int)

    def __eq__(self, other):
        """
        Check if two lineage configurations are equal.

        :param other: Other lineage configuration
        :return: Whether the two lineage configurations are equal
        """
        return self.lineage_dict == other.lineage_dict
