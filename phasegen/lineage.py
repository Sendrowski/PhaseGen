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
        or a dictionary with population names as keys and number of lineages as values.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if isinstance(n, dict):
            # we have a dictionary
            n_lineages = n

        elif isinstance(n, Iterable):
            # we have an iterable
            n_lineages = {f"pop_{i}": n for i, n in enumerate(n)}

        else:
            # assume we have a scalar
            n_lineages = dict(pop_0=n)

        #: Number of lineages per deme
        self.lineages: np.array = np.array(list(n_lineages.values()))

        #: Total number of lineages
        self.n: int = sum(list(n_lineages.values()))

        # warn if the number of lineages is large
        if self.n > 20:
            self._logger.warning(f"Total number of lineages ({self.n}) is large. "
                                 f"Note that the state space and thus the runtime "
                                 f"grows exponentially with the number of lineages.")

        #: Number of populations
        self.n_pops = len(n_lineages)

        #: Names of populations
        self.pop_names = list(n_lineages.keys())

    @property
    def lineage_dict(self) -> Dict[str, int]:
        """
        Get a dictionary with the number of lineages per population.

        :return: Number of lineages per population.
        """
        return dict(zip(self.pop_names, self.lineages))

    def get_initial_states(self, s: 'StateSpace') -> np.ndarray:
        """
        Get initial state vector for the population configuration.

        :param s: State space
        :return: Initial state vector
        """
        # determine the states that correspond to the population configuration
        # it is enough here to focus on the first lineage class
        return (s.states[:, :, :, 0] == self.lineages).all(axis=(1, 2)).astype(int)