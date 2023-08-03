from typing import Dict, List, Iterable

import numpy as np


class PopulationConfig:
    """
    Class to hold the configuration of a population such as the number of lineages.
    """

    def __init__(self, n: int | Dict[str, int] | List[int] | np.ndarray):
        """
        Initialize the population configuration.

        :param n: Number of lineages. Either a single integer if only one population, or a list of integers
        or a dictionary with population names as keys and number of lineages as values.
        """
        if not isinstance(n, dict):

            # assume we have a scalar
            if not isinstance(n, Iterable):
                n_lineages = dict(pop_0=n)
            else:
                # we have an iterable
                n_lineages = {f"pop_{i}": i for i in n}
        else:
            # we have a dictionary
            n_lineages = n

        #: Number of lineages per deme
        self.lineages: np.array = np.array(list(n_lineages.values()))

        #: Total number of lineages
        self.n: int = sum(list(n_lineages.values()))

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

    def get_alpha(self, s: 'StateSpace') -> np.ndarray:
        """
        Get initial state vector for the population configuration.

        :param s: State space
        :return: Initial state vector
        """
        # determine the states that correspond to the population configuration
        return (s.states[:, :, 0] == self.lineages).all(axis=1).astype(int)
