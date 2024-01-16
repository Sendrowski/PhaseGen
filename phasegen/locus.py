import logging

import numpy as np

logger = logging.getLogger('phasegen')


class LocusConfig:
    """
    Class to hold the configuration of the number of loci and with how many independent loci to start.
    """

    def __init__(
            self,
            n: int = 1,
            n_start: int = 1,
            recombination_rate: float = 0
    ):
        """
        Initialize the locus configuration.

        :param n: Number of loci
        :param n_start: Number of loci to start with
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if n < 1:
            raise ValueError("Number of loci must be at least 1.")

        if n > 2:
            raise NotImplementedError("Only 1 or 2 loci are currently supported.")

        if n_start < 1:
            raise ValueError("Number of loci to start with must be at least 1.")

        if recombination_rate < 0:
            raise ValueError("Recombination rate must be non-negative.")

        if n_start > n:
            raise ValueError("Number of loci to start with must be less than or equal to the total number of loci.")

        #: Number of loci
        self.n = n

        #: Number of loci to start with
        self.n_start = n_start

        #: Recombination rate
        self.recombination_rate = recombination_rate

    def get_initial_states(self, s: 'StateSpace') -> np.ndarray:
        """
        Get initial state vector for the locus configuration.
        TODO test this

        :param s: State space
        :return: Initial state vector
        """
        if self.n == 1:
            # every lineage is on the same locus
            return np.ones(s.k)

        if self.n_start == 1:
            # all lineages are shared
            # sum over demes and lineage blocks, and require all loci to have all lineages shared
            return (s.shared.sum(axis=(2, 3)) == s.pop_config.n).all(axis=1).astype(int)

        if self.n_start == 2:
            # no lineage is shared
            # sum over demes and lineage blocks, and require all loci to have zero shared lineages
            return (s.shared.sum(axis=(2, 3)) == 0).all(axis=1).astype(int)
