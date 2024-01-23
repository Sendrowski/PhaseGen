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
            n_unlinked: int = 0,
            recombination_rate: float = 0,
            allow_coalescence: bool = True
    ):
        """
        Initialize the locus configuration.

        :param n: Number of loci
        :param n_unlinked: Number of lineages that are initially unlinked between loci. Defaults to 0 meaning that all
            lineages are initially linked between loci so that the loci are completely linked.
        :param recombination_rate: Recombination rate between loci.
        :param allow_coalescence: Whether to allow coalescence between loci. Defaults to True. If False, then their
            will be no common ancestor between loci once they recombine.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if n < 1:
            raise ValueError("Number of loci must be at least 1.")

        if n > 2:
            raise NotImplementedError("Only 1 or 2 loci are currently supported.")

        if n_unlinked < 0:
            raise ValueError("Number of unlinked lineages must be non-negative.")

        if recombination_rate < 0:
            raise ValueError("Recombination rate must be non-negative.")

        #: Number of loci
        self.n = n

        #: Number of loci to start with
        self.n_unlinked = n_unlinked

        #: Recombination rate
        self.recombination_rate = recombination_rate

        #: Whether to allow coalescence between loci
        self.allow_coalescence = allow_coalescence

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

        # number of lineages linked between loci
        n_linked = max(s.pop_config.n - self.n_unlinked, 0)

        # sum over demes and lineage blocks, and require all loci to have `n_linked` linked lineages
        return (s.linked.sum(axis=(2, 3)) == n_linked).all(axis=1).astype(int)
