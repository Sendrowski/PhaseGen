"""
Locus configuration class.
"""

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
            recombination_rate: float = 0
    ):
        """
        Initialize the locus configuration.

        :param n: Number of loci. Either 1 or 2.
        :param n_unlinked: Number of lineages that are initially unlinked between loci. Defaults to 0 meaning that all
            lineages are initially linked between loci so that the loci are completely linked.
        :param recombination_rate: Recombination rate between loci.
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

        #: Number of loci.
        self.n: int = int(n)

        #: Number of loci to start with.
        self.n_unlinked: int = int(n_unlinked)

        #: Recombination rate.
        self.recombination_rate: float = recombination_rate

        #: Whether to allow coalescence between loci, deprecated
        self._allow_coalescence: float = True

    def _get_initial_states(self, s: 'StateSpace') -> np.ndarray:
        """
        Get (not normalized) initial state vector for the locus configuration.

        :param s: State space
        :return: Initial state vector
        """
        if self.n == 1:
            # every lineage is on the same locus
            return np.ones(s.k)

        # number of lineages linked between loci
        n_linked = max(s.lineage_config.n - self.n_unlinked, 0)

        # sum over demes and lineage blocks, and require all loci to have ``n_linked`` linked lineages
        return (s.linked.sum(axis=(2, 3)) == n_linked).all(axis=1).astype(int)

    def __eq__(self, other):
        """
        Check if two locus configurations are equal.

        :param other: Other locus configuration
        :return: Whether the two locus configurations are equal
        """
        return (
                self.n == other.n
                and self.n_unlinked == other.n_unlinked
                and self.recombination_rate == other.recombination_rate
                and self._allow_coalescence == other._allow_coalescence
        )
