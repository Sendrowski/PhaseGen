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
    ) -> None:
        """
        Initialize the locus configuration.

        :param n: Number of loci. Either 1 or 2.
        :param n_unlinked: Number of lineages that are initially unlinked between loci. Defaults to 0 meaning that all
            lineages are initially linked between loci so that the loci are completely linked. It must not exceed the
            number of lineages.
        :param recombination_rate: Recombination rate between loci.
        :raises ValueError: If ``n`` is not a positive integer, ``n_unlinked`` is not a non-negative integer, or the
            recombination rate is negative.
        :raises NotImplementedError: If ``n`` exceeds 2.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        if n < 1 or not float(n).is_integer():
            raise ValueError(f"Number of loci must be a positive integer, got {n}.")

        if n > 2:
            raise NotImplementedError("Only 1 or 2 loci are currently supported.")

        if n_unlinked < 0 or not float(n_unlinked).is_integer():
            raise ValueError(f"Number of unlinked lineages must be a non-negative integer, got {n_unlinked}.")

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
        r"""
        Get the unnormalized locus factor of :attr:`StateSpace.alpha <phasegen.state_space.StateSpace.alpha>`, the
        indicator of the states consistent with the requested number of loci and initially linked lineages.

        :param s: State space
        :return: Initial state vector
        """
        if self.n == 1:
            # every lineage is on the same locus
            return np.ones(s.k)

        # number of lineages linked between loci
        n_linked = s.lineage_config.n - self.n_unlinked

        # sum over demes and lineage blocks, and require all loci to have ``n_linked`` linked lineages
        return (s.linked.sum(axis=(2, 3)) == n_linked).all(axis=1).astype(int)

    def __eq__(self, other) -> bool:
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
