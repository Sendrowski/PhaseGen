from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from typing import List

import numpy as np
from scipy.linalg import expm, inv

from .coalescent_models import CoalescentModel
from .demography import TimeHomogeneousDemography
from .population import PopConfig


class StateSpace(ABC):
    """
    State space.
    """

    def __init__(
            self,
            pop_config: PopConfig,
            model: CoalescentModel,
            demography: TimeHomogeneousDemography
    ):
        """
        Create a rate matrix.

        :param pop_config: Population configuration.
        :param model: Coalescent model.
        :param demography: Time homogeneous demography (we can only construct a state space
            for a fixed demography).
        """
        #: Coalescent model
        self.model: CoalescentModel = model

        #: Population configuration
        self.pop_config: PopConfig = pop_config

        # we first determine the non-zero states by using default values for the demography
        self.demography = TimeHomogeneousDemography(
            pop_sizes={p: 1 for p in demography.pop_names},
            migration_rates={(p, q): 1 for p, q in product(demography.pop_names, demography.pop_names)}
        )

        # get the rate matrix for the default demography
        default_rate_matrix = np.fromfunction(
            np.vectorize(self._matrix_indices_to_rates, otypes=[float]),
            (self.k, self.k),
            dtype=int
        )

        # indices of non-zero rates
        # this improves performance when computing the rate matrix
        self._non_zero_states = np.where(default_rate_matrix != 0)

        #: Demography
        self.demography: TimeHomogeneousDemography = demography

    @cached_property
    @abstractmethod
    def states(self) -> np.ndarray:
        """
        Get the states.
        """
        pass

    @cached_property
    def s(self) -> np.ndarray:
        """
        Get exit rate vector.
        """
        return -self.S[:-1, :-1] @ self.e[:-1]

    @cached_property
    def e(self) -> np.ndarray:
        """
        Vector with ones of size ``n``.
        """
        return np.ones(self.S.shape[0])

    @cached_property
    def T(self) -> np.ndarray:
        """
        Transition matrix.
        """
        return expm(self.S)

    @cached_property
    def t(self) -> np.ndarray:
        """
        Exit probability vector.
        """
        return 1 - self.T @ self.e

    @cached_property
    def U(self) -> np.ndarray:
        """
        Green matrix (negative inverse of sub-intensity matrix).
        """
        return -inv(self.S[:-1, :-1])

    @cached_property
    def T_inv(self) -> np.ndarray:
        """
        Inverse of transition matrix.
        """
        return inv(self.T)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Get full intensity matrix.
        """
        # obtain intensity matrix
        return self._get_rate_matrix()

    def update_demography(self, demography: TimeHomogeneousDemography):
        """
        Update the demography.

        :param demography: Demography.
        :return: State space.
        """
        # update the demography
        self.demography = demography

        # try to delete rate matrix cache
        try:
            # noinspection all
            del self.S
        except AttributeError:
            pass

    @cached_property
    def k(self) -> int:
        """
        Get number of states.

        :return: The number of states.
        """
        return len(self.states)

    @cached_property
    def m(self) -> int:
        """
        Length of the state vector for a single deme.

        :return: The length
        """
        return self.states.shape[2]

    def _matrix_indices_to_rates(self, i: int, j: int) -> float:
        """
        Get the rate from the state indexed by i to the state indexed by j.

        :param i: Index i.
        :param j: Index j.
        :return: The rate from the state indexed by i to the state indexed by j.
        """
        # get the states
        s1 = self.states[i]
        s2 = self.states[j]

        # get the difference between the states
        diff = s1 - s2

        # get the number of different demes
        n_diff = (diff != 0).any(axis=1).sum()

        # eligible for coalescent event
        if n_diff == 1:
            deme_index = np.where(diff != 0)[0][0]

            rate = self._get_coalescent_rate(n=self.pop_config.n, s1=s1[deme_index], s2=s2[deme_index])

            pop_size = next(self.demography.pop_sizes[self.demography.pop_names[deme_index]])

            return rate / self.model._get_timescale(pop_size)

        # eligible for migration event
        if n_diff == 2:

            # check if one migration event
            if (diff == 1).sum(axis=1).sum() == 1 and (diff == -1).sum(axis=1).sum() == 1:

                # make sure that the other demes are not changing
                if (diff != 0).sum(axis=1).sum() == 2:

                    # make sure that the number of lineages is greater than 1
                    if s1.sum() > 1:
                        # get the indices of the source and destination demes
                        i_source = np.where((diff == 1).sum(axis=1) == 1)[0][0]
                        i_dest = np.where((diff == -1).sum(axis=1) == 1)[0][0]

                        # get the deme names
                        source = self.demography.pop_names[i_source]
                        dest = self.demography.pop_names[i_dest]

                        # get the number of lineages in deme i before migration
                        n_lineages_source = s1[i_source][np.where(diff == 1)[1][0]]

                        # scale migration rate by population size
                        migration_rate = self.demography._migration_rates[(source, dest)]

                        rate = migration_rate * n_lineages_source

                        return rate

        return 0

    def _get_rate_matrix(self) -> np.ndarray:
        """
        Get the rate matrix.

        :return: The rate matrix.
        """
        matrix_indices_to_rates = np.vectorize(self._matrix_indices_to_rates, otypes=[float])

        # create empty matrix
        S = np.zeros((self.k, self.k))

        # fill matrix with non-zero rates
        S[self._non_zero_states] = matrix_indices_to_rates(*self._non_zero_states)

        # fill diagonal with negative sum of row
        S[np.diag_indices_from(S)] = -np.sum(S, axis=1)

        return S

    @abstractmethod
    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
        """
        pass

    @staticmethod
    def _find_vectors(n: int, k: int) -> List[List[int]]:
        """
        Find all vectors of length ``k`` with non-negative integers that sum to ``n``.

        :param n: The sum.
        :param k: The length of the vectors.
        :return: All vectors of length ``k`` with non-negative integers that sum to ``n``.
        """
        if k == 0:
            return [[]]

        if k == 1:
            return [[n]]

        vectors = []
        for i in range(n + 1):
            for vector in StateSpace._find_vectors(n - i, k - 1):
                vectors.append(vector + [i])

        return vectors


class DefaultStateSpace(StateSpace):
    """
    Default rate matrix where there is one state per number of lineages for each deme and locus.
    """

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
        """
        return self.model.get_rate(s1=s1[0], s2=s2[0])

    @cached_property
    @abstractmethod
    def states(self) -> np.ndarray:
        """
        Get the states. Each state describes the number of lineages per deme.
        """
        # the number of lineages
        lineages = np.arange(1, self.pop_config.n + 1)[::-1]

        # iterate over possible number of lineages and find all possible deme configurations
        states = []
        for i in lineages:
            states += self._find_vectors(n=i, k=self.demography.n_pops)

        states = np.array(states)

        # add extra dimension to make it compatible with the other state spaces
        return states.reshape(states.shape + (1,))


class InfiniteAllelesStateSpace(StateSpace):
    r"""
    Infinite alleles rate matrix where there is one state per sample configuration:
    :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`,

    per deme and per locus. This state space can distinguish between different tree topologies
    and is thus used when computing statistics based on the SFS.
    """

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
        """
        return self.model.get_rate_infinite_alleles(n=n, s1=s1, s2=s2)

    @cached_property
    def states(self) -> np.ndarray:
        """
        Get the states. Each state describes the allele configuration per deme.

        :return: The states.
        """
        # the possible allele configurations
        allele_configs = np.array(self._find_sample_configs(m=self.pop_config.n, n=self.pop_config.n))

        # iterate over possible allele configurations and find all possible deme configurations
        states = []
        for config in allele_configs:

            # iterate over possible number of lineages with multiplicity k and
            # find all possible deme configurations
            vectors = []
            for i in config:
                vectors += [self._find_vectors(n=i, k=self.demography.n_pops)]

            # find all possible combinations of deme configurations for each multiplicity
            states += list(product(*vectors))

        # transpose the array to have the deme configurations as columns
        states = np.transpose(np.array(states), (0, 2, 1))

        return states

    @classmethod
    def _find_sample_configs(cls, m: int, n: int) -> List[List[int]]:
        """
        Function to find all vectors of length m such that sum_{i=0}^{m} i*x_{m-i} equals n.

        :param m: Length of the vectors.
        :param n: Target sum.
        :returns: list of vectors satisfying the condition
        """
        # base case, when the length of vector is 0
        # if n is also 0, return an empty vector, otherwise no solutions
        if m == 0:
            return [[]] if n == 0 else []

        vectors = []
        # iterate over possible values for the first component
        for x in range(n // m + 1):  # Adjusted for 1-based index
            # recursively find vectors with one less component and a smaller target sum
            for vector in cls._find_sample_configs(m - 1, n - x * m):  # Adjusted for 1-based index
                # prepend the current component to the recursively found vectors
                vectors.append(vector + [x])  # Reversed vectors

        return vectors
