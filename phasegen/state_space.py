from abc import ABC, abstractmethod
from functools import cached_property
from typing import cast, List

import numpy as np
from scipy.linalg import expm, inv

from .coalescent_models import CoalescentModel
from .demography import TimeHomogeneousDemography


class StateSpace(ABC):

    def __init__(
            self,
            n: int,
            model: CoalescentModel,
            demography: TimeHomogeneousDemography
    ):
        """
        Create a rate matrix.

        :param n: Number of lineages.
        :param model: Coalescent model.
        :param demography: Time homogeneous demography (we can also construct a state space
            for a fixed demography).
        """
        self.n = n
        self.model = model
        self.demography = demography

    @abstractmethod
    def _get_rate_matrix(self) -> np.ndarray:
        """
        Get the sub-intensity matrix for a given number of lineages.
        """
        pass

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
    @abstractmethod
    def S(self) -> np.ndarray:
        """
        Get full intensity matrix.
        """
        pass

    @cached_property
    @abstractmethod
    def k(self) -> int:
        """
        Get number of states.
        """
        pass

    def from_demography(self, demography: TimeHomogeneousDemography) -> 'StateSpace':
        """
        Get a new state space given a demography. Like this we can optimize the state space
        creation for a specific demography.
        TODO optimize state space creation

        :param demography: Demography.
        :return: State space.
        """
        return self.__class__(
            n=self.n,
            model=self.model,
            demography=demography
        )


class DefaultStateSpace(StateSpace):
    """
    Default rate matrix where there is one state per number of lineages for each deme and locus.
    """

    def _get_rate_matrix(self) -> np.ndarray:
        """
        Get the sub-intensity matrix for a given number of lineages.
        Each state corresponds to the number of lineages minus one.

        :return: The rate matrix.
        """

        def matrix_indices_to_rates(i: int, j: int) -> float:
            """
            Get the rate from state i to state j.
            TODO implement for multiple demes

            :param i: State i.
            :param j: State j.
            :return: The rate from state i to state j.
            """
            pop_size = self.demography.pop_size[self.demography.pop_names[0]]

            return self.model.get_rate(b=self.n - i, k=j + 1 - i) / pop_size

        # obtain the rate matrix
        S = np.fromfunction(np.vectorize(matrix_indices_to_rates), (self.n - 1, self.n - 1), dtype=float)

        return cast(np.ndarray, S)

    @cached_property
    @abstractmethod
    def states(self) -> np.ndarray:
        """
        Get the states.
        """
        return np.arange(2, self.n)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Get full intensity matrix.
        """
        # obtain sub-intensity matrix
        S_sub = self._get_rate_matrix()

        # exit rate vector
        # cannot use cached property because it depends on S
        s = -S_sub @ np.ones(S_sub.shape[0])

        # return full intensity matrix
        return np.block([
            [S_sub, s[:, None]],
            [np.zeros(S_sub.shape[0] + 1)]
        ])

    @cached_property
    @abstractmethod
    def k(self) -> int:
        """
        Get number of states.
        """
        return self.n


class InfiniteAllelesStateSpace(StateSpace):
    r"""
    Infinite alleles rate matrix where there is one state per sample configuration:
    :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`,

    per deme and per locus.
    """

    @cached_property
    def states(self) -> np.ndarray:
        """
        Get the states for a given number of lineages.

        :return: The states.
        """
        return np.array(self._find_vectors(m=self.n, n=self.n))

    def _find_vectors(self, m: int, n: int) -> List[List[int]]:
        """
        Function to find all vectors x of length m such that the sum_{i=0}^{m} i*x_{m-i} equals n.

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
            for vector in self._find_vectors(m - 1, n - x * m):  # Adjusted for 1-based index
                # prepend the current component to the recursively found vectors
                vectors.append(vector + [x])  # Reversed vectors

        return vectors

    def _get_rate_matrix(self) -> (np.ndarray, np.ndarray):
        r"""
        Get the intensity matrix for a given number of lineages.
        Each state corresponds to a sample configuration,
        :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`.

        :return: The rate matrix.
        """
        n_states = self.states.shape[0]

        def matrix_indices_to_rates(i: int, j: int) -> float:
            """
            Get the rate from state i to state j.
            TODO implement for multiple demes

            :param i: State i.
            :param j: State j.
            :return: The rate from state i to state j.
            """
            pop_size = self.demography.pop_size[self.demography.pop_names[0]]
            rate = self.model.get_rate_infinite_alleles(n=self.n, s1=self.states[i], s2=self.states[j])

            return rate / pop_size

        S = cast(np.ndarray, np.fromfunction(np.vectorize(matrix_indices_to_rates), (n_states, n_states), dtype=int))

        # fill diagonal with negative sum of row
        S[np.diag_indices_from(S)] = -np.sum(S, axis=1)

        return S

    @cached_property
    def S(self) -> np.ndarray:
        """
        Get full intensity matrix.
        """
        # obtain intensity matrix
        return self._get_rate_matrix()

    @cached_property
    @abstractmethod
    def k(self) -> int:
        """
        Get number of states.

        :return: The number of states.
        """
        return self.partitions(self.n)

    @staticmethod
    def partitions(n: int) -> int:
        """
        Get the number of partitions of n.

        :param n: The number to partition.
        :return: The number of partitions of n.
        """
        parts = [1] + [0] * n

        for t in range(1, n + 1):
            for i, x in enumerate(range(t, n + 1)):
                parts[x] += parts[i]

        return parts[n]
