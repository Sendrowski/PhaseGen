import itertools
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List, Set, Tuple, Dict, cast

import numpy as np
import sympy as sp
from scipy.special import comb, beta


class CoalescentModel(ABC):
    """
    Abstract class for coalescent models.
    """

    def get_rate_matrix(self, n: int) -> np.ndarray:
        """
        Get the sub-intensity matrix for a given number of lineages.
        Each state corresponds to the number of lineages minus one.

        :param n: Number of lineages.
        :return: The rate matrix.
        """

        def matrix_indices_to_rates(i: int, j: int) -> float:
            """
            Convert matrix indices to k out of b lineages.

            :param i:
            :param j:
            :return:
            """
            return self.get_rate(b=int(n - i), k=int(j + 1 - i))

        # Dividing by Ne here produces unstable results for small population
        # sizes (Ne < 1). We thus add it later to the moments.
        return cast(np.ndarray, np.fromfunction(np.vectorize(matrix_indices_to_rates), (n - 1, n - 1)))

    def get_rate_matrix_infinite_alleles(self, n: int) -> np.ndarray:
        r"""
        Get the sub-intensity matrix for a given number of lineages.
        Each state corresponds to a sample configuration,
        :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`.

        :param n: Number of lineages.
        :return: The rate matrix.
        """


    @abstractmethod
    def get_rate(self, b: int, k: int):
        """
        Get exponential rate for a merger of k out of b lineages.

        :param b:
        :param k:
        :return:
        """
        pass

    @abstractmethod
    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :param n: Number of lineages.
        :return:
        """
        pass


class StandardCoalescent(CoalescentModel):
    """
    Standard coalescent model.
    """

    def get_rate(self, b: int, k: int):
        """
        Get exponential rate for a merger of k out of b lineages.

        :param b:
        :param k:
        :return:
        """
        # two lineages can merge with a rate depending on b
        if k == 2:
            return b * (b - 1) / 2

        # the opposite of above
        if k == 1:
            return -self.get_rate(b=b, k=2)

        # no other mergers can happen
        return 0

    def find_vectors(self, m: int, n: int) -> List[List[int]]:
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
            for vector in self.find_vectors(m - 1, n - x * m):  # Adjusted for 1-based index
                # prepend the current component to the recursively found vectors
                vectors.append(vector + [x])  # Reversed vectors

        return vectors

    def find_substates(self, n: int, state: np.ndarray) -> List[np.ndarray]:
        """
        Function to find all substates of a given state that are one coalescence event away.

        :param n: The number of lineages
        :param state: The given state
        :returns: list of substates
        """
        substates = []

        for i in range(n):
            for j in range(n):
                if (i < j and state[i] > 0 and state[j] > 0) or (i == j and state[i] > 1):
                    new_state = state.copy()
                    new_state[i] -= 1
                    new_state[j] -= 1
                    new_state[i + j + 1] += 1

                    substates.append(new_state)

        return substates

    def get_sample_config_probs_explicit_state_space(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.
        This function constructs the state space explicitly and iterates over all possible states which is
        computationally expensive.

        :param n: The number of lineages
        :return:
        """
        # get all possible states
        states = np.array(self.find_vectors(n, n))

        # the number of lineages in each state
        n_lin_states = states.sum(axis=1)

        # the indices of the states with the same number of lineages
        n_lineages = [np.where(n_lin_states == i)[0] for i in np.arange(n + 1)]

        # initialize the probabilities
        probs = cast(Dict[Tuple, float], defaultdict(int))
        probs[tuple(states[0])] = 1

        # iterate over the number of lineages
        for i in np.arange(2, n)[::-1]:

            # iterate over pairs and determine the probability of transitioning from s1 to s2
            for s1, s2 in itertools.product(states[n_lineages[i + 1]], states[n_lineages[i]]):
                # s = self.find_substates(s1)

                probs[tuple(s2)] += probs[tuple(s1)] * self.get_probs(n, s1, s2)

        return probs

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :param n: The number of lineages
        :return:
        """
        # initialize the probabilities
        probs = cast(Dict[Tuple, float], defaultdict(int))

        # states indexed by the number of lineages
        states: List[Set[Tuple[int, ...]]] = [set() for _ in range(n)]
        states[n - 1] = {tuple([n] + [0] * (n - 1))}

        # initialize the probabilities
        probs[tuple(states[n - 1])[0]] = 1

        # iterate over the number of lineages
        for i in np.arange(2, n)[::-1]:

            # iterate over states with i + 1 lineages
            for s1_tuple in states[i]:
                s1 = np.array(s1_tuple)

                # iterate over substates of s1
                for s2 in self.find_substates(n, s1):
                    s2_tuple = tuple(s2)
                    states[i - 1].add(s2_tuple)

                    # determine the probability of transitioning from s1 to s2
                    probs[s2_tuple] += probs[s1_tuple] * self.get_probs(n, s1, s2)

        return probs

    def get_probs(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the probabilities transitioning from s1 to s2 assuming that s1 has one more lineage than s2.

        :param n: The number of lineages
        :param s1: The starting state
        :param s2: The ending state
        :return: The probability of transitioning from s1 to s2
        """
        diff = s1 - s2
        i = s1.sum()

        if np.sum(diff == -1) == 1:

            # if two lineages of the same class coalesce
            if np.sum(diff == 2) == 1 and np.sum(diff == 0) == n - 2:
                # get the number of lineages that were present in s1
                j = s1[diff == 2][0]

                return math.comb(j, 2) / math.comb(i, 2)

            # if two lineages of different classes coalesce
            if np.sum(diff == 1) == 2 and np.sum(diff == 0) == n - 3:
                # get the number of lineages that were present in s1
                j1, j2 = s1[diff == 1]

                return math.comb(j1, 1) * math.comb(j2, 1) / math.comb(i, 2)

        return 0


class BetaCoalescent(CoalescentModel):
    """
    Beta coalescent model.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def get_rate(self, b: int, k: int):
        """
        Get exponential rate for a merger of k out of b lineages.

        :param b:
        :param k:
        :return:
        """
        if k < 1 or k > b:
            return 0

        if k == 1:
            return -np.sum([self.get_rate(b, i) for i in range(2, b + 1)])

        return comb(b, k, exact=True) * beta(k - self.alpha, b - k + self.alpha) / beta(self.alpha, 2 - self.alpha)

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :param n: The number of lineages
        :return:
        """
        raise NotImplementedError()


class LambdaCoalescent(CoalescentModel):
    """
    Lambda coalescent model.
    TODO implement this
    """

    @abstractmethod
    def get_density(self) -> Callable:
        """
        Get the density function of the coalescent model.

        :return:
        """
        pass

    def get_rate(self, i: int, j: int):
        """
        Get exponential rate for a merger of j out of i lineages.

        :param i:
        :param j:
        :return:
        """
        x = sp.symbols('x')
        integrand = x ** (i - 2) * (1 - x) ** (j - i)

        integral = sp.Integral(integrand * self.get_density()(x), (x, 0, 1))
        return float(integral.doit())

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :param n: The number of lineages
        :return:
        """
        raise NotImplementedError()
