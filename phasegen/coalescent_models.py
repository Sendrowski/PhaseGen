import itertools
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List, Set, Tuple, Dict, cast

import numpy as np
from scipy.special import comb, beta


class CoalescentModel(ABC):
    """
    Abstract class for coalescent models.
    """

    @abstractmethod
    def get_rate(self, s1: int, s2: int) -> float:
        """
        Get rate for a merger collapsing k1 lineages into k2 lineages.

        :param s1: Number of lineages in the first state.
        :param s2: Number of lineages in the second state.
        :return: The rate.
        """
        pass

    @abstractmethod
    def get_rate_infinite_alleles(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        r"""
        Get rate between two infinite alleles states.
        :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`.

        :param n: Number of lineages.
        :param s1: Sample configuration 1.
        :param s2: Sample configuration 2.
        :return: The rate.
        """

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
    Standard (Kingman) coalescent model.
    """

    def get_rate(self, s1: int, s2: int) -> float:
        """
        Get rate for a merger collapsing k1 lineages into k2 lineages.

        :param s1: Number of lineages in the first state.
        :param s2: Number of lineages in the second state.
        :return: The rate.
        """
        # not possible
        if s2 > s1:
            return 0

        return self._get_rate(b=s1, k=s1 + 1 - s2)

    @staticmethod
    def _get_rate(b: int, k: int):
        """
        Get positive rate for a merger of k out of b lineages.
        Negative rates will be inferred later

        :param b: Number of lineages.
        :param k: Number of lineages that merge.
        :return: The rate.
        """
        # two lineages can merge with a rate depending on b
        if k == 2:
            return b * (b - 1) / 2

        # no other mergers can happen
        return 0

    def get_rate_infinite_alleles(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get exponential rate for a merger of k out of b lineages.

        :param n: Number of lineages.
        :param s1: Sample configuration 1.
        :param s2: Sample configuration 2.
        :return: The rate.
        """
        diff = s2 - s1

        if np.sum(diff == 1) == 1:

            # if two lineages of different classes coalesce
            if np.sum(diff == -1) == 2 and np.sum(diff == 0) == n - 3:

                # check that (a_1,...,a_n) -> (a_1,...,a_i - 1,...,a_j - 1,...,a_{i+j} + 1,...,a_n)
                if diff[(np.where(diff == -1)[0] + 1).sum() - 1] == 1:
                    rate = s1[np.where(diff == -1)[0]].prod()
                    return rate

            # if two lineages of the same class coalesce
            if np.sum(diff == -2) == 1 and np.sum(diff == 0) == n - 2:

                # check that (a_1,...,a_n) -> (a_1,...,a_i - 2,...,a_2i + 1,...,a_n)
                if diff[2 * (np.where(diff == -2)[0][0] + 1) - 1] == 1:
                    rate = math.comb(s1[np.where(diff == -2)[0][0]], 2)
                    return rate

        return 0

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.
        Note that this currently only works for a single population.

        :param n: The number of lineages
        :return: The probabilities of all possible sample configurations.
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
                for s2 in self._find_substates(n, s1):
                    s2_tuple = tuple(s2)
                    states[i - 1].add(s2_tuple)

                    # determine the probability of transitioning from s1 to s2
                    probs[s2_tuple] += probs[s1_tuple] * self._get_probs(n, s1, s2)

        return probs

    @staticmethod
    def _find_substates(n: int, state: np.ndarray) -> List[np.ndarray]:
        """
        Function to find all substates of a given state that are one coalescence event away.

        :param n: The number of lineages
        :param state: The given state
        :returns: List of substates
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

    def _get_sample_config_probs_explicit_state_space(self, n: int, states: np.ndarray) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.
        This function constructs the state space explicitly and iterates over all possible states which is
        computationally expensive.

        :param n: Number of lineages.
        :param states: Matrix of all possible states.
        :return: The probabilities of all possible sample configurations.
        """
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

                probs[tuple(s2)] += probs[tuple(s1)] * self._get_probs(n, s1, s2)

        return probs

    @staticmethod
    def _get_probs(n: int, s1: np.ndarray, s2: np.ndarray) -> float:
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
        """
        Initialize the beta coalescent model.

        :param alpha: The alpha parameter of the beta coalescent model.
        """
        self.alpha = alpha

    def get_rate(self, s1: int, s2: int) -> float:
        """
        Get rate for a merger collapsing k1 lineages into k2 lineages.

        :param s1: Number of lineages in the first state.
        :param s2: Number of lineages in the second state.
        :return: The rate.
        """
        # not possible
        if s2 > s1:
            return 0

        return self._get_rate(b=s1, k=s1 + 1 - s2)

    def _get_rate(self, b: int, k: int):
        """
        Get positive rate for a merger of k out of b lineages.
        Negative rates will be filled in later.

        :param b: The number of lineages.
        :param k: The number of lineages that merge.
        :return: The rate.
        """
        if k <= 1 or k > b:
            return 0

        return comb(b, k, exact=True) * beta(k - self.alpha, b - k + self.alpha) / beta(self.alpha, 2 - self.alpha)

    def get_rate_infinite_alleles(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the rate of a merger of k out of b lineages under the infinite alleles model.

        :param n: Number of lineages.
        :param s1: First state.
        :param s2: Second state.
        :return: The rate.
        """
        pass

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :param n: The total number of lineages
        :return: The probabilities of all possible sample configurations.
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

        :return: The density function of the coalescent model.
        """
        pass

    def get_rate(self, i: int, j: int):
        """
        Get exponential rate for a merger of j out of i lineages.

        :param i: The number of lineages.
        :param j: The number of lineages that merge.
        :return: The rate.
        """
        """
        x = sp.symbols('x')
        integrand = x ** (i - 2) * (1 - x) ** (j - i)

        integral = sp.Integral(integrand * self.get_density()(x), (x, 0, 1))
        return float(integral.doit())
        """

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :param n: The number of lineages
        :return: The probabilities of all possible sample configurations.
        """
        raise NotImplementedError()
