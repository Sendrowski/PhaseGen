import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List, Set, Tuple, Dict, cast

import numpy as np
from scipy.special import comb, beta


class CoalescentModel(ABC):
    """
    Abstract class for coalescent models.
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

    def get_rate_infinite_alleles(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        r"""
        Get (positive) rate between two infinite alleles states.
        :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`.

        :param n: Number of lineages.
        :param s1: Sample configuration 1.
        :param s2: Sample configuration 2.
        :return: The rate.
        """
        diff = s2 - s1

        # make sure only one class has one more lineage
        if np.sum(diff == 1) == 1 and n == s1.shape[0]:

            # get the index for the class that lost lineages
            where_less = np.where(diff < 0)[0]

            # only continue if there is a class that lost lineages
            if len(where_less) > 0:

                # get the number of lineages that were lost
                diff_less = -diff[where_less]

                # determine the index of the class that gained lineages
                i_more = np.dot(where_less + 1, diff_less) - 1

                # make sure that the class that gained lineages only gained one lineage
                if diff[i_more] == 1:
                    # number of lineages before the merger
                    b = s1[where_less]

                    # determine number of lineages that coalesce
                    k = b - s2[where_less]

                    # get rate
                    rate = self._get_rate_infinite_alleles(n=s1.sum(), b=b, k=k)
                    return rate

        return 0

    @abstractmethod
    def get_generation_time(self, N: float) -> float:
        """
        Get the generation time.

        :param N: The effective population size.
        :return: The generation time.
        """
        pass

    @abstractmethod
    def _get_rate(self, b: int, k: int) -> float:
        """
        Get positive rate for a merger of k out of b lineages.
        Negative rates will be inferred later

        :param b: Number of lineages.
        :param k: Number of lineages that merge.
        :return: The rate.
        """
        pass

    @abstractmethod
    def _get_rate_infinite_alleles(self, n: int, b: np.ndarray[int], k: np.ndarray[int]) -> float:
        """
        Get positive rate for a merger of k_i out of b_i lineages for all i.
        Negative rates will be inferred later

        :param n: Number of lineages.
        :param b: Number of lineages before merge for classes that experience a merger.
        :param k: Number of lineages that merge for classes that experience a merger.
        :return: The rate.
        """
        pass

    @abstractmethod
    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        TODO deprecate this?

        :param n: Number of lineages.
        :return:
        """
        pass


class StandardCoalescent(CoalescentModel):
    """
    Standard (Kingman) coalescent model.
    """

    def get_generation_time(self, N: float) -> float:
        """
        Get the generation time.

        :param N: The effective population size.
        :return: The generation time.
        """
        return N

    def _get_rate(self, b: int, k: int) -> float:
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

    def _get_rate_infinite_alleles(self, n: int, b: np.ndarray[int], k: np.ndarray[int]) -> float:
        """
        Get positive rate for a merger of k_i out of b_i lineages for all i.
        Negative rates will be inferred later

        :param n: Number of lineages.
        :param b: Number of lineages before merge for classes that experience a merger.
        :param k: Number of lineages that merge for classes that experience a merger.
        :return: The rate.
        """
        # if we have a single class
        if b.shape[0] == 1:
            return self._get_rate(b=b[0], k=k[0])

        # if we have a merger from two classes
        if b.shape[0] == 2:
            if k[0] == 1 and k[1] == 1:
                # same as b[0] choose k[0] times b[1] choose k[1]
                return b[0] * b[1]

        # no other mergers are possible
        return 0

    def get_rate_infinite_alleles_dep(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get (positive) rate between two infinite alleles states.

        :param n: Number of lineages.
        :param s1: Sample configuration 1.
        :param s2: Sample configuration 2.
        :return: The rate.
        """
        diff = s2 - s1

        # make sure only one class has one more lineage
        if np.sum(diff == 1) == 1 and n == s1.shape[0]:

            # if two lineages of different classes coalesce
            if np.sum(diff == -1) == 2 and np.sum(diff == 0) == n - 3:

                # check that (a_1,...,a_n) -> (a_1,...,a_i - 1,...,a_j - 1,...,a_{i+j} + 1,...,a_n)
                if diff[(np.where(diff == -1)[0] + 1).sum() - 1] == 1:
                    b = s1[np.where(diff == -1)[0]]
                    rate = b.prod()
                    return rate

            # if two lineages of the same class coalesce
            if np.sum(diff == -2) == 1 and np.sum(diff == 0) == n - 2:

                # check that (a_1,...,a_n) -> (a_1,...,a_i - 2,...,a_2i + 1,...,a_n)
                if diff[2 * (np.where(diff == -2)[0][0] + 1) - 1] == 1:
                    b = s1[np.where(diff == -2)[0][0]]
                    rate = self._get_rate(b=b, k=2)
                    return rate

        return 0

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.
        Note that this currently only works for a single population.

        TODO deprecate this?

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

                return comb(j, 2) / comb(i, 2)

            # if two lineages of different classes coalesce
            if np.sum(diff == 1) == 2 and np.sum(diff == 0) == n - 3:
                # get the number of lineages that were present in s1
                j1, j2 = s1[diff == 1]

                return comb(j1, 1) * comb(j2, 1) / comb(i, 2)

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

    def _get_base_rate(self, b: int, k: int) -> float:
        """
        Get base rate for a merger of k out of b lineages (without number of ways).

        :param b: The number of lineages before the merger.
        :param k: The number of lineages that merge.
        :return: The rate.
        """
        return beta(k - self.alpha, b - k + self.alpha) / beta(self.alpha, 2 - self.alpha)

    def get_generation_time(self, N: float) -> float:
        """
        Get the generation time.

        :param N: The effective population size.
        :return: The generation time.
        """
        m = 1 + 1 / (2 ** (self.alpha - 1) * (self.alpha - 1))

        return m ** self.alpha * N ** (self.alpha - 1) / self.alpha / beta(2 - self.alpha, self.alpha)

    def _get_rate(self, b: int, k: int) -> float:
        """
        Get positive rate for a merger of k out of b lineages.
        Negative rates will be filled in later.

        :param b: The number of lineages before the merger.
        :param k: The number of lineages that merge.
        :return: The rate.
        """
        if k < 1 or k > b:
            return 0

        return comb(b, k, exact=True) * self._get_base_rate(b, k)

    def _get_rate_infinite_alleles(self, n: int, b: np.ndarray[int], k: np.ndarray[int]) -> float:
        """
        Get positive rate for a merger of k_i out of b_i lineages for all i.
        Negative rates will be inferred later

        :param n: Number of lineages.
        :param b: Number of lineages before merge for classes that experience a merger.
        :param k: Number of lineages that merge for classes that experience a merger.
        :return: The rate.
        """
        combinations = np.prod([comb(N=b_i, k=k_i, exact=True) for b_i, k_i in zip(b, k)])

        return combinations * self._get_base_rate(b=n, k=k.sum())

    def get_sample_config_probs(self, n: int) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        TODO deprecate this?

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
