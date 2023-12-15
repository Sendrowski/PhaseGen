from abc import ABC, abstractmethod

import numpy as np
from scipy.special import comb, beta
from scipy.stats import binom


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
    def _get_timescale(self, N: float) -> float:
        """
        Get the timescale.

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


class StandardCoalescent(CoalescentModel):
    """
    Standard (Kingman) coalescent model. Refer to
    `Msprime docs <https://tskit.dev/msprime/docs/stable/api.html?
    highlight=standard+coalescent#msprime.StandardCoalescent>`__
    for more information.
    """

    def _get_timescale(self, N: float) -> float:
        """
        Get the timescale.

        :param N: The effective population size.
        :return: The generation time.
        """
        return N

    def _get_rate(self, b: int, k: int) -> float:
        """
        Get positive rate for a merger of k out of b lineages.

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


class BetaCoalescent(CoalescentModel):
    """
    Beta coalescent model. Refer to
    `Msprime docs <https://tskit.dev/msprime/docs/stable/api.html?highlight=beta+coalescent#msprime.BetaCoalescent>`__
    for more information.

    TODO disagrees with Msprime for large values of alpha
    """

    def __init__(self, alpha: float, scale_time: bool = True):
        """
        Initialize the beta coalescent model.

        :param alpha: The alpha parameter of the beta coalescent model.
        :param scale_time: Whether to scale coalescence time as described in
            `Msprime docs <https://tskit.dev/msprime/docs/stable/api.html?
            highlight=beta+coalescent#msprime.BetaCoalescent>`__. If ``False``, the timescale is set to N.
        """
        if alpha < 1 or alpha > 2:
            raise ValueError("Alpha must be between 1 and 2.")

        #: Whether to scale coalescence time
        self.scale_time: bool = scale_time

        #: The alpha parameter of the beta coalescent model.
        self.alpha: float = alpha

    def _get_base_rate(self, b: int, k: int) -> float:
        """
        Get base rate for a merger of k out of b lineages (without number of ways).

        :param b: The number of lineages before the merger.
        :param k: The number of lineages that merge.
        :return: The rate.
        """
        return beta(k - self.alpha, b - k + self.alpha) / beta(self.alpha, 2 - self.alpha)

    def _get_timescale(self, N: float) -> float:
        """
        Get the timescale.

        :param N: The effective population size.
        :return: The generation time.
        """
        if not self.scale_time:
            return N

        m = 1 + 1 / 2 ** (self.alpha - 1) / (self.alpha - 1)

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

        :param n: Number of lineages.
        :param b: Number of lineages before merge for classes that experience a merger.
        :param k: Number of lineages that merge for classes that experience a merger.
        :return: The rate.
        """
        combinations = np.prod([comb(N=b_i, k=k_i, exact=True) for b_i, k_i in zip(b, k)])

        return combinations * self._get_base_rate(b=n, k=k.sum())


class DiracCoalescent(CoalescentModel):
    """
    Dirac coalescent model. Refer to
    `Msprime docs <https://tskit.dev/msprime/docs/stable/api.html?highlight=dirac+coalescent#msprime.DiracCoalescent>`__
    for more information.

    TODO shorter SFS bins than msprime
    """

    def __init__(self, psi: float, c: float, scale_time: bool = True):
        """
        Initialize the Dirac coalescent model.

        :param psi: The fraction of the population replaced by offspring in one large reproduction event
        :param c: The rate of potential multiple merger events.
        :param scale_time: Whether to scale coalescence time as described in
            `Msprime docs <https://tskit.dev/msprime/docs/stable/api.html?
            highlight=dirac+coalescent#msprime.DiracCoalescent>`__. If ``False``, the timescale is set to N.
        """
        super().__init__()

        if not 0 < psi < 1:
            raise ValueError("Psi must be between 0 and 1.")

        #: The fraction of the population replaced by offspring in one large reproduction event
        self.psi = psi

        #: The rate of potential multiple merger events.
        self.c = c

        #: Whether to scale coalescence time
        self.scale_time: bool = scale_time

        #: The standard coalescent model
        self._standard = StandardCoalescent()

    def _get_timescale(self, N: float) -> float:
        """
        Get the timescale.

        :param N: The effective population size.
        :return: The generation time.
        """
        if not self.scale_time:
            return N

        return N ** 2

    def _get_rate(self, b: int, k: int) -> float:
        """
        Get positive rate for a merger of k out of b lineages.
        Negative rates will be filled in later.

        :param b: The number of lineages before the merger.
        :param k: The number of lineages that merge.
        :return: The rate.
        """
        # rate of binary merger
        rate_binary = self._standard._get_rate(b=b, k=k)

        # probability of multiple merger of k out of b lineages
        p_psi = binom.pmf(k=k, n=b, p=self.psi)

        # rate of multiple merger
        rate_multi = p_psi * self.c

        return rate_binary + rate_multi

    def _get_rate_infinite_alleles(self, n: int, b: np.ndarray[int], k: np.ndarray[int]) -> float:
        """
        Get positive rate for a merger of k_i out of b_i lineages for all i.

        :param n: Number of lineages.
        :param b: Number of lineages before merge for classes that experience a merger.
        :param k: Number of lineages that merge for classes that experience a merger.
        :return: The rate.
        """
        # rate of binary merger
        rate_binary = self._standard._get_rate_infinite_alleles(n=n, b=b, k=k)

        # probability of multiple merger of k out of n lineages
        p_psi = binom.pmf(k=k.sum(), n=n, p=self.psi)

        # rate of multiple merger
        rate_multi = p_psi * self.c

        return rate_binary + rate_multi
