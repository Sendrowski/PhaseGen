import functools
import itertools
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property, lru_cache
from math import factorial
from typing import Generator, List, Callable, cast, Tuple, Dict, Set

import msprime as ms
import numpy as np
import tskit
from matplotlib import pyplot as plt
from multiprocess import Pool
from numpy.linalg import matrix_power
from scipy.linalg import inv, expm, fractional_matrix_power
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from .coalescent_models import StandardCoalescent, CoalescentModel
from .demography import Demography, PiecewiseConstantDemography
from .visualization import Visualization


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        batch_size: int = 1,
        desc: str = None,
) -> np.ndarray:
    """
    Convenience function that parallelizes the given function
    if specified or executes them sequentially otherwise.

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize
    :param pbar: Whether to show a progress bar
    :param batch_size: Number of units to show in the pbar per function
    :param desc: Description for tqdm progress bar
    :return:
    """

    if parallelize and len(data) > 1:
        # parallelize
        iterator = Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    if pbar:
        iterator = tqdm(iterator, total=len(data), unit_scale=batch_size, desc=desc)

    return np.array(list(iterator), dtype=object)


def calculate_sfs(tree: tskit.trees.Tree) -> np.ndarray:
    """
    Calculate the SFS of given tree by looking at mutational opportunities.

    :param tree:
    :return:
    """
    sfs = np.zeros(tree.sample_size + 1)
    for u in tree.nodes():
        if u != tree.root:
            t = tree.get_branch_length(u)
            n = tree.get_num_leaves(u)

            sfs[n] += t

    return sfs


def van_loan(B: np.ndarray, S: np.ndarray, tau: float, k: int = 1) -> np.ndarray:
    """
    Use Van Loan's method to evaluate the integral ∫u S(u)B(u)S(u)du for k = 1,
    and accordingly for higher moments.

    :param B: Matrix B
    :param S: Matrix S
    :param tau: Time to integrate over
    :param k: The kth moment to evaluate
    :return: Evaluated integral
    """
    n = B.shape[0]

    # matrix of zeros
    O = np.zeros_like(B)

    # create compound matrix
    A = np.block([[S if i == j else B if i == j - 1 else O for j in range(k + 1)] for i in range(k + 1)])

    # compute matrix exponential of A to determine integral
    V = expm(tau * A)

    # return upper right block
    return V[:n, -n:]


class ProbabilityDistribution(ABC):
    @abstractmethod
    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.

        :param alpha:
        :return:
        """
        pass

    @abstractmethod
    @cached_property
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.

        :param alpha:
        :return:
        """
        pass

    def touch(self):
        """
        Touch all cached properties.
        """
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, functools.cached_property):
                getattr(self, attr)

    @abstractmethod
    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t:
        :return:
        """
        pass

    @abstractmethod
    def pdf(self, u) -> float | np.ndarray:
        """
        Density function.

        :param u:
        :return:
        """
        pass

    def plot_cdf(
            self,
            x: np.ndarray = np.linspace(0, 20, 100),
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None
    ) -> plt.axis:
        """
        Plot cumulative distribution function.

        :return:
        """
        return Visualization.plot(
            x=x,
            y=self.cdf(x),
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear
        )

    def plot_pdf(
            self,
            x: np.ndarray = np.linspace(0, 20, 100),
            show=True,
            file: str = None,
            clear: bool = True,
            label: str = None
    ) -> plt.axis:
        """
        Plot density function.

        :return:
        """
        return Visualization.plot(
            x=x,
            y=self.pdf(x),
            xlabel='u',
            ylabel='f(u)',
            label=label,
            file=file,
            show=show,
            clear=clear
        )


class PhaseTypeDistribution(ProbabilityDistribution, ABC):
    pass


class ConstantPopSizeDistribution(PhaseTypeDistribution):
    def __init__(self, cd: 'ConstantPopSizeCoalescent', r: np.ndarray | List):
        self.cd = cd
        self.r = np.array(r)

    @cached_property
    def R(self) -> np.ndarray:
        """
        The rewards matrix.

        :return:
        """
        return np.diag(self.r)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Intensity matrix with rewards.

        :return: Intensity matrix
        """
        return self.R @ self.cd.S

    @cached_property
    def T(self) -> np.ndarray:
        """
        Probability transition matrix.

        :return: Transition matrix
        """
        return expm(self.S)

    @cached_property
    def s(self) -> np.ndarray:
        """
        Exit rate vector.

        :return: Exit rate vector
        """
        return -self.S[:-1, :-1] @ self.cd.e[:-1]

    def nth_moment(self, k: int, alpha: np.ndarray = None, Ne: float = None, r: np.ndarray = None) -> float:
        """
        Get the nth moment.

        :param r: Full reward vector
        :param Ne: The effective population size
        :param k: The order of the moment
        :param alpha: Full initial state vector
        :return: The nth moment
        """
        if alpha is None:
            alpha = self.cd.alpha

        if Ne is None:
            Ne = self.cd.Ne

        if r is None:
            r = self.r

        R = self.cd.invert_reward(np.diag(r[:-1]))

        M = matrix_power(self.cd.U @ R, k)

        # self.Ne ** k is the rescaling due to population size
        return Ne ** k * factorial(k) * alpha[:-1] @ M @ self.cd.e[:-1]

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.

        :param alpha:
        :return:
        """
        return self.nth_moment(k=1, alpha=alpha)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.

        :param alpha:
        :return:
        """
        return self.nth_moment(k=2, alpha=alpha) - self.nth_moment(k=1, alpha=alpha) ** 2

    def cdf(self, t) -> float | np.ndarray:
        """
        Vectorized cumulative distribution function.

        :param t:
        :return:
        """

        def cdf(t: float) -> float:
            """
            Cumulative distribution function.

            :param t:
            :return:
            """
            return 1 - self.cd.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], t / self.cd.Ne) @ self.cd.e[:-1]

        return np.vectorize(cdf)(t)

    def pdf(self, u) -> float | np.ndarray:
        """
        Vectorized density function.

        :param u:
        :return:
        """

        def pdf(u: float) -> float:
            """
            Density function.

            :param u:
            :return:
            """
            return self.cd.alpha[:-1] @ fractional_matrix_power(self.T[:-1, :-1], u) @ self.s / self.cd.Ne

        return np.vectorize(pdf)(u)


class VariablePopSizeDistribution(ConstantPopSizeDistribution):
    """
    Variable population size distribution using IPH.
    """

    def __init__(self, cd: 'VariablePopSizeCoalescent', r: np.ndarray | List):
        super().__init__(cd, r)

        self.cd = cd
        self.r = np.array(r)

    def _nth_moment(self, k: int, alpha: Tuple[float] = None, r: Tuple[float] = None) -> (np.ndarray, np.ndarray):
        """
        Get the nth (non-central) moment per epoch.

        :param k: The kth moment
        :param alpha: Full initial value vector
        :param r: Full reward vector
        :return: Moments and absorption probability per epoch
        """
        if alpha is None:
            alpha = self.cd.alpha
        else:
            alpha = np.array(alpha)

        if r is None:
            R = self.R
        else:
            R = np.diag(r)

        # moments conditional on no absorption in previous epoch
        moments = np.zeros(self.cd.n_epochs)

        # moments conditional on no absorption in current epoch
        moments_no_absorp = np.zeros(self.cd.n_epochs + 1)

        # state probabilities conditional on no absorption in previous epoch
        alphas = np.zeros((self.cd.n_epochs + 1, self.cd.n))

        # state probabilities conditional on no absorption in current epoch
        alphas_no_absorp = np.zeros((self.cd.n_epochs + 1, self.cd.n))

        # initial state probabilities
        alphas[0] = alpha
        alphas_no_absorp[0] = np.concatenate([(alpha[:-1] / alpha[:-1].sum()), [0]])

        # iterate through epochs and compute initial values
        for i in range(0, self.cd.n_epochs):

            # Ne of current epoch
            Ne = self.cd.pop_sizes[i]

            # all but last epoch
            # TODO break out of loop if absorbing state is reached
            if i < self.cd.n_epochs - 1:
                # time spent in current epoch scaled by Ne
                tau = (self.cd.times[i + 1] - self.cd.times[i]) / Ne
            else:
                # TODO termine tau dynamically
                tau = 1000

            # calculate transition probabilities for current epoch
            P = fractional_matrix_power(self.cd.T, tau)

            # update alpha for the time spent in the current epoch
            alphas[i + 1] = alphas_no_absorp[i] @ P
            alphas_no_absorp[i + 1] = np.concatenate([(alphas[i + 1][:-1] / alphas[i + 1][:-1].sum()), [0]])

            # obtain sojourn matrix for the current epoch using Van Loan's method
            M = van_loan(S=self.cd.S, B=self.cd.invert_reward(R), tau=tau, k=k)

            # normalize by transition probabilities
            M[P != 0] /= P[P != 0]

            # calculate moments in this epoch given no absorption in the previous epoch
            moments[i] = Ne ** k * factorial(k) * (alphas_no_absorp[i] @ M)[-1]

            # make sojourn matrix symmetric
            M += M.T
            M[np.diag_indices_from(M)] /= 2

            # calculate moments given no absorption in this epoch
            moments_no_absorp[i + 1] = Ne ** k * factorial(k) * alphas_no_absorp[i] @ M @ alphas_no_absorp[i + 1]

        no_absorption = np.cumprod(1 - alphas[:-1, -1])
        absorption_probs = alphas[1:, -1]
        total_absorption_probs = no_absorption * absorption_probs

        return moments, moments_no_absorp[:-1], total_absorption_probs

    def nth_moment(self, k: int, alpha: np.ndarray = None, r: np.ndarray = None, **kwargs) -> float:
        """
        Get the nth (non-central) moment.

        :param k: The kth moment
        :param alpha: Full initial value vector
        :param r: Full reward vector
        :param kwargs: Additional arguments
        :return: Moment
        """
        if alpha is not None:
            alpha = tuple(alpha)

        if r is not None:
            r = tuple(r)

        # get moments per epoch
        moments, moments_no_absorp, absorption_probs = self._nth_moment(k=k, alpha=alpha, r=r)

        if k == 1:
            moments_total = np.cumsum(moments_no_absorp) + moments

            return (moments_total * absorption_probs).sum()

        if k == 2:
            # get means
            means, _, _ = self._nth_moment(k=1, alpha=alpha, r=r)

            m2_total = moments + np.cumsum(moments_no_absorp) + 2 * np.cumsum(np.sqrt(moments_no_absorp)) * means

            return (m2_total * absorption_probs).sum()

    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Time
        :return: Cumulative probability
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not np.all(self.r[:-1] == 1):
            raise NotImplementedError("CDF not implemented for non-default rewards.")

        def cdf(t: float) -> float:
            # get the cumulative coalescent rate up to time t
            cum = self.cd.demography.get_cum_rate(t)

            return 1 - self.cd.alpha[:-1] @ fractional_matrix_power(self.cd.T[:-1, :-1], cum) @ self.cd.e[:-1]

        return np.vectorize(cdf)(t)

    def pdf(self, u: float | np.ndarray) -> float | np.ndarray:
        """
        Density function.

        :param u: Time
        :return: Density
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not np.all(self.r[:-1] == 1):
            raise NotImplementedError("PDF not implemented for non-default reward.")

        def pdf(u: float) -> float:
            # get the cumulative coalescent rate up to time u
            cum = self.cd.demography.get_cum_rate(u)

            # get current coalescent rate
            rate = self.cd.demography.get_rate(u)

            return self.cd.alpha[:-1] @ fractional_matrix_power(self.cd.T[:-1, :-1], cum) @ self.cd.s * rate

        return np.vectorize(pdf)(u)


class SFSDistribution(VariablePopSizeDistribution):
    """
    Variable population size distribution using IPH.
    """

    def find_vectors(self, m: int, n: int) -> List[List[int]]:
        """
        Function to find all vectors x of length m such that the sum_{i=0}^{m} i*x_{m-i} equals n.

        :param m: length of the vectors
        :param n: target sum
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

    def find_substates(self, state: np.ndarray) -> List[np.ndarray]:
        """
        Function to find all substates of a given state that are one coalescence event away.

        :param state: The given state
        :returns: list of substates
        """
        substates = []

        for i in range(self.cd.n):
            for j in range(self.cd.n):
                if (i < j and state[i] > 0 and state[j] > 0) or (i == j and state[i] > 1):
                    new_state = state.copy()
                    new_state[i] -= 1
                    new_state[j] -= 1
                    new_state[i + j + 1] += 1

                    substates.append(new_state)

        return substates

    def get_sample_config_probs_explicit_state_space(self) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.
        This function constructs the state space explicitly and iterates over all possible states which is
        computationally expensive.

        :return:
        """
        # get all possible states
        states = np.array(self.find_vectors(self.cd.n, self.cd.n))

        # the number of lineages in each state
        n_lin_states = states.sum(axis=1)

        # the indices of the states with the same number of lineages
        n_lineages = [np.where(n_lin_states == i)[0] for i in np.arange(self.cd.n + 1)]

        # initialize the probabilities
        probs = cast(Dict[Tuple, float], defaultdict(int))
        probs[tuple(states[0])] = 1

        # iterate over the number of lineages
        for i in np.arange(2, self.cd.n)[::-1]:

            # iterate over pairs and determine the probability of transitioning from s1 to s2
            for s1, s2 in itertools.product(states[n_lineages[i + 1]], states[n_lineages[i]]):
                # s = self.find_substates(s1)

                probs[tuple(s2)] += probs[tuple(s1)] * self.get_probs(s1, s2)

        return probs

    def get_sample_config_probs(self) -> Dict[Tuple, float]:
        """
        Get the probabilities of all possible sample configurations.

        :return:
        """
        # initialize the probabilities
        probs = cast(Dict[Tuple, float], defaultdict(int))

        # states indexed by the number of lineages
        states: List[Set[Tuple[int, ...]]] = [set() for _ in range(self.cd.n)]
        states[self.cd.n - 1] = {tuple([self.cd.n] + [0] * (self.cd.n - 1))}

        # initialize the probabilities
        probs[tuple(states[self.cd.n - 1])[0]] = 1

        # iterate over the number of lineages
        for i in np.arange(2, self.cd.n)[::-1]:

            # iterate over states with i + 1 lineages
            for s1_tuple in states[i]:
                s1 = np.array(s1_tuple)

                # iterate over substates of s1
                for s2 in self.find_substates(s1):
                    s2_tuple = tuple(s2)
                    states[i - 1].add(s2_tuple)

                    # determine the probability of transitioning from s1 to s2
                    probs[s2_tuple] += probs[s1_tuple] * self.get_probs(s1, s2)

        return probs

    def get_probs(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the probabilities transitioning from s1 to s2 assuming that s1 has one more lineage than s2.

        :param s1: The starting state
        :param s2: The ending state
        :return: The probability of transitioning from s1 to s2
        """
        diff = s1 - s2
        i = s1.sum()

        if np.sum(diff == -1) == 1:

            # if two lineages of the same class coalesce
            if np.sum(diff == 2) == 1 and np.sum(diff == 0) == self.cd.n - 2:
                # get the number of lineages that were present in s1
                j = s1[diff == 2][0]

                return math.comb(j, 2) / math.comb(i, 2)

            # if two lineages of different classes coalesce
            if np.sum(diff == 1) == 2 and np.sum(diff == 0) == self.cd.n - 3:
                # get the number of lineages that were present in s1
                j1, j2 = s1[diff == 1]

                return math.comb(j1, 1) * math.comb(j2, 1) / math.comb(i, 2)

        return 0

    def nth_moment(self, k: int, alpha: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Get the nth moment.

        :param k: The order of the moment
        :param alpha: Full initial state vector
        :param kwargs: Additional arguments
        :return: The nth moment
        """
        probs = self.get_sample_config_probs()

        R = np.zeros((self.cd.n, self.cd.n))

        for state, prob in probs.items():
            R[:, self.cd.n - sum(state)] += prob * np.array(state)

        # TODO how to combine state for higher moments?
        if k == 2:
            R = np.array([
                [4., 2., 0.88888889, 0.],
                [0., 1., 1.11111111, 0.],
                [0., 0., 0.88888889, 0.],
                [0., 0., 0., 0.]
            ])

        # probs[(0, 2, 0, 0)] ** 2 * np.array([0, 2, 0, 0]) + probs[(1, 0, 1, 0)] ** 2 * np.array([1, 0, 1, 0]) + 2 * (np.array([1, 0, 1, 0]) + np.array([0, 2, 0, 0])) * probs[(0, 2, 0, 0)] * probs[(1, 0, 1, 0)]

        sfs = np.zeros(self.cd.n + 1)
        for i, r in enumerate(R[:-1]):
            sfs[i + 1] = super().nth_moment(k=k, r=self.cd.invert_reward(r), alpha=alpha)

        return sfs


class EmpiricalDistribution(ProbabilityDistribution):
    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples:
        """
        self.samples = np.array(samples, dtype=float)

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the mean absorption time.

        :param alpha:
        :return:
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float | np.ndarray:
        """
        Get the variance in the absorption time.

        :param alpha:
        :return:
        """
        return np.var(self.samples, axis=0)

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t:
        :return:
        """
        x = np.sort(self.samples)
        y = np.arange(1, len(self.samples) + 1) / len(self.samples)

        return np.interp(t, x, y)

    def pdf(self, u: float | np.ndarray, n_bins: int = 10000, sigma: float = None) -> float | np.ndarray:
        """
        Density function.

        :param sigma:
        :param n_bins:
        :param u:
        :return:
        """
        hist, bin_edges = np.histogram(self.samples, range=(0, max(self.samples)), bins=n_bins, density=True)

        # determine bins for u
        bins = np.minimum(np.sum(bin_edges <= u[:, None], axis=1) - 1, np.full_like(u, n_bins - 1, dtype=int))

        # use proper bins for y values
        y = hist[bins]

        # smooth using gaussian filter
        if sigma is not None:
            y = gaussian_filter1d(y, sigma=sigma)

        return y


class CoalescentDistribution:
    @property
    @abstractmethod
    def tree_height(self) -> ProbabilityDistribution:
        """
        Tree height distribution.

        :return:
        :rtype:
        """
        pass

    @property
    @abstractmethod
    def total_branch_length(self) -> ProbabilityDistribution:
        """
        Total branch length distribution.

        :return:
        :rtype:
        """
        pass


class ConstantPopSizeCoalescent(CoalescentDistribution):
    """
    """

    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray | List = None,
            Ne: float | int = 1
    ):
        self.alpha = None

        # coalescent model
        self.model = model

        # sample size
        self.n = n

        # initial conditions
        if alpha is None:
            self.alpha = np.eye(1, self.n, 0)[0]
        elif len(alpha) != self.n:
            raise Exception(f"alpha should be of length n={self.n} but has instead length {len(alpha)}.")
        else:
            self.alpha = np.array(alpha)

        # effective population size
        self.Ne = Ne

        # obtain sub-intensity matrix
        S_sub = self.get_rate_matrix(self.n, self.model)

        # obtain exit rate vector
        self.s = -S_sub @ self.e[:-1]

        # obtain full intensity matrix
        self.S = np.block([
            [S_sub, self.s[:, None]],
            [np.zeros(self.n)]
        ])

    @staticmethod
    def pad(x: np.ndarray) -> np.ndarray:
        """
        Pad a matrix with a row and column of zeros or a vector with a zero.

        :param x: The matrix or vector to pad.
        :return: The padded matrix or vector.
        """
        if x.ndim == 1:
            return np.pad(x, (0, 1), mode='constant', constant_values=0)

        return np.pad(x, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    @staticmethod
    def invert_reward(r: np.ndarray) -> np.ndarray:
        """
        Invert the reward matrix or reward vector.

        :param r: The reward matrix or reward vector.
        :return: The inverted reward matrix or reward vector.
        """
        if r.ndim == 2:
            r = r[np.diag_indices(r.shape[0])]

            r[r != 0] = 1 / r[r != 0]

            return np.diag(r)

        r[r != 0] = 1 / r[r != 0]

        return r

    @cached_property
    def e(self) -> np.ndarray:
        """
        Get a vector with ones of size 1.

        :return:
        :rtype:
        """
        return np.ones(self.n)

    @cached_property
    def T(self) -> np.ndarray:
        """
        The transition matrix.

        :return:
        """
        return expm(self.S)

    @cached_property
    def t(self) -> np.ndarray:
        """
        The exit probability vector.

        :return:
        :rtype:
        """
        return 1 - self.T * self.e

    @cached_property
    def U(self) -> np.ndarray:
        """
        The Green matrix (negative inverse of sub-intensity matrix).

        :return:
        """
        return -inv(self.S[:-1, :-1])

    @cached_property
    def T_inv(self) -> np.ndarray:
        """
        Inverse of transition matrix.

        :return:
        """
        return inv(self.T)

    @staticmethod
    def get_rate_matrix(n: int, model: CoalescentModel) -> np.ndarray:
        def matrix_indices_to_rates(i: int, j: int) -> float:
            """
            Convert matrix indices to k out of b lineages.

            :param i:
            :param j:
            :return:
            """
            return model.get_rate(b=int(n - i), k=int(j + 1 - i))

        # Define sub-intensity matrix.
        # Dividing by Ne here produces unstable results for small population
        # sizes (Ne < 1). We thus add it later to the moments.
        return cast(np.ndarray, np.fromfunction(np.vectorize(matrix_indices_to_rates), (n - 1, n - 1)))

    @cached_property
    def tree_height(self) -> ConstantPopSizeDistribution:
        """
        Tree height distribution.

        :return:
        :rtype:
        """
        return ConstantPopSizeDistribution(
            cd=self,
            r=self.pad(np.ones(self.n - 1))
        )

    @cached_property
    def total_branch_length(self) -> ConstantPopSizeDistribution:
        """
        Total branch length distribution.

        :return:
        :rtype:
        """
        return ConstantPopSizeDistribution(
            cd=self,
            r=self.pad(1 / np.arange(2, self.n + 1)[::-1])
        )


class VariablePopSizeCoalescent(ConstantPopSizeCoalescent):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            demography: Demography = None
    ):
        """

        :param n:
        :param model:
        :param alpha:
        :param demography:
        """
        super().__init__(
            model=model,
            n=n,
            alpha=alpha,
            Ne=1
        )

        self.demography = demography

        if isinstance(demography, PiecewiseConstantDemography):
            self.times = demography.times
            self.pop_sizes = demography.pop_sizes
            self.n_epochs = len(self.times)
        else:
            self.times = np.array([0])
            self.pop_sizes = [self.Ne]
            self.n_epochs = 1

    @cached_property
    def tree_height(self) -> VariablePopSizeDistribution:
        """
        Tree height distribution.

        :return:
        :rtype:
        """
        return VariablePopSizeDistribution(
            cd=self,
            r=self.pad(np.ones(self.n - 1))
        )

    @cached_property
    def total_branch_length(self) -> VariablePopSizeDistribution:
        """
        Total branch length distribution.

        :return:
        :rtype:
        """
        return VariablePopSizeDistribution(
            cd=self,
            r=self.pad(1 / np.arange(2, self.n + 1)[::-1])
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site-frequency spectrum.

        :return:
        :rtype:
        """
        return SFSDistribution(self, r=self.pad(np.ones(self.n - 1)))


class MsprimeCoalescent(CoalescentDistribution):
    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List,
            times: np.ndarray | List,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True
    ):
        """
        Simulate data using msprime.

        :param n:
        :param pop_sizes:
        :param times:
        :param num_replicates:
        :param n_threads:
        :param parallelize:
        """
        self.sfs_counts = None
        self.total_branch_lengths = None
        self.heights = None

        self.n = n
        self.pop_sizes = pop_sizes
        self.times = times
        self.num_replicates = num_replicates
        self.n_threads = n_threads
        self.parallelize = parallelize

    @lru_cache
    def simulate(self):
        """
        Simulate data using msprime.
        """
        # configure demography
        d = ms.Demography()
        d.add_population(initial_size=self.pop_sizes[0])

        # number of replicates for one thread
        num_replicates = self.num_replicates // self.n_threads

        # add population size change is specified
        for i in range(1, len(self.pop_sizes)):
            d.add_population_parameters_change(time=self.times[i], initial_size=self.pop_sizes[i])

        def simulate_batch(_) -> (np.ndarray, np.ndarray, np.ndarray):
            """
            Simulate statistics.

            :param _:
            :type _:
            :return:
            :rtype:
            """
            # simulate trees
            g: Generator = ms.sim_ancestry(
                samples=self.n,
                num_replicates=num_replicates,
                demography=d,
                model=ms.StandardCoalescent(),
                ploidy=1
            )

            # initialize variables
            heights = np.zeros(num_replicates)
            total_branch_lengths = np.zeros(num_replicates)
            sfs = np.zeros((num_replicates, self.n + 1))

            # iterate over trees and compute statistics
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):
                t: tskit.Tree = ts.first()
                total_branch_lengths[i] = t.total_branch_length
                heights[i] = t.time(t.root)
                sfs[i] = calculate_sfs(t)

            return np.concatenate([[heights.T], [total_branch_lengths.T], sfs.T])

        # parallelize and add up results
        res = np.hstack(parallelize(
            func=simulate_batch,
            data=[None] * self.n_threads,
            parallelize=self.parallelize,
            batch_size=num_replicates,
            desc="Simulating trees"
        ))

        # store results
        self.heights, self.total_branch_lengths, self.sfs_counts = res[0], res[1], res[2:]

    def touch(self):
        """
        Touch cached properties.
        """
        self.total_branch_length.touch()
        self.tree_height.touch()
        self.sfs.touch()

    def drop(self):
        """
        Drop simulated data.
        """
        self.heights = None
        self.total_branch_lengths = None
        self.sfs_counts = None

    @cached_property
    def tree_height(self) -> EmpiricalDistribution:
        """
        Tree height distribution.

        :return:
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.heights)

    @cached_property
    def total_branch_length(self) -> EmpiricalDistribution:
        """
        Total branch length distribution.

        :return:
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.total_branch_lengths)

    @cached_property
    def sfs(self) -> EmpiricalDistribution:
        """
        Site-frequency spectrum.

        :return:
        """
        self.simulate()

        return EmpiricalDistribution(samples=self.sfs_counts.T)
