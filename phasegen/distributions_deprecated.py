import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property, lru_cache
from math import factorial
from typing import Generator, List, Callable, cast, Tuple, Dict

import mpmath as mp
import msprime as ms
import numpy as np
import tskit
from matplotlib import pyplot as plt
from multiprocess import Pool
from scipy.linalg import inv, expm
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from .coalescent_models import StandardCoalescent, CoalescentModel
from .demography import Demography, PiecewiseConstantDemography
from .visualization import Visualization


def set_precision(p: int):
    """
    Set precision to p decimal places.

    :param p:
    :return:
    """
    mp.mp.dps = p


# set default precision
set_precision(20)


def e_i(n: int, i: int = 0) -> mp.matrix:
    """
    Get the nth standard unit vector
    :return:
    """
    return mp.matrix(mp.unitvector(n, i + 1))


def to_numpy(m: mp.matrix) -> np.ndarray:
    """
    Convert mpmath matrix to numpy array.
    :param m:
    :type m:
    :return:
    :rtype:
    """
    return np.array([[float(m[i, j]) for j in range(m.cols)] for i in range(m.rows)])


def fractional_power(m: mp.matrix, p: float) -> mp.matrix:
    """
    Fractional power of mpmath matrix using exponentials.
    TODO this sometimes produces complex values with small negative parts
    :param m:
    :type m:
    :param p:
    :type p:
    :return:
    :rtype:
    """
    return mp.expm(float(p) * mp.logm(m)).apply(mp.re)


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        desc: str = None,
) -> np.ndarray:
    """
    Convenience function that parallelizes the given function
    if specified or executes them sequentially otherwise.

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize
    :param pbar: Whether to show a progress bar
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
        iterator = tqdm(iterator, total=len(data), desc=desc)

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


def van_loan(B: mp.matrix, S: mp.matrix, tau: float) -> mp.matrix:
    """
    Use Van Loan's method to evaluate the integral ∫u S(u)B(u)S(u)du.

    :param B: Matrix B
    :param S: Matrix S
    :param tau: Time to integrate over
    :return: Evaluated integral
    """
    n = B.cols

    O = mp.zeros(n, n)

    upper = mp.matrix(S.T.tolist() + B.T.tolist()).T
    lower = mp.matrix(O.T.tolist() + S.T.tolist()).T

    A = mp.matrix(upper.tolist() + lower.tolist())

    # compute matrix exponential of A to determine integral
    V = mp.expm(float(tau) * A)

    return V[:n, n:]


class ProbabilityDistribution(ABC):
    @abstractmethod
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.
        :param alpha:
        :return:
        """
        pass

    @abstractmethod
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.
        :param alpha:
        :return:
        """
        pass

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
        Visualization.plot(
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
        Visualization.plot(
            x=x,
            y=self.pdf(x),
            xlabel='u',
            ylabel='f(u)',
            label=label,
            file=file,
            show=show,
            clear=clear
        )


class PhaseTypeDistribution(ProbabilityDistribution):
    pass


class ConstantPopSizeDistribution(PhaseTypeDistribution):
    """
    Class for calculating statistics under a constant population size coalescent.

    TODO deprecate class complete to avoid implementing SFS?
    """
    def __init__(self, cd: 'ConstantPopSizeCoalescent', r: np.ndarray | List, S: mp.matrix):
        self.cd = cd
        self.r = np.array(r)

        # reward matrix
        self.R = mp.inverse(mp.diag(self.r))

        # sub-intensity matrix with reward applied
        self.S = self.R * S

        # exit rate vector
        self.s = -self.S * self.cd.e

    @cached_property
    def S_full(self) -> mp.matrix:
        """
        Full intensity matrix

        :return:
        """
        upper = mp.matrix(self.S.T.tolist() + self.s.T.tolist()).T
        lower = mp.zeros(1, self.cd.n)

        return mp.matrix(upper.tolist() + lower.tolist())

    @cached_property
    def T(self) -> mp.matrix:
        """
        The transition matrix.

        :return:
        """
        return mp.expm(self.S)

    @cached_property
    def T_full(self) -> mp.matrix:
        """
        The full transition matrix

        :return:
        """
        return mp.expm(self.S_full)

    @cached_property
    def R_full(self):
        """
        The rewards matrix including the absorbing state.

        :return:
        """
        O = mp.zeros(1, 1)

        upper = mp.matrix(self.R.T.tolist() + O.T.tolist()).T
        lower = mp.matrix(O.T.tolist() + O.T.tolist()).T

        A = mp.matrix(upper.tolist() + lower.tolist())

        return A

    @cached_property
    def t(self) -> mp.matrix:
        """
        The exit probability vector.

        :return:
        :rtype:
        """
        return 1 - self.T * self.cd.e

    @cached_property
    def U(self) -> mp.matrix:
        """
        The Green matrix.
        :return:
        """
        return -mp.inverse(self.S)

    @cached_property
    def T_inv_full(self) -> mp.matrix:
        """
        Inverse of full transition matrix.
        :return:
        """
        return mp.inverse(self.T_full)

    @cached_property
    def T_inv(self) -> mp.matrix:
        """
        Inverse of transition matrix.
        :return:
        """
        return mp.inverse(self.T)

    def nth_moment(self, k: int, alpha: np.ndarray = None) -> float:
        """
        Get the nth moment.
        :param k:
        :param alpha:
        :return:
        """
        alpha = self.cd.get_alpha(alpha)

        # self.Ne ** k is the rescaling due to population size
        return float(self.cd.Ne ** k * factorial(k) * (alpha * self.U ** k * self.cd.e)[0, 0])

    @cached_property
    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.
        :param alpha:
        :return:
        """
        return self.nth_moment(1, alpha)

    @cached_property
    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.
        :param alpha:
        :return:
        """
        return self.nth_moment(2, alpha) - self.nth_moment(1, alpha) ** 2

    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.
        :param t:
        :return:
        """

        def cdf(t: float) -> float:
            return 1 - float((self.cd.alpha * fractional_power(self.T, t / self.cd.Ne) * self.cd.e)[0, 0])

        return np.vectorize(cdf)(t)

    def pdf(self, u) -> float | np.ndarray:
        """
        Density function.
        :param u:
        :return:
        """

        def pdf(u: float) -> float:
            return float((self.cd.alpha * fractional_power(self.T, u) * self.s)[0, 0]) / self.cd.Ne

        return np.vectorize(pdf)(u)


class VariablePopSizeDistribution(ConstantPopSizeDistribution):
    """
    Distribution of absorption times for a variable population size coalescent.
    """

    def __init__(self, cd: 'VariablePopSizeCoalescent', r: np.ndarray | List, S: mp.matrix):
        super().__init__(cd=cd, r=r, S=S)

        # reassign to make the IDE aware of the new subclass
        self.cd = cd

    def get_I_mean(self, i, j, tau) -> mp.matrix:
        """
        Use Van Loan's method to evaluate the integral I(a, b, i=alpha, j=beta)
        in equation (3) in https://doi.org/10.1239/jap/1324046009. Van Loan's method
        is described in section 4 of this paper.

        :param i:
        :param j:
        :param tau:
        :return:
        """
        # determine B so that TBT[a, b] describes the probabilities
        # of transitioning from state 'a' to alpha to beta to 'b'.
        # Note that this step requires a lot of numerical precision.
        B = self.T_inv_full * self.T_full[:, j] * self.T_full[i, :] * self.T_inv_full

        # construct matrix consisting of B and the rate matrices
        Q = self.S_full
        O = mp.zeros(self.cd.n, self.cd.n)

        upper = mp.matrix(Q.T.tolist() + B.T.tolist()).T
        lower = mp.matrix(O.T.tolist() + Q.T.tolist()).T

        A = mp.matrix(upper.tolist() + lower.tolist())

        # compute matrix exponential of A to determine integral
        return mp.expm(float(tau) * A)[:self.cd.n, self.cd.n:]

    def get_I_var(self, i, j, k, l, tau) -> mp.matrix:
        """
        Use Van Loan's method to evaluate the integral I(a, b, i=alpha, j=beta, k=gamma, l=delta)
        in equation (4) in https://doi.org/10.1239/jap/1324046009.  Van Loan's method
        is described in section 4 of this paper.

        :param i:
        :param j:
        :param k:
        :param l:
        :param tau:
        :return:
        """
        # determine B so that TBT[a, b] describes the probabilities
        # of transitioning from state a to alpha to beta to b.
        # Note that this step requires a lot of numerical precision.
        # TODO check if B1 and B2 are correct
        B1 = self.T_inv_full * self.T_full[:, j] * self.T_full[i, :] * self.T_inv_full
        B2 = self.T_inv_full * self.T_full[:, l] * self.T_full[k, :] * self.T_inv_full

        # construct matrix consisting of B and the rate matrices
        Q = self.S_full
        O = mp.zeros(self.cd.n, self.cd.n)
        upper = mp.matrix(Q.T.tolist() + B1.T.tolist() + O.T.tolist()).T
        middle = mp.matrix(O.T.tolist() + Q.T.tolist() + B2.T.tolist()).T
        lower = mp.matrix(O.T.tolist() + O.T.tolist() + Q.T.tolist()).T

        A = mp.matrix(upper.tolist() + middle.tolist() + lower.tolist())

        # compute matrix exponential of A to determine integral
        var = mp.expm(float(tau) * A)[:self.cd.n, 2 * self.cd.n:]

        return var

    def get_mean_sojourn_times(self, i: int, tau: float) -> mp.matrix:
        """
        Get the endpoint-conditioned amount of time spent in state ``i``.

        :param i: State for which to compute the sojourn time.
        :param tau: Time interval for which to compute the sojourn time.
        :return: ``U[a, b]`` describes the amount of time spent in state ``i``,
            given that the state was ``a`` at time 0 and ``b`` at time tau.
        """
        # obtain matrix G using Van Loan's method
        G = self.get_I_mean(i, i, tau / self.cd.Ne)

        # normalize by transition probabilities.
        U = self.normalize_by_T_tau(G, tau)

        # scale by Ne
        mean = float(self.cd.Ne) * U

        return mean

    def get_covarying_sojourn_times(self, i: int, j: int, tau: float) -> mp.matrix:
        """
        :param i:
        :param j:
        :param tau:
        :return:
        """
        # obtain matrix G using Van Loan's method
        G1 = self.get_I_var(i, i, j, j, tau / self.cd.Ne)

        # obtain same matrix with states switched as in
        # https://doi.org/10.2202/1544-6115.1127
        G2 = self.get_I_var(j, j, i, i, tau / self.cd.Ne)

        # add matrices
        G = G1 + G2

        # normalize by transition probabilities.
        U = self.normalize_by_T_tau(G, tau)

        # scale by Ne
        return float(self.cd.Ne ** 2) * U

    def normalize_by_T_tau(self, G: mp.matrix, tau: float) -> mp.matrix:
        """
        Normalize by transition probabilities.
        :param G:
        :type G:
        :param tau:
        :type tau:
        :return:
        :rtype:
        """
        # get transition probabilities over time tau
        T_tau = fractional_power(self.T_full, tau / self.cd.Ne)

        # mpmath fails on division by zero, so we iterate through here.
        U = mp.zeros(G.rows, G.cols)
        for i in range(G.rows):
            for j in range(G.cols):

                # for states where T_tau == 0, i.e. for impossible transitions,
                # we have a sojourn time of 0
                if T_tau[i, j] != 0:
                    # divide by transition probabilities to normalize G
                    # as in https://doi.org/10.2202/1544-6115.1127
                    U[i, j] = G[i, j] / T_tau[i, j]

        return U

    def nth_moment(self, k: int, alpha: np.ndarray = None) -> float:
        """
        Get the nth moment.
        Only the first two moments are currently implemented.
        :param k:
        :param alpha:
        :return:
        """
        if k == 1:
            return self.mean_and_m2[0] if alpha is None else self.mean_and_m2(mp.matrix(alpha))[0]

        if k == 2:
            return self.mean_and_m2[1] if alpha is None else self.mean_and_m2(mp.matrix(alpha))[1]

        raise NotImplementedError('Only the first second moments are implemented.')

    @cached_property
    def mean_and_m2(self, alpha: np.ndarray = None) -> (float, float):
        """
        Calculate the first and second moments in the absorption time.

        We need the absorption probability in a certain epoch and
        the expected absorption time conditional on absorption in that epoch.
        We can get the expected absorption time by conditioning on the endpoint
        and determining the amount of time we spend in the absorbing state.
        We can get the absorption probability from the transition matrix.
        :return:
        """
        alpha = self.cd.get_alpha(alpha)

        # absorption times conditional on when the epoch ends
        absorption_times = np.zeros(self.cd.n_epochs)

        # conditional second moments in absorption time
        absorption_m2 = np.zeros(self.cd.n_epochs)

        # unconditional absorption probabilities
        absorption_probs = np.zeros(self.cd.n_epochs)

        # Probability of not having reached the absorbing state until
        # the current epoch.
        no_absorption = np.zeros(self.cd.n_epochs)

        # iterate over epochs
        for i in range(self.cd.n_epochs):
            # set Ne of current epoch
            self.cd.set_Ne(self.cd.pop_sizes[i])

            # probability of not having reach the absorbing state until now
            no_absorption[i] = np.prod(1 - absorption_probs[:i])

            # we need to end-point condition all but the last epoch
            if i < self.cd.n_epochs - 1:

                # determine tau, the amount of time spend with the
                # current population size
                tau = self.cd.times[i + 1] - self.cd.times[i]

                B = mp.eye(self.cd.n)
                B[self.cd.n - 1, self.cd.n - 1] = 0
                N = self.normalize_by_T_tau(van_loan(S=self.S_full, B=B, tau=tau / self.cd.Ne), tau)

                # Get absorption time depending on initial states.
                # Note that we skip the absorbing state for now as
                # we currently do not allow the chain to start in it.
                # M[:-1, -1] describes the time in the absorbing state,
                # given that it starts in state i at time 0 and ends in
                # the absorbing state at time tau.
                # Distributing these times according to alpha,
                # we obtain the expected time spent in the
                # absorbing state given absorption at time tau.
                # tau minus this is then the expected absorption time.
                # Note that M[:-1, -1] did not work for mpmath
                absorption_times[i] = np.dot(alpha, N[:-1, N.cols - 1])

                # second moment in sojourn time for absorbing state
                M2 = self.get_covarying_sojourn_times(i=self.cd.n - 1, j=self.cd.n - 1, tau=tau)

                # M2[:-1, -1] is the second moment in the time spent in the absorbing
                # state. We convert it here to the second moment in the absorption time
                # and multiply by the initial states as done for the mean.
                absorption_m2[i] = np.dot(alpha, M2[:-1, M2.cols - 1]) + 2 * tau * absorption_times[i] - tau ** 2

                # Get probability of states at time tau.
                # These are the initial state probabilities for the next epoch.
                alpha = (alpha * fractional_power(self.T, tau / self.cd.pop_sizes[i]))

                # absorption probability in current state
                absorption_probs[i] = 1 - float(np.sum(alpha))

                # Normalize alpha.
                # We do this because alpha needs to sum to 1
                # and this alpha is conditional on not having
                # reached absorption yet.
                alpha /= np.sum(alpha)

            else:
                # for the last epoch we can simply calculate the unconditional moments
                absorption_m2[i] = super().nth_moment(alpha=alpha, k=2)
                absorption_times[i] = super().nth_moment(alpha=alpha, k=1)
                absorption_probs[i] = 1

        # Calculate total absorption probabilities i.e. the probability
        # of absorption in epoch i
        total_absorption_probs = no_absorption * absorption_probs

        # The total absorption times are the absorption times within
        # each epoch plus the times spent in the previous epochs
        total_absorption_times = absorption_times + self.cd.times

        # Here we adjust the second moment by the time spent in the
        # previous epochs
        total_absorption_m2 = absorption_m2 + self.cd.times ** 2 + 2 * self.cd.times * absorption_times

        # We finally get the unconditional moments by multiplying
        # with their total absorption probabilities and summing up
        mean = (total_absorption_probs * total_absorption_times).sum()
        m2 = (total_absorption_probs * total_absorption_m2).sum()

        return mean, m2

    def get_alphas(self) -> np.ndarray:
        """
        Get initial values at beginning of epochs.
        :return:
        :rtype:
        """

        # initial values at beginning of epochs.
        alphas = np.zeros(self.cd.n_epochs, dtype=mp.matrix)
        alphas[0] = self.cd.alpha

        # iterate through epochs and compute initial values
        for i in range(self.cd.n_epochs - 1):
            # Ne of current epoch
            Ne = self.cd.pop_sizes[i]

            tau = (self.times[i + 1] - self.times[i]) / Ne

            # update alpha for the time spent in the current epoch
            alphas[i + 1] = alphas[i] * fractional_power(self.T, tau)

        return alphas

    def cdf(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.
        :param t:
        :return:
        """
        # alphas = self.get_alphas()
        alphas = self.alphas

        def cdf(t: float) -> float:
            # determine index of last epoch given t
            j = np.sum(self.times <= t) - 1

            # Ne and start time of the last epoch
            Ne = self.cd.pop_sizes[j]
            start_time = self.times[j]

            # cumulative distribution function at time t
            tau = (t - start_time) / Ne
            return 1 - float((alphas[j] * fractional_power(self.T, tau) * self.cd.e)[0, 0])

        return np.vectorize(cdf)(t)

    def pdf(self, u) -> float | np.ndarray:
        """
        Density function.
        :param u:
        :return:
        """

        def pdf(u: float) -> float:
            # determine index of current epoch
            j = np.sum(self.times <= u) - 1

            # Ne and start time of the last epoch
            Ne = self.cd.pop_sizes[j]
            start_time = self.times[j]

            tau = (u - start_time) / Ne

            # compute density for the current epoch
            return float((self.alphas[j] * fractional_power(self.T, tau) * self.s)[0, 0]) / Ne

        return np.vectorize(pdf)(u)


class VariablePopSizeDistributionManual(VariablePopSizeDistribution):

    def __init__(self, cd: 'VariablePopSizeCoalescent', r: np.ndarray | List, S: mp.matrix):
        """
        Restore the original intensity matrix.

        :return:
        """
        self.cd = cd
        self.r = np.array(r)

        # reward matrix
        self.R = mp.inverse(mp.diag(self.r))

        # sub-intensity matrix with reward applied
        self.S = S

        # exit rate vector
        self.s = -self.S * self.cd.e

    @cached_property
    def mean_and_m2(self, alpha: np.ndarray = None) -> (float, float):
        """
        Calculate the first and second moments in the absorption time.

        We need the absorption probability in a certain epoch and
        the expected absorption time conditional on absorption in that epoch.
        We can get the expected absorption time by conditioning on the endpoint
        and determining the amount of time we spend in the absorbing state.
        We can get the absorption probability from the transition matrix.

        :param alpha: Initial state probabilities
        :return:
        """
        alpha = self.cd.get_alpha(alpha)

        # absorption times conditional on when the epoch ends
        absorption_times = np.zeros(self.cd.n_epochs)

        # sojourn times in the non-absorbing states per epoch given that no absorption has occurred
        sojourn_times = np.zeros((self.cd.n_epochs, self.cd.n - 1))

        # sojourn times in the non-absorbing states per epoch given that no absorption has occurred
        sojourn_times_complete = np.zeros((self.cd.n_epochs, self.cd.n - 1))

        # conditional second moments in absorption time
        absorption_m2 = np.zeros(self.cd.n_epochs)

        # second moments in sojourn time in non-absorbing states per epoch
        sojourn_m2 = np.zeros((self.cd.n_epochs, self.cd.n - 1))

        # unconditional absorption probabilities
        absorption_probs = np.zeros(self.cd.n_epochs)

        # Probability of not having reached the absorbing state until
        # the current epoch.
        no_absorption = np.zeros(self.cd.n_epochs)

        # iterate over epochs
        for i in range(self.cd.n_epochs):
            # set Ne of current epoch
            self.cd.set_Ne(self.cd.pop_sizes[i])

            # probability of not having reach the absorbing state until now
            no_absorption[i] = np.prod(1 - absorption_probs[:i])

            # we need to end-point condition all but the last epoch
            if i < self.cd.n_epochs - 1:

                # determine tau, the amount of time spend with the
                # current population size
                tau = self.cd.times[i + 1] - self.cd.times[i]
            else:
                # for the last epoch we set tau to a large value
                tau = 1000000

            # Get probability of states at time tau.
            # These are the initial state probabilities for the next epoch.
            alpha_next = (alpha * fractional_power(self.T, tau / self.cd.pop_sizes[i]))

            # sojourn time for absorbing state
            M = self.get_mean_sojourn_times(i=self.cd.n - 1, tau=tau)

            # tau minus this is the expected absorption time.
            absorption_times[i] = tau - np.dot(alpha, M[:-1, M.cols - 1])

            for j in range(self.cd.n - 1):
                # Mi[:-1, -1] describes the time spent in state j
                # given that it starts in state i at time 0 and ends in
                # the absorbing state at time tau.
                # Distributing these times according to alpha,
                # we obtain the expected time spent in the
                # absorbing state given absorption at time tau.
                # Note that Mi[:-1, -1] did not work for mpmath
                Mi = self.get_mean_sojourn_times(i=j, tau=tau)
                sojourn_times[i][j] = np.dot(alpha, Mi[:-1, Mi.cols - 1])
                sojourn_times_complete[i][j] = float((alpha * Mi[:-1, :-1] * (alpha_next / np.sum(alpha_next)).T)[0, 0])

            sojourn_times_complete[i] /= sojourn_times_complete[i].sum() / tau

            # second moment in sojourn time for absorbing state
            M2 = self.get_covarying_sojourn_times(i=self.cd.n - 1, j=self.cd.n - 1, tau=tau)

            # M2[:-1, -1] is the second moment in the time spent in the absorbing
            # state. We convert it here to the second moment in the absorption time
            # and multiply by the initial states as done for the mean.
            absorption_m2[i] = np.dot(alpha, M2[:-1, M2.cols - 1]) + 2 * tau * absorption_times[i] - tau ** 2

            for j in range(self.cd.n - 1):
                # M2i[:-1, -1] is the second moment of time spent in the state j.
                M2i = self.get_covarying_sojourn_times(i=j, j=j, tau=tau)
                sojourn_m2[i][j] = np.dot(alpha, M2i[:-1, M2i.cols - 1])

            # absorption probability in current state
            absorption_probs[i] = 1 - float(np.sum(alpha_next))

            # Normalize alpha.
            # We do this because alpha needs to sum to 1
            # and this alpha is conditional on not having
            # reached absorption yet.
            alpha = alpha_next / np.sum(alpha_next)

        # Calculate total absorption probabilities i.e. the probability
        # of absorption in epoch i
        total_absorption_probs = no_absorption * absorption_probs

        # get inverse reward matrix
        R = np.linalg.inv(to_numpy(self.R))

        absorption_times_complete = np.cumsum(np.insert((sojourn_times_complete @ R).sum(axis=1), 0, 0)[:-1])

        # The total absorption times are the absorption times within
        # each epoch plus the times spent in the previous epochs
        total_absorption_times = (sojourn_times @ R).sum(axis=1) + absorption_times_complete

        # Here we adjust the second moment by the time spent in the
        # previous epochs
        total_absorption_m2 = (sojourn_m2 @ R).sum(axis=1) + self.cd.times ** 2 + \
                              2 * self.cd.times * (sojourn_times @ R).sum(axis=1)

        # We finally get the unconditional moments by multiplying
        # with their total absorption probabilities and summing up
        mean = np.dot(total_absorption_probs, total_absorption_times)
        m2 = np.dot(total_absorption_probs, total_absorption_m2)

        return mean, m2


class VariablePopSizeDistributionIPH(VariablePopSizeDistribution):
    """
    Variable population size distribution using IPH.
    """

    def __init__(self, cd: 'VariablePopSizeCoalescentIPH', r: np.ndarray | List, S: mp.matrix):
        super().__init__(cd=cd, r=r, S=S)

        # reassign to make the IDE aware of the new subclass
        self.cd = cd

    def nth_moment_numerical(self, k: int, alpha: np.ndarray = None) -> float:
        """
        Get the nth moment.

        :param k:
        :param alpha:
        :return:
        """
        S = to_numpy(self.S_full)
        R = to_numpy(self.R_full)

        def integrand(u: float, t: float) -> np.ndarray:
            """

            :param u: Current time
            :param t: End time
            :return: Current rate
            """
            P1 = expm(self.cd.demography.get_cum_rate(t=u) * S)
            P2 = expm(self.cd.demography.get_cum_rate(t=t - u) * S)

            return P1 @ R @ P2

        def integrate(func: Callable, lower, upper, n: int) -> float | np.ndarray:
            """
            Integrate over given function using Monte Carlo integration.

            :param func: Function to integrate over
            :param lower: Lower bound
            :param upper: Upper bounds
            :param n: Number of random samples
            :return:
            """
            x = np.random.uniform(low=lower, high=upper, size=n)
            samples = np.array([func(float(z)) for z in x])
            return samples.mean(axis=0) * (upper - lower)

        tau = 100
        mean = integrate(lambda u: integrand(u, tau), 0, tau, n=10000)

        pass

    def nth_moment(self, k: int, alpha: np.ndarray = None, r: np.ndarray = None) -> float:
        """
        Get the nth moment.

        :param k:
        :param alpha:
        :param r: Full reward vector
        :return:
        """
        if r is None:
            r = list(self.r) + [0]

        R = mp.diag(r)

        S = self.cd.tree_height.S_full
        T = self.cd.tree_height.T_full
        e = self.cd.e_full

        means = np.zeros(self.cd.n_epochs)
        alphas = np.zeros(self.cd.n_epochs, dtype=mp.matrix)

        alphas[0] = self.cd.alpha_full

        # iterate through epochs and compute initial values
        for i in range(0, self.cd.n_epochs):
            # Ne of current epoch
            Ne = self.cd.pop_sizes[i]

            if i < self.cd.n_epochs - 1:
                # time spent in current epoch scaled by Ne
                tau = (self.cd.times[i + 1] - self.cd.times[i]) / Ne

                # update alpha for the time spent in the current epoch
                alphas[i + 1] = alphas[i] * fractional_power(T, tau)
            else:
                tau = 1000

            U = van_loan(S=S, B=R, tau=tau)

            means[i] = float(Ne ** k * factorial(k) * (alphas[i] * U ** k * e)[0, 0])

        return means.sum()

    def nth_moment_discrete(self, k: int, alpha: np.ndarray = None) -> float:
        """
        Get the nth moment.

        :param k:
        :param alpha:
        :return:
        """
        N = cast(list, self.cd.pop_sizes.tolist())
        t = cast(list, self.times.tolist())
        S = self.S

        I = mp.eye(self.cd.n - 1)

        c1 = (mp.expm(S * t[1] / N[0]) * (S * t[1] - N[0] * I) + N[0] * I) * mp.powm(S, -2)

        c2 = (mp.expm(S * t[1] / N[0]) * (N[1] * I - S * t[1])) * mp.powm(S, -2)

        t1 = float((self.alphas[0] * c1 * mp.inverse(self.R) * self.s)[0, 0])
        t2 = float((self.alphas[0] * c2 * mp.inverse(self.R) * self.s)[0, 0])

        return t1 + t2


class SFSDistribution(VariablePopSizeDistributionIPH):
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

    @cached_property
    def mean(self, **kwargs) -> np.ndarray:
        """
        Get the nth moment.

        :param kwargs:
        :return:
        """

        states = np.array(self.find_vectors(self.cd.n, self.cd.n))

        # iterate over the number of lineages
        # and merge the states with the same number of lineages
        same = np.where(states.sum(axis=1) == self.cd.n)[0]

        probs = cast(Dict[Tuple, float], defaultdict(int))
        probs[tuple(states[0])] = 1
        states_merged: List[List[int]] = [states[0]]

        for i in np.arange(2, self.cd.n)[::-1]:
            # get the number of states with i lineages
            same_next = np.where(states.sum(axis=1) == i)[0]

            for s1, s2 in itertools.product(states[same], states[same_next]):
                diff = s1 - s2

                if 2 in diff:
                    # get the number of lineages that were present in s1
                    j = s1[diff == 2][0]
                    probs[tuple(s2)] += probs[tuple(s1)] * j / (i + 1) * (j - 1) / i

                if 1 in diff:
                    # get the number of lineages that were present in s1
                    j1, j2 = s1[diff == 1]
                    probs[tuple(s2)] += probs[tuple(s1)] * 2 * j1 / (i + 1) * j2 / i

            same = same_next

        R = np.zeros((self.cd.n, self.cd.n))

        for state, prob in probs.items():
            R[:, self.cd.n - sum(state)] += prob * np.array(state)

        R2 = np.array([
            [4, 2, 2 / 3, 0],
            [0, 1, 2 / 3, 0],
            [0, 0, 2 / 3, 0]
        ])

        return np.array([0.] + [self.nth_moment(k=1, r=r) for r in R[:-1]] + [0.])


class SFSDistributionDeprecated:
    """
    Variable population size distribution using IPH.
    """

    def __init__(self, cd: 'VariablePopSizeCoalescent'):
        """

        :param cd:
        """
        self.cd = cd

    def get_P(self, i: int):
        """

        """

        # TODO extra case if no zero rewards, or no non-zero rewards (e.g. when n = 2)

        r = np.eye(1, self.cd.n - 1, i)[0]

        E_plus = np.where(r > 0)[0]  # Indices of states with positive reward
        E_zero = np.where(r == 0)[0]  # Indices of states with zero reward

        Q = to_numpy(self.cd.S)
        Q_plus_plus = Q[np.ix_(E_plus, E_plus)]
        Q_plus_zero = Q[np.ix_(E_plus, E_zero)]
        Q_zero_plus = Q[np.ix_(E_zero, E_plus)]
        Q_zero_zero = Q[np.ix_(E_zero, E_zero)]

        I = np.eye(Q_zero_zero.shape[0])

        P = Q_plus_plus + Q_plus_zero @ inv(I - Q_zero_zero) @ Q_zero_plus

        alpha = to_numpy(self.cd.alpha)[0]
        pi = alpha[E_plus] + alpha[E_zero] @ inv(I - Q_zero_zero) @ Q_zero_plus

        return P, pi

    @cached_property
    def mean(self) -> np.ndarray:
        """
        Get the nth moment.

        :param k:
        :param alpha:
        :return:
        """
        sfs = np.zeros(self.cd.n + 1)
        for i in range(self.cd.n - 1):
            P, pi = self.get_P(i)

            sfs[self.cd.n - i - 1] = (self.cd.n - i) * self.cd.Ne * (pi * -inv(P) * np.ones((P.shape[0])))

            """N = cast(list, self.cd.pop_sizes.tolist())
            t = cast(list, self.cd.times.tolist())

            I = np.eye(P.shape[0])

            c1 = (expm(P * t[1] / N[0]) @ (P * t[1] - N[0] * I) + N[0] * I) @ inv(P) ** 2

            c2 = (expm(P * t[1] / N[0]) @ (N[1] * I - P * t[1])) @ inv(P) ** 2

            t1 = pi @ c1 @ p
            t2 = pi @ c2 @ p

            sfs[self.cd.n - i] = t1 + t2"""

        return sfs


class EmpiricalDistribution(ProbabilityDistribution):
    def __init__(self, samples: np.ndarray | list):
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
        :type sigma:
        :param n_bins:
        :type n_bins:
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

    @abstractmethod
    def get_n_segregating(self, theta: float = 1.0, alpha: np.ndarray = None) -> np.ndarray:
        """
        The site-frequency spectrum.
        :param theta:
        :type theta:
        :param alpha:
        :type alpha:
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
            alpha: np.ndarray | List | mp.matrix = None,
            Ne: float | int = 1
    ):

        self.alpha = None

        # coalescent model
        self.model = model

        # sample size
        self.n = n

        self.set_alpha(alpha)

        # effective population size
        self.Ne = Ne

        # obtain sub-intensity matrix
        self.S = mp.matrix(self.get_rate_matrix(self.n, self.model))

    @cached_property
    def e(self) -> mp.matrix:
        """
        Get a vector with ones of size n.
        :return:
        :rtype:
        """
        return mp.matrix(np.ones(self.n - 1))

    @cached_property
    def e_full(self) -> mp.matrix:
        """
        Get a vector with ones of size n.
        :return:
        :rtype:
        """
        return mp.matrix(np.ones(self.n))

    @cached_property
    def alpha_full(self) -> mp.matrix:
        """
        Get a vector with ones of size n.
        :return:
        :rtype:
        """
        return mp.matrix(self.get_alpha().T.tolist() + mp.zeros(1).T.tolist()).T

    @staticmethod
    def get_rate_matrix(n: int, model: CoalescentModel):
        def matrix_indices_to_rates(i: int, j: int) -> float:
            """
            Convert matrix indices to k out of b lineages.
            :param i:
            :type i:
            :param j:
            :type j:
            :return:
            :rtype:
            """
            return model.get_rate(b=int(n - i), k=int(j + 1 - i))

        # Define sub-intensity matrix.
        # Dividing by Ne here produces unstable results for small population
        # sizes (Ne < 1). We thus add it later to the moments.
        return np.fromfunction(np.vectorize(matrix_indices_to_rates), (n - 1, n - 1))

    @cached_property
    def tree_height(self) -> ConstantPopSizeDistribution:
        """
        Tree height distribution.
        :return:
        :rtype:
        """
        return ConstantPopSizeDistribution(
            cd=self,
            S=self.S,
            r=np.ones(self.n - 1)
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
            S=self.S,
            r=np.arange(2, self.n + 1)[::-1]
        )

    def set_alpha(self, alpha: np.ndarray | list | mp.matrix = None):
        """
        Set the initial state.
        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        if alpha is not None and len(alpha) != self.n - 1:
            raise Exception(f"alpha should be of length n - 1 = {self.n - 1} but has instead length {len(alpha)}.")

        # initial conditions
        if alpha is None:
            self.alpha = e_i(self.n - 1, 0).T
        else:
            self.alpha = self.get_alpha(alpha)

    def get_alpha(self, alpha: np.ndarray | list | mp.matrix = None):
        """
        Convenience function for obtaining alpha.
        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        if alpha is None:
            return self.alpha

        if isinstance(alpha, (np.ndarray, list)):
            return mp.matrix(alpha).T

        # assume alpha is of type mp.matrix
        return alpha

    def set_Ne(self, Ne: int | float):
        """
        Change the effective population size
        :return:
        """
        # there is nothing we need to do here as all results
        # are rescaled by Ne
        self.Ne = Ne

    def get_n_segregating(self, theta: float = 1.0, alpha: np.ndarray = None) -> np.ndarray:
        """
        Get the SFS i.e. the expected number of segregating sites.
        This follows https://doi.org/10.1016/j.tpb.2019.02.001
        :param theta:
        :type theta:
        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        alpha = self.get_alpha(alpha)

        lam = theta / 2

        # compute resolvent
        I = mp.eye(self.n - 1)
        P = mp.inverse(I - float(1 / lam) * self.total_branch_length.S)
        p = self.e - P * self.e

        n_segregating = np.zeros(self.n + 1)
        P_i = I

        # iterate through number of segregating sites.
        for i in range(self.n):
            n_segregating[i] = (alpha * P_i * p)[0, 0]
            P_i *= P

        return n_segregating


class VariablePopSizeCoalescent(ConstantPopSizeCoalescent):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            demography: Demography = None
    ):
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
            S=self.S,
            r=np.ones(self.n - 1)
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
            S=self.S,
            r=np.arange(2, self.n + 1)[::-1]
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site-frequency spectrum.

        :return:
        :rtype:
        """
        return SFSDistribution(self, r=np.ones(self.n - 1), S=self.S)

    def get_n_segregating(self, theta: float = 1.0, alpha: np.ndarray = None) -> np.ndarray:
        """
        TODO implement this
        :param theta:
        :type theta:
        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        pass


class VariablePopSizeCoalescentManual(ConstantPopSizeCoalescent):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            demography: Demography = None
    ):
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
    def tree_height(self) -> VariablePopSizeDistributionManual:
        """
        Tree height distribution.
        :return:
        :rtype:
        """
        return VariablePopSizeDistributionManual(
            cd=self,
            S=self.S,
            r=np.ones(self.n - 1)
        )

    @cached_property
    def total_branch_length(self) -> VariablePopSizeDistributionManual:
        """
        Total branch length distribution.
        :return:
        :rtype:
        """
        return VariablePopSizeDistributionManual(
            cd=self,
            S=self.S,
            r=np.arange(2, self.n + 1)[::-1]
        )

    def get_n_segregating(self, theta: float = 1.0, alpha: np.ndarray = None) -> np.ndarray:
        """
        TODO implement this
        :param theta:
        :type theta:
        :param alpha:
        :type alpha:
        :return:
        :rtype:
        """
        pass


class VariablePopSizeCoalescentIPH(ConstantPopSizeCoalescent):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            demography: Demography = None
    ):
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
            self.pop_sizes = np.array([self.Ne])
            self.n_epochs = 1

    @cached_property
    def tree_height(self) -> VariablePopSizeDistributionIPH:
        """
        Tree height distribution.
        :return:
        :rtype:
        """
        return VariablePopSizeDistributionIPH(
            cd=self,
            S=self.S,
            r=np.ones(self.n - 1)
        )

    @cached_property
    def total_branch_length(self) -> VariablePopSizeDistributionIPH:
        """
        Total branch length distribution.
        :return:
        :rtype:
        """
        return VariablePopSizeDistributionIPH(
            cd=self,
            S=self.S,
            r=np.arange(2, self.n + 1)[::-1]
        )

    @cached_property
    def sfs(self) -> SFSDistribution:
        """
        Site-frequency spectrum.

        :return:
        :rtype:
        """
        return SFSDistribution(self, r=np.ones(self.n - 1), S=self.S)


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
            # number of replicates for one thread
            num_replicates = self.num_replicates // self.n_threads

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
            desc="Simulating trees"
        ))

        # store results
        self.heights, self.total_branch_lengths, self.sfs_counts = res[0], res[1], res[2:]

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
