import functools
from abc import ABC, abstractmethod
from functools import cached_property
from math import factorial
from typing import Callable
from typing import List
from scipy.special import comb

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import seaborn as sns
import sympy as sp
from scipy.special import beta


def set_precision(p: int):
    """
    Set precision to p decimal places.
    :param p:
    :type p:
    :return:
    :rtype:
    """
    mp.mp.dps = p


# set default precision
set_precision(20)


class CoalescentModel(ABC):
    @abstractmethod
    def get_rate(self, b: int, k: int):
        """
        Get exponential rate for a merger of k out of b lineages.
        :param b:
        :type b:
        :param k:
        :type k:
        :return:
        :rtype:
        """
        pass


class StandardCoalescent(CoalescentModel):
    def get_rate(self, b: int, k: int):
        """
        Get exponential rate for a merger of k out of b lineages.
        :param b:
        :type b:
        :param k:
        :type k:
        :return:
        :rtype:
        """
        # two lineages can merge with a rate depending on b
        if k == 2:
            return b * (b - 1) / 2

        # the opposite of above
        if k == 1:
            return -self.get_rate(b=b, k=2)

        # no other mergers can happen
        return 0


class BetaCoalescent(CoalescentModel):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def get_rate(self, b: int, k: int):
        if k < 1 or k > b:
            return 0

        if k == 1:
            return -np.sum([self.get_rate(b, i) for i in range(2, b + 1)])

        return comb(b, k, exact=True) * beta(k - self.alpha, b - k + self.alpha) / beta(self.alpha, 2 - self.alpha)


class LambdaCoalescent(CoalescentModel):
    @abstractmethod
    def get_density(self) -> Callable:
        pass

    def get_rate(self, i: int, j: int):
        x = sp.symbols('x')
        integrand = x ** (i - 2) * (1 - x) ** (j - i)

        integral = sp.Integral(integrand * self.get_density()(x), (x, 0, 1))
        return float(integral.doit())


class Demography:
    pass


class PiecewiseConstantDemography(Demography):
    """
    Demographic scenario where containing a number
    of instantaneous population size changes.
    """

    def __init__(self, pop_sizes: np.ndarray | List, times: np.ndarray | List):
        """
        The population sizes and times these changes occur backwards in time.
        We need to start with a population size at time 0 but this time
        can be omitted in which case len(pop_size) == len(times) + 1.
        :param pop_sizes:
        :param times:
        """
        if len(pop_sizes) == 0:
            raise Exception('At least one population size must be provided')

        # add 0 if no times are specified
        if len(times) < len(pop_sizes):
            times = [0] + list(times)

        if len(times) != len(pop_sizes):
            raise Exception('The specified number of times population size change occurs'
                            'and the number of population sizes does not match.')

        self.pop_sizes: np.ndarray = np.array(pop_sizes)
        self.times: np.ndarray = np.array(times)


class PhaseTypeDistribution:
    pass


class CoalescentDistribution(PhaseTypeDistribution):
    """
    """

    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray | List | mp.matrix = None,
            r: np.ndarray | List = None,
            Ne: float | int = 1
    ):

        # coalescent model
        self.r = None
        self.alpha = None
        self.model = model

        # sample size
        self.n = n

        self.set_alpha(alpha)

        self.set_reward(r)

        # effective population size
        self.Ne = Ne

        # obtain sub-intensity matrix.
        S = mp.matrix(self.get_rate_matrix(self.n, self.model))

        # apply reward
        self.S = mp.inverse(mp.diag(self.r)) * S

        # vector of ones
        self.e = mp.matrix(np.ones(n - 1))

        # exit rate vector
        self.s = -self.S * self.e

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
        # TODO can probably optimized by using self.T
        :return:
        """
        return mp.expm(self.S_full)

    @cached_property
    def t(self) -> mp.matrix:
        """
        The exit probability vector.
        :return:
        :rtype:
        """
        return 1 - self.T * self.e

    @cached_property
    def U(self) -> mp.matrix:
        """
        The Green matrix
        :return:
        """
        return -mp.inverse(self.S)

    @cached_property
    def T_inv_full(self) -> mp.matrix:
        """
        Inverse of full transition matrix
        :return:
        """
        return mp.inverse(self.T_full)

    @cached_property
    def T_inv(self) -> mp.matrix:
        """
        Inverse of transition matrix
        :return:
        """
        return mp.inverse(self.T)

    @cached_property
    def S_full(self) -> mp.matrix:
        """
        Full intensity matrix
        :return:
        """
        upper = np.concatenate([to_numpy(self.S), to_numpy(self.s)], axis=1)
        lower = np.zeros((1, self.n))

        return mp.matrix(np.concatenate([upper, lower], axis=0).tolist())

    @cached_property
    def default_reward(self) -> np.ndarray:
        """
        The default reward, which is also used for
        the moments of the tree height.
        :return:
        :rtype:
        """
        return np.ones(self.n - 1)

    @cached_property
    def total_branch_length_reward(self) -> np.ndarray:
        """
        Reward used to compute the total branch length.
        :return:
        :rtype:
        """
        return np.arange(2, self.n + 1)[::-1]

    def set_alpha(self, alpha: np.ndarray | List | mp.matrix = None):
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
            self.alpha = self.e_i(self.n, 0)
        elif isinstance(alpha, (np.ndarray, list)):
            self.alpha = mp.matrix(alpha)
        else:
            # assume we have instance of mp.matrix
            self.alpha = alpha

    def set_reward(self, r: np.ndarray | List = None):
        """
        Set the initial state.
        :return:
        :rtype:
        """
        if r is not None and len(r) != self.n - 1:
            raise Exception(f"r should be of length n - 1 = {self.n - 1} but has instead length {len(r)}.")

        # initial conditions
        if r is None:
            self.r = self.default_reward
        else:
            self.r = np.array(r)

    def set_Ne(self, Ne: int | float):
        """
        Change the effective population size
        :return:
        """
        # there is nothing we need to do here as all results
        # are rescaled by Ne
        self.Ne = Ne

    def nth_moment(self, k: int, alpha: np.ndarray = None) -> float:
        """
        Get the nth moment.
        :param k:
        :param alpha:
        :return:
        """
        if alpha is None:
            alpha = self.alpha
        else:
            alpha = mp.matrix(alpha)

        # self.Ne ** k is the rescaling due to population size
        return float(self.Ne ** k * factorial(k) * (alpha * self.U ** k * self.e)[0, 0])

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

    def sfs(self, theta: float = 1.0, alpha: np.ndarray = None) -> np.ndarray:
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
        if alpha is None:
            alpha = self.alpha
        else:
            alpha = mp.matrix(alpha)

        lam = theta / 2

        # compute resolvent
        I = mp.eye(self.n - 1)
        P = mp.inverse(I - float(1 / lam) * mp.inverse(mp.diag(self.total_branch_length_reward)) * self.S)
        p = self.e - P * self.e

        sfs = np.zeros(self.n + 1)
        P_i = I

        # iterate through number of segregating sites.
        # TODO for all entries to sum up to 1 we need n -> inf
        # TODO can we simply normalize?
        for i in range(self.n):
            sfs[i] = (alpha[:-1].T * P_i * p)[0, 0]
            P_i *= P

        return sfs

    def F(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.
        :param t:
        :return:
        """

        def F(t: float) -> float:
            return 1 - (self.alpha * fractional_matrix_power(self.T, self.Ne * t) * self.e)[0]

        return np.vectorize(F)(t)

    def f(self, u) -> float | np.ndarray:
        """
        Density function.
        :param u:
        :return:
        """

        def f(u: float) -> float:
            return self.alpha * fractional_matrix_power(self.T, self.Ne * u) * self.s

        return np.vectorize(f)(u)

    @staticmethod
    def clear_show_save(func: Callable) -> Callable:
        """
        Decorator for clearing current figure in the beginning
        and showing or saving produced plot subsequently.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # clear current figure
            plt.clf()

            # execute function
            func(*args, **kwargs)

            # show or save
            return CoalescentDistribution.show_and_save(
                file=kwargs['file'] if 'file' in kwargs else None,
                show=kwargs['show'] if 'show' in kwargs else None
            )

        return wrapper

    @staticmethod
    def show_and_save(file: str = None, show=True) -> plt.axis:
        """
        Show and save plot.
        :param file:
        :type file:
        :param show:
        :return:
        """
        # save figure if file path given
        if file is not None:
            plt.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0.1)

        # show figure if specified
        if show:
            plt.show()

        # return axis
        return plt.gca()

    @clear_show_save
    def plot_F(self, t_min: float = 0, t_max: float = 10, show=True, file: str = None) -> plt.axis:
        """
        Plot cumulative distribution function.
        :return:
        """
        t = np.linspace(t_min, t_max, 100)
        sns.lineplot(x=t, y=self.F(t))

        # set axis labels
        plt.xlabel('t')
        plt.ylabel('F(t)')

    @clear_show_save
    def plot_f(self, u_min: float = 0, u_max: float = 10, show=True, file: str = None) -> plt.axis:
        """
        Plot density function.
        :return:
        """
        t = np.linspace(u_min, u_max, 100)
        sns.lineplot(x=t, y=self.f(t))

        # set axis labels
        plt.xlabel('u')
        plt.ylabel('f(u)')

    @staticmethod
    def e_i(n: int, i: int = 0) -> np.matrix:
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


def fractional_matrix_power(m: mp.matrix, p: float) -> mp.matrix:
    """
    Fractional power of mpmath matrix using exponentials.
    TODO this produces complex value with small negative parts
    :param m:
    :type m:
    :param p:
    :type p:
    :return:
    :rtype:
    """
    return mp.expm(float(p) * mp.logm(m)).apply(mp.re)


class VariablePopulationSizeCoalescentDistribution(CoalescentDistribution):
    def __init__(
            self,
            n: int = 2,
            model: CoalescentModel = StandardCoalescent(),
            alpha: np.ndarray = None,
            r: np.ndarray = None,
            demography: Demography = None
    ):
        super().__init__(
            model=model,
            n=n,
            alpha=alpha,
            r=r,
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
        # of transitioning from state a to alpha to beta to b.
        # TODO this is numerically very unstable.
        B = self.T_inv_full * self.T_full[:, j] * self.T_full[i, :] * self.T_inv_full
        # B2 = expm(-self.S_full + self.S_full[:, [j]] + self.S_full[[i], :]) - self.S_full

        # construct matrix consisting of B and the rate matrices
        Q = self.S_full
        O = mp.zeros(self.n, self.n)

        upper = mp.matrix(Q.T.tolist() + B.T.tolist()).T
        lower = mp.matrix(O.T.tolist() + Q.T.tolist()).T

        A = mp.matrix(upper.tolist() + lower.tolist())

        # compute matrix exponential of A to determine integral
        return mp.expm(float(tau) * A)[:self.n, self.n:]

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
        # TODO this is numerically very unstable.
        # TODO check if B1 and B2 are correct
        B1 = self.T_inv_full * self.T_full[:, j] * self.T_full[i, :] * self.T_inv_full
        B2 = self.T_inv_full * self.T_full[:, l] * self.T_full[k, :] * self.T_inv_full

        # construct matrix consisting of B and the rate matrices
        Q = self.S_full
        O = mp.zeros(self.n, self.n)
        upper = mp.matrix(Q.T.tolist() + B1.T.tolist() + O.T.tolist()).T
        middle = mp.matrix(O.T.tolist() + Q.T.tolist() + B2.T.tolist()).T
        lower = mp.matrix(O.T.tolist() + O.T.tolist() + Q.T.tolist()).T

        A = mp.matrix(upper.tolist() + middle.tolist() + lower.tolist())

        # compute matrix exponential of A to determine integral
        var = mp.expm(float(tau) * A)[:self.n, 2 * self.n:]

        return var

    def get_mean_sojourn_times(self, i: int, tau: float) -> mp.matrix:
        """
        Get the endpoint-conditioned amount of time spent in state i.
        U[a, b] describes the amount of time spent in state i,
        given that the state was ´a´ at time 0 and ´b´ at time tau.
        :param i:
        :param tau:
        :return:
        """
        # obtain matrix G using Van Loan's method
        G = self.get_I_mean(i, i, tau / self.Ne)

        # normalize by transition probabilities.
        U = self.normalize_by_T_tau(G, tau)

        # scale by Ne
        mean = float(self.Ne) * U

        return mean

    def get_covarying_sojourn_times(self, i: int, j: int, tau: float) -> mp.matrix:
        """
        :param i:
        :param j:
        :param tau:
        :return:
        """
        # obtain matrix G using Van Loan's method
        G1 = self.get_I_var(i, i, j, j, tau / self.Ne)

        # obtain same matrix with states switched as in
        # https://doi.org/10.2202/1544-6115.1127
        G2 = self.get_I_var(j, j, i, i, tau / self.Ne)

        # add matrices
        G = G1 + G2

        # normalize by transition probabilities.
        U = self.normalize_by_T_tau(G, tau)

        # scale by Ne
        return float(self.Ne ** 2) * U

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
        T_tau = fractional_matrix_power(self.T_full, tau / self.Ne)

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
            return self.mean_and_var[0] if alpha is None else self.mean_and_var(np.matrix(alpha).T)[0]

        if k == 2:
            return self.mean_and_var[1] if alpha is None else self.mean_and_var(np.matrix(alpha).T)[1]

        raise NotImplementedError('Only the first second moments are implemented.')

    @cached_property
    def mean_and_var(self, alpha: np.ndarray = None) -> (float, float):
        """
        Calculate the first and second moments in the absorption time.

        We need the absorption probability in a certain epoch and
        the expected absorption time conditional on absorption in that epoch.
        We can get the expected absorption time by conditioning on the endpoint
        and determining the amount of time we spend in the absorbing state.
        We can get the absorption probability from the transition matrix.
        :return:
        """
        if alpha is None:
            alpha = mp.matrix(self.alpha).T
        else:
            alpha = mp.matrix(alpha).T

        # absorption times conditional on when the epoch ends
        absorption_times = np.zeros(self.n_epochs)

        # conditional second moments in absorption time
        absorption_m2 = np.zeros(self.n_epochs)

        # unconditional absorption probabilities
        absorption_probs = np.zeros(self.n_epochs)

        # Probability of not having reached the absorbing state until
        # the current epoch.
        no_absorption = np.zeros(self.n_epochs)

        # iterate over epochs
        for i in range(self.n_epochs):
            # set Ne of current epoch
            self.set_Ne(self.pop_sizes[i])

            # probability of not having reach the absorbing state until now
            no_absorption[i] = np.prod(1 - absorption_probs[:i])

            # we need to end-point condition all but the last epoch
            if i < self.n_epochs - 1:

                # determine tau, the amount of time spend with the
                # current population size
                tau = self.times[i + 1] - self.times[i]

                # sojourn time for absorbing state
                M = self.get_mean_sojourn_times(i=self.n - 1, tau=tau)

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
                absorption_times[i] = tau - np.dot(alpha, M[:-1, M.cols - 1])

                # second moment in sojourn time for absorbing state
                M2 = self.get_covarying_sojourn_times(i=self.n - 1, j=self.n - 1, tau=tau)

                # M2[:-1, -1] is the second moment in the time spent in the absorbing
                # state. We convert it here to the second moment in the absorption time
                # and multiply by the initial states as done for the mean.
                absorption_m2[i] = np.dot(alpha, M2[:-1, M.cols - 1]) + 2 * tau * absorption_times[i] - tau ** 2

                # Get probability of states at time tau.
                # These are the initial state probabilities for the next epoch.
                alpha = (alpha * fractional_matrix_power(self.T, tau / self.pop_sizes[i])).apply(mp.re)

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
        total_absorption_times = absorption_times + self.times

        # Here we adjust the second moment by the time spent in the
        # previous epochs
        total_absorption_m2 = absorption_m2 + self.times ** 2 + 2 * self.times * absorption_times

        # We finally get the unconditional moments by multiplying
        # with their total absorption probabilities and summing up
        mean = np.dot(total_absorption_probs, total_absorption_times)
        m2 = np.dot(total_absorption_probs, total_absorption_m2)

        return mean, m2
