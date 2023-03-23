from abc import ABC, abstractmethod
from functools import cached_property
from math import factorial
from typing import Callable, Union
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sympy as sp
from numpy.linalg import inv, matrix_power
from scipy.linalg import expm, fractional_matrix_power

import rewards


class CoalescentModel(ABC):
    @abstractmethod
    def get_rate(self, i: int, j: int):
        pass


class StandardCoalescent(CoalescentModel):
    def get_rate(self, i: int, j: int):
        if j == 2:
            return i * (i - 1) / 2

        if j == 1:
            return -i * (i - 1) / 2

        return 0


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
    TODO consider using scipy.sparse.linalg as matrices might be sparse
    """

    def __init__(
            self,
            model: CoalescentModel,
            n: int,
            alpha: np.ndarray | List = None,
            r: rewards.Reward | np.ndarray | List = None,
            Ne: float | int = 1
    ):
        if len(alpha) != n - 1:
            raise Exception(f"alpha should be of length n - 1 = {n - 1} but has instead length {len(alpha)}.")

        # initial conditions
        if alpha is None:
            self.alpha = self.e_i(0)
        else:
            self.alpha = np.array(alpha)

        # coalescent model
        self.model = model

        # sample size
        self.n = n

        if r is None:
            self.r = rewards.Default().get_reward(n)

        elif isinstance(r, rewards.Reward):
            self.r = r.get_reward(n)

        else:
            # the reward
            self.r = np.array(r)

        # effective population size
        self.Ne = Ne

        def matrix_indices_to_rates(i: int, j: int) -> float:
            return model.get_rate(n - i, j + 1 - i)

        # Define sub-intensity matrix.
        # Dividing by Ne here produces unstable results for small population
        # sizes (Ne < 1). We thus add it later to the moments.
        S = np.fromfunction(np.vectorize(matrix_indices_to_rates), (n - 1, n - 1))

        # apply reward
        self.S = np.diag(1 / self.r) @ S

        # vector of ones
        self.e = np.ones(n - 1)

        # exit vector
        self.s = -self.S @ self.e

    @cached_property
    def T(self) -> np.ndarray:
        """
        The transition matrix.
        :return:
        """
        return expm(self.S)

    @cached_property
    def T_full(self) -> np.ndarray:
        """
        The full transition matrix
        # TODO can probably optimized by using self.T
        :return:
        """
        return expm(self.S_full)

    @cached_property
    def U(self) -> np.ndarray:
        """
        The Green matrix
        :return:
        """
        return -inv(self.S)

    @cached_property
    def T_inv_full(self) -> np.ndarray:
        """
        Inverse of full transition matrix
        :return:
        """
        return inv(self.T_full)

    @cached_property
    def T_inv(self) -> np.ndarray:
        """
        Inverse of transition matrix
        :return:
        """
        return inv(self.T)

    @cached_property
    def S_full(self) -> np.ndarray:
        """
        Full intensity matrix
        :return:
        """
        return np.concatenate([np.concatenate([self.S, self.s[:, None]], axis=1), np.zeros((1, self.n))], axis=0)

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

        # self.Ne ** k is the rescaling due to population size
        return (self.Ne ** k) * factorial(k) * alpha @ matrix_power(self.U, k) @ self.e

    def mean(self, alpha: np.ndarray = None) -> float:
        """
        Get the mean absorption time.
        :param alpha:
        :return:
        """
        return self.nth_moment(1, alpha)

    def var(self, alpha: np.ndarray = None) -> float:
        """
        Get the variance in the absorption time.
        :param alpha:
        :return:
        """
        return self.nth_moment(2, alpha) - self.nth_moment(1, alpha) ** 2

    def set_reward(self, r: Union[rewards.Reward, np.ndarray, List]) -> 'CoalescentDistribution':
        """
        Change the reward.
        :param r:
        :return:
        """
        if isinstance(r, rewards.Reward):
            r = r.get_reward(self.n)

        # create a new instance
        # TODO this can be improved
        return CoalescentDistribution(
            model=self.model,
            n=self.n,
            alpha=self.alpha,
            r=r,
            Ne=self.Ne
        )

    def F(self, t) -> float | np.ndarray:
        """
        Cumulative distribution function.
        :param t:
        :return:
        """

        def F(t):
            return 1 - (self.alpha @ fractional_matrix_power(self.T, self.Ne * t) @ self.e[:, None])[0]

        return np.vectorize(F)(t)

    def f(self, u) -> float | np.ndarray:
        """
        Density function.
        :param u:
        :return:
        """

        def f(u):
            return self.alpha @ fractional_matrix_power(self.T, self.Ne * u) @ self.s

        return np.vectorize(f)(u)

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

        return self.show_and_save(file, show)

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

        return self.show_and_save(file, show)

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

    @staticmethod
    def e_i(n: int, i: int = 0) -> np.ndarray:
        """
        Get the nth standard unit vector
        :return:
        """
        return np.eye(1, n - 1, i)[0]


class VariablePopulationSizeCoalescentDistribution(CoalescentDistribution):
    def __init__(
            self,
            model: CoalescentModel,
            n: int,
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

    def get_I_mean(self, i, j, tau):
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
        B = self.T_inv_full @ self.T_full[:, [j]] @ self.T_full[[i], :] @ self.T_inv_full
        # B2 = expm(-self.S_full + self.S_full[:, [j]] + self.S_full[[i], :]) - self.S_full

        # construct matrix consisting of B and the rate matrices
        Q = self.S_full
        O = np.zeros((self.n, self.n))
        A = np.concatenate([np.concatenate([Q, B], axis=1),
                            np.concatenate([O, Q], axis=1)], axis=0)

        # compute matrix exponential of A to determine integral
        return fractional_matrix_power(expm(A), tau)[:self.n, self.n:]

    def get_I_var(self, i, j, k, l, tau):
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
        B1 = self.T_inv_full @ self.T_full[:, [j]] @ self.T_full[[i], :] @ self.T_inv_full
        B2 = self.T_inv_full @ self.T_full[:, [l]] @ self.T_full[[k], :] @ self.T_inv_full

        # construct matrix consisting of B and the rate matrices
        Q = self.S_full
        O = np.zeros((self.n, self.n))
        A = np.concatenate([np.concatenate([Q, B1, O], axis=1),
                            np.concatenate([O, Q, B2], axis=1),
                            np.concatenate([O, O, Q], axis=1)], axis=0)

        # compute matrix exponential of A to determine integral
        var = fractional_matrix_power(expm(A), tau)[:self.n, 2 * self.n:]

        return var

    def get_mean_sojourn_times(self, i: int, tau: float) -> np.ndarray:
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

        # get transition probabilities over time tau
        T_tau = fractional_matrix_power(self.T_full, tau / self.Ne)

        # for states where T_tau == 0, i.e. for impossible transitions,
        # we have a sojourn time of 0
        not_zero = T_tau != 0

        # initialize matrix
        U = np.zeros_like(G)

        # divide by transition probabilities to normalize G
        U[not_zero] = G[not_zero] / T_tau[not_zero]

        # scale by Ne
        mean = self.Ne * U

        return mean

    def get_covarying_sojourn_times(self, i: int, j: int, tau: float) -> np.ndarray:
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

        # get transition probabilities over time tau
        T_tau = fractional_matrix_power(self.T_full, tau / self.Ne)

        # for states where T_tau == 0, i.e. for impossible transitions,
        # we have a sojourn time of 0
        not_zero = T_tau != 0

        # initialize matrix
        U = np.zeros_like(G)

        # divide by transition probabilities to normalize G
        # as in https://doi.org/10.2202/1544-6115.1127
        U[not_zero] = G[not_zero] / T_tau[not_zero]

        # scale by Ne
        return self.Ne ** 2 * U

    @property
    def mean_and_var(self, **kwargs) -> (float, float):
        """
        Calculate the first and second moments in the absorption time.

        We need the absorption probability in a certain epoch and
        the expected absorption time conditional on absorption in that epoch.
        We can get the expected absorption time by conditioning on the endpoint
        and determining the amount of time we spend in the absorbing state.
        We can get the absorption probability from the transition matrix.
        :param **kwargs:
        :type **kwargs:
        :return:
        """

        # absorption times conditional on when the epoch ends
        absorption_times = np.zeros(self.n_epochs)

        # second moments in absorption time
        absorption_m2 = np.zeros(self.n_epochs)

        # unconditional absorption probabilities
        absorption_probs = np.zeros(self.n_epochs)

        # probability of not having reach the absorbing state until now
        no_absorption = np.zeros(self.n_epochs)

        # initial state of current epoch
        alpha = self.alpha

        # iterate over epochs
        for i in range(self.n_epochs):
            # set Ne of current epoch
            self.set_Ne(self.pop_sizes[i])

            # probability of not having reach the absorbing state until now
            no_absorption[i] = np.prod(1 - absorption_probs)

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
                absorption_times[i] = tau - np.dot(alpha, M[:-1, -1])

                # second moment in sojourn time for absorbing state
                M2 = self.get_covarying_sojourn_times(i=self.n - 1, j=self.n - 1, tau=tau)

                # M2[:-1, -1] is the second moment in the time spent in the absorbing
                # state. We convert it here to the second moment in the absorption time
                # and multiply by the initial states as done for the mean.
                absorption_m2[i] = np.dot(alpha, M2[:-1, -1]) + 2 * tau * absorption_times[i] - tau ** 2

                # Get probability of states at time tau.
                # These are the initial state probabilities for the next epoch.
                alpha = alpha @ fractional_matrix_power(self.T, tau / self.pop_sizes[i])

                # absorption probability in current state
                absorption_probs[i] = 1 - np.sum(alpha)

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

    @property
    def mean(self, **kwargs) -> float:
        return self.mean_and_var[0]

    @property
    def var(self, **kwargs) -> float:
        return self.mean_and_var[1] - self.mean ** 2

    def set_reward(self, r: Union[rewards.Reward, np.ndarray, List]) -> 'VariablePopulationSizeCoalescentDistribution':
        """
        Change the reward.
        :param r:
        :return:
        """
        if isinstance(r, rewards.Reward):
            r = r.get_reward(self.n)

        # create a new instance
        # TODO this can be improved
        return VariablePopulationSizeCoalescentDistribution(
            model=self.model,
            n=self.n,
            alpha=self.alpha,
            r=r,
            demography=self.demography
        )
