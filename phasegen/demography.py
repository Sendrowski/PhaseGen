from abc import abstractmethod
from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt


class Demography:
    """
    Base class for demographic scenarios.
    """

    def get_rate(self, t: float) -> float:
        """
        Get the coalescence rate.

        :param t: Time at which to get the rate.
        :return: Coalescence rate.
        """
        pass

    @abstractmethod
    def get_cum_rate(self, t: float) -> float:
        """
        Get the cumulative coalescence rate.

        :param t: Time at which to get the cumulative rate.
        :return: Cumulative coalescence rate.
        """
        pass


class ConstantDemography(Demography):
    """
    Demographic scenario where the population size is constant.
    """

    def __init__(self, pop_size: float):
        """
        Create a constant demographic scenario.

        :param pop_size: Population size.
        """
        if pop_size <= 0:
            raise Exception('Population size must be greater than zero.')

        self.pop_size: float = pop_size

    def get_rate(self, t: float) -> float:
        """
        Get the coalescence rate.

        :param t: Time at which to get the rate.
        :return: Coalescence rate.
        """
        return 1 / self.pop_size

    def get_cum_rate(self, t: float) -> float:
        """
        Get the cumulative coalescence rate.

        :param t: Time at which to get the cumulative rate.
        :return: Cumulative coalescence rate.
        """
        return t / self.pop_size

    def plot(self, show: bool = True) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :return: Axes object.
        """
        plt.plot([0, 1], [self.pop_size, self.pop_size])

        return self.finalize_plot(show=show)

    @staticmethod
    def finalize_plot(show: bool = True) -> plt.Axes:
        """
        Finalize the plot.

        :param show: Whether to show the plot.
        :return: Axes object.
        """
        ax = plt.gca()

        ax.set_xlabel('t')
        ax.set_ylabel('N(t)')

        if show:
            plt.show()

        return ax


class PiecewiseConstantDemography(ConstantDemography):
    """
    Piecewise constant demographic scenario.
    """

    def __init__(self, pop_sizes: np.ndarray | List, times: np.ndarray | List):
        """
        The population sizes and times these changes occur backwards in time.
        We need to start with a population size at time 0 but this time
        can be omitted in which case len(pop_size) == len(times) + 1.

        :param pop_sizes: Population sizes.
        :param times: Times at which the population sizes change.
        """
        if (np.array(pop_sizes) <= 0).any():
            raise Exception('All population sizes must be greater than zero.')

        super().__init__(pop_size=pop_sizes[0])

        if len(pop_sizes) == 0:
            raise Exception('At least one population size must be provided')

        # add 0 if no times are specified
        if len(times) < len(pop_sizes):
            times = [0] + list(times)

        if len(times) != len(pop_sizes):
            raise Exception('Number of population sizes and times must match.')

        self.pop_sizes: np.ndarray = np.array(pop_sizes)
        self.times: np.ndarray = np.array(times)

        self.tau = self.times[1:] - self.times[:-1]

    def get_rate(self, t: float) -> float:
        """
        Get the coalescence rate.

        :param t: Time at which to get the rate.
        :return: Coalescence rate.
        """
        # obtain index of previous epoch
        i = np.sum(self.times <= t) - 1

        # return rate
        return 1 / self.pop_sizes[i]

    def get_cum_rate(self, t: float) -> float:
        """
        Get the cumulative coalescence rate.

        :param t: Time at which to get the cumulative rate.
        :return: Cumulative coalescence rate.
        """
        # obtain index of previous epoch
        i = np.sum(self.times <= t) - 1

        if i <= 0:
            return super().get_cum_rate(t)

        # return cumulative rate
        return np.dot(self.tau[:i], 1 / self.pop_sizes[:i]) + (t - self.times[i]) / self.pop_sizes[i]

    def plot(self, show: bool = True) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :return: Axes object.
        """
        plt.step(self.times, self.pop_sizes, where='post')

        return self.finalize_plot(show=show)


class ContinuousDemography(PiecewiseConstantDemography):
    """
    Continuous demographic scenario (which is discretized).
    """

    def __init__(self, trajectory: Callable[[float], float]):
        """
        Create a continuous demographic scenario.
        TODO determine dynamically based on the transition probabilities.

        :param trajectory: Function that returns the population size at a given time.
        """
        #: Trajectory function
        self.trajectory = np.vectorize(trajectory)

        start_time = 0
        end_time = 5
        n_points = 100000
        n_epochs = 100

        # compute the cumulative sum and normalize it to be between 0 and 1
        x = np.linspace(start_time, end_time, n_points)
        y = self.trajectory(x)

        # generate the discretized points
        population_changes = np.linspace(y[0], y[-1], n_epochs)
        indices = np.array([(np.abs(y - p)).argmin() for p in population_changes])

        pop_sizes = y[indices]
        times = x[indices]

        super().__init__(pop_sizes=pop_sizes, times=times)

    def plot(self, show: bool = True) -> plt.Axes:
        """
        Plot the population size over time.

        :param show: Whether to show the plot.
        :return: Axes object.
        """
        plt.plot(self.times, self.trajectory(self.times), label='original')
        plt.step(self.times, self.pop_sizes, where='post', label='discretized')
        plt.legend()

        return self.finalize_plot(show=show)


class ExponentialDemography(ContinuousDemography):
    """
    Demographic scenario where the population size grows exponentially.
    """

    def __init__(self, growth_rate: float, N0: float = 1):
        """
        :param growth_rate: Exponential growth rate so that at time ``t`` in the past we have
            ``N0 * exp(- growth_rate * t)``.
        :param N0: Initial population size (only used if growth_rate is specified).
        """
        super().__init__(trajectory=lambda t: N0 * np.exp(- growth_rate * t))
