from abc import abstractmethod
from typing import List

import numpy as np


class Demography:

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
        :param pop_size: Population size.

        """
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


class PiecewiseConstantDemography(ConstantDemography):
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

        # return probability
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

        # return cumulative probability
        return np.dot(self.tau[:i], 1 / self.pop_sizes[:i]) + (t - self.times[i]) / self.pop_sizes[i]
