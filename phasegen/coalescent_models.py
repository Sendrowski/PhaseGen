from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import sympy as sp
from scipy.special import comb, beta


class CoalescentModel(ABC):
    """
    Abstract class for coalescent models.
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
