"""
Simulate moments of given population scenario using phase-type theory.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-02-25"

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp
import rewards
from typing import Callable, Union
from scipy.linalg import expm
from numpy.linalg import inv, matrix_power
from math import factorial
import JSON

try:
    testing = False
    n = snakemake.params.n
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n = 10  # sample size
    out = "scratch/ph.json"


class CoalescentModel(ABC):
    @abstractmethod
    def get_rate(self, i: int, j: int):
        pass


class StandardCoalscent(CoalescentModel):
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
        integrant = x ** (i - 2) * (1 - x) ** (j - i)

        integral = sp.Integral(integrant * self.get_density()(x), (x, 0, 1))
        return float(integral.doit())


class PhaseTypeDistribution:
    pass


class CoalescentDistribution(PhaseTypeDistribution):

    def __init__(self, model: CoalescentModel, n: int, alpha: np.ndarray = None, r: np.ndarray = None):
        # initial conditions
        self.alpha = alpha

        # coalescent model
        self.model = model

        # sample size
        self.n = n

        if r is None:
            r = rewards.Default()

        if isinstance(r, rewards.Reward):
            r = r.get_reward(n)

        # the reward
        self.r = r

        def matrix_indices_to_rates(i: int, j: int):
            return model.get_rate(n - i, j + 1 - i)

        # define rate matrix
        self.P = np.fromfunction(np.vectorize(matrix_indices_to_rates), (n - 1, n - 1), dtype=float)

        # apply reward
        self.P = np.diag(1 / self.r) @ self.P

        # calculate Green matrix
        self.U = -inv(self.P)

    def nth_moment(self, k: int) -> float:
        """
        Get the nth moment.
        :param k:
        :param P:
        :param alpha:
        :return:
        """
        return factorial(k) * (self.alpha @ matrix_power(self.U, k) @ np.ones(n - 1))[0]

    def mean(self) -> float:
        """
        Get the mean.
        :param P:
        :param alpha:
        :return:
        """
        return self.nth_moment(1)

    def var(self) -> float:
        """
        Get the variance
        :param P:
        :param alpha:
        :return:
        """
        return self.nth_moment(2) - self.nth_moment(1) ** 2

    def set_reward(self, r: Union[rewards.Reward, np.ndarray]) -> 'CoalescentDistribution':
        """
        Change the reward.
        :param r:
        :return:
        """
        if isinstance(r, rewards.Reward):
            r = r.get_reward(self.n)

        return CoalescentDistribution(model=self.model, n=self.n, alpha=self.alpha, r=r)


alpha = np.eye(1, n - 1, 0)
cd = CoalescentDistribution(StandardCoalscent(), n=n, alpha=alpha)

height = dict(
    mu=cd.mean(),
    var=cd.var()
)

cd = cd.set_reward(rewards.TotalBranchLength())

total_branch_length = dict(
    mu=cd.mean(),
    var=cd.var()
)

JSON.save(dict((k, globals()[k]) for k in ['n', 'height', 'total_branch_length']), out)
