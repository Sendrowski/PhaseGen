from abc import ABC, abstractmethod

import numpy as np


class Reward(ABC):
    @abstractmethod
    def get_reward(self, n: int):
        pass


class Default(Reward):
    def get_reward(self, n: int):
        return np.ones(n - 1)


class TotalBranchLength(Reward):
    def get_reward(self, n: int):
        return np.arange(2, n + 1)[::-1]
