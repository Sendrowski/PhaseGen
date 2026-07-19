"""
Poisson log-likelihood utilities, vendored from fastDFE to avoid the dependency.
"""

import numpy as np
from scipy.special import factorial


class Likelihood:
    """
    Utilities for computing Poisson likelihoods.
    """

    #: Epsilon for numerical stability
    eps = 1e-50

    @staticmethod
    def add_epsilon(x: np.ndarray) -> np.ndarray:
        """
        Add epsilon to zero counts.

        :param x: Array to add epsilon to
        :return: Array with epsilon added to zero counts
        """
        x = x.astype(float)

        # replace 0s with epsilon to avoid log(0)
        x[x == 0] = Likelihood.eps

        return x

    @staticmethod
    def log_poisson(mu: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Compute log(Poisson(mu, k)).

        :param mu: Mean of Poisson distribution
        :param k: Number of events
        :return: log(Poisson(mu, k))
        """
        mu = Likelihood.add_epsilon(mu)

        return k * np.log(mu) - mu - Likelihood.log_factorial(k)

    @staticmethod
    def log_factorial_stirling(n: np.ndarray | float) -> np.ndarray | float:
        """
        Use Stirling's approximation for values larger than n_threshold.
        https://en.wikipedia.org/wiki/Stirling%27s_approximation

        :param n: n
        :return: log(n!)
        """
        return 0.5 * np.log(2 * np.pi * n) + n * np.log(n / np.e) + np.log(1 + 1 / (12 * n))

    @staticmethod
    def log_factorial(n: np.ndarray, n_threshold: int = 100) -> np.ndarray:
        """
        Compute log(n!).

        :param n: n
        :param n_threshold: Threshold for using Stirling's approximation
        :return: log(n!)
        """
        x = np.zeros_like(n, dtype=np.float64)

        low = n <= n_threshold

        # exact for small n, Stirling's approximation for large n
        x[low] = np.log(factorial(n[low]))
        x[~low] = Likelihood.log_factorial_stirling(n[~low])

        return x
