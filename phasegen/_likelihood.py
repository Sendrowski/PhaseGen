"""
Poisson log-likelihood utilities, vendored from fastDFE to avoid the dependency.
"""

import numpy as np
from scipy.special import gammaln


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
    def log_factorial(n: np.ndarray) -> np.ndarray:
        """
        Compute log(n!).

        :param n: n
        :return: log(n!)
        """
        return gammaln(np.asarray(n, dtype=np.float64) + 1)
