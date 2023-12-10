from math import factorial
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, expon, rv_continuous

from phasegen.distributions import _HermiteExpansion


class DistributionTestCase(TestCase):
    """
    Test distributions.
    """

    @staticmethod
    def test_expand_exponential():
        """
        Test expanding an exponential distribution.
        """
        x = np.linspace(0, 10, 100)
        mu = 1
        sigma = 10
        dist: rv_continuous = expon
        n_moments = 100

        # get moments
        moments = np.array([dist.moment(k) for k in range(n_moments)])

        # get y-values
        y_approx = _HermiteExpansion.pdf(x, moments, mu=mu, sigma=sigma)
        y_ref = norm.pdf(x, loc=mu, scale=sigma)
        y_exact = dist.pdf(x)

        # plot data
        plt.plot(x, y_approx, label='approx')
        plt.plot(x, y_ref, label='ref')
        plt.plot(x, y_exact, label='exact')

        plt.legend()
        plt.show()

        pass

    @staticmethod
    def test_expand_norm():
        """
        Test expanding a normal distribution.
        """
        x = np.linspace(0, 10, 100)
        mu = 1
        sigma = 10
        dist: rv_continuous = norm(loc=2, scale=10)
        n_moments = 60

        # get moments
        moments = np.array([dist.moment(k) for k in range(1, n_moments + 1)])

        # get y-values
        y_approx = _HermiteExpansion.pdf(x, moments, mu=mu, sigma=sigma)
        y_ref = norm.pdf(x, loc=mu, scale=sigma)
        y_exact = dist.pdf(x)

        # plot data
        plt.plot(x, y_approx, label='approx', linestyle='--')
        plt.plot(x, y_ref, label='ref', linestyle='--')
        plt.plot(x, y_exact, label='exact', linestyle='dotted')

        plt.legend()
        plt.show()

        pass
