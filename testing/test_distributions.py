import itertools
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, expon, rv_continuous
from statsmodels.distributions.edgeworth import cumulant_from_moments, ExpandedNormal
from statsmodels.stats.moment_helpers import mnc2cum

import phasegen as pg
from phasegen.distributions import _EdgeworthExpansion
from phasegen.distributions import _GramCharlierExpansion


class DistributionTestCase(TestCase):
    """
    Test distributions.
    """

    @staticmethod
    def get_test_coalescent() -> pg.Coalescent:
        """
        Get a test coalescent.
        """
        return pg.Coalescent(
            demography=pg.PiecewiseConstantDemography(
                pop_sizes=dict(
                    pop_0={0: 1, 0.2: 5},
                    pop_1={0: 0.4, 0.1: 3, 0.25: 0.3},
                    pop_2={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 0.1},
                    ('pop_1', 'pop_2'): {0: 0.2, 0.1: 0.3},
                    ('pop_2', 'pop_0'): {0: 0.4, 0.1: 0.5, 0.2: 0.6},
                    ('pop_0', 'pop_2'): {0: 0.7, 0.1: 0.8, 0.2: 0.9, 0.3: 1},
                    ('pop_2', 'pop_1'): {0: 0.1}
                }
            ),
            n=pg.PopConfig(dict(
                pop_0=1,
                pop_1=2,
                pop_2=3
            ))
        )

    def test_quantile_raises_error_below_0(self):
        """
        Test quantile function raises error when quantile is below 0.
        """
        with self.assertRaises(ValueError) as context:
            self.get_test_coalescent().tree_height.quantile(-0.1)

        self.assertEqual(str(context.exception), 'Quantile must be between 0 and 1.')

    def test_quantile_raises_error_above_1(self):
        """
        Test quantile function raises error when quantile is above 1.
        """
        with self.assertRaises(ValueError) as context:
            self.get_test_coalescent().tree_height.quantile(1.1)

        self.assertEqual(str(context.exception), 'Quantile must be between 0 and 1.')

    def test_quantile(self):
        """
        Test quantile function.
        """
        dist = self.get_test_coalescent().tree_height

        for (quantile, tol) in itertools.product([0, 0.01, 0.5, 0.99, 1], [1e-1, 1e-5, 1e-10]):
            self.assertAlmostEqual(dist.cdf(dist.quantile(quantile, tol=tol)), quantile, delta=tol)

    @staticmethod
    def test_expand_exponential_gram_charlier():
        """
        Test expanding an exponential distribution.
        """
        x = np.linspace(-10, 10, 100)
        mu = 1
        sigma = 10
        dist: rv_continuous = expon
        n_moments = 3

        # get moments
        moments = np.array([dist.moment(k) for k in range(n_moments)])

        # get y-values
        y_approx = _GramCharlierExpansion.pdf(x, moments, mu=mu, sigma=sigma)
        y_ref = norm.pdf(x, loc=mu, scale=sigma)
        y_exact = dist.pdf(x)

        # plot data
        plt.plot(x, y_approx, label='approx', alpha=0.5)
        plt.plot(x, y_ref, label='ref', alpha=0.5)
        plt.plot(x, y_exact, label='truth', alpha=0.5)

        plt.legend()
        plt.show()

        pass

    @staticmethod
    def test_expand_norm_gram_charlier():
        """
        Test expanding a normal distribution.
        """
        x = np.linspace(-10, 20, 100)
        mu = 1
        sigma = 5
        dist: rv_continuous = norm(loc=7, scale=5)
        n_moments = 10

        # get moments
        moments = np.array([dist.moment(k) for k in range(1, n_moments + 1)])

        # get y-values
        y_approx = _GramCharlierExpansion.pdf(x, moments, mu=mu, sigma=sigma)
        y_ref = norm.pdf(x, loc=mu, scale=sigma)
        y_exact = dist.pdf(x)

        # plot data
        plt.plot(x, y_approx, label='approx', linestyle='-')
        plt.plot(x, y_ref, label='ref', linestyle='--')
        plt.plot(x, y_exact, label='truth', linestyle='dotted')

        plt.legend()
        plt.show()

        pass

    @staticmethod
    def test_expand_exponential_compare_with_statsmodels():
        """
        Test expanding an exponential distribution.
        """
        x = np.linspace(-10, 10, 100)
        dist: rv_continuous = expon
        n_moments = 4  # higher moments not implemented by statsmodels

        # get moments
        moments = np.array([dist.moment(k) for k in range(1, n_moments + 1)])

        cumulants = _EdgeworthExpansion.cumulants_from_moments(moments)

        ee = _EdgeworthExpansion(cumulants)
        ee2 = ExpandedNormal(cumulants)

        # get y-values
        y_approx = ee._pdf(x)
        y_approx2 = ee2.pdf(x)

        np.testing.assert_almost_equal(y_approx, y_approx2)

    @staticmethod
    def test_expand_exponential_edgeworth_pdf():
        """
        Test expanding an exponential distribution.
        """
        x = np.linspace(0, 3, 100)
        dist: rv_continuous = expon
        n_moments = 12

        # get moments
        moments = np.array([dist.moment(k) for k in range(1, n_moments + 1)])

        cumulants = _EdgeworthExpansion.cumulants_from_moments(moments)

        ee = _EdgeworthExpansion(cumulants)

        # get y-values
        y_approx = ee._pdf(x)
        y_ref = norm.pdf(x, loc=moments[0], scale=np.sqrt(moments[1]) - moments[0])
        y_exact = dist.pdf(x)

        # plot data
        plt.plot(x, y_approx, label='approx', alpha=0.5)
        plt.plot(x, y_ref, label='ref', alpha=0.5)
        plt.plot(x, y_exact, label='truth', alpha=0.5)

        plt.legend()
        plt.show()

        pass

    @staticmethod
    def test_expand_exponential_edgeworth_cdf():
        """
        Test expanding an exponential distribution.
        """
        x = np.linspace(0, 3, 100)
        dist: rv_continuous = expon
        n_moments = 12

        # get moments
        moments = np.array([dist.moment(k) for k in range(1, n_moments + 1)])

        cumulants = _EdgeworthExpansion.cumulants_from_moments(moments)

        ee = _EdgeworthExpansion(cumulants)

        # get y-values
        y_approx = ee._cdf(x)
        y_ref = norm.cdf(x, loc=moments[0], scale=np.sqrt(moments[1]) - moments[0])
        y_exact = dist.cdf(x)

        # plot data
        plt.plot(x, y_approx, label='approx', alpha=0.5)
        plt.plot(x, y_ref, label='ref', alpha=0.5)
        plt.plot(x, y_exact, label='truth', alpha=0.5)

        plt.legend()
        plt.show()

        pass

    def test_generate_partitions(self):
        """
        Test generating partitions.
        """
        partitions = [list(_EdgeworthExpansion._generate_partitions(n)) for n in range(1, 5)]

        expected = [
            [[(1, 1)]],
            [[(1, 2)], [(2, 1)]],
            [[(1, 3)], [(1, 1), (2, 1)], [(3, 1)]],
            [[(1, 4)], [(1, 2), (2, 1)], [(2, 2)], [(1, 1), (3, 1)], [(4, 1)]]
        ]

        self.assertEqual(
            tuple(tuple(tuple(b) for b in a) for a in expected),
            tuple(tuple(tuple(b) for b in a) for a in partitions)
        )

    @staticmethod
    def test_cumulants_from_moments_normal():
        """
        Test cumulants from moments for a normal distribution.
        """
        dist = norm(loc=1, scale=2)

        moments = np.array([dist.moment(k) for k in range(1, 11)])

        cumulants = _EdgeworthExpansion.cumulants_from_moments(moments)

        np.testing.assert_almost_equal(
            cumulants,
            np.array([1, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
            decimal=7
        )

    @staticmethod
    def test_cumulants_from_moments_exponential():
        """
        Test cumulants from moments for an exponential distribution.
        """
        dist = expon
        n_moments = 4  # higher moments not implemented by statsmodels

        moments = np.array([dist.moment(k) for k in range(1, n_moments + 1)])

        cumulants = _EdgeworthExpansion.cumulants_from_moments(moments)

        cumulants_expected = [cumulant_from_moments(moments, k) for k in range(1, n_moments + 1)]

        cumulants_expected2 = mnc2cum(moments)

        np.testing.assert_array_equal(
            cumulants,
            cumulants_expected
        )
