"""
Test norms.
"""

import unittest

import numpy as np
import scipy.stats as stats

import phasegen as pg


class NormTestCase(unittest.TestCase):
    """
    Test norms.
    """

    def test_L2Norm(self):
        """
        Test the L2 norm.
        """
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        expected_result = np.linalg.norm(a - b, ord=2)
        self.assertEqual(pg.L2Norm().compute(a, b), expected_result)

    def test_L1Norm(self):
        """
        Test the L1 norm.
        """
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        expected_result = np.linalg.norm(a - b, ord=1)
        self.assertEqual(pg.L1Norm().compute(a, b), expected_result)

    def test_LInfNorm(self):
        """
        Test the LInf norm.
        """
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        expected_result = np.linalg.norm(a - b, ord=np.inf)
        self.assertEqual(pg.LInfNorm().compute(a, b), expected_result)

    def test_poisson_likelihood(self):
        """
        Test the Poisson likelihood.
        """
        observed = [2, 3, 4, 5]
        modelled = [2.5, 3.5, 4.5, 5.5]

        expected_result = sum(stats.poisson.logpmf(observed, modelled))
        actual_result = pg.PoissonLikelihood().compute(observed, modelled)

        self.assertAlmostEqual(-expected_result, actual_result, places=7)

    def test_poisson_likelihood_pass_SFS(self):
        """
        Test the Poisson likelihood.
        """
        observed = pg.SFS([2, 3, 4, 5])
        modelled = pg.SFS([2.5, 3.5, 4.5, 5.5])

        expected_result = sum(stats.poisson.logpmf(observed.data, modelled.data))
        actual_result = pg.PoissonLikelihood().compute(observed, modelled)

        self.assertAlmostEqual(-expected_result, actual_result, places=7)

    def test_poisson_likelihood_pass_2SFS(self):
        """
        Test the Poisson likelihood.
        """
        observed = pg.SFS2([[2, 3], [4, 5]])
        modelled = pg.SFS2([[2.5, 3.5], [4.5, 5.5]])

        expected_result = stats.poisson.logpmf(observed.data, modelled.data).sum()
        actual_result = pg.PoissonLikelihood().compute(observed, modelled)

        self.assertAlmostEqual(-expected_result, actual_result, places=7)
