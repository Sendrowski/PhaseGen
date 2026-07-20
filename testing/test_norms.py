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

    def test_L2Norm_2d_is_elementwise_not_spectral(self):
        """
        Regression test for bug #13: on 2-D input the L2 norm must be the element-wise
        Euclidean distance sqrt(sum((a-b)**2)) (flattened vector norm), not the matrix
        (spectral / largest-singular-value) norm np.linalg.norm(a-b).
        """
        a = np.array([[3., 0., 0.], [0., 4., 0.]])
        b = np.zeros((2, 3))

        # element-wise Euclidean distance: sqrt(3**2 + 4**2) = 5.0
        expected_elementwise = np.sqrt(np.sum((a - b) ** 2))
        self.assertAlmostEqual(expected_elementwise, 5.0, places=12)

        # pre-fix behaviour was np.linalg.norm(a - b) == 4.0 (the largest singular value)
        matrix_norm = np.linalg.norm(a - b, ord=2)
        self.assertAlmostEqual(matrix_norm, 4.0, places=12)
        self.assertNotAlmostEqual(matrix_norm, expected_elementwise, places=6)

        self.assertAlmostEqual(pg.L2Norm().compute(a, b), expected_elementwise, places=12)

    def test_L1Norm_2d_is_elementwise_not_max_column_sum(self):
        """
        Regression test for bug #13: on 2-D input the L1 norm must be the element-wise
        Manhattan distance sum(|a-b|) (flattened vector norm), not the matrix 1-norm
        (max column sum).
        """
        a = np.ones((2, 2))
        b = np.zeros((2, 2))

        # element-wise Manhattan distance: sum of four ones = 4.0
        expected_elementwise = np.sum(np.abs(a - b))
        self.assertAlmostEqual(expected_elementwise, 4.0, places=12)

        # pre-fix behaviour was np.linalg.norm(a - b, ord=1) == 2.0 (max column sum)
        matrix_norm = np.linalg.norm(a - b, ord=1)
        self.assertAlmostEqual(matrix_norm, 2.0, places=12)
        self.assertNotAlmostEqual(matrix_norm, expected_elementwise, places=6)

        self.assertAlmostEqual(pg.L1Norm().compute(a, b), expected_elementwise, places=12)

    def test_LInfNorm_2d_is_elementwise_not_max_row_sum(self):
        """
        Regression test for bug #13: on 2-D input the L-infinity norm must be the
        element-wise Chebyshev distance max(|a-b|) (flattened vector norm), not the
        matrix inf-norm (max row sum).
        """
        a = np.ones((2, 2))
        b = np.zeros((2, 2))

        # element-wise Chebyshev distance: max abs element = 1.0
        expected_elementwise = np.max(np.abs(a - b))
        self.assertAlmostEqual(expected_elementwise, 1.0, places=12)

        # pre-fix behaviour was np.linalg.norm(a - b, ord=np.inf) == 2.0 (max row sum)
        matrix_norm = np.linalg.norm(a - b, ord=np.inf)
        self.assertAlmostEqual(matrix_norm, 2.0, places=12)
        self.assertNotAlmostEqual(matrix_norm, expected_elementwise, places=6)

        self.assertAlmostEqual(pg.LInfNorm().compute(a, b), expected_elementwise, places=12)

    def test_norms_scalar_and_1d_unchanged(self):
        """
        Regression test for bug #13: scalar and 1-D inputs must be unaffected by the
        flatten fix (1-D vector norm == matrix/vector norm of the same input).
        """
        a = np.array([1., 2., 3.])
        b = np.array([4., 5., 6.])

        self.assertAlmostEqual(pg.L2Norm().compute(a, b), np.linalg.norm(a - b, ord=2), places=12)
        self.assertAlmostEqual(pg.L1Norm().compute(a, b), np.linalg.norm(a - b, ord=1), places=12)
        self.assertAlmostEqual(pg.LInfNorm().compute(a, b), np.linalg.norm(a - b, ord=np.inf), places=12)

        # scalars
        self.assertAlmostEqual(pg.L2Norm().compute(3.0, 7.0), 4.0, places=12)
        self.assertAlmostEqual(pg.L1Norm().compute(3.0, 7.0), 4.0, places=12)
        self.assertAlmostEqual(pg.LInfNorm().compute(3.0, 7.0), 4.0, places=12)

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
        observed = pg.TwoSFS([[2, 3], [4, 5]])
        modelled = pg.TwoSFS([[2.5, 3.5], [4.5, 5.5]])

        expected_result = stats.poisson.logpmf(observed.data, modelled.data).sum()
        actual_result = pg.PoissonLikelihood().compute(observed, modelled)

        self.assertAlmostEqual(-expected_result, actual_result, places=7)

    def test_multinomial_likelihood(self):
        """
        Test the Multinomial likelihood.
        """
        observed = [10, 20, 30]
        modelled = [0.2, 0.3, 0.5]

        # Manually compute expected value
        expected_result = -np.sum(np.array(observed) * np.log(modelled))
        actual_result = pg.MultinomialLikelihood().compute(observed, modelled)

        self.assertAlmostEqual(expected_result, actual_result, places=7)

    def test_multinomial_likelihood_unnormalized(self):
        """
        Test Multinomial likelihood with unnormalized modelled values.
        """
        observed = [5, 15, 30]
        modelled = [2, 3, 5]  # Not normalized

        # Normalize modelled
        modelled = np.array(modelled) / sum(modelled)
        expected_result = -np.sum(np.array(observed) * np.log(modelled))
        actual_result = pg.MultinomialLikelihood().compute(observed, [2, 3, 5])

        self.assertAlmostEqual(expected_result, actual_result, places=7)
