"""
Test matrix exponentiation.
"""

from testing import TestCase

import numpy as np
import pytest

import phasegen as pg


class ExpmTestCase(TestCase):
    """
    Test matrix exponentiation.
    """

    def test_scipy_backend_precision(self):
        """
        The SciPy backend accepts NumPy floating types and their names, and applies the precision to both the dense
        exponential and its action. The annotated literal ``'np.float32'`` made every dense exponential raise, and the
        action ignored the precision.
        """
        import scipy.sparse as sp

        for precision in (np.float32, 'float32'):
            backend = pg.SciPyExpmBackend(precision=precision)
            self.assertEqual(backend.compute(np.eye(2)).dtype, np.float32)
            self.assertEqual(backend.compute_action(sp.csr_matrix(np.eye(2)), np.ones(2)).dtype, np.float32)

        np.testing.assert_allclose(pg.SciPyExpmBackend().compute_action(sp.csr_matrix(np.eye(2)), np.ones(2)), np.e)

        for precision in ('np.float32', int):
            with self.assertRaisesRegex(TypeError, "floating-point type"):
                pg.SciPyExpmBackend(precision=precision)
