"""
Test matrix exponentiation.
"""
from unittest import TestCase

import numpy as np

import phasegen as pg


class ExpmTestCase(TestCase):
    """
    Test matrix exponentiation.
    """

    def test_expm_different_backends(self):
        """
        Test matrix exponential for medium-sized matrix.
        """

        coal = pg.Coalescent(n=10)

        A = pg.TensorFlowExpmBackend().compute(coal.block_counting_state_space.S)
        B = pg.SciPyExpmBackend().compute(coal.block_counting_state_space.S)
        C = pg.JaxExpmBackend().compute(coal.block_counting_state_space.S)
        D = pg.expm.PyTorchExpmBackend().compute(coal.block_counting_state_space.S)

        np.testing.assert_array_almost_equal(A, B)
        np.testing.assert_array_almost_equal(A, C)
        np.testing.assert_array_almost_equal(A, D)

        pass
