"""
Test matrix exponentiation.
"""
import importlib.util

from testing import TestCase

import numpy as np
import pytest

import phasegen as pg


class ExpmTestCase(TestCase):
    """
    Test matrix exponentiation.
    """

    @pytest.mark.slow
    def test_expm_different_backends(self):
        """
        Test that the available matrix exponentiation backends agree on a medium-sized matrix. The optional backends
        (TensorFlow, Jax, PyTorch) are only checked if their underlying package is installed.
        """
        S = pg.Coalescent(n=10).block_counting_state_space.S

        # SciPy is always available and serves as the reference
        reference = pg.SciPyExpmBackend().compute(S)

        # optional backends, keyed by the module they require
        optional_backends = {
            'tensorflow': pg.TensorFlowExpmBackend,
            'jax': pg.JaxExpmBackend,
            'torch': pg.expm.PyTorchExpmBackend,
        }

        for module, backend in optional_backends.items():
            if importlib.util.find_spec(module) is None:
                continue

            np.testing.assert_array_almost_equal(backend().compute(S), reference)
