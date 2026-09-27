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

    def test_scipy_backend_precision(self):
        """
        The SciPy backend accepts NumPy floating types and their names, and applies the precision to both the dense
        exponential and its action. The annotated literal ``'np.float32'`` made every dense exponential raise, and the
        action ignored the precision. Half precision is rejected at construction, since scipy.sparse has no float16
        and every statistic routed through the sparse action raised.
        """
        import scipy.sparse as sp

        for precision in (np.float32, 'float32'):
            backend = pg.SciPyExpmBackend(precision=precision)
            self.assertEqual(backend.compute(np.eye(2)).dtype, np.float32)
            self.assertEqual(backend.compute_action(sp.csr_matrix(np.eye(2)), np.ones(2)).dtype, np.float32)

        np.testing.assert_allclose(pg.SciPyExpmBackend().compute_action(sp.csr_matrix(np.eye(2)), np.ones(2)), np.e)

        for precision in ('np.float32', int, np.float16, 'float16'):
            with self.assertRaisesRegex(TypeError, "floating-point type"):
                pg.SciPyExpmBackend(precision=precision)

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


def test_a_backend_returning_read_only_arrays_gives_multi_epoch_moments():
    """Multi-epoch moments work with a backend whose arrays are read-only, as those of Jax are. Regression: the
    in-place writes of the Van Loan moments raised on every multi-epoch moment."""
    class ReadOnly(pg.SciPyExpmBackend):
        def compute(self, m):
            out = super().compute(m)
            out.flags.writeable = False
            return out

    coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}}))
    expected = coal.tree_height.var

    previous = pg.Backend.backend
    try:
        pg.Backend.backend = ReadOnly()
        got = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}})).tree_height.var
    finally:
        pg.Backend.backend = previous

    assert got == pytest.approx(expected, rel=1e-12)
