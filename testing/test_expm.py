"""
Test matrix exponentiation.
"""
import importlib.util
import math

from testing import TestCase

import numpy as np
import scipy.linalg
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
        and every statistic routed through the sparse action raised. Extended precision is rejected too, since
        scipy.linalg.expm raises on it.
        """
        import scipy.sparse as sp

        for precision in (np.float32, 'float32'):
            backend = pg.SciPyExpmBackend(precision=precision)
            self.assertEqual(backend.compute(np.eye(2)).dtype, np.float32)
            self.assertEqual(backend.compute_action(sp.csr_matrix(np.eye(2)), np.ones(2)).dtype, np.float32)

        np.testing.assert_allclose(pg.SciPyExpmBackend().compute_action(sp.csr_matrix(np.eye(2)), np.ones(2)), np.e)

        # long double is plain double precision on some platforms, such as arm64 macOS
        extended = (np.longdouble, 'longdouble') if np.dtype(np.longdouble) != np.dtype(np.float64) else ()

        for precision in ('np.float32', int, np.float16, 'float16') + extended:
            with self.assertRaisesRegex(TypeError, "np.float32 or np.float64"):
                pg.SciPyExpmBackend(precision=precision)

    def test_scipy_backend_scales_a_large_norm(self):
        """
        A sub-intensity generator of 1-norm 1e39 with one absorbing state exponentiates to its limit, in which all
        mass sits in the absorbing state. A moderate argument takes the plain SciPy path unchanged. Regression:
        scipy.linalg.expm chose 2^31 squarings and did not return.
        """
        a = np.triu(np.random.default_rng(1).random((8, 8)))
        np.fill_diagonal(a, 0)
        a[np.diag_indices(8)] = -a.sum(axis=1)

        limit = np.zeros((8, 8))
        limit[:, -1] = 1

        np.testing.assert_allclose(pg.SciPyExpmBackend().compute(a * 1e39), limit, atol=1e-12)
        np.testing.assert_allclose(pg.SciPyExpmBackend().compute(a), scipy.linalg.expm(a), rtol=1e-14)

        # a nilpotent argument exponentiates to its finite power series, which the ten squarings beyond the bound
        # reproduce to about 1e-11
        n = np.diag(np.ones(4), k=1) * 1e39
        exact = sum(np.linalg.matrix_power(n, k) / math.factorial(k) for k in range(5))

        np.testing.assert_allclose(pg.SciPyExpmBackend().compute(n), exact, rtol=1e-9)

    @pytest.mark.slow
    def test_expm_different_backends(self):
        """
        Test that the available matrix exponentiation backends agree on a medium-sized matrix. The optional backends
        (TensorFlow, Jax, PyTorch) are only checked if their underlying package is installed.
        """
        S = pg.Coalescent(n=10).state_spaces['block_counting'].S

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


@pytest.mark.parametrize('stiff', [False, True])
def test_expm_multiply_interval_steps_where_stiff(stiff):
    """
    The interval action matches the actions of each multiple. Below the stiff threshold it takes one interval action
    of the backend, above it it applies the steps one by one, each stopping once the vector is stationary.
    """
    from unittest.mock import patch
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    from phasegen.expm import Backend

    rng = np.random.default_rng(0)
    S = sp.random(30, 30, density=0.2, random_state=1).toarray()
    S -= np.diag(S.sum(axis=1) + 0.5)
    a = sp.csr_matrix(S.T * (1e3 if stiff else 1.0))
    b = rng.uniform(size=30)
    stop, num = 2.0, 7

    with patch.object(Backend.backend, 'compute_action_interval', wraps=Backend.backend.compute_action_interval) as spy:
        out = Backend.expm_multiply_interval(a, b, stop, num)

    assert spy.call_count == (0 if stiff else 1)
    assert out.shape == (num, 30)

    ref = np.stack([expm_multiply(a * (stop * (j + 1) / num), b) for j in range(num)])
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())


def test_default_backend_interval_action():
    """The interval action of a backend without its own repeats the dense exponential of one step."""
    from phasegen.expm import ExpmBackend

    class Dense(ExpmBackend):
        def compute(self, m):
            return scipy.linalg.expm(m)

    a = np.array([[-1.0, 1.0], [0.0, -2.0]])
    b = np.array([1.0, 2.0])
    out = Dense().compute_action_interval(a, b, 3.0, 3)

    np.testing.assert_allclose(out, [scipy.linalg.expm(a * s) @ b for s in (1, 2, 3)], rtol=1e-13)
