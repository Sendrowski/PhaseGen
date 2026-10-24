"""
Tests of :class:`phasegen._block_expm.BlockTriangularExpm`, the structured exponential of the Van Loan matrices and of
the augmented occupation generator, against the generic dense exponential.
"""
import numpy as np
import pytest
import scipy.linalg

import phasegen as pg
from phasegen._block_expm import BlockTriangularExpm
from phasegen.distributions import _moments
from phasegen.expm import SciPyExpmBackend


def _generator(n: int, rng: np.random.Generator, absorbing: int = 0) -> np.ndarray:
    """A sparse random sub-generator of size ``n`` whose last ``absorbing`` rows are zero."""
    S = rng.random((n, n)) * (rng.random((n, n)) < 0.4)
    np.fill_diagonal(S, 0)
    np.fill_diagonal(S, -S.sum(1) - 0.3 * rng.random(n))
    if absorbing:
        S[n - absorbing:] = 0
    return S


def _van_loan(S: np.ndarray, k: int, rng: np.random.Generator, zero_rewards: bool = False) -> np.ndarray:
    """A block-bidiagonal Van Loan matrix with ``S`` on the ``k + 1`` diagonal blocks and diagonal rewards above."""
    n = len(S)
    A = np.zeros(((k + 1) * n, (k + 1) * n))
    for i in range(k + 1):
        A[i * n:(i + 1) * n, i * n:(i + 1) * n] = S
        if i < k and not zero_rewards:
            A[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = np.diag(rng.random(n))
    return A


def _occupation(S: np.ndarray) -> np.ndarray:
    """The augmented occupation generator ``[[S, I], [0, 0]]``."""
    n = len(S)
    A = np.zeros((2 * n, 2 * n))
    A[:n, :n] = S
    A[:n, n:] = np.eye(n)
    return A


def _cases():
    rng = np.random.default_rng(3)
    cases = []
    for n in [1, 5, 30]:
        for tau in [0.0, 1e-9, 0.2, 7.0, 1e3]:
            cases.append((f'occupation n={n} tau={tau}', _occupation(_generator(n, rng)) * tau, n, [False, True]))
            for k in [1, 2, 3]:
                cases.append((f'van loan k={k} n={n} tau={tau}', _van_loan(_generator(n, rng), k, rng) * tau, n,
                              [False] * (k + 1)))
    S = _generator(12, rng, absorbing=3)  # singular generator, as with zero migration or absorbing states
    cases.append(('singular occupation', _occupation(S) * 5.0, 12, [False, True]))
    cases.append(('singular van loan', _van_loan(S, 2, rng) * 5.0, 12, [False] * 3))
    cases.append(('zero rewards', _van_loan(_generator(10, rng), 2, rng, zero_rewards=True) * 3.0, 10, [False] * 3))
    return cases


@pytest.mark.parametrize('name, A, n, scalar', _cases(), ids=lambda x: x if isinstance(x, str) else '')
def test_structured_expm_matches_the_generic_one(name, A, n, scalar):
    """The structured exponential agrees with :func:`scipy.linalg.expm` and is exactly zero below the diagonal. The
    error of scaling and squaring grows with the norm of the matrix, as with ``tau = 1e3``, where SciPy's reference is
    off by about 1e-10 of its largest entry on some BLAS builds."""
    E = BlockTriangularExpm(n, scalar).compute(A)
    ref = scipy.linalg.expm(A)

    tol = 1e-12 * max(1.0, np.linalg.norm(A, 1))
    np.testing.assert_allclose(E, ref, rtol=0, atol=tol * np.abs(ref).max())
    for i in range(1, len(scalar)):
        assert not E[i * n:(i + 1) * n, :i * n].any()


def _patterned(n: int, scalar: list, rng: np.random.Generator) -> np.ndarray:
    """A block upper triangular matrix with ``c I`` on the scalar diagonal blocks and one shared matrix elsewhere."""
    K = len(scalar)
    D = rng.standard_normal((n, n))
    A = np.zeros((K * n, K * n))
    for i in range(K):
        A[i * n:(i + 1) * n, i * n:(i + 1) * n] = rng.standard_normal() * np.eye(n) if scalar[i] else D
        A[i * n:(i + 1) * n, (i + 1) * n:] = rng.standard_normal((n, (K - i - 1) * n))
    return A


@pytest.mark.parametrize('n', [4, 25])
@pytest.mark.parametrize('scalar', [
    [False, True, False], [False, True, True], [True, False, False], [False, False, True], [False] * 4,
    [True, True, False, True]
], ids=str)
def test_structured_expm_with_a_scalar_block_before_the_last(n, scalar):
    """
    A scalar diagonal block that is neither first nor last carries the products of the blocks above it with those to
    its right. Without them every pattern with such a block missed the paths through it and disagreed with
    :func:`scipy.linalg.expm` by order one.
    """
    A = _patterned(n, scalar, np.random.default_rng(7))
    ref = scipy.linalg.expm(A)

    np.testing.assert_allclose(BlockTriangularExpm(n, scalar).compute(A), ref, rtol=0, atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize('k', [1, 3])
@pytest.mark.parametrize('tau', [60.0, 1e3])
def test_structured_expm_of_a_triangular_matrix_keeps_its_accuracy_over_many_squarings(k, tau):
    """
    The Van Loan matrix of a lineage-counting chain is upper triangular. Recomputing its diagonal and superdiagonal
    exactly after each squaring keeps the exponential within 1e-14 of :func:`scipy.linalg.expm`, which does the same,
    where the plain squarings drift by 4e-12 at ``tau = 60`` and 3e-11 at ``tau = 1000``.
    """
    n = 30
    rates = np.arange(n, 0, -1) * (np.arange(n, 0, -1) - 1) / 2.0
    S = np.diag(-rates) + np.diag(rates[:-1], 1)
    A = np.zeros(((k + 1) * n, (k + 1) * n))
    for i in range(k + 1):
        A[i * n:(i + 1) * n, i * n:(i + 1) * n] = S
        if i < k:
            A[i * n:(i + 1) * n, (i + 1) * n:(i + 2) * n] = np.diag((rates > 0).astype(float))

    E = BlockTriangularExpm(n, [False] * (k + 1)).compute(A * tau)
    ref = scipy.linalg.expm(A * tau)

    np.testing.assert_allclose(E, ref, rtol=1e-14, atol=1e-300)


def test_structured_expm_scales_an_extreme_norm():
    """A 1-norm beyond what the unscaled powers hold is halved first, matching the scaled generic exponential."""
    rng = np.random.default_rng(5)
    A = _van_loan(_generator(6, rng), 2, rng) * 1e60
    E = BlockTriangularExpm(6, [False] * 3).compute(A)

    assert np.isfinite(E).all()
    np.testing.assert_allclose(E, SciPyExpmBackend().compute(A), rtol=0, atol=1e-12)


def test_backward_error_squarings_of_a_non_normal_matrix():
    """
    The extra squarings of Al-Mohy & Higham (2009, eq. 5.1) for ``[[1, 100], [0, -1]]``, whose powers have norms far
    below those of the powers of its modulus, and its exponential ``[[e, 100 sinh(1)], [0, 1 / e]]``.
    """
    A = np.array([[1.0, 100.0], [0.0, -1.0]])
    h = BlockTriangularExpm(1, [True, True])

    assert [h._ell(A, 101.0, m) for m in (3, 5, 7, 9, 13)] == [7, 3, 1, 0, 0]
    np.testing.assert_allclose(h.compute(A), [[np.e, 100 * np.sinh(1)], [0, 1 / np.e]], rtol=1e-15)


@pytest.mark.parametrize('n, epochs, stat', [
    (30, 4, 'tree_height.m3'), (30, 4, 'total_branch_length.var'), (10, 20, 'tree_height.m3'), (10, 20, 'sfs.mean'),
    (10, 12, 'sfs.var'), ((3, 3), 12, 'sfs.mean')
])
def test_moments_do_not_depend_on_the_structured_exponential(monkeypatch, n, epochs, stat):
    """
    The moments whose dense Van Loan and occupation exponentials of all epochs take ``BlockTriangularExpm``, in
    stacks, agree with those of the generic exponential.
    """
    def moments():
        times = np.linspace(0, 3, epochs, endpoint=False)
        pops = ['pop_0'] if isinstance(n, int) else ['pop_0', 'pop_1']
        coal = pg.Coalescent(
            n=n if isinstance(n, int) else dict(zip(pops, n)),
            demography=pg.Demography(
                pop_sizes={
                    p: {float(t): 1 + 0.5 * np.sin(i + j) for i, t in enumerate(times)} for j, p in enumerate(pops)
                },
                migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.3} if len(pops) == 2 else None
            )
        )
        if stat == 'tree_height.m3':
            return np.atleast_1d(coal.tree_height.moment(3, center=False))
        if stat == 'total_branch_length.var':
            return np.atleast_1d(coal.total_branch_length.moment(2, center=False))
        return np.asarray(getattr(coal.sfs, stat.split('.')[1]).data)

    monkeypatch.setattr(_moments.MomentEvaluator, '_block_expm_applies', staticmethod(lambda *args: True))
    structured = moments()
    monkeypatch.setattr(_moments.MomentEvaluator, '_block_expm_applies', staticmethod(lambda *args: False))

    np.testing.assert_allclose(structured, moments(), rtol=1e-12)
