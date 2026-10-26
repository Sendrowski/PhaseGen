"""
Tests for the uniformized exponential and integral of a Metzler sub-generator
(``phasegen.distributions._uniformization._Uniformization``).
"""
from unittest.mock import patch

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.special import gammaln

from phasegen.distributions import _uniformization
from phasegen.distributions._uniformization import _Uniformization


def _subgenerator(n: int, scale: float, seed: int = 0, exit_scale: float = None) -> np.ndarray:
    """A random dense sub-generator with exits to absorption, at rates up to ``exit_scale`` (``scale`` by default)."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (n, n)) * (rng.uniform(0, 1, (n, n)) < 0.4) * scale
    np.fill_diagonal(A, 0)
    np.fill_diagonal(A, -A.sum(axis=1) - rng.uniform(0, 1, n) * (scale if exit_scale is None else exit_scale))
    return A


def _van_loan(A: np.ndarray, v: np.ndarray, tau: float):
    """The exponential and its integral applied to ``v`` from one dense exponential of the Van Loan matrix."""
    n = len(v)
    E = sla.expm(np.block([[A, np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]]) * tau)
    return v @ E[:n, :n], v @ E[:n, n:]


@pytest.mark.parametrize('scale, exit_scale, tau', [(1.0, None, 0.3), (1.0, None, 4.0), (50.0, 0.01, 30.0)],
                         ids=['short', 'long', 'chunked'])
def test_uniformization_matches_the_van_loan_exponential(scale, exit_scale, tau):
    """Both products agree with the dense Van Loan exponential, also when the step is split into chunks. The exits of
    the chunked case are slow, so that most of the mass is left after the last chunk."""
    A = _subgenerator(12, scale, exit_scale=exit_scale)
    v = np.random.default_rng(1).uniform(0, 1, 12)
    lam = -A.diagonal().min()
    assert (lam * tau > _uniformization._MAX_CHUNK) == (scale == 50.0)

    u, occ = _Uniformization(A).propagate(v, tau)
    u_ref, occ_ref = _van_loan(A, v, tau)

    if exit_scale is not None:
        assert u_ref.sum() > 0.5 * v.sum()
    np.testing.assert_allclose(u, u_ref, rtol=1e-12, atol=1e-14 * v.sum())
    np.testing.assert_allclose(occ, occ_ref, rtol=1e-12)


def test_uniformization_keeps_tiny_entries_nonnegative_and_accurate():
    """Down a chain where each entry is about a thousand times smaller than the one before, every entry above the
    truncation floor keeps its relative accuracy against the closed form, far below the resolution of the vector's
    norm, and none is negative or above the closed form."""
    n, rate, a = 20, 1.0, 1e-3
    # exits at rate 1, of which a fraction a moves one step down the chain: entry k at time t is (a t)^k e^{-t} / k!
    A = np.diag(-np.full(n, rate)) + np.diag(np.full(n - 1, a * rate), 1)
    v = np.zeros(n)
    v[0] = 1
    t = 2.0

    u, occ = _Uniformization(A).propagate(v, t)

    k = np.arange(n)
    exact = np.exp(k * np.log(a * t) - t - gammaln(k + 1))
    above = exact > 100 * _uniformization._FLOOR
    assert exact[-1] < _uniformization._FLOOR and exact[above].min() < 1e-40
    assert (u >= 0).all() and (occ >= 0).all() and (u <= exact * (1 + 1e-12)).all()
    np.testing.assert_allclose(u[above], exact[above], rtol=1e-12)


def test_uniformization_of_a_zero_generator_and_step():
    """A zero generator holds the vector, and a zero step leaves it unchanged with zero integral."""
    v = np.array([0.25, 0.75])

    u, occ = _Uniformization(np.zeros((2, 2))).propagate(v, 2.0)
    np.testing.assert_array_equal(u, v)
    np.testing.assert_array_equal(occ, 2 * v)

    u, occ = _Uniformization(_subgenerator(2, 1.0)).propagate(v, 0.0)
    np.testing.assert_array_equal(u, v)
    np.testing.assert_array_equal(occ, 0 * v)


def test_uniformization_truncates_at_the_floor():
    """Entries below the truncation floor relative to the norm may be truncated, but never above it, and the
    normalization of the truncated weights raises none by more than the floor."""
    with patch.object(_uniformization, '_FLOOR', 1e-10):
        n, a, t = 12, 1e-3, 2.0
        A = np.diag(-np.ones(n)) + np.diag(np.full(n - 1, a), 1)
        v = np.zeros(n)
        v[0] = 1
        u, _ = _Uniformization(A).propagate(v, t)

    k = np.arange(n)
    exact = np.exp(k * np.log(a * t) - t - gammaln(k + 1))
    big = exact > 1e-8
    np.testing.assert_allclose(u[big], exact[big], rtol=1e-2)
    assert (u <= exact * (1 + 1e-12) / (1 - 1e-10)).all()


def test_uniformization_keeps_the_deep_tail_of_a_chunked_step():
    """Down a chain of equal rates, the state reached after k jumps holds the Poisson probability of k at mean lam t.
    Over two chunks, the states far down the chain are reached only by series terms deep in the Poisson tail of a
    chunk, and every one above the truncation floor keeps its relative accuracy."""
    n, lam, t = 1500, 1.0, 1000.0
    A = np.diag(-np.full(n, lam)) + np.diag(np.full(n - 1, lam), 1)
    v = np.zeros(n)
    v[0] = 1
    assert np.ceil(lam * t / _uniformization._MAX_CHUNK) == 2

    u, _ = _Uniformization(A).propagate(v, t)

    k = np.arange(n)
    exact = np.exp(k * np.log(lam * t) - lam * t - gammaln(k + 1))
    above = exact > 1e-35
    assert exact[above].min() < 1e-34 and k[above].max() > 1350
    np.testing.assert_allclose(u[above], exact[above], rtol=1e-10)


def test_uniformization_weights_sum_to_one():
    """The truncated Poisson weights are normalized. Regression: they summed to 1 - 1.9e-13 at a mean of 500, a
    relative bias that every chunk of a long step reapplied."""
    w, Q = _Uniformization._weights(500.0)

    assert abs(w.sum() - 1) <= 4 * np.finfo(float).eps
    assert abs(Q[0] - (1 - w[0])) <= 4 * np.finfo(float).eps


def test_uniformization_of_a_stiff_step_over_many_chunks():
    """A fast state draining into a slow one over ten thousand chunks keeps the accuracy of the occupation and of the
    vector, against the closed form evaluated at 60 digits. Regression: the unnormalized Poisson weights biased each
    chunk by 2e-13, which accumulated to a relative error of 4e-10 in the occupation and 2e-9 in the vector."""
    A = np.array([[-1e3, 1e3 - 1e-3], [0, -1e-3]])
    tau = 5000.0
    assert np.ceil(1e3 * tau / _uniformization._MAX_CHUNK) == 1e4

    u, occ = _Uniformization(A).propagate(np.array([1.0, 0.0]), tau)

    np.testing.assert_allclose(u[1], 0.0067379469990854670966, rtol=5e-10)
    np.testing.assert_allclose(occ, [0.001, 993.2610530009145329], rtol=1e-10)


def test_uniformization_stops_once_the_vector_vanishes():
    """A fast exit over twenty thousand chunks empties the vector within the first few, after which no chunk is
    evaluated, and the occupation keeps its closed form. Regression: every remaining chunk ran its series on the zero
    vector, which made a deep bottleneck at a long epoch time take 23 s in place of 1 s."""
    A = np.array([[-1e3, 0.5e3], [0, -1e3]])
    tau = 1e4
    assert np.ceil(1e3 * tau / _uniformization._MAX_CHUNK) == 2e4

    calls = []
    series = _uniformization._series

    def counting(*args):
        calls.append(1)
        return series(*args)

    with patch.object(_uniformization, '_series', counting):
        u, occ = _Uniformization(A).propagate(np.array([1.0, 0.0]), tau)

    assert len(calls) < 10
    np.testing.assert_array_equal(u, 0)
    np.testing.assert_allclose(occ, [1e-3, 0.5e-3], rtol=1e-12)


@pytest.mark.parametrize('factor', [0.99, 1.01])
@pytest.mark.parametrize('rate_factor', [0.99, 1.01])
def test_uniformization_drops_entries_just_below_the_support_floor(factor, rate_factor):
    """Over a long step, a fast state draining into a slow one is dropped when it holds just below the support floor
    relative to the 1-norm, and kept just above it. States holding far less that no entry above the floor reaches are
    kept when they exit at most the rate ratio times as fast as the states reachable from those above the floor, and
    dropped just above. Once a kept fast state has drained over the first chunk, those faster than the rate ratio
    allows relative to the rest are dropped as well, and the rest of the step is taken in one chunk. Both products
    agree with the closed form to a few unit roundoffs."""
    r = rate_factor * _uniformization._SUPPORT_RATE_RATIO
    A = np.array([[-1.0, 0, 0, 0], [1e6, -1e6, 0, 0], [0, 0, -0.5, 0], [0, 0, 0, -r]])
    v = np.array([1.0, 0, 1e-30, 1e-30])
    v[1] = eps = factor * _uniformization._SUPPORT_FLOOR * v.sum()
    tau = 1.0
    fast_kept = factor > 1
    kept = rate_factor < 1 or fast_kept
    # the time over which the last state is propagated: the step, the first chunk, or nothing
    held = tau if rate_factor < 1 else tau / np.ceil(1e6 * tau / _uniformization._MAX_CHUNK) * fast_kept

    keep = _Uniformization(A)._support(v)
    u, occ = _Uniformization(A).propagate(v, tau)

    # the drained state adds about eps relative to the slow state it drains into
    u_ref = [np.exp(-tau) * (1 + eps * fast_kept), 0, 1e-30 * np.exp(-0.5 * tau),
             1e-30 * np.exp(-r * tau) * (rate_factor < 1)]
    occ_ref = [-np.expm1(-tau) * (1 + eps * fast_kept), eps / 1e6 * fast_kept, 1e-30 * -np.expm1(-0.5 * tau) / 0.5,
               1e-30 * -np.expm1(-r * held) / r]
    np.testing.assert_array_equal(keep, [True, fast_kept, True, kept])
    np.testing.assert_allclose(u, u_ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(occ, occ_ref, rtol=1e-12, atol=0)


def test_uniformization_drops_a_drained_fast_residue_after_its_first_chunk():
    """A residue of 1e-9 on a state draining a million times faster than the bulk sets the rate of the first chunk
    only. Once it has drained, the rest of the step is taken at the rate of the bulk in a single chunk, and both
    products keep the closed form. Regression: the whole step was taken at the drain rate in 20000 chunks, which made a
    Dirac split whose derived deme shrinks afterwards take 282 s in place of 0.08 s."""
    A = np.array([[-1.0, 0], [1e6, -1e6]])
    v = np.array([1.0, 1e-9])
    tau = 10.0
    assert np.ceil(1e6 * tau / _uniformization._MAX_CHUNK) == 2e4

    calls = []
    series = _uniformization._series

    def counting(*args):
        calls.append(1)
        return series(*args)

    with patch.object(_uniformization, '_series', counting):
        u, occ = _Uniformization(A).propagate(v, tau)

    c = 1e-9 * 1e6 / (1e6 - 1)
    assert len(calls) == 2
    np.testing.assert_allclose(u, [np.exp(-tau) * (1 + c), 0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(occ, [-np.expm1(-tau) * (1 + c) - c / 1e6, 1e-15], rtol=1e-12)
