"""
Tests for the sparse matrix-exponential-action moment computation.

For large state spaces moments are computed via the action of the matrix exponential on a vector
(``MomentEvaluator._advance_action``) instead of the dense Van Loan propagator. These tests force both paths and check
that they agree across state-space types, models, moment orders, multiple epochs and cross-moments.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.settings import Settings

#: force the dense / action path respectively
DENSE = 10 ** 9
ACTION = 0


def _demography(pop_sizes, migration_rate=1.0, two_epoch=False):
    from itertools import product

    pops = list(pop_sizes)
    sizes = {p: ({0: pop_sizes[p], 1.0: pop_sizes[p] * 0.4} if two_epoch else pop_sizes[p]) for p in pops}

    if len(pops) == 1:
        return pg.Demography(pop_sizes=sizes)

    migration_rates = {(a, b): migration_rate for a, b in product(pops, repeat=2) if a != b}
    return pg.Demography(pop_sizes=sizes, migration_rates=migration_rates)


def _both_paths(make, get):
    """Compute the statistic with the dense and the action path."""
    Settings.expm_action_min_dim = DENSE
    dense = np.asarray(get(make()))

    Settings.expm_action_min_dim = ACTION
    action = np.asarray(get(make()))

    return dense, action


CASES = [
    ("tree height mean n=5", lambda: pg.Coalescent(n=5), lambda c: c.tree_height.mean),
    ("tree height var n=6", lambda: pg.Coalescent(n=6), lambda c: c.tree_height.var),
    ("tree height tiny n=2", lambda: pg.Coalescent(n=2), lambda c: c.tree_height.mean),
    ("sfs mean 2-deme n=2+2", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs mean beta n=5", lambda: pg.Coalescent(n=5, model=pg.BetaCoalescent(alpha=1.5)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs mean dirac n=5", lambda: pg.Coalescent(n=5, model=pg.DiracCoalescent(psi=0.5, c=1.0)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("jsfs mean 2-deme n=2+2", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75)),
     lambda c: np.asarray(c.jsfs.mean.data)),
    ("jsfs var 2-deme n=2+2", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_demography({'pop_0': 1.0, 'pop_1': 1.0}, 1.0)),
     lambda c: np.asarray(c.jsfs.var.data)),
    ("tree height mean 2-epoch n=6", lambda: pg.Coalescent(
        n=6, demography=_demography({'pop_0': 1.0}, two_epoch=True)), lambda c: c.tree_height.mean),
    ("sfs mean 2-epoch 2-deme", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_demography({'pop_0': 1.0, 'pop_1': 1.0}, 0.75, two_epoch=True)),
     lambda c: np.asarray(c.sfs.mean.data)),
]


@pytest.mark.parametrize("label, make, get", CASES, ids=[c[0] for c in CASES])
def test_action_matches_dense(label, make, get):
    """The action and dense moment paths agree to floating-point tolerance."""
    dense, action = _both_paths(make, get)
    np.testing.assert_allclose(action, dense, rtol=1e-9, atol=1e-11, err_msg=label)


def test_action_cross_moment_matches_dense():
    """A cross-moment (k=2 with two different rewards) agrees between the action and dense paths."""
    make = lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_demography({'pop_0': 1.0, 'pop_1': 1.0}, 1.0))

    get = lambda c: c.moment(
        k=2, rewards=[pg.JointSFSReward((1, 0)), pg.JointSFSReward((0, 1))])

    dense, action = _both_paths(make, get)
    np.testing.assert_allclose(action, dense, rtol=1e-9, atol=1e-11)


def test_action_accumulate_over_time_matches_dense():
    """Accumulation over a grid of end times agrees between the action and dense paths."""
    make = lambda: pg.Coalescent(n=8)
    get = lambda c: np.asarray(c.tree_height.accumulate(k=1, end_times=np.linspace(0.1, 3, 15)))

    dense, action = _both_paths(make, get)
    np.testing.assert_allclose(action, dense, rtol=1e-8, atol=1e-10)


def test_expm_multiply_steps_end_at_stationarity():
    """An argument of large norm is applied in steps that end once the vector is stationary, matching the dense
    exponential, and an argument within ``_ACTION_STEP_NORM`` takes a single action. Regression: the cost of the action
    grew linearly in the norm, so a generator that absorbs its mass early in a long step took a step count in
    proportion to the step length."""
    import scipy.linalg as sla
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from phasegen.expm import Backend, _ACTION_STEP_NORM

    S = np.array([[-3.0, 2.0, 1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    p = np.array([1.0, 0.0, 0.0])
    calls = []

    def action(a, b):
        calls.append(float(abs(a).sum(axis=0).max()))
        return spla.expm_multiply(a, b)

    for scale, n_max in ((1e9, 3), (1.0, 1), (_ACTION_STEP_NORM / 6, 1)):
        calls.clear()
        A = sp.csr_matrix(S * scale).T.tocsr()
        v = Backend.expm_multiply(A, p, action=action)
        np.testing.assert_allclose(v, sla.expm(S * scale).T @ p, rtol=1e-12, atol=1e-15)
        assert 1 <= len(calls) <= n_max and calls[0] <= _ACTION_STEP_NORM * (1 + 1e-12)
        if n_max == 1:
            np.testing.assert_array_equal(v, spla.expm_multiply(A, p))


def test_expm_multiply_steps_continue_while_the_exponential_grows():
    """A matrix whose exponential grows in norm (logarithmic 1-norm 30) is not stopped early: the remaining change
    is bounded by the growth factor of ``_GrowthBound``, on ``Backend.expm_multiply`` and on ``_TapedAction``.
    Regression: the stop test bounded the remaining change by the remaining fraction times the derivative, valid only
    for exponentials that do not grow, so a component of 1e-19 that grows by e^30 was dropped after the first step
    and the result was off by 1e-6."""
    import scipy.linalg as sla
    import scipy.sparse as sp
    from phasegen.expm import Backend
    from phasegen.distributions._gradient import _TapedAction

    # a fast transposed generator moving the mass of state 0 to state 1, next to a growing decoupled state 2
    A = np.array([[-5e4, 0.0, 0.0], [5e4, 0.0, 0.0], [0.0, 0.0, 30.0]])
    v = np.array([1.0, 0.0, 1e-19])
    exact = sla.expm(A) @ v

    np.testing.assert_allclose(Backend.expm_multiply(sp.csr_matrix(A), v), exact, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(_TapedAction(sp.csr_matrix(A)).apply(v)[0], exact, rtol=1e-10, atol=1e-15)


def test_tree_height_terms_bounded_by_beta_alpha_close_to_two():
    """The tree-height density and CDF at 2 - alpha = 1e-6 on the sparse action, within a budget of Taylor terms,
    against the dense exponential. Regression: the propagation ended only once the transient entries were exactly
    unchanged by a step, which the noise of the action prevents, so it went on in steps doubling in norm until the
    end of the epoch and the cost grew with the rates as 1 / (2 - alpha)."""
    from scipy.sparse.linalg import _expm_multiply

    t = np.array([1e-7, 1e-6, 1e-5, 1.0, 3.0])

    def tree_height():
        return pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=2 - 1e-6)).tree_height

    Settings.expm_action_min_dim = DENSE
    th = tree_height()
    pdf, cdf = th.pdf(t), th.cdf(t)

    Settings.expm_action_min_dim = ACTION
    core, terms = _expm_multiply._expm_multiply_simple_core, [0]

    def counted(A, B, t_, mu, m_star, s, *args, **kwargs):
        terms[0] += m_star * s
        return core(A, B, t_, mu, m_star, s, *args, **kwargs)

    _expm_multiply._expm_multiply_simple_core = counted
    try:
        th = tree_height()
        np.testing.assert_allclose(th.pdf(t), pdf, rtol=1e-10, atol=1e-12 * pdf.max())
        np.testing.assert_allclose(th.cdf(t), cdf, rtol=1e-10, atol=1e-15)
    finally:
        _expm_multiply._expm_multiply_simple_core = core

    assert 0 < terms[0] < 40000
