"""
Tests of :class:`phasegen._blas.BlasThreadLimit`, the limit on the BLAS threads during moment evaluations.
"""
import numpy as np
import pytest
import threadpoolctl

import phasegen as pg
from phasegen._blas import BlasThreadLimit
from phasegen.distributions import _moments
from phasegen.settings import Settings


def _blas_threads():
    return [p['num_threads'] for p in threadpoolctl.threadpool_info() if p['user_api'] == 'blas']


def test_moments_run_with_the_configured_blas_threads_and_restore_them(monkeypatch):
    """The BLAS libraries run with ``Settings.blas_threads`` inside a moment and with their own number after it."""
    seen = []
    accumulate = _moments.MomentEvaluator._accumulate_closed_form

    def spy(self, *args, **kwargs):
        seen.append(_blas_threads())
        return accumulate(self, *args, **kwargs)

    monkeypatch.setattr(_moments.MomentEvaluator, '_accumulate_closed_form', spy)
    with threadpoolctl.threadpool_limits(2, user_api='blas'):
        before = _blas_threads()
        pg.Coalescent(n=5).tree_height.moment(2)
        after = _blas_threads()

    assert seen and all(t == [1] * len(t) for t in seen)
    assert after == before


def test_blas_threads_none_leaves_the_threads(monkeypatch):
    """With ``Settings.blas_threads = None`` the moment runs with the threads the caller set."""
    seen = []
    accumulate = _moments.MomentEvaluator._accumulate_closed_form

    def spy(self, *args, **kwargs):
        seen.append(_blas_threads())
        return accumulate(self, *args, **kwargs)

    monkeypatch.setattr(_moments.MomentEvaluator, '_accumulate_closed_form', spy)
    monkeypatch.setattr(Settings, 'blas_threads', None)
    with threadpoolctl.threadpool_limits(2, user_api='blas'):
        pg.Coalescent(n=5).tree_height.moment(2)

    assert seen and all(t == [2] * len(t) for t in seen)


def test_moments_do_not_depend_on_the_blas_threads(monkeypatch):
    """The moments with one BLAS thread equal those with the caller's threads."""
    def moments():
        coal = pg.Coalescent(n=12, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.4, 1.5: 2.0}}))
        return np.array([coal.tree_height.moment(3), coal.total_branch_length.var, *coal.sfs.mean.data])

    limited = moments()
    monkeypatch.setattr(Settings, 'blas_threads', None)

    np.testing.assert_allclose(limited, moments(), rtol=1e-13)


def _threads_inside(monkeypatch, method: str, evaluate) -> list:
    """The BLAS threads seen by the numerical-stability checks run inside ``MomentEvaluator.<method>``."""
    seen, depth = [], [0]
    target = getattr(_moments.MomentEvaluator, method)
    check = _moments.MomentEvaluator._check_numerical_stability

    def wrapped(self, *args, **kwargs):
        depth[0] += 1
        try:
            return target(self, *args, **kwargs)
        finally:
            depth[0] -= 1

    def spy(self, *args, **kwargs):
        if depth[0]:
            seen.append(_blas_threads())
        return check(self, *args, **kwargs)

    monkeypatch.setattr(_moments.MomentEvaluator, method, wrapped)
    monkeypatch.setattr(_moments.MomentEvaluator, '_check_numerical_stability', spy)
    with threadpoolctl.threadpool_limits(2, user_api='blas'):
        evaluate()

    return seen


def _two_demes() -> pg.Coalescent:
    return pg.Coalescent(
        n={'pop_0': 3, 'pop_1': 3},
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1.0, 0.5: 0.4}, 'pop_1': {0: 0.5}},
            migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 0.5}
        )
    )


def test_batched_sfs_mean_runs_with_the_configured_blas_threads(monkeypatch):
    """
    The batched closed-form ``sfs.mean`` runs ``MomentEvaluator._occupation_times`` with ``Settings.blas_threads``.
    It ran with the caller's threads, as only ``MomentEvaluator.accumulate`` limited them.
    """
    seen = _threads_inside(monkeypatch, '_occupation_times', lambda: _two_demes().sfs.mean)

    assert seen and all(t == [1] * len(t) for t in seen)


def test_batched_jsfs_accumulate_runs_with_the_configured_blas_threads(monkeypatch):
    """
    The batched ``jsfs.accumulate(1, ...)`` runs ``MomentEvaluator._mean_occupation_grid`` with
    ``Settings.blas_threads``. It ran with the caller's threads, as only ``MomentEvaluator.accumulate`` limited them.
    """
    seen = _threads_inside(
        monkeypatch, '_mean_occupation_grid', lambda: _two_demes().jsfs.accumulate(1, [0.25, 1.0, 2.0])
    )

    assert seen and all(t == [1] * len(t) for t in seen)


def _threads_in_two_point_occupation(monkeypatch, evaluate) -> list:
    """
    The BLAS threads seen by ``MomentEvaluator._two_point_occupation`` and by each call of the solve it returns, under
    a caller's limit of four threads.
    """
    seen = []
    target = _moments.MomentEvaluator._two_point_occupation

    def spy(self):
        seen.append(_blas_threads())
        out = target(self)
        if out is None:
            return None
        m, solve, idx_t = out

        def spied_solve(b):
            seen.append(_blas_threads())
            return solve(b)

        return m, spied_solve, idx_t

    monkeypatch.setattr(_moments.MomentEvaluator, '_two_point_occupation', spy)
    with threadpoolctl.threadpool_limits(4, user_api='blas'):
        evaluate()

    return seen


@pytest.mark.parametrize('evaluate', [
    lambda: pg.Coalescent(n=6).sfs.var,
    lambda: pg.Coalescent(n=6).sfs.cov,
    lambda: _two_demes().jsfs.cov,
    lambda: pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2.mean,
], ids=['sfs.var', 'sfs.cov', 'jsfs.cov', 'sfs2.mean'])
def test_two_point_occupation_runs_with_the_configured_blas_threads(monkeypatch, evaluate):
    """
    The batched second-order spectra factorize and solve in ``MomentEvaluator._two_point_occupation`` with
    ``Settings.blas_threads``. They ran with the caller's threads, as only ``MomentEvaluator.accumulate``,
    ``MomentEvaluator._occupation_times`` and ``MomentEvaluator._mean_occupation_grid`` limited them.
    """
    seen = _threads_in_two_point_occupation(monkeypatch, evaluate)

    assert seen and all(t == [1] * len(t) for t in seen)


def test_stacked_sfs_cov_runs_with_the_configured_blas_threads(monkeypatch):
    """
    The stacked closed form of ``fsfs.cov`` across epochs runs ``MomentEvaluator._accumulate_closed_form`` with
    ``Settings.blas_threads``. It ran with the caller's threads, as ``SFSDistribution.cov`` calls it directly.
    """
    seen = []
    accumulate = _moments.MomentEvaluator._accumulate_closed_form

    def spy(self, *args, **kwargs):
        seen.append(_blas_threads())
        return accumulate(self, *args, **kwargs)

    monkeypatch.setattr(_moments.MomentEvaluator, '_accumulate_closed_form', spy)
    coal = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.4, 1.5: 2.0}}))
    with threadpoolctl.threadpool_limits(4, user_api='blas'):
        coal.fsfs.cov

    assert seen and all(t == [1] * len(t) for t in seen)


@pytest.mark.parametrize('evaluate', [
    lambda: _two_demes().tree_height.gradient(),
    lambda: _two_demes().gradient(),
    lambda: _two_demes().sfs.gradient(),
    lambda: pg.Gradient(
        lambda N: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: N}})),
        lambda coal: coal.sfs.mean
    ).compute(dict(N=0.4)),
], ids=['tree_height.gradient', 'Coalescent.gradient', 'sfs.gradient', 'Gradient.compute'])
def test_gradients_run_with_the_configured_blas_threads(monkeypatch, evaluate):
    """The exact gradients run ``MomentEvaluator._occupation_times`` with ``Settings.blas_threads``."""
    seen = _threads_inside(monkeypatch, '_occupation_times', evaluate)

    assert seen and all(t == [1] * len(t) for t in seen)


def test_nested_blas_thread_limits_restore_the_outer_limit():
    """An inner exit of a nested limit keeps the outer limit, and the outer exit restores the caller's threads."""
    limit = BlasThreadLimit()
    with threadpoolctl.threadpool_limits(2, user_api='blas'):
        with limit:
            with limit:
                inner = _blas_threads()
            between = _blas_threads()
        after = _blas_threads()

    assert inner == [1] * len(inner) and between == [1] * len(between) and after == [2] * len(after)
