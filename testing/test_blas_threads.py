"""
Tests of :class:`phasegen._blas.BlasThreadLimit`, the limit on the BLAS threads during moment evaluations.
"""
import numpy as np
import threadpoolctl

import phasegen as pg
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
