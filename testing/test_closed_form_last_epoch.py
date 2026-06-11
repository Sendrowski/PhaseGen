"""
Tests for the closed-form evaluation of the final (unbounded) epoch of a moment-to-absorption
(:attr:`phasegen.settings.Settings.closed_form_last_epoch`).

The closed form must reproduce the default matrix-exponential path bit-for-bit wherever absorption is almost sure,
and must fall back gracefully on the degenerate cases: absorption occurring (almost surely) before the last epoch,
and demographies that never absorb (disconnected demes).
"""
import warnings

import numpy as np
import pytest

import phasegen as pg
import phasegen.distributions as _pgd
from phasegen.settings import Settings
from testing import TestCase

_HI, _LO = 10 ** 12, 0


@pytest.fixture(autouse=True)
def _restore_flag():
    """Restore the closed-form / path-selection settings after each test."""
    prev = (Settings.closed_form_last_epoch, Settings.expm_action_min_dim, _pgd._CLOSED_FORM_SPARSE_MIN_N)
    yield
    Settings.closed_form_last_epoch, Settings.expm_action_min_dim, _pgd._CLOSED_FORM_SPARSE_MIN_N = prev


def _both(fn):
    """Evaluate ``fn`` with the closed form off then on, returning both arrays."""
    Settings.closed_form_last_epoch = False
    off = np.asarray(fn(), dtype=float)
    Settings.closed_form_last_epoch = True
    on = np.asarray(fn(), dtype=float)
    Settings.closed_form_last_epoch = False
    return off, on


def _four_paths(fn):
    """
    Evaluate ``fn`` under all four moment solvers, forcing each via the path-selection thresholds:
    dense/sparse matrix exponential (closed form off) and dense/sparse closed form (closed form on).
    """
    # (closed_form, expm_action_min_dim, closed_form_sparse_min_n)
    configs = {
        "dense-expm": (False, _HI, _HI),
        "sparse-expm": (False, _LO, _HI),
        "dense-cf": (True, _HI, _HI),
        "sparse-cf": (True, _HI, _LO),
    }
    out = {}
    for name, (cf, eamd, cfmin) in configs.items():
        Settings.closed_form_last_epoch = cf
        Settings.expm_action_min_dim = eamd
        _pgd._CLOSED_FORM_SPARSE_MIN_N = cfmin
        out[name] = np.asarray(fn(), dtype=float)
    return out


# demography helpers
_MIG = pg.Demography(
    pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
    migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0}
)
_3EPOCH = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.4: 0.3, 1.2: 2.0}})

_MIG_2EPOCH = pg.Demography(
    pop_sizes={'pop_0': {0: 1.0, 0.5: 0.4}, 'pop_1': {0: 1.0}},
    migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 0.6}
)

_CASES = {
    # tree height: increasing moment order, single and multi epoch / deme
    "tree_height.mean n=5": lambda: pg.Coalescent(n=5).tree_height.mean,
    "tree_height.var n=6": lambda: pg.Coalescent(n=6).tree_height.var,
    "tree_height.moment3 n=6": lambda: pg.Coalescent(n=6).tree_height.moment(3, center=False),
    "total_branch_length.mean n=6": lambda: pg.Coalescent(n=6).total_branch_length.mean,
    "3-epoch tree_height.var": lambda: pg.Coalescent(n=5, demography=_3EPOCH).tree_height.var,
    # site-frequency spectrum (block-counting state space), incl. a cross-moment of two different reward bins
    "sfs.mean n=6": lambda: pg.Coalescent(n=6).sfs.mean.data,
    "sfs.var n=5": lambda: pg.Coalescent(n=5).sfs.var.data,
    "sfs cross-moment E[L1 L3] n=6":
        lambda: pg.Coalescent(n=6).moment(2, (pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(3)), center=False),
    # multiple-merger models (the transient back-substitution is model-agnostic)
    "beta sfs.mean n=6": lambda: pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs.mean.data,
    "dirac tree_height.var n=6":
        lambda: pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.5, c=1.0)).tree_height.var,
    # multi-deme, single and multi epoch, joint SFS
    "2-deme tree_height.mean": lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG).tree_height.mean,
    "2-deme jsfs.mean": lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG).jsfs.mean.data,
    "2-epoch 2-deme jsfs.mean":
        lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG_2EPOCH).jsfs.mean.data,
}


@pytest.mark.parametrize("name", list(_CASES), ids=list(_CASES))
def test_closed_form_matches_default(name):
    """The closed form reproduces the default path to floating-point precision (single/multi epoch and deme)."""
    off, on = _both(_CASES[name])
    np.testing.assert_allclose(on, off, rtol=1e-8, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("name", list(_CASES), ids=list(_CASES))
def test_all_four_solvers_consistent(name):
    """
    All four moment solvers agree: dense and sparse matrix-exponential, and dense and sparse closed form. Each is
    forced via the path-selection thresholds so the small test state spaces still exercise the sparse paths.
    """
    res = _four_paths(_CASES[name])
    ref = res["dense-expm"]
    for path, val in res.items():
        np.testing.assert_allclose(val, ref, rtol=1e-7, atol=1e-10, err_msg=f"{name}: {path} vs dense-expm")


class ClosedFormLastEpochTestCase(TestCase):
    """
    Validate the closed-form last-epoch evaluation against the matrix-exponential default.
    """

    def test_absorption_certain_predicate(self):
        """The structural predicate is True for a connected demography and False for disconnected demes."""
        connected = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG)
        assert connected.tree_height._absorption_certain_in_last_epoch() is True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disconnected = pg.Coalescent(
                n={'pop_0': 2, 'pop_1': 2},
                demography=pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 1.0})
            )
        assert disconnected.tree_height._absorption_certain_in_last_epoch() is False

    def test_no_absorption_falls_back(self):
        """Disconnected demes never absorb: the closed form is not applied and the result matches the default path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coal = pg.Coalescent(
                n={'pop_0': 2, 'pop_1': 2},
                demography=pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 1.0})
            )
            # the guard prevents the (singular) closed-form solve; both paths agree
            off, on = _both(lambda: coal.tree_height.mean)
        np.testing.assert_allclose(on, off, rtol=1e-8)

    def test_absorption_before_last_epoch(self):
        """A demography that coalesces almost entirely in an early epoch (tiny later population size) is handled
        exactly by the closed form, since its connected last epoch still has a non-singular transient block."""
        coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.05: 1e-6}}))
        assert coal.tree_height._absorption_certain_in_last_epoch() is True
        off, on = _both(lambda: coal.tree_height.mean)
        np.testing.assert_allclose(on, off, rtol=1e-8)
