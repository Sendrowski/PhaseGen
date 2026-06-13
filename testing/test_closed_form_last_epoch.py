"""
Tests for the closed-form evaluation of the final (unbounded) epoch of a moment-to-absorption
(:attr:`phasegen.settings.Settings.closed_form_last_epoch`).

The closed form must reproduce the default matrix-exponential path bit-for-bit wherever absorption is almost sure,
and must fall back gracefully on the degenerate cases: absorption occurring (almost surely) before the last epoch,
and demographies that never absorb (disconnected demes).
"""
import contextlib
import warnings

import numpy as np
import pytest

import phasegen as pg
import phasegen.distributions as _pgd
from phasegen.settings import Settings
from testing import TestCase

_HI, _LO = 10 ** 12, 0

# the closed-form / batched dispatch entry points (every one is gated on ``_use_closed_form``)
_CF_ENTRY_POINTS = ("_accumulate_closed_form", "_occupation_times", "_two_point_occupation")


@contextlib.contextmanager
def _count_closed_form_calls():
    """Count calls to the closed-form / batched entry points while still executing them normally."""
    counts = {name: 0 for name in _CF_ENTRY_POINTS}
    originals = {name: getattr(_pgd.PhaseTypeDistribution, name) for name in _CF_ENTRY_POINTS}

    def make(name, orig):
        def wrapper(self, *args, **kwargs):
            counts[name] += 1
            return orig(self, *args, **kwargs)

        return wrapper

    try:
        for name in _CF_ENTRY_POINTS:
            setattr(_pgd.PhaseTypeDistribution, name, make(name, originals[name]))
        yield counts
    finally:
        for name in _CF_ENTRY_POINTS:
            setattr(_pgd.PhaseTypeDistribution, name, originals[name])


@pytest.fixture(autouse=True)
def _restore_flag():
    """Restore the closed-form / path-selection settings after each test."""
    prev = (Settings.closed_form_last_epoch, Settings.expm_action_min_dim,
            Settings.closed_form_sparse_min_states)
    yield
    (Settings.closed_form_last_epoch, Settings.expm_action_min_dim,
     Settings.closed_form_sparse_min_states) = prev


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
    # (closed_form_last_epoch, expm_action_min_dim, closed_form_sparse_min_states); _HI/_LO force the
    # dense/sparse sub-paths
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
        Settings.closed_form_sparse_min_states = cfmin
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
    # joint-SFS second moments, batched through the shared two-point occupation (single and multi epoch)
    "2-deme jsfs.cov": lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG).jsfs.cov,
    "2-deme jsfs.var": lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG).jsfs.var.data,
    "2-epoch 2-deme jsfs.cov":
        lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG_2EPOCH).jsfs.cov,
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


# a battery covering every moment-evaluation kind: scalar moments, mean spectra, single- and multi-epoch
# covariances, and a multi-deme case
_3EPOCH_DEMO = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.4: 0.3, 1.2: 2.0}})
_DISPATCH_BATTERY = {
    "tree_height.mean": lambda: pg.Coalescent(n=6).tree_height.mean,
    "tree_height.var": lambda: pg.Coalescent(n=6).tree_height.var,
    "sfs.mean": lambda: pg.Coalescent(n=6).sfs.mean.data,
    "sfs.cov 1-epoch": lambda: pg.Coalescent(n=6).sfs.cov.data,
    "sfs.cov 3-epoch": lambda: pg.Coalescent(n=6, demography=_3EPOCH_DEMO).sfs.cov.data,
    "jsfs.mean": lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG).jsfs.mean.data,
    "jsfs.cov": lambda: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_MIG).jsfs.cov,
}


@pytest.mark.parametrize("name", list(_DISPATCH_BATTERY), ids=list(_DISPATCH_BATTERY))
def test_disabled_uses_only_matrix_exponential(name):
    """
    Fully disabling the closed form falls back to the matrix
    exponential for *every* moment kind: none of the closed-form / batched entry points is ever reached.
    """
    Settings.closed_form_last_epoch = False
    with _count_closed_form_calls() as counts:
        _DISPATCH_BATTERY[name]()
    assert sum(counts.values()) == 0, f"{name}: closed-form/batched path used while disabled: {counts}"


# the documented dispatch: which closed-form / batched entry points each moment kind reaches when enabled.
# (covariances also touch ``_occupation_times`` because they subtract the batched mean.)
_EXPECTED_PATHS = {
    "tree_height.mean": {"_accumulate_closed_form"},
    "tree_height.var": {"_accumulate_closed_form"},
    # single-population SFS mean flattens (k=1, standard); the small lineage-counting space then uses the closed form
    "sfs.mean": {"_accumulate_closed_form"},
    # the cov's mean flattens (closed form on the lineage space); the covariance itself uses the two-point operator
    "sfs.cov 1-epoch": {"_two_point_occupation", "_accumulate_closed_form"},
    # multi-epoch covariance: the batched single-epoch operator is declined (returns None) -> per-pair closed form
    "sfs.cov 3-epoch": {"_two_point_occupation", "_accumulate_closed_form"},
    # joint SFS is multi-population, so flattening does not apply: mean uses the batched occupation
    "jsfs.mean": {"_occupation_times"},
    "jsfs.cov": {"_two_point_occupation", "_occupation_times"},
}


@pytest.mark.parametrize("name", list(_EXPECTED_PATHS), ids=list(_EXPECTED_PATHS))
def test_enabled_dispatch_paths(name):
    """With the closed form enabled, each moment kind takes exactly its documented dispatch path (see
    :meth:`PhaseTypeDistribution._use_closed_form`)."""
    Settings.closed_form_last_epoch = True
    with _count_closed_form_calls() as counts:
        _DISPATCH_BATTERY[name]()
    used = {name_ for name_, n in counts.items() if n > 0}
    assert used == _EXPECTED_PATHS[name], f"{name}: reached {used}, expected {_EXPECTED_PATHS[name]} ({counts})"


@pytest.mark.parametrize("demography", [None, _3EPOCH], ids=["1-epoch", "3-epoch"])
def test_closed_form_matches_time_accumulation_and_is_grid_stable(demography):
    """
    The closed-form back-substitution moment to absorption equals the limit of the matrix-exponential time
    accumulation (the incremental ``Q @= expm(V dt)`` threading), and that accumulation is numerically stable on a
    fine time grid: finite, monotonic, and grid-independent down to small steps. Run single- and multi-epoch (the
    latter also exercises the finite-epoch Van Loan within the closed form).
    """
    th = pg.Coalescent(n=8, demography=demography).tree_height

    # closed-form (back-substitution) mean to absorption
    Settings.closed_form_last_epoch = True
    cf_mean = float(th.mean)

    # matrix-exponential accumulation over a large interval, on a fine and a coarse grid with the same endpoint
    Settings.closed_form_last_epoch = False
    t_end = th.t_max * 3
    fine = th.accumulate(k=1, end_times=np.linspace(0, t_end, 800), center=False)    # small steps
    coarse = th.accumulate(k=1, end_times=np.linspace(0, t_end, 20), center=False)   # large steps

    assert np.all(np.isfinite(fine)), "accumulation produced non-finite values on a fine grid"
    assert np.all(np.diff(fine) >= -1e-10), "mean accumulation is not monotonic (small-step instability)"
    # grid-independent: the incremental threading gives the same endpoint regardless of step size
    np.testing.assert_allclose(fine[-1], coarse[-1], rtol=1e-9)
    # the time accumulation converges to the closed-form (back-substitution) value
    np.testing.assert_allclose(fine[-1], cf_mean, rtol=1e-3)


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

    def test_no_absorption_raises_in_both_modes(self):
        """Disconnected demes never absorb: there is no almost-sure absorption time, so both the closed-form and
        the matrix-exponential path must raise rather than return the doubling-search ceiling."""
        coal = pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 2},
            demography=pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 1.0})
        )
        for cf in (False, True):
            Settings.closed_form_last_epoch = cf
            with self.assertRaisesRegex(ValueError, "does not absorb"):
                _ = coal.tree_height.mean
        Settings.closed_form_last_epoch = False

    def test_absorption_before_last_epoch(self):
        """A demography that coalesces almost entirely in an early epoch (tiny later population size) is handled
        exactly by the closed form, since its connected last epoch still has a non-singular transient block."""
        coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.05: 1e-6}}))
        assert coal.tree_height._absorption_certain_in_last_epoch() is True
        off, on = _both(lambda: coal.tree_height.mean)
        np.testing.assert_allclose(on, off, rtol=1e-8)
