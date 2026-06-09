"""
Tests for the sparse matrix-exponential-action moment computation.

For large state spaces moments are computed via the action of the matrix exponential on a vector
(``_accumulate_action``) instead of the dense Van Loan propagator. These tests force both paths and check that they
agree across state-space types, models, moment orders, multiple epochs and cross-moments.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.settings import Settings

#: force the dense / action path respectively
DENSE = 10 ** 9
ACTION = 0


@pytest.fixture(autouse=True)
def _restore_threshold():
    """Restore the global threshold after each test."""
    prev = Settings.expm_action_min_dim
    yield
    Settings.expm_action_min_dim = prev


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
