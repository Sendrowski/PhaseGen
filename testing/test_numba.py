"""
Tests for the numba-accelerated state-space construction.

These verify that the numba path reproduces the pure-Python construction (up to the allowed state reordering) across
state-space types and coalescent models, and that forcing the Python fallback gives identical results.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.settings import Settings
from phasegen.state_space import (
    LineageCountingStateSpace,
    BlockCountingStateSpace,
    JointBlockCountingStateSpace,
)


@pytest.fixture(autouse=True)
def _restore_numba_setting():
    """Restore the global numba setting after each test."""
    prev = Settings.use_numba
    yield
    Settings.use_numba = prev


def _demography(pop_sizes, migration_rate=1.0):
    """Two-or-more-deme demography with symmetric migration (or a single deme with no migration)."""
    from itertools import product

    pops = list(pop_sizes)

    if len(pops) == 1:
        return pg.Demography(pop_sizes=pop_sizes)

    migration_rates = {(a, b): migration_rate for a, b in product(pops, repeat=2) if a != b}

    return pg.Demography(pop_sizes=pop_sizes, migration_rates=migration_rates)


def _build_S(make_state_space, use_numba):
    """Build a state space with the given construction path and return its states and rate matrix."""
    Settings.use_numba = use_numba
    ss = make_state_space()
    rows = ss.lineages[:, 0, :, :].reshape(ss.k, -1)
    return rows, np.asarray(ss.S)


def _assert_S_parity(make_state_space, label, atol=1e-11):
    """Assert that the numba and Python rate matrices agree up to a state (row/column) permutation."""
    rows_p, S_p = _build_S(make_state_space, use_numba=False)
    rows_n, S_n = _build_S(make_state_space, use_numba=True)

    assert len(rows_n) == len(rows_p), f"{label}: state count {len(rows_n)} != {len(rows_p)}"

    index = {tuple(r): i for i, r in enumerate(rows_p)}
    perm = np.array([index[tuple(r)] for r in rows_n])  # numba index -> python index

    reordered = np.zeros_like(S_p)
    reordered[np.ix_(perm, perm)] = S_n

    assert np.abs(reordered - S_p).max() < atol, f"{label}: max|S| diff {np.abs(reordered - S_p).max():.2e}"


# state spaces spanning all three types and all three models
STATE_SPACES = [
    ("lineage 1-deme n=8 std",
     lambda: LineageCountingStateSpace(lineage_config=pg.LineageConfig(8),
                                       epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("lineage 2-deme n=3+3 std",
     lambda: LineageCountingStateSpace(lineage_config=pg.LineageConfig({'pop_0': 3, 'pop_1': 3}),
                                       epoch=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75).get_epoch(0))),
    ("block n=7 std",
     lambda: BlockCountingStateSpace(lineage_config=pg.LineageConfig(7),
                                     epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("block n=7 beta",
     lambda: BlockCountingStateSpace(lineage_config=pg.LineageConfig(7), model=pg.BetaCoalescent(alpha=1.5),
                                     epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("block n=7 dirac",
     lambda: BlockCountingStateSpace(lineage_config=pg.LineageConfig(7), model=pg.DiracCoalescent(psi=0.5, c=1.0),
                                     epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("joint 2-deme n=3+3 std",
     lambda: JointBlockCountingStateSpace(lineage_config=pg.LineageConfig({'pop_0': 3, 'pop_1': 3}),
                                          epoch=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75).get_epoch(0))),
    ("joint 3-deme n=2+1+1 std",
     lambda: JointBlockCountingStateSpace(lineage_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 1, 'pop_2': 1}),
                                          epoch=_demography({'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.0}).get_epoch(0))),
]


@pytest.mark.parametrize("label, make", STATE_SPACES, ids=[s[0] for s in STATE_SPACES])
def test_rate_matrix_parity_up_to_permutation(label, make):
    """The numba rate matrix matches the pure-Python one up to a state permutation."""
    _assert_S_parity(make, label)


# end-to-end moment parity through the public API
MOMENT_CASES = [
    ("sfs n=6", lambda: pg.Coalescent(n=6), lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs beta n=6", lambda: pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs dirac n=6", lambda: pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.5, c=1.0)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("tree height variance n=10", lambda: pg.Coalescent(n=10), lambda c: c.tree_height.var),
    ("jsfs 2-deme n=2+2", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75)), lambda c: np.asarray(c.jsfs.mean.data)),
]


@pytest.mark.parametrize("label, make, get", MOMENT_CASES, ids=[m[0] for m in MOMENT_CASES])
def test_moment_parity_numba_vs_python(label, make, get):
    """Moments computed via the numba and Python construction paths agree to floating-point tolerance."""
    Settings.use_numba = True
    numba = np.asarray(get(make()))

    Settings.use_numba = False
    python = np.asarray(get(make()))

    np.testing.assert_allclose(numba, python, atol=1e-10, err_msg=label)


def test_two_loci_use_python_path():
    """The 2-locus (recombination) case is not numba-accelerated and must fall back to the Python construction."""
    coal = pg.Coalescent(n=4, loci=2, recombination_rate=1.0)
    ss = coal.lineage_counting_state_space

    assert not ss._use_numba()
    # still computes correctly via the Python path
    assert coal.tree_height.mean > 0


def test_fallback_setting_disables_numba():
    """Setting ``Settings.use_numba = False`` routes construction through the pure-Python path."""
    Settings.use_numba = False
    ss = BlockCountingStateSpace(lineage_config=pg.LineageConfig(5),
                                 epoch=_demography({'pop_0': 1.0}).get_epoch(0))

    assert not ss._use_numba()
    assert ss.S.shape[0] == ss.k
