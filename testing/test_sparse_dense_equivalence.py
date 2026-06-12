"""
Equivalence of sparse and dense rate-matrix construction (and the moment paths that consume it).

The rate matrix is stored densely for small state spaces and sparsely once the transient count exceeds
:attr:`Settings.dense_rate_matrix_max_states` (see :meth:`StateSpace._construct_numba`). Likewise the moment
paths switch to sparse linear algebra (expm-action, sparse closed-form LU) above
:attr:`Settings.expm_action_min_dim` / :attr:`Settings.closed_form_sparse_min_states`. These thresholds only
change *how* a result is computed, never *what*; this module pins that down at small, fast sizes by forcing the
construction sparse-vs-dense and the moment paths sparse-vs-dense and asserting bit-for-bit (construction) and
floating-point (moments) agreement across every state-space type.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import phasegen as pg
from phasegen.settings import Settings
from phasegen.state_space import (
    LineageCountingStateSpace,
    BlockCountingStateSpace,
    JointBlockCountingStateSpace,
)

# a value comfortably above every state count exercised here, used to force *dense* storage / dense paths
_HUGE = 10 ** 9


@pytest.fixture(autouse=True)
def _restore_settings():
    """Restore the sparse/dense thresholds (and numba) after each test."""
    saved = {
        name: getattr(Settings, name)
        for name in (
            'use_numba',
            'dense_rate_matrix_max_states',
            'expm_action_min_dim',
            'closed_form_sparse_min_states',
        )
    }
    yield
    for name, value in saved.items():
        setattr(Settings, name, value)


def _demography(pop_sizes, migration_rate=1.0):
    """Single-deme (no migration) or multi-deme symmetric-migration demography."""
    from itertools import product

    pops = list(pop_sizes)

    if len(pops) == 1:
        return pg.Demography(pop_sizes=pop_sizes)

    migration_rates = {(a, b): migration_rate for a, b in product(pops, repeat=2) if a != b}

    return pg.Demography(pop_sizes=pop_sizes, migration_rates=migration_rates)


# one state space of every type / model, kept small so both storage modes are cheap
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
]


def _build_S(make_state_space, dense_max_states):
    """Build a state space forcing the given dense/sparse storage and return (state rows, rate matrix)."""
    Settings.use_numba = True
    Settings.dense_rate_matrix_max_states = dense_max_states
    ss = make_state_space()
    rows = ss.lineages[:, 0, :, :].reshape(ss.k, -1)
    return rows, ss.S


@pytest.mark.parametrize("label, make", STATE_SPACES, ids=[s[0] for s in STATE_SPACES])
def test_construction_sparse_matches_dense(label, make):
    """Forcing sparse storage yields a (densified) rate matrix bit-for-bit identical to dense storage."""
    rows_dense, S_dense = _build_S(make, dense_max_states=_HUGE)
    rows_sparse, S_sparse = _build_S(make, dense_max_states=0)

    # storage really differs
    assert not sp.issparse(S_dense), f"{label}: expected dense storage"
    assert sp.issparse(S_sparse), f"{label}: expected sparse storage"

    # same states in the same order (same kernel, only final storage differs)
    assert np.array_equal(rows_dense, rows_sparse), f"{label}: state ordering differs between storage modes"

    dense = np.asarray(S_dense)
    densified = np.asarray(S_sparse.todense())

    assert dense.shape == densified.shape, f"{label}: shape mismatch"
    assert np.abs(dense - densified).max() == 0.0, f"{label}: sparse vs dense rate matrix differs"


# a two-epoch single-deme demography (population size changes at t = 0.5) and a two-epoch two-deme demography
# (sizes change at t = 0.6, with symmetric migration). The multi-epoch cases below exercise the finite-epoch
# Van Loan / per-epoch occupation accumulation, which the time-homogeneous cases do not reach.
def _two_epoch_single():
    return pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.3}})


def _two_epoch_two_deme():
    return pg.Demography(
        pop_sizes={'pop_0': {0: 1.0, 0.6: 0.4}, 'pop_1': {0: 1.5, 0.6: 2.0}},
        migration_rates={('pop_0', 'pop_1'): 0.75, ('pop_1', 'pop_0'): 0.75},
    )


# end-to-end statistics spanning the mean and covariance moment paths over all state-space types, for both
# time-homogeneous (single-epoch) and time-inhomogeneous (multi-epoch) demographies
MOMENT_CASES = [
    ("sfs n=6 mean", lambda: pg.Coalescent(n=6), lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs n=6 cov", lambda: pg.Coalescent(n=6), lambda c: np.asarray(c.sfs.cov.data)),
    ("sfs beta n=6 mean", lambda: pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("tree height var n=10", lambda: pg.Coalescent(n=10), lambda c: np.asarray(c.tree_height.var)),
    ("jsfs 2-deme n=2+2 mean", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75)), lambda c: np.asarray(c.jsfs.mean.data)),
    ("jsfs 2-deme n=2+2 cov", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75)), lambda c: np.asarray(c.jsfs.cov.data)),
    ("sfs2 n=4 r=1 mean", lambda: pg.Coalescent(n=4, loci=2, recombination_rate=1.0),
     lambda c: np.asarray(c.sfs2.mean.data)),
    # --- multi-epoch (time-inhomogeneous) ---
    ("sfs n=6 multi-epoch mean", lambda: pg.Coalescent(n=6, demography=_two_epoch_single()),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs n=6 multi-epoch cov", lambda: pg.Coalescent(n=6, demography=_two_epoch_single()),
     lambda c: np.asarray(c.sfs.cov.data)),
    ("tree height var n=10 multi-epoch",
     lambda: pg.Coalescent(n=10, demography=_two_epoch_single()), lambda c: np.asarray(c.tree_height.var)),
    ("jsfs 2-deme n=2+2 multi-epoch mean", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_two_epoch_two_deme()), lambda c: np.asarray(c.jsfs.mean.data)),
    ("jsfs 2-deme n=2+2 multi-epoch cov", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2}, demography=_two_epoch_two_deme()), lambda c: np.asarray(c.jsfs.cov.data)),
]


@pytest.mark.parametrize("label, make, get", MOMENT_CASES, ids=[m[0] for m in MOMENT_CASES])
def test_moment_paths_sparse_matches_dense(label, make, get):
    """Forcing the moment paths fully sparse vs fully dense yields the same statistic to floating-point tolerance."""
    # fully dense: dense rate matrix and dense expm / closed-form everywhere
    Settings.dense_rate_matrix_max_states = _HUGE
    Settings.expm_action_min_dim = _HUGE
    Settings.closed_form_sparse_min_states = _HUGE
    dense = get(make())

    # fully sparse: sparse rate matrix and sparse expm-action / closed-form everywhere
    Settings.dense_rate_matrix_max_states = 0
    Settings.expm_action_min_dim = 0
    Settings.closed_form_sparse_min_states = 0
    sparse = get(make())

    np.testing.assert_allclose(sparse, dense, atol=1e-10, rtol=1e-8, err_msg=label)
