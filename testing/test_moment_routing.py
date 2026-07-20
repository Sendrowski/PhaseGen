"""
Routing tests for the moment-evaluation engine (:class:`phasegen.distributions._moments.MomentEvaluator`).

These pin down *which* path the dispatch takes — flattening vs closed-form vs matrix-exponential, and the
dense/sparse sub-paths — rather than the numeric result (covered by ``test_sparse_dense_equivalence`` /
``test_closed_form_last_epoch``). They guard the refactor that split the engine into a mixin and unified the
dense/sparse Van Loan builder and LU factorization.
"""
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

import phasegen as pg
from phasegen.settings import Settings
from phasegen.distributions import PhaseTypeDistribution
from phasegen.distributions._moments import MomentEvaluator


@pytest.fixture(autouse=True)
def _restore_settings():
    saved = {
        name: getattr(Settings, name)
        for name in ('flatten_block_counting', 'closed_form_last_epoch', 'expm_action_min_dim',
                     'closed_form_sparse_min_states', 'dense_rate_matrix_max_states')
    }
    yield
    for name, value in saved.items():
        setattr(Settings, name, value)


def _spy(method):
    """Patch a MomentEvaluator method with a pass-through spy and return the mock (call-counting)."""
    return patch.object(PhaseTypeDistribution, method, autospec=True,
                        side_effect=getattr(PhaseTypeDistribution, method))


# ----------------------------------------------------------------------------------------------------------------
# _flattening_applies truth table
# ----------------------------------------------------------------------------------------------------------------

def test_flattening_applies_standard_single_pop_first_moment():
    """Flattening applies to the first moment of the single-population, single-locus standard coalescent SFS."""
    sfs = pg.Coalescent(n=6).sfs
    assert sfs._flattening_applies(1) is True
    # but not for the second moment (covariance)
    assert sfs._flattening_applies(2) is False


def test_flattening_not_applied_for_mmc():
    """Multiple-merger models are excluded (the jump-chain block-size law does not reconstruct the reward)."""
    assert pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs._flattening_applies(1) is False
    assert pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.5, c=1.0)).sfs._flattening_applies(1) is False


def test_flattening_not_applied_for_multiple_populations():
    """The joint (multi-population) SFS uses a joint block-counting space, which is not flattened."""
    coal = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
                                 migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0}),
    )
    assert coal.jsfs._flattening_applies(1) is False


def test_flattening_respects_global_switch():
    """The ``flatten_block_counting`` setting gates the predicate."""
    sfs = pg.Coalescent(n=6).sfs
    Settings.flatten_block_counting = False
    assert sfs._flattening_applies(1) is False
    Settings.flatten_block_counting = True
    assert sfs._flattening_applies(1) is True


# ----------------------------------------------------------------------------------------------------------------
# unified Van Loan builder and LU solver
# ----------------------------------------------------------------------------------------------------------------

def test_van_loan_matrix_dense_and_sparse_agree():
    """The single builder yields a dense array or a sparse CSR matrix that densify to the same block structure."""
    S = np.array([[-2.0, 2.0], [0.0, -1.0]])
    R = [np.array([1.0, 0.5])]

    dense = MomentEvaluator._van_loan_matrix(R, S, k=1, sparse=False)
    sparse = MomentEvaluator._van_loan_matrix(R, sp.csr_matrix(S), k=1, sparse=True)

    assert not sp.issparse(dense)
    assert sp.issparse(sparse)
    np.testing.assert_allclose(dense, sparse.toarray())

    # block-bidiagonal: S on the diagonal blocks, diag(R) on the super-diagonal, zero below
    np.testing.assert_allclose(dense[:2, :2], S)
    np.testing.assert_allclose(dense[2:, 2:], S)
    np.testing.assert_allclose(dense[:2, 2:], np.diag(R[0]))
    np.testing.assert_allclose(dense[2:, :2], 0.0)


def test_lu_solver_dense_and_sparse_solve_correctly():
    """Both factorizations solve ``A x = b`` and the callable is reusable across right-hand sides."""
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b1, b2 = np.array([5.0, 5.0]), np.array([1.0, 0.0])

    for solve in (MomentEvaluator._lu_solver(A, sparse=False),
                  MomentEvaluator._lu_solver(sp.csc_matrix(A), sparse=True)):
        np.testing.assert_allclose(solve(b1), np.linalg.solve(A, b1))
        np.testing.assert_allclose(solve(b2), np.linalg.solve(A, b2))


def test_block_triangular_order_detects_grading_and_falls_back():
    """The SCC ordering reorders a graded (block-triangular) generator and declines a single-SCC one."""
    # graded chain: three lineage levels 3->2->1, with a 2-state migration cycle inside level 2 (one SCC of size 2,
    # the rest singletons) -> a valid block-triangular permutation must exist
    idx = {'L3': 0, 'L2a': 1, 'L2b': 2, 'L1': 3}
    A = np.zeros((4, 4))
    A[idx['L3'], idx['L2a']] = 1.0          # coalescence 3 -> 2
    A[idx['L2a'], idx['L2b']] = 1.0         # migration within level 2 (cycle)
    A[idx['L2b'], idx['L2a']] = 1.0         # migration back -> SCC {L2a, L2b}
    A[idx['L2a'], idx['L1']] = 1.0          # coalescence 2 -> 1
    A[idx['L2b'], idx['L1']] = 1.0
    np.fill_diagonal(A, -A.sum(axis=1) - 1.0)   # make it a proper invertible sub-generator

    perm = MomentEvaluator._block_triangular_order(A)
    assert perm is not None and sorted(perm.tolist()) == [0, 1, 2, 3]
    # the migration pair lands in a contiguous diagonal block (adjacent in the ordering)
    pos = {state: int(np.where(perm == i)[0][0]) for state, i in idx.items()}
    assert abs(pos['L2a'] - pos['L2b']) == 1

    # the NATURAL-ordered solve via this permutation is still correct
    solve = MomentEvaluator._lu_solver(sp.csc_matrix(A), sparse=True)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(solve(b), np.linalg.solve(A, b))

    # a single strongly-connected matrix offers no triangular structure -> fall back (None)
    cyc = np.array([[-1.0, 1.0], [1.0, -1.0]])
    assert MomentEvaluator._block_triangular_order(cyc) is None


# ----------------------------------------------------------------------------------------------------------------
# path dispatch (closed-form vs matrix-exponential, dense vs sparse action)
# ----------------------------------------------------------------------------------------------------------------

def test_closed_form_path_taken_when_enabled():
    """With the closed form enabled, the moment to absorption routes through ``_accumulate_closed_form`` (and not
    the sparse-action sub-path)."""
    Settings.closed_form_last_epoch = True
    coal = pg.Coalescent(n=5)
    with _spy('_accumulate_closed_form') as cf, _spy('_accumulate_action') as action:
        _ = coal.tree_height.mean
    assert cf.call_count >= 1
    assert action.call_count == 0


def test_matrix_exponential_path_when_closed_form_disabled():
    """With the closed form disabled, the closed-form sub-path is not taken; the dispatcher ``_accumulate`` runs
    the matrix-exponential path instead."""
    Settings.closed_form_last_epoch = False
    coal = pg.Coalescent(n=5)
    with _spy('_accumulate_closed_form') as cf, _spy('_accumulate') as dispatch:
        _ = coal.tree_height.mean
    assert cf.call_count == 0
    assert dispatch.call_count >= 1


def test_sparse_action_path_taken_below_threshold():
    """A zero ``expm_action_min_dim`` forces the sparse matrix-exponential action (``_accumulate_action``)."""
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 0
    coal = pg.Coalescent(n=5)
    with _spy('_accumulate_action') as action:
        _ = coal.tree_height.mean
    assert action.call_count >= 1


def test_dense_expm_path_taken_above_threshold():
    """A huge ``expm_action_min_dim`` keeps the dense Van Loan exponential (no action path)."""
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 10 ** 12
    coal = pg.Coalescent(n=5)
    with _spy('_accumulate_action') as action:
        _ = coal.tree_height.mean
    assert action.call_count == 0


@pytest.mark.parametrize('force_sparse', [False, True])
def test_accumulate_restores_input_order_for_unsorted_times(force_sparse):
    """``accumulate`` must return moments aligned to the *input* end-time order, not the internal sorted order.
    Regression for the inverse-permutation bug (``argsort`` instead of ``argsort(argsort(...))``) which mis-attached
    moments for >= 3 unsorted times, in both the dense (``_accumulate``) and sparse-action (``_accumulate_action``)
    paths."""
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 0 if force_sparse else 10 ** 12

    th = pg.Coalescent(n=5).tree_height
    times = [3.0, 0.5, 2.0, 1.0]  # >= 3 entries and unsorted -> exposes the permutation bug

    acc = th.accumulate(k=1, end_times=times)
    ref = np.array([th.accumulate(k=1, end_times=[t])[0] for t in times])

    np.testing.assert_allclose(acc, ref, rtol=1e-8, atol=1e-10)


def test_flattened_path_taken_for_sfs_mean():
    """The single-population standard SFS mean routes through the flattened accumulation."""
    Settings.flatten_block_counting = True
    coal = pg.Coalescent(n=6)
    with _spy('_accumulate_flattened') as flat:
        _ = coal.sfs.mean
    assert flat.call_count >= 1


# ----------------------------------------------------------------------------------------------------------------
# windowed (start_time > 0) higher moments  --  bug-scan-2026-07-19 #2
# ----------------------------------------------------------------------------------------------------------------

def test_windowed_second_moment_is_true_windowed_moment():
    """A windowed (``start_time > 0``) k>=2 moment is E[(H - a)_+^k], NOT the naive m_end - m_start subtraction
    of two cumulative-from-0 moments (which is only additive for the mean).

    Regression for #2: ``pg.Coalescent(n=3).tree_height.moment(k=2, center=False, start_time=0.4)`` returned the
    naive subtraction 2.742 (= E[H^2] - E[min(H,0.4)^2]) instead of the true windowed second moment 1.9775."""
    th = pg.Coalescent(n=3).tree_height

    # true windowed E[(H - 0.4)_+^2]; pre-fix returned the naive subtraction 2.742
    assert np.isclose(th.moment(k=2, center=False, start_time=0.4), 1.9774941145611176, rtol=1e-5)

    # the start_time = 0 case is the ordinary second moment and must be UNCHANGED by the fix
    assert np.isclose(th.moment(k=2, center=False, start_time=0), 2.888888888888889, rtol=1e-5)


def test_windowed_variance_and_std_use_cross_terms():
    """Centered variance / std with ``start_time > 0`` must be the true windowed centered moment, not the
    difference of the two cumulative variances (which omits the -2 E[Y_a Y_b] + 2 E[Y_a^2] cross terms).

    Regression for #2: centered variance at ``start_time=0.4`` returned 1.107 instead of the correct 1.065."""
    th = pg.Coalescent(n=3).tree_height

    var = th.moment(k=2, center=True, start_time=0.4)
    # true windowed centered second moment; pre-fix returned 1.107
    assert np.isclose(var, 1.0649322611477698, rtol=1e-5)
    # std is its square root
    assert np.isclose(np.sqrt(var), 1.0319555519244856, rtol=1e-5)

    # the start_time = 0 centered second moment (ordinary variance) is UNCHANGED by the fix
    assert np.isclose(th.moment(k=2, center=True, start_time=0), 1.1111111111111112, rtol=1e-5)


# ----------------------------------------------------------------------------------------------------------------
# facade default reward for windowed / finite-end moments  --  bug-scan-2026-07-19 #4
# ----------------------------------------------------------------------------------------------------------------

def test_facade_moment_uses_tree_height_default_reward():
    """The ``Coalescent`` facade default moment must reward the transient tree-height states ([1,1,1,0]), not the
    absorbing state (the wrong UnitReward [1,1,1,1]), so a windowed / finite-end default moment integrates the
    branch length, not the absorbing indicator.

    Regression for #4: with a nonzero start_time the facade default returned the absorbing-state integral instead
    of the tree-height moment."""
    # windowed default moment must equal the explicit tree_height windowed moment; pre-fix returned 62.999
    facade = pg.Coalescent(n=4, start_time=1.0).moment(1)
    explicit = pg.Coalescent(n=4).tree_height.moment(1, start_time=1.0)
    assert np.isclose(facade, explicit, rtol=1e-5)
    assert np.isclose(facade, 0.6456699297251958, rtol=1e-5)

    # finite end_time default moment; pre-fix returned 1.0 (the end time itself, absorbing reward)
    assert np.isclose(pg.Coalescent(n=4).moment(1, end_time=1.0), 0.8543300702748016, rtol=1e-5)

    # the unbounded mean (start_time=0, end_time=None) takes the flattened path and is UNCHANGED
    assert np.isclose(pg.Coalescent(n=4).moment(1), 1.5, rtol=1e-5)


def test_facade_accumulate_uses_tree_height_default_reward():
    """``Coalescent.accumulate`` with the default reward accumulates the tree height and saturates near the mean
    1.5, rather than returning the end times themselves (the absorbing-reward artifact).

    Regression for #4: pre-fix returned exactly the end times [0.5, 1, 2, 10]."""
    acc = pg.Coalescent(n=4).accumulate(1, [0.5, 1, 2, 10])
    # tree-height accumulation saturating near the mean 1.5; pre-fix returned [0.5, 1, 2, 10]
    np.testing.assert_allclose(
        acc, [0.48096196, 0.85433007, 1.25722254, 1.49991828], rtol=1e-5
    )


def test_windowed_moment_infinite_end_time_matches_to_absorption():
    """Regression for the scan-2 finding: a windowed moment (start_time>0) with an explicit ``end_time=np.inf`` must
    return the finite to-absorption windowed moment, as ``end_time=None`` does, not crash with a NaN 'ill-conditioned
    rate matrix' error."""
    c = pg.Coalescent(n=3)

    # pre-fix: end_time=np.inf exponentiated over an infinite step in the windowed Van Loan loop and raised
    # ValueError('NaN value encountered when computing moment. This is likely due to an ill-conditioned rate matrix.')
    assert np.isclose(c.moment(k=2, start_time=0.5, end_time=np.inf), c.moment(k=2, start_time=0.5), rtol=1e-5)
