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


def test_flattened_path_taken_for_sfs_mean():
    """The single-population standard SFS mean routes through the flattened accumulation."""
    Settings.flatten_block_counting = True
    coal = pg.Coalescent(n=6)
    with _spy('_accumulate_flattened') as flat:
        _ = coal.sfs.mean
    assert flat.call_count >= 1
