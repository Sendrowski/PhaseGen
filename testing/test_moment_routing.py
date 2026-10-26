"""
Routing tests for the moment-evaluation engine (:class:`phasegen.distributions._moments.MomentEvaluator`).

These pin down *which* path the dispatch takes — flattening vs closed-form vs matrix-exponential, and the
dense/sparse sub-paths — rather than the numeric result (covered by ``test_sparse_dense_equivalence`` /
``test_closed_form_last_epoch``). They guard the refactor that split the engine into a mixin and unified the
dense/sparse Van Loan builder and LU factorization.
"""
import math
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

import phasegen as pg
from phasegen.settings import Settings
from phasegen.distributions import PhaseTypeDistribution, TreeHeightDistribution
from phasegen.distributions._moments import MomentEvaluator
from phasegen.expm import Backend


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


_MIGRATION = dict(pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.3})


@pytest.mark.parametrize('coal, key, max_fill', [
    (pg.Coalescent(n=10), 'lineage_counting', 1.0),
    (pg.Coalescent(n=10), 'block_counting', 1.0),
    (pg.Coalescent(n=10, model=pg.BetaCoalescent(alpha=1.5)), 'block_counting', 1.0),
    (pg.Coalescent(n=8, model=pg.DiracCoalescent(psi=0.7, c=50)), 'block_counting', 1.0),
    (pg.Coalescent(n={'a': 4, 'b': 4}, demography=pg.Demography(**_MIGRATION)), 'lineage_counting', 3),
    (pg.Coalescent(n={'a': 4, 'b': 4}, demography=pg.Demography(**_MIGRATION)), 'block_counting', 6),
    (pg.Coalescent(n={'a': 4, 'b': 4}, demography=pg.Demography(**_MIGRATION)), 'joint_block_counting', 11),
    (pg.Coalescent(n=4, loci=pg.LocusConfig(n=2, recombination_rate=1)), 'lineage_counting', 3),
    (pg.Coalescent(n=4, loci=pg.LocusConfig(n=2, recombination_rate=1)), 'two_locus_block_counting', 4),
], ids=['lineage', 'block', 'block-beta', 'block-dirac', 'lineage-2-demes', 'block-2-demes', 'joint-2-demes',
        'lineage-2-loci', 'two-locus'])
def test_state_spaces_factorize_in_block_triangular_form(coal, key, max_fill):
    """
    The ordering of ``_block_triangular_order`` brings ``-T`` of every state-space type into block upper
    triangular form over its strongly connected components, so the ``NATURAL``-ordered LU fills only inside the
    diagonal blocks: not at all on graded single-deme spaces, and within about twice the measured fill under
    migration or recombination. On the joint space of two demes with ``n=6+6`` the default COLAMD ordering fills
    15 times more and factorizes four orders of magnitude slower.
    """
    import scipy.sparse.csgraph as csg
    import scipy.sparse.linalg as spla

    ss = coal.state_spaces[key]
    t = np.where(~np.asarray(ss.absorbing))[0]
    A = -sp.csr_matrix(ss.S)[t][:, t]

    perm = MomentEvaluator._block_triangular_order(A)
    assert perm is not None and np.array_equal(np.sort(perm), np.arange(len(t)))

    P = A[perm][:, perm].tocoo()
    _, labels = csg.connected_components(abs(P), directed=True, connection='strong')
    below = P.row > P.col
    assert np.all(labels[P.row[below]] == labels[P.col[below]])

    lu = spla.splu(P.tocsc(), permc_spec='NATURAL')
    assert (lu.L.nnz + lu.U.nnz - len(t)) / A.nnz <= max_fill


# ----------------------------------------------------------------------------------------------------------------
# path dispatch (closed-form vs matrix-exponential, dense vs sparse action)
# ----------------------------------------------------------------------------------------------------------------

def test_closed_form_path_taken_when_enabled():
    """With the closed form enabled, the moment to absorption routes through ``_accumulate_closed_form`` (and not
    the sparse-action sub-path)."""
    Settings.closed_form_last_epoch = True
    coal = pg.Coalescent(n=5)
    with _spy('_accumulate_closed_form') as cf, _spy('_action_operator') as action:
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
    """A zero ``expm_action_min_dim`` forces the sparse matrix-exponential action (``_action_operator``)."""
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 0
    coal = pg.Coalescent(n=5)
    with _spy('_action_operator') as action:
        _ = coal.tree_height.mean
    assert action.call_count >= 1


def test_dense_expm_path_taken_above_threshold():
    """A huge ``expm_action_min_dim`` keeps the dense Van Loan exponential (no action path)."""
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 10 ** 12
    coal = pg.Coalescent(n=5)
    with _spy('_action_operator') as action:
        _ = coal.tree_height.mean
    assert action.call_count == 0


@pytest.mark.parametrize('force_sparse', [False, True])
def test_accumulate_restores_input_order_for_unsorted_times(force_sparse):
    """``accumulate`` must return moments aligned to the *input* end-time order, not the internal sorted order.
    Regression for the inverse-permutation bug (``argsort`` instead of ``argsort(argsort(...))``) which mis-attached
    moments for >= 3 unsorted times, with both the dense and the sparse-action exponential of ``_accumulate_windowed``.
    """
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 0 if force_sparse else 10 ** 12

    th = pg.Coalescent(n=5).tree_height
    times = [3.0, 0.5, 2.0, 1.0]  # >= 3 entries and unsorted -> exposes the permutation bug

    acc = th.accumulate(k=1, end_times=times)
    ref = np.array([th.accumulate(k=1, end_times=[t])[0] for t in times])

    np.testing.assert_allclose(acc, ref, rtol=1e-8, atol=1e-10)


def test_flattened_path_taken_for_sfs_mean():
    """The single-population standard SFS mean routes through the flattened rewards and not the block-counting closed
    form."""
    Settings.flatten_block_counting = True
    coal = pg.Coalescent(n=6)
    with _spy('_flattened_weights') as flat, _spy('_accumulate_closed_form') as cf:
        _ = coal.sfs.mean
    assert flat.call_count >= 1
    assert cf.call_count == 0


@pytest.mark.parametrize('end_time', [None, 2.0])
@pytest.mark.parametrize('folded', [False, True])
def test_windowed_sfs_mean_flattens(end_time, folded):
    """A windowed SFS mean (``start_time > 0``) of the standard single-deme coalescent is accumulated on the flattened
    lineage-counting space and agrees with the block-counting evaluation. Regression: the window bypassed the
    flattening, up to 2700 times slower."""
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.2, 1: 2}})

    def mean() -> np.ndarray:
        coal = pg.Coalescent(n=6, demography=demography, start_time=0.5, end_time=end_time)
        return np.asarray((coal.fsfs if folded else coal.sfs).mean.data)

    with _spy('_accumulate_flattened') as flat:
        flattened = mean()
    assert flat.call_count >= 1

    Settings.flatten_block_counting = False
    np.testing.assert_allclose(flattened, mean(), rtol=1e-10)


def test_closed_form_finite_epochs_dense_on_stiff_rates():
    """The finite epochs of the closed form use the dense Van Loan exponential on stiff rates below
    ``expm_action_min_dim`` even where the last-epoch LU is sparse. Regression: a sparse LU forced the sparse action,
    15 to 30 times slower on lineage-counting chains from 256 states on."""
    Settings.closed_form_sparse_min_states = 0
    th = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.2, 1: 2}})).tree_height

    with patch.object(Backend, 'expm_multiply', side_effect=Backend.expm_multiply) as action:
        m = th.moment(2, start_time=0.2)
    assert action.call_count == 0

    # the block-counting space of 17 lineages is not stiff and keeps the sparse action
    Settings.flatten_block_counting = False
    sfs = pg.Coalescent(n=17, demography=th.demography).sfs
    reward = pg.rewards.CombinedReward([sfs.reward, sfs._get_sfs_reward(2)])
    with patch.object(Backend, 'expm_multiply', side_effect=Backend.expm_multiply) as action:
        PhaseTypeDistribution.moment(sfs, k=2, rewards=(reward, reward))
    assert action.call_count > 0

    Settings.expm_action_min_dim = 0
    assert m == pytest.approx(pg.Coalescent(n=10, demography=th.demography).tree_height.moment(2, start_time=0.2),
                              rel=1e-10)


def test_closed_form_finite_epochs_take_action_below_sparse_lu_threshold():
    """The finite epochs of the closed form take the sparse action on non-stiff rates where the last-epoch LU is
    dense. Regression: the dense LU forced the dense Van Loan exponential, 4 to 13 times slower on a 10-epoch
    two-deme SFS mean at 110 to 185 transient states."""
    pairs = [('pop_0', 'pop_1'), ('pop_1', 'pop_0')]
    times = np.linspace(0, 2, 10)
    coal = pg.Coalescent(n={'pop_0': 4, 'pop_1': 3}, demography=pg.Demography(
        pop_sizes={'pop_0': {t: 1 + 0.5 * np.sin(t) for t in times}, 'pop_1': {t: 1.5 for t in times}},
        migration_rates={pr: 0.5 for pr in pairs}))
    assert coal.state_spaces['block_counting'].k < Settings.closed_form_sparse_min_states

    with patch('phasegen.distributions._moments.spla.expm_multiply', side_effect=sp.linalg.expm_multiply) as action:
        mean = coal.sfs.mean.data
    assert action.call_count > 0

    sfs = coal.sfs
    reward = pg.rewards.CombinedReward([sfs.reward, sfs._get_sfs_reward(2)])
    with patch.object(Backend, 'expm_multiply', side_effect=Backend.expm_multiply) as action:
        var = PhaseTypeDistribution.moment(sfs, k=2, rewards=(reward, reward))
    assert action.call_count > 0

    Settings.expm_action_min_dim = 10 ** 9
    with patch('phasegen.distributions._moments._EXPM_COST_RATIO', 0.0):
        dense = pg.Coalescent(n=coal._lineages.lineage_dict, demography=coal.demography)
        np.testing.assert_allclose(mean, dense.sfs.mean.data, rtol=1e-12)
        assert var == pytest.approx(PhaseTypeDistribution.moment(dense.sfs, k=2, rewards=(reward, reward)), rel=1e-12)


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


@pytest.mark.parametrize("closed_form", [True, False], ids=["closed-form", "matrix-exponential"])
def test_infinite_end_time_in_grid_accumulates_until_absorption(closed_form):
    """An infinite end time accumulates until absorption on every path, including a grid that also holds finite
    times and the batched spectrum mean. It previously reached the closed form only as the sole end time, and the
    matrix exponential over an infinite step returned NaN."""
    Settings.closed_form_last_epoch = closed_form

    th = pg.Coalescent(n=2).tree_height
    np.testing.assert_allclose(th.accumulate(1, [1.0, np.inf]), [1 - np.exp(-1), 1.0], rtol=1e-6)
    np.testing.assert_allclose(th.accumulate(2, [np.inf, 1.0])[0], th.var, rtol=1e-6)

    beta = pg.Coalescent(n=3, model=pg.BetaCoalescent(alpha=1.5))
    np.testing.assert_allclose(beta.tree_height.accumulate(1, [1.0, np.inf])[1], beta.tree_height.mean, rtol=1e-6)
    np.testing.assert_allclose(beta.sfs.accumulate(1, [np.inf, 1.0])[0], beta.sfs.mean.data, rtol=1e-6)


def test_explicit_infinite_end_time_without_certain_absorption_in_last_epoch():
    """``moment(end_time=np.inf)`` with a zero start time resolves like ``end_time=None`` when some transient state
    of the last epoch cannot be absorbed (an isolated empty deme), instead of factorizing the singular ``-T`` and
    raising a NaN error."""
    demography = pg.Demography(pop_sizes={'a': {0: 1.0}, 'b': {0: 1.0}})
    th = pg.Coalescent(n={'a': 2, 'b': 0}, demography=demography).tree_height

    np.testing.assert_allclose(th.moment(1, end_time=np.inf), th.mean, rtol=1e-12)
    np.testing.assert_allclose(th.mean, 1.0, rtol=1e-6)


@pytest.mark.parametrize("k", [1, 2])
def test_moment_rejects_invalid_window(k):
    """A reversed window or a negative start time raises instead of returning a negative mean, a zero second
    moment, or silently treating the negative start as 0."""
    th = pg.Coalescent(n=2).tree_height

    with pytest.raises(ValueError, match="End time must be greater than equal start time"):
        th.moment(k, start_time=0.8, end_time=0.2)

    with pytest.raises(ValueError, match="Start time must be greater than or equal to 0"):
        th.moment(k, start_time=-1.0, end_time=0.5)


def test_two_locus_rewards_route_to_two_locus_state_space():
    """
    State-space routing chose only among the joint, lineage-counting and block-counting spaces, so two-locus rewards
    fell through to the block-counting space, which raised NotImplementedError for two loci.
    """
    from phasegen.rewards import TwoLocusSFSReward

    c = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
    m = c.moment(2, [TwoLocusSFSReward(0, 1), TwoLocusSFSReward(1, 1)], center=False)

    assert m == pytest.approx(np.asarray(c.sfs2.mean.data)[1, 1], rel=1e-8)


def test_windowed_mean_accumulates_until_absorption_not_until_the_absorption_estimate():
    """
    ``moment`` replaced an infinite end time by the internal absorption-time estimate whenever the start time was
    positive, closing the window ``[start_time, inf)`` at that estimate. For the Dirac coalescent below the estimate
    fell well below the mean tree height, so the evaluated window covered about a tenth of the intended span and the
    mean total branch length came back as 9.3777e10 against 1.5476e12 from an explicit end time far beyond
    absorption, a factor of 16.5.
    """
    from phasegen.distributions.phase_type import TreeHeightDistribution

    model = pg.DiracCoalescent(psi=0.5, c=1.0)
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1e6}})
    start_time = 5e11

    def total_branch_length(**kwargs) -> float:
        return pg.Coalescent(n=4, model=model, demography=demography, **kwargs).total_branch_length.mean

    windowed = total_branch_length(start_time=start_time)
    reference = total_branch_length(end_time=1e14) - total_branch_length(end_time=start_time)

    assert windowed == pytest.approx(reference, rel=1e-9)

    # the estimate bounds only where the window may open, not how far the accumulation runs
    with patch.object(TreeHeightDistribution, '_get_absorption_time', lambda self: start_time):
        assert total_branch_length(start_time=start_time) == pytest.approx(windowed, rel=1e-12)


@pytest.mark.parametrize("model", [pg.StandardCoalescent(), pg.BetaCoalescent(alpha=1.5)], ids=["standard", "beta"])
def test_accumulate_passes_start_time_through(model):
    """Coalescent.accumulate, SFSDistribution.accumulate and SFSDistribution.get_accumulation accumulate from
    ``start_time`` as PhaseTypeDistribution.accumulate does. Regression: they had no ``start_time`` parameter. The
    Beta coalescent takes the batched SFS mean accumulation and the standard coalescent the per-bin one."""
    coal = pg.Coalescent(n=4, model=model)
    start, end = 0.3, 1.5

    assert coal.accumulate(1, [end], start_time=start)[0] == pytest.approx(
        coal.moment(1, start_time=start, end_time=end), rel=1e-8)
    assert coal.accumulate(2, [end], start_time=start)[0] == pytest.approx(
        coal.moment(2, start_time=start, end_time=end), rel=1e-8)

    np.testing.assert_allclose(coal.sfs.accumulate(1, [end], start_time=start)[0],
                               coal.sfs.moment(1, start_time=start, end_time=end).data, rtol=1e-8, atol=1e-14)
    assert coal.sfs.get_accumulation(2, 1, end, start_time=start) == pytest.approx(
        coal.sfs.moment(2, start_time=start, end_time=end).data[1], rel=1e-8)


def test_batched_spectrum_means_validate_the_window():
    """The batched k = 1 means of the SFS and the joint SFS reject a negative start time and a start time beyond
    almost sure absorption, as every other moment path does. Regression: they returned the Beta SFS for a negative
    start and round-off noise for an empty window."""
    beta = pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=1.5))
    mig = pg.Demography(pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
                        migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0})
    jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=mig).jsfs

    for dist in (beta.sfs, jsfs):
        with pytest.raises(ValueError, match="Start time must be greater than or equal to 0"):
            dist.moment(1, start_time=-1.0)

        with pytest.raises(ValueError, match="beyond the time of almost sure absorption"):
            dist.moment(1, start_time=1e6)

        with pytest.raises(ValueError, match="End time must be greater than equal start time"):
            dist.moment(1, start_time=1.0, end_time=0.5)


def test_identical_rewards_accumulate_one_ordering():
    """A moment of identical rewards accumulates a single ordering, all k! orderings being the same. Regression: the
    ninth moment accumulated 9! identical orderings and took 0.97 s instead of 1 ms."""
    th = pg.Coalescent(n=3).tree_height

    with _spy('_accumulate') as acc:
        m = th.moment(7, center=False)

    assert acc.call_count == 1
    assert m == pytest.approx(pg.Coalescent(n=3).tree_height.moment(7, center=False, permute=False), rel=1e-14)


@pytest.mark.parametrize('start_time', [20.0, 30.0])
def test_windowed_mean_near_absorption_is_accurate(start_time):
    """The mean of the tree height of two lineages over ``[start_time, inf)`` is ``exp(-start_time)`` to full
    relative precision. Regression: it was the difference of two cumulative means, off by 4.9e-3 relative at 30."""
    th = pg.Coalescent(n=2).tree_height

    assert th.moment(1, start_time=start_time) == pytest.approx(np.exp(-start_time), rel=1e-10)


@pytest.mark.parametrize('force_sparse', [False, True])
def test_accumulate_passes_nan_end_time_through(caplog, force_sparse):
    """A NaN end time gives a NaN moment without a warning and leaves the other end times unchanged. Regression: it
    logged a warning blaming an ill-conditioned rate matrix."""
    Settings.closed_form_last_epoch = False
    Settings.expm_action_min_dim = 0 if force_sparse else 10 ** 12
    th = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}})).tree_height

    with caplog.at_level('WARNING'):
        out = th.accumulate(k=2, end_times=[1.0, np.nan, 2.0])

    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], th.accumulate(k=2, end_times=[1.0, 2.0]), rtol=1e-12)
    assert not [r for r in caplog.records if r.levelname == 'WARNING']


@pytest.mark.parametrize('rewards, error', [
    (pg.TreeHeightReward(), ValueError),
    ([1.0], TypeError),
    ([pg.TreeHeightReward(), 'a'], TypeError),
    ([pg.TreeHeightReward()], ValueError),
])
def test_phase_type_moment_validates_rewards(rewards, error):
    """
    PhaseTypeDistribution.moment raises ValueError for a single bare reward or a wrong number of rewards, and
    TypeError for an entry that is not a reward, as Coalescent.moment does. A bare reward and a non-reward entry
    raised AttributeError or a TypeError about len() before the rewards were validated.
    """
    with pytest.raises(error, match='reward'):
        pg.Coalescent(n=4).tree_height.moment(2, rewards=rewards)


@pytest.mark.parametrize('k', [2, 3, 4])
def test_central_moments_of_one_reward_from_one_closed_form(k):
    """A central moment of a single reward takes its lower raw moments from the closed form of the highest order, and
    agrees with the centering by separately evaluated raw moments."""
    coal = pg.Coalescent(n={'a': 3, 'b': 2}, demography=pg.Demography(
        pop_sizes={'a': {0: 1, 0.5: 0.3}, 'b': {0: 0.7}}, migration_rates={('a', 'b'): 0.3, ('b', 'a'): 0.6}))
    dist = coal.total_branch_length
    raw = [float(dist.moment(i, center=False)) if i else 1.0 for i in range(k + 1)]
    expected = sum(math.comb(k, i) * raw[i] * (-raw[1]) ** (k - i) for i in range(k + 1))

    assert dist._powers_apply(k, (dist.reward,) * k, (np.inf,))
    np.testing.assert_allclose(dist._raw_powers(k, dist.reward, 0.0), raw, rtol=1e-12)
    assert float(dist.moment(k)) == pytest.approx(expected, rel=1e-10)


def test_closed_forms_of_all_orders_share_the_last_epoch_factorization(monkeypatch):
    """The closed forms of the orders of a moment, whose epoch lists coincide, factorize the last epoch once.
    Regression: the factorization was kept per order, so a third cross-moment factorized the same block thrice."""
    import traceback

    calls = []
    lu_solver = MomentEvaluator._lu_solver
    monkeypatch.setattr(MomentEvaluator, '_lu_solver', staticmethod(lambda *a, **k: calls.append(
        '_last_epoch_solver' in traceback.format_stack()[-2]) or lu_solver(*a, **k)))

    coal = pg.Coalescent(n={'pop_0': 3, 'pop_1': 3}, demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1, 0.5: 2}, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 0.5,
                                                                          ('pop_1', 'pop_0'): 0.5}))
    tbl = pg.TotalBranchLengthReward()
    coal.moment(3, rewards=[pg.TreeHeightReward(), tbl, tbl])

    assert sum(calls) == 1


# ----------------------------------------------------------------------------------------------------------------
# dense exponential against sparse action: one cost model for every caller
# ----------------------------------------------------------------------------------------------------------------

_THREE_EPOCHS = {0: 1.0, 0.4: 0.3, 1.2: 2.0}


def _one_deme(n, model=None, **kwargs) -> pg.Coalescent:
    """One deme with three epochs, on the block-counting state space."""
    Settings.flatten_block_counting = False
    return pg.Coalescent(n=n, model=model, demography=pg.Demography(pop_sizes={'pop_0': _THREE_EPOCHS}), **kwargs)


def _bin_variance(coal: pg.Coalescent) -> float:
    """The second moment of the first SFS bin, by the per-bin accumulation."""
    sfs = coal.sfs
    reward = pg.rewards.CombinedReward([sfs.reward, sfs._bins()[0][1]])
    return PhaseTypeDistribution.moment(sfs, k=2, rewards=(reward, reward))


def _dispatch(read, spectra_only: bool = False) -> list:
    """The shapes and the outcomes of every dense-against-action decision taken while evaluating ``read``, without
    those of the tree heights propagated alongside a spectrum if ``spectra_only``."""
    seen = []
    original = MomentEvaluator._dense_expm_is_faster

    def spy(self, *args, **kwargs):
        dense = original(self, *args, **kwargs)
        if not (spectra_only and isinstance(self, TreeHeightDistribution)):
            seen.append((args[:3], dense))
        return dense

    with patch.object(MomentEvaluator, '_dense_expm_is_faster', spy):
        read()

    return seen


_BETA = pg.BetaCoalescent(alpha=1.5)


@pytest.mark.parametrize('read, dense, spectrum', [
    (lambda: _one_deme(10).tree_height.var, True, False),
    (lambda: _bin_variance(_one_deme(15, model=_BETA)), False, True),
    (lambda: _one_deme(10, end_time=1.5).tree_height.var, True, False),
    (lambda: _bin_variance(_one_deme(15, model=_BETA, end_time=1.5)), False, True),
    (lambda: _one_deme(8, model=_BETA).sfs.mean, True, True),
    (lambda: _one_deme(15, model=_BETA).sfs.mean, False, True),
    (lambda: _one_deme(8, model=_BETA).sfs._mean_occupation_grid(np.linspace(0.1, 3, 30)), True, True),
    (lambda: _one_deme(15, model=_BETA).sfs._mean_occupation_grid(np.linspace(0.1, 3, 30)), False, True),
], ids=['closed form 30', 'closed form 528', 'window 30', 'window 528', 'occupation 42', 'occupation 350',
        'time grid 44', 'time grid 352'])
def test_every_caller_takes_the_dense_exponential_on_small_spaces_and_the_action_on_large(read, dense, spectrum):
    """The closed-form and windowed moments, the occupation times and the occupation on a time grid each decide
    through ``_dense_expm_is_faster``, dense on spaces of a few tens of states and the action on hundreds. The tree
    height a spectrum propagates alongside decides on its own space."""
    seen = _dispatch(read, spectra_only=spectrum)

    assert seen and all(outcome is dense for _, outcome in seen)


def test_dense_expm_rule_boundaries():
    """Dense exactly where the summed step norm and the fixed costs reach the cost of the dense exponentials, that of a
    pass recorded for its gradient scaled up, never at or above ``Settings.expm_action_min_dim`` in their size or
    without steps, and dense without reading a generator when the fixed costs alone exceed it."""
    from phasegen.distributions import _moments

    sfs = _one_deme(10).sfs
    ss = sfs.state_space
    epochs = sfs._get_epochs_until_unbounded(2)
    steps = sfs._closed_form_steps(epochs, 0.0)
    d_action, d_dense, count = 9 * ss.k, 3 * ss.k, 4

    norm = 0.0
    for epoch, tau in steps[:-1]:
        ss.update_epoch(epoch)
        norm += float(abs(ss.S).sum(axis=0).max()) * tau
    ratio = ((norm + 2 * _moments._ACTION_STEP_NORM_OFFSET) * (d_action + _moments._ACTION_DIM_OFFSET) /
             (count * 2 * d_dense ** 3))

    with patch.object(_moments, '_EXPM_COST_RATIO', ratio * (1 + 1e-9)):
        assert not sfs._dense_expm_is_faster(d_action, d_dense, count, steps)
        assert sfs._dense_expm_is_faster(d_action, d_dense, count, steps, n_dense=1)
    with patch.object(_moments, '_EXPM_COST_RATIO', ratio * (1 - 1e-9)):
        assert sfs._dense_expm_is_faster(d_action, d_dense, count, steps)
        assert not sfs._dense_expm_is_faster(d_action, d_dense, count, steps, taped=True)
    with patch.object(_moments, '_EXPM_COST_RATIO', ratio / _moments._TAPED_DENSE_COST * (1 - 1e-9)):
        assert sfs._dense_expm_is_faster(d_action, d_dense, count, steps, taped=True)
    with patch.object(_moments, '_EXPM_COST_RATIO', ratio * (1 - 1e-9)):
        Settings.expm_action_min_dim = d_dense
        assert not sfs._dense_expm_is_faster(d_action, d_dense, count, steps)
        Settings.expm_action_min_dim = d_dense + 1
        assert sfs._dense_expm_is_faster(d_action, d_dense, count, steps)
        assert not sfs._dense_expm_is_faster(d_action, d_dense, count, [(epochs[-1], 0.0)])

    # the fixed costs alone decide without reading a generator
    with patch.object(_moments, '_EXPM_COST_RATIO', 0.0), \
            patch.object(type(ss), 'update_epoch', side_effect=AssertionError('generator read')):
        assert sfs._dense_expm_is_faster(d_action, d_dense, count, steps)


def test_grid_steps_count_one_action_per_point_and_one_exponential_per_step_length():
    """A propagation from ``start_time`` through an equispaced grid takes one action per grid point and per remainder
    to an epoch end, and one dense exponential per epoch and step length."""
    sfs = _one_deme(4).sfs

    # within the first epoch [0, 0.4): four equal steps of 0.05 from the start
    steps, n_dense = sfs._grid_steps(0.1, np.array([0.15, 0.2, 0.25, 0.3]))
    assert [tau for _, tau in steps] == pytest.approx([0.05] * 4)
    assert n_dense == 1

    # across the boundary at 0.4: a step of 0.3, the remainder of 0.1, then steps of 0.2 and 0.3 in the second epoch
    steps, n_dense = sfs._grid_steps(0.0, np.array([0.3, 0.6, 0.9]))
    assert [e.start_time for e, _ in steps] == [0, 0, 0.4, 0.4]
    assert [tau for _, tau in steps] == pytest.approx([0.3, 0.1, 0.2, 0.3])
    assert n_dense == 4

    # no end time after the start
    assert sfs._grid_steps(0.5, np.array([0.2, 0.5])) == ([], 0)


def _transform_dispatch(read) -> list:
    """The dimensions and outcomes of every dense-against-action decision of the Laplace transform taken while
    evaluating ``read``."""
    from phasegen.distributions import reward

    seen = []
    original = reward._takes_action

    def spy(dim, batch, n_epochs, epochs):
        action = original(dim, batch, n_epochs, epochs)
        seen.append((dim, batch, action))
        return action

    with patch.object(reward, '_takes_action', spy):
        read()

    return seen


@pytest.mark.parametrize('read, action', [
    (lambda: _one_deme(5).distribution(pg.TotalBranchLengthReward()).cdf(np.linspace(0.1, 6, 5)), False),
    (lambda: _one_deme(20).distribution(pg.UnfoldedSFSReward(1)).cdf(np.array([0.5, 1.0])), True),
    (lambda: _one_deme(5).joint(pg.TreeHeightReward(), pg.TotalBranchLengthReward()).lst_taylor(-0.5j), False),
    (lambda: _one_deme(14).sfs.joint(1, 2)._lst_taylor_batch(-1j * np.linspace(0.1, 30, 64), 'a', 2), True),
    (lambda: _one_deme(4).joint(pg.TreeHeightReward(), pg.TotalBranchLengthReward()).conditional('a', 1.0).mean,
     False),
], ids=['transform 5', 'transform 627', 'taylor 15', 'taylor 405', 'jump heights'])
def test_transform_takes_dense_exponentials_on_small_spaces_and_the_action_on_large(read, action):
    """The batched transform, its Taylor coefficients and the jump heights of the conditionals decide through the
    cost model, with dense Pade exponentials on spaces of a few states and the Taylor action on batches of hundreds,
    a single argument such as an atom being left to the dense exponentials there."""
    seen = [outcome for _, batch, outcome in _transform_dispatch(read) if batch > 1 or not action]

    assert seen and all(outcome is action for outcome in seen)


def test_transform_rule_boundaries():
    """The transform takes the action exactly where the bound of the cost model for its batch exceeds the summed Taylor
    step norm, always from ``Settings.expm_action_min_dim`` on, and the dense exponentials without reading a norm
    where the fixed costs of the actions alone exceed them."""
    from phasegen.distributions import reward, _moments

    dim, batch = 40, 8
    rng = np.random.default_rng(0)
    diag = -rng.uniform(1, 3, (batch, dim)) + 0j
    off_norm = rng.uniform(0, 2, dim)
    epochs = lambda: [[(diag[:5], off_norm), (diag[5:], off_norm)], [(2 * diag, 2 * off_norm)]]

    taylor = reward._TaylorAction(diag, off_norm).norm + reward._TaylorAction(2 * diag, 2 * off_norm).norm

    # the ratio at which the bound equals the norm, by linearity of the bound in the ratio
    unit = MomentEvaluator._dense_expm_norm_bound(dim, dim, 1, 2, 2, batch=batch)
    with patch.object(_moments, '_EXPM_COST_RATIO', 0.0):
        offset = MomentEvaluator._dense_expm_norm_bound(dim, dim, 1, 2, 2, batch=batch)
    ratio = _moments._EXPM_COST_RATIO * (taylor - offset) / (unit - offset)

    with patch.object(_moments, '_EXPM_COST_RATIO', ratio * (1 + 1e-9)):
        assert reward._takes_action(dim, batch, 2, epochs())
    with patch.object(_moments, '_EXPM_COST_RATIO', ratio * (1 - 1e-9)):
        assert not reward._takes_action(dim, batch, 2, epochs())
        assert not reward._takes_action(dim, batch, 0, [])

    def unread():
        raise AssertionError('norm read')
        yield

    # the fixed costs of the actions alone exceed the dense exponentials, without reading a norm
    with patch.object(_moments, '_EXPM_COST_RATIO', 0.0):
        assert not reward._takes_action(dim, batch, 2, unread())

        Settings.expm_action_min_dim = dim
        assert reward._takes_action(dim, batch, 2, epochs())
        Settings.expm_action_min_dim = dim + 1
        assert not reward._takes_action(dim, batch, 2, epochs())


# ----------------------------------------------------------------------------------------------------------------
# the tree-height distribution functions decide by the same cost model
# ----------------------------------------------------------------------------------------------------------------

def _two_demes(n: int, sizes=1.0, migration: float = 1.0) -> pg.Coalescent:
    """Two demes of ``n`` lineages each, the first of the given sizes."""
    return pg.Coalescent(n={'a': n, 'b': n}, demography=pg.Demography(
        pop_sizes={'a': sizes, 'b': 1.0}, migration_rates={('a', 'b'): migration, ('b', 'a'): migration}))


def _forced(dense: bool):
    """Force every dense-against-action decision to ``dense``."""
    return patch.object(MomentEvaluator, '_dense_expm_is_faster', lambda self, *args, **kwargs: dense)


@pytest.mark.parametrize('end, dense', [(1e-3, False), (None, True)], ids=['short horizon', 'to absorption'])
def test_tree_height_sweep_decides_once_by_the_cost_model(end, dense):
    """On 230 states, five equispaced times within a short horizon take one interval action, while a hundred up to
    the time of almost sure absorption reuse one dense exponential, as one decision for the whole sweep."""
    th = _two_demes(10).tree_height
    t = np.linspace(0, th.t_max if end is None else end, 5 if end else 100)

    seen = _dispatch(lambda: th._sweep(t))

    assert seen == [((230, 230, 1), dense)]


def test_tree_height_sweep_values_do_not_depend_on_the_decision():
    """The CDF and density of the sweep agree between the dense exponentials and the action, across epochs and on
    an irregular grid with an infinite time."""
    th = _two_demes(3, sizes=_THREE_EPOCHS).tree_height
    t = np.concatenate([np.sort(np.random.default_rng(0).uniform(0, th.t_max, 40)), [np.inf]])

    with _forced(True):
        dense = th._sweep(t)
    with _forced(False):
        action = th._sweep(t)

    np.testing.assert_allclose(action[0], dense[0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(action[1], dense[1], rtol=0, atol=1e-13 * dense[1].max())


def test_sweep_steps_count_one_action_per_run_and_one_exponential_per_step_length():
    """A sweep takes one interval action per run of several equal steps and one action per other step and per
    remainder to an epoch end, one dense exponential per epoch and step length, and neither for an infinite time."""
    th = _one_deme(4).tree_height

    # [0, 0.4): three steps of 0.05 and the remainder 0.25; [0.4, 1.2): two steps of 0.3 and the remainder 0.2
    plan = th._sweep_plan(np.array([0.05, 0.1, 0.15, 0.7, 1.0, np.inf]))
    steps, n_dense = th._sweep_steps(plan)

    assert [e.start_time for e, _ in steps] == [0, 0, 0.4, 0.4]
    assert [tau for _, tau in steps] == pytest.approx([0.15, 0.25, 0.6, 0.2])
    assert n_dense == 4
    assert [len(runs) for _, runs, _ in plan] == [1, 1, 1]


@pytest.mark.parametrize('read, k, dense', [
    (lambda: pg.Coalescent(n=10).tree_height, 10, True),
    (lambda: _two_demes(10).tree_height, 230, True),
], ids=['10 states', '230 states'])
def test_quantile_grid_decides_once_by_the_cost_model(read, k, dense):
    """The quantile grid weighs the dense propagators of its expected segment widths, per epoch below ``t_max``,
    against an action per expected advance, over twice the span of the grid."""
    th = read()
    t_max = th.t_max
    seen = []
    original = MomentEvaluator._dense_expm_is_faster

    def spy(self, *args, **kwargs):
        seen.append(args)
        return original(self, *args, **kwargs)

    with patch.object(MomentEvaluator, '_dense_expm_is_faster', spy):
        th.quantile(0.5)

    from phasegen.distributions.phase_type import _ExpmFunction
    d_action, d_dense, count, steps, n_dense = seen[-1]
    assert (d_action, d_dense, count) == (k, k, 1)
    assert len(steps) == _ExpmFunction._expected_advances
    assert sum(tau for _, tau in steps) == pytest.approx(2 * t_max)
    assert n_dense == _ExpmFunction._expected_widths
    assert original(th, *seen[-1]) is dense


def test_quantile_grid_values_do_not_depend_on_the_decision():
    """The quantiles read off the grid agree between the dense propagators and the action to the tolerance of the
    grid."""
    q = np.array([0.01, 0.5, 0.99])
    with _forced(True):
        dense = _two_demes(3, sizes=_THREE_EPOCHS).tree_height.quantile(q)
    with _forced(False):
        action = _two_demes(3, sizes=_THREE_EPOCHS).tree_height.quantile(q)

    np.testing.assert_allclose(action, dense, rtol=1e-8)
