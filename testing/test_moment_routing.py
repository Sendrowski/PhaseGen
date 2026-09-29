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
    sfs = pg.Coalescent(n=17, demography=th.demography).sfs
    reward = pg.rewards.CombinedReward([sfs.reward, sfs._get_sfs_reward(2)])
    with patch.object(Backend, 'expm_multiply', side_effect=Backend.expm_multiply) as action:
        PhaseTypeDistribution.moment(sfs, k=2, rewards=(reward, reward))
    assert action.call_count > 0

    Settings.expm_action_min_dim = 0
    assert m == pytest.approx(pg.Coalescent(n=10, demography=th.demography).tree_height.moment(2, start_time=0.2),
                              rel=1e-10)


@pytest.mark.parametrize('force_sparse', [False, True])
@pytest.mark.parametrize('folded', [False, True])
def test_multi_epoch_sfs_cov_stacks_bins(force_sparse, folded):
    """The multi-epoch SFS covariance evaluates one closed form per second bin, all first bins stacked in one extended
    vector, and matches the per-pair cross-moments. Regression: one closed form per ordered bin pair."""
    Settings.closed_form_sparse_min_states = 0 if force_sparse else 10 ** 9
    Settings.expm_action_min_dim = 0 if force_sparse else 10 ** 9
    coal = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 0.2, 1: 2}}))
    sfs = coal.fsfs if folded else coal.sfs
    indices = sfs._get_indices()

    with _spy('_accumulate_closed_form') as cf:
        cov = np.asarray(sfs.cov.data)
    assert cf.call_count == (len(indices) if force_sparse else len(indices) * (len(indices) + 1))

    mean = np.asarray(sfs.mean.data)
    for i in indices:
        for j in indices:
            rewards = tuple(pg.rewards.CombinedReward([sfs.reward, sfs._get_sfs_reward(b)]) for b in (i, j))
            m_ij = PhaseTypeDistribution.moment(sfs, k=2, rewards=rewards, center=False, permute=False)
            m_ji = PhaseTypeDistribution.moment(sfs, k=2, rewards=rewards[::-1], center=False, permute=False)
            assert cov[i, j] == pytest.approx((m_ij + m_ji) / 2 - mean[i] * mean[j], rel=1e-10, abs=1e-12)


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
