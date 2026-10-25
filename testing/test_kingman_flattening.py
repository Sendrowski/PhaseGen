"""
Second moments of the site-frequency spectrum flattened onto the lineage-counting state space for one population and
one locus under the standard coalescent (``MomentEvaluator._flattening_applies`` with ``k = 2``).
"""
from unittest.mock import patch

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import PhaseTypeDistribution
from phasegen.distributions._kingman import KingmanBlockKernel
from phasegen.rewards import UnfoldedSFSReward as U, FoldedSFSReward as F, CombinedReward, TotalBranchLengthReward, \
    UnitReward, TreeHeightReward
from phasegen.settings import Settings

DEMOGRAPHIES = {
    'constant': {0: 1.0},
    'three_epochs': {0: 1.0, 0.5: 0.3, 2: 2.0},
    'stiff': {0: 1e-3, 0.01: 1e4, 5e4: 1e-2},
}


def _coal(n: int, sizes: str = 'three_epochs', **kwargs) -> pg.Coalescent:
    return pg.Coalescent(n, demography=pg.Demography(pop_sizes={'pop_0': DEMOGRAPHIES[sizes]}), **kwargs)


def _unflattened(f):
    """Evaluate ``f`` with flattening disabled."""
    Settings.flatten_block_counting = False
    try:
        return f()
    finally:
        Settings.flatten_block_counting = True


def _assert_close(actual, desired, rtol=1e-11):
    """Agreement relative to the largest entry, as entries of a covariance cancel."""
    actual, desired = np.asarray(actual, dtype=float), np.asarray(desired, dtype=float)
    np.testing.assert_allclose(actual, desired, rtol=0, atol=rtol * max(np.abs(desired).max(), 1e-300))


def _fu_cov(n: int) -> np.ndarray:
    """The covariance of the branch lengths of the unfolded SFS under a constant size (Fu, 1995), in units where a
    pair coalesces at rate 1, so that :math:`\\operatorname{Cov}[L_i, L_j] = 4 \\sigma_{ij}`."""
    a = lambda m: sum(1 / i for i in range(1, m))
    beta = lambda i: 2 * n * (a(n + 1) - a(i)) / ((n - i + 1) * (n - i)) - 2 / (n - i)
    sigma = np.zeros((n - 1, n - 1))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                if 2 * i < n:
                    s = beta(i + 1)
                elif 2 * i == n:
                    s = 2 * (a(n) - a(i)) / (n - i) - 1 / i ** 2
                else:
                    s = beta(i) - 1 / i ** 2
            else:
                hi, lo = max(i, j), min(i, j)
                if hi + lo < n:
                    s = (beta(hi + 1) - beta(hi)) / 2
                elif hi + lo == n:
                    s = (a(n) - a(hi)) / (n - hi) + (a(n) - a(lo)) / (n - lo) - (beta(hi) + beta(lo + 1)) / 2 - \
                        1 / (hi * lo)
                else:
                    s = (beta(lo) - beta(lo + 1)) / 2 - 1 / (hi * lo)
            sigma[i - 1, j - 1] = s
    return 4 * sigma


@pytest.mark.parametrize('n', [2, 3, 10, 40])
def test_constant_size_covariance_matches_fu(n):
    """The flattened covariance of the unfolded SFS under a constant size equals the closed form of Fu (1995)."""
    _assert_close(_coal(n, 'constant').sfs.cov.data[1:n, 1:n], _fu_cov(n), rtol=1e-12)


def test_literal_values():
    """Covariances of a small case against stored values of the block-counting evaluation."""
    coal = _coal(4)
    _assert_close(coal.sfs.cov.data[1:4, 1:4], [
        [0.4730675462864411, -0.0652984048778864, 0.1344052662526054],
        [-0.0652984048778864, 0.4171004211904813, -0.07684926261119007],
        [0.1344052662526054, -0.07684926261119007, 0.16313682025237042]
    ], rtol=1e-13)
    _assert_close(coal.fsfs.cov.data[1:3, 1:3], [
        [0.9050148990440223, -0.1421476674890766],
        [-0.1421476674890766, 0.4171004211904813]
    ], rtol=1e-13)
    _assert_close(_coal(4, start_time=0.2, end_time=1.5).sfs.cov.data[1:4, 1:4], [
        [0.269929364034348, -0.03467549550684379, 0.06233746916270477],
        [-0.03467549550684379, 0.23643218915838363, -0.06343896136086177],
        [0.06233746916270477, -0.06343896136086177, 0.08248634015514864]
    ], rtol=1e-13)


@pytest.mark.parametrize('n', [2, 3, 7])
@pytest.mark.parametrize('sizes', list(DEMOGRAPHIES))
@pytest.mark.parametrize('window', [(0, None), (0.3, None), (0, 1.0), (0.4, 2.5)])
def test_spectrum_moments_match_unflattened(n, sizes, window):
    """The covariance, variance, correlation and second moments of the unfolded and folded spectra agree with the
    block-counting evaluation, also within a window."""

    def moments():
        coal = _coal(n, sizes, start_time=window[0], end_time=window[1])
        return [coal.sfs.cov.data, coal.sfs.var.data, coal.sfs.corr.data, coal.sfs.moment(2, center=False).data,
                coal.fsfs.cov.data, coal.fsfs.var.data, coal.fsfs.moment(2).data]

    for flattened, reference in zip(moments(), _unflattened(moments)):
        _assert_close(flattened, reference)


@pytest.mark.parametrize('sizes', ['constant', 'three_epochs'])
def test_cross_moments_match_unflattened(sizes):
    """Cross-moments of two SFS bins of any kind, ordered or averaged over their orderings, central or raw, and
    accumulated to several end times agree with the block-counting evaluation."""

    def moments():
        coal = _coal(6, sizes)
        return [
            coal.moment(2, rewards=(U(2), U(3))),
            coal.moment(2, rewards=(U(2), U(3)), center=False, permute=False),
            coal.moment(2, rewards=(U(3), U(2)), center=False, permute=False),
            coal.moment(2, rewards=(F(1), U(4)), start_time=0.2),
            coal.sfs.get_cov(1, 5),
            coal.fsfs.get_corr(1, 3),
            coal.sfs.accumulate(2, end_times=[3.0, 0.2, np.inf, 1.0]),
            coal.sfs.moment(2, end_time=1.5).data,
            coal.sfs.bin(2).var,
            coal.fsfs.bin(1).var,
            PhaseTypeDistribution.moment(coal.sfs, 2, rewards=(CombinedReward([UnitReward(), U(1)]), U(2)),
                                         start_time=0.1, end_time=2.0),
        ]

    for flattened, reference in zip(moments(), _unflattened(moments)):
        _assert_close(flattened, reference)


@pytest.mark.parametrize('setting, value', [('expm_action_min_dim', 0), ('closed_form_last_epoch', False),
                                            ('regularize', False)])
def test_settings_match_unflattened(setting, value):
    """The sparse action, the integration to the time of almost sure absorption and the unbalanced exponentials agree
    with the block-counting evaluation under the same setting."""
    setattr(Settings, setting, value)

    def moments():
        coal = _coal(5, start_time=0.1)
        return [coal.sfs.cov.data, coal.fsfs.cov.data]

    for flattened, reference in zip(moments(), _unflattened(moments)):
        _assert_close(flattened, reference, rtol=1e-9)


def test_gradient_matches_unflattened():
    """The value and the exact Jacobian of a statistic reading the covariance and the variance of the spectrum are
    those of the block-counting evaluation."""

    def compute():
        return pg.Gradient(
            coal=lambda t, Ne: pg.Coalescent(5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, t: Ne, 2.0: 1.5}})),
            statistic=lambda c: np.concatenate([c.sfs.cov.data.ravel(), c.fsfs.var.data])
        ).compute(dict(t=0.3, Ne=0.4))

    (value, jacobian), (value_ref, jacobian_ref) = compute(), _unflattened(compute)
    _assert_close(value, value_ref)
    _assert_close(jacobian, jacobian_ref, rtol=1e-10)


def test_routes_through_lineage_counting_space():
    """The covariance, variance and second moments of the spectrum and the cross-moments of SFS bins read the level
    occupation and never build the block-counting state space."""
    coal = _coal(12)

    with patch.object(PhaseTypeDistribution, '_level_cross_moments', autospec=True,
                      side_effect=PhaseTypeDistribution._level_cross_moments) as spy:
        coal.sfs.cov, coal.sfs.var, coal.fsfs.corr, coal.sfs.moment(2), coal.moment(2, rewards=(U(1), F(3)))

    assert spy.call_count >= 1
    assert 'states' not in coal.state_spaces['block_counting'].__dict__


def test_flattening_predicate():
    """The second moment is flattened for two SFS bins with polymorphic blocks of the standard coalescent on one
    population and one locus only."""
    sfs = _coal(6).sfs
    assert sfs._flattening_applies(2, (U(1), U(5)))
    assert sfs._flattening_applies(2, (F(3), CombinedReward([UnitReward(), U(2)])))

    assert not sfs._flattening_applies(2)
    assert not sfs._flattening_applies(2, (U(1), TotalBranchLengthReward()))
    assert not sfs._flattening_applies(2, (U(0), U(1)))
    assert not sfs._flattening_applies(2, (F(4), U(1)))
    assert not sfs._flattening_applies(2, (CombinedReward([U(1), U(2)]), U(1)))
    assert not sfs._flattening_applies(2, (CombinedReward([TreeHeightReward(), U(1)]), U(1)))
    assert not sfs._flattening_applies(3, (U(1), U(2), U(3)))

    Settings.flatten_block_counting = False
    assert not sfs._flattening_applies(2, (U(1), U(2)))
    Settings.flatten_block_counting = True

    for coal in [
        pg.Coalescent(6, model=pg.BetaCoalescent(alpha=1.5)),
        pg.Coalescent(6, model=pg.DiracCoalescent(psi=0.5, c=1.0)),
        pg.Coalescent(n={'pop_0': 3, 'pop_1': 3}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1})),
    ]:
        assert not coal.sfs._flattening_applies(2, (U(1), U(2)))

    two_loci = pg.Coalescent(n=4, loci=2, recombination_rate=1.0)
    assert not two_loci.sfs2._flattening_applies(2, (U(1), U(2)))


def test_kernel_matches_enumerated_block_moments():
    """The contraction of an occupation with unit mass on one level equals the expected product of the block counts
    of that level, enumerated over the block-counting state space."""
    n = 5
    space = pg.Coalescent(n).state_spaces['block_counting']
    kernel = KingmanBlockKernel(n)

    # the level k of each state, and the probability of each state given its level
    levels = space.lineages.sum(axis=(1, 2, 3))
    probs = space._state_probs
    counts = space.lineages[:, 0, 0, :n - 1]

    for k in range(2, n + 1):
        M = np.zeros((n - 1, n - 1))
        M[n - k, n - k] = 1.0
        X = kernel.contract(M)
        at = levels == k
        expected = np.einsum('s,si,sj->ij', probs[at], counts[at], counts[at])
        np.testing.assert_allclose(X, expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize('n', [2, 3, 10, 30])
def test_lineage_counting_states_decrease_in_lineages(n):
    """
    The flattened cross-moments take the level matrix as upper triangular, which needs the single-deme
    lineage-counting states ordered by decreasing number of lineages, so that the generator is upper triangular.
    """
    ss = _coal(n).state_spaces['lineage_counting']
    counts = ss.lineages.sum(axis=(1, 2, 3))
    S = ss.S.toarray() if hasattr(ss.S, 'toarray') else np.asarray(ss.S)

    assert np.all(np.diff(counts) < 0)
    assert not np.tril(S, -1).any()
