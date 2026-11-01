"""
Tests of the second moments of spectra from one forward pass over all bins (``MomentEvaluator._bin_second_moments``):
agreement with the cross-moments of each pair of bins in every regime the pass reaches, values pinned against the
per-pair evaluation, its routing, and the boundaries of its exponential dispatch.
"""
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

import phasegen as pg
from phasegen.distributions import PhaseTypeDistribution, _moments
from phasegen.expm import Backend, _GrowthBound
from phasegen.rewards import CombinedReward
from phasegen.settings import Settings

_THREE = {0: 1.0, 0.4: 0.3, 1.2: 2.0}


@pytest.fixture(autouse=True)
def _unflattened():
    """The single-deme spectra of the standard coalescent take the forward pass on the block-counting state space.
    The conftest restores the setting."""
    Settings.flatten_block_counting = False


def _single(n=5, sizes=None, model=None, start_time=0, end_time=None) -> pg.Coalescent:
    """One deme with three epochs by default."""
    return pg.Coalescent(n=n, model=model, demography=pg.Demography(pop_sizes={'pop_0': sizes or _THREE}),
                         start_time=start_time, end_time=end_time)


def _two_demes(n0=2, n1=2, start_time=0, end_time=None) -> pg.Coalescent:
    """Two demes with migration and two epochs."""
    return pg.Coalescent(n={'pop_0': n0, 'pop_1': n1}, start_time=start_time, end_time=end_time,
                         demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.4}, 'pop_1': {0: 1.0}},
                                                  migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 0.6}))


def _spy(method):
    """Patch a method of the moment evaluator with a pass-through spy counting its calls."""
    return patch.object(PhaseTypeDistribution, method, autospec=True,
                        side_effect=getattr(PhaseTypeDistribution, method))


def _per_pair(dist) -> np.ndarray:
    """The covariance of the bins of a spectrum from the cross-moment of each ordered pair of bins."""
    rewards = [CombinedReward([dist.reward, reward]) for _, reward in dist._bins()]
    mean = np.array([PhaseTypeDistribution.moment(dist, k=1, rewards=(r,)) for r in rewards])
    ordered = np.array([[PhaseTypeDistribution.moment(dist, k=2, rewards=(a, b), center=False, permute=False)
                         for b in rewards] for a in rewards])

    return (ordered + ordered.T) / 2 - np.outer(mean, mean)


def _batched(dist) -> np.ndarray:
    """The covariance of the bins of a spectrum over its polymorphic bins, in the order of ``_bins``."""
    cov = np.asarray(getattr(dist.cov, 'data', dist.cov))
    positions = [pos for pos, _ in dist._bins()]

    return np.array([[cov[a + b] for b in positions] for a in positions])


_REGIMES = {
    'three epochs': (lambda: _single().sfs, {}),
    'three epochs folded': (lambda: _single().fsfs, {}),
    'one epoch': (lambda: _single(sizes={0: 1.0}).sfs, {}),
    'one bin': (lambda: _single(n=2).sfs, {}),
    'beta three epochs': (lambda: _single(model=pg.BetaCoalescent(alpha=1.5)).sfs, {}),
    'dirac two epochs': (lambda: _single(model=pg.DiracCoalescent(psi=0.5, c=1), sizes={0: 1, 0.7: 0.2}).sfs, {}),
    'stiff sizes': (lambda: _single(sizes={0: 1e-3, 0.01: 1e4, 5e4: 1e-2}).sfs, {}),
    'start in a finite epoch': (lambda: _single(start_time=0.6).sfs, {}),
    'start in the last epoch': (lambda: _single(start_time=1.5).sfs, {}),
    'start in the only epoch': (lambda: _single(sizes={0: 1.0}, start_time=0.5).sfs, {}),
    'finite window': (lambda: _single(start_time=0.6, end_time=2.0).sfs, {}),
    'finite end': (lambda: _single(end_time=1.0).sfs, {}),
    'closed form disabled': (lambda: _single().sfs, {'closed_form_last_epoch': False}),
    'forced action': (lambda: _single().sfs, {'expm_method': 'action'}),
    'forced dense': (lambda: _single().sfs, {'expm_method': 'dense'}),
    'sparse factorization': (lambda: _single().sfs, {'closed_form_sparse_min_states': 0}),
    'two demes sfs': (lambda: _two_demes().sfs, {}),
    'two demes jsfs': (lambda: _two_demes().jsfs, {}),
    'two demes jsfs window': (lambda: _two_demes(start_time=0.3, end_time=1.5).jsfs, {}),
    'two demes jsfs forced action': (lambda: _two_demes().jsfs, {'expm_method': 'action'}),
}


@pytest.mark.parametrize('name', list(_REGIMES), ids=list(_REGIMES))
def test_covariance_matches_the_cross_moments_of_each_pair(name):
    """The covariance of the bins equals the cross-moments of each ordered pair of bins, centered by their means."""
    make, settings = _REGIMES[name]
    for key, value in settings.items():
        setattr(Settings, key, value)

    batched = _batched(make())
    reference = _per_pair(make())

    np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-12 * np.abs(reference).max())


@pytest.mark.parametrize('read', [
    lambda: _single().sfs.cov,
    lambda: _single().sfs.var,
    lambda: _single().sfs.corr,
    lambda: _single(sizes={0: 1.0}).sfs.cov,
    lambda: _single(start_time=1.5).sfs.var,
    lambda: _single(start_time=0.6, end_time=2.0).fsfs.cov,
    lambda: _two_demes().jsfs.cov,
    lambda: _two_demes().jsfs.var,
    lambda: pg.Coalescent(n=4, loci=2, recombination_rate=0.7,
                          demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 0.5}})).sfs2.mean,
    lambda: pg.Coalescent(n=4, loci=2, recombination_rate=0.7,
                          demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 0.5}})).sfs2.cov,
], ids=['sfs.cov', 'sfs.var', 'sfs.corr', 'sfs.cov one epoch', 'sfs.var start in the last epoch', 'fsfs.cov window',
        'jsfs.cov', 'jsfs.var', 'sfs2.mean', 'sfs2.cov'])
def test_second_moments_of_spectra_take_one_forward_pass(read):
    """Each second moment of a spectrum is read from one forward pass, never from per-bin or per-pair moments."""
    with _spy('_bin_second_moments') as forward, _spy('_accumulate_closed_form') as closed, \
            _spy('_accumulate_windowed') as windowed, _spy('_raw_powers') as powers:
        read()

    assert forward.call_count == 1
    assert closed.call_count == windowed.call_count == powers.call_count == 0


def test_second_moments_equal_the_per_pair_values():
    """Entries of the second moments, as the per-pair and per-bin evaluations gave them."""
    rtol = 1e-12

    sfs = _single(n=6).sfs
    assert sfs.cov.data[1, 2] == pytest.approx(-0.06609360636277128, rel=rtol)
    assert sfs.cov.data[2, 5] == pytest.approx(-0.05453862264194538, rel=rtol)
    assert sfs.var.data[3] == pytest.approx(0.9736311013115428, rel=rtol)
    assert _single(n=6).fsfs.cov.data[1, 3] == pytest.approx(-0.07235352050915567, rel=rtol)

    windowed = _single(n=6, start_time=0.6, end_time=2.0).sfs
    assert windowed.cov.data[1, 2] == pytest.approx(-0.006453006283565935, rel=rtol)
    assert windowed.var.data[1] == pytest.approx(0.07666372117516553, rel=rtol)
    assert _single(n=6, start_time=1.5).sfs.cov.data[1, 4] == pytest.approx(-0.004419241116600708, rel=rtol)

    jsfs = _two_demes().jsfs
    assert np.asarray(jsfs.cov)[1, 0, 0, 1] == pytest.approx(-0.4234888256450624, rel=rtol)
    assert jsfs.var.data[1, 1] == pytest.approx(3.552387167079909, rel=rtol)

    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=0.7,
                         demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 0.5}})).sfs2
    assert sfs2.mean.data[1, 2] == pytest.approx(0.8871406096802578, rel=rtol)
    assert sfs2.cov.data[1, 1] == pytest.approx(0.4249968182667958, rel=1e-11)


@pytest.mark.parametrize('offset, action', [(0, True), (1, False)])
def test_windowed_pass_takes_the_action_from_the_threshold(offset, action, monkeypatch):
    """In a finite window, the pass takes the stacked action from the Van Loan dimension at which
    ``Settings.expm_method = 'auto'`` caps the dense exponentials on, and one dense exponential per bin below, with the
    same values."""
    reference = _batched(_single(end_time=2.0).sfs)
    sfs = _single(end_time=2.0).sfs
    monkeypatch.setattr(_moments, '_EXPM_ACTION_MIN_DIM', 3 * sfs.state_space.k + offset)

    with _spy('_block_expms') as dense, patch.object(Backend, 'expm_multiply', side_effect=Backend.expm_multiply) as act:
        batched = _batched(sfs)

    stacked = [c for c in act.call_args_list if c.kwargs.get('blocks') == 2 * len(sfs._bins()) + 1]
    assert bool(stacked) is action and (dense.call_count > 0) is not action
    np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-12 * np.abs(reference).max())


@pytest.mark.parametrize('dense', [True, False])
@pytest.mark.parametrize('end_time', [None, 2.0], ids=['closed', 'window'])
def test_pass_follows_the_stacked_dense_exponential_rule(dense, end_time):
    """The finite epochs take the dense exponentials exactly where ``_dense_expm_is_faster`` holds, with the same
    values, until absorption and in a window."""
    reference = _per_pair(_single(end_time=end_time).sfs)
    sfs = _single(end_time=end_time).sfs

    with patch.object(PhaseTypeDistribution, '_dense_expm_is_faster', autospec=True, return_value=dense), \
            _spy('_block_expms') as blocks:
        batched = _batched(sfs)

    assert (blocks.call_count > 0) is dense
    np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-12 * np.abs(reference).max())


def _stacked_dispatch(make) -> bool:
    """Whether the second moments of the spectrum made by ``make`` take the stacked action."""
    action = patch.object(Backend, 'expm_multiply', side_effect=Backend.expm_multiply)
    with action as act, _spy('_block_expms') as dense:
        make().cov

    stacked = any(c.kwargs.get('blocks', 1) > 3 for c in act.call_args_list)
    assert stacked is not (dense.call_count > 0)

    return stacked


_SHORT = {0: 1.0, 0.1: 0.5, 0.2: 2.0}


@pytest.mark.parametrize('make, action', [
    (lambda: _single(n=5).sfs, False),
    (lambda: _single(n=10).sfs, True),
    (lambda: _single(n=6, model=pg.BetaCoalescent(alpha=1.5)).sfs, False),
    (lambda: _single(n=10, model=pg.BetaCoalescent(alpha=1.5)).sfs, True),
    (lambda: _single(n=6, model=pg.DiracCoalescent(psi=0.5, c=1)).sfs, False),
    (lambda: _single(n=5, end_time=2.0).sfs, False),
    (lambda: _two_demes(3, 3, start_time=0.3, end_time=1.5).jsfs, True),
    (lambda: _two_demes(2, 2).sfs, False),
], ids=['kingman n=5', 'kingman n=10', 'beta n=6', 'beta n=10', 'dirac n=6', 'window n=5', 'window jsfs 3+3',
        'two demes sfs 2+2'])
def test_stacked_dispatch_on_the_block_counting_space(make, action):
    """The forward pass takes the dense exponentials on small block-counting spaces and the stacked action from about
    20 states on, until absorption and in a window."""
    assert _stacked_dispatch(make) is action


@pytest.mark.parametrize('n, sizes, action', [(10, _THREE, False), (30, _THREE, False), (30, _SHORT, True)],
                         ids=['n=10', 'n=30 long epochs', 'n=30 short epochs'])
def test_stacked_dispatch_on_the_lineage_counting_space(n, sizes, action):
    """Flattened onto the lineage-counting space, the forward pass over the ``n - 1`` levels takes the dense
    exponentials on stiff epochs and the stacked action on short ones."""
    Settings.flatten_block_counting = True
    sfs = _single(n=n, sizes=sizes).sfs

    with _spy('_level_cross_moments') as flattened:
        assert _stacked_dispatch(lambda: sfs) is action

    assert flattened.call_count == 1


def test_stiff_stacked_action_stops_once_stationary():
    """An epoch whose generator has a 1-norm far above the stepping threshold is applied in norm-doubling steps that
    stop once the stacked vector is stationary, and matches the dense exponentials."""
    make = lambda: _single(sizes={0: 1.0, 0.5: 1e-4, 2.0: 1.0}).sfs
    Settings.expm_method = 'dense'
    reference = _batched(make())

    Settings.expm_method = 'action'
    with patch.object(Backend, '_stationary', side_effect=Backend._stationary) as stationary:
        batched = _batched(make())

    assert stationary.call_count > 0
    np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-11 * np.abs(reference).max())


def test_growth_bound_depth_bounds_the_exponential():
    """The growth bound with the depth of the coupling chain bounds the mean norm of the exponential of a matrix of five
    diagonal blocks coupled along chains of three blocks, and is tighter than the bound over all five blocks."""
    rng = np.random.default_rng(0)
    n = 4
    S = rng.uniform(0, 1, (n, n))
    np.fill_diagonal(S, 0)
    np.fill_diagonal(S, -S.sum(axis=1) - 0.5)

    # blocks p, u_1, u_2, w_1, w_2 of the forward pass, transposed for the column action
    A = np.zeros((5 * n, 5 * n))
    for b in range(3):
        A[b * n:(b + 1) * n, b * n:(b + 1) * n] = S.T
    for a in range(2):
        A[(1 + a) * n:(2 + a) * n, :n] = np.diag(rng.uniform(1, 3, n))
        A[(3 + a) * n:(4 + a) * n, (1 + a) * n:(2 + a) * n] = 2.0 * np.eye(n)
    A = sp.csr_matrix(A * 3.0)

    import scipy.linalg as sla
    for f in (0.5, 1.0, 4.0):
        s = np.linspace(0, f, 401)
        norms = [np.abs(sla.expm(x * A.toarray())).sum(axis=0).max() for x in s]
        mean_norm = float(np.sum((np.array(norms[1:]) + norms[:-1]) / 2 * np.diff(s)) / f)
        tight, loose = _GrowthBound(A, abs(A), 5, depth=3)(f), _GrowthBound(A, abs(A), 5)(f)
        assert mean_norm <= tight * (1 + 1e-6) and tight < loose


def test_jacobian_of_second_moments_is_unchanged():
    """The exact Jacobian of entries of ``sfs.cov``, ``jsfs.var`` and ``jsfs.cov``, read through the forward pass,
    equals the one from the per-pair values, and the backward pass agrees with the values read."""
    coal = lambda N1, m: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1.0, 0.5: N1}, 'pop_1': {0: 1.0}},
        migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): 0.6}))
    gradient = pg.Gradient(coal, lambda c: np.array([c.sfs.cov.data[1, 2], c.jsfs.var.data[1, 1],
                                                     np.asarray(c.jsfs.cov)[1, 0, 0, 1]]))
    params, bounds = {'N1': 0.4, 'm': 1.0}, {'N1': (0.05, 5), 'm': (0.01, 5)}

    exact = gradient._exact(params, bounds)
    assert exact is not None

    value, jacobian = exact
    np.testing.assert_allclose(value, [-1.3180612934735434, 3.552387167079909, -0.4234888256450624], rtol=1e-12)
    np.testing.assert_allclose(jacobian, [[-1.2468038558119652, 0.7500612562903464],
                                          [4.319325382546423, 0.15202572093266548],
                                          [-0.4638229069345021, 0.1346383190704073]], rtol=1e-11)
