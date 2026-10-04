"""
Parity of the accumulation over time of the empirical joint and two-locus spectra of :class:`MsprimeCoalescent`, and
of every accumulation of :class:`SampledCoalescent`, with the exact distributions: ``accumulate``,
``get_accumulation`` and ``plot_accumulation``. Each estimate is compared with the exact value within four standard
errors of the replicates, over one and two epochs, two demes and two loci.
"""
import functools

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions.empirical import _EmpiricalJointSFSAccumulation, _EmpiricalTwoLocusSFSAccumulation
from phasegen.distributions.spectra import JointSFSDistribution, TwoLocusSFSDistribution

#: Number of trajectories of every sampled coalescent.
N = 4000

#: Number of replicates of every msprime simulation.
N_MS = 2000

#: Times at which the accumulation is compared.
TIMES = (0.4, 1.0, 2.5)

#: Number of standard errors within which an estimate must lie.
N_SE = 4

TWO_DEMES = pg.Demography(pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.3})

TWO_DEMES_2_EPOCHS = pg.Demography(pop_sizes={'a': {0: 1, 0.5: 0.3}, 'b': {0: 2}},
                                   migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.3})

#: The exact coalescents, by scenario name.
SCENARIOS = {
    '1_epoch': pg.Coalescent(n=4),
    '2_epochs': pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2}})),
    '2_demes': pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES),
    '2_demes_2_epochs': pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES_2_EPOCHS),
    '2_loci': pg.Coalescent(n=3, loci=2, recombination_rate=1),
    '2_loci_2_epochs': pg.Coalescent(n=3, loci=2, recombination_rate=1,
                                     demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2}})),
}

#: The rewards compared on the coalescent-level accumulation, by scenario name.
REWARDS = {
    '1_epoch': [pg.TreeHeightReward(), pg.TotalBranchLengthReward(), pg.UnfoldedSFSReward(1)],
    '2_epochs': [pg.TreeHeightReward(), pg.UnfoldedSFSReward(2)],
    '2_demes': [pg.TreeHeightReward(), pg.CombinedReward([pg.TotalBranchLengthReward(), pg.DemeReward('a')]),
                pg.JointSFSReward((1, 1))],
    '2_loci': [pg.TreeHeightReward(), pg.RestrictedReward(pg.TotalBranchLengthReward(), locus=1),
               pg.TwoLocusSFSReward(0, 1)],
}


@functools.lru_cache(maxsize=None)
def _sampled(name: str) -> pg.distributions.SampledCoalescent:
    """The seeded sampled coalescent of a scenario."""
    return SCENARIOS[name].to_empirical(n_samples=N, seed=list(SCENARIOS).index(name) + 1)


@functools.lru_cache(maxsize=None)
def _simulated(name: str) -> pg.distributions.MsprimeCoalescent:
    """The seeded serial msprime simulation of a scenario."""
    pg.Settings.use_pbar = False

    return SCENARIOS[name].to_msprime(num_replicates=N_MS, parallelize=False, n_threads=1,
                                      seed=list(SCENARIOS).index(name) + 1)


def _standard_error(values: list, center: bool) -> np.ndarray:
    """
    Standard error of the sample cross-moment of per-replicate values, each of shape ``(len(times), N)``, from the
    spread of the per-replicate products.
    """
    if center and len(values) > 1:
        values = [v - v.mean(axis=1, keepdims=True) for v in values]

    products = np.prod(values, axis=0)

    return products.std(axis=1) / np.sqrt(products.shape[1])


def _assert_close(exact, estimate, se) -> None:
    """Assert that the estimate lies within ``N_SE`` standard errors of the exact value."""
    exact, estimate, se = np.asarray(exact, float), np.asarray(estimate, float), np.asarray(se, float)

    assert np.all(np.abs(estimate - exact) <= N_SE * se + 1e-10), (exact, estimate, se)


def _records(dist):
    """The per-replicate records behind the accumulation of an empirical distribution."""
    return dist._accumulator._coalescent._trajectory_records()


def _jsfs_se(dist, k: int) -> np.ndarray:
    """Standard error of the accumulated moment of every joint spectrum bin, of shape ``(len(TIMES),) + shape``."""
    records, shape = _records(dist), dist._accumulator.shape
    se = np.zeros((len(TIMES),) + shape)

    for config in dist._accumulator._get_configs():
        values = records.accumulated(pg.JointSFSReward(config), 0.0, np.array(TIMES))
        se[(slice(None),) + config] = _standard_error([values] * k, True)

    return se


def _sfs2_se(dist, k: int, n: int) -> np.ndarray:
    """Standard error of the accumulated moment of every two-locus bin, symmetrized over the loci."""
    records = _records(dist)
    lengths = {(l, i): records.accumulated(pg.TwoLocusSFSReward(l, i), 0.0, np.array(TIMES))
               for l in (0, 1) for i in range(1, n)}
    se = np.zeros((len(TIMES), n + 1, n + 1))

    for i in range(1, n):
        for j in range(1, n):
            se[:, i, j] = _standard_error([lengths[0, i] * lengths[1, j]] * k, True)

    return (se + se.transpose(0, 2, 1)) / 2


# ---- MsprimeCoalescent: joint and two-locus spectra ----------------------------------------------------------------


@pytest.mark.parametrize('name', ['2_demes', '2_demes_2_epochs'])
def test_msprime_jsfs_accumulate_matches_exact(name):
    """The accumulated mean and variance of every joint spectrum bin match the exact accumulation."""
    coal, ms = SCENARIOS[name], _simulated(name)

    for k in (1, 2):
        exact, estimate = coal.jsfs.accumulate(k, TIMES), ms.jsfs.accumulate(k, TIMES)
        assert estimate.shape == exact.shape == (len(TIMES), 3, 2)
        _assert_close(exact, estimate, _jsfs_se(ms.jsfs, k))


@pytest.mark.parametrize('name', ['2_loci', '2_loci_2_epochs'])
def test_msprime_sfs2_accumulate_matches_exact(name):
    """The accumulated mean and variance of every two-locus bin match the exact accumulation."""
    coal, ms = SCENARIOS[name], _simulated(name)

    for k in (1, 2):
        exact, estimate = coal.sfs2.accumulate(k, TIMES), ms.sfs2.accumulate(k, TIMES)
        assert estimate.shape == exact.shape == (len(TIMES), 4, 4)
        _assert_close(exact, estimate, _sfs2_se(ms.sfs2, k, 3))


def test_msprime_spectra_accumulate_to_their_moments():
    """
    Accumulated to absorption, the joint and two-locus spectra give the moments of the statistics simulated from the
    same seeded genealogies, the descendant vector of every lineage being recorded.
    """
    jsfs, sfs2 = _simulated('2_demes').jsfs, _simulated('2_loci').sfs2

    np.testing.assert_allclose(jsfs.accumulate(1, [np.inf])[0], jsfs.mean.data, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(jsfs.accumulate(2, [np.inf], center=False)[0], jsfs.m2.data, rtol=1e-12, atol=1e-14)

    mean = np.asarray(sfs2.mean.data)
    np.testing.assert_allclose(sfs2.accumulate(1, [np.inf])[0], (mean + mean.T) / 2, rtol=1e-12, atol=1e-14)


def test_msprime_spectra_errors():
    """End times beyond the end of the coalescent and descendant vectors that do not exist raise."""
    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=TWO_DEMES, end_time=0.6)
    ms = coal.to_msprime(num_replicates=50, parallelize=False, n_threads=1, seed=1)

    with pytest.raises(ValueError, match='end time of the coalescent'):
        ms.jsfs.accumulate(1, [0.7])

    two_loci = pg.Coalescent(n=3, loci=2, recombination_rate=1, end_time=0.6).to_msprime(
        num_replicates=50, parallelize=False, n_threads=1, seed=1)
    with pytest.raises(ValueError, match='end time of the coalescent'):
        two_loci.sfs2.accumulate(1, [0.7])

    np.testing.assert_array_equal(two_loci.sfs2.accumulate(1, []).shape, (0, 4, 4))
    assert two_loci.sfs2.accumulate(0, [0.3])[0, 1, 1] == 1

    for config in ((3, 0), (0, 0), (1, 0, 0)):
        with pytest.raises(ValueError, match='descendant vector'):
            ms.accumulate(1, [0.5], [pg.JointSFSReward(config)])

    ms._touch()
    ms._drop()
    with pytest.raises(NotImplementedError, match='MsprimeCoalescent'):
        ms.jsfs.accumulate(1, [0.5])


# ---- SampledCoalescent -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['1_epoch', '2_epochs', '2_demes', '2_loci'])
def test_sampled_accumulate_matches_exact(name):
    """The accumulated means, variances and covariances with the first reward match the exact accumulation."""
    coal, sampled, rewards = SCENARIOS[name], _sampled(name), REWARDS[name]

    for reward in rewards:
        for k, pair in ((1, [reward]), (2, [rewards[0], reward])):
            records = sampled._accumulator(k, pair)._coalescent._trajectory_records()
            se = _standard_error([records.accumulated(r, 0.0, np.array(TIMES)) for r in pair], True)
            _assert_close(coal.accumulate(k, TIMES, pair), sampled.accumulate(k, TIMES, pair), se)


@pytest.mark.parametrize('name', ['1_epoch', '2_epochs', '2_demes', '2_loci'])
def test_sampled_distribution_accumulate_matches_exact(name):
    """The accumulation of every distribution of a scenario matches the exact one."""
    coal, sampled = SCENARIOS[name], _sampled(name)

    for attr in ('tree_height', 'total_branch_length'):
        dist = getattr(sampled, attr)
        values = _records(dist).accumulated(dist._accumulator.reward, 0.0, np.array(TIMES))
        _assert_close(getattr(coal, attr).accumulate(1, TIMES), dist.accumulate(1, TIMES),
                      _standard_error([values], True))

    if coal.locus_config.n == 1:
        for attr, bin_reward in (('sfs', pg.UnfoldedSFSReward), ('fsfs', pg.FoldedSFSReward)):
            exact, estimate = getattr(coal, attr).accumulate(1, TIMES), getattr(sampled, attr).accumulate(1, TIMES)
            assert estimate.shape == exact.shape == (len(TIMES), coal.n + 1)

            records = _records(getattr(sampled, attr))
            se = np.array([_standard_error([records.accumulated(bin_reward(i), 0.0, np.array(TIMES))], True)
                           for i in range(coal.n + 1)]).T
            _assert_close(exact, estimate, se)
            _assert_close(getattr(coal, attr).get_accumulation(1, 1, 1.0),
                          getattr(sampled, attr).get_accumulation(1, 1, 1.0), se[1, 1])

    if coal.lineage_config.n_pops > 1:
        _assert_close(coal.jsfs.accumulate(2, TIMES), sampled.jsfs.accumulate(2, TIMES), _jsfs_se(sampled.jsfs, 2))

    if coal.locus_config.n == 2:
        _assert_close(coal.sfs2.accumulate(2, TIMES), sampled.sfs2.accumulate(2, TIMES),
                      _sfs2_se(sampled.sfs2, 2, coal.n))


def test_sampled_accumulation_follows_the_samples():
    """
    The trajectories recorded again from the seed of a statistic give its samples, also when sampled in batches, and
    those of the coalescent-level accumulation give the sampled moments.
    """
    sampled = _sampled('2_epochs')

    np.testing.assert_allclose(sampled.tree_height.accumulate(1, [np.inf]), [sampled.tree_height.mean], rtol=1e-12)
    np.testing.assert_allclose(sampled.sfs.accumulate(1, [np.inf])[0], sampled.sfs.mean.data, rtol=1e-12)
    np.testing.assert_allclose(sampled.fsfs.get_accumulation(2, 1, [np.inf], center=False),
                               [sampled.fsfs.m2.data[1]], rtol=1e-12)

    rewards = [pg.TreeHeightReward(), pg.UnfoldedSFSReward(1)]
    np.testing.assert_allclose(sampled.accumulate(2, [np.inf], rewards), [sampled.moment(2, rewards)], rtol=1e-12)

    jsfs = _sampled('2_demes').jsfs
    np.testing.assert_allclose(jsfs.accumulate(1, [np.inf])[0], jsfs.mean.data, rtol=1e-12, atol=1e-14)

    sfs2 = _sampled('2_loci').sfs2
    mean = np.asarray(sfs2.mean.data)
    np.testing.assert_allclose(sfs2.accumulate(1, [np.inf])[0], (mean + mean.T) / 2, rtol=1e-12, atol=1e-14)

    # batches of trajectories, each from its own spawned generator
    batch = pg.Settings.sample_batch_size
    try:
        pg.Settings.sample_batch_size = 300
        batched = SCENARIOS['2_epochs'].to_empirical(n_samples=1000, seed=4)
        np.testing.assert_allclose(batched.sfs.accumulate(1, [np.inf])[0], batched.sfs.mean.data, rtol=1e-12)
    finally:
        pg.Settings.sample_batch_size = batch


def test_sampled_window_and_errors():
    """The window of the accumulation is bounded by the end time of the coalescent, and is linear in its ends."""
    sampled = pg.Coalescent(n=4, end_time=0.6).to_empirical(n_samples=1000, seed=3)

    np.testing.assert_allclose(sampled.accumulate(1, [0.6]), [sampled.moment(1)], rtol=1e-12)
    np.testing.assert_allclose(sampled.tree_height.accumulate(1, [0.6]), [sampled.tree_height.mean], rtol=1e-12)

    full = sampled.accumulate(1, [0.2, 0.5])
    np.testing.assert_allclose(sampled.accumulate(1, [0.5], start_time=0.2), [full[1] - full[0]], rtol=1e-12)
    np.testing.assert_array_equal(sampled.accumulate(0, [0.3]), [1])
    assert sampled.accumulate(1, []).shape == (0,)

    for call in (lambda: sampled.accumulate(1, [0.3, 0.7]), lambda: sampled.tree_height.accumulate(1, [0.7]),
                 lambda: sampled.sfs.accumulate(1, [0.7]), lambda: sampled.plot_accumulation(end_times=[0.7])):
        with pytest.raises(ValueError, match='end time of the coalescent'):
            call()

    with pytest.raises(ValueError, match='single Reward'):
        sampled.accumulate(1, [0.3], pg.TreeHeightReward())

    with pytest.raises(ValueError, match='must be 2'):
        sampled.accumulate(2, [0.3], [pg.TreeHeightReward()])

    sampled._touch()
    sampled._drop()
    assert '_sampled_trajectories' not in sampled.__dict__

    with pytest.raises(NotImplementedError, match='SampledCoalescent'):
        sampled.tree_height.accumulate(1, [0.3])


# ---- plotting ------------------------------------------------------------------------------------------------------


def test_plotting_reuses_exact_code_path():
    """The empirical plots of the joint and two-locus spectra are drawn by the plotting functions of the exact ones."""
    assert _EmpiricalJointSFSAccumulation._plot_accumulation_data is JointSFSDistribution._plot_accumulation_data
    assert _EmpiricalTwoLocusSFSAccumulation._plot_accumulation_data is \
           TwoLocusSFSDistribution._plot_accumulation_data
    assert pg.distributions.EmpiricalJointSFSDistribution.plot_accumulation is \
           pg.distributions.PhaseTypeDistribution.plot_accumulation


@pytest.mark.parametrize('name, attr, source', [
    ('2_demes', 'jsfs', _simulated), ('2_demes', 'jsfs', _sampled),
    ('2_loci', 'sfs2', _simulated), ('2_loci', 'sfs2', _sampled)
])
def test_plot_accumulation_draws_accumulation(name, attr, source):
    """The plots of the spectra draw their accumulation, with the labels of the exact plots."""
    exact, dist = getattr(SCENARIOS[name], attr), getattr(source(name), attr)

    expected = exact._plot_accumulation_data(end_times=TIMES)
    drawn = dist._plot_accumulation_data(end_times=TIMES)
    assert drawn.labels == expected.labels and drawn.title == expected.title

    ax = dist.plot_accumulation(end_times=TIMES, show=False)
    np.testing.assert_allclose(ax.get_lines()[1].get_ydata(), drawn.y[1])
    plt.close('all')


def test_sampled_plot_accumulation_draws_accumulation():
    """The plots of SampledCoalescent draw the accumulation at the default end times."""
    sampled = _sampled('1_epoch')
    times = sampled._accumulator(2, None)._default_end_times()
    ax = sampled.plot_accumulation(k=2, show=False)
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), sampled.accumulate(2, times))
    plt.close('all')

    ax = sampled.sfs.plot_accumulation(show=False)
    assert len(ax.get_lines()) == SCENARIOS['1_epoch'].n - 1
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), sampled.sfs._accumulator._default_end_times())
    plt.close('all')


def test_signatures_match_exact():
    """The accumulation members take the parameters of their exact counterparts."""
    import inspect

    exact, sampled = SCENARIOS['2_demes'], _sampled('2_demes')
    pairs = [(exact, sampled), (exact.jsfs, sampled.jsfs), (SCENARIOS['2_loci'].sfs2, _sampled('2_loci').sfs2),
             (SCENARIOS['1_epoch'].sfs, _sampled('1_epoch').sfs)]

    for a, b in pairs:
        for name in ('accumulate', 'plot_accumulation', 'get_accumulation'):
            if hasattr(a, name):
                assert inspect.signature(getattr(a, name)).parameters.keys() == \
                       inspect.signature(getattr(b, name)).parameters.keys(), (type(b).__name__, name)
