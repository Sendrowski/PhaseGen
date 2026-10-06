"""
Parity of the coalescent-level methods of :class:`MsprimeCoalescent` and :class:`SampledCoalescent` with the exact
:class:`Coalescent`: ``joint``, ``moment``, ``accumulate`` and ``plot_accumulation``, and the distribution-level
``accumulate`` and ``plot_accumulation``. Each estimate is compared with the exact value within four standard errors
of the replicates, over one and two epochs, two demes and two loci.
"""
import functools

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import EmpiricalPhaseTypeDistribution
from phasegen.distributions.empirical import _EmpiricalAccumulation, _EmpiricalSFSAccumulation
from phasegen.distributions.phase_type import PhaseTypeDistribution
from phasegen.distributions.spectra import SFSDistribution

#: Number of replicates of every simulation.
N = 5000

#: Times at which the accumulation is compared, late enough for the replicates to resolve the coalescences before
#: them.
TIMES = (0.4, 1.0, 2.5)

#: Number of standard errors within which an estimate must lie.
N_SE = 4


def _scenarios() -> dict:
    """The exact coalescents and the rewards compared on them, by scenario name."""
    two_demes = pg.Demography(pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.3})

    return {
        '1_epoch': (pg.Coalescent(n=4), dict(), [
            pg.TreeHeightReward(), pg.TotalBranchLengthReward(), pg.UnfoldedSFSReward(1), pg.FoldedSFSReward(2)
        ]),
        '2_epochs': (pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2}})), dict(), [
            pg.TreeHeightReward(), pg.TotalBranchLengthReward(), pg.UnfoldedSFSReward(2)
        ]),
        '2_demes': (pg.Coalescent(n={'a': 2, 'b': 2}, demography=two_demes), dict(record_migration=True), [
            pg.TreeHeightReward(),
            pg.RestrictedReward(pg.TotalBranchLengthReward(), pop='a'),
            pg.CombinedReward([pg.TreeHeightReward(), pg.DemeReward('b')]),
            pg.CombinedReward([pg.UnfoldedSFSReward(1), pg.DemeReward('b')])
        ]),
        '2_loci': (pg.Coalescent(n=3, loci=2, recombination_rate=1), dict(), [
            pg.TreeHeightReward(),
            pg.TotalTreeHeightReward(),
            pg.TotalBranchLengthReward(),
            pg.RestrictedReward(pg.TotalBranchLengthReward(), locus=1),
            pg.CombinedReward([pg.TreeHeightReward(), pg.LocusReward(0)])
        ])
    }


SCENARIOS = _scenarios()


@functools.lru_cache(maxsize=None)
def _simulated(name: str) -> tuple:
    """One seeded serial simulation of a scenario, simulated on first use, with its exact coalescent and rewards."""
    pg.Settings.use_pbar = False
    coal, kwargs, rewards = SCENARIOS[name]
    seed = list(SCENARIOS).index(name) + 1

    return coal, coal.to_msprime(num_replicates=N, parallelize=False, n_threads=1, seed=seed, **kwargs), rewards


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


def _values(ms: pg.distributions.MsprimeCoalescent, reward: pg.Reward, times, start_time: float = 0.0) -> np.ndarray:
    """The per-replicate accumulation of a reward."""
    return ms._trajectory_records().accumulated(reward, start_time, np.asarray(times, dtype=float))


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_accumulate_mean_matches_exact(scenario):
    """The mean accumulation of every reward matches the exact accumulation."""
    coal, ms, rewards = _simulated(scenario)

    for reward in rewards:
        se = _standard_error([_values(ms, reward, TIMES)], center=True)
        _assert_close(coal.accumulate(1, TIMES, [reward]), ms.accumulate(1, TIMES, [reward]), se)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_accumulate_second_moments_match_exact(scenario):
    """The accumulated variance of each reward and the covariance of the first reward with each other match."""
    coal, ms, rewards = _simulated(scenario)

    for reward in rewards:
        pair = [rewards[0], reward]
        se = _standard_error([_values(ms, r, TIMES) for r in pair], center=True)
        _assert_close(coal.accumulate(2, TIMES, pair), ms.accumulate(2, TIMES, pair), se)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_moment_matches_exact(scenario):
    """The moments until absorption, raw and central, of orders one to three, match the exact moments."""
    coal, ms, rewards = _simulated(scenario)
    end = [np.inf]

    for reward in rewards:
        for k, center in ((1, True), (2, True), (2, False), (3, True)):
            values = [_values(ms, reward, end)] * k
            _assert_close(coal.moment(k, [reward] * k, center=center), ms.moment(k, [reward] * k, center=center),
                          _standard_error(values, center))


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_windowed_moment_matches_exact(scenario):
    """The moment over a window ``[s, t]`` matches the exact windowed moment."""
    coal, ms, rewards = _simulated(scenario)

    for reward in rewards:
        values = [_values(ms, reward, [1.0], start_time=0.3)] * 2
        _assert_close(coal.moment(2, [reward] * 2, start_time=0.3, end_time=1.0),
                      ms.moment(2, [reward] * 2, start_time=0.3, end_time=1.0), _standard_error(values, True))


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_joint_matches_exact(scenario):
    """The means and the covariance of the joint distribution of two rewards match the exact joint distribution."""
    coal, ms, rewards = _simulated(scenario)
    a, b = rewards[0], rewards[-1]

    exact, joint = coal.joint(a, b), ms.joint(a, b)
    values = [_values(ms, r, [np.inf]) for r in (a, b)]

    assert isinstance(joint, pg.distributions.EmpiricalJointDistribution)
    _assert_close(exact.mean, joint.mean, [_standard_error([v], True)[0] for v in values])
    _assert_close(exact.cov, joint.cov, _standard_error(values, True))

    # cached per pair of rewards, as for the exact coalescent
    assert ms.joint(a, b) is joint


@pytest.mark.parametrize('scenario', ['1_epoch', '2_epochs', '2_demes'])
def test_distribution_accumulate_matches_exact(scenario):
    """The accumulation of the tree height, the total branch length and the spectra match the exact ones."""
    coal, ms, _ = _simulated(scenario)

    for name, reward in (('tree_height', pg.TreeHeightReward()), ('total_branch_length', pg.TotalBranchLengthReward())):
        se = _standard_error([_values(ms, reward, TIMES)], True)
        _assert_close(getattr(coal, name).accumulate(1, TIMES), getattr(ms, name).accumulate(1, TIMES), se)

    for name, bin_reward in (('sfs', pg.UnfoldedSFSReward), ('fsfs', pg.FoldedSFSReward)):
        exact, estimate = getattr(coal, name).accumulate(1, TIMES), getattr(ms, name).accumulate(1, TIMES)
        assert estimate.shape == exact.shape == (len(TIMES), coal.n + 1)

        se = np.array([_standard_error([_values(ms, bin_reward(i), TIMES)], True) for i in range(coal.n + 1)]).T
        _assert_close(exact, estimate, se)


def test_plotting_reuses_exact_code_path():
    """The empirical plots are drawn by the plotting functions of the exact distributions."""
    assert EmpiricalPhaseTypeDistribution.plot_accumulation is PhaseTypeDistribution.plot_accumulation
    assert _EmpiricalAccumulation.plot_accumulation is PhaseTypeDistribution.plot_accumulation
    assert _EmpiricalAccumulation._plot_accumulation_data is PhaseTypeDistribution._plot_accumulation_data
    assert _EmpiricalSFSAccumulation._plot_accumulation_data is SFSDistribution._plot_accumulation_data


@pytest.mark.parametrize('scenario', ['1_epoch', '2_loci'])
def test_plot_accumulation_draws_accumulation(scenario):
    """The coalescent-level and distribution-level plots draw the accumulation at the default end times."""
    coal, ms, _ = _simulated(scenario)
    times = ms._accumulator()._default_end_times()

    ax = ms.plot_accumulation(k=2, show=False)
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), ms.accumulate(2, times))
    plt.close('all')

    ax = ms.total_branch_length.plot_accumulation(show=False)
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), ms.total_branch_length.accumulate(1, times))
    assert ax.get_title() == coal.total_branch_length._plot_accumulation_data().title
    plt.close('all')

    if coal.locus_config.n == 1:
        ax = ms.sfs.plot_accumulation(show=False)
        assert len(ax.get_lines()) == coal.n - 1
        np.testing.assert_allclose(ax.get_lines()[1].get_ydata(), ms.sfs.accumulate(1, times)[:, 2])
        plt.close('all')


def test_seeded_trajectories_pair_with_statistics():
    """
    The trajectories reproduce the statistics of the same seeded genealogies, whether recorded together with them
    or by a second simulation after them.
    """
    pg.Settings.use_pbar = False

    for loci in (1, 2):
        coal = pg.Coalescent(n=4, loci=loci, recombination_rate=1 if loci == 2 else 0)

        before = coal.to_msprime(num_replicates=100, parallelize=False, n_threads=5, seed=7)
        acc = before.accumulate(1, [np.inf])

        after = coal.to_msprime(num_replicates=100, parallelize=False, n_threads=5, seed=7)
        mean = after.tree_height.mean
        assert after._trajectories is None

        np.testing.assert_allclose(acc, [mean], rtol=1e-12)
        np.testing.assert_allclose(after.accumulate(1, [np.inf]), [mean], rtol=1e-12)
        joint = after.joint(pg.TreeHeightReward(), pg.TotalBranchLengthReward())
        np.testing.assert_allclose(before.tree_height.samples, joint.marginal('a').samples, rtol=1e-12)

        if loci == 1:
            np.testing.assert_allclose(after.sfs.accumulate(1, [np.inf])[0], after.sfs.mean.data, rtol=1e-12)


def test_end_time_of_coalescent():
    """A coalescent ended before absorption accumulates the rewards until its end time, and refuses later end times."""
    pg.Settings.use_pbar = False
    coal = pg.Coalescent(n=4, end_time=0.6)
    ms = coal.to_msprime(num_replicates=N, parallelize=False, n_threads=1, seed=3)

    for reward in (pg.TreeHeightReward(), pg.TotalBranchLengthReward(), pg.UnfoldedSFSReward(1)):
        values = [_values(ms, reward, [0.6])] * 2
        _assert_close(coal.moment(2, [reward] * 2, center=False), ms.moment(2, [reward] * 2, center=False),
                      _standard_error(values, False))

    np.testing.assert_allclose(ms._accumulator()._default_end_times()[-1], 0.6)
    ms.tree_height.plot_accumulation(show=False)
    plt.close('all')

    # the simulation stops at the end time, so later end times have no replicates to accumulate over
    np.testing.assert_allclose(ms.accumulate(1, [0.6]), ms.moment(1))

    for call in (lambda: ms.accumulate(1, [0.3, 0.7]), lambda: ms.moment(1, end_time=1.0),
                 lambda: ms.moment(2, end_time=np.inf), lambda: ms.tree_height.accumulate(1, [0.7]),
                 lambda: ms.sfs.accumulate(1, [0.7])):
        with pytest.raises(ValueError, match='end time of the coalescent'):
            call()


def test_window_boundaries():
    """Accumulation is linear in the window, vanishes on an empty window and saturates after absorption."""
    _, ms, rewards = _simulated('1_epoch')

    for reward in rewards:
        full = ms.accumulate(1, [0.4, 2.5], [reward])
        windowed = ms.accumulate(1, [2.5], [reward], start_time=0.4)
        np.testing.assert_allclose(windowed, full[1] - full[0], rtol=1e-12)

        np.testing.assert_array_equal(ms.accumulate(1, [0.0, 0.2], [reward], start_time=0.2), [0, 0])
        np.testing.assert_array_equal(ms.accumulate(2, [0.2], [reward] * 2, start_time=0.5), [0])

    np.testing.assert_allclose(ms.accumulate(1, [1e6]), ms.accumulate(1, [np.inf]), rtol=1e-12)
    np.testing.assert_array_equal(ms.accumulate(0, TIMES), np.ones(len(TIMES)))
    assert ms.moment(0) == 1
    assert ms.accumulate(1, []).shape == (0,)


def test_sampled_coalescent_matches_exact():
    """The joint distribution and the moments of SampledCoalescent match the exact coalescent."""
    for coal, rewards in (
            (pg.Coalescent(n=4), (pg.TreeHeightReward(), pg.UnfoldedSFSReward(1))),
            (pg.Coalescent(n=3, loci=2, recombination_rate=1),
             (pg.TwoLocusSFSReward(0, 1), pg.TwoLocusSFSReward(1, 1)))
    ):
        sampled = coal.to_empirical(n_samples=N, seed=5)
        samples = sampled._sample_rewards(rewards, 'moment')
        values = [s[None, :] for s in samples.T]

        _assert_close(coal.moment(2, rewards), sampled.moment(2, rewards), _standard_error(values, True))
        _assert_close(coal.moment(1, rewards[:1]), sampled.moment(1, rewards[:1]),
                      _standard_error(values[:1], True))

        joint = sampled.joint(*rewards)
        _assert_close(coal.joint(*rewards).cov, joint.cov, _standard_error(values, True))

        # reproducible from the seed
        assert coal.to_empirical(n_samples=N, seed=5).moment(2, rewards) == sampled.moment(2, rewards)

    with pytest.raises(NotImplementedError, match='window of the wrapped coalescent'):
        sampled.moment(1, end_time=1.0)


def test_errors():
    """Unsupported rewards and arguments raise as the exact coalescent does."""
    ms = pg.Coalescent(n=4).to_msprime(num_replicates=50, parallelize=False, n_threads=1, seed=1)

    with pytest.raises(NotImplementedError, match='Supported are'):
        ms.accumulate(1, TIMES, [pg.LineageReward(2)])

    with pytest.raises(NotImplementedError, match='Supported are'):
        ms.joint(pg.TreeHeightReward(), pg.ProductReward([pg.TreeHeightReward(), pg.TotalBranchLengthReward()]))

    with pytest.raises(ValueError, match='single Reward'):
        ms.moment(1, pg.TreeHeightReward())

    with pytest.raises(ValueError, match='must be 2'):
        ms.accumulate(2, TIMES, [pg.TreeHeightReward()])

    with pytest.raises(ValueError, match='Start time'):
        ms.accumulate(1, TIMES, start_time=-1)

    with pytest.raises(TypeError):
        ms.joint(pg.TreeHeightReward(), [pg.TreeHeightReward()])

    with pytest.raises(ValueError, match='Locus 2 does not exist'):
        ms.accumulate(1, TIMES, [pg.RestrictedReward(pg.TotalBranchLengthReward(), locus=2)])

    with pytest.raises(ValueError, match='polymorphic class'):
        ms.accumulate(1, TIMES, [pg.TwoLocusSFSReward(0, 4)])

    # the demes of several demes are resolved only with migration recording
    coal = pg.Coalescent(n={'a': 1, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))
    unresolved = coal.to_msprime(num_replicates=10, parallelize=False, n_threads=1, seed=1)

    with pytest.raises(ValueError, match='record_migration'):
        unresolved.accumulate(1, TIMES, [pg.RestrictedReward(pg.TreeHeightReward(), pop='a')])

    # the tree height of several loci does not decompose over the demes
    two_loci = pg.Coalescent(n=3, loci=2, recombination_rate=1).to_msprime(
        num_replicates=50, parallelize=False, n_threads=1, seed=1)
    with pytest.raises(NotImplementedError, match='additively'):
        two_loci.accumulate(1, TIMES, [pg.RestrictedReward(pg.TreeHeightReward(), pop='pop_0')])

    # a distribution without simulated genealogies
    with pytest.raises(NotImplementedError, match='MsprimeCoalescent'):
        pg.Coalescent(n=3).tree_height.to_empirical(100, seed=1).accumulate(1, TIMES)


def test_drop_clears_trajectories():
    """Dropping the simulated data drops the trajectories and the accumulators of the distributions."""
    pg.Settings.use_pbar = False
    ms = pg.Coalescent(n=3).to_msprime(num_replicates=50, parallelize=False, n_threads=1, seed=1)
    ms.accumulate(1, TIMES)
    ms._touch()
    ms._drop()

    assert ms._trajectories is None
    assert ms.tree_height._accumulator is None

    with pytest.raises(NotImplementedError, match='MsprimeCoalescent'):
        ms.tree_height.accumulate(1, TIMES)
