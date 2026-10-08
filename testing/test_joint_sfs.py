"""
Test the joint (multi-population) site-frequency spectrum.

Fast, simulation-free invariants (state-space structure, marginal consistency, the ``JointSFS`` container, automatic
reward routing) are plain tests; the msprime comparisons are marked ``slow`` so they can be deselected with
``-m "not slow"``.
"""
import json
from math import prod
from pathlib import Path

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent
from phasegen.state_space import JointBlockCountingStateSpace

# directory holding the independent ``moments`` joint-SFS references (see ``scripts/generate_jsfs_reference.py``)
MOMENTS_REFERENCE_DIR = Path(__file__).resolve().parent.parent / 'results' / 'jsfs_reference'

# configs for which a ``moments`` equilibrium-island reference is generated
MOMENTS_REFERENCE_CONFIGS = [
    '1_epoch_2_pops_n_4_jsfs',
    '1_epoch_2_pops_n_6_jsfs',
    '1_epoch_2_pops_n_6_asym_jsfs',
    '1_epoch_2_pops_n_8_jsfs',
]


@pytest.fixture(scope="module")
def two_pop_coalescent(symmetric_demography):
    """
    A small two-population coalescent reused (built once) across the fast tests.
    """
    return pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.5}, migration_rate=0.75)
    )


# sample configurations used for the simulation-free marginal-consistency check
MARGINAL_CONFIGS = [
    {'pop_0': 2, 'pop_1': 2},
    {'pop_0': 3, 'pop_1': 1},
    {'pop_0': 2, 'pop_1': 1, 'pop_2': 1},
]

# msprime comparison cases: (n, pop_sizes, migration_rate, model, seed)
MSPRIME_CASES = [
    pytest.param({'pop_0': 2, 'pop_1': 2}, {'pop_0': 1.0, 'pop_1': 1.5}, 0.75, None, 42, id="2pop_n4"),
    pytest.param({'pop_0': 3, 'pop_1': 1}, {'pop_0': 1.0, 'pop_1': 0.7}, 1.0, None, 43, id="2pop_n4_asym"),
    pytest.param({'pop_0': 1, 'pop_1': 1, 'pop_2': 1}, {'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.0}, 1.0, None, 44,
                 id="3pop_n3"),
    pytest.param({'pop_0': 2, 'pop_1': 1}, {'pop_0': 1.0, 'pop_1': 1.0}, 1.0, pg.BetaCoalescent(alpha=1.5), 45,
                 id="beta_2pop_n3"),
]


def test_state_space_structure(symmetric_demography):
    """
    Test the basic structure of the joint block-counting state space.
    """
    n = {'pop_0': 2, 'pop_1': 1}
    s = JointBlockCountingStateSpace(
        lineage_config=pg.LineageConfig(n),
        epoch=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0}).get_epoch(0)
    )

    # number of block types is prod(n_p + 1) - 1 (excluding the all-zero vector)
    assert len(s.block_configs) == prod(v + 1 for v in n.values()) - 1

    # the initial state is unique and alpha is a probability distribution
    assert s.alpha.sum() == pytest.approx(1.0)
    assert (s.alpha > 0).sum() == 1

    # the intensity matrix has zero row sums (valid generator) and at least one absorbing state
    assert np.allclose(s.S.sum(axis=1), 0)
    assert any(state.is_absorbing() for state in s.states)


def test_multiple_loci_not_implemented(symmetric_demography):
    """
    The joint SFS only supports a single locus; using multiple loci (i.e. recombination) must raise a clear
    ``NotImplementedError`` from every entry point.
    """
    n = {'pop_0': 2, 'pop_1': 2}
    demography = symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0})

    # constructing the state space directly with more than one locus
    with pytest.raises(NotImplementedError):
        JointBlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n),
            locus_config=pg.LocusConfig(n=2),
            epoch=demography.get_epoch(0)
        )

    # via the public joint SFS distribution
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=n, demography=demography, loci=2, recombination_rate=1.0).jsfs.mean

    # via automatic reward routing
    with pytest.raises(NotImplementedError):
        pg.Coalescent(n=n, demography=demography, loci=2, recombination_rate=1.0).moment(
            k=1, rewards=[pg.JointSFSReward((1, 0))]
        )


def test_mean_is_jointsfs_container(two_pop_coalescent):
    """
    The joint SFS mean is a :class:`~phasegen.spectrum.JointSFS` container of the right shape, with non-negative
    entries and zero monomorphic corners.
    """
    mean = two_pop_coalescent.jsfs.mean

    assert isinstance(mean, pg.JointSFS)
    assert mean.n_pops == 2
    assert mean.shape == (3, 3)
    assert mean[0, 0] == 0
    assert mean[2, 2] == 0
    assert np.all(mean.data >= 0)


def test_jointsfs_container_2d_and_3d(symmetric_demography):
    """
    The joint SFS is a 2-D array for two populations (plottable directly) and a genuine 3-D array for three
    populations (marginalizable to 2-D for plotting).
    """
    # two populations -> 2-D
    jsfs2 = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0})
    ).jsfs.mean
    assert jsfs2.n_pops == 2
    jsfs2.plot(show=False)
    jsfs2.plot_surface(show=False)

    # three populations -> 3-D, marginalize to 2-D
    jsfs3 = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 1, 'pop_2': 1},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.0})
    ).jsfs.mean
    assert jsfs3.n_pops == 3
    assert jsfs3.shape == (3, 2, 2)

    marginal = jsfs3.marginalize((0, 1))
    assert marginal.shape == (3, 2)

    # marginalizing onto a single population reproduces the per-deme totals (summing the rest)
    assert np.allclose(marginal.data, jsfs3.data.sum(axis=2))

    jsfs3.plot(pops=(0, 1), show=False)
    jsfs3.plot_surface(pops=(0, 1), show=False)


@pytest.mark.parametrize("n", MARGINAL_CONFIGS)
def test_marginal_consistency_with_single_population_sfs(symmetric_demography, n):
    """
    The pooled joint SFS (summing over all configurations with the same total allele count) must reproduce the
    single-population SFS under the same demography. This holds analytically and requires no simulation.
    """
    coal = pg.Coalescent(
        n=n,
        demography=symmetric_demography({pop: 1.0 + i * 0.3 for i, pop in enumerate(n)})
    )

    jsfs = coal.jsfs.mean
    sfs = coal.sfs.mean.data

    # collapse the joint SFS onto the total allele count
    pooled = np.zeros(coal.lineage_config.n + 1)
    for config in np.ndindex(jsfs.shape):
        pooled[sum(config)] += jsfs[config]

    np.testing.assert_allclose(pooled, sfs, atol=1e-10)


def test_single_population_jsfs_raises():
    """
    The joint SFS is a spectrum *across* populations, so accessing it for a single population is an error
    (the single-population spectrum is ``sfs``). A single population with multiple epochs (size-change times as
    ``pop_sizes`` keys) is still one population and must also raise.
    """
    with pytest.raises(ValueError, match="at least two populations"):
        _ = pg.Coalescent(n=5).jsfs

    demo = pg.Demography(pop_sizes={0: 1.0, 1: 1.1})
    with pytest.raises(ValueError, match="at least two populations"):
        _ = pg.Coalescent(n=5, demography=demo).jsfs


def test_moment_accumulate_auto_routing(two_pop_coalescent):
    """
    Passing a :class:`JointSFSReward` to :meth:`Coalescent.moment` or :meth:`Coalescent.accumulate` must
    automatically route to the joint state space and agree with the explicit ``jsfs`` distribution.
    """
    mean = two_pop_coalescent.jsfs.mean

    for config in [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
        m = two_pop_coalescent.moment(k=1, rewards=[pg.JointSFSReward(config)])
        assert m == pytest.approx(mean[config], abs=1e-9)

    # accumulate at a large time approaches the bin mean
    acc = two_pop_coalescent.accumulate(k=1, end_times=[100.0], rewards=[pg.JointSFSReward((1, 1))])
    assert float(acc[0]) == pytest.approx(mean[(1, 1)], abs=1e-6)


def test_jsfs_accumulate(two_pop_coalescent):
    """
    ``jsfs.accumulate`` returns the whole spectrum over time, converging to the bin means and agreeing per bin with
    ``Coalescent.accumulate``; ``plot_accumulation`` must also run.
    """
    jsfs = two_pop_coalescent.jsfs
    mean = jsfs.mean

    end_times = [0.5, 2.0, 100.0]
    acc = jsfs.accumulate(k=1, end_times=end_times)

    # shape is a leading time axis plus the spectrum shape
    assert acc.shape == (len(end_times),) + jsfs.shape

    # accumulation at a large end time converges to the bin means
    np.testing.assert_allclose(acc[-1], np.asarray(mean), atol=1e-9)

    # each bin's accumulation matches Coalescent.accumulate for that JointSFSReward
    for config in [(1, 0), (1, 1), (2, 0)]:
        single = two_pop_coalescent.accumulate(k=1, end_times=[2.0], rewards=[pg.JointSFSReward(config)])
        assert acc[(1,) + config] == pytest.approx(float(single[0]), abs=1e-9)

    # centered second-moment accumulation and plotting run without error
    assert jsfs.accumulate(k=2, end_times=[2.0], center=True).shape == (1,) + jsfs.shape
    jsfs.plot_accumulation(k=1, end_times=np.linspace(0, 3, 20), show=False)


def test_jsfs_composes_with_existing_rewards(two_pop_coalescent):
    """
    Weighting a joint SFS bin by :class:`DemeReward` partitions it by deme of residence; summing over demes
    recovers the full bin mean. Also checks the cross-bin covariance (diagonal equals variance, symmetric).
    """
    from phasegen.rewards import CombinedReward

    mean = two_pop_coalescent.jsfs.mean

    for config in [(1, 0), (1, 1), (2, 0)]:
        per_deme = sum(
            two_pop_coalescent.moment(k=1, rewards=[CombinedReward([pg.DemeReward(pop), pg.JointSFSReward(config)])])
            for pop in ['pop_0', 'pop_1']
        )
        assert per_deme == pytest.approx(mean[config], abs=1e-9)

    # the cross-bin covariance matches the variance on its diagonal and is symmetric
    cov = two_pop_coalescent.jsfs.cov
    assert cov.shape == (3, 3, 3, 3)
    assert cov[(1, 0) + (1, 0)] == pytest.approx(two_pop_coalescent.jsfs.var[1, 0], abs=1e-9)
    assert cov[(1, 0) + (0, 1)] == pytest.approx(cov[(0, 1) + (1, 0)], abs=1e-9)


def test_jsfs_incompatible_reward_stacking_raises(two_pop_coalescent):
    """
    Stacking a ``JointSFSReward`` with a reward based on a different, incompatible state space (e.g. a
    single-population SFS reward) must raise a clear error rather than silently misroute.
    """
    from phasegen.rewards import CombinedReward

    with pytest.raises(ValueError):
        two_pop_coalescent.moment(k=1, rewards=[CombinedReward([pg.UnfoldedSFSReward(1), pg.JointSFSReward((1, 0))])])


def test_joint_cdf_plot_grid_honours_diagonal_reduction():
    """
    Regression for the ``JointCDF`` plot/surface path (``_grid_values``) skipping the ``R_a = R_b`` diagonal
    reduction that the callable ``__call__`` applies. For a diagonal joint (identical rewards) the law is singular
    on the diagonal and the CDF must equal ``P(R <= min(x, y))``; the plot grid must reproduce the pointwise callable
    exactly. Pre-fix, ``_grid_values`` built the 2D cosine box expansion of the diagonal-singular measure instead, so
    the grid disagreed with ``__call__`` (max abs error ~0.009) and was not constant along ``min(x, y)``.
    """
    d = pg.Coalescent(n=3).joint(pg.TotalBranchLengthReward(), pg.TotalBranchLengthReward())
    assert d._ratio == 1.0

    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([1.0, 2.0, 3.0])

    grid = d.cdf._grid_values(xs, ys)
    pointwise = np.array([[float(d.cdf(x, y)) for y in ys] for x in xs])

    # the plot grid must match the callable evaluated pointwise on the same grid (pre-fix they differed)
    np.testing.assert_allclose(grid, pointwise, atol=1e-12)

    # and it must be constant along each min(x, y) contour, e.g. the x = 1 row is P(R <= 1) throughout
    np.testing.assert_allclose(grid[0], grid[0, 0], atol=1e-12)


def test_jsfs_per_bin_curves_honour_non_unit_reward(symmetric_demography):
    """
    Regression for the ``JointSFS`` per-bin cdf/pdf/quantile aggregate dropping ``self.reward``. Under a non-unit
    spectrum reward (here a deme-restricted view) the per-bin distribution must combine it with the bin's
    ``JointSFSReward`` exactly as the moment/mean/cov paths do; its ``mean`` must equal ``jsfs.moment(1)[config]`` and
    its cdf must differ from the default (unit-reward) per-bin cdf. Pre-fix, the per-bin distribution was built from
    ``JointSFSReward(config)`` alone, ignoring ``self.reward``, giving the wrong (unit-reward) curve.
    """
    from phasegen.distributions.spectra import JointSFSDistribution
    from phasegen.rewards import CombinedReward, JointSFSReward

    coal = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=symmetric_demography({'pop_0': 1.0, 'pop_1': 1.5}, migration_rate=0.75)
    )

    reward = pg.DemeReward('pop_0')
    jsfs = JointSFSDistribution(
        state_space=coal.joint_block_counting_state_space,
        tree_height=coal.tree_height,
        demography=coal.demography,
        reward=reward
    )

    means = jsfs.moment(1)
    config = (0, 1)  # a bin with non-zero deme-restricted mean
    assert means[config] > 1e-6

    # the per-bin distribution built with the combined reward (the fixed path) reproduces the spectrum mean exactly
    bin_dist = jsfs.distribution(reward=CombinedReward([jsfs.reward, JointSFSReward(config)]))
    assert bin_dist.mean == pytest.approx(float(means[config]), abs=1e-9)

    # the public per-bin cdf aggregate honours self.reward, so it matches the combined-reward bin distribution ...
    t = 0.5
    assert float(jsfs.cdf(t)[config]) == pytest.approx(float(bin_dist.cdf(t)), abs=1e-9)

    # ... and differs from the pre-fix behaviour, which used JointSFSReward(config) alone (the unit-reward curve)
    unit_cdf = jsfs.distribution(reward=JointSFSReward(config)).cdf(t)
    assert abs(float(jsfs.cdf(t)[config]) - float(unit_cdf)) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("n, pop_sizes, migration_rate, model, seed", MSPRIME_CASES)
def test_jsfs_mean_matches_msprime(symmetric_demography, n, pop_sizes, migration_rate, model, seed):
    """
    Compare the analytical joint SFS mean against msprime across demographies, initializations and coalescent models.
    """
    demography = symmetric_demography(pop_sizes, migration_rate)
    model_kwargs = {} if model is None else dict(model=model)

    ana = pg.Coalescent(n=n, demography=demography, **model_kwargs).jsfs.mean
    ms = MsprimeCoalescent(
        n=n, demography=demography, num_replicates=100000, n_threads=1, seed=seed, **model_kwargs
    ).jsfs.mean

    np.testing.assert_allclose(np.asarray(ana), np.asarray(ms), atol=0.05, err_msg=f"Mismatch for config {n}")


@pytest.mark.slow
def test_jsfs_second_moment_matches_msprime(symmetric_demography):
    """
    Compare the analytical second (non-central) moment of the joint SFS against msprime.
    """
    n = {'pop_0': 2, 'pop_1': 1}
    demography = symmetric_demography({'pop_0': 1.0, 'pop_1': 1.0})

    ana = pg.Coalescent(n=n, demography=demography).jsfs.moment(k=2, center=False)
    ms = MsprimeCoalescent(n=n, demography=demography, num_replicates=100000, n_threads=1, seed=46).jsfs.m2

    np.testing.assert_allclose(np.asarray(ana), np.asarray(ms), rtol=0.05, atol=0.3)


@pytest.mark.slow
@pytest.mark.parametrize("name", MOMENTS_REFERENCE_CONFIGS)
def test_jsfs_matches_moments(name):
    """
    Compare the analytical joint SFS against an independent ``moments`` reference (precomputed via snakemake;
    skipped if absent). Both spectra are normalized, so the comparison probes the spectrum shape.
    """
    reference_file = MOMENTS_REFERENCE_DIR / f'{name}.json'

    if not reference_file.exists():
        pytest.skip(
            f"moments reference {reference_file} not generated; "
            f"run `snakemake results/jsfs_reference/{name}.json`"
        )

    with open(reference_file) as f:
        reference = json.load(f)

    # rebuild the PhaseGen coalescent directly from the (self-contained) reference metadata
    demography = pg.Demography(
        pop_sizes=reference['pop_sizes'],
        migration_rates={(src, dest): rate for src, dest, rate in reference['migration_rates']}
    )
    jsfs = np.asarray(pg.Coalescent(n=reference['n'], demography=demography).jsfs.mean)
    jsfs = jsfs / jsfs.sum()

    # moments is a diffusion approximation, so allow a small absolute tolerance on the normalized spectrum
    np.testing.assert_allclose(jsfs, np.array(reference['jsfs']), atol=0.01, err_msg=f"Mismatch for config {name}")


def test_jsfs_moment_infinite_end_time(two_pop_coalescent):
    """
    An explicit infinite end time is accumulation until absorption. jsfs.moment passed it on unresolved, which
    exponentiated over an infinite step and raised ValueError for the mean and for windowed moments.
    """
    jsfs = two_pop_coalescent.jsfs

    np.testing.assert_allclose(
        np.asarray(jsfs.moment(k=1, end_time=np.inf).data), np.asarray(jsfs.mean.data), rtol=1e-10, atol=1e-12
    )

    for k in (1, 2):
        np.testing.assert_allclose(
            np.asarray(jsfs.moment(k=k, start_time=0.3, end_time=np.inf).data),
            np.asarray(jsfs.moment(k=k, start_time=0.3).data),
            rtol=1e-8,
            atol=1e-12
        )


def test_jsfs_demes_cov(two_pop_coalescent):
    """
    jsfs.demes.get_cov, cov and corr must evaluate, and the deme covariances of a bin must sum to its variance, since
    the deme branch lengths partition the bin branch length. JointSFSDistribution.moment took no rewards, so all three
    raised TypeError.

    That summation identity is ``Var(sum_p X_p)`` for the per-deme branch lengths ``X_p`` of a bin, so it is the same
    number under any relabelling of the demes, and the other two assertions checked only shapes. A mis-attribution
    inside ``MarginalDemeDistributions.get_cov`` -- looking each deme up one position along the canonical deme axis,
    which would corrupt every per-deme covariance and correlation of the tree height, total branch length, SFS and
    joint SFS alike -- left the whole of this file and testing/test_rewards.py green. The diagonal entries are pinned
    against each deme's own variance, computed by a separate call of the moment engine, and the two demes here differ
    by up to 0.436 in that diagonal, so a permutation of the deme axis breaks it.
    """
    jsfs = two_pop_coalescent.jsfs
    pops = jsfs.lineage_config.pop_names

    total = sum(np.asarray(jsfs.demes.get_cov(p, q).data) for p in pops for q in pops)

    np.testing.assert_allclose(total, np.asarray(jsfs.var.data), rtol=1e-8, atol=1e-12)
    assert np.asarray(jsfs.demes.cov).shape == (len(pops), len(pops)) + jsfs.shape
    assert np.asarray(jsfs.demes.corr).shape == (len(pops), len(pops)) + jsfs.shape

    cov = np.asarray(jsfs.demes.cov)
    corr = np.asarray(jsfs.demes.corr)
    variances = [np.asarray(jsfs.demes[p].var.data) for p in pops]

    # the demes are told apart: they differ in population size, so their bin variances differ
    assert np.abs(variances[0] - variances[1]).max() > 0.1

    for i, p in enumerate(pops):
        np.testing.assert_allclose(np.asarray(jsfs.demes.get_cov(p, p).data), variances[i], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(cov[i, i], variances[i], rtol=1e-8, atol=1e-12)

    # the off-diagonal correlation is built from the same two demes as the off-diagonal covariance
    off = np.asarray(jsfs.demes.get_cov(pops[0], pops[1]).data)
    np.testing.assert_allclose(cov[0, 1], off, rtol=1e-8, atol=1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = off / np.sqrt(variances[0] * variances[1])
    finite = np.isfinite(expected)
    np.testing.assert_allclose(corr[0, 1][finite], expected[finite], rtol=1e-8, atol=1e-12)


def test_jsfs_joint_restricted_by_spectrum_reward(two_pop_coalescent):
    """
    The joint distribution of two bins of a deme view must carry the view's reward, so its marginal means and
    covariance equal those of the view. It used the bare bin rewards and so described the full joint spectrum.
    """
    view = two_pop_coalescent.jsfs.demes['pop_0']
    a, b = (1, 0), (0, 1)

    jd = view.joint(a, b)

    np.testing.assert_allclose(jd.mean, [view.mean.data[a], view.mean.data[b]], rtol=1e-10)
    np.testing.assert_allclose(jd.cov, view.get_cov(a, b), rtol=1e-8)


@pytest.mark.parametrize('config', [(3, 0), (1,), (1, 0, 0), (0, 0), (2, 2), (0.5, 1), (True, 0)])
def test_jsfs_invalid_config_raises_value_error(two_pop_coalescent, config):
    """An out-of-range, wrongly sized, monomorphic or non-integral descendant configuration raises ValueError from
    every per-bin entry point. Regression: a bare KeyError from the reward, the absorbing-state error for the full
    configuration, a silently floored non-integral entry, and a failure deferred to first use."""
    jsfs = two_pop_coalescent.jsfs

    for call in (
            lambda: jsfs.bin(*config),
            lambda: jsfs.get_cov(config, (1, 0)),
            lambda: jsfs.joint(config, (1, 0)),
            lambda: jsfs.cdf._plot_data(configs=[config], t=[1.0])
    ):
        with pytest.raises(ValueError, match='descendant configuration'):
            call()


def test_jsfs_valid_configs_pass_validation(two_pop_coalescent):
    """Every polymorphic configuration passes validation unchanged, as integers."""
    jsfs = two_pop_coalescent.jsfs

    assert [jsfs._bin_config(c) for c in jsfs._get_configs()] == list(jsfs._get_configs())
    assert jsfs._bin_config(np.array([1, 0])) == (1, 0) and jsfs._bin_config((1.0, 2.0)) == (1, 2)


def test_bin_distributions_are_served_when_the_cache_is_off(two_pop_coalescent):
    """With ``Settings.cache`` off, a stored per-bin distribution is still served and a new one is not stored.
    Regression: the stored entry was bypassed and its fit rebuilt on every call."""
    for spectrum, key, other in ((pg.Coalescent(n=4).sfs, 1, 2), (two_pop_coalescent.jsfs, (1, 0), (0, 1))):
        stored = spectrum._bin_distribution(key)
        pg.Settings.cache = False

        assert spectrum._bin_distribution(key) is stored
        assert spectrum._bin_distribution(other) is not spectrum._bin_distribution(other)
        pg.Settings.cache = True


def test_empirical_joint_spectrum_keeps_its_statistics_when_dropped():
    """The fourth moment is a joint spectrum as the lower ones, and the covariance survives freeing the samples."""
    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))
    e = coal.jsfs.to_empirical(2000, seed=1)
    cov = e.cov

    assert isinstance(e.m4, type(e.m3))

    e._drop()
    np.testing.assert_array_equal(e.cov, cov)
    assert {'mean', 'var', 'cov'} <= set(e._standard_errors)


def test_joint_plot_accumulation_takes_a_label():
    """The joint spectrum plots its accumulation through the shared method, with a legend label."""
    import matplotlib
    matplotlib.use('Agg')

    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))

    ax = coal.jsfs.plot_accumulation(end_times=[0.5, 1.0], show=False, label='x')
    assert ax.get_lines()


def test_joint_sfs_moments_keep_population_names_exact_sampled_and_msprime():
    """
    The mean, variance, standard deviation and second moment of the joint SFS carry the population names in the
    exact, sampled and msprime coalescents alike, so their plots label the axes by population.
    """
    from phasegen.distributions import MsprimeCoalescent

    demo = pg.Demography(pop_sizes={'CEU': 1, 'CHB': 1}, migration_rates={('CEU', 'CHB'): 1, ('CHB', 'CEU'): 1})
    n = {'CEU': 2, 'CHB': 2}
    exact = pg.Coalescent(n=n, demography=demo)

    for coal in [exact, exact.to_empirical(200, seed=1),
                 MsprimeCoalescent(n=n, demography=demo, num_replicates=200, seed=1, parallelize=False)]:
        jsfs = coal.jsfs
        for spectrum in (jsfs.mean, jsfs.var, jsfs.std, jsfs.moment(2)):
            assert spectrum.pop_names == ['CEU', 'CHB'], type(coal).__name__

