"""
Tests for the public ``sample`` / ``to_empirical`` methods of the distributions and the
:class:`~phasegen.distributions.SampledCoalescent`. These check that PhaseGen's own trajectory sampler reproduces
the exact analytic statistics (up to Monte-Carlo error), the self-consistency that the scenario suite exercises
end-to-end via the nested ``tolerance.empirical`` blocks in ``test_scenarios.py`` configs.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import SampledCoalescent
from phasegen.distributions.coalescent import AbstractCoalescent
from phasegen.distributions.empirical import MsprimeCoalescent

N_SAMPLES = 50000


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


def test_sample_scalar_stat_shapes_and_mean():
    """Sampling a scalar statistic returns ``(n_samples,)`` and reproduces the analytic mean."""
    for dist in (pg.Coalescent(n=6).tree_height, pg.Coalescent(n=6).total_branch_length):
        s = dist.sample(N_SAMPLES)
        assert s.shape == (N_SAMPLES,)
        assert s.mean() == pytest.approx(dist.mean, rel=0.02)


def test_sample_sfs_shape_and_mean():
    """SFS ``sample`` returns ``(n_samples, n + 1)`` whose mean matches the analytic SFS."""
    sfs = pg.Coalescent(n=6).sfs
    s = sfs.sample(N_SAMPLES)
    assert s.shape == (N_SAMPLES, 7)
    np.testing.assert_allclose(s.mean(axis=0), np.asarray(sfs.mean.data), atol=0.05)


def test_sample_jsfs_shape_and_mean():
    """Joint SFS ``sample`` returns ``(n_samples, *shape)`` whose mean matches the analytic joint SFS."""
    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1.5},
                        migration_rates={('p0', 'p1'): 0.75, ('p1', 'p0'): 0.75})
    jsfs = pg.Coalescent(n={'p0': 3, 'p1': 3}, demography=dem).jsfs
    s = jsfs.sample(N_SAMPLES)
    assert s.shape == (N_SAMPLES,) + jsfs.shape
    np.testing.assert_allclose(s.mean(axis=0), np.asarray(jsfs.mean.data), atol=0.05)


def test_sample_sfs2_outer_product_mean():
    """Two-locus SFS ``sample`` returns ``(n_samples, n+1, n+1)`` matching the (symmetrized) cross-moment mean."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    s = sfs2.sample(N_SAMPLES)
    assert s.shape == (N_SAMPLES, 5, 5)
    np.testing.assert_allclose(s.mean(axis=0), np.asarray(sfs2.mean.data), atol=0.15)


def test_to_empirical_per_deme_and_locus_match_analytic():
    """The empirical per-deme / per-locus breakdowns reproduce the analytic marginals."""
    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1.5},
                        migration_rates={('p0', 'p1'): 0.75, ('p1', 'p0'): 0.75})
    th = pg.Coalescent(n={'p0': 3, 'p1': 3}, demography=dem).tree_height
    e = th.to_empirical(N_SAMPLES)

    assert e.mean == pytest.approx(th.mean, rel=0.02)
    for p in ('p0', 'p1'):
        assert e.demes[p].mean == pytest.approx(th.demes[p].mean, rel=0.03)

    # per-locus breakdown on a two-locus tree height
    th2 = pg.Coalescent(n=3, loci=2, recombination_rate=1.0).tree_height
    e2 = th2.to_empirical(N_SAMPLES)
    for locus in (0, 1):
        assert e2.loci[locus].mean == pytest.approx(th2.loci[locus].mean, rel=0.03)


def test_to_empirical_sfs2_cross_moment():
    """The empirical two-locus cross-moment reproduces the analytic two-locus SFS entry."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    e = sfs2.to_empirical(N_SAMPLES)
    assert e.cross_moment(1, 1) == pytest.approx(np.asarray(sfs2.mean.data)[1, 1], rel=0.05)


def test_empirical_joint_marginal_conditional_match_analytic():
    """The empirical joint (sampler) marginals and conditionals reproduce the exact
    :class:`~phasegen.distributions.JointRewardDistribution` ones — the sanity check
    :class:`~phasegen.distributions.EmpiricalJointRewardDistribution` enables."""
    coal = pg.Coalescent(n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.25: 0.08, 0.7: 1.0}}))
    ana = coal.sfs.joint_distribution(1, 2)
    emp = coal.sfs.to_empirical(200000).joint_distribution(1, 2)

    assert emp.corr() == pytest.approx(ana.corr(), abs=0.03)
    np.testing.assert_allclose(emp.mean, ana.mean, rtol=0.03)
    assert emp.cdf(1.5, 0.5) == pytest.approx(ana.cdf(1.5, 0.5), abs=0.02)

    # the marginal reproduces the analytic marginal (and the direct bin)
    assert emp.marginal('a').mean == pytest.approx(ana.marginal('a').mean, rel=0.02)
    assert emp.marginal('b').mean == pytest.approx(coal.sfs.bin(2).mean, rel=0.02)

    # the conditional shifts with the (negative) correlation, matching the analytic conditional mean
    for v in (0.5, 1.3):
        assert emp.conditional('a', v).mean == pytest.approx(ana.conditional('a', v).mean, abs=0.04)

    # an explicit window is honoured; invalid selectors raise
    assert emp.conditional('a', 0.5, window=0.1).samples.size < emp.conditional('a', 0.5, window=0.5).samples.size
    with pytest.raises(ValueError):
        emp.marginal('c')
    with pytest.raises(ValueError):
        emp.conditional('c', 0.5)


def test_coalescent_to_empirical_returns_sampled_coalescent():
    """``Coalescent.to_empirical`` mirrors ``to_msprime``: it returns a :class:`SampledCoalescent` whose per-statistic
    distributions match the exact analytic coalescent."""
    coal = pg.Coalescent(n=6)
    emp = coal.to_empirical(N_SAMPLES, seed=42)

    assert isinstance(emp, SampledCoalescent)
    assert emp.n_samples == N_SAMPLES
    assert emp.tree_height.mean == pytest.approx(coal.tree_height.mean, rel=0.02)
    np.testing.assert_allclose(np.asarray(emp.sfs.mean), np.asarray(coal.sfs.mean.data), atol=0.05)


def test_to_empirical_exposes_n_samples():
    """``to_empirical`` records the sample count on the empirical object, surviving ``drop``."""
    e = pg.Coalescent(n=5).tree_height.to_empirical(12345)
    assert e.n_samples == 12345
    e.touch(np.linspace(0, 5, 20))
    e.drop()
    assert e.n_samples == 12345  # retained for the serialized fixture

    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1},
                        migration_rates={('p0', 'p1'): 1, ('p1', 'p0'): 1})
    assert pg.Coalescent(n={'p0': 2, 'p1': 2}, demography=dem).jsfs.to_empirical(9999).n_samples == 9999
    assert pg.Coalescent(n=3, loci=2, recombination_rate=1).sfs2.to_empirical(8888).n_samples == 8888


def test_tree_height_per_deme_gated_for_multiple_loci():
    """Per-deme tree height is ill-posed under recombination (max over loci is not additively decomposable), so the
    accessor raises for multiple loci; the additive ``total_branch_length.demes`` is available instead."""
    c = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
    with pytest.raises(NotImplementedError):
        _ = c.tree_height.demes

    # the additive per-deme decomposition is well-defined and sums to the total
    dem = pg.Demography(pop_sizes={'p0': 2, 'p1': 1},
                        migration_rates={('p0', 'p1'): 1, ('p1', 'p0'): 1})
    tbl = pg.Coalescent(n={'p0': 1, 'p1': 1}, loci=2, recombination_rate=1.0, demography=dem).total_branch_length
    assert tbl.demes['p0'].mean + tbl.demes['p1'].mean == pytest.approx(tbl.mean, rel=1e-6)

    # single-locus per-deme tree height remains available
    assert pg.Coalescent(n={'p0': 2, 'p1': 2}, demography=dem).tree_height.demes['p0'].mean > 0


def test_batched_sampling_matches_single_pass():
    """Batching the ensemble (small ``sample_batch_size``) preserves shape and the CTMC law, including across epochs."""
    from scipy import stats
    from phasegen.settings import Settings

    dem = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 1.0: 2.0}})  # piecewise-constant: forces an epoch crossing
    saved = Settings.sample_batch_size
    try:
        for c in (pg.Coalescent(n=8), pg.Coalescent(n=8, demography=dem)):
            d = c.tree_height
            np.random.seed(7)
            Settings.sample_batch_size = None
            single = d.sample(20000)
            np.random.seed(7)
            Settings.sample_batch_size = 2500  # several batches incl. a short final one
            batched = d.sample(20000)
            assert batched.shape == single.shape == (20000,)
            assert stats.ks_2samp(single, batched).pvalue > 0.01
            assert batched.mean() == pytest.approx(d.mean, rel=0.02)
    finally:
        Settings.sample_batch_size = saved


def test_sampled_coalescent_matches_analytic():
    """``SampledCoalescent`` exposes empirical distributions consistent with the analytic coalescent."""
    c = pg.Coalescent(n=6)
    sampled = SampledCoalescent(coalescent=c, n_samples=N_SAMPLES, seed=42)

    assert sampled.tree_height.mean == pytest.approx(c.tree_height.mean, rel=0.02)
    np.testing.assert_allclose(np.asarray(sampled.sfs.mean), np.asarray(c.sfs.mean.data), atol=0.05)
    np.testing.assert_allclose(np.asarray(sampled.fsfs.mean), np.asarray(c.fsfs.mean.data), atol=0.05)


def test_sampled_and_msprime_share_facade():
    """``SampledCoalescent`` and ``MsprimeCoalescent`` implement the same ``AbstractCoalescent`` facade, so the
    comparison framework can use them interchangeably as the empirical (candidate) operand."""
    sampled = SampledCoalescent(coalescent=pg.Coalescent(n=6), n_samples=100)
    ms = MsprimeCoalescent(n=6)  # cheap: msprime simulation is lazy (triggered on stat access, not construction)

    assert isinstance(sampled, AbstractCoalescent) and isinstance(ms, AbstractCoalescent)

    # the per-statistic distributions and lifecycle hooks the comparison framework relies on (checked on the class
    # to avoid triggering the lazy cached_property simulations)
    for name in ('tree_height', 'total_branch_length', 'sfs', 'fsfs', 'jsfs', 'sfs2', 'touch', 'drop'):
        assert hasattr(SampledCoalescent, name) and hasattr(MsprimeCoalescent, name), name

    # the delegated configuration both expose as instance attributes
    for name in ('lineage_config', 'locus_config', 'demography', 'model', 'n'):
        assert hasattr(sampled, name) and hasattr(ms, name), name


@pytest.mark.slow
def test_msprime_touch_grids_sfs_on_its_own_support():
    """``MsprimeCoalescent.touch`` must cache each spectrum on its **own** support, not the tree-height grid.

    Regression for the bug where ``touch`` passed the tree-height grid ``t = _get_cached_times(self.tree_height)`` to
    ``self.sfs.touch`` / ``self.fsfs.touch``. Individual SFS bin branch lengths are not bounded by the tree height (the
    summed singleton branches routinely exceed the TMRCA), so caching the SFS cdf/pdf on the tree-height grid truncated
    the SFS tail: the serialized cdf never reached 1 and the comparison asserted nothing above the tree height. The fix
    passes ``_get_cached_times(self.sfs)`` / ``_get_cached_times(self.fsfs)`` instead.
    """
    ms = MsprimeCoalescent(n=6, num_replicates=2000, n_threads=1, parallelize=False, seed=42)
    ms.touch()

    # the SFS's own support genuinely extends beyond the tree height (else this test would assert nothing)
    assert np.max(ms.sfs.samples) > np.max(ms.tree_height.samples)
    assert np.max(ms.fsfs.samples) > np.max(ms.tree_height.samples)

    tree_grid = ms.tree_height._cache['t']

    # the grid the SFS/fSFS were cached on equals their own support grid, and differs from (extends beyond) the
    # tree-height grid -- pre-fix both were touched with ``tree_grid`` and the tail above it was truncated
    for dist in (ms.sfs, ms.fsfs):
        np.testing.assert_array_equal(dist._cache['t'], MsprimeCoalescent._get_cached_times(dist))
        assert dist._cache['t'][-1] > tree_grid[-1]
        assert dist._cache['t'][-1] == pytest.approx(float(np.max(dist.samples)))


@pytest.mark.slow
def test_msprime_jsfs_n_samples_is_actual_replicate_count():
    """The empirical joint SFS ``n_samples`` must be the actual averaged replicate count ``n_total``, not the requested
    ``num_replicates``.

    Regression for the bug where ``jsfs`` recorded ``n_samples=self.num_replicates`` while the moments are normalised
    over ``n_total = (num_replicates // n_threads) * n_threads``. When ``num_replicates`` is not a multiple of
    ``n_threads`` the two differ (here 103 requested, 100 simulated), overstating the replicate count and biasing the
    tolerance tuner's noise floor tighter than the true sampling error. The fix passes ``n_samples=n_total``.
    """
    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1},
                        migration_rates={('p0', 'p1'): 1, ('p1', 'p0'): 1})
    # 103 is not divisible by 4 threads -> 25 per thread -> 100 replicates actually simulated and averaged
    ms = MsprimeCoalescent(n={'p0': 2, 'p1': 2}, demography=dem,
                           num_replicates=103, n_threads=4, parallelize=False, seed=1)

    assert ms.jsfs.n_samples == ms.n_total == 100
    assert ms.n_total != ms.num_replicates  # pre-fix n_samples was num_replicates (103)


@pytest.mark.slow
def test_sampled_and_msprime_agree():
    """The two empirical backends sample the *same* coalescent process, so their shared statistics agree within
    Monte-Carlo error (a cross-check independent of the analytic reference)."""
    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1.5},
                        migration_rates={('p0', 'p1'): 0.75, ('p1', 'p0'): 0.75})
    n = {'p0': 3, 'p1': 3}
    reps = 100000

    sampled = SampledCoalescent(coalescent=pg.Coalescent(n=n, demography=dem), n_samples=reps, seed=1)
    ms = MsprimeCoalescent(n=n, demography=dem, num_replicates=reps, seed=1, parallelize=False)

    assert sampled.tree_height.mean == pytest.approx(ms.tree_height.mean, rel=0.03)
    assert sampled.total_branch_length.mean == pytest.approx(ms.total_branch_length.mean, rel=0.03)
    np.testing.assert_allclose(np.asarray(sampled.sfs.mean), np.asarray(ms.sfs.mean), rtol=0.05, atol=0.05)
    np.testing.assert_allclose(np.asarray(sampled.jsfs.mean), np.asarray(ms.jsfs.mean), rtol=0.1, atol=0.05)


def _assert_same_law(sampled, ms, rel: float = 0.03, atol: float = 0.02):
    """Assert two empirical coalescents sample the same law: the means, and the whole tree-height CDF.

    The CDF is what makes this bite. Means are insensitive to *where* the mass sits, so a sampler that mishandled an
    epoch boundary -- putting coalescences on the wrong side of it -- could still land the mean. Both CDFs are
    empirical, so the scale is the two-sample Kolmogorov-Smirnov noise, ``~1.4 sqrt(2 / n)``; ``atol`` sits an order of
    magnitude above it, making this a structural check rather than a precision bound.
    """
    assert sampled.tree_height.mean == pytest.approx(ms.tree_height.mean, rel=rel)
    assert sampled.total_branch_length.mean == pytest.approx(ms.total_branch_length.mean, rel=rel)
    np.testing.assert_allclose(np.asarray(sampled.sfs.mean), np.asarray(ms.sfs.mean), rtol=0.05, atol=0.05)

    t = np.linspace(0, float(ms.tree_height.quantile(0.99)), 50)
    np.testing.assert_allclose(sampled.tree_height.cdf(t), ms.tree_height.cdf(t), atol=atol)


@pytest.mark.slow
def test_sampled_and_msprime_agree_across_epochs():
    """The sampler against msprime under a **time-inhomogeneous** demography -- the one place its epoch handling can
    be wrong without anything else noticing.

    Within an epoch the sampler takes ``H / lambda`` from an ``Exp(1)`` hazard budget; at a boundary it advances the
    walker to the boundary, consumes ``lambda * duration`` of the budget, and carries the remainder into the next
    epoch. Nothing else validates that carry-over against an *independent* implementation: the scenario suite's
    ``tolerance.empirical`` blocks compare the sampler against PhaseGen's own analytics, which share its epoch grid, so
    a bug in the grid would agree with itself.

    The bottleneck is deep and short, so a walker that mis-crossed a boundary would coalesce in the wrong epoch.
    """
    dem = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.3: 0.05, 0.5: 1.0}})
    reps = 100000

    sampled = SampledCoalescent(coalescent=pg.Coalescent(n=6, demography=dem), n_samples=reps, seed=2)
    ms = MsprimeCoalescent(n=6, demography=dem, num_replicates=reps, seed=2, parallelize=False)

    _assert_same_law(sampled, ms)


def test_sampler_is_scale_equivariant():
    """Rescaling every population size by ``c`` rescales every sampled time by exactly ``c``.

    The sampler draws ``H / lambda`` from a hazard budget, which carries no absolute time scale, so this must hold to
    the last digit rather than merely within Monte-Carlo error -- and the same seed gives the same trajectories, so the
    sampled means are compared as an identity, not a statistic. Worth pinning: an absolute constant slipped into a
    scale-free computation is a recurring failure here (the atom probe once tested ``phi(1e8)`` rather than
    ``phi(1e8 / tau)``, which invented atoms for small populations), and no scenario config samples at an extreme
    ``N``.
    """
    ref = None
    for scale in (1e-8, 1e-4, 1.0, 1e4, 1e8):
        c = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': scale}))
        sampled = c.to_empirical(20000, seed=11)

        # the mean in units of the population size: identical across scales, and equal to the analytic value
        normalized = float(sampled.tree_height.mean) / scale
        assert normalized == pytest.approx(float(c.tree_height.mean) / scale, rel=0.02)

        if ref is None:
            ref = normalized
        assert normalized == pytest.approx(ref, rel=1e-12)


@pytest.mark.slow
def test_sampled_and_msprime_agree_across_a_zero_rate_epoch():
    """The sampler against msprime when an epoch has **no migration at all**, so the demes are isolated until they
    reconnect.

    This is the ``lambda = 0`` branch of the hazard budget: a walker in a state it cannot leave consumes no hazard and
    simply waits out the epoch, accruing reward. Getting that wrong (consuming budget, or dividing by a zero rate)
    would be invisible to a time-homogeneous test, and the isolated phase forces the two demes' lineages to survive it
    before they can ever coalesce with one another.
    """
    dem = pg.Demography(
        pop_sizes={'p0': 1.0, 'p1': 1.0},
        migration_rates={('p0', 'p1'): {0: 0.0, 0.5: 1.0}, ('p1', 'p0'): {0: 0.0, 0.5: 1.0}}
    )
    n = {'p0': 2, 'p1': 2}
    reps = 100000

    sampled = SampledCoalescent(coalescent=pg.Coalescent(n=n, demography=dem), n_samples=reps, seed=3)
    ms = MsprimeCoalescent(n=n, demography=dem, num_replicates=reps, seed=3, parallelize=False)

    _assert_same_law(sampled, ms)
