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

#: Seed passed explicitly to every sampler call, which draws from its own ``numpy.random.Generator``.
SEED = 42


def test_sample_scalar_stat_shapes_and_mean():
    """Sampling a scalar statistic returns ``(n_samples,)`` and reproduces the analytic mean."""
    for dist in (pg.Coalescent(n=6).tree_height, pg.Coalescent(n=6).total_branch_length):
        s = dist.sample(N_SAMPLES, seed=SEED)
        assert s.shape == (N_SAMPLES,)
        assert s.mean() == pytest.approx(dist.mean, rel=0.02)


def test_sample_sfs_shape_and_mean():
    """SFS ``sample`` returns ``(n_samples, n + 1)`` whose mean matches the analytic SFS."""
    sfs = pg.Coalescent(n=6).sfs
    s = sfs.sample(N_SAMPLES, seed=SEED)
    assert s.shape == (N_SAMPLES, 7)
    np.testing.assert_allclose(s.mean(axis=0), np.asarray(sfs.mean.data), atol=0.05)


def test_sample_jsfs_shape_and_mean():
    """Joint SFS ``sample`` returns ``(n_samples, *shape)`` whose mean matches the analytic joint SFS."""
    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1.5},
                        migration_rates={('p0', 'p1'): 0.75, ('p1', 'p0'): 0.75})
    jsfs = pg.Coalescent(n={'p0': 3, 'p1': 3}, demography=dem).jsfs
    s = jsfs.sample(N_SAMPLES, seed=SEED)
    assert s.shape == (N_SAMPLES,) + jsfs.shape
    np.testing.assert_allclose(s.mean(axis=0), np.asarray(jsfs.mean.data), atol=0.05)


def test_sample_sfs2_outer_product_mean():
    """Two-locus SFS ``sample`` returns ``(n_samples, n+1, n+1)`` matching the (symmetrized) cross-moment mean."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    s = sfs2.sample(N_SAMPLES, seed=SEED)
    assert s.shape == (N_SAMPLES, 5, 5)
    np.testing.assert_allclose(s.mean(axis=0), np.asarray(sfs2.mean.data), atol=0.15)


def test_to_empirical_per_deme_and_locus_match_analytic():
    """The empirical per-deme / per-locus breakdowns reproduce the analytic marginals."""
    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1.5},
                        migration_rates={('p0', 'p1'): 0.75, ('p1', 'p0'): 0.75})
    th = pg.Coalescent(n={'p0': 3, 'p1': 3}, demography=dem).tree_height
    e = th.to_empirical(N_SAMPLES, seed=SEED)

    assert e.mean == pytest.approx(th.mean, rel=0.02)
    for p in ('p0', 'p1'):
        assert e.demes[p].mean == pytest.approx(th.demes[p].mean, rel=0.03)

    # per-locus breakdown on a two-locus tree height
    th2 = pg.Coalescent(n=3, loci=2, recombination_rate=1.0).tree_height
    e2 = th2.to_empirical(N_SAMPLES, seed=SEED)
    for locus in (0, 1):
        assert e2.loci[locus].mean == pytest.approx(th2.loci[locus].mean, rel=0.03)


def test_empirical_deme_covariance_matches_the_deme_distributions():
    """
    ``demes.cov`` must be the covariance of the per-deme samples that ``demes`` holds, and its standard error must be
    computed from them. For the tree height the matrix was built from the maximum over loci while the deme
    distributions sum over loci, so its diagonal was about half their variance.
    """
    dem = pg.Demography(pop_sizes={'a': {0: 1}, 'b': {0: 1}}, migration_rates={('a', 'b'): {0: 0.5}, ('b', 'a'): {0: 0.5}})
    th = pg.Coalescent(n={'a': 2, 'b': 1}, demography=dem, loci=2, recombination_rate=1.0).tree_height
    e = th.to_empirical(2000, seed=SEED)

    per_deme = np.array([e.demes[p].samples for p in e.pops])

    np.testing.assert_allclose(e.demes.cov, np.cov(per_deme, bias=True), rtol=1e-12)
    np.testing.assert_allclose(np.diag(e.demes.cov), [e.demes[p].var for p in e.pops], rtol=1e-12)

    e._cache_standard_errors()
    np.testing.assert_allclose(
        e._standard_errors['demes.cov'],
        e._matrix_block_standard_error(per_deme, lambda x: np.cov(x, bias=True), 100),
        rtol=1e-12
    )


def test_to_empirical_sfs2_cross_moment():
    """The empirical two-locus cross-moment reproduces the analytic two-locus SFS entry."""
    sfs2 = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).sfs2
    e = sfs2.to_empirical(N_SAMPLES, seed=SEED)
    assert e.cross_moment(1, 1) == pytest.approx(np.asarray(sfs2.mean.data)[1, 1], rel=0.05)


def test_empirical_joint_marginal_conditional_match_analytic():
    """The empirical joint (sampler) marginals and conditionals reproduce the exact
    :class:`~phasegen.distributions.JointRewardDistribution` ones — the sanity check
    :class:`~phasegen.distributions.EmpiricalJointDistribution` enables."""
    coal = pg.Coalescent(n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.25: 0.08, 0.7: 1.0}}))
    ana = coal.sfs.joint_distribution(1, 2)
    emp = coal.sfs.to_empirical(200000, seed=SEED).joint_distribution(1, 2)

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
    """``to_empirical`` records the sample count on the empirical object, surviving ``_drop``."""
    e = pg.Coalescent(n=5).tree_height.to_empirical(12345, seed=SEED)
    assert e.n_samples == 12345
    e._touch(np.linspace(0, 5, 20))
    e._drop()
    assert e.n_samples == 12345  # retained for the serialized fixture

    dem = pg.Demography(pop_sizes={'p0': 1, 'p1': 1},
                        migration_rates={('p0', 'p1'): 1, ('p1', 'p0'): 1})
    assert pg.Coalescent(n={'p0': 2, 'p1': 2}, demography=dem).jsfs.to_empirical(9999, seed=SEED).n_samples == 9999
    assert pg.Coalescent(n=3, loci=2, recombination_rate=1).sfs2.to_empirical(8888, seed=SEED).n_samples == 8888


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
            Settings.sample_batch_size = None
            single = d.sample(20000, seed=7)
            Settings.sample_batch_size = 2500  # several batches incl. a short final one
            batched = d.sample(20000, seed=7)
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
    sampled = SampledCoalescent(coalescent=pg.Coalescent(n=6), n_samples=100, seed=SEED)
    ms = MsprimeCoalescent(n=6)  # cheap: msprime simulation is lazy (triggered on stat access, not construction)

    assert isinstance(sampled, AbstractCoalescent) and isinstance(ms, AbstractCoalescent)

    # the per-statistic distributions and lifecycle hooks the comparison framework relies on (checked on the class
    # to avoid triggering the lazy cached_property simulations)
    for name in ('tree_height', 'total_branch_length', 'sfs', 'fsfs', 'jsfs', 'sfs2', '_touch', '_drop'):
        assert hasattr(SampledCoalescent, name) and hasattr(MsprimeCoalescent, name), name

    # the delegated configuration both expose as instance attributes
    for name in ('lineage_config', 'locus_config', 'demography', 'model', 'n'):
        assert hasattr(sampled, name) and hasattr(ms, name), name


@pytest.mark.slow
def test_msprime_touch_grids_sfs_on_its_own_support():
    """``MsprimeCoalescent._touch`` must cache each spectrum on its **own** support, not the tree-height grid.

    Regression for the bug where ``_touch`` passed the tree-height grid ``t = _get_cached_times(self.tree_height)`` to
    ``self.sfs._touch`` / ``self.fsfs._touch``. SFS bin branch lengths are not bounded by the tree height (the
    summed singleton branches routinely exceed the TMRCA), so caching the SFS cdf/pdf on the tree-height grid truncated
    the SFS tail: the serialized cdf never reached 1 and the comparison asserted nothing above the tree height. The fix
    passes ``_get_cached_times(self.sfs)`` / ``_get_cached_times(self.fsfs)`` instead.
    """
    ms = MsprimeCoalescent(n=6, num_replicates=2000, n_threads=1, parallelize=False, seed=42)
    ms._touch()

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


def test_empirical_cdf_is_zero_below_the_smallest_sample():
    """The empirical CDF interpolated between ``(Y_(m), m / N)`` and was clamped to ``1 / N`` below the smallest
    sample, so it reported positive probability for values no realisation reached. It must be zero there, for scalar
    and per-bin samples, and keep the post-jump value at an atom."""
    e = pg.distributions.EmpiricalDistribution([1.0, 2.0, 3.0, 4.0])
    assert e.cdf(0.5) == 0.0
    assert e.cdf(-10.0) == 0.0
    assert e.cdf(1.0) == pytest.approx(0.25)

    spectrum = pg.distributions.EmpiricalSFSDistribution([[0.0, 1.0, 0.0], [0.0, 2.0, 0.5], [0.0, 3.0, 1.0]])
    np.testing.assert_array_equal(spectrum.cdf(-1.0), np.zeros(3))
    assert spectrum.cdf(0.0)[2] == pytest.approx(1 / 3)  # the atom at zero holds one of three realisations


def test_empirical_var_is_diagonal_of_cov():
    """``var`` normalised by ``1 / N`` while ``cov`` normalised by ``1 / (N - 1)``, so the variance of a bin
    disagreed with the diagonal of the covariance matrix. Every empirical covariance uses ``1 / N``."""
    rng = np.random.default_rng(0)
    samples = rng.exponential(size=(50, 4))

    e = pg.distributions.EmpiricalDistribution(samples)
    np.testing.assert_allclose(np.diag(e.cov), e.var, rtol=1e-12)
    np.testing.assert_allclose(e.var, e.moment(2), rtol=1e-12)

    spectrum = pg.distributions.EmpiricalSFSDistribution(samples)
    np.testing.assert_allclose(np.diag(np.asarray(spectrum.cov.data)), np.asarray(spectrum.var.data), rtol=1e-12)

    joint = pg.distributions.EmpiricalJointDistribution(samples[:, 0], samples[:, 1])
    assert joint.cov() == pytest.approx(e.cov[0, 1], rel=1e-12)

    per_deme = pg.distributions.EmpiricalPhaseTypeDistribution(samples.T.reshape(1, 4, 50), pops=list('abcd'))
    np.testing.assert_allclose(np.diag(per_deme.demes.cov), [per_deme.demes[p].var for p in 'abcd'], rtol=1e-12)


def test_sampled_sfs_has_no_mutation_configs():
    """``SFSDistribution.to_empirical`` stored all-zero mutation counts, so ``mutation_configs`` reported mass one on
    the configuration without mutations. A spectrum sampled from branch lengths carries no mutation counts, so the
    configuration accessors raise, while touching and dropping it for a comparison still works."""
    sfs = pg.Coalescent(n=4).sfs.to_empirical(500, seed=0)

    with pytest.raises(ValueError, match="no mutation counts"):
        _ = sfs.mutation_configs
    with pytest.raises(ValueError, match="no mutation counts"):
        sfs.get_mutation_config([0, 0, 0])
    with pytest.raises(ValueError, match="no mutation counts"):
        next(sfs.get_mutation_configs())

    sfs._touch(np.linspace(0, 2, 5))
    sfs._drop()
    with pytest.raises(ValueError, match="no mutation counts"):
        _ = sfs.mutation_configs


def test_sampled_coalescent_accepts_a_generator_seed():
    """``Coalescent.to_empirical`` raised TypeError at the first statistic access when given a
    ``numpy.random.Generator``, which every per-distribution sampler accepts. Equally seeded generators must yield
    equal statistics."""
    coal = pg.Coalescent(n=3)

    a = coal.to_empirical(500, seed=np.random.default_rng(SEED))
    b = coal.to_empirical(500, seed=np.random.default_rng(SEED))

    assert isinstance(a.seed, int)
    assert np.isfinite(a.tree_height.mean)
    np.testing.assert_array_equal(a.tree_height.samples, b.tree_height.samples)
    np.testing.assert_array_equal(a.sfs.samples, b.sfs.samples)


def test_msprime_mutation_configs_survive_drop_and_serialization():
    """The configuration frequencies of a spectrum with mutation counts are persisted by ``_touch``, remain available
    after ``_drop`` frees the counts, and are restored by jsonpickle under the serialized key ``mutation_configs``."""
    import jsonpickle

    ms = MsprimeCoalescent(n=4, num_replicates=200, n_threads=1, parallelize=False, simulate_mutations=True,
                           mutation_rate=1.0, seed=1)
    sfs = ms.sfs
    expected = dict(sfs.mutation_configs)
    assert sum(expected.values()) == pytest.approx(1.0)

    sfs._touch(np.linspace(0, 2, 5))
    sfs._drop()
    assert dict(sfs.mutation_configs) == expected

    restored = jsonpickle.decode(jsonpickle.encode(sfs, keys=True), keys=True)
    assert 'mutation_configs' in jsonpickle.encode(sfs, keys=True)
    assert restored.get_mutation_config(next(iter(expected))) == pytest.approx(expected[next(iter(expected))])


def test_msprime_sfs_without_simulated_mutations_has_no_mutation_configs():
    """``MsprimeCoalescent`` with ``simulate_mutations=False`` filled its spectra with all-zero mutation counts, so
    ``mutation_configs`` reported mass one on the configuration without mutations. Its unfolded and folded spectra
    carry no mutation counts, so the configuration accessors raise, while touching and dropping still works."""
    ms = MsprimeCoalescent(n=4, num_replicates=200, n_threads=1, parallelize=False, seed=1)

    for sfs in (ms.sfs, ms.fsfs):
        with pytest.raises(ValueError, match="no mutation counts"):
            _ = sfs.mutation_configs
        with pytest.raises(ValueError, match="no mutation counts"):
            sfs.get_mutation_config([0, 0, 0])

    ms._touch()
    ms._drop()
    with pytest.raises(ValueError, match="no mutation counts"):
        _ = ms.sfs.mutation_configs


def test_empirical_tree_height_total_is_the_maximum_of_the_per_locus_heights():
    """The empirical total aggregated over loci before summing over demes, so the tree height was the sum over demes
    of the deepest locus in each deme. The tree height of several loci is the deepest per-locus height, and a
    locus's height is the sum over demes of the time its lineages spend there. With one locus spending its whole
    height in each deme, the old total doubled the height.

    The test went through a hand-built ``EmpiricalPhaseTypeDistribution`` whose ``locus_agg`` it passed itself, so it
    pinned only the order of the two aggregations and never the choice made by
    ``TreeHeightDistribution._empirical_locus_agg``, which is what makes ``tree_height.to_empirical`` a tree height
    rather than a total. Replacing that override with the base-class sum over loci left the whole file green while
    the empirical two-locus tree height for ``n = 3`` rose from 1.685 to 2.658 against an exact 1.685."""
    from phasegen.distributions.empirical import EmpiricalPhaseTypeDistribution

    # the public path: the two-locus tree height of the library's own sampler against the exact mean
    th = pg.Coalescent(n=3, loci=2, recombination_rate=1.0).tree_height
    e = th.to_empirical(N_SAMPLES, seed=SEED)
    se = e.samples.std(ddof=1) / np.sqrt(N_SAMPLES)

    assert abs(e.mean - th.mean) < 4 * se
    for locus in (0, 1):
        se_locus = e.loci[locus].samples.std(ddof=1) / np.sqrt(N_SAMPLES)
        assert abs(e.loci[locus].mean - th.loci[locus].mean) < 4 * se_locus

    # the sum over loci, which the base class would have taken, is far outside that band
    assert e.loci[0].mean + e.loci[1].mean > e.mean + 100 * se

    # shape (loci, demes, replicates), asymmetric in both axes so that exchanging them changes every aggregate:
    # per-locus totals 3 + 1 = 4 and 0 + 4 = 4, per-deme totals 3 + 0 = 3 and 1 + 4 = 5
    samples = np.array([[[3.0], [1.0]], [[0.0], [4.0]]])

    tree_height = EmpiricalPhaseTypeDistribution(samples, pops=['a', 'b'], locus_agg=lambda x: x.max(axis=0))
    total = EmpiricalPhaseTypeDistribution(samples, pops=['a', 'b'])

    # the maximum over loci of the per-locus totals, not the sum over demes of the per-deme maxima (7), not the
    # maximum over demes of the per-deme totals (5), and not the sum over loci (8)
    assert tree_height.mean == 4.0
    assert total.mean == 8.0
    assert [tree_height.loci[locus].mean for locus in (0, 1)] == [4.0, 4.0]
    assert [tree_height.demes[pop].mean for pop in ('a', 'b')] == [3.0, 5.0]


def test_msprime_statistics_come_from_one_simulation_without_caching():
    """``MsprimeCoalescent.simulate`` was memoized with ``phasegen.caching.cache``, which stores nothing under
    ``Settings.cache = False``, so the memo could not act as the run-once latch it was being used as. Every statistic
    re-ran the ancestry simulation and overwrote the arrays in place, and with the default ``seed=None`` two
    statistics then described two different tree sets: for two lineages, ``total_branch_length.mean`` is twice
    ``tree_height.mean`` by construction, yet the two disagreed, and repeated access to one statistic returned
    different numbers."""
    pg.Settings.cache = False
    try:
        ms = MsprimeCoalescent(n=2, num_replicates=200, n_threads=1, parallelize=False)  # seed=None

        assert ms.total_branch_length.mean == 2 * ms.tree_height.mean
        assert ms.tree_height.mean == ms.tree_height.mean

        # _touch must persist the statistics it touched, so that _drop reaches the same objects
        ms._touch()
        touched = ms.tree_height
        ms._drop()

        assert ms.tree_height is touched
        assert ms.heights is None
    finally:
        pg.Settings.cache = True
