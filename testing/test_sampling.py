"""
Tests for the public ``sample`` / ``to_empirical`` methods of the distributions and the
:class:`~phasegen.distributions.SampledCoalescent`. These check that PhaseGen's own trajectory sampler reproduces
the exact analytic statistics (up to Monte-Carlo error), the self-consistency that the sampled-scenario suite
(``*_sampled`` configs in ``test_scenarios.py``) exercises end-to-end.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import SampledCoalescent

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
