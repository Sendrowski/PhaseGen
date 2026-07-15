"""
Fast msprime ground-truth tests.

These exercise the msprime-backed :class:`~phasegen.distributions.MsprimeCoalescent` paths (used as simulation
ground truth in the scenario comparisons) with tiny samples and few replicates, so the code is covered without the
cost of the slow comparison suite. They assert only that the statistics are produced and finite, not their accuracy
(the slow scenario tests validate accuracy against the analytical results).
"""
import math

import jsonpickle
import numpy as np

import phasegen as pg
from phasegen.distributions.empirical import EmpiricalDistribution
from testing import TestCase


class MsprimeGroundTruthTestCase(TestCase):
    """
    Drive the msprime ground-truth statistics on small samples.
    """

    @staticmethod
    def _ms(coal: pg.Coalescent, **kwargs) -> 'pg.distributions.MsprimeCoalescent':
        """Small, serial, seeded msprime simulation of the given coalescent."""
        return coal.to_msprime(num_replicates=50, parallelize=False, n_threads=1, seed=42, **kwargs)

    def test_single_population_statistics(self):
        """Tree height, branch length and (folded) SFS ground truth for a single population."""
        ms = self._ms(pg.Coalescent(n=4))

        assert np.isfinite(ms.tree_height.mean)
        assert np.isfinite(ms.total_tree_height.mean)
        assert np.isfinite(ms.total_branch_length.mean)
        assert np.asarray(ms.sfs.mean).shape == (5,)
        assert np.asarray(ms.fsfs.mean) is not None

        # round-trip back to the analytical coalescent
        assert isinstance(ms.to_phasegen(), pg.Coalescent)

    def test_multi_population_statistics(self):
        """Joint SFS, F_ST and Patterson f-statistics ground truth across four populations."""
        pops = [f'pop_{i}' for i in range(4)]
        demography = pg.Demography(
            pop_sizes={p: 1.0 for p in pops},
            migration_rates={(a, b): 1.0 for a in pops for b in pops if a != b}
        )
        coal = pg.Coalescent(n={p: 2 for p in pops}, demography=demography)
        ms = self._ms(coal, record_migration=True)

        assert np.asarray(ms.jsfs.mean).ndim == 4
        assert np.isfinite(ms.fst)
        assert np.isfinite(ms.f2('pop_0', 'pop_1'))
        assert np.isfinite(ms.f3('pop_0', 'pop_1', 'pop_2'))
        assert np.isfinite(ms.f4('pop_0', 'pop_1', 'pop_2', 'pop_3'))

    def test_two_locus_statistics(self):
        """Two-locus SFS ground truth under recombination."""
        ms = self._ms(pg.Coalescent(n=2, loci=2, recombination_rate=1.0))

        assert np.asarray(ms.sfs2.mean.data).ndim == 2

    def test_joint_sfs_n_samples_is_the_replicate_count_not_the_cap(self):
        """``EmpiricalJointSFSDistribution.n_samples`` must record the replicate count the moments were averaged over,
        not the (capped) number of retained per-replicate samples -- otherwise the tuner's jSFS noise floor is
        inflated by ``sqrt(num_replicates / cap)``. It defaults to ``len(samples)`` only when not passed."""
        from phasegen.distributions.empirical import EmpiricalJointSFSDistribution

        moments = np.ones((3, 2, 2))
        capped = np.ones((10, 2, 2))  # a capped subset of a larger simulation

        d = EmpiricalJointSFSDistribution(moments=moments, samples=capped, n_samples=1000)
        self.assertEqual(d.n_samples, 1000)
        d.drop()
        self.assertIsNone(d.samples)
        self.assertEqual(d.n_samples, 1000)  # survives the drop, so a serialized comparison keeps the true count

        # backwards-compatible default: the sample length when no explicit count is given
        self.assertEqual(EmpiricalJointSFSDistribution(moments=moments, samples=capped).n_samples, 10)

    def test_sfs_empirical_does_not_compute_unused_deme_matrices(self):
        """The SFS empirical overrides ``demes`` to a plain per-deme dict, so its deme-deme cov/corr are never read;
        they must not be computed (an expensive ``2*(n-1)**2`` pass per construction) but left ``None``."""
        ms = self._ms(pg.Coalescent(n=4))

        self.assertIsNone(ms.sfs.pops_cov)
        self.assertIsNone(ms.sfs.pops_corr)


class MsprimeSurfaceCachingTestCase(TestCase):
    """
    Drive the empirical pairwise-surface caching (the *only* joint ground-truth path serialized into the comparison
    fixtures after the point-based pairwise comparison was retired). These exercise ``cache_joint_surface`` on each
    empirical spectrum and ``cache_loci_joint_surface`` for the cross-locus joint, on tiny seeded simulations, and
    assert the grid invariants the surface comparison relies on (shape, finiteness, a valid -- bounded, monotone --
    empirical CDF). Accuracy is validated by the slow scenario suite.
    """

    @staticmethod
    def _ms(coal: pg.Coalescent, **kwargs) -> 'pg.distributions.MsprimeCoalescent':
        """Small, serial, seeded msprime simulation of the given coalescent (enough replicates for a stable grid)."""
        return coal.to_msprime(num_replicates=500, parallelize=False, n_threads=1, seed=42, **kwargs)

    def _assert_valid_surface(self, xs, ys, cdf, pdf, n_grid: int = 25):
        """A cached surface is a square ``n_grid`` grid: a finite, bounded-in-[0,1], monotone-non-decreasing empirical
        joint CDF and a finite density."""
        xs, ys, cdf, pdf = (np.asarray(a, dtype=float) for a in (xs, ys, cdf, pdf))
        assert xs.shape == (n_grid,) and ys.shape == (n_grid,)
        assert cdf.shape == (n_grid, n_grid) and pdf.shape == (n_grid, n_grid)
        assert np.isfinite(cdf).all() and np.isfinite(pdf).all()
        assert (cdf >= -1e-9).all() and (cdf <= 1 + 1e-9).all()
        # the empirical joint CDF P(L_i <= x, L_j <= y) is non-decreasing along each axis
        assert (np.diff(cdf, axis=0) >= -1e-9).all()
        assert (np.diff(cdf, axis=1) >= -1e-9).all()

    def test_sfs_surface(self):
        """Within-tree SFS pairwise surface for a single population."""
        ms = self._ms(pg.Coalescent(n=4))
        ms.sfs.cache_joint_surface([(1, 3)])

        assert len(ms.sfs._joint_surface) == 1
        i, j, xs, ys, cdf, pdf = ms.sfs._joint_surface[0]
        assert (i, j) == (1, 3)
        self._assert_valid_surface(xs, ys, cdf, pdf)

    def test_jsfs_surface(self):
        """Within-tree joint (multi-population) SFS surface, indexed by descendant configuration."""
        demography = pg.Demography(
            pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
            migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0}
        )
        ms = self._ms(pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=demography))
        ms.jsfs.cache_joint_surface([((0, 1), (1, 0))])

        ca, cb, xs, ys, cdf, pdf = ms.jsfs._joint_surface[0]
        assert (ca, cb) == ((0, 1), (1, 0))
        self._assert_valid_surface(xs, ys, cdf, pdf)

    def test_two_locus_surface(self):
        """Cross-locus two-locus SFS surface under recombination."""
        ms = self._ms(pg.Coalescent(n=3, loci=2, recombination_rate=1.0))
        ms.sfs2.cache_joint_surface([(1, 2)])

        i, j, xs, ys, cdf, pdf = ms.sfs2._joint_surface[0]
        assert (i, j) == (1, 2)
        self._assert_valid_surface(xs, ys, cdf, pdf)

    def test_loci_surface(self):
        """Cross-locus joint surface of the per-locus total branch length at the two loci."""
        ms = self._ms(pg.Coalescent(n=2, loci=2, recombination_rate=1.0))
        ms.total_branch_length.cache_loci_joint_surface([(0, 1)])

        l1, l2, xs, ys, cdf, pdf = ms.total_branch_length._loci_joint_surface[0]
        assert (l1, l2) == (0, 1)
        self._assert_valid_surface(xs, ys, cdf, pdf)

    def test_standard_errors_match_the_closed_forms(self):
        """The block estimator reproduces the exact standard errors of an exponential sample, for which the moments of
        every order are known. The closed forms are only available here (they need the ``2k``-th moment for the ``k``-th
        raw moment, and the fourth central moment for the variance), which is why the estimator exists."""
        n, scale = 200_000, 2.0
        dist = EmpiricalDistribution(np.random.default_rng(0).exponential(scale, n))
        dist.cache_standard_errors()

        # for Exp(scale): E[X^k] = k! scale^k, so Var[m_k_hat] = (E[X^2k] - E[X^k]^2) / n
        moment = lambda k: float(math.factorial(k)) * scale ** k

        for k, name in [(1, 'mean'), (2, 'm2'), (3, 'm3'), (4, 'm4')]:
            exact = np.sqrt((moment(2 * k) - moment(k) ** 2) / n)
            self.assertAlmostEqual(float(dist.standard_errors[name]) / exact, 1, delta=0.25)

        # Var[var_hat] = (mu4 - mu2^2) / n, with the central moments mu2 = scale^2 and mu4 = 9 scale^4
        exact_var = np.sqrt((9 * scale ** 4 - scale ** 4) / n)
        self.assertAlmostEqual(float(dist.standard_errors['var']) / exact_var, 1, delta=0.25)

    def test_standard_errors_survive_the_drop(self):
        """The standard errors are cached by ``touch`` and outlive the samples, so a serialized comparison can still
        tell how much of a discrepancy against its ground truth is that ground truth's own Monte-Carlo noise."""
        ms = self._ms(pg.Coalescent(n=4))
        ms.touch()
        ms.drop()

        for dist in (ms.tree_height, ms.total_branch_length, ms.sfs):
            self.assertIsNone(dist.samples)
            for name in ('mean', 'var', 'm3', 'm4'):
                self.assertTrue(np.all(np.isfinite(dist.standard_errors[name])))
                self.assertTrue(np.all(np.asarray(dist.standard_errors[name]) >= 0))

        # the per-bin standard errors of a spectrum line up with its bins
        self.assertEqual(np.shape(ms.sfs.standard_errors['mean']), np.shape(np.asarray(ms.sfs.mean)))

    def test_standard_errors_cover_deme_and_locus_matrices(self):
        """The block estimator covers the deme-deme and locus-locus covariance / correlation matrices that
        ``demes.cov`` / ``loci.corr`` expose (keyed ``"demes.cov"`` etc.), so the tuner has a real noise floor for
        those leaves. The bogus scalar ``cov`` / ``corr`` of the 1-D total is not cached (corrcoef of a 1-D sample is
        the constant 1). Single-deme / single-locus cases have no matrix and cache none."""
        from phasegen.distributions.empirical import EmpiricalPhaseTypeDistribution

        rng = np.random.default_rng(3)
        base = rng.exponential(1.0, 40_000)
        # 2 loci x 3 demes, correlated through a shared component
        samples = np.array([[base + rng.normal(0, 0.2, base.size) for _ in range(3)],
                            [0.7 * base + rng.normal(0, 0.2, base.size) for _ in range(3)]])
        dist = EmpiricalPhaseTypeDistribution(pops=['p0', 'p1', 'p2'], samples=samples)
        dist.cache_standard_errors()

        self.assertEqual(dist.standard_errors['demes.cov'].shape, (3, 3))
        self.assertEqual(dist.standard_errors['loci.cov'].shape, (2, 2))
        self.assertIn('demes.corr', dist.standard_errors)
        self.assertIn('loci.corr', dist.standard_errors)
        self.assertTrue(np.all(dist.standard_errors['demes.cov'] >= 0))
        # the 1-D total's scalar cov/corr are skipped
        self.assertNotIn('cov', dist.standard_errors)
        self.assertNotIn('corr', dist.standard_errors)

        # a single deme has no deme-deme matrix; a single locus no locus-locus matrix
        single = EmpiricalPhaseTypeDistribution(pops=['p0'], samples=samples[:, :1, :])
        single.cache_standard_errors()
        self.assertNotIn('demes.cov', single.standard_errors)

    def test_surface_survives_serialization(self):
        """The cached surface (numpy grids) round-trips through the jsonpickle serialization used for the fixtures
        (numpy handlers are registered on importing phasegen)."""
        ms = self._ms(pg.Coalescent(n=4))
        sfs = ms.sfs
        sfs.cache_joint_surface([(1, 3)])
        sfs.samples = None  # the fixture serializes after the raw per-replicate samples are dropped

        restored = jsonpickle.decode(jsonpickle.encode(sfs, keys=True), keys=True)
        i, j, xs, ys, cdf, pdf = restored._joint_surface[0]
        assert (i, j) == (1, 3)
        self._assert_valid_surface(xs, ys, cdf, pdf)
        assert restored.samples is None  # only the surface ground truth remains, not the bulky samples
