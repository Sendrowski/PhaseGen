"""
Fast msprime ground-truth tests.

These exercise the msprime-backed :class:`~phasegen.distributions.MsprimeCoalescent` paths (used as simulation
ground truth in the scenario comparisons) with tiny samples and few replicates, so the code is covered without the
cost of the slow comparison suite. They assert only that the statistics are produced and finite, not their accuracy
(the slow scenario tests validate accuracy against the analytical results).
"""
import jsonpickle
import numpy as np

import phasegen as pg
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
