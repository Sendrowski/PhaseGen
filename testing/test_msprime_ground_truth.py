"""
Fast msprime ground-truth tests.

These exercise the msprime-backed :class:`~phasegen.distributions.MsprimeCoalescent` paths (used as simulation
ground truth in the scenario comparisons) with tiny samples and few replicates, so the code is covered without the
cost of the slow comparison suite. They assert only that the statistics are produced and finite, not their accuracy
(the slow scenario tests validate accuracy against the analytical results).
"""
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
