"""
Targeted fast tests filling small coverage gaps in utility helpers, demographic events and the folded SFS.
"""
import numpy as np

import phasegen as pg
from phasegen.utils import take_n, takewhile_inclusive
from testing import TestCase


class CoverageGapsTestCase(TestCase):
    """
    Small, fast tests that exercise otherwise-uncovered helper paths.
    """

    def test_take_n_and_takewhile_inclusive(self):
        """The ``take_n`` and ``takewhile_inclusive`` iterator helpers."""
        self.assertEqual(list(take_n(range(10), 3)), [0, 1, 2])
        # takewhile_inclusive keeps the first element that fails the predicate
        self.assertEqual(list(takewhile_inclusive(lambda x: x < 3, [1, 2, 3, 4])), [1, 2, 3])

    def test_population_split_demography(self):
        """A population split builds valid epochs and plots, exercising the split event and migration plotting."""
        d = pg.Demography(
            pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
            events=[pg.PopulationSplit(time=1.0, derived='pop_0', ancestral='pop_1')]
        )

        # building the epochs applies the split event
        epochs = list(d.get_epochs(np.array([0.0, 1.5])))
        self.assertEqual(len(epochs), 2)

        d.plot_migration(show=False)
        d.plot_pop_sizes(show=False)

    def test_folded_sfs_mean(self):
        """The folded SFS distribution produces a non-trivial mean."""
        folded = pg.Coalescent(n=4).fsfs.mean
        self.assertGreater(np.asarray(folded.data).sum(), 0)

    def test_single_locus_and_population_guards(self):
        """The single-locus SFS and multi-population statistics raise clear errors for invalid configurations."""
        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=2, loci=2, recombination_rate=1.0).sfs

        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=2, loci=2, recombination_rate=1.0).fsfs

        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=3).sfs2

        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=3).fst

    def test_tree_height_density_cdf_quantile(self):
        """Evaluate the tree-height CDF, density and quantile, exercising the numerical paths."""
        coal = pg.Coalescent(n=4)
        t = np.linspace(0.1, 3, 5)

        self.assertTrue(np.all(np.isfinite(coal.tree_height.cdf(t))))
        self.assertTrue(np.all(np.isfinite(coal.tree_height.pdf(t))))
        self.assertTrue(np.isfinite(coal.tree_height.quantile(0.5)))
