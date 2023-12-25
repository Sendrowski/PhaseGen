from unittest import TestCase

import numpy as np

import phasegen as pg
from phasegen.distributions import _MsprimeCoalescent


class CoalescentTestCase(TestCase):
    """
    Test coalescents.
    """

    def test_simple_coalescent(self):
        """
        Test simple coalescent.
        """
        coal = pg.Coalescent(
            n=pg.PopConfig(n=2),
            model=pg.StandardCoalescent(),
            demography=pg.Demography([pg.PopSizeChange(pop='pop_0', time=0, size=1)])
        )

        m = coal.tree_height.mean
        coal.tree_height.plot_pdf()

        self.assertAlmostEqual(m, 1)

    def get_complex_demography(self):
        """
        Get complex demography.
        """
        return pg.Demography([
            pg.PopSizeChanges({'pop_0': {0: 1, 0.2: 1.2, 0.4: 1.4}, 'pop_1': {0: 1, 0.2: 1.2, 0.4: 1.4}}),
            pg.MigrationRateChanges({
                ('pop_0', 'pop_1'): {0: 0, 0.2: 0.2, 0.4: 0.4},
                ('pop_1', 'pop_0'): {0: 0, 0.3: 0.3, 0.6: 0.6},
                ('pop_1', 'pop_2'): {0: 0, 0.4: 0.4, 0.8: 0.8},
                ('pop_2', 'pop_1'): {0: 0, 0.5: 0.5, 1: 1}
            })
        ])

    def test_complex_coalescent(self):
        """
        Test complex coalescent.
        """
        coal = pg.Coalescent(
            n=pg.PopConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        )

        m = coal.tree_height.mean
        coal.tree_height.plot_pdf()

        self.assertAlmostEqual(m, 5.91979, delta=5)

    def test_demes_complex_coalescent(self):
        """
        Test deme-wise complex coalescents.
        """
        coals = [
            # pg.Coalescent(
            #    n=pg.PopConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            #    model=pg.BetaCoalescent(alpha=1.7),
            #    demography=self.get_complex_demography(),
            #    parallelize=False
            # ),
            _MsprimeCoalescent(
                n_threads=1,
                num_replicates=1000,
                n=pg.PopConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
                model=pg.BetaCoalescent(alpha=1.7),
                demography=self.get_complex_demography(),
                record_migration=True
            )
        ]

        for coal in coals:
            # make sure the deme-wise tree heights add upp
            np.testing.assert_array_almost_equal(
                coal.tree_height.mean,
                np.sum([coal.tree_height.demes[p].mean for p in coal.demography.pop_names], axis=0),
                decimal=14
            )

            # make sure the deme-wise total branch lengths add upp
            np.testing.assert_array_almost_equal(
                coal.total_branch_length.mean,
                np.sum([coal.total_branch_length.demes[p].mean for p in coal.demography.pop_names], axis=0),
                decimal=14
            )

            # make sure the deme-wise SFS add upp
            np.testing.assert_array_almost_equal(
                coal.sfs.mean.data,
                np.sum([coal.sfs.demes[p].mean.data for p in coal.demography.pop_names], axis=0),
                decimal=14
            )
