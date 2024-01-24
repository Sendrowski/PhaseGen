from unittest import TestCase

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent


class CoalescentTestCase(TestCase):
    """
    Test coalescents.
    """

    def test_simple_coalescent(self):
        """
        Test simple coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(n=2),
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

    def test_t_max_standard_coalescent(self):
        """
        Test time until almost sure absorption for standard coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.StandardCoalescent()
        )

        t = coal.tree_height._t_max

        self.assertEqual(t, 32)

    def test_t_max_complex_coalescent(self):
        """
        Test time until almost sure absorption for complex coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        )

        t = coal.tree_height._t_max

        self.assertEqual(t, 129)

    def test_t_max_exponential_growth(self):
        """
        Test time until almost sure absorption for exponential growth.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.StandardCoalescent(),
            demography=pg.Demography([pg.ExponentialPopSizeChanges(
                initial_size={'pop_0': 1},
                growth_rate={'pop_0': 1},
                start_time={'pop_0': 0}
            )])
        )

        t = coal.tree_height._t_max

        self.assertEqual(t, 3.7)

    def test_complex_coalescent(self):
        """
        Test complex coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
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
            pg.Coalescent(
                n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
                model=pg.BetaCoalescent(alpha=1.7),
                demography=self.get_complex_demography(),
                parallelize=False
            ),
            MsprimeCoalescent(
                n_threads=1,
                num_replicates=1000,
                n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
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
                decimal=8
            )

            # make sure the deme-wise total branch lengths add upp
            np.testing.assert_array_almost_equal(
                coal.total_branch_length.mean,
                np.sum([coal.total_branch_length.demes[p].mean for p in coal.demography.pop_names], axis=0),
                decimal=8
            )

            # make sure the deme-wise SFS add upp
            np.testing.assert_array_almost_equal(
                coal.sfs.mean.data,
                np.sum([coal.sfs.demes[p].mean.data for p in coal.demography.pop_names], axis=0),
                decimal=8
            )

    @pytest.mark.skip(reason="Too slow")
    def test_msprime_complex_coalescent(self):
        """
        Test msprime complex coalescent.
        """
        coal = MsprimeCoalescent(
            n_threads=100,
            parallelize=True,
            num_replicates=1000000,
            n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            # model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography(),
            record_migration=True
        )

        coal.touch()

        pass

    def test_msprime_coalescent_two_loci(self):
        """
        Test msprime coalescent.
        """
        coal = MsprimeCoalescent(
            n_threads=1,
            parallelize=False,
            num_replicates=1000,
            n=pg.LineageConfig(2),
            loci=2,
            recombination_rate=10,
            model=pg.StandardCoalescent(),
            demography=pg.Demography([pg.PopSizeChange(pop='pop_0', time=0, size=1)])
        )

        m = coal.tree_height.mean

        pass

    def test_two_loci_one_deme_n_2_tree_height(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        )

        self.assertAlmostEqual(1, coal.tree_height.loci[0].mean)
        self.assertAlmostEqual(1, coal.tree_height.loci[0].var)

        self.assertAlmostEqual(2, coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),)))

        pass

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_two_loci_one_deme_n_2_sfs(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            loci=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        _ = coal.tree_height.mean
        coal.sfs.mean.plot()

        pass

    def test_two_loci_one_deme_n_4(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(4),
            loci=pg.LocusConfig(n=2, recombination_rate=1),
        )

        marginal = pg.Coalescent(n=pg.LineageConfig(4))

        # assert total branch length to be twice as long as marginal
        self.assertAlmostEqual(marginal.total_branch_length.mean * 2, coal.total_branch_length.mean)

        self.assertAlmostEqual(marginal.total_branch_length.mean, coal.total_branch_length.loci[0].mean)
        self.assertAlmostEqual(marginal.total_branch_length.var, coal.total_branch_length.loci[0].var)

        # assert total tree height to be twice as long as marginal
        self.assertAlmostEqual(marginal.tree_height.mean * 2, coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),)))

        # assert marginal locus moments
        self.assertAlmostEqual(marginal.tree_height.mean, coal.tree_height.loci[0].mean)
        self.assertAlmostEqual(marginal.tree_height.var, coal.tree_height.loci[0].var)

        pass

    def test_two_loci_two_demes(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig([2, 2]),
            loci=pg.LocusConfig(n=2, recombination_rate=1.11),
        )

        pass

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_two_loci_one_deme_n_2(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig([2]),
            loci=pg.LocusConfig(n=2, recombination_rate=1.11),
        )

        coal.sfs.mean.plot()

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_two_loci_one_deme_n_linked(self):
        """
        Test SFS for two loci with different numbers of linked lineages.
        """
        m = []
        n = 5

        for n_unlinked in range(n + 1):
            coal = pg.Coalescent(
                n=pg.LineageConfig(n),
                loci=pg.LocusConfig(
                    n=2,
                    recombination_rate=0,
                    n_unlinked=n_unlinked,
                    allow_coalescence=False
                )
            )

            m += [coal.sfs.mean.data]

        m = np.array(m)

        pass

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_two_loci_one_deme_linked_coalescence(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(4),
            loci=pg.LocusConfig(
                n=2,
                recombination_rate=0,
                n_unlinked=0,
                allow_coalescence=False
            )
        )

        # sfs = coal.sfs.mean.data

        rates1, states1 = coal.block_counting_state_space._get_outgoing_rates(19)
        rates2, states2 = coal.block_counting_state_space._get_outgoing_rates(states1[0])

        pass

    def test_beta_4_n(self):
        """
        Test beta coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(4),
            model=pg.BetaCoalescent(alpha=1.7)
        )

        m = coal.tree_height.mean

        pass

    def test_2_loci_sfs_raises_not_implemented_error(self):
        """
        Test two loci SFS raises NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            coal = pg.Coalescent(
                n=pg.LineageConfig(4),
                loci=pg.LocusConfig(2)
            )

            _ = coal.sfs
