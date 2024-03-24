"""
Test coalescents.
"""

from unittest import TestCase

import numpy as np
import pytest

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent


class CoalescentTestCase(TestCase):
    """
    Test coalescents.
    """

    def get_simple_coalescent(self):
        """
        Get simple coalescent.
        """
        return pg.Coalescent(
            n=pg.LineageConfig(n=2),
            model=pg.StandardCoalescent(),
            demography=pg.Demography([pg.PopSizeChange(pop='pop_0', time=0, size=1)])
        )

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

    def get_complex_coalescent(self):
        """
        Get complex coalescent.
        """
        return pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        )

    def test_simple_coalescent(self):
        """
        Test simple coalescent.
        """
        coal = self.get_simple_coalescent()

        m = coal.tree_height.mean
        coal.tree_height.plot_pdf()

        self.assertAlmostEqual(m, 1)

    def test_t_max_standard_coalescent(self):
        """
        Test time until almost sure absorption for standard coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.StandardCoalescent()
        )

        t = coal.tree_height.t_max

        self.assertEqual(t, 63)

    def test_t_max_complex_coalescent(self):
        """
        Test time until almost sure absorption for complex coalescent.
        """
        coal = self.get_complex_coalescent()

        t = coal.tree_height.t_max

        self.assertTrue(150 < t < 200)

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

        t = coal.tree_height.t_max

        self.assertTrue(3 < t < 5)

    def test_complex_coalescent(self):
        """
        Test complex coalescent.
        """
        coal = self.get_complex_coalescent()

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

    @pytest.mark.skip(reason="not needed")
    def test_n_2_2_loci_default_state_space_unlinked(self):
        """
        Test n=2, 2 loci, default state space.
        """
        means = []
        m2 = []

        for n_unlinked in range(3):
            coal = pg.Coalescent(
                demography=pg.Demography(
                    pop_sizes=dict(
                        pop_0={0: 1},
                        pop_1={0: 1}
                    ),
                    migration_rates={
                        ('pop_0', 'pop_1'): {0: 1},
                        ('pop_1', 'pop_0'): {0: 1},
                    }
                ),
                n=pg.LineageConfig(dict(
                    pop_0=1,
                    pop_1=1
                )),
                loci=pg.LocusConfig(n=2, recombination_rate=0, n_unlinked=n_unlinked, allow_coalescence=False)
            )

            means += [coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),))]
            m2 += [coal.tree_height.moment(2, (pg.TotalTreeHeightReward(),) * 2)]

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_2_2_loci_default_state_space_completely_unlinked(self):
        """
        Test n=2, 2 loci, default state space.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 1},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=1,
                pop_1=1
            )),
            loci=pg.LocusConfig(n=2, recombination_rate=0, n_unlinked=2, allow_coalescence=False)
        )

        m = coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),))
        m2 = coal.tree_height.moment(2, (pg.TotalTreeHeightReward(),) * 2)

        coal.default_state_space._plot_rates('tmp/test_n_2_2_loci_default_state_space_completely_unlinked')

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_2_1_locus_default_state_space(self):
        """
        Test n=2, 1 locus, default state space.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 1},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=1,
                pop_1=1
            ))
        )

        m = coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),))

        coal.default_state_space._plot_rates('tmp/test_n_2_1_locus_default_state_space')

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_3_2_loci_default_state_space_unlinked(self):
        """
        Test n=3, 2 loci, default state space.
        """
        means = []
        m2 = []

        for n_unlinked in range(4):
            coal = pg.Coalescent(
                demography=pg.Demography(
                    pop_sizes=dict(
                        pop_0={0: 1},
                        pop_1={0: 1}
                    ),
                    migration_rates={
                        ('pop_0', 'pop_1'): {0: 1},
                        ('pop_1', 'pop_0'): {0: 1}
                    }
                ),
                n=pg.LineageConfig(dict(
                    pop_0=2,
                    pop_1=1
                )),
                loci=pg.LocusConfig(n=2, recombination_rate=0, n_unlinked=n_unlinked, allow_coalescence=False)
            )

            means += [coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),))]
            m2 += [coal.tree_height.moment(2, (pg.TotalTreeHeightReward(),) * 2)]

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_3_2_loci_default_state_space_completely_linked(self):
        """
        Test n=3, 2 loci, default state space.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 1},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1}
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=2,
                pop_1=1
            )),
            loci=pg.LocusConfig(n=2, recombination_rate=0, n_unlinked=0, allow_coalescence=False)
        )

        m = coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),))
        m2 = coal.tree_height.moment(2, (pg.TotalTreeHeightReward(),) * 2)

        coal.default_state_space._plot_rates('tmp/test_n_3_2_loci_default_state_space_completely_linked')

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_3_1_locus_default_state_space(self):
        """
        Test n=3, 1 locus, default state space.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 1},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=2,
                pop_1=1
            ))
        )

        m = coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),))

        coal.default_state_space._plot_rates('tmp/test_n_3_1_locus_default_state_space')

        pass

    def test_beta_coalescent_n_2_alpha_close_to_2_default_state_space(self):
        """
        Test beta coalescent with default state space for n = 2.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.BetaCoalescent(alpha=1.999)
        )

        # coalescent time coincides with timescale in this case
        self.assertAlmostEqual(coal.tree_height.mean, coal.model._get_timescale(1), places=15)

        pass

    def test_serialize_coalescent(self):
        """
        Test serialization of coalescent.
        """
        coal = self.get_complex_coalescent()

        coal.to_file('tmp/test_serialize_simple_coalescent.json')

        coal2 = pg.Coalescent.from_file('tmp/test_serialize_simple_coalescent.json')

        self.assertEqual(coal.tree_height.mean, coal2.tree_height.mean)

    def test_coalescent_negative_end_time_raises_value_error(self):
        """
        Test negative end time raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                end_time=-1
            ).tree_height

        self.assertTrue('End time' in str(context.exception))

    def test_coalescent_negative_start_time_raises_value_error(self):
        """
        Test negative start time raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                start_time=-1
            ).tree_height

        self.assertTrue('Start time' in str(context.exception))

    def test_end_time_before_start_time_raises_value_error(self):
        """
        Test end time before start time raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                start_time=1,
                end_time=0
            ).tree_height

        self.assertTrue('End time' in str(context.exception))

    def test_start_greater_than_t_abs_raises_value_error(self):
        """
        Test start time greater than t_abs raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                start_time=100000
            ).tree_height.mean

        self.assertTrue('start time' in str(context.exception))

    def test_start_time_equal_end_time_zero_moments(self):
        """
        Test start time equal to end time gives zero moments.
        """
        coal = pg.Coalescent(
            n=2,
            start_time=1,
            end_time=1
        )

        self.assertEqual(coal.tree_height.mean, 0)
        self.assertEqual(coal.tree_height.var, 0)

    def test_simple_coalescent_start_time(self):
        """
        Test simple coalescent start time moment.
        """
        coal = pg.Coalescent(
            n=2,
            start_time=1
        )

        _ = coal.tree_height.mean
        _ = coal.tree_height.var
        _ = coal.total_branch_length.mean
        _ = coal.total_branch_length.var
        _ = coal.sfs.mean
        _ = coal.sfs.corr
        _ = coal.tree_height.pdf(1)
        _ = coal.tree_height.cdf(1)

    def test_plot_accumulation(self):
        """
        Test moments.
        """
        coal = self.get_complex_coalescent()

        values = np.linspace(0, coal.tree_height.quantile(0.99), 10)

        coal.tree_height.plot_accumulation(1, values)
        coal.tree_height.plot_accumulation(2, values)
        coal.sfs.plot_accumulation(1, values)
        coal.sfs.plot_accumulation(2, values)

    def test_large_accumulation_equal_moments(self):
        """
        Test large accumulation equals moments.
        """
        coal = self.get_complex_coalescent()

        self.assertEqual(
            coal.tree_height.accumulate(1, coal.tree_height.t_max),
            coal.tree_height.moment(1)

        )

        self.assertEqual(
            coal.tree_height.accumulate(2, coal.tree_height.t_max),
            coal.tree_height.moment(2)
        )

        np.testing.assert_array_equal(
            coal.sfs.accumulate(1, coal.tree_height.t_max),
            coal.sfs.moment(1).data
        )

    def test_precision_regularization_large_N(self):
        """
        Make sure regularization works for large rates.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1e40}})
        )

        lamb = coal.tree_height._get_regularization_factor(coal.default_state_space.S)

        self.assertTrue(1e39 <= lamb <= 1e41)

        self.assertAlmostEqual(coal.tree_height.mean, 1.5e40, delta=1e26)

    def test_precision_regularization_small_N(self):
        """
        Make sure regularization works for small rates.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1e-40}})
        )

        lamb = coal.tree_height._get_regularization_factor(coal.default_state_space.S)

        self.assertTrue(1e-41 <= lamb <= 1e-39)

        self.assertAlmostEqual(coal.tree_height.mean, 1.5e-40, delta=1e-26)

    def test_warning_disconnected_demes(self):
        """
        Make sure disconnected demes raise warning.
        """
        with self.assertLogs(level='WARNING', logger=pg.logger) as cm:
            pg.Demography(
                pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 1}}
            )

        self.assertTrue('zero migration rates' in cm.output[0])

    def test_value_error_extreme_imprecision(self):
        """
        Make sure extreme imprecision raises ValueError.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1e-40}, 'pop_1': {0: 1e40}},
                migration_rates={('pop_0', 'pop_1'): {0: 1}}
            )
        )

        _ = coal.tree_height.mean

        self.assertNoLogs(level='CRITICAL', logger=coal._logger)

    def test_value_error_large_imprecision(self):
        """
        Make sure no warning is raised for large imprecision.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1e10}, 'pop_1': {0: 1e4}},
                migration_rates={('pop_0', 'pop_1'): {0: 1e4}}
            ),
        )

        coal.tree_height.plot_cdf()
        coal.tree_height.plot_accumulation(1)

        self.assertNoLogs(level='WARNING', logger=coal._logger)

    def test_low_recombination_rate(self):
        """
        Test low recombination rate.
        """
        coal = pg.Coalescent(
            n=4,
            loci=pg.LocusConfig(n=2, recombination_rate=1e11)
        )

        with self.assertLogs(level='WARNING', logger=coal.tree_height._logger) as cm:
            _ = coal.tree_height.mean

        self.assertIn("numerical instability", cm.output[0])
