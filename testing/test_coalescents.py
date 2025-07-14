"""
Test coalescents.
"""
import unittest
from itertools import islice
from typing import cast
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

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

        self.assertEqual(t, 64)

    def test_t_max_complex_coalescent(self):
        """
        Test time until almost sure absorption for complex coalescent.
        """
        coal = self.get_complex_coalescent()

        t = coal.tree_height.t_max

        self.assertEqual(t, 128)

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
        Validate first moments for deme-wise complex coalescent.
        """
        pg.Settings.parallelize = False

        coals = [
            pg.Coalescent(
                n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
                model=pg.BetaCoalescent(alpha=1.7),
                demography=self.get_complex_demography()
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

        pg.Settings.parallelize = True

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

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
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

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_two_loci_one_deme_n_2(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig([2]),
            loci=pg.LocusConfig(n=2, recombination_rate=1.11),
        )

        coal.sfs.mean.plot()

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
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

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
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
    def test_n_2_2_loci_lineage_counting_state_space_unlinked(self):
        """
        Test n=2, 2 loci, lineage-counting state space.
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
    def test_n_2_2_loci_lineage_counting_state_space_completely_unlinked(self):
        """
        Test n=2, 2 loci, lineage-counting state space.
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

        coal.lineage_counting_state_space.plot_rates(
            'scratch/test_n_2_2_loci_lineage_counting_state_space_completely_unlinked')

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_2_1_locus_lineage_counting_state_space(self):
        """
        Test n=2, 1 locus, lineage-counting state space.
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

        coal.lineage_counting_state_space.plot_rates('scratch/test_n_2_1_locus_lineage_counting_state_space')

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_3_2_loci_lineage_counting_state_space_unlinked(self):
        """
        Test n=3, 2 loci, lineage-counting state space.
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
    def test_n_3_2_loci_lineage_counting_state_space_completely_linked(self):
        """
        Test n=3, 2 loci, lineage-counting state space.
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

        coal.lineage_counting_state_space.plot_rates(
            'scratch/test_n_3_2_loci_lineage_counting_state_space_completely_linked')

        pass

    @pytest.mark.skip(reason="not needed")
    def test_n_3_1_locus_lineage_counting_state_space(self):
        """
        Test n=3, 1 locus, lineage-counting state space.
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

        coal.lineage_counting_state_space.plot_rates('scratch/test_n_3_1_locus_lineage_counting_state_space')

        pass

    def test_beta_coalescent_n_2_alpha_close_to_2_lineage_counting_state_space(self):
        """
        Test beta coalescent with lineage-counting state space for n = 2.
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

        coal.to_file('scratch/test_serialize_simple_coalescent.json')

        coal2 = pg.Coalescent.from_file('scratch/test_serialize_simple_coalescent.json')

        self.assertEqual(coal.tree_height.mean, coal2.tree_height.mean)

    def test_serialize_assert_getstate_method_called(self):
        """
        Test serialization of coalescent.
        """
        coal = self.get_complex_coalescent()

        with patch.object(coal, '__getstate__', return_value=None) as mock_getstate:
            try:
                coal.to_file('scratch/test_serialize_simple_coalescent.json')
            except Exception:
                pass

            mock_getstate.assert_called_once()

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

    def test_plot_accumulation_center_permute(self):
        """
        Test accumulation plot for center and permute.
        """
        coal = pg.Coalescent(
            n=3
        )

        values = np.linspace(0, coal.tree_height.quantile(0.99), 10)
        rewards = (pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2))

        fig, ax = plt.subplots(1)

        for i, kwargs in enumerate([
            dict(center=True, permute=True),
            dict(center=False, permute=True),
            dict(center=True, permute=False),
            dict(center=False, permute=False)
        ]):
            coal.plot_accumulation(
                k=2,
                end_times=values,
                rewards=rewards,
                ax=ax,
                show=False,
                label=str(kwargs),
                **kwargs
            )

        plt.show()

    def test_plot_accumulation(self):
        """
        Test accumulation plot.
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
            coal.tree_height.accumulate(1, [coal.tree_height.t_max])[0],
            coal.tree_height.moment(1)
        )

        self.assertEqual(
            coal.tree_height.accumulate(2, [coal.tree_height.t_max])[0],
            coal.tree_height.moment(2)
        )

        np.testing.assert_array_equal(
            coal.sfs.accumulate(1, [coal.tree_height.t_max])[:, 0],
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

        lamb = coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S)

        self.assertTrue(1e39 <= lamb <= 1e41)

        self.assertAlmostEqual(coal.tree_height.mean, 1.5e40, delta=1e27)

    def test_precision_regularization_small_N(self):
        """
        Make sure regularization works for small rates.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1e-40}})
        )

        lamb = coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S)

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

    def test_symmetric_deme_covariance(self):
        """
        Make sure deme covariance is symmetric.
        """
        coal = self.get_complex_coalescent()

        np.testing.assert_array_equal(coal.tree_height.demes.cov, coal.tree_height.demes.cov.T)
        np.testing.assert_array_equal(coal.tree_height.demes.corr, coal.tree_height.demes.corr.T)

        # check that diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(coal.tree_height.demes.corr), 1)

    def test_variance(self):
        """
        Test kurtosis.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        var = coal.moment(2, rewards, center=True)

        self.assertEqual(var, coal.moment(2, rewards, center=False) - coal.moment(1, rewards[:1], center=False) ** 2)

    def test_kurtosis(self):
        """
        Test kurtosis.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        kurtosis = coal.moment(3, rewards, center=True)

        m1 = coal.moment(1, rewards[:1], center=False)
        m2 = coal.moment(2, rewards[:2], center=False)
        m3 = coal.moment(3, rewards, center=False)

        self.assertAlmostEqual(kurtosis, m3 - 3 * m2 * m1 + 2 * m1 ** 3)

    def test_skewness(self):
        """
        Test skewness.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        skewness = coal.moment(4, rewards, center=True)

        m4 = coal.moment(4, rewards, center=False)
        m1 = coal.moment(1, rewards[:1], center=False)
        m3m1 = coal.moment(3, rewards[:3], center=False) * m1
        m2m2 = coal.moment(2, rewards[:2], center=False) * m1 ** 2

        self.assertAlmostEqual(skewness, m4 - 4 * m3m1 + 6 * m2m2 - 3 * m1 ** 4)

    def test_central_m5(self):
        """
        Test central 5th moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        moment = coal.moment(5, rewards, center=True)

        m5 = coal.moment(5, rewards, center=False)
        m1 = coal.moment(1, rewards[:1], center=False)
        m4m1 = coal.moment(4, rewards[:4], center=False) * m1
        m3m2 = coal.moment(3, rewards[:3], center=False) * m1 ** 2
        m2m3 = coal.moment(2, rewards[:2], center=False) * m1 ** 3

        self.assertAlmostEqual(moment, m5 - 5 * m4m1 + 10 * m3m2 - 10 * m2m3 + 4 * m1 ** 5)

    def test_central_2nd_order_cross_moment(self):
        """
        Test 2nd order central cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(3)
        )

        moment = coal.moment(2, rewards, center=True)
        xy = coal._raw_moment(2, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(3)))
        yx = coal._raw_moment(2, (pg.UnfoldedSFSReward(3), pg.UnfoldedSFSReward(2)))
        x = coal._raw_moment(1, (pg.UnfoldedSFSReward(2),))
        y = coal._raw_moment(1, (pg.UnfoldedSFSReward(3),))

        self.assertAlmostEqual(moment, (xy + yx) / 2 - x * y)

    def test_central_3rd_order_cross_moment(self):
        """
        Test 3rd order cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(3),
            pg.UnfoldedSFSReward(4)
        )

        moment = coal.moment(3, rewards, center=True)
        xyz = coal.moment(3, rewards, center=False)
        xy = coal.moment(2, tuple(np.array(rewards)[[0, 1]]), center=False)
        xz = coal.moment(2, tuple(np.array(rewards)[[0, 2]]), center=False)
        yz = coal.moment(2, tuple(np.array(rewards)[[1, 2]]), center=False)
        xy_centered = coal.moment(2, tuple(np.array(rewards)[[0, 1]]))
        xz_centered = coal.moment(2, tuple(np.array(rewards)[[0, 2]]))
        yz_centered = coal.moment(2, tuple(np.array(rewards)[[1, 2]]))
        x = coal.moment(1, (pg.UnfoldedSFSReward(2),))
        y = coal.moment(1, (pg.UnfoldedSFSReward(3),))
        z = coal.moment(1, (pg.UnfoldedSFSReward(4),))

        self.assertAlmostEqual(moment, xyz - xy * z - xz * y - yz * x + 2 * x * y * z)
        self.assertAlmostEqual(moment, xyz - xy_centered * z - xz_centered * y - yz_centered * x - x * y * z)

    def test_3rd_order_uncentered_cross_moment(self):
        """
        Test 3rd order uncentered cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(4)
        )

        moment = coal.moment(3, rewards, center=False)
        xyz = coal._raw_moment(3, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(4)))
        xzy = coal._raw_moment(3, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(4), pg.UnfoldedSFSReward(2)))
        yxz = coal._raw_moment(3, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(4)))
        yzx = coal._raw_moment(3, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(4), pg.UnfoldedSFSReward(2)))
        zxy = coal._raw_moment(3, (pg.UnfoldedSFSReward(4), pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(2)))
        zyx = coal._raw_moment(3, (pg.UnfoldedSFSReward(4), pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(2)))

        self.assertAlmostEqual(moment, (xyz + xzy + yxz + yzx + zxy + zyx) / 6)

    def test_3rd_order_uncentered_partial_cross_moment(self):
        """
        Test 3rd order uncentered partial cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(3),
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(4)
        )

        moment = coal.moment(3, rewards, center=False, permute=False)
        moment_raw = coal._raw_moment(3, (pg.UnfoldedSFSReward(3), pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(4)))

        self.assertAlmostEqual(moment, moment_raw)

    def test_uncentered_cross_moments_msprime(self):
        """
        Test higher-order uncentered cross-moments against Msprime coalescent.
        """
        coal = self.get_complex_coalescent()
        ms = coal.to_msprime(num_replicates=100000)

        # test uncentered moments
        for indices in [[2, 3, 4], [1, 1, 4], [1, 1, 1], [4, 2, 1]]:
            m_ms = np.mean(ms.sfs.samples[:, indices].prod(axis=1))
            m_ph = coal.moment(3, tuple(pg.UnfoldedSFSReward(l) for l in indices), center=False)

            self.assertLess(2 * np.abs((m_ms - m_ph) / (m_ms + m_ph)), 0.07)

    def compare_centered_sfs_cross_moments_msprime(self):
        """
        Test higher-order centered SFS cross-moments against Msprime coalescent.
        """
        coal = pg.Coalescent(n=6)
        ms = coal.to_msprime(num_replicates=1000000, n_threads=100, seed=42)

        data = [
            dict(moments=[2, 3, 4, 4, 4], tol=0.1),
            dict(moments=[1, 2, 3, 4, 5], tol=0.1),
            dict(moments=[1, 2, 3, 4], tol=0.1),
            dict(moments=[3, 2, 3, 4], tol=0.1),
            dict(moments=[1, 1, 4], tol=0.1),
            dict(moments=[1, 1, 1], tol=0.1),
            dict(moments=[4, 2, 1], tol=0.1),
            dict(moments=[2, 3], tol=0.1)
        ]

        # test centered moments
        for config in data:
            moments, tol = config['moments'], config['tol']
            m_ph = coal.moment(len(moments), tuple(pg.UnfoldedSFSReward(l) for l in moments), center=True)
            m_ms = np.mean((ms.sfs.samples[:, moments] - ms.sfs.samples[:, moments].mean(axis=0)).prod(axis=1))

            self.assertLess(2 * np.abs((m_ms - m_ph) / (m_ms + m_ph)), tol)

    def test_mutation_configuration_probability_mass_close_to_one(self):
        """
        Test mutation configuration probability mass is close to one.
        """
        coal = pg.Coalescent(n=5)

        ms = coal.to_msprime(
            num_replicates=1000,
            seed=42,
            n_threads=1,
            parallelize=False,
            mutation_rate=0.01,
            simulate_mutations=True,
        )

        self.assertAlmostEqual(1, sum(map(lambda x: x[1], islice(ms.sfs.get_mutation_configs(), 100))))
        self.assertAlmostEqual(1, sum(map(lambda x: x[1],
                                          islice(coal.sfs.get_mutation_configs(theta=ms.mutation_rate), 100))))

    def test_pdf_large_N(self):
        """
        Test plotting pdf for different population sizes.
        """
        Ns = np.array([1e-30, 1e-20, 1e-10, 1, 1e10, 1e20, 1e30])
        _, axs = plt.subplots(len(Ns), 1, figsize=(5, 10))
        data = []

        for i, N in enumerate(Ns):
            coal = pg.Coalescent(
                n=4,
                demography=pg.Demography(
                    pop_sizes=cast(float, N)
                )
            )

            t = np.linspace(0, coal.tree_height.quantile(0.99), 200)
            data += [coal.tree_height.pdf(t=t)]
            coal.tree_height.plot_pdf(ax=axs[i], label=f'N={N}', show=False, t=t)

        plt.show()

        data = np.array(data)
        self.assertTrue((np.var(data.T * Ns[None, :], axis=1) < 1e-10).all())

    def test_accumulate_same_as_moment(self):
        """
        Test accumulation is the same as moment with adjusted end time.
        """
        coal = pg.Coalescent(n=4)

        rewards = [
            (pg.TreeHeightReward(),),
            (pg.TotalTreeHeightReward(),),
            (pg.TreeHeightReward(), pg.TreeHeightReward()),
            (pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)),
        ]

        times = np.linspace(0, coal.tree_height.quantile(0.99), 10)

        for reward in rewards:
            moments = [coal.moment(k=len(reward), rewards=reward, end_time=t) for t in times]
            accumulation = coal.accumulate(k=len(reward), rewards=reward, end_times=times)

            np.testing.assert_array_almost_equal(moments, accumulation)

    def test_accumulate_same_as_moment_sfs(self):
        """
        Test accumulation is the same as moment with adjusted end time for SFS-based moments.
        """
        coal = pg.Coalescent(n=4)

        times = np.linspace(0, coal.tree_height.quantile(0.99), 10)

        for k in [1, 2]:
            moments = np.array([coal.sfs.moment(k=k, end_time=t).data for t in times]).T
            accumulation = coal.sfs.accumulate(k=k, end_times=times)

            np.testing.assert_array_almost_equal(moments, accumulation)

    def test_get_cov_sfs(self):
        """
        Test get_cov method for SFS.
        """
        n = 4
        coal = pg.Coalescent(n=n)

        cov = coal.sfs.cov.data
        cov2 = np.array([[coal.sfs.get_cov(i, j) for i in range(n + 1)] for j in range(n + 1)])

        np.testing.assert_array_almost_equal(cov, cov2)

    def test_get_corr_sfs(self):
        """
        Test get_corr method for SFS.
        """
        n = 4
        coal = pg.Coalescent(n=n)

        corr = coal.sfs.corr.data
        corr2 = np.array([[coal.sfs.get_corr(i, j) for i in range(n + 1)] for j in range(n + 1)])

        np.testing.assert_array_almost_equal(corr, corr2)

    def test_disable_regularization(self):
        """
        Test disabling regularization.
        """
        pg.Settings.regularize = False

        coal = pg.Coalescent(n=4)

        self.assertEqual(1, coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S))
        self.assertEqual(1, coal.total_branch_length._get_regularization_factor(coal.lineage_counting_state_space.S))
        self.assertEqual(1, coal.sfs._get_regularization_factor(coal.block_counting_state_space.S))

        pg.Settings.regularize = True

    def test_enable_regularization(self):
        """
        Test enabling regularization.
        """
        pg.Settings.regularize = True

        coal = pg.Coalescent(n=4)

        self.assertNotEqual(1, coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S))
        self.assertNotEqual(
            1, coal.total_branch_length._get_regularization_factor(coal.lineage_counting_state_space.S)
        )
        self.assertNotEqual(1, coal.sfs._get_regularization_factor(coal.block_counting_state_space.S))

    def test_fewer_than_2_lineages_raises_error(self):
        """
        Test fewer than 2 lineages raises error.
        """
        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=1)

    def test_recombination_tree_height_covariance(self):
        """
        Test tree height covariance against theoretical expectations
        """
        covs = []
        covs_exp = []
        ps = [10 ** -i for i in range(10)[::-1]]

        for p in ps:
            coal = pg.Coalescent(
                n=2,
                loci=pg.LocusConfig(n=2, recombination_rate=p / 2)
            )

            cov = coal.tree_height.loci.cov[0, 1]
            cov_exp = (p + 18) / (p ** 2 + 13 * p + 18)

            covs += [cov]
            covs_exp += [cov_exp]

        plt.plot(covs, label='Observed')
        plt.plot(covs_exp, label='Expected')
        plt.xticks(range(len(covs)), ps)
        plt.legend()
        plt.show()

        np.testing.assert_allclose(covs, covs_exp, atol=1e-14, rtol=0)

    def test_rescale_S_single_kingman(self):
        """
        Test rate matrix rescaling for Kingman coalescent with a single population.
        """
        coal = pg.Coalescent(n=6)
        self.assertFalse('S' in coal.block_counting_state_space.__dict__)

        _ = coal.block_counting_state_space.S
        coal.block_counting_state_space.update_epoch(pg.Epoch(pop_sizes={'pop_0': 3}))

        np.testing.assert_array_almost_equal(
            coal.block_counting_state_space.S * 3,
            pg.Coalescent(n=6).block_counting_state_space.S
        )
        self.assertTrue('S' in coal.block_counting_state_space.__dict__)

    def test_rescale_S_beta(self):
        """
        Test rate matrix rescaling for Beta coalescent with a single population.
        """
        coal1 = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.7))
        coal2 = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.7),
                              demography=pg.Demography(pop_sizes={'pop_0': {0: 3}}))

        r = coal2.block_counting_state_space._get_scaling_factor(
            epoch_prev=next(coal1.demography.epochs),
            epoch_next=next(coal2.demography.epochs)
        )

        np.testing.assert_array_almost_equal(
            coal1.block_counting_state_space.S * r,
            coal2.block_counting_state_space.S
        )

    def test_rescale_S_dirac(self):
        """
        Test rate matrix rescaling for Dirac coalescent with a single population.
        """
        coal1 = pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.4, c=5))
        coal2 = pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.4, c=5),
                              demography=pg.Demography(pop_sizes={'pop_0': {0: 3}}))

        r = coal2.block_counting_state_space._get_scaling_factor(
            epoch_prev=next(coal1.demography.epochs),
            epoch_next=next(coal2.demography.epochs)
        )

        np.testing.assert_array_almost_equal(
            coal1.block_counting_state_space.S * r,
            coal2.block_counting_state_space.S
        )

    def test_rescale_S_multi_pop(self):
        """
        Test that rescaling does not cache S for multiple populations.
        """
        coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2})
        _ = coal.block_counting_state_space.S
        coal.block_counting_state_space.update_epoch(pg.Epoch(pop_sizes={'pop_0': 3}))

        self.assertFalse('S' in coal.block_counting_state_space.__dict__)

    def test_flattened_block_counting_standard_coalescent_2_epochs(self):
        """
        Make sure flattening block counting states works correctly.
        """
        pg.Backend.register(pg.SciPyExpmBackend())
        pg.Settings.flatten_block_counting = True
        times = np.linspace(0, 30, 10)
        n = 10
        demography = pg.Demography(
            pop_sizes={'pop_0': {0: 1, 1: 10}}
        )

        coal_flattened = pg.Coalescent(n=n, demography=demography)
        flattened = np.array([coal_flattened.sfs.get_accumulation(1, i, times) for i in range(10)])

        # make sure state probabilities are cached
        self.assertTrue('_state_probs' in coal_flattened.block_counting_state_space.__dict__)

        pg.Settings.flatten_block_counting = False
        coal_original = pg.Coalescent(n=n, demography=demography)
        original = np.array([coal_original.sfs.get_accumulation(1, i, times) for i in range(10)])

        # make sure state probabilities are not cached
        self.assertFalse('_state_probs' in coal_original.block_counting_state_space.__dict__)

        np.testing.assert_array_almost_equal(original, flattened, decimal=14)
        pg.Settings.flatten_block_counting = True

    def test_sample_empirical_pdf(self):
        """
        Test empirical PDF sampling against exact PDF.
        """
        coal = pg.Coalescent(
            n=10,
            model=pg.BetaCoalescent(alpha=1.7),
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1, 1: 10}}
            )
        )

        exact = coal.moment(1, (pg.UnfoldedSFSReward(2),))

        with pg.Settings.set_pbar():
            empirical = coal._sample(10000, (pg.UnfoldedSFSReward(2),)).mean()

        rel_diff = np.abs(empirical - exact) / exact

        self.assertLessEqual(rel_diff, 0.05)

    def test_sample_empirical_cdf(self):
        """
        Test empirical CDF sampling against exact CDF.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 1, 'pop_1': 1, 'pop_2': 1}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        )

        t = np.linspace(0, coal.tree_height.quantile(0.99), 100)
        empirical = coal.tree_height._empirical_cdf(n_samples=1000, t=t)
        exact = coal.tree_height.cdf(t=t)
        plt.plot(t, empirical, label='Empirical CDF')
        plt.plot(t, exact, label='Exact CDF')
        plt.legend()
        plt.show()

        rel_diff = np.abs(empirical - exact) / exact

        self.assertLess(rel_diff[20:].mean(), 0.02)

    def test_plot_empirical_cdf(self):
        """
        Test plotting empirical CDF.
        """
        pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 1, 'pop_1': 1, 'pop_2': 1}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        ).tree_height._plot_empirical_cdf()

    def test_compare_state_reward_flattened(self):
        """
        Test that flattened state rewards match the original state rewards.
        """
        coal = pg.Coalescent(n=10)
        k = coal.block_counting_state_space.k

        pg.Settings.flatten_block_counting = True
        flattened = [coal.moment(1, rewards=(pg.StateReward(i),)) for i in range(k)]
        self.assertTrue('_state_probs' in coal.block_counting_state_space.__dict__)

        coal = pg.Coalescent(n=10)
        pg.Settings.flatten_block_counting = False
        original = [coal.moment(1, rewards=(pg.StateReward(i),)) for i in range(k)]
        self.assertFalse('_state_probs' in coal.block_counting_state_space.__dict__)

        np.testing.assert_array_almost_equal(flattened, original)
        pg.Settings.flatten_block_counting = True

    @unittest.skip("Flattening block counting states for beta coalescent with two epochs doesn't work.")
    def test_flattened_block_counting_beta_coalescent_2_epochs(self):
        """
        Flattening the block counting states for MMCs with two doesn't work.
        """
        pg.Backend.register(pg.SciPyExpmBackend())
        pg.Settings.flatten_block_counting = True
        n = 10
        model = pg.BetaCoalescent(alpha=1.7)
        demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 10}})

        coal_flattened = pg.Coalescent(n=n, model=model, demography=demography)
        flattened = coal_flattened.sfs.mean.data
        self.assertTrue('_state_probs' in coal_flattened.block_counting_state_space.__dict__)

        pg.Settings.flatten_block_counting = False
        coal_original = pg.Coalescent(n=n, model=model, demography=demography)
        original = coal_original.sfs.mean.data
        self.assertFalse('_state_probs' in coal_original.block_counting_state_space.__dict__)

        self.assertGreater(np.nanmean(np.abs(flattened - original) / original), 0.01)
        pg.Settings.flatten_block_counting = True

    def test_not_flattened_block_counting_beta_coalescent(self):
        """
        Make sure that not flattening block counting states works correctly.
        """
        coal_original = pg.Coalescent(
            n=3,
            model=pg.BetaCoalescent(alpha=1.7),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1}})
        )
        _ = coal_original.sfs.mean.data

        # make sure state probabilities were not accessed
        self.assertFalse('_state_probs' in coal_original.block_counting_state_space.__dict__)

    def test_beta_coalescent_state_props(self):
        """
        Compare state probabilities of beta coalescent with empirical sampling.
        """
        coal = pg.Coalescent(
            n=10,
            model=pg.BetaCoalescent(1.7),
            demography=pg.Demography(pop_sizes={'pop_0': 1})
        )

        samples, probs_empirical = coal.sfs._sample(10000, record_visits=True)

        probs = coal.block_counting_state_space._state_probs

        self.assertLess((np.abs(probs_empirical - probs) / probs).mean(), 0.08)
