"""
Test distributions.
"""

import itertools
from unittest import TestCase

import numpy as np
import pytest

import phasegen as pg


class DistributionTestCase(TestCase):
    """
    Test distributions.
    """

    @staticmethod
    def get_test_coalescent() -> pg.Coalescent:
        """
        Get a test coalescent.
        """
        return pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 1, 0.2: 5},
                    pop_1={0: 0.4, 0.1: 3, 0.25: 0.3},
                    pop_2={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 0.1},
                    ('pop_1', 'pop_2'): {0: 0.2, 0.1: 0.3},
                    ('pop_2', 'pop_0'): {0: 0.4, 0.1: 0.5, 0.2: 0.6},
                    ('pop_0', 'pop_2'): {0: 0.7, 0.1: 0.8, 0.2: 0.9, 0.3: 1},
                    ('pop_2', 'pop_1'): {0: 0.1}
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=1,
                pop_1=2,
                pop_2=3
            ))
        )

    def test_quantile_raises_error_below_0(self):
        """
        Test quantile function raises error when quantile is below 0.
        """
        with self.assertRaises(ValueError) as context:
            self.get_test_coalescent().tree_height.quantile(-0.1)

        self.assertEqual(str(context.exception), 'Specified quantile must be between 0 and 1.')

    def test_quantile_raises_error_above_1(self):
        """
        Test quantile function raises error when quantile is above 1.
        """
        with self.assertRaises(ValueError) as context:
            self.get_test_coalescent().tree_height.quantile(1.1)

        self.assertEqual(str(context.exception), 'Specified quantile must be between 0 and 1.')

    def test_update_transition_matrix(self):
        """
        Test _update function against CDF for different times.
        """
        dist = self.get_test_coalescent().tree_height
        e = dist.reward._get(dist.state_space)
        alpha = dist.state_space.alpha

        for t in [0, 0.001, 0.01, 0.1, 1, 10, 100]:

            u, T, epoch = dist._update(t, 0, np.eye(dist.state_space.k), next(dist.demography.epochs))

            self.assertAlmostEqual(1 - alpha @ T @ e, dist.cdf(t))

    def test_quantile(self):
        """
        Test quantile function.
        """
        dist = self.get_test_coalescent().tree_height

        for (quantile, tol) in itertools.product([0, 0.01, 0.5, 0.99, 1], [1e-1, 1e-5, 1e-10]):
            self.assertAlmostEqual(dist.cdf(dist.quantile(quantile, precision=tol)), quantile, delta=tol)

    def test_tree_height_per_population(self):
        """
        Test population means.
        """
        dist = self.get_test_coalescent().tree_height

        m_demes = {pop: dist.moment(1, rewards=(pg.TreeHeightReward().prod(pg.DemeReward(pop)),)) for pop in
                   dist.demography.pop_names}
        m = dist.moment(1)

        self.assertAlmostEqual(m, sum(m_demes.values()), delta=1e-8)

        pass

    def test_total_branch_length_per_population(self):
        """
        Test population means.
        """
        dist = self.get_test_coalescent().total_branch_length

        m_demes = {pop: dist.moment(1, rewards=(pg.TotalBranchLengthReward().prod(pg.DemeReward(pop)),)) for pop
                   in dist.demography.pop_names}
        m = dist.moment(1)

        self.assertAlmostEqual(m, sum(m_demes.values()), delta=1e-10)

        pass

    @pytest.mark.skip(reason="Fix later")
    def test_n_4_2_loci_wrong_lineage_config_raises_error(self):
        """
        How to solve errors when passing additional populations? Be more rigorous?

        TODO Fix this.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 2},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_2', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=2,
                pop_1=2
            )),
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        )

        _ = coal.tree_height.mean

    def test_n_4_2_loci_lineage_counting_state_space(self):
        """
        Test n=4, 2 loci, lineage-counting state space.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 2},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=2,
                pop_1=2
            )),
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        )

        _ = coal.lineage_counting_state_space.S

        pass

    def test_folded_mean_sfs_test_coalescent(self):
        """
        Test folded SFS.
        """
        coal = self.get_test_coalescent()

        observed = coal.fsfs.mean
        expected = coal.sfs.mean.fold()

        np.testing.assert_array_almost_equal(observed.data, expected.data)

    def test_folded_mean_sfs_n_10(self):
        """
        Test folded SFS.
        """
        coal = pg.Coalescent(
            n=10
        )

        observed = coal.fsfs.mean
        expected = coal.sfs.mean.fold()

        np.testing.assert_array_almost_equal(observed.data, expected.data)

    def test_folded_mean_sfs_n_11(self):
        """
        Test folded SFS.
        """
        coal = pg.Coalescent(
            n=11
        )

        observed = coal.fsfs.mean
        expected = coal.sfs.mean.fold()

        np.testing.assert_array_almost_equal(observed.data, expected.data)

    def test_lineage_reward_basic_coalescent_lineage_counting_state_space(self):
        """
        Test lineage reward for basic coalescent and lineage-counting state space.
        """
        coal = pg.Coalescent(
            n=10
        )

        times = [coal.moment(1, rewards=(pg.LineageReward(i),)) for i in range(2, 11)[::-1]]

        np.testing.assert_array_almost_equal(times, [1 / (i * (i - 1) / 2) for i in range(2, 11)[::-1]])

    def test_lineage_reward_basic_coalescent_block_counting_state_space(self):
        """
        Test lineage reward for basic coalescent and block-counting state space.
        """
        coal = pg.Coalescent(
            n=10
        )

        # make sure lineage-counting state space is not supported
        r = pg.ProductReward([pg.rewards.BlockCountingUnitReward(), pg.LineageReward(2)])
        self.assertFalse(pg.Reward.support(pg.LineageCountingStateSpace, [r]))

        times = [coal.moment(1, rewards=(pg.ProductReward([
            pg.rewards.BlockCountingUnitReward(), pg.LineageReward(i)]),)) for i in range(2, 11)[::-1]]

        np.testing.assert_array_almost_equal(times, [1 / (i * (i - 1) / 2) for i in range(2, 11)[::-1]])

    def test_lineage_reward_2_demes(self):
        """
        Test lineage reward for a 2-deme coalescent.
        """
        coal = pg.Coalescent(
            n={'pop_0': 6, 'pop_1': 4},
            demography=pg.Demography(
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                },
            )
        )

        times = [coal.moment(1, rewards=(pg.LineageReward(i),)) for i in range(2, 11)[::-1]]

        # check that times add up to tree height
        self.assertAlmostEqual(sum(times), coal.tree_height.mean)

    def test_lineage_reward_2_loci(self):
        """
        Test lineage reward for a 2-locus coalescent.
        """
        coal = pg.Coalescent(
            n=6,
            loci=2,
            recombination_rate=0
        )

        times = [coal.moment(1, rewards=(pg.LineageReward(i),)) for i in range(3, 14)[::-1]]

        # check that times add up to tree height
        self.assertAlmostEqual(sum(times), coal.tree_height.mean)
