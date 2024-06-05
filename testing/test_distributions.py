"""
Test distributions.
"""

import itertools
from typing import Sequence, List
from unittest import TestCase

import numpy as np
import pytest
from matplotlib import pyplot as plt

import phasegen as pg
from phasegen.utils import multiset_permutations


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

    def test_multiset_permutations(self):
        """
        Test multiset permutations.
        """
        self.assertEqual(list(multiset_permutations(())), [()])
        self.assertEqual(len(list(multiset_permutations([1] * 10 + [2] * 2))), 66)

        for sets in [
            [],
            [1],
            [1, 1, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 2, 4],
            [1, 2, 3, 3, 5]
        ]:
            # compare with itertools
            self.assertEqual(set(multiset_permutations(sets)), set(itertools.permutations(sets)))

    def test_sampling_formula(self):
        """
        Test sampling formula.
        """
        coal = pg.Coalescent(
            n=3
        )

        self.assertAlmostEqual(coal.sfs.get_mutation_config(config=[0, 0], theta=1), 1 / 6)
        self.assertAlmostEqual(coal.sfs.get_mutation_config(config=[0, 0], theta=0), 1)
        self.assertAlmostEqual(coal.sfs.get_mutation_config(config=[0, 1], theta=0), 0)

        pass

    def test_plot_prob_10_singletons_2_doubletons(self):
        """
        Test plot of probability of 10 singletons and 2 doubletons.
        """
        coal = pg.Coalescent(
            n=3
        )

        x = np.linspace(0, 30, 100)
        y = np.array([coal.sfs.get_mutation_config(config=[10, 2], theta=x) for x in x])

        plt.plot(x, y)
        plt.show()

        pass

    def test_consume_mutation_configs_threshold(self):
        """
        Test consume_sample_generator with threshold.
        """
        coal = pg.Coalescent(n=5)

        it = coal.sfs.get_mutation_configs(theta=1)
        samples = list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))

        configs, probs = zip(*samples)

        plt.plot(probs)
        # use configs as x-ticks labels
        plt.xticks(range(len(configs)), [str(config) for config in configs], rotation=90)
        plt.tight_layout()
        plt.show()

        pass

    def test_get_mutation_config_negative_theta_raises_error(self):
        """
        Test that sampling with negative theta raises an error.
        """
        with self.assertRaises(ValueError) as context:
            pg.Coalescent(n=5).sfs.get_mutation_config(config=[1, 1], theta=-1)

    def test_get_mutation_config_zero_theta(self):
        """
        Test that sampling with zero theta returns probability of 1 for no mutations and 0 otherwise.
        """
        coal = pg.Coalescent(n=5)

        self.assertEqual(coal.sfs.get_mutation_config(config=[0, 0, 0, 0], theta=0), 1)
        self.assertEqual(coal.sfs.get_mutation_config(config=[0, 1, 0, 0], theta=0), 0)
        self.assertEqual(coal.sfs.get_mutation_config(config=[1, 0, 0, 0], theta=0), 0)
        self.assertEqual(coal.sfs.get_mutation_config(config=[1, 1, 0, 0], theta=0), 0)

    def test_get_mutation_config_more_than_one_epoch_raises_not_implemented_error(self):
        """
        Test that sampling with more than one epoch raises a NotImplementedError.
        """
        with self.assertRaises(NotImplementedError) as context:
            pg.Coalescent(
                n=5,
                demography=pg.Demography(
                    pop_sizes={'pop_0': {0: 5, 1: 10}}
                )
            ).sfs.get_mutation_config(config=[1, 1], theta=1)

    def test_get_mutation_config_incorrect_length_value_error(self):
        """
        Test that an error is raised when the length of the configuration is not equal to the number of lineages
        minus one.
        """
        with self.assertRaises(ValueError) as context:
            pg.Coalescent(n=5).sfs.get_mutation_config(config=[1, 1, 1], theta=1)

    def test_unfold_folded_config_odd_number_of_lineages(self):
        """
        Test unfolding folded block configurations for odd number of lineages.
        """
        coal = pg.Coalescent(n=5)

        folded = [2, 2]
        configs = coal.fsfs._unfold(folded)

        for unfolded in configs:
            np.testing.assert_array_equal(
                pg.SFS([0] + list(folded) + [0] * 3).data,
                pg.SFS([0] + list(unfolded) + [0]).fold().data
            )

    def test_unfolded_folded_configs(self):
        """
        Test unfolding folded block configurations.
        """

        def compare(n: int, config: Sequence[int], unfolded: List[Sequence[int]]):
            """
            Compare folded and unfolded configurations.

            :param n: Number of lineages.
            :param config: Folded configuration.
            :param unfolded: Unfolded configurations.
            :raises AssertionError: If the configurations are not equal.
            """
            observed = pg.Coalescent(n=n).fsfs._unfold(config)
            expected = set(tuple(c) for c in unfolded)

            return self.assertEqual(observed, expected)

        compare(2, [0], [(0,)])
        compare(2, [1], [(1,)])
        compare(2, [5], [(5,)])
        compare(3, [0], [(0, 0)])
        compare(3, [1], [(0, 1), (1, 0)])
        compare(3, [2], [(2, 0), (0, 2), (1, 1)])
        compare(5, [1, 1], [(1, 1, 0, 0), (0, 1, 0, 1), (0, 0, 1, 1), (1, 0, 1, 0)])

    def test_unfold_folded_config_even_number_of_lineages(self):
        """
        Test unfolding folded block configurations for even number of lineages.
        """
        coal = pg.Coalescent(n=4)

        folded = [2, 2]
        configs = coal.fsfs._unfold(folded)

        for unfolded in configs:
            np.testing.assert_array_equal(
                pg.SFS([0] + list(folded) + [0] * 2).data,
                pg.SFS([0] + list(unfolded) + [0]).fold().data
            )

    def test_get_folded_mutation_config(self):
        """
        Test that the folded SFS probability is equal to the sum of the unfolded SFS probabilities.
        """
        coal = pg.Coalescent(n=5)

        for config in [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 0],
            [0, 2],
            [1, 2],
            [2, 1],
            [2, 2]
        ]:
            p_folded = coal.fsfs.get_mutation_config(config=config, theta=1)

            p_unfolded = [coal.sfs.get_mutation_config(config=u, theta=1) for u in coal.fsfs._unfold(config)]

            self.assertAlmostEqual(p_folded, sum(p_unfolded))
