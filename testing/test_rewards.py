"""
Test reward classes.
"""

from unittest import TestCase

import numpy as np
from numpy import testing

import phasegen as pg


class RewardsTestCase(TestCase):
    """
    Test reward classes.
    """

    @staticmethod
    def test_tree_height_reward_default_state_space():
        """
        Test tree height reward for default state space.
        """
        s = pg.DefaultStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward()._get(s)

        testing.assert_array_equal(r, [1, 1, 1, 0])

    def test_tree_height_reward_block_counting_state_space(self):
        """
        Test tree height reward for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward()._get(s)

        testing.assert_array_equal(r[s._get_old_ordering()], [1, 1, 1, 1, 0])

    @staticmethod
    def test_total_branch_length_reward_default_state_space():
        """
        Test total branch length reward for default state space.
        """
        s = pg.DefaultStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TotalBranchLengthReward()._get(s)

        testing.assert_array_equal(r[s._get_old_ordering()], [4, 3, 2, 0])

    def test_total_branch_length_reward_block_counting_state_space(self):
        """
        Test total branch length reward for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TotalBranchLengthReward()._get(s)

        testing.assert_array_equal(r[s._get_old_ordering()], [4, 3, 2, 2, 0])

    @staticmethod
    def test_product_reward():
        """
        Test product reward.
        """
        r1 = pg.CustomReward(lambda _: np.diag([1, 2, 0, 4]))
        r2 = pg.CustomReward(lambda _: np.diag([1, 1, 2, 3]))
        r3 = pg.CustomReward(lambda _: np.diag([1, 0, 1, 1]))

        r = pg.ProductReward([r1, r2, r3])

        testing.assert_array_equal(r._get(None), np.array([
            [1., 0., 0., 0.],
            [0., 0, 0., 0.],
            [0., 0., 0, 0.],
            [0., 0., 0., 12.]]
        ))

    def test_use_rewards_for_wrong_state_space_raises_error(self):
        """
        Test that using rewards for the wrong state space raises an error.
        """
        coal = pg.Coalescent(n=5)

        with self.assertRaises(NotImplementedError) as context:
            _ = coal.tree_height.moment(1, (pg.UnfoldedSFSReward(2),))

    def test_supports_state_space(self):
        """
        Test that rewards support state space.
        """
        self.assertTrue(pg.Reward.support(pg.DefaultStateSpace, [pg.TreeHeightReward()]))
        self.assertTrue(pg.Reward.support(pg.BlockCountingStateSpace, [pg.TreeHeightReward()]))
        self.assertTrue(
            pg.Reward.support(pg.DefaultStateSpace, [pg.TreeHeightReward(), pg.TotalBranchLengthReward()])
        )
        self.assertFalse(pg.Reward.support(pg.DefaultStateSpace, [pg.TreeHeightReward(), pg.UnfoldedSFSReward(2)]))

        self.assertTrue(
            pg.Reward.support(pg.BlockCountingStateSpace, [pg.ProductReward([pg.TreeHeightReward()])])
        )
        self.assertTrue(pg.Reward.support(pg.DefaultStateSpace, [pg.ProductReward([pg.TreeHeightReward()])]))
        self.assertFalse(pg.Reward.support(
            pg.DefaultStateSpace, [pg.ProductReward([pg.TreeHeightReward(), pg.UnfoldedSFSReward(2)])]
        ))
