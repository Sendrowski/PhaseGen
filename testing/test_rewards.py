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
            pop_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward().get(s)

        testing.assert_array_equal(r, [1, 1, 1, 0])

    def test_tree_height_reward_block_counting_state_space(self):
        """
        Test tree height reward for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward().get(s)

        testing.assert_array_equal(r, [1, 1, 1, 1, 0])

    @staticmethod
    def test_total_branch_length_reward_default_state_space():
        """
        Test total branch length reward for default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TotalBranchLengthReward().get(s)

        testing.assert_array_equal(r, [4, 3, 2, 0])

    def test_total_branch_length_reward_block_counting_state_space(self):
        """
        Test total branch length reward for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TotalBranchLengthReward().get(s)

        testing.assert_array_equal(r, [4, 3, 2, 2, 0])

    @staticmethod
    def test_product_reward():
        """
        Test product reward.
        """
        r1 = pg.CustomReward(lambda _: np.diag([1, 2, 0, 4]))
        r2 = pg.CustomReward(lambda _: np.diag([1, 1, 2, 3]))
        r3 = pg.CustomReward(lambda _: np.diag([1, 0, 1, 1]))

        r = pg.ProductReward([r1, r2, r3])

        testing.assert_array_equal(r.get(None), np.array([
            [1., 0., 0., 0.],
            [0., 0, 0., 0.],
            [0., 0., 0, 0.],
            [0., 0., 0., 12.]]
        ))
