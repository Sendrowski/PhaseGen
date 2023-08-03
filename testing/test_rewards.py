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
            pop_config=pg.PopulationConfig(n=4),
            demography=pg.TimeHomogeneousDemography(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward().get(s)

        testing.assert_array_equal(r, np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.]]
        ))
