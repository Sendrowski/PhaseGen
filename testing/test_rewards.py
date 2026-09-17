"""
Test reward classes.
"""

from testing import TestCase
from testing.state_space_parity import old_ordering

import numpy as np
import pytest
from numpy import testing

import phasegen as pg


class RewardsTestCase(TestCase):
    """
    Test reward classes.
    """

    @staticmethod
    def test_tree_height_reward_lineage_counting_state_space():
        """
        Test tree height reward for lineage-counting state space.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward()._get(s)

        testing.assert_array_equal(r, [1, 1, 1, 0])

    def test_tree_height_reward_block_counting_state_space(self):
        """
        Test tree height reward for block-counting state space.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TreeHeightReward()._get(s)

        testing.assert_array_equal(r[old_ordering(s)], [1, 1, 1, 1, 0])

    def test_total_tree_height_reward_block_counting_state_space(self):
        """
        Regression test for bug #12: TotalTreeHeightReward crashed on a BlockCountingStateSpace.

        Before the fix, TotalTreeHeightReward._get delegated to LocusReward, which only handles
        LineageCountingStateSpace, so evaluating it on a BlockCountingStateSpace raised
        NotImplementedError('Unsupported state space type for reward LocusReward: BlockCountingStateSpace').
        The fix implements the block-counting case directly. Also assert the lineage-counting path
        is unchanged so the fix did not alter it.
        """
        s_block = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r_block = pg.TotalTreeHeightReward()._get(s_block)

        testing.assert_array_equal(r_block[old_ordering(s_block)], [1, 1, 1, 1, 0])

        s_lineage = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r_lineage = pg.TotalTreeHeightReward()._get(s_lineage)

        testing.assert_array_equal(r_lineage, [1, 1, 1, 0])

    @staticmethod
    def test_total_branch_length_reward_lineage_counting_state_space():
        """
        Test total branch length reward for lineage-counting state space.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TotalBranchLengthReward()._get(s)

        testing.assert_array_equal(r[old_ordering(s)], [4, 3, 2, 0])

    def test_total_branch_length_reward_block_counting_state_space(self):
        """
        Test total branch length reward for block-counting state space.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        r = pg.TotalBranchLengthReward()._get(s)

        testing.assert_array_equal(r[old_ordering(s)], [4, 3, 2, 2, 0])

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
        self.assertTrue(pg.Reward.support(pg.LineageCountingStateSpace, [pg.TreeHeightReward()]))
        self.assertTrue(pg.Reward.support(pg.BlockCountingStateSpace, [pg.TreeHeightReward()]))
        self.assertTrue(
            pg.Reward.support(pg.LineageCountingStateSpace, [pg.TreeHeightReward(), pg.TotalBranchLengthReward()])
        )
        self.assertFalse(pg.Reward.support(pg.LineageCountingStateSpace, [pg.TreeHeightReward(), pg.UnfoldedSFSReward(2)]))

        self.assertTrue(
            pg.Reward.support(pg.BlockCountingStateSpace, [pg.ProductReward([pg.TreeHeightReward()])])
        )
        self.assertTrue(pg.Reward.support(pg.LineageCountingStateSpace, [pg.ProductReward([pg.TreeHeightReward()])]))
        self.assertFalse(pg.Reward.support(
            pg.LineageCountingStateSpace, [pg.ProductReward([pg.TreeHeightReward(), pg.UnfoldedSFSReward(2)])]
        ))

    def test_combined_reward(self):
        """
        Test combined reward.
        """
        rewards = [pg.LineageReward(5), pg.TreeHeightReward()]
        r = pg.CombinedReward(rewards)

        # no change
        self.assertEqual(tuple(r.rewards), tuple(rewards))

        rewards = [pg.TotalBranchLengthReward(), pg.LocusReward(1)]
        r = pg.CombinedReward(rewards)

        # reward transformation to TotalBranchLengthLocusReward
        self.assertEqual(tuple(r.rewards), tuple([pg.TotalBranchLengthLocusReward(1)]))

        rewards = [pg.TreeHeightReward(), pg.LocusReward(2), pg.TreeHeightReward(), pg.TotalBranchLengthReward()]
        r = pg.CombinedReward(rewards)

        # reward transformation to TotalBranchLengthLocusReward
        self.assertEqual(
            set(r.rewards),
            {pg.TreeHeightReward(), pg.TreeHeightReward(), pg.TotalBranchLengthLocusReward(2)}
        )


def _two_island_demography() -> pg.Demography:
    """Two islands with sizes 1 (``pop_0``) and 5 (``pop_1``) and symmetric migration rate 1."""
    return pg.Demography(
        pop_sizes={'pop_0': 1.0, 'pop_1': 5.0},
        migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0}
    )


@pytest.mark.parametrize("n", [{'pop_1': 2}, {'pop_0': 0, 'pop_1': 2}, {'pop_1': 2, 'pop_0': 0}])
def test_deme_reward_independent_of_lineage_order(n):
    """
    DemeReward looked up the deme in the sorted demography order while the state-space deme axis follows the
    lineage configuration, so a lineage dict not in sorted order swapped the per-deme marginals. The expected values
    come from the three-state chain (both lineages in ``pop_1``, one per deme, both in ``pop_0``) with sub-generator
    ``[[-2.2, 2, 0], [1, -2, 1], [0, 2, -3]]``, whose expected sojourns from the first state are ``[10/7, 15/7, 5/7]``.
    """
    c = pg.Coalescent(n=n, demography=_two_island_demography())

    assert c.tree_height.demes['pop_1'].mean == pytest.approx(10 / 7 + 15 / 14, rel=1e-8)
    assert c.tree_height.demes['pop_0'].mean == pytest.approx(5 / 7 + 15 / 14, rel=1e-8)


def test_deme_reward_auto_added_populations_follow_demography_order():
    """
    Populations present only in the demography were appended to the lineage configuration in set-iteration order,
    so the deme axis, and with the sorted-name lookup the per-deme marginals, depended on the hash seed.
    """
    demo = pg.Demography(
        pop_sizes={'a': 1.0, 'b': 1.0, 'c': 5.0},
        migration_rates={(p, q): 1.0 for p in 'abc' for q in 'abc' if p != q}
    )
    c = pg.Coalescent(n={'a': 2}, demography=demo)
    ref = pg.Coalescent(n={'a': 2, 'b': 0, 'c': 0}, demography=demo)

    assert c.lineage_config.pop_names == ['a', 'b', 'c']

    for p in 'abc':
        assert c.tree_height.demes[p].mean == pytest.approx(ref.tree_height.demes[p].mean, rel=1e-8)


@pytest.mark.parametrize("reward", [
    pg.UnfoldedSFSReward(-1), pg.UnfoldedSFSReward(-3), pg.UnfoldedSFSReward(5),
    pg.FoldedSFSReward(-1), pg.FoldedSFSReward(5)
])
def test_sfs_reward_index_out_of_range_raises(reward):
    """
    SFS rewards indexed the block axis with ``index - 1`` unchecked, so a negative index wrapped around to another bin
    (``UnfoldedSFSReward(-1)`` returned bin ``n - 1``) and ``FoldedSFSReward(-1)`` raised an unrelated IndexError.
    """
    with pytest.raises(ValueError, match="index must lie in"):
        pg.Coalescent(n=4).moment(1, [reward])


@pytest.mark.parametrize("flatten", [True, False])
def test_sfs_reward_matches_every_spectrum_entry(flatten):
    """
    Each SFS reward index ``0, ..., n`` gives the matching entry of the spectrum mean. ``FoldedSFSReward(3)`` for
    ``n = 4`` counted blocks of sizes 3 and 1, returning folded bin 1 instead of the empty entry 3.
    """
    prev = pg.Settings.flatten_block_counting
    pg.Settings.flatten_block_counting = flatten

    try:
        c = pg.Coalescent(n=4)

        for i in range(5):
            assert c.moment(1, [pg.UnfoldedSFSReward(i)]) == pytest.approx(c.sfs.mean.data[i], abs=1e-10)
            assert c.moment(1, [pg.FoldedSFSReward(i)]) == pytest.approx(c.fsfs.mean.data[i], abs=1e-10)
    finally:
        pg.Settings.flatten_block_counting = prev
