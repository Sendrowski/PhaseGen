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

        # the locus reward restricts the branch-length reward instead of multiplying it
        self.assertEqual(tuple(r.rewards), tuple([pg.RestrictedReward(pg.TotalBranchLengthReward(), locus=1)]))

        rewards = [pg.TotalBranchLengthReward(), pg.LocusReward(1), pg.DemeReward('pop_0')]
        r = pg.CombinedReward(rewards)

        # both restrictions are applied
        self.assertEqual(tuple(r.rewards), tuple([
            pg.RestrictedReward(pg.RestrictedReward(pg.TotalBranchLengthReward(), locus=1), pop='pop_0')
        ]))

        # the reward vector of the restricted branch length is the lineage count of the focal locus alone
        c = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
        s = c.lineage_counting_state_space

        testing.assert_array_equal(
            pg.CombinedReward([pg.TotalBranchLengthReward(), pg.LocusReward(1)])._get(s),
            s.lineages.sum(axis=(2, 3))[:, 1] * (s.lineages.sum(axis=(2, 3))[:, 1] > 1)
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


def _two_deme_demography() -> pg.Demography:
    """Two demes of sizes 1 (``a``) and 1.5 (``b``) with asymmetric migration."""
    return pg.Demography(
        pop_sizes={'a': 1.0, 'b': 1.5},
        migration_rates={('a', 'b'): 0.6, ('b', 'a'): 0.3}
    )


@pytest.mark.parametrize("r", [0.0, 1.0, 50.0])
def test_per_deme_branch_length_independent_of_recombination_rate(r):
    """
    The deme restriction was a fraction of the lineage count pooled over both loci, multiplied onto the two-locus
    branch-length reward, instead of the per-locus lineage count of the deme summed over loci. The per-deme
    two-locus branch length therefore drifted with the recombination rate (deme ``a``: 8.0741 at ``r = 0``, 8.0411
    at ``r = 0.5``, 7.9873 at ``r = 50``), although every locus is marginally a single-locus coalescent, so it must
    be twice the single-locus value at every rate.
    """
    single = pg.Coalescent(n={'a': 2, 'b': 1}, demography=_two_deme_demography())
    two = pg.Coalescent(n={'a': 2, 'b': 1}, demography=_two_deme_demography(), loci=2, recombination_rate=r)

    for pop in ('a', 'b'):
        assert two.total_branch_length.demes[pop].mean == pytest.approx(
            2 * single.total_branch_length.demes[pop].mean, rel=1e-8
        )

    assert sum(two.total_branch_length.demes[pop].mean for pop in ('a', 'b')) == pytest.approx(
        two.total_branch_length.mean, rel=1e-8
    )


def test_deme_and_locus_marginals_commute():
    """
    ``demes[p].loci[l]`` nested the deme restriction inside the locus restriction, which the reward rewrite did not
    see, so it multiplied the lineage count pooled over both loci by the locus indicator: the per-locus pieces of
    deme ``a`` summed to 14.0047 against a deme total of 8.0254, while ``loci[l].demes[p]`` gave 8.0254. Both
    orderings name the same marginal and must agree.
    """
    c = pg.Coalescent(n={'a': 2, 'b': 1}, demography=_two_deme_demography(), loci=2, recombination_rate=1.0)

    for pop in ('a', 'b'):
        by_locus = [c.total_branch_length.loci[locus].demes[pop].mean for locus in range(2)]
        by_deme = [c.total_branch_length.demes[pop].loci[locus].mean for locus in range(2)]

        assert by_deme == pytest.approx(by_locus, rel=1e-8)
        assert sum(by_deme) == pytest.approx(c.total_branch_length.demes[pop].mean, rel=1e-8)

    # with a single deme the deme restriction is vacuous, so both marginals equal the per-locus value
    single_deme = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
    pop = single_deme.lineage_config.pop_names[0]

    for locus in range(2):
        assert single_deme.total_branch_length.demes[pop].loci[locus].mean == pytest.approx(
            single_deme.total_branch_length.loci[locus].mean, rel=1e-8
        )


@pytest.mark.parametrize("reward", [
    pg.rewards.UnitReward(), pg.DemeReward('pop_0'), pg.CustomReward(lambda s: np.ones(s.k))
])
def test_reward_non_zero_on_absorbing_states_is_rejected(reward):
    """
    A reward that is non-zero on an absorbing state kept accruing after absorption in every finite epoch, so
    splitting an epoch into identical epochs changed the moment (``UnitReward`` moved from 1.5 to 1.5092 with one
    redundant boundary and to 3.0896 with three). Such a reward does not accumulate until absorption and is
    rejected.
    """
    c = pg.Coalescent(n=4)

    with pytest.raises(ValueError, match="does not define an accumulation until absorption"):
        c.moment(1, rewards=[reward])


def test_state_reward_on_an_absorbing_state_is_rejected():
    """
    A reward supported on the absorbing set alone accumulates zero, since the accumulation stops at absorption, but
    the engine returned the time the finite epochs spend absorbed: 0 for one epoch, 0.0092 with one redundant
    identical boundary and 1.5896 with three, on both the closed-form and the matrix-exponential route. It is
    rejected rather than having its absorbing entry dropped, while a transient state reward is accumulated as before.
    """
    c = pg.Coalescent(n=4)
    absorbing = np.where(c.block_counting_state_space.absorbing)[0]
    transient = np.where(~c.block_counting_state_space.absorbing)[0]

    with pytest.raises(ValueError, match="does not define an accumulation until absorption"):
        c.moment(1, rewards=[pg.StateReward(int(absorbing[0]))])

    assert c.moment(1, rewards=[pg.StateReward(int(transient[0]))], center=False) > 0


def test_per_deme_tree_height_invariant_to_epoch_splitting():
    """
    The per-deme tree height was the deme fraction, which is one on the absorbing state holding the last lineage,
    multiplied by the transient indicator, and the multi-epoch path collected it after absorption: the per-deme
    heights summed to 4.8301 instead of the tree height 4.7269 once a redundant epoch boundary was added. The
    per-deme rewards vanish on the absorbing states, so the marginals are unchanged by splitting an epoch.
    """
    one = pg.Demography(pop_sizes={'a': 1.0, 'b': 1.5}, migration_rates={('a', 'b'): 0.6, ('b', 'a'): 0.3})
    split = pg.Demography(
        pop_sizes={'a': {0: 1.0, 2.0: 1.0}, 'b': {0: 1.5, 2.0: 1.5}},
        migration_rates={('a', 'b'): 0.6, ('b', 'a'): 0.3}
    )

    for demography in (one, split):
        c = pg.Coalescent(n={'a': 2, 'b': 2}, demography=demography)
        per_deme = {pop: c.tree_height.demes[pop].mean for pop in ('a', 'b')}

        assert per_deme['a'] == pytest.approx(1.829352321, rel=1e-8)
        assert sum(per_deme.values()) == pytest.approx(c.tree_height.mean, rel=1e-8)


@pytest.mark.parametrize("end_time", [1.0, np.inf], ids=["finite-end-time", "until-absorption"])
@pytest.mark.parametrize("k", [1, 2])
def test_reward_non_zero_on_absorbing_states_is_rejected_over_a_window(k, end_time):
    """
    The guard rejecting a reward with mass on an absorbing state ran only after the windowed early return, so a
    moment over a window opening after time zero was accepted while the same reward raised on the accumulation from
    zero. What came back was the deterministic window length raised to the k-th power, the reward accruing after
    absorption up to the internal absorption-time estimate: ``UnitReward`` gave 4032.25 = (64 - 0.5)**2 for n = 2, 4
    and 8 alike, 64 being that estimate rather than any property of the coalescent.
    """
    c = pg.Coalescent(n=4, start_time=0.5)

    with pytest.raises(ValueError, match="does not define an accumulation until absorption"):
        c.moment(k, rewards=[pg.rewards.UnitReward()] * k, end_time=end_time, center=False)

    with pytest.raises(ValueError, match="does not define an accumulation until absorption"):
        c.moment(k, rewards=[pg.StateReward(int(np.where(c.block_counting_state_space.absorbing)[0][0]))] * k,
                 end_time=end_time, center=False)


def test_flattened_accumulation_still_rejects_a_reward_with_absorbing_mass():
    """
    The flattening route onto the lineage-counting state space is taken before the absorbing-state guard, so the
    guard must still reject a reward with mass on an absorbing state there, without building the block-counting
    state space for the rewards that do vanish on them.
    """
    sfs = pg.Coalescent(n=6).sfs
    assert sfs._flattening_applies(1) is True

    with pytest.raises(ValueError, match="does not define an accumulation until absorption"):
        sfs._accumulate(1, (np.inf,), (pg.rewards.UnitReward(),))

    assert sfs._accumulate(1, (np.inf,), (pg.rewards.UnfoldedSFSReward(1),))[0] == pytest.approx(2.0, rel=1e-10)


def _asymmetric_slow_migration() -> pg.Demography:
    """Two demes of sizes 1 (``pop_0``) and 2 (``pop_1``) with backward migration rates 0.2 (``pop_0`` to ``pop_1``)
    and 0.5 (``pop_1`` to ``pop_0``), slow enough that a lineage's deme stays informative about how many samples it
    subtends."""
    return pg.Demography(
        pop_sizes={'pop_0': 1.0, 'pop_1': 2.0},
        migration_rates={('pop_0', 'pop_1'): 0.2, ('pop_1', 'pop_0'): 0.5}
    )


def _block_count_reward(state_space_type, pop_index: int, blocks) -> pg.CustomReward:
    """The number of blocks of the given sizes residing in the given deme at the first locus, read straight off the
    state space as an independent reference for the deme-resolved reward parts."""
    return pg.CustomReward(
        lambda s: s.lineages[:, 0, pop_index, blocks].sum(axis=1),
        supports=lambda t: t is state_space_type
    )


def test_per_deme_sfs_counts_the_blocks_residing_in_the_deme():
    """
    The per-deme spectra weighted each frequency class by the global fraction of lineages in the deme,
    a_j(x) n_d(x) / n(x), instead of counting the class-j blocks that reside in the deme, a_j^(d)(x). Under
    asymmetric, slow migration the two disagree: for the model below, sfs.demes['pop_0'].mean was
    [0, 3.3666, 1.9707, 1.10596] and sfs.demes['pop_1'].mean was [0, 2.73957, 1.45575, 0.75109], against the
    MsprimeCoalescent estimates (200000 replicates, record_migration=True, seed 42)
    [0, 3.00442, 2.13824, 1.30285] +- [0, 0.00515, 0.00621, 0.0044] and
    [0, 3.10201, 1.30385, 0.54825] +- [0, 0.00569, 0.00534, 0.0031], a maximum relative deviation of 15.1% and 37.0%
    (27 to 70 simulation standard errors). The deme-resolved count brings this to 0.49% and 0.63% (0.14 to 1.55
    standard errors).
    """
    c = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_asymmetric_slow_migration())
    pops = c.lineage_config.pop_names
    n = c.lineage_config.n

    for pop in pops:
        i = pops.index(pop)

        unfolded = [
            c.moment(1, rewards=[_block_count_reward(pg.BlockCountingStateSpace, i, [j - 1])], center=False)
            for j in range(1, n)
        ]
        testing.assert_allclose(np.array(c.sfs.demes[pop].mean.data)[1:n], unfolded, rtol=1e-10)

        folded = [
            c.moment(
                1,
                rewards=[_block_count_reward(pg.BlockCountingStateSpace, i, sorted({j - 1, n - j - 1}))],
                center=False
            )
            for j in range(1, n // 2 + 1)
        ]
        testing.assert_allclose(np.array(c.fsfs.demes[pop].mean.data)[1:n // 2 + 1], folded, rtol=1e-10)

    # the per-deme spectra partition the pooled one
    for spectrum in (c.sfs, c.fsfs):
        testing.assert_allclose(
            np.sum([np.array(spectrum.demes[pop].mean.data) for pop in pops], axis=0),
            np.array(spectrum.mean.data),
            rtol=1e-10
        )

    # the fraction-weighted quantity sits far from the deme-resolved one, so the assertion discriminates
    fraction_weighted = c.moment(
        1,
        rewards=[pg.ProductReward([pg.RestrictedReward(pg.TreeHeightReward(), pop='pop_0'), pg.UnfoldedSFSReward(3)])],
        center=False
    )
    assert abs(fraction_weighted - c.sfs.demes['pop_0'].mean.data[3]) > 0.19


def test_per_deme_sfs_agrees_between_the_batched_and_the_per_bin_path():
    """
    The batched closed-form mean of the spectrum multiplied the distribution's reward by the bin reward directly
    instead of combining them, so it bypassed the deme-resolved parts and kept returning the fraction-weighted
    per-deme spectrum ([0, 3.3666, 1.9707, 1.10596] for ``pop_0``) while the per-bin path returned the deme-resolved
    one ([0, 3.00315, 2.13087, 1.30925]).
    """
    demography = _asymmetric_slow_migration()
    n = {'pop_0': 2, 'pop_1': 2}
    default = pg.Settings.closed_form_last_epoch

    spectra = {}
    try:
        for closed_form in (True, False):
            pg.Settings.closed_form_last_epoch = closed_form
            c = pg.Coalescent(n=n, demography=demography)
            spectra[closed_form] = {
                pop: (np.array(c.sfs.demes[pop].mean.data), np.array(c.fsfs.demes[pop].mean.data))
                for pop in c.lineage_config.pop_names
            }
    finally:
        pg.Settings.closed_form_last_epoch = default

    for pop, (sfs, fsfs) in spectra[True].items():
        testing.assert_allclose(sfs, spectra[False][pop][0], rtol=1e-8)
        testing.assert_allclose(fsfs, spectra[False][pop][1], rtol=1e-8)


def test_per_deme_joint_sfs_counts_the_lineages_residing_in_the_deme():
    """
    Weighting a joint SFS bin by DemeReward split the bin by the global fraction of lineages in the deme rather than
    by the deme the subtending lineages reside in, so the per-deme pieces of bin (1, 0) under the asymmetric,
    slow-migration model were [1.26932, 0.97259] instead of [1.82762, 0.41428].
    """
    c = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=_asymmetric_slow_migration())
    pops = c.lineage_config.pop_names
    ss = c.joint_block_counting_state_space

    for config in [(1, 0), (0, 1), (1, 1), (2, 0)]:
        block = ss.block_index[config]

        per_deme = [
            c.moment(1, rewards=[pg.CombinedReward([pg.DemeReward(pop), pg.JointSFSReward(config)])], center=False)
            for pop in pops
        ]
        expected = [
            c.moment(
                1,
                rewards=[_block_count_reward(pg.JointBlockCountingStateSpace, i, [block])],
                center=False
            )
            for i in range(len(pops))
        ]

        testing.assert_allclose(per_deme, expected, rtol=1e-10)
        assert sum(per_deme) == pytest.approx(c.jsfs.mean[config], rel=1e-10)


def test_deme_restricted_tree_height_is_refused_for_multiple_loci():
    """
    The shared parts of the height rewards sum, over loci and demes, to the total tree height reward rather than to
    the tree height reward, so a deme restriction of TreeHeightReward silently returned the total tree height: for
    two loci at recombination rate 1 with n = 3 and a single deme, where the restriction is vacuous,
    CombinedReward([TreeHeightReward(), DemeReward('pop_0')]) gave 2.6666666666666665, the exact two-locus
    E[sum of the per-locus heights], instead of the tree height 1.6852380952380952. The per-locus restriction, which
    needs these parts, is unaffected.
    """
    c = pg.Coalescent(n=3, loci=2, recombination_rate=1.0)
    pop = c.lineage_config.pop_names[0]

    for reward in (
            pg.RestrictedReward(pg.TreeHeightReward(), pop=pop),
            pg.CombinedReward([pg.TreeHeightReward(), pg.DemeReward(pop)])
    ):
        with pytest.raises(NotImplementedError, match='does not decompose additively over the 2 loci'):
            c.moment(1, rewards=[reward])

    # the per-locus height, and the per-locus height restricted to the only deme, are the single-locus E[T_MRCA]
    for reward in (
            pg.RestrictedReward(pg.TreeHeightReward(), locus=0),
            pg.RestrictedReward(pg.RestrictedReward(pg.TreeHeightReward(), locus=0), pop=pop),
            pg.CombinedReward([pg.TreeHeightReward(), pg.LocusReward(0), pg.DemeReward(pop)]),
            pg.CombinedReward([pg.TreeHeightReward(), pg.DemeReward(pop), pg.LocusReward(0)])
    ):
        assert c.moment(1, rewards=[reward]) == pytest.approx(4 / 3, rel=1e-12)

    # the rewards that are additive over loci are unaffected, and their restriction to the only deme is the identity
    for reward, expected in (
            (pg.TotalTreeHeightReward(), 2 * 4 / 3),
            (pg.TotalBranchLengthReward(), 2 * 3.0)
    ):
        assert c.moment(1, rewards=[pg.CombinedReward([reward, pg.DemeReward(pop)])]) == pytest.approx(
            c.moment(1, rewards=[reward]), rel=1e-12
        )
        assert c.moment(1, rewards=[reward]) == pytest.approx(expected, rel=1e-12)

    # with a single locus the deme restriction of the tree height reward is the tree height itself
    single = pg.Coalescent(n=3)
    assert single.moment(
        1, rewards=[pg.CombinedReward([pg.TreeHeightReward(), pg.DemeReward(single.lineage_config.pop_names[0])])]
    ) == pytest.approx(single.tree_height.mean, rel=1e-12)


def test_sum_of_deme_rewards_restricts_to_the_union_of_demes():
    """A SumReward of DemeRewards combined with the SFS reward restricts it to the lineages residing in those
    demes, so it equals the sum of the per-deme spectra. Regression: the sum fell through to an elementwise product,
    weighting each frequency class by the share of all lineages in the demes (3.4157 against 3.5216 in bin 1)."""
    coal = pg.Coalescent(
        n={'pop_0': 3, 'pop_1': 2, 'pop_2': 0},
        demography=pg.Demography(
            pop_sizes={'pop_0': 3, 'pop_1': 0.5, 'pop_2': 0.1},
            events=[pg.SymmetricMigrationRateChanges(pops=['pop_0', 'pop_1', 'pop_2'], rate=1)]
        )
    )

    union = coal.sfs.moment(1, (pg.SumReward([pg.DemeReward('pop_0'), pg.DemeReward('pop_1')]),))
    parts = coal.sfs.demes['pop_0'].mean.data + coal.sfs.demes['pop_1'].mean.data

    np.testing.assert_allclose(union.data, parts, rtol=1e-10)


def test_restrictions_stacked_on_a_union_resolve_the_residence():
    """A restriction applied on top of a union restricts the residence-resolved parts of the union. Regression: the
    union's SumReward split its value by lineage counts, so two intersecting deme unions gave 2.7520 against 2.2411
    for the one deme they share, and a locus union restricted to a deme gave 3.232744 against 3.232323."""
    coal = pg.Coalescent(
        n={'pop_0': 3, 'pop_1': 2, 'pop_2': 1},
        demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1},
            events=[pg.SymmetricMigrationRateChanges(pops=['pop_0', 'pop_1', 'pop_2'], rate=1)]
        )
    )
    sfs = pg.UnfoldedSFSReward(1)

    stacked = pg.CombinedReward([
        sfs,
        pg.SumReward([pg.DemeReward('pop_0'), pg.DemeReward('pop_1')]),
        pg.SumReward([pg.DemeReward('pop_1'), pg.DemeReward('pop_2')])
    ])
    single = pg.CombinedReward([sfs, pg.DemeReward('pop_1')])

    assert coal.moment(1, [stacked], center=False) == pytest.approx(coal.moment(1, [single], center=False), rel=1e-10)

    coal = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 1},
        loci=pg.LocusConfig(n=2, recombination_rate=1),
        demography=pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 1,
                                                                                    ('pop_1', 'pop_0'): 1})
    )
    height = pg.TotalTreeHeightReward()

    union = pg.CombinedReward([height, pg.SumReward([pg.LocusReward(0), pg.LocusReward(1)]), pg.DemeReward('pop_0')])
    deme = pg.CombinedReward([height, pg.DemeReward('pop_0')])

    assert coal.moment(1, [union], center=False) == pytest.approx(coal.moment(1, [deme], center=False), rel=1e-10)


def test_a_sum_of_sfs_rewards_keeps_its_deme_resolution_inside_a_product():
    """A sum of residence-resolving rewards resolves residence as a factor of a product, so multiplying it by the
    unit reward leaves its deme restriction unchanged. Regression: the sum was split by lineage counts once it was a
    factor, giving 2.8348 against 2.8764."""
    coal = pg.Coalescent(
        n={'a': 2, 'b': 2},
        demography=pg.Demography(pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 1, ('b', 'a'): 0.5})
    )
    both = pg.SumReward([pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)])

    def mean(*rewards):
        return coal.moment(1, [pg.CombinedReward(list(rewards))], center=False)

    per_bin = mean(pg.UnfoldedSFSReward(1), pg.DemeReward('a')) + mean(pg.UnfoldedSFSReward(2), pg.DemeReward('a'))

    assert mean(both, pg.DemeReward('a')) == pytest.approx(per_bin, rel=1e-10)
    assert mean(both, pg.rewards.UnitReward(), pg.DemeReward('a')) == pytest.approx(per_bin, rel=1e-10)


def test_custom_rewards_with_equal_representations_are_distinct():
    """Custom rewards compare by the identity of their function. Regression: they compared by its string form, so two
    callables printing alike shared cached moments, and the second reward was served the first one's mean."""
    class Scaled:
        def __init__(self, c):
            self.c = c

        def __call__(self, state_space):
            return self.c * pg.TreeHeightReward()._get(state_space)

        def __repr__(self):
            return 'Scaled'

    supports = lambda s: s is pg.LineageCountingStateSpace
    coal = pg.Coalescent(n=3)

    one = coal.moment(1, [pg.CustomReward(Scaled(1), supports=supports)], center=False)
    five = coal.moment(1, [pg.CustomReward(Scaled(5), supports=supports)], center=False)

    assert five == pytest.approx(5 * one, rel=1e-12)


def test_a_reward_supporting_the_two_locus_space_uses_block_counting_on_one_locus():
    """The two-locus space is chosen only for two loci and one deme. Regression: a custom reward supporting it was
    routed there on a single-locus coalescent, which raised."""
    reward = pg.CustomReward(
        lambda s: pg.TreeHeightReward()._get(s),
        supports=lambda s: s in (pg.BlockCountingStateSpace, pg.TwoLocusBlockCountingStateSpace)
    )
    coal = pg.Coalescent(n=3)

    assert isinstance(coal._select_state_space([reward]), pg.BlockCountingStateSpace)
    assert coal.moment(1, [reward], center=False) == pytest.approx(coal.tree_height.mean, rel=1e-12)


def test_custom_reward_accepts_an_unhashable_callable():
    """The function of a custom reward is hashed by identity, so a callable without a hash of its own works.
    Regression: hashing the callable itself raised TypeError for a dataclass instance."""
    from dataclasses import dataclass

    @dataclass
    class Height:
        scale: float = 1.0

        def __call__(self, state_space):
            return self.scale * pg.TreeHeightReward()._get(state_space)

    reward = pg.CustomReward(Height(), supports=lambda s: s is pg.LineageCountingStateSpace)

    assert pg.Coalescent(n=3).moment(1, [reward], center=False) == pytest.approx(4 / 3, rel=1e-12)


def test_restricting_a_product_of_a_restriction_again_is_exact():
    """A restriction resolves residence, so a product or combination wrapping it keeps its parts when restricted again.
    Regression: the product re-split the restricted branch length by lineage counts over every deme, giving 1.7575
    instead of 3.0022 for deme a restricted twice, and 1.2447 instead of 0 for deme a then deme b."""
    coal = pg.Coalescent(
        n={'a': 2, 'b': 2},
        demography=pg.Demography(pop_sizes={'a': 1, 'b': 0.5}, migration_rates={('a', 'b'): 0.7, ('b', 'a'): 0.3})
    )
    tbl = pg.TotalBranchLengthReward()

    def mean(reward):
        return coal.moment(1, [reward], center=False)

    flat = mean(pg.CombinedReward([tbl, pg.DemeReward('a')]))

    assert mean(pg.CombinedReward([pg.CombinedReward([tbl, pg.DemeReward('a')]), pg.DemeReward('a')])) == \
        pytest.approx(flat, rel=1e-12)
    assert mean(pg.CombinedReward([pg.CombinedReward([tbl, pg.DemeReward('a')]), pg.DemeReward('b')])) == \
        pytest.approx(0, abs=1e-12)


def test_custom_rewards_differing_in_support_are_distinct():
    """The support predicate of a custom reward decides its state space, so it enters equality. Regression: the same
    function with two supports shared cached moments, and the block-counting variant was served 3.6667 instead of
    2.0."""
    def bin_one(state_space):
        return pg.UnfoldedSFSReward(1)._get(state_space) if isinstance(state_space, pg.BlockCountingStateSpace) \
            else pg.TotalBranchLengthReward()._get(state_space)

    coal = pg.Coalescent(n=4)
    lineage = pg.CustomReward(bin_one, supports=lambda s: s is pg.LineageCountingStateSpace)
    block = pg.CustomReward(bin_one, supports=lambda s: s is pg.BlockCountingStateSpace)

    coal.moment(1, [lineage], center=False)

    assert coal.moment(1, [block], center=False) == pytest.approx(coal.sfs.mean.data[1], rel=1e-12)


def test_rewards_no_state_space_supports_raise_value_error():
    """Rewards that no state space of a two-locus coalescent supports raise the documented ValueError. Regression:
    the fallback to the single-locus block-counting space raised NotImplementedError."""
    coal = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1))

    with pytest.raises(ValueError, match='not jointly compatible'):
        coal._select_state_space([pg.TwoLocusSFSReward(1, 1), pg.UnfoldedSFSReward(1)])


def test_lineage_reward_is_restricted_to_one_locus():
    """LineageReward counts the lineages of one locus. Regression: on two loci it summed the lineage counts over the
    loci, so LineageReward(2) always raised the absorbing-state error and other counts gave no coalescence time. On one
    locus, LineageReward(2) is the time with two lineages, 1 for the standard coalescent."""
    for n in (2, 4):
        assert pg.Coalescent(n=n).moment(1, [pg.LineageReward(2)], center=False) == pytest.approx(1, rel=1e-12)

    coal = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1))

    for k in (2, 3):
        with pytest.raises(ValueError, match='single locus, but the coalescent has 2 loci'):
            coal.moment(1, [pg.LineageReward(k)], center=False)


def test_deme_and_locus_rewards_name_an_unknown_deme_or_locus():
    """DemeReward and LocusReward raise a ValueError naming an unknown deme or locus. Regression: DemeReward leaked the
    bare ValueError of list.index, LocusReward an IndexError, and LocusReward(-1) wrapped round to the last locus."""
    demes = pg.Coalescent(n={'a': 2, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}
    ))

    with pytest.raises(ValueError, match='Population c does not exist'):
        demes.moment(1, [pg.DemeReward('c')], center=False)

    loci = pg.Coalescent(n=2, loci=pg.LocusConfig(n=2, recombination_rate=1))

    for locus in (2, -1):
        with pytest.raises(ValueError, match=f'Locus {locus} does not exist'):
            loci.moment(1, [pg.LocusReward(locus)], center=False)


def test_distribution_rejects_a_sequence_of_rewards():
    """Coalescent.distribution, Coalescent.joint_distribution and their PhaseTypeDistribution counterparts take single
    rewards and raise a TypeError naming the argument for a list. Regression: the list reached the state-space
    selection and raised an AttributeError, the PhaseTypeDistribution path accepted it and raised an AttributeError
    at the first evaluation, and the message of the Coalescent path named a tuple for a list."""
    coal = pg.Coalescent(n=3)

    for dist in (coal, coal.total_branch_length):
        with pytest.raises(TypeError, match='reward must be a single Reward, but got a sequence'):
            dist.distribution([pg.TreeHeightReward()])

        with pytest.raises(TypeError, match='reward_b must be a single Reward, but got a sequence'):
            dist.joint_distribution(pg.TreeHeightReward(), [pg.TotalBranchLengthReward()])

    with pytest.raises(TypeError, match='reward must be a single Reward, but got str'):
        coal.total_branch_length.distribution('tree_height')


@pytest.mark.parametrize("inner", [
    [pg.DemeReward('a')],
    [pg.DemeReward('a'), pg.rewards.UnitReward()],
])
def test_a_nested_combined_reward_restricts_as_its_members_do(inner):
    """A combined reward member contributes its members, so nesting leaves the reward and its hash unchanged.
    Regression: a restricting combined reward was taken as a factor splitting by lineage counts, giving an SFS bin
    1 mean of 1.9215 against 1.5996 for deme a."""
    coal = pg.Coalescent(
        n={'a': 2, 'b': 2},
        demography=pg.Demography(pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 1, ('b', 'a'): 0.5})
    )
    sfs = pg.UnfoldedSFSReward(1)
    nested = pg.CombinedReward([pg.CombinedReward(inner), sfs])
    flat = pg.CombinedReward(inner + [sfs])

    assert hash(nested) == hash(flat)
    assert coal.moment(1, [nested], center=False) == pytest.approx(coal.moment(1, [flat], center=False), rel=1e-12)


@pytest.mark.parametrize("make", [
    lambda: pg.UnfoldedSFSReward(1.5),
    lambda: pg.FoldedSFSReward(0.5),
    lambda: pg.LineageReward(2.7),
    lambda: pg.LocusReward(-0.5),
    lambda: pg.rewards.TwoLocusSFSReward(0, 1.7),
    lambda: pg.RestrictedReward(pg.TreeHeightReward(), locus=0.5),
    lambda: pg.JointSFSReward((1.5, 0)),
    lambda: pg.UnfoldedSFSReward(True),
])
def test_a_non_integral_reward_index_raises(make):
    """Reward indices, lineage counts, loci and frequency classes must be integers. Regression: they were truncated
    by int(), selecting another bin, lineage count or locus, and LocusReward(-0.5) passed the existence check as
    locus 0."""
    with pytest.raises(ValueError, match='must be an integer'):
        make()


def test_integral_reward_indices_of_any_numeric_type_are_accepted():
    """Integral values of any numeric type select the same bin as the integer."""
    assert pg.UnfoldedSFSReward(np.int64(2)).index == pg.UnfoldedSFSReward(2.0).index == 2
    assert pg.JointSFSReward((np.float64(1), 0)).config == (1, 0)


@pytest.mark.parametrize("config", [(3, 0), (1, 0, 0), (0, 0), (-1, 1)])
def test_a_joint_sfs_bin_outside_the_sample_raises_a_value_error(config):
    """A descendant vector that is not one of the sample raises a ValueError naming the admissible range. Regression:
    it raised a bare KeyError."""
    coal = pg.Coalescent(
        n={'a': 2, 'b': 1},
        demography=pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1})
    )

    with pytest.raises(ValueError, match='descendant configuration must hold one integer count per population'):
        _ = coal.jsfs.bin(*config).mean


def test_a_non_integral_joint_sfs_bin_raises():
    """jsfs.bin rejects a non-integral count. Regression: it truncated the count."""
    coal = pg.Coalescent(
        n={'a': 2, 'b': 1},
        demography=pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1})
    )

    with pytest.raises(ValueError, match='descendant configuration must hold one integer count per population'):
        coal.jsfs.bin(1.5, 0)


@pytest.mark.parametrize("cls", [pg.ProductReward, pg.SumReward, pg.CombinedReward])
def test_an_empty_composite_reward_raises(cls):
    """A composite reward needs a member. Regression: an empty one raised a bare IndexError when accumulated."""
    with pytest.raises(ValueError, match='needs at least one reward'):
        cls([])


@pytest.mark.parametrize("func", [lambda s: np.ones(s.k - 1), lambda s: None, lambda s: np.ones((s.k, 2))])
def test_a_custom_reward_of_the_wrong_shape_raises(func):
    """A custom reward must give one entry per state. Regression: a wrong length raised a bare numpy IndexError."""
    coal = pg.Coalescent(n=4)

    with pytest.raises(ValueError, match='must be a vector of length'):
        coal.moment(1, [pg.CustomReward(func)], center=False)



@pytest.mark.parametrize("reward", [
    pg.TwoLocusSFSReward(0, 1),
    pg.ProductReward([pg.LocusReward(0), pg.UnfoldedSFSReward(1)]),
    pg.CustomReward(lambda s: np.ones(s.k), supports=lambda s: s is pg.TwoLocusBlockCountingStateSpace)
])
def test_rewards_without_a_single_locus_space_are_rejected(reward):
    """
    On one locus the block-counting space was chosen without checking that the rewards support it, so a reward that
    supports no single-locus space failed inside the reward with a NotImplementedError, or was evaluated on a space it
    excludes, instead of raising the documented ValueError.
    """
    with pytest.raises(ValueError, match="not jointly compatible"):
        pg.Coalescent(n=3).moment(1, rewards=[reward])


def test_rewards_evaluated_on_the_single_locus_block_counting_space_route_there():
    """LineageReward, a restriction to the single locus and StateReward are evaluated on the block-counting space."""
    coal = pg.Coalescent(n=4)
    sfs = pg.UnfoldedSFSReward(1)
    expected = coal.moment(1, rewards=[sfs])

    assert coal.moment(1, rewards=[pg.RestrictedReward(sfs, locus=0)]) == pytest.approx(expected, rel=1e-12)
    assert coal.moment(1, rewards=[pg.CombinedReward([pg.LocusReward(0), sfs])]) == pytest.approx(expected, rel=1e-12)
    assert np.isfinite(coal.moment(2, rewards=[pg.LineageReward(2), sfs]))

    transient = np.where(~coal.block_counting_state_space.absorbing)[0]
    assert coal.moment(1, rewards=[pg.StateReward(int(transient[0]))]) > 0


def test_locus_trivial_rewards_evaluate_on_the_joint_state_space():
    """On a single locus, the tree height restricted to locus 0 and the total tree height equal the tree height. They
    also evaluate on the joint block-counting space, which a joint-SFS reward requires. Regression: they raised
    ValueError there."""
    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 1, ('b', 'a'): 0.5}))
    jsfs = pg.JointSFSReward((1, 0))

    expected = coal.moment(2, rewards=(jsfs, pg.TreeHeightReward()))

    for reward in [pg.RestrictedReward(pg.TreeHeightReward(), locus=0),
                   pg.CombinedReward([pg.LocusReward(0), pg.TreeHeightReward()]),
                   pg.TotalTreeHeightReward()]:
        assert coal.moment(2, rewards=(jsfs, reward)) == pytest.approx(expected, rel=1e-10)


def test_state_reward_validates_its_index():
    """Regression: a non-integer, negative or out-of-range state gave an all-zero reward and a moment of zero."""
    assert pg.StateReward(1.0).state == 1
    assert hash(pg.StateReward(1.0)) == hash(pg.StateReward(1))

    for state in [1.5, -1]:
        with pytest.raises(ValueError):
            pg.StateReward(state)

    with pytest.raises(ValueError, match='does not exist'):
        pg.Coalescent(n=4).moment(1, rewards=[pg.StateReward(1000)])
