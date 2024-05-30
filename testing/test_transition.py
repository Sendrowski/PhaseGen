"""
Test Transition class.
"""

from unittest import TestCase

import numpy as np
import pytest

import phasegen as pg
from phasegen.state_space_old import Transition, State


class TransitionTestCase(TestCase):
    """
    Test Transition class.
    """

    def test_simple_coalescence_n_2(self):
        """
        Test simple coalescence for n = 2.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2]]]),
            marginal2=np.array([[[1]]]),
            linked1=np.array([[[0]]]),
            linked2=np.array([[[0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_eligible_coalescence)
        self.assertTrue(t.is_coalescence)
        self.assertTrue(t.is_unlinked_coalescence)

        self.assertFalse(t.is_eligible_migration)
        self.assertFalse(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

        self.assertFalse(t.is_recombination)

        self.assertEqual(1, t.get_rate())

    def test_block_coalescence_n_5(self):
        """
        Test block coalescence for n = 5.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=5)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2, 0, 1, 0, 0]]]),
            marginal2=np.array([[[1, 0, 0, 1, 0]]]),
            linked1=np.array([[[0, 0, 0, 0, 0]]]),
            linked2=np.array([[[0, 0, 0, 0, 0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_eligible_coalescence)
        self.assertTrue(t.is_coalescence)
        self.assertTrue(t.is_unlinked_coalescence)

        self.assertFalse(t.is_eligible_migration)

        self.assertEqual(2, t.get_rate())

    def test_linked_coalescence_two_loci_lineage_counting_state_space(self):
        """
        Test linked coalescence for two loci, lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[3]], [[3]]]),
            linked1=np.array([[[2]], [[2]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_linked_coalescence)

        self.assertFalse(t.is_unlinked_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_linked_coalescence_two_loci_has_insufficient_linked_lineages(self):
        """
        Test whether we detect insufficient linked lineages for linked coalescence.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[3]], [[3]]]),
            linked1=np.array([[[1]], [[1]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertFalse(t.has_sufficient_linked_lineages_linked_coalescence)

    def test_linked_coalescence_two_loci_invalid_lineage_reduction(self):
        """
        Test whether we detect invalid lineage reduction for linked coalescence.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[3]], [[3]]]),
            linked1=np.array([[[3]], [[3]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertFalse(t.is_valid_lineage_reduction_unlinked_coalescence)

    def test_migration_two_demes_lineage_counting_state_space(self):
        """
        Test migration for two demes, lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 1, 'pop_1': 1}),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 2},
                migration_rates={('pop_0', 'pop_1'): 1.11}
            )
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2], [0]]]),
            marginal2=np.array([[[1], [1]]]),
            linked1=np.array([[[0], [0]]]),
            linked2=np.array([[[0], [0]]])
        )

        self.assertEqual(2.22, t.get_rate())

    def test_migration_more_than_one_lineage_invalid(self):
        """
        Test whether we detect invalid migration for more than one lineage.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 1, 'pop_1': 1}),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 2},
                migration_rates={('pop_0', 'pop_1'): 1.11}
            )
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2], [0]]]),
            marginal2=np.array([[[0], [2]]]),
            linked1=np.array([[[0], [0]]]),
            linked2=np.array([[[0], [0]]])
        )

        self.assertFalse(t.is_one_migration_event)

    def test_recombination_two_loci_lineage_counting_state_space(self):
        """
        Test recombination for two loci, lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[3]]]),
            linked1=np.array([[[2]], [[2]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_recombination)

        self.assertFalse(t.is_locus_coalescence)

        self.assertEqual(2.22, t.get_rate())

    def test_locus_coalescence_two_loci_lineage_counting_state_space(self):
        """
        Test locus coalescence for two loci, lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[3]]]),
            linked1=np.array([[[1]], [[1]]]),
            linked2=np.array([[[2]], [[2]]])
        )

        self.assertTrue(t.is_locus_coalescence)

        self.assertFalse(t.is_recombination)

        self.assertEqual(4, t.get_rate())

    def test_unlinked_coalescence_two_loci_lineage_counting_state_space(self):
        """
        Test unlinked coalescence for two loci, lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[4]], [[3]]]),
            linked1=np.array([[[3]], [[3]]]),
            linked2=np.array([[[3]], [[3]]])
        )

        t.get_rate()

    def test_is_not_linked_coalescence_reverse_coalescence(self):
        """
        Test whether we detect non-linked coalescence.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[0, 1]], [[0, 1]]]),
            marginal2=np.array([[[2, 0]], [[2, 0]]]),
            linked1=np.array([[[0, 1]], [[0, 1]]]),
            linked2=np.array([[[2, 0]], [[2, 0]]])
        )

        self.assertFalse(t.is_linked_coalescence)

    def test_state_is_absorbing(self):
        """
        Test whether we detect absorbing states.
        """
        # lineage-counting state space
        self.assertTrue(State.is_absorbing(np.array([[[1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[2]]])))

        # lineage-counting state space, two demes
        self.assertTrue(State.is_absorbing(np.array([[[1], [0]]])))
        self.assertFalse(State.is_absorbing(np.array([[[1], [1]]])))

        # lineage-counting state space, two loci
        self.assertTrue(State.is_absorbing(np.array([[[1]], [[1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[1]], [[2]]])))  # both loci must be absorbing

        # block-counting state space
        self.assertTrue(State.is_absorbing(np.array([[[0, 1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[1, 0]]])))

        # block-counting state space, two demes
        self.assertTrue(State.is_absorbing(np.array([[[0, 1], [0, 0]]])))
        self.assertFalse(State.is_absorbing(np.array([[[0, 1], [0, 1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[0, 1], [1, 0]]])))

    def test_mixed_coalescence_with_linked_lineages_lineage_counting_state_space_two_loci(self):
        """
        Test unlinked coalescence with linked lineages for lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[1]], [[2]]]),
            marginal2=np.array([[[1]], [[1]]]),
            linked1=np.array([[[1]], [[1]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertFalse(t.is_unlinked_coalescence)
        self.assertTrue(t.is_mixed_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_mixed_coalescence_no_linked_lineages_lineage_counting_state_space_two_loci(self):
        """
        Test unlinked coalescence with linked lineages for lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[1]], [[2]]]),
            marginal2=np.array([[[1]], [[1]]]),
            linked1=np.array([[[0]], [[0]]]),
            linked2=np.array([[[0]], [[0]]])
        )

        self.assertFalse(t.is_mixed_coalescence)
        self.assertTrue(t.is_unlinked_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_mixed_coalescent_no_unlinked_lineages_lineage_counting_state_space_two_loci(self):
        """
        Test unlinked coalescence with linked lineages for lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[1]], [[2]]]),
            marginal2=np.array([[[1]], [[1]]]),
            linked1=np.array([[[1]], [[2]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertFalse(t.is_mixed_coalescence)
        self.assertFalse(t.is_unlinked_coalescence)

        self.assertEqual(0, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_mixed_coalescence_block_counting_state_space_two_loci_n_2(self):
        """
        Test mixed coalescence for block-counting state space.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[0, 1]], [[2, 0]]]),
            marginal2=np.array([[[0, 1]], [[0, 1]]]),
            linked1=np.array([[[0, 1]], [[1, 0]]]),
            linked2=np.array([[[0, 1]], [[0, 1]]])
        )

        self.assertTrue(t.is_mixed_coalescence)

    def test_mixed_coalescence_lineage_counting_state_space_two_loci_n_3(self):
        """
        Test mixed coalescence for lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here one of the coalescing lineages has to be linked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[2]]]),
            linked1=np.array([[[1]], [[1]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_mixed_coalescence)
        self.assertTrue(t.is_unlinked_coalescence)

        self.assertEqual(3, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_mixed_coalescence_block_counting_state_space_two_loci_n_3(self):
        """
        Test unlinked coalescence with linked lineages for block-counting state space.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here one of the coalescing lineages has to be linked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3, 0, 0]], [[3, 0, 0]]]),
            marginal2=np.array([[[3, 0, 0]], [[1, 1, 0]]]),
            linked1=np.array([[[1, 0, 0]], [[1, 0, 0]]]),
            linked2=np.array([[[1, 0, 0]], [[0, 1, 0]]])
        )

        self.assertTrue(t.is_mixed_coalescence)
        self.assertFalse(t.is_unlinked_coalescence)

        self.assertEqual(2, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_unlinked_coalescence_block_counting_state_space_two_loci_n_3(self):
        """
        Test unlinked coalescence with linked lineages for block-counting state space.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unlinked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3, 0, 0]], [[3, 0, 0]]]),
            marginal2=np.array([[[3, 0, 0]], [[1, 1, 0]]]),
            linked1=np.array([[[1, 0, 0]], [[1, 0, 0]]]),
            linked2=np.array([[[1, 0, 0]], [[1, 0, 0]]])
        )

        self.assertTrue(t.is_unlinked_coalescence)
        self.assertFalse(t.is_mixed_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_unlinked_coalescence_lineage_counting_state_space_two_loci_n_3(self):
        """
        Test unlinked coalescence with linked lineages for lineage-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unlinked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[2]]]),
            linked1=np.array([[[1]], [[1]]]),
            linked2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_unlinked_coalescence)
        self.assertTrue(t.is_mixed_coalescence)

        self.assertEqual(3, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_linked_coalescence_two_loci_n_3_same_rate_across_loci(self):
        """
        Test linked coalescence for equal lineage blocks when the coalescence rate is the same across loci.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[1, 1, 0]], [[3, 0, 0]]]),
            marginal2=np.array([[[0, 0, 1]], [[1, 1, 0]]]),
            linked1=np.array([[[1, 1, 0]], [[2, 0, 0]]]),
            linked2=np.array([[[0, 0, 1]], [[0, 1, 0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_linked_coalescence)

        self.assertEqual(1, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_linked_coalescence_two_loci_n_4_different_rate_across_loci(self):
        """
        Test linked coalescence for unequal lineage blocks when the coalescence rate is different across loci.
        What to do when rates are different across loci? Select the minimum?
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2, 1, 0]], [[4, 0, 0]]]),
            marginal2=np.array([[[1, 0, 1]], [[2, 1, 0]]]),
            linked1=np.array([[[2, 1, 0]], [[3, 0, 0]]]),
            linked2=np.array([[[1, 0, 1]], [[1, 1, 0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_linked_coalescence)

        self.assertEqual(2, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_mixed_coalescence_only_possible_if_only_one_locus_changes(self):
        """
        Test whether we detect mixed coalescence when only one locus changes.
        """
        # lineage-counting state space
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2, 0]], [[0, 1]]]),
            marginal2=np.array([[[0, 1]], [[0, 1]]]),
            linked1=np.array([[[1, 0]], [[0, 1]]]),
            linked2=np.array([[[0, 0]], [[0, 0]]])
        )

        self.assertFalse(t.is_mixed_coalescence)

    def test_bug_lineage_counting_state_space_two_loci_n_2(self):
        """
        Test mixed coalescence for block-counting state space.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2]], [[2]]]),
            marginal2=np.array([[[2]], [[1]]]),
            linked1=np.array([[[1]], [[1]]]),
            linked2=np.array([[[0]], [[0]]])
        )

        self.assertFalse(t.is_mixed_coalescence)

        self.assertEqual(0, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_mixed_coalescence_block_counting_state_space_2_loci_n_3(self):
        """
        Test mixed coalescence for block-counting state space.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[3, 0, 0]], [[1, 1, 0]]]),
            marginal2=np.array([[[3, 0, 0]], [[0, 0, 1]]]),
            linked1=np.array([[[1, 0, 0]], [[0, 1, 0]]]),
            linked2=np.array([[[1, 0, 0]], [[0, 0, 1]]])
        )

        self.assertTrue(t.is_mixed_coalescence)

        self.assertEqual(1, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_invalid_mixed_coalescence_linked_lineage_in_wrong_place_2_loci_n_4(self):
        """
        Test whether we detect invalid mixed coalescence.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unlinked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[4, 0, 0, 0]], [[2, 1, 0, 0]]]),
            marginal2=np.array([[[4, 0, 0, 0]], [[1, 0, 1, 0]]]),
            linked1=np.array([[[1, 0, 0, 0]], [[0, 1, 0, 0]]]),
            linked2=np.array([[[1, 0, 0, 0]], [[1, 0, 0, 0]]])
        )

        self.assertFalse(t.is_mixed_coalescence)

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_invalid_coalescence_too_many_linked_lineages_2_loci_n_4(self):
        """
        Test whether we detect invalid mixed coalescence.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unlinked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[4, 0, 0, 0]], [[2, 1, 0, 0]]]),
            marginal2=np.array([[[4, 0, 0, 0]], [[0, 2, 0, 0]]]),
            linked1=np.array([[[2, 0, 0, 0]], [[2, 0, 0, 0]]]),
            linked2=np.array([[[2, 0, 0, 0]], [[0, 2, 0, 0]]])
        )

        self.assertFalse(t.is_linked_coalescence)
        self.assertFalse(t.is_unlinked_coalescence)
        self.assertFalse(t.is_mixed_coalescence)

        self.assertEqual(0, t.get_rate())

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_valid_mixed_coalescence_linked_lineage_in_right_place_2_loci_n_4(self):
        """
        Test whether we detect valid mixed coalescence.
        """
        s = pg.state_space_old.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unlinked
        t = Transition(
            state_space=s,
            marginal1=np.array([[[4, 0, 0, 0]], [[2, 1, 0, 0]]]),
            marginal2=np.array([[[4, 0, 0, 0]], [[1, 0, 1, 0]]]),
            linked1=np.array([[[1, 0, 0, 0]], [[0, 1, 0, 0]]]),
            linked2=np.array([[[1, 0, 0, 0]], [[0, 0, 1, 0]]])
        )

        self.assertTrue(t.is_mixed_coalescence)

    def test_multiple_merger_lineage_counting_state_space_n_4(self):
        """
        Test multiple merger.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            model=pg.BetaCoalescent(alpha=1.5)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]]]),
            marginal2=np.array([[[2]]]),
            linked1=np.array([[[0]]]),
            linked2=np.array([[[0]]])
        )

        self.assertTrue(t.is_coalescence)

    def get_lineage_counting_state_space_2_demes_2_loci(self):
        """
        Get lineage-counting state space for two demes, two loci.
        """
        return pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 1, 'pop_1': 1}),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 1},
                migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}
            )
        )

    def test_linked_migration_complete_linkage_one_source_lineage_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, complete linkage, one source lineage.
        """

        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[1], [1]], [[1], [1]]]),
            marginal2=np.array([[[0], [2]], [[0], [2]]]),
            linked1=np.array([[[1], [1]], [[1], [1]]]),
            linked2=np.array([[[0], [2]], [[0], [2]]])
        )

        self.assertTrue(t.is_linked_migration)
        self.assertFalse(t.is_unlinked_migration)

        self.assertEqual(t.get_rate_linked_migration(), 1)

    def test_linked_migration_complete_linkage_two_source_lineages_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, complete linkage, two source lineages.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[1], [1]], [[1], [1]]]),
            linked1=np.array([[[2], [0]], [[2], [0]]]),
            linked2=np.array([[[1], [1]], [[1], [1]]]),
        )

        self.assertTrue(t.is_linked_migration)
        self.assertFalse(t.is_unlinked_migration)

        self.assertEqual(t.get_rate_linked_migration(), 2)

    def test_linked_migration_partial_linkage_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, partial linkage.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[1], [1]], [[1], [1]]]),
            linked1=np.array([[[1], [0]], [[1], [0]]]),
            linked2=np.array([[[0], [1]], [[0], [1]]])
        )

        self.assertTrue(t.is_linked_migration)
        self.assertFalse(t.is_unlinked_migration)

        self.assertEqual(t.get_rate_linked_migration(), 1)

    def test_unlinked_migration_locus_1_no_linkage_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, no linkage, locus 1.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[1], [1]], [[2], [0]]]),
            linked1=np.array([[[0], [0]], [[0], [0]]]),
            linked2=np.array([[[0], [0]], [[0], [0]]])
        )

        self.assertTrue(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

        self.assertEqual(t.get_rate_unlinked_migration(), 2)

    def test_unlinked_migration_locus_2_no_linkage_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, no linkage, locus 2.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[2], [0]], [[1], [1]]]),
            linked1=np.array([[[0], [0]], [[0], [0]]]),
            linked2=np.array([[[0], [0]], [[0], [0]]])
        )

        self.assertTrue(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

        self.assertEqual(t.get_rate_unlinked_migration(), 2)

    def test_unlinked_migration_from_one_lineage_each_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, no linkage, locus 2.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[1], [1]]]),
            marginal2=np.array([[[2], [0]], [[0], [2]]]),
            linked1=np.array([[[0], [0]], [[0], [0]]]),
            linked2=np.array([[[0], [0]], [[0], [0]]])
        )

        self.assertTrue(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

        self.assertEqual(t.get_rate_unlinked_migration(), 1)

    def test_unlinked_migration_locus_1_partial_linkage_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, partial linkage, locus 1.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[1], [1]], [[2], [0]]]),
            linked1=np.array([[[1], [0]], [[1], [0]]]),
            linked2=np.array([[[1], [0]], [[1], [0]]])
        )

        self.assertTrue(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

        self.assertEqual(t.get_rate_unlinked_migration(), 1)

    def test_unlinked_migration_locus_2_partial_linkage_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, partial linkage, locus 2.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[2], [0]], [[1], [1]]]),
            linked1=np.array([[[1], [0]], [[1], [0]]]),
            linked2=np.array([[[1], [0]], [[1], [0]]])
        )

        self.assertTrue(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

        self.assertEqual(t.get_rate_unlinked_migration(), 1)

    def test_unlinked_migration_not_enough_lineages_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, not enough lineages.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[2], [0]], [[1], [1]]]),
            linked1=np.array([[[2], [0]], [[2], [0]]]),
            linked2=np.array([[[2], [0]], [[2], [0]]])
        )

        self.assertFalse(t.is_unlinked_migration)

    def test_invalid_linked_migration_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, invalid.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[1], [1]], [[1], [1]]]),
            marginal2=np.array([[[0], [2]], [[1], [0]]]),
            linked1=np.array([[[1], [1]], [[1], [1]]]),
            linked2=np.array([[[0], [2]], [[1], [0]]])
        )

        self.assertFalse(t.is_unlinked_migration)
        self.assertFalse(t.is_linked_migration)

    def test_invalid_coalescence_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, invalid coalescence.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[1], [1]], [[1], [1]]]),
            marginal2=np.array([[[1], [1]], [[0], [1]]]),
            linked1=np.array([[[0], [1]], [[0], [1]]]),
            linked2=np.array([[[0], [1]], [[0], [1]]])
        )

        self.assertFalse(t.is_unlinked_coalescence)
        self.assertFalse(t.is_mixed_coalescence)

    def test_linked_migration_unequal_number_of_linked_lineages_two_demes_lineage_counting_state_space(self):
        """
        Test linked migration for two demes, lineage-counting state space, unequal number of linked lineages.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[0], [2]], [[1], [1]]]),
            marginal2=np.array([[[1], [1]], [[2], [0]]]),
            linked1=np.array([[[0], [2]], [[1], [1]]]),
            linked2=np.array([[[1], [1]], [[2], [0]]])
        )

        self.assertTrue(t.is_linked_migration)

        self.assertEqual(t.get_rate_linked_migration(), 1)

    def test_recombination_only_one_eligible_lineage_in_deme_two_demes_lineage_counting_state_space(self):
        """
        Test recombination for two demes, lineage-counting state space, only one eligible lineage in deme.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[1], [1]], [[1], [1]]]),
            marginal2=np.array([[[1], [1]], [[1], [1]]]),
            linked1=np.array([[[1], [1]], [[1], [1]]]),
            linked2=np.array([[[0], [1]], [[0], [1]]])
        )

        self.assertTrue(t.is_recombination)

        self.assertEqual(t.get_rate(), 1.11)

    def test_locus_coalescence_two_possible_lineages_per_deme_n_2_lineage_counting_state_space(self):
        """
        Test locus coalescence for two possible lineages per deme, n = 2, lineage-counting state space.
        """

        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[2], [0]]]),
            marginal2=np.array([[[2], [0]], [[2], [0]]]),
            linked1=np.array([[[0], [0]], [[0], [0]]]),
            linked2=np.array([[[1], [0]], [[1], [0]]])
        )

        self.assertTrue(t.is_locus_coalescence)

        self.assertEqual(t.get_rate(), 4)

    def test_unlinked_back_migration_lineage_counting_state_space_2_loci_n_2(self):
        """
        Test unlinked back migration for lineage-counting state space, n = 2.
        """
        t = Transition(
            state_space=self.get_lineage_counting_state_space_2_demes_2_loci(),
            marginal1=np.array([[[2], [0]], [[1], [1]]]),
            marginal2=np.array([[[2], [0]], [[2], [0]]]),
            linked1=np.array([[[0], [0]], [[0], [0]]]),
            linked2=np.array([[[0], [0]], [[0], [0]]])
        )

        self.assertTrue(t.is_unlinked_migration)

        self.assertEqual(t.get_rate(), 1)
