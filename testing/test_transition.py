from unittest import TestCase

import numpy as np

import phasegen as pg
from phasegen.transition import Transition, State


class TransitionTestCase(TestCase):
    """
    Test Transition class.
    """

    def test_simple_coalescence_n_2(self):
        """
        Test simple coalescence for n = 2.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2]]]),
            marginal2=np.array([[[1]]]),
            shared1=np.array([[[0]]]),
            shared2=np.array([[[0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_eligible_coalescence)
        self.assertTrue(t.is_coalescence)
        self.assertTrue(t.is_unshared_coalescence)

        self.assertFalse(t.is_eligible_migration)
        self.assertFalse(t.is_migration)

        self.assertFalse(t.is_recombination)

        self.assertEqual(1, t.get_rate())

    def test_block_coalescence_n_5(self):
        """
        Test block coalescence for n = 5.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=5)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2, 0, 1, 0, 0]]]),
            marginal2=np.array([[[1, 0, 0, 1, 0]]]),
            shared1=np.array([[[0, 0, 0, 0, 0]]]),
            shared2=np.array([[[0, 0, 0, 0, 0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_eligible_coalescence)
        self.assertTrue(t.is_coalescence)
        self.assertTrue(t.is_unshared_coalescence)

        self.assertFalse(t.is_eligible_migration)

        self.assertEqual(2, t.get_rate())

    def test_shared_coalescence_two_loci_default_state_space(self):
        """
        Test shared coalescence for two loci, default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[3]], [[3]]]),
            shared1=np.array([[[2]], [[2]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_shared_coalescence)

        self.assertFalse(t.is_unshared_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_shared_coalescence_two_loci_has_insufficient_shared_lineages(self):
        """
        Test whether we detect insufficient shared lineages for shared coalescence.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[3]], [[3]]]),
            shared1=np.array([[[1]], [[1]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertFalse(t.has_sufficient_shared_lineages_shared_coalescence)

    def test_shared_coalescence_two_loci_invalid_lineage_reduction(self):
        """
        Test whether we detect invalid lineage reduction for shared coalescence.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[3]], [[3]]]),
            shared1=np.array([[[3]], [[3]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertFalse(t.is_valid_lineage_reduction_unshared_coalescence)

    def test_migration_two_demes_default_state_space(self):
        """
        Test migration for two demes, default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=2),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 2},
                migration_rates={('pop_0', 'pop_1'): 1.11}
            )
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2], [0]]]),
            marginal2=np.array([[[1], [1]]]),
            shared1=np.array([[[0], [0]]]),
            shared2=np.array([[[0], [0]]])
        )

        self.assertEqual(2.22, t.get_rate())

    def test_migration_more_than_one_lineage_invalid(self):
        """
        Test whether we detect invalid migration for more than one lineage.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=2),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 2},
                migration_rates={('pop_0', 'pop_1'): 1.11}
            )
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2], [0]]]),
            marginal2=np.array([[[0], [2]]]),
            shared1=np.array([[[0], [0]]]),
            shared2=np.array([[[0], [0]]])
        )

        self.assertFalse(t.is_one_migration_event)

    def test_recombination_two_loci_default_state_space(self):
        """
        Test recombination for two loci, default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[3]]]),
            shared1=np.array([[[2]], [[2]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_recombination)

        self.assertFalse(t.is_locus_coalescence)

        self.assertEqual(2.22, t.get_rate())

    def test_locus_coalescence_two_loci_default_state_space(self):
        """
        Test locus coalescence for two loci, default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[3]]]),
            shared1=np.array([[[1]], [[1]]]),
            shared2=np.array([[[2]], [[2]]])
        )

        self.assertTrue(t.is_locus_coalescence)

        self.assertFalse(t.is_recombination)

        self.assertEqual(4, t.get_rate())

    def test_unshared_coalescence_two_loci_default_state_space(self):
        """
        Test unshared coalescence for two loci, default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[4]], [[4]]]),
            marginal2=np.array([[[4]], [[3]]]),
            shared1=np.array([[[3]], [[3]]]),
            shared2=np.array([[[3]], [[3]]])
        )

        t.get_rate()

    def test_is_not_shared_coalescence_reverse_coalescence(self):
        """
        Test whether we detect non-shared coalescence.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=2)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[0, 1]], [[0, 1]]]),
            marginal2=np.array([[[2, 0]], [[2, 0]]]),
            shared1=np.array([[[0, 1]], [[0, 1]]]),
            shared2=np.array([[[2, 0]], [[2, 0]]])
        )

        self.assertFalse(t.is_shared_coalescence)

    def test_state_is_absorbing(self):
        """
        Test whether we detect absorbing states.
        """
        # default state space
        self.assertTrue(State.is_absorbing(np.array([[[1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[2]]])))

        # default state space, two demes
        self.assertTrue(State.is_absorbing(np.array([[[1], [0]]])))
        self.assertFalse(State.is_absorbing(np.array([[[1], [1]]])))

        # default state space, two loci
        self.assertTrue(State.is_absorbing(np.array([[[1]], [[1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[1]], [[2]]])))  # both loci must be absorbing

        # block counting state space
        self.assertTrue(State.is_absorbing(np.array([[[0, 1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[1, 0]]])))

        # block counting state space, two demes
        self.assertTrue(State.is_absorbing(np.array([[[0, 1], [0, 0]]])))
        self.assertFalse(State.is_absorbing(np.array([[[0, 1], [0, 1]]])))
        self.assertFalse(State.is_absorbing(np.array([[[0, 1], [1, 0]]])))

    def test_unshared_coalescence_with_shared_lineages_default_state_space_two_loci(self):
        """
        Test unshared coalescence with shared lineages for default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[1]], [[2]]]),
            marginal2=np.array([[[1]], [[1]]]),
            shared1=np.array([[[1]], [[1]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_unshared_coalescence)
        self.assertTrue(t.is_mixed_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_mixed_coalescence_block_counting_state_space_two_loci_n_2(self):
        """
        Test mixed coalescence with for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[0, 1]], [[2, 0]]]),
            marginal2=np.array([[[0, 1]], [[0, 1]]]),
            shared1=np.array([[[0, 1]], [[1, 0]]]),
            shared2=np.array([[[0, 1]], [[0, 1]]])
        )

        self.assertTrue(t.is_mixed_coalescence)

    def test_mixed_coalescence_default_state_space_two_loci_n_3(self):
        """
        Test mixed coalescence for default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here one of the coalescing lineages has to be shared
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[2]]]),
            shared1=np.array([[[1]], [[1]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_mixed_coalescence)
        self.assertTrue(t.is_unshared_coalescence)

        self.assertEqual(3, t.get_rate())

    def test_mixed_coalescence_block_counting_state_space_two_loci_n_3(self):
        """
        Test unshared coalescence with shared lineages for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here one of the coalescing lineages has to be shared
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3, 0, 0]], [[3, 0, 0]]]),
            marginal2=np.array([[[3, 0, 0]], [[1, 1, 0]]]),
            shared1=np.array([[[1, 0, 0]], [[1, 0, 0]]]),
            shared2=np.array([[[1, 0, 0]], [[0, 1, 0]]])
        )

        self.assertTrue(t.is_mixed_coalescence)
        self.assertFalse(t.is_unshared_coalescence)

        self.assertEqual(2, t.get_rate())

    def test_unshared_coalescence_block_counting_state_space_two_loci_n_3(self):
        """
        Test unshared coalescence with shared lineages for block counting state space.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unshared
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3, 0, 0]], [[3, 0, 0]]]),
            marginal2=np.array([[[3, 0, 0]], [[1, 1, 0]]]),
            shared1=np.array([[[1, 0, 0]], [[1, 0, 0]]]),
            shared2=np.array([[[1, 0, 0]], [[1, 0, 0]]])
        )

        self.assertTrue(t.is_unshared_coalescence)
        self.assertFalse(t.is_mixed_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_unshared_coalescence_default_state_space_two_loci_n_3(self):
        """
        Test unshared coalescence with shared lineages for default state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        # here both of the coalescing lineages have to be unshared
        t = Transition(
            state_space=s,
            marginal1=np.array([[[3]], [[3]]]),
            marginal2=np.array([[[3]], [[2]]]),
            shared1=np.array([[[1]], [[1]]]),
            shared2=np.array([[[1]], [[1]]])
        )

        self.assertTrue(t.is_unshared_coalescence)
        self.assertTrue(t.is_mixed_coalescence)

        self.assertEqual(3, t.get_rate())

    def test_shared_coalescence_two_loci_n_3_same_rate_across_loci(self):
        """
        Test shared coalescence for equal lineage blocks when the coalescence rate is the same across loci.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[1, 1, 0]], [[3, 0, 0]]]),
            marginal2=np.array([[[0, 0, 1]], [[1, 1, 0]]]),
            shared1=np.array([[[1, 1, 0]], [[2, 0, 0]]]),
            shared2=np.array([[[0, 0, 1]], [[0, 1, 0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_shared_coalescence)

        self.assertEqual(1, t.get_rate())

    def test_shared_coalescence_two_loci_n_4_different_rate_across_loci(self):
        """
        Test shared coalescence for unequal lineage blocks when the coalescence rate is different across loci.
        TODO what to do when rates are different across loci? Select the minimum?
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2, 1, 0]], [[4, 0, 0]]]),
            marginal2=np.array([[[1, 0, 1]], [[2, 1, 0]]]),
            shared1=np.array([[[2, 1, 0]], [[3, 0, 0]]]),
            shared2=np.array([[[1, 0, 1]], [[1, 1, 0]]])
        )

        self.assertTrue(t.is_eligible)
        self.assertTrue(t.is_shared_coalescence)

        self.assertEqual(2, t.get_rate())

    def test_mixed_coalescence_only_possible_if_only_one_locus_changes(self):
        """
        Test whether we detect mixed coalescence when only one locus changes.
        """
        # default state space
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2, 0]], [[0, 1]]]),
            marginal2=np.array([[[0, 1]], [[0, 1]]]),
            shared1=np.array([[[1, 0]], [[0, 1]]]),
            shared2=np.array([[[0, 0]], [[0, 0]]])
        )

        self.assertFalse(t.is_mixed_coalescence)

    def test_bug_default_state_space_two_loci_n_2(self):
        """
        Test mixed coalescence with for block counting state space.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        t = Transition(
            state_space=s,
            marginal1=np.array([[[2]], [[2]]]),
            marginal2=np.array([[[2]], [[1]]]),
            shared1=np.array([[[1]], [[1]]]),
            shared2=np.array([[[0]], [[0]]])
        )

        self.assertFalse(t.is_mixed_coalescence)

        self.assertEqual(0, t.get_rate())