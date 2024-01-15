from unittest import TestCase

import numpy as np

import phasegen as pg
from phasegen.transition import Transition


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

        self.assertFalse(t.is_eligible_recombination)
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

        self.assertFalse(t.is_eligible_recombination)

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

        self.assertFalse(t.is_valid_lineage_reduction_marginal_coalescence)

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

    def test_forward_recombination_two_loci_default_state_space(self):
        """
        Test forward recombination for two loci, default state space.
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

        self.assertTrue(t.is_forward_recombination)

        self.assertFalse(t.is_backward_recombination)

        self.assertEqual(2.22, t.get_rate())

    def test_backward_recombination_two_loci_default_state_space(self):
        """
        Test backward recombination for two loci, default state space.
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

        self.assertTrue(t.is_backward_recombination)

        self.assertFalse(t.is_forward_recombination)

        self.assertEqual(4, t.get_rate())
