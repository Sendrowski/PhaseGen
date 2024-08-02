"""
Test LocusConfig class.
"""

from unittest import TestCase

import numpy as np

import phasegen as pg


class LocusConfigTestCase(TestCase):
    """
    Test LocusConfig class.
    """

    def test_fewer_than_one_loci_raises_value_error(self):
        """
        Test that fewer than one loci raises a ValueError.
        """
        with self.assertRaises(ValueError):
            pg.LocusConfig(0)

    def test_more_than_two_loci_raises_not_implemented_error(self):
        """
        Test that more than two loci raises a NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            pg.LocusConfig(3)

    def test_negative_unlinked_raises_value_error(self):
        """
        Test that a negative number of unlinked lineages raises a ValueError.
        """
        with self.assertRaises(ValueError):
            pg.LocusConfig(1, -1)

    def test_negative_recombination_rate_raises_value_error(self):
        """
        Test that a negative recombination rate raises a ValueError.
        """
        with self.assertRaises(ValueError):
            pg.LocusConfig(1, 0, -1)

    def test_equality(self):
        """
        Test that the equality operator works as expected.
        """
        self.assertEqual(pg.LocusConfig(1), pg.LocusConfig(1))
        self.assertEqual(pg.LocusConfig(1, 0), pg.LocusConfig(1, 0))
        self.assertEqual(pg.LocusConfig(1, 0, 0), pg.LocusConfig(1, 0, 0))

        self.assertNotEqual(pg.LocusConfig(1), pg.LocusConfig(2))
        self.assertNotEqual(pg.LocusConfig(1, 0), pg.LocusConfig(1, 1))
        self.assertNotEqual(pg.LocusConfig(1, 0, 0), pg.LocusConfig(1, 0, 1))
        self.assertNotEqual(pg.LocusConfig(1, 0, 0), pg.LocusConfig(1, 1, 0))
        self.assertNotEqual(pg.LocusConfig(1, 0, 0), pg.LocusConfig(2, 0, 0))
        self.assertNotEqual(pg.LocusConfig(1, 0, 0), pg.LocusConfig(2, 1, 0))
        self.assertNotEqual(pg.LocusConfig(1, 0, 0), pg.LocusConfig(2, 1, 3))

    def test_get_initial_states(self):
        """
        Test that the _get_initial_states method works as expected.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(1),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        # array of ones for one locus
        np.testing.assert_array_equal(s.locus_config._get_initial_states(s), [1, 1, 1, 1])

        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(2),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        # make sure we have one state with 4 linked lineages each
        starting = np.array(s.states)[s.locus_config._get_initial_states(s).astype(bool)]
        self.assertEqual(len(starting), 1)
        np.testing.assert_array_equal(starting[0].data, ([[[4]], [[4]]], [[[4]], [[4]]]))

        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(2, n_unlinked=2),
            epoch=pg.Epoch(),
            model=pg.StandardCoalescent()
        )

        # make sure we have one state with 2 linked lineages each
        starting = np.array(s.states)[s.locus_config._get_initial_states(s).astype(bool)]
        for state in starting:
            np.testing.assert_array_equal(state.linked, [[[2]], [[2]]])
