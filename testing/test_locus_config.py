"""
Test LocusConfig class.
"""

from unittest import TestCase

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
