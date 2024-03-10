"""
Test LineageConfig class.
"""

from unittest import TestCase

import numpy as np
from numpy import testing

import phasegen as pg


class LineageConfigTestCase(TestCase):
    """
    Test LineageConfig class.
    """

    def test_pop_config_from_list(self):
        """
        Test LineageConfig from list.
        """
        p = pg.LineageConfig([1, 2, 3])

        self.assertDictEqual(p.lineage_dict, {'pop_0': 1, 'pop_1': 2, 'pop_2': 3})

    def test_pop_config_from_dict(self):
        """
        Test LineageConfig from dict.
        """
        p = pg.LineageConfig({'b': 1, 'a': 2, 'c': 3})

        self.assertDictEqual(p.lineage_dict, {'a': 2, 'b': 1, 'c': 3})

    def test_pop_config_from_scalar(self):
        """
        Test LineageConfig from scalar.
        """
        p = pg.LineageConfig(3)

        self.assertDictEqual(p.lineage_dict, {'pop_0': 3})
