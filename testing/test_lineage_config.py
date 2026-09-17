"""
Test LineageConfig class.
"""

from testing import TestCase

import numpy as np
import pytest
from numpy import testing

import phasegen as pg


class LineageConfigTestCase(TestCase):
    """
    Test LineageConfig class.
    """

    def test_lineage_config_from_list(self):
        """
        Test LineageConfig from list.
        """
        p = pg.LineageConfig([1, 2, 3])

        self.assertDictEqual(p.lineage_dict, {'pop_0': 1, 'pop_1': 2, 'pop_2': 3})

    def test_lineage_config_from_dict(self):
        """
        Test LineageConfig from dict.
        """
        p = pg.LineageConfig({'b': 1, 'a': 2, 'c': 3})

        self.assertDictEqual(p.lineage_dict, {'a': 2, 'b': 1, 'c': 3})

    def test_lineage_config_from_scalar(self):
        """
        Test LineageConfig from scalar.
        """
        p = pg.LineageConfig(3)

        self.assertDictEqual(p.lineage_dict, {'pop_0': 3})

    def test_equality(self):
        """
        Test equality.
        """
        self.assertEqual(pg.LineageConfig(3), pg.LineageConfig([3]))
        self.assertEqual(pg.LineageConfig(3), pg.LineageConfig({'pop_0': 3}))
        self.assertEqual(pg.LineageConfig([3, 2]), pg.LineageConfig({'pop_0': 3, 'pop_1': 2}))

        self.assertNotEqual(pg.LineageConfig(3), pg.LineageConfig(4))
        self.assertNotEqual(pg.LineageConfig(3), pg.LineageConfig([3, 3]))
        self.assertNotEqual(pg.LineageConfig(3), pg.LineageConfig({'pop_1': 3}))
        self.assertNotEqual(pg.LineageConfig({'pop_0': 3, 'pop_1': 2}), pg.LineageConfig({'pop_0': 3, 'pop1': 3}))


@pytest.mark.parametrize("n", [{'pop_0': 3, 'pop_1': -1}, [2.7, 0.9], 2.5])
def test_negative_or_non_integral_lineage_counts_raise(n):
    """
    Per-population counts were cast with ``int`` and only their sum was checked, so a negative count passed whenever
    the total was at least 2 and failed later with a misleading NaN error, and ``[2.7, 0.9]`` was silently truncated
    to ``[2, 0]``.
    """
    with pytest.raises(ValueError, match="non-negative integers"):
        pg.LineageConfig(n)


def test_integral_float_and_numpy_lineage_counts_accepted():
    """
    Integral floats, as passed from R, and numpy integers remain valid lineage counts.
    """
    assert pg.LineageConfig([2.0, 1.0]) == pg.LineageConfig([2, 1])
    assert pg.LineageConfig(np.array([2, 1])) == pg.LineageConfig([2, 1])


def test_non_numeric_lineage_count_raises_type_error():
    """
    A non-numeric lineage count raises a TypeError.
    """
    with pytest.raises(TypeError):
        pg.LineageConfig({'pop_0': None, 'pop_1': 2})
