from unittest import TestCase

import numpy as np
import pytest
import phasegen as pg


class CoalescentModelTestCase(TestCase):
    """
    Test coalescent models.
    """

    def test_standard_coalescent_infinite_alleles_n_2(self):
        """
        Test standard coalescent with infinite alleles for n = 2.
        """
        model = pg.StandardCoalescent()

        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([2, 0]), np.array([0, 1])), 1)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([0, 1]), np.array([2, 1])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([2, 0]), np.array([2, 0])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([2, 0]), np.array([0, 2])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([3, 0]), np.array([1, 2])), 0)

    def test_standard_coalescent_infinite_alleles_n_3(self):
        """
        Test standard coalescent with infinite alleles for n = 3.
        """
        model = pg.StandardCoalescent()

        self.assertEqual(model.get_rate_infinite_alleles(3, np.array([3, 0, 0]), np.array([1, 1, 0])), 3)
        self.assertEqual(model.get_rate_infinite_alleles(3, np.array([3, 0, 0]), np.array([0, 0, 1])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(3, np.array([3, 0, 0, 0]), np.array([1, 1, 0, 0])), 0)

    def test_standard_coalescent_infinite_alleles_n_10(self):
        """
        Test beta coalescent with infinite alleles for n = 10.
        """
        model = pg.StandardCoalescent()

        def f(s1, s2):
            return model.get_rate_infinite_alleles(n=10, s1=np.array(s1), s2=np.array(s2))

        self.assertAlmostEqual(f([2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 4)

    def test_beta_coalescent_infinite_alleles_n_2(self):
        """
        Test beta coalescent with infinite alleles for n = 2.
        """
        model = pg.BetaCoalescent(alpha=1.5)

        # n = 2
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([2, 0]), np.array([0, 1])), 1)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([0, 1]), np.array([2, 1])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([2, 0]), np.array([2, 0])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([2, 0]), np.array([0, 2])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(2, np.array([3, 0]), np.array([1, 2])), 0)

    def test_beta_coalescent_infinite_alleles_n_3(self):
        """
        Test beta coalescent with infinite alleles for n = 3.
        """
        model = pg.BetaCoalescent(alpha=2 - 1e-10)

        self.assertAlmostEqual(model.get_rate_infinite_alleles(3, np.array([3, 0, 0]), np.array([1, 1, 0])), 3)
        self.assertAlmostEqual(model.get_rate_infinite_alleles(3, np.array([3, 0, 0]), np.array([0, 0, 1])), 0)
        self.assertEqual(model.get_rate_infinite_alleles(3, np.array([3, 0, 0, 0]), np.array([1, 1, 0, 0])), 0)

    @pytest.mark.skip(reason="Not properly implemented yet.")
    def test_beta_coalescent_infinite_alleles_n_10(self):
        """
        Test beta coalescent with infinite alleles for n = 10.
        """
        model = pg.BetaCoalescent(alpha=1.5)

        def f(s1, s2):
            return model.get_rate_infinite_alleles(n=10, s1=np.array(s1), s2=np.array(s2))

        self.assertAlmostEqual(f([2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 0.234375)
        self.assertAlmostEqual(f([2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]), 0)
        self.assertAlmostEqual(f([2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 5.46875)
        self.assertAlmostEqual(f([2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]), 0)
