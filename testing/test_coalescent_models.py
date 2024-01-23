from unittest import TestCase

import numpy as np
import pytest

import phasegen as pg


class CoalescentModelTestCase(TestCase):
    """
    Test coalescent models.
    """

    def test_standard_coalescent_block_counting_n_2(self):
        """
        Test standard coalescent with block counting state space for n = 2.
        """
        model = pg.StandardCoalescent()

        self.assertEqual(model.get_rate_block_counting(2, np.array([2, 0]), np.array([0, 1])), 1)
        self.assertEqual(model.get_rate_block_counting(2, np.array([0, 1]), np.array([2, 1])), 0)
        self.assertEqual(model.get_rate_block_counting(2, np.array([2, 0]), np.array([2, 0])), 0)
        self.assertEqual(model.get_rate_block_counting(2, np.array([2, 0]), np.array([0, 2])), 0)
        self.assertEqual(model.get_rate_block_counting(2, np.array([3, 0]), np.array([1, 2])), 0)

    def test_standard_coalescent_block_counting_n_3(self):
        """
        Test standard coalescent with block counting state space for n = 3.
        """
        model = pg.StandardCoalescent()

        self.assertEqual(model.get_rate_block_counting(3, np.array([3, 0, 0]), np.array([1, 1, 0])), 3)
        self.assertEqual(model.get_rate_block_counting(3, np.array([3, 0, 0]), np.array([0, 0, 1])), 0)
        self.assertEqual(model.get_rate_block_counting(3, np.array([3, 0, 0, 0]), np.array([1, 1, 0, 0])), 0)

    def test_standard_coalescent_block_counting_n_10(self):
        """
        Test beta coalescent with block counting state space for n = 10.
        """
        model = pg.StandardCoalescent()

        def f(s1, s2):
            return model.get_rate_block_counting(n=10, s1=np.array(s1), s2=np.array(s2))

        self.assertAlmostEqual(f([2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 4)

    @staticmethod
    def test_beta_coalescent_default_state_space_compare_with_paper():
        """
        Test against result in paper "Phase-type distributions in population genetics"
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=5),
            model=pg.BetaCoalescent(alpha=1.5, scale_time=False),
            epoch=pg.Epoch()
        )

        np.testing.assert_array_almost_equal(s.S[:-1, :-1], np.array([
            [-6.5625, 5.46875, 0.78125, 0.234375],
            [0.0000, -4.37500, 3.75000, 0.500000],
            [0.0000, 0.00000, -2.50000, 2.250000],
            [0.0000, 0.00000, 0.00000, -1.000000]
        ]))

    def test_beta_coalescent_block_counting_n_2_alpha_1_5(self):
        """
        Test beta coalescent with block counting state space for n = 2.
        """
        model = pg.BetaCoalescent(alpha=1.5)

        # n = 2
        self.assertEqual(model.get_rate_block_counting(2, np.array([2, 0]), np.array([0, 1])), 1)
        self.assertEqual(model.get_rate_block_counting(2, np.array([0, 1]), np.array([2, 1])), 0)
        self.assertEqual(model.get_rate_block_counting(2, np.array([2, 0]), np.array([2, 0])), 0)
        self.assertEqual(model.get_rate_block_counting(2, np.array([2, 0]), np.array([0, 2])), 0)
        self.assertEqual(model.get_rate_block_counting(2, np.array([3, 0]), np.array([1, 2])), 0)

    def test_beta_coalescent_block_counting_n_3_alpha_close_to_2(self):
        """
        Test beta coalescent with block counting state space for n = 3.
        """
        model = pg.BetaCoalescent(alpha=2 - 1e-10)

        self.assertAlmostEqual(model.get_rate_block_counting(3, np.array([3, 0, 0]), np.array([1, 1, 0])), 3)
        self.assertAlmostEqual(model.get_rate_block_counting(3, np.array([3, 0, 0]), np.array([0, 0, 1])), 0)
        self.assertEqual(model.get_rate_block_counting(3, np.array([3, 0, 0, 0]), np.array([1, 1, 0, 0])), 0)

    def test_beta_coalescent_block_counting_n_3_alpha_1_5(self):
        """
        Test beta coalescent with block counting state space for n = 3.
        """
        model = pg.BetaCoalescent(alpha=1.5)

        def f(s1, s2):
            return model.get_rate_block_counting(n=3, s1=np.array(s1), s2=np.array(s2))

        self.assertEqual(f([3, 0, 0], [3, 0, 0]), 0)
        self.assertEqual(f([3, 0, 0], [0, 1, 0]), 0)
        self.assertEqual(f([3, 0, 0], [2, 0, 0]), 0)
        self.assertEqual(f([3, 0, 0], [4, 0, 0]), 0)
        self.assertEqual(f([3, 0, 0], [1, 0, 1]), 0)
        self.assertEqual(f([3, 0, 0], [1, 1, 1]), 0)
        self.assertEqual(f([3, 0, 0], [1, 2, 0]), 0)
        self.assertAlmostEqual(f([3, 0, 0], [0, 0, 1]), 0.25)
        self.assertAlmostEqual(f([3, 0, 0], [1, 1, 0]), 2.25)

        self.assertAlmostEqual(f([1, 1, 0], [0, 0, 1]), 1)

    def test_beta_coalescent_positive_correlation(self):
        """
        Test beta coalescent with positive correlation.
        """
        n = 10

        c = pg.Coalescent(
            n=pg.LineageConfig(n=n),
            model=pg.BetaCoalescent(alpha=1.1),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1}})
        )

        c2 = pg.Coalescent(
            n=pg.LineageConfig(n=n),
            model=pg.StandardCoalescent(),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1}})
        )

        # many more positive entries for beta coalescent
        self.assertAlmostEqual(0.4, np.sum(c.sfs.cov.data > 0) / (n + 1) ** 2, delta=0.01)
        self.assertAlmostEqual(0.14, np.sum(c2.sfs.cov.data > 0) / (n + 1) ** 2, delta=0.01)

    @staticmethod
    def test_beta_coalescent_get_generation_time():
        """
        Test beta coalescent generation time.
        """
        pg.BetaCoalescent(alpha=1.999)._get_timescale(1)

    @pytest.mark.skip(reason="Not finished")
    def test_dirac_coalescent_n_5(self):
        """
        Test Dirac coalescent with block counting state space for n = 2.
        """
        model = pg.DiracCoalescent(psi=0.8, c=1)

        self.assertAlmostEqual(
            model.get_rate_block_counting(5, np.array([5, 0, 0, 0, 0]), np.array([3, 1, 0, 0, 0])), 8.33333333)

        self.assertEqual(model.get_rate_block_counting(5, np.array([5, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 1])), 5)
