from unittest import TestCase

import numpy as np
import pytest
from numpy import testing

import phasegen as pg


class StateSpaceTestCase(TestCase):
    """
    Test StateSpace class.
    """

    @staticmethod
    def test_default_intensity_matrix_n_4():
        """
        Test default intensity matrix for n = 4.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=4),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        testing.assert_array_equal(s.S, np.array([[-6., 6., 0., 0.],
                                                  [0., -3., 3., 0.],
                                                  [0., 0., -1., 1.],
                                                  [0., 0., 0., -0.]]))

    @staticmethod
    def test_n_2_2_demes():
        """
        Test n = 2, 2 demes.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 2}
            )
        )

        np.testing.assert_array_equal(s.S, np.array([[-1., 0., 0., 1., 0.],
                                                     [0., -0., 0., 0., 0.],
                                                     [0., 0., -0.5, 0., 0.5],
                                                     [0., 0., 0., -0., 0.],
                                                     [0., 0., 0., 0., -0.]]))

    def test_2_loci_default_state_space_n_2(self):
        """
        Test two loci, n = 2.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        expected = np.array(
            [[-6., 1., 1., 0., 4., 0., 0., 0., 0.],
             [0., -3., 0., 1., 0., 2., 0., 0., 0.],
             [0., 0., -3., 1., 0., 0., 2., 0., 0.],
             [0., 0., 0., -0., 0., 0., 0., 0., 0.],
             [1.11, 0., 0., 0., -4.11, 1., 1., 0., 1.],
             [0., 1.11, 0., 0., 0., -2.11, 0., 1., 0.],
             [0., 0., 1.11, 0., 0., 0., -2.11, 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., -0., 0.],
             [0., 0., 0., 0., 2.22, 0., 0., 1., -3.22]]
        )

        np.testing.assert_array_almost_equal(s.S, expected)

    @staticmethod
    def test_2_loci_default_state_space_n_3():
        """
        Test two loci, n = 3.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        expected = np.array(
            [[-15., 3., 0., 3., 0., 0., 0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., -10., 1., 0., 3., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., -6., 0., 0., 3., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., -10., 3., 0., 1., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., -6., 1., 0., 1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., -3., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., -6., 3., 0., 0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., -3., 1., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1.11, 0., 0., 0., 0., 0., 0., 0., 0., -11.11, 3., 0., 3., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.],
             [0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -7.11, 1., 0., 3., 0., 0., 0., 0., 0., 2., 0., 0., 0.],
             [0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -4.11, 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -7.11, 3., 0., 1., 0., 0., 0., 0., 2., 0., 0.],
             [0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -4.11, 1., 0., 1., 0., 0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -2.11, 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -4.11, 3., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -2.11, 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., 0., -8.22, 2., 2., 0., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., 0., -5.22, 0., 2., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., -5.22, 2., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., -3.22, 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.33, 0., 0., 3., -6.33]]
        )

        np.testing.assert_array_almost_equal(s.S, expected)

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_2(self):
        """
        Test two loci, one deme, two lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2)
        )

        s.S

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_3(self):
        """
        Test two loci, one deme, two lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        _ = s.S

        pass

    def test_default_state_space_two_loci_one_deme_n_4(self):
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

        pass

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_4(self):
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

        pass

    def test_default_state_space_two_loci_two_demes_n_4(self):
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        _ = s.S

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_two_demes_n_4(self):
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        s._matrix_indices_to_rates(223, 400)

        _ = s.S

    def test_default_state_space_size(self):
        """
        Test default state space size.
        """
        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(10)
        ).k, 10)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(20)
        ).k, 20)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 65)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 494)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 815)

    def test_block_counting_state_space_size(self):
        """
        Test block counting state space size.
        """
        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(10)
        ).k, 42)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(20)
        ).k, 627)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 481)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 2580)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 35581)
