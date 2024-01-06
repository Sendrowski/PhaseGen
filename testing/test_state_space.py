from unittest import TestCase

import numpy as np
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
            pop_config=pg.PopConfig(n=4),
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
            pop_config=pg.PopConfig(n=2),
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

    def test_default_state_space_two_loci_one_deme_four_lineages(self):
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1})
        )

        s.S

    def test_block_counting_state_space_two_loci_one_deme_four_lineages(self):
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(n=4),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1})
        )

        s.S

    def test_default_state_space_two_loci_two_demes_four_lineages(self):
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.PopConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        s.S

    def test_block_counting_state_space_two_loci_two_demes_four_lineages(self):
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        s._matrix_indices_to_rates(223, 400)

        s.S

    def test_default_state_space_size(self):
        """
        Test default state space size.
        """
        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.PopConfig(10)
        ).k, 10)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.PopConfig(20)
        ).k, 20)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.PopConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 65)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.PopConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 494)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.PopConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 815)

    def test_block_counting_state_space_size(self):
        """
        Test block counting state space size.
        """
        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(10)
        ).k, 42)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig(20)
        ).k, 627)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 481)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 2580)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.PopConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 35581)
