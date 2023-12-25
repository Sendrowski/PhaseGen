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
    def test_n_3_2_demes():
        """
        Test n = 3, 2 demes.
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
