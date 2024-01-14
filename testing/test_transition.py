from unittest import TestCase

import numpy as np
from numpy import testing

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

        self.assertTrue(t.is_eligible_coalescence)
        self.assertTrue(t.is_coalescence)
        self.assertTrue(t.is_marginal_coalescence)
