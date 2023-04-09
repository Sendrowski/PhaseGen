from unittest import TestCase

import numpy as np


class CustomTestCase(TestCase):

    @staticmethod
    def diff(a, b):
        """
        Difference.
        """
        return a - b

    def diff_rel(self, a, b):
        """
        Relative difference.
        """
        return self.diff(a, b) / a

    def diff_max_abs(self, a, b):
        """
        Maximum absolute difference.
        """
        return np.max(np.abs(self.diff(a, b)))

    def diff_rel_max_abs(self, a, b):
        """
        Maximum absolute relative difference.
        """
        return np.max(np.abs(self.diff_rel(a, b)))
