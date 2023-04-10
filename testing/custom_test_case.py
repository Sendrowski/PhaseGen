import inspect
import re
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt


def add_method_name_as_title(func):
    """
    Add method name as plot title.
    This wrapper only works for test functions.
    :param func:
    :type func:
    :return:
    :rtype:
    """
    def wrapper(*args, **kwargs):
        plt.title(str(args[0]).split(' ')[0])

        return func(*args, **kwargs)

    return wrapper


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
