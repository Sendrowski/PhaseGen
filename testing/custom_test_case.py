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
    pass
