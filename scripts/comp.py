import numpy as np


def diff(a, b):
    """
    Difference.
    """
    return a - b


def diff_rel(a, b):
    """
    Relative difference.
    """
    return diff(a, b) / a


def diff_max_abs(a, b):
    """
    Maximum absolute difference.
    """
    return np.max(np.abs(diff(a, b)))


def diff_rel_max_abs(a, b):
    """
    Maximum absolute relative difference.
    """
    return np.max(np.abs(diff_rel(a, b)))
