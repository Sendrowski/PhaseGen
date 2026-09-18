"""Shared helpers for the distributions package."""
import functools

import numpy as np
from typing import Callable


def _make_hashable(func: Callable) -> Callable:
    """
    Decorator that makes a function hashable by converting non-hashable arguments to hashable ones.
    """

    @functools.wraps(func)
    def wrapper(self, *args: tuple, **kwargs: dict) -> 'Any':
        """
        Wrapper function.

        :param self: Self.
        :return: The result of the function.
        """
        args = list(args)

        for i, arg in enumerate(args):
            if isinstance(arg, (list, np.ndarray)):
                args[i] = tuple(arg)

        for key, value in kwargs.items():
            if isinstance(value, (list, np.ndarray)):
                kwargs[key] = tuple(value)

        return func(self, *args, **kwargs)

    return wrapper


def _validate_order(k: 'int | float') -> int:
    """
    Normalise the order of a moment to an integer.

    :param k: The order of the moment, an integer or a float with integral value.
    :return: The order as an integer.
    :raises TypeError: if ``k`` is not a number.
    :raises ValueError: if ``k`` is not integral or is smaller than one.
    """
    if isinstance(k, bool) or not isinstance(k, (int, float, np.integer, np.floating)):
        raise TypeError(f"The order k must be an integer, but got {type(k).__name__}.")

    if not float(k).is_integer():
        raise ValueError(f"The order k must be an integer, but got {k}.")

    k = int(k)

    if k < 1:
        raise ValueError(f"The order k must be at least 1, but got {k}.")

    return k
