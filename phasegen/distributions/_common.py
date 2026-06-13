"""Shared helpers for the distributions package."""
import functools

import numpy as np
from typing import Callable


def _make_hashable(func: Callable) -> Callable:
    """
    Decorator that makes a function hashable by converting non-hashable arguments to hashable ones.
    """

    @functools.wraps(func)
    def wrapper(self, *args: tuple, **kwargs: dict):
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
