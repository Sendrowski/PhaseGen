"""Shared helpers for the distributions package."""
import functools

import numpy as np
from typing import Callable

from ..rewards import Reward


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
    :raises ValueError: if ``k`` is not integral or is negative.
    """
    if isinstance(k, bool) or not isinstance(k, (int, float, np.integer, np.floating)):
        raise TypeError(f"The order k must be an integer, but got {type(k).__name__}.")

    if not float(k).is_integer():
        raise ValueError(f"The order k must be an integer, but got {k}.")

    k = int(k)

    if k < 0:
        raise ValueError(f"The order k must be non-negative, but got {k}.")

    return k


def _validate_reward(reward: Reward, name: str = 'reward') -> None:
    """
    Check that a single reward was passed.

    :param reward: The argument to check.
    :param name: The name of the argument, for the error message.
    :raises TypeError: if ``reward`` is not a :class:`~phasegen.rewards.Reward`.
    """
    if not isinstance(reward, Reward):
        got = 'a sequence' if isinstance(reward, (list, tuple)) else type(reward).__name__
        raise TypeError(f"{name} must be a single {Reward.__name__}, but got {got}.")
