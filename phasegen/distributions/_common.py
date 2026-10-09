"""Shared helpers for the distributions package."""
import copy
import functools

import numpy as np
from typing import Callable, Sequence, Tuple

from ..rewards import Reward

#: Number of samples ``to_empirical`` draws by default.
N_EMPIRICAL_SAMPLES = 100000


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


def _validate_rewards(rewards: Sequence[Reward] | None, k: int) -> None:
    """
    Check that the rewards of a moment of order ``k`` are a sequence of rewards.

    :param rewards: The rewards, ``None`` for the default rewards.
    :param k: The order of the moment.
    :raises ValueError: if a single :class:`~phasegen.rewards.Reward` is passed instead of a sequence.
    :raises TypeError: if an entry of ``rewards`` is not a :class:`~phasegen.rewards.Reward`.
    """
    if isinstance(rewards, Reward):
        raise ValueError(
            f"rewards must be a sequence of {k} rewards, but a single {Reward.__name__} instance was given. "
            f"Wrap it in a list, e.g. rewards=[reward]."
        )

    for i, reward in enumerate(rewards or []):
        _validate_reward(reward, f"rewards[{i}]")


def _validate_reward_count(rewards: Sequence[Reward], k: int) -> None:
    """
    Check that a moment of order ``k`` has ``k`` rewards.

    :param rewards: The rewards.
    :param k: The order of the moment.
    :raises ValueError: if the number of rewards differs from ``k``.
    """
    if len(rewards) != k:
        raise ValueError(f"Number of specified rewards for moment of order {k} must be {k}.")


def _validate_start_time(start_time: float) -> None:
    """
    Check that a start time is non-negative.

    :param start_time: The start time.
    :raises ValueError: if ``start_time`` is negative or NaN.
    """
    if not start_time >= 0:
        raise ValueError(f"Start time must be greater than or equal to 0, got {start_time}.")


def _frequency_class(i: 'int | float', n: int) -> int:
    """
    Validate a frequency class of a spectrum array.

    :param i: The frequency class, an integer from 0 to ``n``.
    :param n: The number of lineages.
    :return: The frequency class as an integer.
    :raises ValueError: if ``i`` is not an integer from 0 to ``n``.
    """
    if isinstance(i, bool) or not float(i).is_integer() or not 0 <= i <= n:
        raise ValueError(f"The frequency class must be an integer from 0 to {n}, got {i}.")

    return int(i)


def _polymorphic_class(i: 'int | float', first: int, last: int) -> int:
    """
    Validate a polymorphic frequency class.

    :param i: The frequency class, an integer from ``first`` to ``last``.
    :param first: The first polymorphic class.
    :param last: The last polymorphic class.
    :return: The frequency class as an integer.
    :raises ValueError: if ``i`` is not an integer from ``first`` to ``last``.
    """
    if isinstance(i, bool) or not float(i).is_integer() or not first <= i <= last:
        raise ValueError(f"The frequency class must be a polymorphic class from {first} to {last}, got {i}.")

    return int(i)


def _descendant_config(config: Sequence[int], full: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Validate the descendant configuration of a polymorphic joint SFS bin.

    :param config: The descendant configuration, one count per population.
    :param full: The sample size of each population.
    :return: The configuration as a tuple of integers.
    :raises ValueError: if ``config`` is not the descendant configuration of a polymorphic joint SFS bin.
    """
    config = tuple(config)

    if (
            len(config) != len(full)
            or any(isinstance(c, bool) or not float(c).is_integer() for c in config)
            or not all(0 <= c <= n for c, n in zip(config, full))
            or not any(config)
            or tuple(int(c) for c in config) == full
    ):
        raise ValueError(
            f"The descendant configuration must hold one integer count per population, each from 0 to its sample "
            f"size {full}, and be neither all zero nor {full}, got {config}."
        )

    return tuple(int(c) for c in config)


def _sqrt_spectrum(var):
    """
    The entrywise square root of a variance spectrum, with entries that rounding leaves marginally below zero read as
    zero, keeping the type and metadata of the spectrum, such as its population names.

    :param var: The variance spectrum.
    :return: The standard-deviation spectrum.
    """
    out = copy.deepcopy(var)
    out.data = np.maximum(np.asarray(var.data, dtype=float), 0.0) ** 0.5

    return out
