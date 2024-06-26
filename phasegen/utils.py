"""
Utility functions.
"""
import itertools
from typing import Callable, List, Sequence, Generator, Tuple, Any, Iterable, Iterator

import numpy as np
from multiprocess.pool import Pool
from tqdm import tqdm


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        batch_size: int = None,
        desc: str = None,
        dtype: type = float,
        delay: int = 0
) -> np.ndarray:
    """
    Parallelize given function or execute sequentially.

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize
    :param pbar: Whether to show a progress bar
    :param batch_size: Number of units to show in the pbar per function
    :param desc: Description for tqdm progress bar
    :param dtype: Data type of the results
    :param delay: Delay for tqdm progress bar
    :return: Array of results
    """
    if parallelize and len(data) > 1:
        # parallelize
        iterator = Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    if pbar:
        iterator = tqdm(iterator, total=len(data), unit_scale=batch_size, desc=desc, delay=delay)

    return np.array(list(iterator), dtype=dtype)


def multiset_permutations(items: Sequence) -> Generator[Tuple, None, None]:
    """
    Generate multiset permutations.
    Adapted from https://stackoverflow.com/questions/19676109/how-to-generate-all-the-permutations-of-a-multiset

    :param items: Items to permute
    :return: Permutations
    """

    def visit(head):
        """
        Visit the head of the permutation.
        """
        return tuple(
            u[i] for i in map(E.__getitem__, itertools.accumulate(range(N - 1), lambda e, N: nxts[e], initial=head))
        )

    u = list(set(items))

    # special case: empty multiset
    if len(u) == 0:
        yield ()
        return

    # special case: single element multiset
    if len(u) == 1:
        yield (u[0],) * len(items)
        return

    E = list(sorted(map(u.index, items)))
    N = len(E)
    nxts = list(range(1, N)) + [None]
    head = 0
    i, ai, aai = N - 3, N - 2, N - 1

    yield visit(head)

    while aai is not None or E[ai] > E[head]:
        # before k
        before = (i if aai is None or E[i] > E[aai] else ai)
        k = nxts[before]

        if E[k] > E[head]:
            i = k

        nxts[before], nxts[k], head = nxts[k], head, k
        ai = nxts[i]
        aai = nxts[ai]

        yield visit(head)


def takewhile_inclusive(predicate: Callable[[Any], bool], iterable: Iterable) -> Iterator:
    """
    Take items from the iterable while the predicate is true, including the last item.

    :param predicate: A function that returns a boolean.
    :param iterable: An iterable.
    :return: An iterator.
    """
    iterator = iter(iterable)

    for item in iterator:
        yield item
        if not predicate(item):
            break


def take_n(iterable: Iterable, n: int) -> Iterator:
    """
    Take n items from the iterable.

    :param iterable: An iterable.
    :param n: Number of items to take.
    :return: An iterator.
    """
    iterator = iter(iterable)

    for _ in range(int(n)):
        yield next(iterator)
