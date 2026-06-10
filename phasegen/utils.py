"""
Utility functions.
"""
import itertools
import sys
from typing import Callable, List, Sequence, Generator, Tuple, Any, Iterable, Iterator

import multiprocess as mp
import numpy as np
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

    On macOS the worker pool uses the ``spawn`` start method instead of ``multiprocess``'s default ``fork``.
    Forking a process that has already initialized threaded native libraries (numba/llvmlite, and on macOS
    the Accelerate BLAS and libdispatch) copies their internal locks in a held state, so the first such call
    in the child deadlocks; ``spawn`` starts a fresh interpreter and sidesteps the inherited locks. The
    platform default is kept elsewhere (``fork`` on Linux), where it is safe and avoids the per-worker
    re-import cost of ``spawn``. Because ``spawn`` re-imports the caller's module, on macOS callers must be
    import-safe (guard top-level code with ``if __name__ == '__main__':``) and ``func``/``data`` must be
    picklable (handled here by ``dill`` via ``multiprocess``).

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
    def with_pbar(it: Iterable) -> Iterable:
        """Optionally wrap an iterator in a tqdm progress bar."""
        if pbar:
            return tqdm(it, total=len(data), unit_scale=batch_size, desc=desc, delay=delay)

        return it

    if parallelize and len(data) > 1:
        # spawn on macOS (fork there deadlocks once numba/Accelerate are loaded); platform default elsewhere
        ctx = mp.get_context('spawn') if sys.platform == 'darwin' else mp.get_context()
        # consume the lazy imap iterator while the pool is still open
        with ctx.Pool() as pool:
            return np.array(list(with_pbar(pool.imap(func, data))), dtype=dtype)

    return np.array(list(with_pbar(map(func, data))), dtype=dtype)


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
