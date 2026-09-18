"""
Utility functions.
"""
import itertools
import logging
import sys
from types import FunctionType
from typing import Callable, Dict, List, Sequence, Generator, Tuple, Any, Iterable, Iterator, Optional

import dill
import multiprocess as mp
import numpy as np
from tqdm import tqdm

from .expm import Backend
from .settings import Settings

logger = logging.getLogger('phasegen')


class _ConfiguredCall:
    """
    A function together with a snapshot of the process-global configuration of the process that wrapped it, which it
    applies in the process that calls it.

    The configuration is the public :class:`~phasegen.settings.Settings` attributes and the registered matrix
    exponentiation backend, both of which a worker started by ``spawn`` holds at the value declared in the package.
    Applying it immediately before each call also gives it precedence over a module that registers a backend or
    assigns a setting when the worker imports it.
    """

    def __init__(self, func: Callable) -> None:
        """
        Wrap a function together with the configuration of the calling process.

        :param func: Function to call in the worker process.
        """
        #: Function to call.
        self.func: Callable = func

        #: Values of the settings, by name.
        self.settings: Dict[str, Any] = {
            name: getattr(Settings, name) for name, value in vars(Settings).items()
            if not name.startswith('_') and not isinstance(value, (staticmethod, classmethod, property, FunctionType))
        }

        #: Registered matrix exponentiation backend, serialized by reference, or ``None`` if it cannot be serialized.
        self.backend: Optional[bytes] = self._dump_backend()

    @staticmethod
    def _dump_backend() -> Optional[bytes]:
        """
        Serialize the registered matrix exponentiation backend.

        :return: The serialized backend, or ``None`` if it cannot be serialized.
        """
        try:
            # by reference: the backend classes live in the package the worker imports anyway, and serializing them
            # by value would rebuild the modules they reference in the worker's namespace, replacing objects an
            # already-imported module holds by identity
            return dill.dumps(Backend.backend)
        except Exception as e:
            logger.warning(
                'Could not serialize the registered matrix exponentiation backend %r, so the worker processes use '
                'the default backend instead: %s', Backend.backend, e
            )

            return None

    def __call__(self, item: Any) -> Any:
        """
        Apply the configuration and call the wrapped function.

        :param item: Item passed to the wrapped function.
        :return: Return value of the wrapped function.
        """
        for name, value in self.settings.items():
            setattr(Settings, name, value)

        if self.backend is not None:
            Backend.register(dill.loads(self.backend))

        return self.func(item)


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

    Each call in a worker runs under the calling process's :class:`~phasegen.settings.Settings` and its registered
    matrix exponentiation backend, which the ``spawn`` start method would otherwise reset to their defaults, so that
    the result does not depend on whether it was computed in the calling process or in a worker.

    On macOS the worker pool uses the ``spawn`` start method, not the ``fork`` default of ``multiprocess``.
    Forking a process that has initialized threaded native libraries (numba/llvmlite, and on macOS the
    Accelerate BLAS and libdispatch) copies their internal locks in a held state, so the first such call in
    the child deadlocks. ``spawn`` starts a fresh interpreter and sidesteps the inherited locks. The
    platform default is kept elsewhere (``fork`` on Linux), where it is safe and avoids the per-worker
    re-import cost of ``spawn``. Because ``spawn`` re-imports the caller's module, on macOS callers must be
    import-safe (guard top-level code with ``if __name__ == '__main__':``) and ``func``/``data`` must be
    picklable (handled here by ``dill`` via ``multiprocess``).

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize, overridden by ``Settings.parallelize = False``
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

    if parallelize and Settings.parallelize and len(data) > 1:
        # spawn on macOS (fork there deadlocks once numba/Accelerate are loaded); platform default elsewhere
        ctx = mp.get_context('spawn') if sys.platform == 'darwin' else mp.get_context()
        try:
            # consume the lazy imap iterator while the pool is still open
            with ctx.Pool() as pool:
                return np.array(list(with_pbar(pool.imap(_ConfiguredCall(func), data))), dtype=dtype)
        except RuntimeError as e:
            # ``spawn`` re-imports the caller's module in every worker; if the entry point is not import-safe the
            # worker re-runs the top-level code (re-spawning, repeated side effects such as plots) and multiprocessing
            # raises an opaque "bootstrapping phase" error. Replace it with actionable guidance.
            if 'bootstrapping phase' in str(e) or 'freeze_support' in str(e):
                raise RuntimeError(
                    "Could not start worker processes for parallelize=True. On this platform the workers use the "
                    "'spawn' start method, which re-imports your script, so the entry point must be import-safe. "
                    "Either pass parallelize=False (recommended for interactive or plotting use), or guard the "
                    "top-level code of your script with `if __name__ == '__main__':`."
                ) from e
            raise

    return np.array(list(with_pbar(map(func, data))), dtype=dtype)


def multiset_permutations(items: Sequence) -> Generator[Tuple, None, None]:
    """
    Generate multiset permutations.
    Adapted from https://stackoverflow.com/questions/19676109/how-to-generate-all-the-permutations-of-a-multiset

    :param items: Items to permute
    :return: Permutations
    """

    def visit(head) -> tuple:
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
        try:
            yield next(iterator)
        except StopIteration:
            return
