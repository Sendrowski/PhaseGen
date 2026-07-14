"""
Shared pytest fixtures and hooks for the test suite.
"""
import os
from itertools import product

import pytest

_THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')


def pytest_configure(config):
    """
    Refuse to run with unpinned BLAS/numba thread pools.

    An unpinned run is not merely slow (4x, under ``-n auto``): the thread count changes the summation order of the
    matrix exponential, and a tolerance tuned against one core count is then not reproducible on another. That is how
    the conditional identity of the psi = 0.5, c = 50 Dirac came to sit on a bound it only met on this machine. A
    warning would scroll past unread in a 700-test run, so this is an error. Set ``PHASEGEN_ALLOW_THREADS=1`` to run
    unpinned deliberately (e.g. when timing a single test).
    """
    if os.environ.get('PHASEGEN_ALLOW_THREADS'):
        return

    from testing import blas_pinned_late

    if blas_pinned_late:
        raise pytest.UsageError(
            "numpy was imported before the thread limits were set, so BLAS has already sized its pool and "
            f"{', '.join(_THREAD_VARS)} no longer bite. Run pytest from the repository root (so the root conftest is "
            "picked up), or set the variables in the environment before invoking it."
        )

    if unpinned := [v for v in _THREAD_VARS if os.environ.get(v) != '1']:
        raise pytest.UsageError(
            f"unpinned thread pools ({', '.join(unpinned)}); tolerances are tuned against single-threaded summation "
            "order and the suite is 4x slower without them. Set them to 1, or PHASEGEN_ALLOW_THREADS=1 to override."
        )


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Close all matplotlib figures after every test. This is autouse and lives in ``conftest`` so that it applies to
    every test (including plain ``unittest.TestCase`` classes), preventing matplotlib global state (e.g. a log-scaled
    axis) from leaking between tests and causing order-dependent plotting failures under non-interactive backends.
    """
    yield

    import matplotlib.pyplot as plt

    plt.close('all')


@pytest.fixture(autouse=True)
def _restore_closed_form_setting():
    """
    Snapshot and restore ``Settings.closed_form_last_epoch`` around every test, so tests that pin it (e.g. to
    validate the matrix-exponential path) do not leak the value into later tests.
    """
    import phasegen as pg

    original = pg.Settings.closed_form_last_epoch

    yield

    pg.Settings.closed_form_last_epoch = original


@pytest.fixture(scope="session")
def symmetric_demography():
    """
    Factory fixture returning a function that builds a :class:`~phasegen.demography.Demography` with the given
    population sizes and symmetric migration between all population pairs.
    """
    import phasegen as pg

    def _make(pop_sizes: dict, migration_rate: float = 1.0) -> 'pg.Demography':
        pops = list(pop_sizes)
        migration_rates = {(a, b): migration_rate for a, b in product(pops, repeat=2) if a != b}

        return pg.Demography(pop_sizes=pop_sizes, migration_rates=migration_rates)

    return _make
