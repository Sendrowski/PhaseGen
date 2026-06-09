"""
Shared pytest fixtures and hooks for the test suite.
"""
from itertools import product

import pytest


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
