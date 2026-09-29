"""
Tests for :mod:`phasegen.utils`.
"""
import logging

import numpy as np
import pytest

from phasegen import utils
from phasegen.settings import Settings


def test_parallelize_spawn_guard_message(monkeypatch):
    """When the worker pool cannot bootstrap (e.g. a non-import-safe entry point under the macOS 'spawn' start
    method), ``parallelize`` replaces the opaque multiprocessing error with actionable guidance."""

    class _FakePool:
        def __enter__(self):
            raise RuntimeError("An attempt has been made to start a new process before the current process has "
                               "finished its bootstrapping phase.")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(utils.mp, 'get_context', lambda *a, **k: type('Ctx', (), {'Pool': lambda self, processes=None: _FakePool()})())

    with pytest.raises(RuntimeError, match="import-safe"):
        utils.parallelize(func=lambda x: x, data=[1, 2, 3], parallelize=True, pbar=False)


def test_parallelize_setting_runs_sequentially(monkeypatch):
    """``Settings.parallelize = False`` runs a ``parallelize=True`` call in the calling process, without a worker
    pool."""

    def no_pool(*args, **kwargs):
        raise AssertionError("a worker pool was requested")

    monkeypatch.setattr(utils.mp, 'get_context', no_pool)
    monkeypatch.setattr(Settings, 'parallelize', False)

    result = utils.parallelize(func=lambda x: 2 * x, data=[1, 2, 3], parallelize=True, pbar=False)

    np.testing.assert_array_equal(result, [2, 4, 6])


def _read_worker_config(_) -> tuple:
    """
    Report the configuration a worker process sees, and a quantity that depends on it.

    :param _: Ignored.
    :return: The values of three settings, the name and precision of the registered backend, the level of the
        ``phasegen`` logger, and the mean tree height of the standard coalescent of four lineages.
    """
    import phasegen as pg

    return (
        Settings.dehoog_degree,
        Settings.closed_form_last_epoch,
        Settings.max_state_space_size,
        type(pg.Backend.backend).__name__,
        str(pg.Backend.backend.precision),
        logging.getLogger('phasegen').level,
        float(pg.Coalescent(n=4).tree_height.mean)
    )


def test_parallelize_workers_inherit_settings_and_backend(monkeypatch):
    """The ``spawn`` start method re-imports the package in each worker, which reverted every ``Settings`` attribute
    and the registered matrix exponentiation backend to its declared default. A computation performed in a worker
    therefore ran under a configuration the caller had not asked for: ``max_state_space_size`` no longer bounded the
    state space, and a backend registered with single precision was replaced by the double-precision default, which
    moved the mean tree height in the eighth significant digit. A level set on the ``phasegen`` logger was reset to
    INFO in the same way. The pool is forced to ``spawn``, since a forked worker inherits the caller's state on any
    platform."""
    import phasegen as pg

    get_context = utils.mp.get_context
    monkeypatch.setattr(utils.mp, 'get_context', lambda *args, **kwargs: get_context('spawn'))

    original = pg.Backend.backend
    log = logging.getLogger('phasegen')
    level = log.level

    Settings.dehoog_degree = 4
    Settings.closed_form_last_epoch = False
    Settings.max_state_space_size = 12345
    pg.Backend.register(pg.SciPyExpmBackend(precision=np.float32))
    log.setLevel(logging.ERROR)

    try:
        expected = _read_worker_config(0)
        results = utils.parallelize(func=_read_worker_config, data=[0, 1], parallelize=True, pbar=False, dtype=object)
    finally:
        pg.Backend.register(original)
        log.setLevel(level)

    assert expected[:6] == (4, False, 12345, 'SciPyExpmBackend', 'float32', logging.ERROR)

    for result in results:
        assert tuple(result[:6]) == expected[:6]
        np.testing.assert_allclose(float(result[6]), expected[6], rtol=1e-12)


def test_parallel_workers_do_not_warn_about_the_backend(caplog):
    """The worker restores the caller's matrix-exponential backend. Regression: it did so through the deprecated
    Backend.register, so every worker logged a deprecation warning the caller never triggered."""
    import logging

    # the worker side runs in the caller's process here, so its log records reach caplog
    call = utils._ConfiguredCall(_read_worker_config)

    log = logging.getLogger('phasegen')
    log.addHandler(caplog.handler)
    try:
        caplog.clear()
        call(0)
    finally:
        log.removeHandler(caplog.handler)

    assert not any('deprecated' in r.getMessage() for r in caplog.records)


def test_plot_clear_false_draws_onto_the_current_axes():
    """The clear parameter of the plot methods decides whether a plot starts a new figure. Regression: it was never
    read, so clear=False could not overlay."""
    import matplotlib.pyplot as plt
    import phasegen as pg

    plt.close('all')
    dist = pg.Coalescent(n=3).tree_height

    ax1 = dist.cdf.plot(show=False)
    ax2 = dist.pdf.plot(show=False, clear=False)

    assert ax2 is ax1
    assert len(ax2.get_lines()) == 2

    ax3 = dist.cdf.plot(show=False)
    assert len(ax3.get_lines()) == 1
    plt.close('all')


def test_worker_pool_is_sized_by_the_cpu_allocation_and_the_data(monkeypatch):
    """The pool starts no more workers than the CPUs the process may run on and the items to map. Regression: it
    started one per CPU of the machine, 192 on a SLURM job allocated 2."""
    import phasegen.utils as utils

    sizes = []
    real = utils.mp.get_context

    class Context:
        def __init__(self, ctx):
            self._ctx = ctx

        def Pool(self, processes=None):
            sizes.append(processes)
            return self._ctx.Pool(processes)

    monkeypatch.setattr(utils.os, 'sched_getaffinity', lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(utils.mp, 'get_context', lambda *args: Context(real(*args)))

    utils.parallelize(abs, [-1.0, -2.0, -3.0], pbar=False)
    utils.parallelize(abs, [-1.0], pbar=False)
    monkeypatch.setattr(utils.os, 'sched_getaffinity', lambda pid: set(range(8)), raising=False)
    utils.parallelize(abs, [-1.0, -2.0, -3.0], pbar=False)

    assert sizes == [2, 3]


def test_use_pbar_governs_the_msprime_simulation_bar(capsys):
    """Settings.use_pbar decides whether MsprimeCoalescent.simulate shows its progress bar. Regression: the bar was
    always shown, and use_pbar reached only the pure-Python state-space construction."""
    from phasegen.distributions.empirical import MsprimeCoalescent

    for enabled in (False, True):
        with Settings.set_pbar(enabled):
            MsprimeCoalescent(n=2, num_replicates=20, parallelize=False, seed=0).simulate()

        assert ('Simulating trees' in capsys.readouterr().err) == enabled


def test_identical_warnings_outside_a_computation_are_not_deduplicated():
    """Repeats of a warning logged outside any cached computation, such as constructor validation, were dropped."""
    from phasegen import DeduplicatingFilter

    record = logging.LogRecord('phasegen.MsprimeCoalescent', logging.WARNING, '', 0, 'message', None, None)
    log_filter = DeduplicatingFilter()

    assert [log_filter.filter(record) for _ in range(3)] == [True, True, True]


def test_public_names_are_exported():
    """MultinomialLikelihood was importable from phasegen but missing from its __all__."""
    import phasegen as pg

    assert 'MultinomialLikelihood' in pg.__all__


def test_log_factorial_is_exact():
    """The Stirling approximation above 100 was off by up to 3.4e-7 at 101."""
    from scipy.special import gammaln
    from phasegen._likelihood import Likelihood

    n = np.array([0, 1, 5, 100, 101, 1000])

    np.testing.assert_array_equal(Likelihood.log_factorial(n), gammaln(n + 1.0))
