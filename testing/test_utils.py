"""
Tests for :mod:`phasegen.utils`.
"""
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

    monkeypatch.setattr(utils.mp, 'get_context', lambda *a, **k: type('Ctx', (), {'Pool': lambda self: _FakePool()})())

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
    :return: The values of three settings, the name and precision of the registered backend, and the mean tree height
        of the standard coalescent of four lineages.
    """
    import phasegen as pg

    return (
        Settings.dehoog_degree,
        Settings.closed_form_last_epoch,
        Settings.max_state_space_size,
        type(pg.Backend.backend).__name__,
        str(pg.Backend.backend.precision),
        float(pg.Coalescent(n=4).tree_height.mean)
    )


def test_parallelize_workers_inherit_settings_and_backend():
    """The ``spawn`` start method re-imports the package in each worker, which reverted every ``Settings`` attribute
    and the registered matrix exponentiation backend to its declared default. A computation performed in a worker
    therefore ran under a configuration the caller had not asked for: ``max_state_space_size`` no longer bounded the
    state space, and a backend registered with single precision was replaced by the double-precision default, which
    moved the mean tree height in the eighth significant digit."""
    import phasegen as pg

    original = pg.Backend.backend

    Settings.dehoog_degree = 4
    Settings.closed_form_last_epoch = False
    Settings.max_state_space_size = 12345
    pg.Backend.register(pg.SciPyExpmBackend(precision=np.float32))

    try:
        expected = _read_worker_config(0)
        results = utils.parallelize(func=_read_worker_config, data=[0, 1], parallelize=True, pbar=False, dtype=object)
    finally:
        pg.Backend.register(original)

    assert expected[:5] == (4, False, 12345, 'SciPyExpmBackend', 'float32')

    for result in results:
        assert tuple(result[:5]) == expected[:5]
        np.testing.assert_allclose(float(result[5]), expected[5], rtol=1e-12)


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
