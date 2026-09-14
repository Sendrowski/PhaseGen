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
