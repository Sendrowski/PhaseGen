"""
Caching utilities honoring a global on/off switch (:attr:`phasegen.settings.Settings.cache`).

These are drop-in replacements for :func:`functools.cached_property` and :func:`functools.cache`. When caching is
disabled (``Settings.cache = False``) they act as a frozen cache: stored values are still served, but no new
entries are stored, so anything not yet computed is recomputed on every access. This is useful for debugging
(forcing fresh computation of new results, profiling without cache hits masking cost) while keeping expensive
results that are cached (e.g. a deserialized msprime comparison or a constructed state space).
"""
import functools
import weakref

from .settings import Settings

#: A monotonically increasing "computation epoch", bumped each time an *outermost* cached/memoized computation
#: starts (a moment, a spectrum, ...). The deduplicating log filter uses it to scope deduplication to a single
#: coalescent computation: identical log records collapse within one computation and re-emit in the next. Tracked
#: here because every computation funnels through one of the caching decorators below.
computation_epoch = 0
_computation_depth = 0


def _enter_computation() -> None:
    """Mark the start of a (possibly nested) cached/memoized computation; bump the epoch only at the outermost."""
    global _computation_depth, computation_epoch
    if _computation_depth == 0:
        computation_epoch += 1
    _computation_depth += 1


def _exit_computation() -> None:
    """Mark the end of a cached/memoized computation."""
    global _computation_depth
    _computation_depth -= 1


class cached_property(functools.cached_property):
    """
    Like :class:`functools.cached_property`, but only stores the computed value when :attr:`Settings.cache` is
    ``True``. A cached value (present in the instance ``__dict__``) is always returned. When caching is disabled, an
    uncached property is recomputed on each access without being stored.

    Subclasses :class:`functools.cached_property` (inheriting its ``__init__`` / ``__set_name__``) so that tools that
    special-case it, notably Sphinx autodoc, which then documents the property with its return-type annotation,
    recognise it. Only :meth:`__get__` is overridden, to add the cache gating and the computation-epoch tracking.
    """

    def __get__(self, instance, owner=None) -> 'Any':
        # A non-data descriptor: once the value is in the instance ``__dict__`` Python returns it directly without
        # calling ``__get__`` (so already-cached values are always used, even with caching disabled). ``__get__``
        # only runs on a miss, and then stores only when caching is enabled.
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError("Cannot use cached_property instance without calling __set_name__ on it.")

        _enter_computation()
        try:
            value = self.func(instance)
        finally:
            _exit_computation()

        if Settings.cache:
            try:
                instance.__dict__[self.attrname] = value
            except AttributeError:  # e.g. __slots__ without __dict__: cannot cache
                pass
        return value


def cache(func) -> 'Callable':
    """
    Like :func:`functools.cache`, but only stores new results when :attr:`Settings.cache` is ``True``. Existing
    memoized results are always served. With caching disabled, an un-memoized call is recomputed and not stored.

    All ``@cache``-decorated functions in the library are methods, so the memo is kept per instance in a
    :class:`weakref.WeakKeyDictionary` keyed on ``self``. The cached results of an instance, which may reference its
    state space and dense rate matrix, are dropped when the instance is garbage-collected.
    """
    memo = weakref.WeakKeyDictionary()  # self -> {argkey: result}

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> 'Any':
        key = functools._make_key(args, kwargs, typed=False)
        entries = memo.get(self)
        if entries is not None and key in entries:
            return entries[key]
        _enter_computation()
        try:
            result = func(self, *args, **kwargs)
        finally:
            _exit_computation()
        if Settings.cache:
            memo.setdefault(self, {})[key] = result
        return result

    return wrapper
