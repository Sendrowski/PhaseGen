"""
Caching utilities honoring a global on/off switch (:attr:`phasegen.settings.Settings.cache`).

These are drop-in replacements for :func:`functools.cached_property` and :func:`functools.cache`. When caching is
disabled (``Settings.cache = False``) they act as a *frozen* cache: values that are already cached are still
served, but no new entries are stored, so anything not yet computed is recomputed on every access. This is useful
for debugging (forcing fresh computation of new results, profiling without cache hits masking cost) while never
discarding or bypassing expensive results that were already cached (e.g. a deserialized msprime comparison or a
constructed state space).
"""
import functools

from .settings import Settings

_MISSING = object()

#: A monotonically increasing "computation epoch", bumped each time an *outermost* cached/memoized computation
#: starts (a moment, a spectrum, ...). The deduplicating log filter uses it to scope deduplication to a single
#: coalescent computation: identical log records collapse within one computation and re-emit in the next. Tracked
#: here because every computation funnels through one of the caching decorators below.
computation_epoch = 0
_computation_depth = 0


def _enter_computation():
    """Mark the start of a (possibly nested) cached/memoized computation; bump the epoch only at the outermost."""
    global _computation_depth, computation_epoch
    if _computation_depth == 0:
        computation_epoch += 1
    _computation_depth += 1


def _exit_computation():
    """Mark the end of a cached/memoized computation."""
    global _computation_depth
    _computation_depth -= 1


class cached_property:
    """
    Like :class:`functools.cached_property`, but only stores the computed value when :attr:`Settings.cache` is
    ``True``. An already-cached value (present in the instance ``__dict__``) is always returned; when caching is
    disabled, an uncached property is recomputed on each access instead of being stored.
    """

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = getattr(func, '__doc__', None)

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                f"Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
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


def cache(func):
    """
    Like :func:`functools.cache`, but only stores new results when :attr:`Settings.cache` is ``True``. Existing
    memoized results are always served; with caching disabled, an un-memoized call is recomputed and not stored.
    Exposes ``cache_clear`` / ``cache_info`` like :func:`functools.cache`.
    """
    memo = {}
    hits = misses = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        key = functools._make_key(args, kwargs, typed=False)
        try:
            result = memo[key]
            hits += 1
            return result
        except KeyError:
            misses += 1
        _enter_computation()
        try:
            result = func(*args, **kwargs)
        finally:
            _exit_computation()
        if Settings.cache:
            memo[key] = result
        return result

    def cache_clear():
        nonlocal hits, misses
        memo.clear()
        hits = misses = 0

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = lambda: functools._CacheInfo(hits, misses, None, len(memo))
    return wrapper
