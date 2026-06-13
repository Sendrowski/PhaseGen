"""
Settings for the PhaseGen application.
"""
from contextlib import contextmanager


class Settings:
    #: Whether to flatten the block-counting state space when possible.
    #: In certain cases, this can be achieved by computing block probabilities
    #: and adjusting the rewards of the lineage-counting state space accordingly.
    #: This can substantially speed up computations.
    flatten_block_counting: bool = True

    #: Whether to show a progress bar for certain operations such as state space generation.
    use_pbar: bool = False

    #: Whether to regularize the intensity matrix for numerical stability.
    regularize: bool = True

    #: Whether to cache the rate matrix for different epochs which increases performance.
    cache_epochs: bool = True

    #: Global switch for property/result memoization (the ``cached_property`` and ``cache`` decorators in
    #: :mod:`phasegen.caching`). Set to ``False`` to force every cached property, moment and intermediate result to
    #: recompute on each access. This is meant for debugging (ruling out stale cached state, or profiling the true
    #: cost of a computation without cache hits masking it) and will be slower. Note this is distinct from
    #: :attr:`cache_epochs`, which toggles the separate per-epoch rate-matrix cache.
    cache: bool = True

    #: Whether to use the numba-accelerated state-space construction when numba is available. Set to ``False`` to
    #: force the pure-Python construction path.
    use_numba: bool = True

    #: Van Loan matrix dimension (``(k + 1) * n_states``) at or above which moments are computed via the sparse
    #: matrix-exponential action (Krylov/Taylor) instead of forming the dense propagator. The action exploits the
    #: sparsity of the rate matrix and is much faster for large state spaces, but slower for small ones. Set to a
    #: very large value to always use the dense path, or to 0 to always use the action.
    expm_action_min_dim: int = 1500

    #: Whether to evaluate the final (unbounded) epoch of a moment-to-absorption in closed form (a linear solve with
    #: the transient sub-generator) instead of exponentiating the Van Loan matrix over the estimated absorption time.
    #: The closed form is exact and faster (it never forms the dense matrix exponential, avoids the absorption-time
    #: heuristic, and enables the batched spectrum paths that share one solve across all bins). It applies only when
    #: absorption is almost sure; otherwise the code falls back to the matrix-exponential path. Enabled by default.
    #: Gates the moment-to-absorption path (``moment`` / ``_accumulate`` / ``_accumulate_closed_form``), the mean
    #: spectrum (``_occupation_times``) and the single-epoch covariance spectrum (``_two_point_occupation``); the
    #: independent dense/sparse crossovers (:attr:`expm_action_min_dim`, :attr:`closed_form_sparse_min_states`) sit
    #: below it and change only how, not what, is computed. The off switch mainly exists to validate against the
    #: matrix-exponential path.
    closed_form_last_epoch: bool = True

    #: Transient-state count at or above which the closed-form last-epoch path (see
    #: :attr:`closed_form_last_epoch`) factors the transient sub-generator with a sparse LU (and applies
    #: the sparse matrix-exponential action for its finite-epoch / occupation steps) instead of a dense LU. This is
    #: the closed-form analogue of :attr:`expm_action_min_dim` and, like it, only changes how the result is
    #: computed, never the result. The crossover is on the transient-state count alone (independent of the moment
    #: order). Set to a very large value to always use the dense path, or to 0 to always use the sparse path.
    closed_form_sparse_min_states: int = 1200

    #: State count at or above which the constructed rate matrix is kept sparse instead of dense. The moment code
    #: works with either, so this is purely a memory/speed tradeoff: a dense matrix is faster where it fits but costs
    #: ``n_states**2`` memory, which becomes prohibitive for large state spaces. The default keeps the dense matrix under ~0.5 GB. Set to a very large value to always build dense, or to 0 to always build sparse.
    dense_rate_matrix_max_states: int = 8000

    #: Maximum number of states the construction will build before aborting with a :class:`MemoryError`. This guards
    #: against a prohibitively large state space (which grows steeply with the sample size; e.g. the single-deme
    #: Raise it if you have the memory for a larger space.
    max_state_space_size: int = 1_000_000

    @staticmethod
    @contextmanager
    def set_pbar(enabled: bool = True):
        """
        Context manager to temporarily enable or disable the progress bar.
        """
        prev = Settings.use_pbar
        Settings.use_pbar = enabled
        try:
            yield
        finally:
            Settings.use_pbar = prev