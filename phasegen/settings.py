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

    #: Whether to parallelize phase-type computations across multiple CPU cores.
    #: This may improve performance in some cases, but can also be detrimental due to
    #: inter-process data copying and can lead to hanging processes.
    parallelize: bool = False

    #: Whether to regularize the intensity matrix for numerical stability.
    regularize: bool = True

    #: Whether to cache the rate matrix for different epochs which increases performance.
    cache_epochs: bool = True

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
    #: This is exact and substantially faster (it never forms the dense matrix exponential and avoids the
    #: absorption-time heuristic). It applies only when absorption is almost sure; otherwise the code falls back to
    #: the matrix-exponential path. Within the closed form, the transient sub-generator is factored with a dense LU
    #: below, and a sparse LU at or above, a transient-state count of ``_CLOSED_FORM_SPARSE_MIN_N`` (this is a
    #: separate crossover from :attr:`expm_action_min_dim`, which governs the matrix-exponential path).
    #: Off by default: the per-solve speedup is real, but the current per-call setup overhead (the
    #: absorption-certainty check and transient-block extraction) makes it a net slowdown across many small moments.
    #: Enable it for workloads dominated by large single moments-to-absorption.
    closed_form_last_epoch: bool = False

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