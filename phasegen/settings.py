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
    #: order). The sparse LU reorders ``-T`` into block-triangular form via its strongly-connected-component
    #: condensation (no coalescent transition raises the lineage/block count, so the blocks are the small migration /
    #: recombination cycles) and factors it with ``NATURAL`` column ordering — near-zero-fill block back-substitution
    #: — which moves the dense/sparse crossover down to a few hundred transient states across single-deme (acyclic),
    #: migration, and two-locus spaces. Set to a very large value to always use the dense path, or to 0 to always use
    #: the sparse path.
    closed_form_sparse_min_states: int = 256

    #: State count at or above which the constructed rate matrix is kept sparse instead of dense. The moment code
    #: works with either, so this is purely a memory/speed tradeoff: a dense matrix is faster where it fits but costs
    #: ``n_states**2`` memory, which becomes prohibitive for large state spaces. The default keeps the dense matrix under ~0.5 GB. Set to a very large value to always build dense, or to 0 to always build sparse.
    dense_rate_matrix_max_states: int = 8000

    #: Maximum number of states the construction will build before aborting with a :class:`MemoryError`. This guards
    #: against a prohibitively large state space (which grows steeply with the sample size; e.g. the single-deme
    #: Raise it if you have the memory for a larger space.
    max_state_space_size: int = 1_000_000

    #: Upper quantile used as the default right end of CDF/PDF/quantile plots. The plot grid runs from 0 to this
    #: quantile so the view is not stretched by a heavy upper tail (mean + many standard deviations can extend far
    #: past where the mass is, especially for skewed distributions). Lower it to zoom in on the bulk, raise it
    #: (towards 1) to show more of the tail.
    plot_endpoint_quantile: float = 0.9

    #: Default number of grid points for the 1D distribution-function plots (cdf / pdf / quantile curves). Raise it
    #: for smoother curves, lower it for faster plotting. The 2D joint heatmaps/surfaces keep their own (coarser)
    #: per-axis resolution for performance.
    plot_n_grid: int = 200

    #: How the **1D** cached curve representation (``cdf_curve`` / ``pdf_curve`` and the quantile-curve inversion
    #: behind the plots and any many-query application use) is built, for univariate / marginal / conditional reward
    #: distributions. ``'cos'`` (default) uses the fast Fourier-cosine series: ~13x quicker to build than de Hoog, with
    #: a clean finite-difference-derived PDF, at the cost of a small (~1% sub-grid) near-origin bias on sharp / heavy-
    #: tailed features (an SFS bin's near-zero peak). ``'dehoog'`` builds an accurate monotone PCHIP spline through de
    #: Hoog Laplace-inversion values at adaptive points -- accurate everywhere (no Gibbs ringing) but ~13x slower to
    #: build -- as the high-precision opt-in. The exact per-point ``cdf()`` / ``pdf()`` (de Hoog) are unaffected.
    inversion_method: str = 'cos'

    #: How the **2D** joint density / CDF callables (``JointRewardDistribution.pdf`` / ``cdf``) are inverted.
    #: ``'cos'`` (default) uses the fast 2D Fourier-cosine expansion (built once, then cheap per point); ``'dehoog'``
    #: uses the accurate per-point nested inversion (inner Gaver-Stehfest, outer de Hoog), which has no fixed-window
    #: bias for skewed multi-epoch rewards but costs a full inversion per point. It defaults to ``'cos'`` (unlike the
    #: 1D default) because the 2D de Hoog samples the CDF on a grid and is slow; pass ``exact=True`` / ``exact=False``
    #: to a call to override per query. The de Hoog *density* is the mixed derivative of a spline through the (clean)
    #: nested-de-Hoog box CDF on an adaptive grid (mirroring the 1D ``pdf_curve``), never a direct density inversion
    #: (which is numerically unusable). The 2D *plots* always use the fast cosine grid unless ``exact=True`` is passed.
    inversion_method_2d: str = 'cos'

    #: Relative tolerance for building the accurate de Hoog + monotone-spline curve representation (the default
    #: ``cdf_curve`` / ``pdf_curve`` and the quantile-by-bisection on it). The adaptive build refines until the
    #: spline matches the de Hoog inversion to about this fraction of the range, then every query is a cheap spline
    #: evaluation. Tighter -> more accurate (and more de Hoog solves in the one-time build); the default ~1e-3 gives a
    #: CDF / quantile accuracy of a few 1e-5. Distinct from :attr:`plot_adaptive_tol`, which only sets the (coarser)
    #: visual grid of the per-point ``exact=True`` plots.
    inversion_tol: float = 1e-3

    #: Relative tolerance for the adaptive plot grid used by the exact (de Hoog) ``cdf()`` / ``pdf()`` plots. The grid
    #: starts coarse and bisects any interval whose midpoint deviates from the chord by more than this fraction of the
    #: curve's range, concentrating the expensive de Hoog evaluations where the curve bends (e.g. the near-zero atom
    #: spike of an SFS bin) instead of wasting them on flat tails. Lower it for finer curves, raise it for fewer
    #: evaluations; the point count is still capped at :attr:`plot_n_grid`.
    plot_adaptive_tol: float = 2e-3

    #: Degree of the de Hoog numerical Laplace inversion used by the exact per-point ``cdf()`` / ``pdf()`` (and
    #: ``exact=True`` plots). The inversion evaluates the transform at ``2 * degree + 1`` contour nodes (each a linear
    #: solve), so the cost is linear in the degree. Accuracy is *non-monotonic*: it improves up to ~15 (near machine
    #: precision) and then degrades as the ill-conditioned QD recurrence amplifies roundoff at fixed precision. The
    #: default of 15 is the sweet spot -- both more accurate and faster than mpmath's own default (20). Lower it
    #: (e.g. 8-10) for a further speed-up at still-excellent accuracy (~1e-9 to 1e-11).
    dehoog_degree: int = 15

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