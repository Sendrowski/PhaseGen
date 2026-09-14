"""
Settings for the PhaseGen application.
"""
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class _SettingsMeta(type):
    """
    Metaclass rejecting assignment to a setting that does not exist.

    Settings are class-level attributes, so a mistyped or removed name would otherwise bind a new attribute that
    nothing reads, leaving the caller to believe a flag was set when the behaviour is unchanged.
    """

    def __setattr__(cls, name: str, value: Any):
        """
        Set a setting, rejecting names that are not declared on the class.

        :param name: Name of the setting.
        :param value: Value to assign.
        :raises AttributeError: If no setting of that name exists.
        """
        if not name.startswith('_') and not hasattr(cls, name):
            available = ', '.join(sorted(n for n in vars(cls) if not n.startswith('_')))
            raise AttributeError(
                f"{cls.__name__} has no setting {name!r}, so assigning it would have no effect. "
                f"Available settings: {available}."
            )

        super().__setattr__(name, value)


class Settings(metaclass=_SettingsMeta):
    """
    Global configuration flags governing caching, state-space construction, sampling, and the numerical backends.
    The attributes are class-level and read directly (e.g. ``Settings.use_pbar = True``). Assigning a name that is
    not declared here raises :class:`AttributeError`.
    """
    #: Whether to flatten the block-counting state space onto the lineage-counting one with adjusted rewards where
    #: possible, which can substantially speed up computations.
    flatten_block_counting: bool = True

    #: Whether to show a progress bar for long-running operations.
    use_pbar: bool = False

    #: Whether to allow parallel computation over worker processes. Set to ``False`` to run everything sequentially,
    #: e.g. for a complete stack trace when debugging.
    parallelize: bool = True

    #: Whether to regularize the intensity matrix for numerical stability.
    regularize: bool = True

    #: Whether to cache the rate matrix for different epochs which increases performance.
    cache_epochs: bool = True

    #: Whether to memoize cached properties and results. Set to ``False`` to recompute on every access when debugging.
    #: Distinct from :attr:`cache_epochs`.
    cache: bool = True

    #: Whether to use the numba-accelerated state-space construction when numba is available. Set to ``False`` to
    #: force the pure-Python construction path.
    use_numba: bool = True

    #: Van Loan matrix dimension at or above which moments use the sparse matrix-exponential action instead of the
    #: dense propagator. Set to 0 or very large to force either path.
    expm_action_min_dim: int = 1500

    #: Whether to evaluate the unbounded last epoch of a moment in closed form, by a linear solve with the transient
    #: sub-generator. Falls back to the matrix exponential when absorption is not almost sure. Set to ``False`` to
    #: validate against that path.
    closed_form_last_epoch: bool = True

    #: Transient-state count at or above which the closed-form last epoch uses a sparse LU instead of a dense one,
    #: changing cost but not result. Set to 0 to always use the sparse path, or very large to always use the dense path.
    closed_form_sparse_min_states: int = 256

    #: State count at or above which the rate matrix is stored sparse. A dense matrix is faster but needs
    #: ``n_states**2`` memory, about 0.5 GB at the default.
    dense_rate_matrix_max_states: int = 8000

    #: Maximum number of states to construct before raising :class:`MemoryError`. Raise it if memory permits.
    max_state_space_size: int = 1_000_000

    #: Maximum number of trajectories :meth:`PhaseTypeDistribution.sample()
    #: <phasegen.distributions.PhaseTypeDistribution.sample>` simulates per batch, bounding peak memory without
    #: changing the result. Set to ``None`` to disable batching.
    sample_batch_size: Optional[int] = 1_000_000

    #: Upper quantile used as the default right end of distribution-function plots.
    plot_endpoint_quantile: float = 0.9

    #: Default number of grid points for 1D distribution-function plots.
    plot_n_grid: int = 200

    #: Default number of grid points per axis for the heatmap of a joint density.
    plot_joint_pdf_n_grid: int = 120

    #: Default number of grid points per axis for the 3D surface of a joint density.
    plot_joint_pdf_surface_n_grid: int = 80

    #: Default number of grid points per axis for the heatmap and 3D surface of a joint CDF.
    plot_joint_cdf_n_grid: int = 60

    #: Default right end of the time axis of demography plots.
    plot_demography_end_time: float = 10.0

    #: Default number of time points of demography plots.
    plot_demography_n_grid: int = 1000

    #: Quantile of the inferred tree height used as the default right end of the time axis of inference plots.
    plot_inference_quantile: float = 0.99

    #: Default number of time points of inference plots.
    plot_inference_n_grid: int = 100

    #: Degree of the de Hoog Laplace inversion behind the exact CDF and density. The cost is linear in the degree.
    #: Accuracy peaks near the default and degrades above it.
    dehoog_degree: int = 15

    #: Quantile above which the CDF grid of an accumulated-reward distribution switches from the cosine fit to exact
    #: de Hoog nodes, which resolve the far tail. Set to ``None`` to use the cosine fit throughout.
    dehoog_tail_quantile: Optional[float] = 0.98

    #: Whether to log a warning when a numerical inversion looks imprecise: a negative density, a non-monotone CDF, or
    #: a violated law of total expectation. Set to ``False`` to silence these checks.
    check_inversions: bool = True

    @staticmethod
    @contextmanager
    def set_pbar(enabled: bool = True) -> Iterator[None]:
        """
        Context manager to temporarily enable or disable the progress bar.

        :param enabled: Whether to show the progress bar within the context.
        """
        prev = Settings.use_pbar
        Settings.use_pbar = enabled
        try:
            yield
        finally:
            Settings.use_pbar = prev
