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
    #: Whether to evaluate the mean site-frequency spectrum of a single population and locus under the standard
    #: coalescent on the lineage-counting state space, see :meth:`PhaseTypeDistribution.moment()
    #: <phasegen.distributions.PhaseTypeDistribution.moment>`.
    flatten_block_counting: bool = True

    #: Whether to show a progress bar for long-running operations.
    use_pbar: bool = False

    #: Whether to allow parallel computation over worker processes. Set to ``False`` to run everything sequentially,
    #: e.g. for a complete stack trace when debugging.
    parallelize: bool = True

    #: Whether to balance the Van Loan matrix by a diagonal similarity before exponentiation, see
    #: :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
    regularize: bool = True

    #: Whether to cache the rate matrix for different epochs which increases performance.
    cache_epochs: bool = True

    #: Whether to memoize cached properties and results. Set to ``False`` to recompute on every access when debugging.
    #: Distinct from :attr:`cache_epochs`.
    cache: bool = True

    #: Whether to use the numba-accelerated state-space construction when numba is available. Set to ``False`` to
    #: force the pure-Python construction path.
    use_numba: bool = True

    #: Matrix dimension at or above which a matrix exponential is applied to a vector by the sparse action algorithm
    #: and not formed densely. It is compared against the Van Loan dimension for moments (see
    #: :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`), against the
    #: number of states for the tree-height distribution functions, and against the number of transient states for
    #: the multi-epoch mutational configurations. The result is unchanged. Set to 0 or very large to force either path.
    expm_action_min_dim: int = 1500

    #: Whether to evaluate moments until absorption with the Green's matrix of the unbounded last epoch, see
    #: :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`. Set to ``False``
    #: to use the matrix exponential up to the estimated absorption time.
    closed_form_last_epoch: bool = True

    #: Number of transient states at or above which linear solves with the transient block of the last epoch use a
    #: sparse block-triangular LU factorization and not a dense one. It applies to the closed-form moments (see
    #: :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`), the last-epoch
    #: solve of the Laplace transform and the multi-epoch mutational configurations. The result is unchanged. Set to 0
    #: to always use the sparse path, or very large to always use the dense path.
    closed_form_sparse_min_states: int = 256

    #: State count at or above which the rate matrix is stored sparse. A dense matrix is faster but needs
    #: ``n_states**2`` memory, about 0.5 GB at the default.
    dense_rate_matrix_max_states: int = 8000

    #: Maximum number of states to construct before raising :class:`MemoryError`. Raise it if memory permits.
    max_state_space_size: int = 1_000_000

    #: Maximum number of trajectories :meth:`PhaseTypeDistribution.sample()
    #: <phasegen.distributions.PhaseTypeDistribution.sample>` simulates per batch, bounding peak memory. A fixed seed
    #: yields different draws for different batch sizes, all from the same distribution. Set to ``None`` to disable
    #: batching.
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

    #: Degree :math:`D` of the de Hoog Laplace inversion, which evaluates the transform at :math:`2D + 1` points, as
    #: described at :class:`~phasegen.distributions.RewardDistribution`. The cost is linear in the degree.
    dehoog_degree: int = 15

    #: CDF level above which the grid of an accumulated-reward distribution carries exact de Hoog values in place of
    #: the cosine expansion, as described at :class:`~phasegen.distributions.RewardDistribution`. Set to ``None`` to
    #: use the cosine expansion throughout.
    dehoog_tail_quantile: Optional[float] = 0.98

    #: Whether to log a warning when a numerical inversion looks imprecise, such as a non-monotone cosine CDF (see
    #: :class:`~phasegen.distributions.RewardDistribution`). Set to ``False`` to silence these checks.
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
