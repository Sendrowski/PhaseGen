"""
Empirical distributions estimated from simulated genealogies -- via msprime (:class:`MsprimeCoalescent`) or
PhaseGen's own trajectory sampler (:class:`SampledCoalescent`) -- together with the containers that compute
statistics from the sampled realisations.
"""

import logging
import math
from ..caching import cached_property
from typing import Generator, List, Callable, Tuple, Dict, Iterator, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from ..coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from ..demography import Demography
from ..initial import InitialDistribution
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..settings import Settings
from ..spectrum import SFS, TwoSFS, JointSFS, TwoLocusSFS
from ..utils import parallelize

from .base import DensityAwareDistribution, CumulativeDistributionFunction, DensityFunction, DistributionFunction, \
    QuantileFunction
from .spectra import FoldedSFSDistribution, SFSDistribution, TajimaSFSMixin, UnfoldedSFSDistribution
from .mutation_configs import MutationConfig, MutationLayout
from .coalescent import AbstractCoalescent, Coalescent

if TYPE_CHECKING:
    import msprime
    import tskit
    from ..visualization import _CurveData

logger = logging.getLogger('phasegen')

#: The largest Beta-coalescent alpha msprime accepts.
_MSPRIME_BETA_ALPHA_MAX = 1.991


class EmpiricalJointSFSDistribution:  # pragma: no cover
    r"""
    Empirical joint site-frequency spectrum, built by
    :meth:`JointSFSDistribution.to_empirical() <phasegen.distributions.JointSFSDistribution.to_empirical>` or by
    :class:`~phasegen.distributions.MsprimeCoalescent`. It holds the raw sample moments

    .. math::

        \hat M_o(\mathbf{c}) = \frac{1}{N} \sum_{m=1}^{N} L_{m\mathbf{c}}^o, \qquad o = 1, 2, 3,

    where :math:`L_{m\mathbf{c}}` is the branch length of replicate :math:`m = 1, \dots, N` whose descendants number
    :math:`c_p` in population :math:`p`, for the descendant vector :math:`\mathbf{c} = (c_0, \dots, c_{P-1})` over
    :math:`P` populations. :attr:`mean`, :attr:`m2` and :attr:`m3` return :math:`\hat M_1`, :math:`\hat M_2` and
    :math:`\hat M_3`, and :attr:`var` is :math:`\hat M_2 - \hat M_1^2`.
    """

    def __init__(self, moments: np.ndarray, samples: np.ndarray = None, n_samples: int = None) -> None:
        """
        Initialize the distribution.

        :param moments: Raw moments per descendant configuration of orders one to three, stacked along the first
            axis, of shape ``(3, n_0 + 1, ..., n_{P-1} + 1)``.
        :param samples: Optional per-replicate joint SFS branch lengths, of shape ``(N, n_0 + 1, ...)``, possibly a
            capped subset of the replicates the moments were averaged over.
        :param n_samples: The number of replicates :math:`N` the moments were averaged over. Defaults to the length
            of ``samples``, and must be given when ``samples`` is a capped subset.
        """
        #: Non-central moments per descendant configuration, indexed by order minus one.
        self._moments: np.ndarray = np.asarray(moments)

        #: Joint SFS branch lengths, one row per simulated replicate, of shape
        #: ``(n_samples, n_0 + 1, ..., n_{P-1} + 1)``, possibly a capped subset of the replicates. ``None`` once freed
        #: for serialization.
        self.samples: np.ndarray | None = None if samples is None else np.asarray(samples)

        #: Number of replicates the moments were averaged over, retained when the samples are freed so that it is
        #: recorded in a serialized comparison. Not ``len(samples)`` when the samples are a capped subset.
        self.n_samples: Optional[int] = (
            n_samples if n_samples is not None else (None if samples is None else np.asarray(samples).shape[0]))

        #: Cached full-grid joint surface ground truth: ``[(config_a, config_b, xs, ys, cdf_grid, pdf_grid), ...]``.
        self._joint_surface: list = []

    def _cache_joint_surface(self, pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], n_grid: int = 25,
                            q_max: float = 0.95) -> None:
        """Cache the ground truth of ``EmpiricalPhaseTypeSFSDistribution._cache_joint_surface`` per configuration
        pair."""
        s = self.samples
        n = s.shape[0]
        self._joint_surface = []
        for ca, cb in pairs:
            li, lj = s[(slice(None),) + tuple(ca)], s[(slice(None),) + tuple(cb)]
            xs = np.linspace(0.0, float(np.quantile(li, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(lj, q_max)), n_grid)
            cdf = ((li[:, None] <= xs[None, :]).astype(float).T @ (lj[:, None] <= ys[None, :]).astype(float)) / n
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._joint_surface.append((tuple(ca), tuple(cb), xs, ys, cdf, pdf))

    def _drop(self) -> None:
        """Drop the (large) per-replicate samples once the joint ground truth has been cached."""
        self.samples = None

    @property
    def mean(self) -> JointSFS:
        """
        Mean of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[0])

    @property
    def m2(self) -> JointSFS:
        """
        Second (non-central) moment of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[1])

    @property
    def m3(self) -> JointSFS:
        """
        Third (non-central) moment of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[2])

    @property
    def var(self) -> JointSFS:
        """
        Variance of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[1] - self._moments[0] ** 2)

    @property
    def data(self) -> np.ndarray:
        """
        The mean joint site-frequency spectrum array.
        """
        return self._moments[0]


class _EmpiricalFunction:  # pragma: no cover
    """Mixin building the plot data of an empirical function object: one curve for a sample vector (a scalar
    distribution), one per polymorphic bin for a replicate-by-bin sample matrix (a spectrum)."""

    def _empirical_curves(
            self,
            grid: np.ndarray | None,
            bins: Sequence[int] | None,
            n_points: int | None
    ) -> '_CurveData':
        """
        The curves of this function over ``grid``, by default over :attr:`Settings.plot_n_grid` points up to the
        largest :attr:`Settings.plot_endpoint_quantile` sample quantile of the included bins. The density is a cell
        average, so its default cells are coarsened with the sample size, and its curves are drawn at the cell centres.

        :param grid: Points to evaluate at (the left cell edges for a density), ``None`` for the default grid.
        :param bins: Bins to include for a spectrum, ``None`` for all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves.
        """
        from ..visualization import _CurveData

        samples = np.asarray(self._distribution.samples)
        per_bin = samples.ndim == 2
        columns = [int(i) for i in (self._distribution._polymorphic_bins() if bins is None else np.atleast_1d(bins))] \
            if per_bin else []

        included = samples[:, columns] if per_bin else samples
        x = DistributionFunction._default_grid(
            self.kind, grid, n_points, lambda: np.quantile(included, Settings.plot_endpoint_quantile, axis=0).max()
        )

        if self.kind == 'pdf':
            if grid is None:
                # a cell holding a handful of replicates comes out as noise, so the cells grow with the sample size
                x = np.linspace(x[0], x[-1], int(np.clip(np.sqrt(samples.shape[0]), 20, 100)))
            values = np.asarray(self(x))
            edges = np.append(x, 2 * x[-1] - x[-2])
            x = edges[:-1] + np.diff(edges) / 2
        else:
            values = np.asarray(self(x))

        y = values.T[columns] if per_bin else values[None]
        name = dict(pdf='PDF', cdf='CDF', quantile='quantile function')[self.kind]

        return _CurveData(
            x=x,
            y=y,
            labels=[str(i) for i in columns] if per_bin else [''],
            xlabel='q' if self.kind == 'quantile' else 't',
            ylabel=dict(pdf='f(t)', cdf='F(t)', quantile='quantile')[self.kind],
            title=f'SFS bin {name}s' if per_bin else name[0].upper() + name[1:],
            legend_title='bin' if per_bin else None
        )

    def plot(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the empirical function, one curve per polymorphic bin for a spectrum. The density is drawn at the cell
        centres.

        :param ax: Axes to plot on.
        :param t: Points to evaluate at, the left cell edges for a density. By default,
            :attr:`~phasegen.settings.Settings.plot_n_grid` points (for a density between 20 and 100 cells, growing
            with the square root of the sample size) up to the largest
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` sample quantile.
        :param bins: Bins to plot for a spectrum. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid of a CDF.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(t=t, bins=bins, n_points=n_points), file=file,
                                         show=show, clear=clear, label=label, title=title, **kwargs)


class _EmpiricalCumulativeDistributionFunction(_EmpiricalFunction, CumulativeDistributionFunction):  # pragma: no cover
    """The empirical CDF of ``EmpiricalDistribution`` (see its class docstring), per column for 2-D samples, of shape
    ``(len(t), n + 1)`` for a spectrum."""

    def __call__(self, t) -> 'np.ndarray':
        # sort along the replicate axis (axis 0); for 2-D (per-bin) samples this must not be the default last axis,
        # which would sort across bins within a replicate and produce a meaningless ECDF
        samples = self._distribution.samples
        x = np.sort(samples, axis=0)
        y = np.arange(1, len(samples) + 1) / len(samples)

        if x.ndim == 1:
            return np.interp(t, x, y, left=0.0)

        if x.ndim == 2:
            return np.stack([np.interp(t, x_, y, left=0.0) for x_ in x.T], axis=-1)

        raise ValueError("Samples must be 1 or 2 dimensional.")

    def _plot_data(self, t: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The empirical CDF curves :meth:`plot` draws, one per polymorphic bin for a spectrum.

        :param t: Points to evaluate at. By default, :attr:`Settings.plot_n_grid` points up to the largest
            :attr:`Settings.plot_endpoint_quantile` sample quantile.
        :param bins: Bins to include for a spectrum, all polymorphic bins by default.
        :param n_points: Number of points of the default grid.
        :return: The curves.
        """
        return self._empirical_curves(t, bins, n_points)


class _EmpiricalQuantileFunction(_EmpiricalFunction, QuantileFunction):  # pragma: no cover
    """The sample quantile of ``EmpiricalDistribution`` (see its class docstring), per column for 2-D samples."""

    def __call__(self, q) -> 'np.ndarray':
        # over the replicate axis (axis 0); for 2-D (per-bin) samples this gives one quantile per bin (shape
        # ``(len(q), n_bins)`` for an array ``q``), as the default flattening would mix bins together
        return np.quantile(self._distribution.samples, q=q, axis=0)

    def _plot_data(self, q: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The empirical quantile curves :meth:`plot` draws, one per polymorphic bin for a spectrum.

        :param q: Probabilities to evaluate at. By default, :attr:`Settings.plot_n_grid` points from
            ``1 - Settings.plot_endpoint_quantile`` to :attr:`Settings.plot_endpoint_quantile`.
        :param bins: Bins to include for a spectrum, all polymorphic bins by default.
        :param n_points: Number of points of the default grid.
        :return: The curves.
        """
        return self._empirical_curves(q, bins, n_points)

    def plot(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the empirical quantile function (value versus probability), one curve per polymorphic bin for a spectrum.

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate at. By default, :attr:`~phasegen.settings.Settings.plot_n_grid` points
            from ``1 - Settings.plot_endpoint_quantile`` to :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param bins: Bins to plot for a spectrum. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, bins=bins, n_points=n_points), file=file,
                                         show=show, clear=clear, label=label, title=title, **kwargs)


class _EmpiricalDensityFunction(_EmpiricalFunction, DensityFunction):  # pragma: no cover
    """The cell-average density of ``EmpiricalDistribution`` (see its class docstring), per column for 2-D samples,
    of shape ``(len(t), n + 1)`` for a spectrum. ``Comparison._cell_average`` integrates the exact density over the
    same cells, so both sides estimate the same functional."""

    def __call__(self, t) -> 'np.ndarray':
        samples = self._distribution.samples
        t = np.atleast_1d(np.asarray(t, dtype=float))

        if t.size < 2:
            raise ValueError("The empirical density is a cell average, so it needs a grid of at least two points.")

        edges = np.append(t, 2 * t[-1] - t[-2])
        widths = np.diff(edges)

        if samples.ndim == 1:
            return self._cell_density(samples, edges, widths)

        if samples.ndim == 2:
            return np.stack([self._cell_density(s, edges, widths) for s in samples.T], axis=-1)

        raise ValueError("Samples must be 1 or 2 dimensional.")

    def _plot_data(self, t: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The cell-average density curves :meth:`plot` draws at the cell centres, one per polymorphic bin for a spectrum.

        :param t: Left edges of the cells. By default, between 20 and 100 cells, growing with the square root of the
            sample size, up to the largest :attr:`Settings.plot_endpoint_quantile` sample quantile.
        :param bins: Bins to include for a spectrum, all polymorphic bins by default.
        :param n_points: Unused, as the default cells follow the sample size.
        :return: The curves.
        """
        return self._empirical_curves(t, bins, n_points)

    @staticmethod
    def _cell_density(samples: np.ndarray, edges: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """The cell-average density of one sample vector. Normalised by the *total* replicate count, not by the
        positive one, so the atom at 0 lowers the sub-density instead of being redistributed over the cells."""
        counts, _ = np.histogram(samples[samples > 0], bins=edges)

        return counts / samples.size / widths


class EmpiricalDistribution(DensityAwareDistribution):  # pragma: no cover
    r"""
    Probability distribution estimated from :math:`N` realisations :math:`Y_1, \dots, Y_N`, such as the statistics
    of genealogies simulated by :class:`~phasegen.distributions.MsprimeCoalescent` or the accumulated rewards drawn by
    :meth:`PhaseTypeDistribution.sample() <phasegen.distributions.PhaseTypeDistribution.sample>`, which
    :class:`~phasegen.distributions.SampledCoalescent` collects for every statistic of a coalescent.
    Every estimator applies to each entry of a spectrum separately, and :math:`Y_{(1)} \le \dots \le Y_{(N)}` are the
    order statistics. The moments are described at
    :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.

    - :attr:`cdf`: linear interpolation between the points :math:`(Y_{(m)}, m / N)`, zero below :math:`Y_{(1)}` and
      one from :math:`Y_{(N)}` on. At an atom it takes the value after the jump.
    - :attr:`quantile`: the linearly interpolated sample quantile, :math:`Y_{(\eta)}` at the position
      :math:`\eta = (N - 1)\, q + 1`.
    - :attr:`pdf`: for a grid :math:`x_0 < x_1 < \dots` of left cell edges, the fraction of realisations in each cell
      divided by its width,

      .. math::

          \hat f_g = \frac{\#\{m : Y_m > 0,\ x_g \le Y_m < x_{g+1}\}}{N\, (x_{g+1} - x_g)}.

      Zero realisations are excluded but counted in :math:`N`, so the cells carry the mass :math:`1 - p_0`.
    - :attr:`cov` and :attr:`corr`: the sample covariance with normalisation :math:`1 / N`, so that :attr:`var` is
      its diagonal, and the correlation derived from it. Entries without variance are set to zero.
    """
    # the cdf / pdf / quantile evaluation lives on these sample-based function objects; the distribution supplies the
    # ``samples`` they read (the per-bin spectrum case is handled by the same objects, on 2-D samples)
    _cdf_function = _EmpiricalCumulativeDistributionFunction
    _pdf_function = _EmpiricalDensityFunction
    _quantile_function = _EmpiricalQuantileFunction

    def __init__(self, samples: np.ndarray | list) -> None:
        """
        Initialize the distribution from the given realisations.

        :param samples: The realisations, of shape ``(N,)``, or ``(N, n + 1)`` for a spectrum.
        """
        super().__init__()

        self._cache = None

        #: Sampled values, one row per replicate, of shape ``(n_samples,)``, or ``(n_samples, n + 1)`` for a spectrum.
        #: ``None`` once freed for serialization.
        self.samples: np.ndarray | None = np.array(samples, dtype=float)

        #: Number of samples, retained when the samples are freed so that it is recorded in a serialized comparison.
        self.n_samples: int = self.samples.shape[0]

        #: Standard error of each moment statistic, estimated from blocks of the samples, retained when they are freed.
        self._standard_errors: dict = {}

    def _polymorphic_bins(self) -> range:
        """
        The polymorphic frequency classes of a spectrum sample, ``1, ..., n - 1``.

        :return: The frequency classes.
        """
        return range(1, self.samples.shape[1] - 1)

    def _touch(self, t: np.ndarray) -> None:
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super()._touch()

        # probability grid for the quantile function (kept off the extreme tails, where the empirical quantile is
        # noisy and -- for SFS bins with an atom at 0 -- flat at 0 below the atom mass)
        q = np.linspace(0.05, 0.95, 50)

        self._cache = dict(
            t=t,
            cdf=self.cdf(t),
            pdf=self.pdf(t),
            q=q,
            quantile=self.quantile(q)
        )

        self._cache_standard_errors()

    #: Statistics :meth:`_cache_standard_errors` estimates a standard error for.
    _STANDARD_ERROR_STATISTICS = ('mean', 'var', 'm2', 'm3', 'm4', 'cov', 'corr')

    def _cache_standard_errors(self, n_blocks: int = 100) -> None:
        """
        Cache the standard error of each moment statistic so that it survives ``_drop``. The samples are split into
        ``B`` disjoint blocks of equal size, the statistic is evaluated on each, and the standard error at the full
        sample size is the standard deviation across blocks divided by ``sqrt(B)``, valid for nonlinear statistics.

        :param n_blocks: Number of blocks ``B``, reduced to half the sample size for small samples. Nothing is cached
            below two blocks.
        """
        n_blocks = min(n_blocks, self.samples.shape[0] // 2)

        if n_blocks < 2:
            return

        blocks = self.samples[:self.samples.shape[0] // n_blocks * n_blocks]
        blocks = blocks.reshape(n_blocks, -1, *self.samples.shape[1:])

        # the base class' statistics are plain numpy; the subclasses only wrap the identical numerics in an SFS type
        stats = [EmpiricalDistribution(block) for block in blocks]

        self._standard_errors = {}
        for name in self._STANDARD_ERROR_STATISTICS:
            if name in ('cov', 'corr') and self.samples.ndim == 1:
                continue  # a 1-D sample has no covariance/correlation: corrcoef is the constant 1, SE a bogus 0
            values = np.array([np.asarray(getattr(s, name), dtype=float) for s in stats])
            self._standard_errors[name] = np.std(values, axis=0) / np.sqrt(n_blocks)

    def _drop(self) -> None:
        """
        Drop simulated samples.
        """
        self.samples = None

    @cached_property
    def mean(self) -> float | np.ndarray:
        """
        Sample mean, see :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self) -> float | np.ndarray:
        """
        Sample variance, see :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return np.var(self.samples, axis=0)

    @cached_property
    def m2(self) -> float | np.ndarray:
        """
        Second raw sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return np.mean(self.samples ** 2, axis=0)

    @cached_property
    def m3(self) -> float | np.ndarray:
        """
        Third raw sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return np.mean(self.samples ** 3, axis=0)

    @cached_property
    def m4(self) -> float | np.ndarray:
        """
        Fourth raw sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return np.mean(self.samples ** 4, axis=0)

    @cached_property
    def cov(self) -> float | np.ndarray:
        """
        Sample covariance matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.cov(self.samples, rowvar=False, bias=True))

    @cached_property
    def corr(self) -> float | np.ndarray:
        """
        Sample correlation matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.corrcoef(self.samples, rowvar=False))

    def moment(self, k: int, center: bool = True) -> float | np.ndarray:
        r"""
        The :math:`k`-th sample moment of the realisations :math:`Y_1, \dots, Y_N` of
        :class:`~phasegen.distributions.EmpiricalDistribution`,

        .. math::

            \frac{1}{N} \sum_{m=1}^{N} (Y_m - \hat\mu)^k,

        with :math:`\hat\mu` the sample mean for a central moment (``center=True`` and :math:`k \ge 2`) and
        :math:`\hat\mu = 0` for a raw moment. :attr:`var` is the central moment of order two, and :attr:`m2`,
        :attr:`m3` and :attr:`m4` are raw moments.

        :param k: Order :math:`k \ge 1` of the moment.
        :param center: Whether to center the moment around the sample mean :math:`\hat\mu`.
        :return: The :math:`k`-th moment, per entry for a spectrum.
        """
        samples = self.samples - np.mean(self.samples, axis=0) if (center and k > 1) else self.samples

        return np.mean(samples ** k, axis=0)


class EmpiricalSFSDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    Empirical site-frequency spectrum of one deme, with the estimators of
    :class:`~phasegen.distributions.EmpiricalDistribution` applied per frequency class.

    The following example estimates the mean spectrum of the branches residing in ``pop_0`` from 1000 sampled
    trajectories.

    ::

        coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
        ))

        sfs_0 = coal.sfs.to_empirical(1000, seed=1).demes['pop_0'].mean
    """

    #: Whether the spectrum is folded. Static for backward compatibility.
    folded: bool = False

    def __init__(self, samples: np.ndarray | list, folded: bool = False) -> None:
        """
        Initialize the distribution from the given realisations.

        :param samples: The sampled spectra, of shape ``(N, n + 1)``.
        :param folded: Whether the spectrum is folded.
        """
        super().__init__(samples)

        self.folded = folded

    def _polymorphic_bins(self) -> range:
        """
        The polymorphic frequency classes, ``1, ..., n // 2`` for a folded spectrum and ``1, ..., n - 1`` otherwise.

        :return: The frequency classes.
        """
        n = self.samples.shape[1] - 1

        return range(1, n // 2 + 1) if self.folded else range(1, n)

    @cached_property
    def mean(self) -> SFS:
        """
        Sample mean spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().mean)

    @cached_property
    def var(self) -> SFS:
        """
        Sample variance spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().var)

    @cached_property
    def m2(self) -> SFS:
        """
        Second raw sample moment spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().m2)

    @cached_property
    def cov(self) -> TwoSFS:
        """
        Sample covariance matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        return TwoSFS(super().cov)

    @cached_property
    def corr(self) -> TwoSFS:
        """
        Sample correlation matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        return TwoSFS(super().corr)


class DictContainer(dict):  # pragma: no cover
    """
    Empirical marginal distributions keyed by deme name or locus index, with their covariance and correlation matrices
    as ``cov`` and ``corr``, ordered as the keys.
    """

    #: Covariance matrix of the marginals.
    cov: Optional[np.ndarray] = None

    #: Correlation matrix of the marginals.
    corr: Optional[np.ndarray] = None

    @classmethod
    def _of_spectra(cls, dists: dict, data: np.ndarray) -> 'DictContainer':
        """
        The container of marginal spectra, with the covariance and correlation of each frequency class across the
        marginals, of shape ``(k, k, n + 1)``, ``nan`` where a marginal has zero variance.

        :param dists: The marginal spectra.
        :param data: Their branch lengths, of shape ``(k, N, n + 1)``.
        :return: The container.
        """
        centered = data - data.mean(axis=1, keepdims=True)
        cov = np.einsum('anj,bnj->abj', centered, centered) / data.shape[1]
        sd = np.sqrt(np.einsum('aaj->aj', cov))

        container = cls(dists)
        container.cov = cov
        with np.errstate(divide='ignore', invalid='ignore'):
            container.corr = cov / (sd[:, None] * sd[None, :])

        return container

    def _index(self, key) -> int:
        """
        The position of a key.

        :param key: Deme name or locus index.
        :return: The position.
        :raises ValueError: If there is no marginal of that key.
        """
        if key not in self:
            raise ValueError(f"There is no marginal distribution {key}.")

        return list(self).index(key)

    def get_cov(self, d1, d2) -> float | np.ndarray:
        """
        Get the covariance between two marginal distributions.

        :param d1: Deme name or locus index of the first marginal distribution.
        :param d2: Deme name or locus index of the second marginal distribution.
        :return: The covariance, one per frequency class for spectra.
        :raises ValueError: If there is no marginal of either key.
        """
        return np.atleast_2d(self.cov)[self._index(d1), self._index(d2)]

    def get_corr(self, d1, d2) -> float | np.ndarray:
        """
        Get the correlation coefficient between two marginal distributions.

        :param d1: Deme name or locus index of the first marginal distribution.
        :param d2: Deme name or locus index of the second marginal distribution.
        :return: The correlation coefficient, one per frequency class for spectra.
        :raises ValueError: If there is no marginal of either key.
        """
        return np.atleast_2d(self.corr)[self._index(d1), self._index(d2)]


class EmpiricalPhaseTypeDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    Empirical distribution of an accumulated reward with per-deme and per-locus breakdowns, built by
    :meth:`PhaseTypeDistribution.to_empirical() <phasegen.distributions.PhaseTypeDistribution.to_empirical>` or by
    :class:`~phasegen.distributions.MsprimeCoalescent`. Its estimators are those of
    :class:`~phasegen.distributions.EmpiricalDistribution`.

    The following example estimates the mean and variance of the tree height of two loci and the mean tree height
    at the first locus from 1000 sampled trajectories.

    ::

        emp = pg.Coalescent(n=3, loci=2, recombination_rate=1).tree_height.to_empirical(1000, seed=1)

        mean, var = emp.mean, emp.var
        height_0 = emp.loci[0].mean
    """
    #: Whether the samples resolve the demes, so that :attr:`demes` is available.
    resolves_demes: bool = True

    #: Whether the total is the sum over loci, so that the per-deme samples summed over loci decompose it. Static for
    #: backward compatibility.
    _locus_additive: bool = True

    #: Cached windowed-conditional ground truth of the locus pairs, see ``_cache_windowed_conditional``. Declared at
    #: class level for payloads serialized without it.
    _loci_windowed_conditional: list = []

    def __init__(
            self,
            samples: np.ndarray | list,
            pops: List[str],
            locus_agg: Callable = lambda x: x.sum(axis=0),
            resolves_demes: bool = True
    ) -> None:
        """
        Create object.

        :param samples: Realisations per locus, deme and replicate, of shape ``(loci, demes, N)``.
        :param pops: List of population names.
        :param locus_agg: Aggregation over the locus axis of the per-locus totals, which sum over demes, forming the
            total. The sum by default, the maximum for the tree height.
        :param resolves_demes: Whether the samples resolve the demes. Otherwise :attr:`demes` raises.
        """
        over_loci = samples.sum(axis=0).astype(float)
        over_demes = samples.sum(axis=1).astype(float)

        total = locus_agg(over_demes).astype(float)

        super().__init__(total)

        #: Population names
        self.pops = pops

        #: Samples by deme and locus
        self._samples = samples

        self.resolves_demes = resolves_demes

        #: Whether the total is the sum over loci, so that the per-deme samples summed over loci decompose it.
        self._locus_additive: bool = over_demes.shape[0] == 1 or np.array_equal(total, over_demes.sum(axis=0))

        #: Cross-locus full-grid joint surface ground truth: ``[(l1, l2, xs, ys, cdf_grid, pdf_grid), ...]``.
        self._loci_joint_surface: list = []

        #: Cached windowed-conditional ground truth of the locus pairs, see ``_cache_windowed_conditional``.
        self._loci_windowed_conditional: list = []

        # zero-variance demes/loci make corrcoef divide by zero; the resulting NaNs are expected here, so
        # silence the benign warning
        with np.errstate(divide='ignore', invalid='ignore'):
            #: Covariance matrix for the demes, of the per-deme samples summed over loci, ``None`` unless the samples
            #: resolve the demes
            self.pops_cov: np.ndarray | None = np.cov(over_loci, bias=True) if resolves_demes else None

            #: Correlation matrix for the demes, of the per-deme samples summed over loci, ``None`` unless the samples
            #: resolve the demes
            self.pops_corr: np.ndarray | None = np.corrcoef(over_loci) if resolves_demes else None

            #: Correlation matrix for the loci
            self.loci_corr: np.ndarray = np.corrcoef(over_demes)

            #: Covariance matrix for the loci
            self.loci_cov: np.ndarray = np.cov(over_demes, bias=True)

    def _touch(self, t: np.ndarray) -> None:
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super()._touch(t)

        if self._defines_demes:
            [d._touch(t) for d in self.demes.values()]
        [l._touch(t) for l in self.loci.values()]

    def _drop(self) -> None:
        """
        Drop simulated samples.
        """
        super()._drop()

        self._samples = None

        if self._defines_demes:
            [d._drop() for d in self.demes.values()]
        [l._drop() for l in self.loci.values()]

    def _cache_standard_errors(self, n_blocks: int = 100) -> None:
        """
        Block standard errors of the totals, as ``EmpiricalDistribution._cache_standard_errors``, and, except for a
        spectrum, of the deme and locus covariance and correlation matrices, keyed ``"demes.cov"``, ``"demes.corr"``,
        ``"loci.cov"`` and ``"loci.corr"``.
        """
        super()._cache_standard_errors(n_blocks)

        if self._samples is None:
            return

        def cov(x: np.ndarray) -> np.ndarray:
            return np.cov(x, bias=True)

        demes = (('demes.cov', self._samples.sum(axis=0), cov),  # (n_demes, n_rep), summed over loci
                 ('demes.corr', self._samples.sum(axis=0), np.corrcoef)) if self._defines_demes else ()
        for key, data, fn in demes + (
            ('loci.cov', self._samples.sum(axis=1), cov),  # (n_loci, n_rep), summed over demes
            ('loci.corr', self._samples.sum(axis=1), np.corrcoef),
        ):
            se = self._matrix_block_standard_error(data, fn, n_blocks)
            if se is not None:
                self._standard_errors[key] = se

    @staticmethod
    def _matrix_block_standard_error(data: np.ndarray, fn, n_blocks: int) -> Optional[np.ndarray]:
        """Standard error of a matrix statistic (``np.cov`` / ``np.corrcoef``) of ``data`` (shape ``(series, reps)``),
        by the same block subsampling :meth:`EmpiricalDistribution._cache_standard_errors` uses: the spread of the
        per-block matrix divided by ``sqrt(n_blocks)``. Returns ``None`` when there is nothing to correlate (fewer
        than two series, e.g. a single deme or single locus) or too few replicates to block."""
        if data.ndim != 2 or data.shape[0] < 2:
            return None

        n_reps = data.shape[1]
        n_blocks = min(n_blocks, n_reps // 2)
        if n_blocks < 2:
            return None

        blocks = data[:, :n_reps // n_blocks * n_blocks].reshape(data.shape[0], n_blocks, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mats = np.array([fn(blocks[:, b, :]) for b in range(n_blocks)])

        return np.std(mats, axis=0) / np.sqrt(n_blocks)

    @cached_property
    def demes(self) -> Dict[str, EmpiricalDistribution]:
        """
        Empirical distribution of each deme, summed over loci, with the deme covariance and correlation matrices as
        ``cov`` and ``corr``.

        :return: Dictionary of distributions.
        :raises ValueError: If the samples do not resolve the demes.
        :raises NotImplementedError: If there are several loci and the total is not their sum, as for the tree height.
        """
        self._check_resolves_demes()

        if not self._locus_additive:
            raise NotImplementedError(
                "Per-deme statistics are not defined for multiple loci when the total is not the sum over loci, as "
                "for the tree height, the maximum over loci. Use total_branch_length.demes, or a single locus."
            )

        demes = DictContainer(
            {pop: EmpiricalDistribution(self._samples.sum(axis=0)[i]) for i, pop in enumerate(self.pops)}
        )

        demes.cov = self.pops_cov
        demes.corr = self.pops_corr

        return demes

    @property
    def _untouched(self) -> Tuple[str, ...]:
        """
        The per-deme statistics when :attr:`demes` is not available.
        """
        return () if self._defines_demes else ('demes',)

    @property
    def _defines_demes(self) -> bool:
        """
        Whether :attr:`demes` is available: the samples resolve the demes and the total is the sum over loci.
        """
        return self.resolves_demes and self._locus_additive

    def _check_resolves_demes(self) -> None:
        """
        :raises ValueError: If the samples do not resolve the demes.
        """
        if not self.resolves_demes:
            raise ValueError(
                "Per-deme statistics of MsprimeCoalescent require the migration history. "
                "Simulate with record_migration=True."
            )

    @cached_property
    def loci(self) -> Dict[int, EmpiricalDistribution]:
        """
        Empirical distribution of each locus, summed over demes, with the locus covariance and correlation matrices as
        ``cov`` and ``corr``.

        :return: Dictionary of distributions.
        """
        loci = DictContainer(
            {i: EmpiricalDistribution(self._samples[i].sum(axis=0)) for i in range(self._samples.shape[0])}
        )

        loci.cov = self.loci_cov
        loci.corr = self.loci_corr

        return loci

    def _locus_samples(self, locus: int) -> np.ndarray:
        """Per-replicate accumulated reward at a single locus (summed over demes), matching :attr:`loci`."""
        return self._samples[locus].sum(axis=0)

    def _cache_loci_joint_surface(self, pairs: List[Tuple[int, int]], n_grid: int = 25, q_max: float = 0.95) -> None:
        """Cache the ground truth of ``EmpiricalPhaseTypeSFSDistribution._cache_joint_surface`` per locus pair."""
        self._loci_joint_surface = []
        for l1, l2 in pairs:
            a, b = self._locus_samples(l1), self._locus_samples(l2)
            n = len(a)
            xs = np.linspace(0.0, float(np.quantile(a, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(b, q_max)), n_grid)
            cdf = ((a[:, None] <= xs[None, :]).astype(float).T @ (b[:, None] <= ys[None, :]).astype(float)) / n
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._loci_joint_surface.append((int(l1), int(l2), xs, ys, cdf, pdf))

    def _cache_windowed_conditional(self, specs: List[tuple], loci: bool = False, n_grid: int = 500,
                                    q_max: float = 0.999) -> None:
        """
        Cache, per conditioning window of a pair, the plain window mean of the other reward (not the local-linear
        mean), its standard error and its step CDF over a grid, as ``[(i, j, on, v, h, n_win, mean, mean_se, ys,
        cdf), ...]``. The comparison averages the exact conditional over the same window, so both sides estimate the
        same functional.

        :param specs: ``(i, j, on, value, half_width)`` windows to cache, the values fixed by the exact marginal.
        :param loci: Whether ``(i, j)`` is a pair of loci, cached as ``_loci_windowed_conditional``, or a pair of
            ``_pair_samples``, cached as ``_windowed_conditional``.
        :param n_grid: Points of the CDF grid.
        :param q_max: Quantile of the windowed samples the grid runs to.
        :raises ValueError: If a window holds no replicates at all.
        """
        cached = []

        for i, j, on, v, h in specs:
            a, b = (self._locus_samples(i), self._locus_samples(j)) if loci else self._pair_samples(i, j)
            cond, other = (a, b) if on == 'a' else (b, a)
            sel = other[np.abs(cond - v) <= h]

            if sel.size == 0:
                raise ValueError(
                    f"No replicate of pair ({i}, {j}) falls in the conditioning window R_{on} = {v:g} +- {h:g}, so "
                    f"the windowed conditional cannot be estimated there."
                )

            ys = np.linspace(0.0, float(np.quantile(sel, q_max)), n_grid)

            # from the sorted sample, not an (n_win x n_grid) boolean matrix, which at this resolution would be
            # hundreds of millions of entries
            cdf = np.searchsorted(np.sort(sel), ys, side='right') / sel.size

            cached.append(
                (int(i), int(j), on, float(v), float(h), int(sel.size), float(sel.mean()),
                 float(sel.std() / np.sqrt(sel.size)), ys, cdf)
            )

        setattr(self, '_loci_windowed_conditional' if loci else '_windowed_conditional', cached)


class _WindowedConditional(EmpiricalDistribution):  # pragma: no cover
    """
    The replicates kept by ``EmpiricalJointDistribution.conditional``, with the local-linear mean described there.

    :param samples: The other reward over the kept replicates.
    :param offsets: The conditioning values of the kept replicates minus the conditioning value.
    :param window: Half-width of the window, the scale of the weights.
    """

    def __init__(self, samples: np.ndarray, offsets: np.ndarray, window: float) -> None:
        super().__init__(samples)

        #: Conditioning offsets ``R_on - value`` of the selected replicates.
        self._offsets = np.asarray(offsets, dtype=float)

        #: Half-width of the window.
        self._window = float(window)

    @cached_property
    def mean(self) -> float:
        """Local-linear window mean, see
        :meth:`EmpiricalJointDistribution.conditional() <phasegen.distributions.EmpiricalJointDistribution.conditional>`.
        """
        x, y, h = self._offsets, self.samples, self._window

        if h <= 0 or x.size < 3:
            return float(np.mean(y))

        w = (1.0 - np.minimum(np.abs(x / h), 1.0) ** 3) ** 3
        sw, swx, swx2 = w.sum(), (w * x).sum(), (w * x * x).sum()
        det = sw * swx2 - swx ** 2

        if not np.isfinite(det) or abs(det) < 1e-300:
            return float(np.mean(y))

        return float((swx2 * (w * y).sum() - swx * (w * x * y).sum()) / det)


class EmpiricalJointDistribution:  # pragma: no cover
    r"""
    Empirical joint distribution of two accumulated rewards :math:`R_a` and :math:`R_b` from paired realisations, the
    sampled counterpart of :class:`~phasegen.distributions.JointRewardDistribution`. Its marginals, joint CDF,
    covariance and correlation are the sample estimators of :class:`~phasegen.distributions.EmpiricalDistribution`,
    with the normalisation :math:`1 / N` for :math:`N` replicates.

    .. warning::
        :meth:`EmpiricalJointDistribution.conditional()
        <phasegen.distributions.EmpiricalJointDistribution.conditional>` averages over a window of conditioning
        values and is therefore only an approximate check on the exact conditional distribution.

    The following example estimates the correlation of the branch lengths of the first two frequency classes from
    1000 sampled trajectories.

    ::

        emp = pg.Coalescent(n=5).sfs.to_empirical(1000, seed=1)

        corr = emp.joint_distribution(1, 2).corr
    """

    def __init__(self, samples_a: np.ndarray, samples_b: np.ndarray) -> None:
        """
        :param samples_a: Per-replicate realisations of the first reward.
        :param samples_b: Per-replicate realisations of the second reward.
        """
        #: Per-replicate realisations of the two rewards.
        self._a = np.asarray(samples_a, dtype=float)
        self._b = np.asarray(samples_b, dtype=float)

    def marginal(self, which: str = 'a') -> EmpiricalDistribution:
        """
        The empirical marginal distribution of the first reward (``which='a'``) or the second (``which='b'``).

        :param which: Which reward's marginal, ``'a'`` or ``'b'``.
        :return: The empirical marginal distribution.
        :raises ValueError: If ``which`` is not ``'a'`` or ``'b'``.
        """
        if which not in ('a', 'b'):
            raise ValueError("`which` must be 'a' or 'b'.")
        return EmpiricalDistribution(self._a if which == 'a' else self._b)

    def conditional(self, on: str = 'a', value: float = 0.0, window: float = None) -> EmpiricalDistribution:
        r"""
        The empirical conditional distribution of one reward given that the other, the conditioning reward, is close
        to ``value``, the sampled counterpart of
        :meth:`JointRewardDistribution.conditional() <phasegen.distributions.JointRewardDistribution.conditional>`.

        With :math:`c_m` the conditioning reward and :math:`y_m` the other reward of replicate :math:`m`, the estimate
        keeps the replicates with :math:`|c_m - v| \le h`, where :math:`v` is ``value`` and :math:`h` the half-width
        ``window``. By default :math:`h` is the smallest half-width keeping a number of replicates that grows with
        the sample size. The cdf, pdf, quantile and variance are those of
        :class:`~phasegen.distributions.EmpiricalDistribution` over the kept replicates. The mean is the intercept
        :math:`\beta_0` of the local-linear fit minimizing

        .. math::

            \sum_{|c_m - v| \le h} w_m \bigl(y_m - \beta_0 - \beta_1 (c_m - v)\bigr)^2,

        with tricube weights :math:`w_m = (1 - |c_m - v|^3 / h^3)^3`, which removes the bias of the plain window mean
        where the conditional mean changes with :math:`v`. Every estimate remains an average over the window.

        :param on: Which reward to condition on, ``'a'`` for :math:`R_a` or ``'b'`` for :math:`R_b`.
        :param value: The conditioning value :math:`v`.
        :param window: The half-width :math:`h`, in units of the conditioning reward. ``None`` for the default.
        :return: The empirical conditional distribution of the other reward.
        :raises ValueError: If ``on`` is not ``'a'`` or ``'b'``, or no replicate falls in the window.
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        cond, other = (self._a, self._b) if on == 'a' else (self._b, self._a)
        distance = np.abs(cond - value)
        if window is None:
            k = min(max(200, cond.size // 50), cond.size - 1)
            window = float(np.partition(distance, k)[k])
        mask = distance <= window
        if not mask.any():
            raise ValueError(f"No samples within window {window:g} of {value:g}.")

        return _WindowedConditional(other[mask], cond[mask] - value, float(window))

    def conditional_on_atom(self, on: str = 'a') -> Tuple[float, EmpiricalDistribution]:
        """
        The empirical conditional distribution of the other reward given the atom event that the conditioning reward
        is zero, with the atom's mass.

        Unlike :meth:`EmpiricalJointDistribution.conditional()
        <phasegen.distributions.EmpiricalJointDistribution.conditional>`, this needs no window: the atom event has
        positive probability, so the conditioning set is exactly the replicates in which the conditioning reward is
        zero, and the estimate carries no bandwidth bias.

        :param on: Which reward to condition on, ``'a'`` or ``'b'``.
        :return: The atom's mass ``P(R_{on} = 0)`` and the distribution of the other reward over those replicates.
        :raises ValueError: If ``on`` is not ``'a'`` / ``'b'``, or no replicate has the conditioning reward at zero.
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")

        cond, other = (self._a, self._b) if on == 'a' else (self._b, self._a)
        empty = cond == 0.0

        if not empty.any():
            raise ValueError(f"No replicate has R_{on} = 0, so the atom conditional cannot be estimated.")

        return float(empty.mean()), EmpiricalDistribution(other[empty])

    def cdf(self, x: float, y: float) -> float:
        r"""
        The empirical joint CDF, the fraction of replicates with :math:`R_a \le x` and :math:`R_b \le y`.

        :param x: Threshold for :math:`R_a`.
        :param y: Threshold for :math:`R_b`.
        :return: The empirical joint probability.
        """
        return float(((self._a <= x) & (self._b <= y)).mean())

    @property
    def mean(self) -> np.ndarray:
        r"""The pair of sample means of :math:`R_a` and :math:`R_b`."""
        return np.array([self._a.mean(), self._b.mean()])

    @property
    def cov(self) -> float:
        """The sample covariance of the two rewards, with the normalisation of
        :class:`~phasegen.distributions.EmpiricalDistribution`."""
        return float(np.cov(self._a, self._b, bias=True)[0, 1])

    @property
    def corr(self) -> float:
        """The empirical Pearson correlation of the two rewards."""
        return float(np.corrcoef(self._a, self._b)[0, 1])


class EmpiricalPhaseTypeSFSDistribution(EmpiricalPhaseTypeDistribution, TajimaSFSMixin):  # pragma: no cover
    """
    Empirical site-frequency spectrum with a per-deme breakdown, built by
    :meth:`UnfoldedSFSDistribution.to_empirical() <phasegen.distributions.UnfoldedSFSDistribution.to_empirical>` or
    by :class:`~phasegen.distributions.MsprimeCoalescent`. The estimators of
    :class:`~phasegen.distributions.EmpiricalDistribution` apply per frequency class.

    The following example estimates the mean and covariance of the spectrum from 1000 sampled trajectories.

    ::

        emp = pg.Coalescent(n=5).sfs.to_empirical(1000, seed=1)

        mean, cov = emp.mean, emp.cov
    """

    #: Static for backward compatibility.
    _layout_lineages: LineageConfig | InitialDistribution | None = None

    #: Static for backward compatibility.
    _layout_loci: LocusConfig | InitialDistribution | None = None

    def _tajima_n(self) -> int:
        # derive n from the (serialized) mean vector so this works on fixtures restored without ``n``
        return len(np.asarray(self.mean)) - 1

    def _polymorphic_bins(self) -> range:
        """
        The polymorphic frequency classes, ``1, ..., n // 2`` for a folded spectrum and ``1, ..., n - 1`` otherwise.

        :return: The frequency classes.
        """
        n = self.samples.shape[1] - 1

        return range(1, n // 2 + 1) if self._folded else range(1, n)

    def _tajima_mean(self) -> np.ndarray:
        n = self._tajima_n()
        return np.asarray(self.mean)[1:n]

    def _tajima_cov(self) -> np.ndarray:
        n = self._tajima_n()
        return np.asarray(self.cov)[1:n, 1:n]

    def __init__(
            self,
            branch_lengths: np.ndarray,
            mutations: Optional[np.ndarray],
            pops: List[str],
            sfs_dist: Type[SFSDistribution],
            locus_agg: Callable = lambda x: x.sum(axis=0),
            resolves_demes: bool = True,
            lineage_config: LineageConfig | InitialDistribution = None,
            locus_config: LocusConfig | InitialDistribution = None
    ) -> None:
        """
        Initialize the distribution from the given realisations.

        :param branch_lengths: Branch lengths per locus, deme, replicate and frequency class, of shape
            ``(loci, demes, N, n + 1)``.
        :param mutations: Mutation counts per locus, deme, replicate and polymorphic frequency class, or ``None`` for
            a spectrum without mutations.
        :param pops: List of population names.
        :param sfs_dist: SFS distribution class.
        :param locus_agg: Aggregation function for loci.
        :param resolves_demes: Whether the branch lengths resolve the demes. Otherwise :attr:`demes` raises.
        :param lineage_config: The lineages the replicates start from, recorded by :meth:`mutation_layout`. By default,
            ``n`` lineages in one deme.
        :param locus_config: The loci the replicates start from, recorded by :meth:`mutation_layout`. By default, one
            locus.
        """
        over_loci = locus_agg(branch_lengths).astype(float)

        EmpiricalDistribution.__init__(self, over_loci.sum(axis=0))

        #: Population names
        self.pops = pops

        #: Number of lineages
        self.n = branch_lengths.shape[-1] - 1

        #: SFS distribution class
        self._sfs_dist = sfs_dist

        #: Branch length samples by deme and locus
        self._samples = branch_lengths

        #: Mutation counts by deme and locus, ``None`` for a spectrum without mutations
        self._mutations = mutations

        self.resolves_demes = resolves_demes

        #: The lineages the replicates start from.
        self._layout_lineages: LineageConfig | InitialDistribution = lineage_config

        #: The loci the replicates start from.
        self._layout_loci: LocusConfig | InitialDistribution = locus_config

        #: Relative frequency yielded by the most recently started
        #: :meth:`EmpiricalPhaseTypeSFSDistribution.get_mutation_configs()
        #: <phasegen.distributions.EmpiricalPhaseTypeSFSDistribution.get_mutation_configs>` iterator.
        self.generated_mass = 0

        #: Atom-conditional ground truth: ``[(i, j, on, mass, dist), ...]``, see
        #: ``_cache_atom_conditional``. Survives ``_drop`` and is serialized with the comparison.
        self._atom_conditional: list = []

        #: Cached windowed-conditional ground truth, see ``_cache_windowed_conditional``.
        self._windowed_conditional: list = []

    @cached_property
    def mean(self) -> SFS:
        """
        Sample mean spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().mean)

    @cached_property
    def var(self) -> SFS:
        """
        Sample variance spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().var)

    @cached_property
    def m2(self) -> SFS:
        """
        Second raw sample moment spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().m2)

    @cached_property
    def cov(self) -> TwoSFS:
        """
        Sample covariance matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        return TwoSFS(super().cov)

    @cached_property
    def corr(self) -> TwoSFS:
        """
        Sample correlation matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        return TwoSFS(super().corr)

    def _touch(self, t: np.ndarray) -> None:
        """
        Touch as ``EmpiricalPhaseTypeDistribution._touch`` and persist ``mutation_configs`` when mutation counts exist.

        :param t: Times to cache properties for.
        """
        super()._touch(t)

        if self._mutations is not None:
            self.__dict__['mutation_configs'] = self._config_frequencies()

    def _drop(self) -> None:
        """
        Drop simulated samples.
        """
        super()._drop()

        self._mutations = None

    def cross_moment(self, i: int, j: int) -> float:
        r"""
        Sample cross-moment :math:`N^{-1} \sum_{m} L_{mi} L_{mj}` of the branch lengths :math:`L_{mi}` and
        :math:`L_{mj}` subtending ``i`` and ``j`` of the :math:`n` lineages in replicate :math:`m = 1, \dots, N`, the
        sampled counterpart of
        :meth:`JointRewardDistribution.moment() <phasegen.distributions.JointRewardDistribution.moment>` ``(1, 1)``.

        :param i: First frequency class.
        :param j: Second frequency class.
        :return: The empirical cross-moment.
        """
        return float((self.samples[:, i] * self.samples[:, j]).mean())

    def joint_cdf(self, i: int, j: int, x: float, y: float) -> float:
        r"""
        Empirical joint CDF of two SFS bins, the fraction of replicates whose branch lengths :math:`L_i` and
        :math:`L_j` subtending ``i`` and ``j`` of the :math:`n` lineages satisfy :math:`L_i \le x` and
        :math:`L_j \le y`, the sampled counterpart of :attr:`JointRewardDistribution.cdf
        <phasegen.distributions.JointRewardDistribution.cdf>`.

        :param i: First frequency class.
        :param j: Second frequency class.
        :param x: Threshold for :math:`L_i`.
        :param y: Threshold for :math:`L_j`.
        :return: The empirical joint probability.
        """
        return float(((self.samples[:, i] <= x) & (self.samples[:, j] <= y)).mean())

    def _cache_joint_surface(self, pairs: List[Tuple[int, int]], n_grid: int = 25, q_max: float = 0.95) -> None:
        """
        Pre-compute, for each requested bin pair, the empirical joint CDF and density over a 2D grid (spanning each
        bin's support up to its ``q_max`` quantile), for the full-grid surface comparison. The density is the mixed
        second difference of the CDF grid (grid spacing = bandwidth). Stored as
        ``self._joint_surface = [(i, j, xs, ys, cdf_grid, pdf_grid), ...]`` and serialized with the comparison.
        """
        s = self.samples
        n = s.shape[0]
        self._joint_surface = []
        for i, j in pairs:
            li, lj = s[:, i], s[:, j]
            xs = np.linspace(0.0, float(np.quantile(li, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(lj, q_max)), n_grid)
            # empirical joint CDF on the grid: P(L_i <= x_a, L_j <= y_b) = (1/N) sum_r 1{li_r<=x_a} 1{lj_r<=y_b}
            a = (li[:, None] <= xs[None, :]).astype(float)  # (N, X)
            b = (lj[:, None] <= ys[None, :]).astype(float)  # (N, Y)
            cdf = (a.T @ b) / n  # (X, Y)
            # density via the mixed second difference of the CDF surface (no separate bandwidth needed)
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._joint_surface.append((int(i), int(j), xs, ys, cdf, pdf))

    def _cache_atom_conditional(self, pairs: List[Tuple[int, int]], n_grid: int = 100) -> None:
        """
        Cache, per bin pair and conditioning axis, ``EmpiricalJointDistribution.conditional_on_atom`` as
        ``self._atom_conditional = [(i, j, on, mass, dist), ...]``, each ``dist`` touched on its own support and
        dropped. An axis without an atom is recorded with zero mass and no distribution, which the comparison asserts.

        :param pairs: Bin pairs to cache.
        :param n_grid: Points of the cdf / pdf grid each conditional is cached on.
        """
        s = np.asarray(self.samples)
        self._atom_conditional = []

        for i, j in pairs:
            jd = EmpiricalJointDistribution(s[:, i], s[:, j])
            for on in ('a', 'b'):
                try:
                    mass, dist = jd.conditional_on_atom(on)
                except ValueError:
                    self._atom_conditional.append((int(i), int(j), on, 0.0, None))
                    continue

                dist._touch(np.linspace(0.0, float(np.max(dist.samples)), n_grid))
                dist._drop()
                self._atom_conditional.append((int(i), int(j), on, mass, dist))

    def _pair_samples(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Per-replicate branch lengths of frequency classes ``i`` and ``j``."""
        s = np.asarray(self.samples)
        return s[:, i], s[:, j]

    def joint_distribution(self, i: int, j: int) -> 'EmpiricalJointDistribution':
        """
        The empirical joint distribution of the branch lengths of bins ``i`` and ``j``, from the per-replicate
        samples, the sampled counterpart of
        :meth:`UnfoldedSFSDistribution.joint_distribution()
        <phasegen.distributions.UnfoldedSFSDistribution.joint_distribution>`, exposing the same
        :meth:`~EmpiricalJointDistribution.marginal` and :meth:`~EmpiricalJointDistribution.conditional`
        slices for a sanity check against the exact joint.

        :param i: First frequency class.
        :param j: Second frequency class.
        :return: The empirical joint reward distribution of ``(L_i, L_j)``.
        :raises ValueError: If the per-replicate samples have been dropped.
        """
        if self.samples is None:
            raise ValueError("The per-replicate samples have been dropped; joint_distribution needs them.")
        s = np.asarray(self.samples)
        return EmpiricalJointDistribution(s[:, i], s[:, j])

    @cached_property
    def demes(self) -> Dict[str, EmpiricalDistribution]:
        """
        Empirical spectrum of each deme, summed over loci, with the per-class covariance and correlation across demes
        as ``cov`` and ``corr``.

        :return: Dictionary of distributions.
        :raises ValueError: If the branch lengths do not resolve the demes.
        """
        self._check_resolves_demes()

        data = self._samples.sum(axis=0)

        return DictContainer._of_spectra(
            {pop: EmpiricalSFSDistribution(data[i], folded=self._folded) for i, pop in enumerate(self.pops)}, data
        )

    @cached_property
    def loci(self) -> Dict[int, EmpiricalSFSDistribution]:
        """
        Empirical spectrum of each locus, summed over demes, with the per-class covariance and correlation across loci
        as ``cov`` and ``corr``.

        :return: Dictionary of distributions.
        """
        data = self._samples.sum(axis=1)

        return DictContainer._of_spectra(
            {i: EmpiricalSFSDistribution(data[i], folded=self._folded) for i in range(data.shape[0])}, data
        )

    @property
    def _folded(self) -> bool:
        """Whether the spectrum is folded."""
        return issubclass(self._sfs_dist, FoldedSFSDistribution)

    def mutation_layout(self) -> MutationLayout:
        """
        The layout of the configurations, one bin per polymorphic frequency class as
        :meth:`UnfoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>` or
        :meth:`FoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.FoldedSFSDistribution.mutation_layout>`.

        :return: The layout.
        """
        return SFSDistribution._layout_of(
            LineageConfig(self.n) if self._layout_lineages is None else self._layout_lineages,
            LocusConfig() if self._layout_loci is None else self._layout_loci,
            folded=self._folded,
            demes=False
        )

    @property
    def mutation_configs(self) -> Dict[MutationConfig, float]:
        """
        Relative frequency of each mutational configuration among the simulated replicates, of the mutation counts
        summed over loci.

        :return: Dictionary from configuration to relative frequency.
        :raises ValueError: If the spectrum carries no mutation counts, as a spectrum built by
            :meth:`UnfoldedSFSDistribution.to_empirical() <phasegen.distributions.UnfoldedSFSDistribution.to_empirical>`.
        """
        layout = self.mutation_layout()

        return {layout.config(c): p for c, p in self._config_frequencies().items()}

    @mutation_configs.setter
    def mutation_configs(self, configs: Dict[MutationConfig, float]) -> None:
        """
        Store the configuration frequencies.

        :param configs: Dictionary from configuration to relative frequency.
        """
        self.__dict__['mutation_configs'] = {tuple(int(k) for k in c): p for c, p in configs.items()}

    def _config_frequencies(self) -> Dict[Tuple[int, ...], float]:
        """
        The configuration frequencies keyed by the plain tuples of the counts, as they are stored and serialized.

        :return: Dictionary from configuration to relative frequency.
        :raises ValueError: If the spectrum carries no mutation counts.
        """
        # stored under the name of the property, so a serialized comparison restores it through the setter, and
        # persisted by ``_touch`` only when mutation counts exist
        if 'mutation_configs' in self.__dict__:
            return self.__dict__['mutation_configs']

        if self._mutations is None:
            raise ValueError(
                "This spectrum carries no mutation counts (it was sampled from branch lengths only, or its samples "
                "were dropped), so mutational configuration frequencies are unavailable."
            )

        configs = {}

        # the mutations of a replicate summed over loci and demes, as the branch lengths of the moments
        for config in self._mutations.sum(axis=(0, 1)):
            key = tuple(int(c) for c in config)
            configs[key] = configs.get(key, 0) + 1 / self._mutations.shape[2]

        if Settings.cache:
            self.__dict__['mutation_configs'] = configs

        return configs

    def get_mutation_config(self, config: Union[MutationConfig, Sequence[int], int]) -> float:
        """
        Relative frequency of a mutational configuration among the simulated replicates, the sampled counterpart of
        :meth:`UnfoldedSFSDistribution.get_mutation_config()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`, which defines configurations.

        :param config: A :class:`~phasegen.distributions.MutationConfig`, or one mutation count per frequency class of
            the default layout, a single count for one class.
        :return: The fraction of replicates showing the configuration, 0 for a configuration no replicate shows.
        :raises ValueError: If ``config`` does not have one non-negative integer per frequency class, is a
            :class:`~phasegen.distributions.MutationConfig` of another layout, or the spectrum carries no mutation
            counts.
        """
        frequencies = self._config_frequencies()
        layout = self.mutation_layout()

        if not isinstance(config, MutationConfig):
            config = layout.config((config,) if np.isscalar(config) else config)
        elif config.layout != layout:
            raise ValueError(f"The configuration must have the layout {layout}, got {config.layout}.")

        return frequencies.get(tuple(config), 0)

    def get_mutation_configs(self) -> Iterator[Tuple[MutationConfig, float]]:
        """
        Sampled counterpart of :meth:`UnfoldedSFSDistribution.get_mutation_configs()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs>` with ``order='count'``, yielding the
        relative frequencies of the configurations among the simulated replicates.

        :return: An iterator over pairs of configuration and relative frequency.
        """
        # reset generated mass
        self.generated_mass = 0

        # iterate over number of mutations
        i = 0
        while True:
            # iterate over configurations
            for config in self.mutation_layout().configs(i):
                p = self._config_frequencies().get(tuple(config), 0)
                self.generated_mass += p
                yield config, p

            # increase counter for number of mutations
            i += 1


class EmpiricalTwoLocusSFSDistribution:  # pragma: no cover
    r"""
    Empirical two-locus site-frequency spectrum, built by
    :meth:`TwoLocusSFSDistribution.to_empirical() <phasegen.distributions.TwoLocusSFSDistribution.to_empirical>` or
    by :class:`~phasegen.distributions.MsprimeCoalescent`. Its :attr:`mean` has the entries
    :math:`N^{-1} \sum_{m=1}^{N} L^0_{mi} L^1_{mj}`, where :math:`L^\ell_{mi}` is the branch length subtending
    :math:`i` of the :math:`n` lineages at locus :math:`\ell \in \{0, 1\}` in replicate :math:`m = 1, \dots, N`. The
    mean is not symmetrized over the two loci.
    """

    def __init__(self, mean: np.ndarray, left: np.ndarray = None, right: np.ndarray = None) -> None:
        """
        :param mean: The sample mean two-locus SFS array.
        :param left: Optional per-replicate locus-0 SFS branch lengths, of shape ``(N, n + 1)``.
        :param right: Optional per-replicate locus-1 SFS branch lengths, of shape ``(N, n + 1)``.
        """
        self._mean = np.asarray(mean)
        self._left = None if left is None else np.asarray(left)
        self._right = None if right is None else np.asarray(right)

        #: Number of samples, retained when the samples are freed so that it is recorded in a serialized comparison.
        self.n_samples: Optional[int] = None if left is None else np.asarray(left).shape[0]

    @property
    def mean(self) -> TwoLocusSFS:
        """Mean two-locus SFS."""
        return TwoLocusSFS(self._mean)

    def _drop(self) -> None:
        """Drop the per-replicate samples (the mean is retained)."""
        self._left = None
        self._right = None

    def cross_moment(self, i: int, j: int) -> float:
        r"""
        Empirical cross-locus moment :math:`\mathbb{E}[L^0_i\, L^1_j]`, the two-locus SFS entry, estimated as in
        :meth:`EmpiricalPhaseTypeSFSDistribution.cross_moment()
        <phasegen.distributions.EmpiricalPhaseTypeSFSDistribution.cross_moment>`.

        :param i: Locus-0 frequency class.
        :param j: Locus-1 frequency class.
        :return: The empirical cross-moment.
        """
        return float((self._left[:, i] * self._right[:, j]).mean())

    def joint_cdf(self, i: int, j: int, x: float, y: float) -> float:
        r"""
        Empirical cross-locus joint CDF :math:`P(L^0_i \le x, L^1_j \le y)`, estimated as in
        :meth:`EmpiricalPhaseTypeSFSDistribution.joint_cdf()
        <phasegen.distributions.EmpiricalPhaseTypeSFSDistribution.joint_cdf>`.

        :param i: Locus-0 frequency class.
        :param j: Locus-1 frequency class.
        :param x: Threshold for ``L^0_i``.
        :param y: Threshold for ``L^1_j``.
        :return: The empirical joint probability.
        """
        return float(((self._left[:, i] <= x) & (self._right[:, j] <= y)).mean())

    def _cache_joint_surface(self, pairs: List[Tuple[int, int]], n_grid: int = 25, q_max: float = 0.95) -> None:
        """Pre-compute the joint surface ground truth of ``EmpiricalPhaseTypeSFSDistribution._cache_joint_surface``
        for each cross-locus bin pair ``(i, j)``, locus-0 class ``i`` against locus-1 class ``j``."""
        n = self._left.shape[0]
        self._joint_surface = []
        for i, j in pairs:
            li, rj = self._left[:, i], self._right[:, j]
            xs = np.linspace(0.0, float(np.quantile(li, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(rj, q_max)), n_grid)
            cdf = ((li[:, None] <= xs[None, :]).astype(float).T @ (rj[:, None] <= ys[None, :]).astype(float)) / n
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._joint_surface.append((int(i), int(j), xs, ys, cdf, pdf))


class _ReplicateStatistic:  # pragma: no cover
    """
    A per-replicate statistic accumulated in a **single pass** over the simulated tree sequences (the observer
    pattern: :meth:`MsprimeCoalescent.simulate` iterates the trees once and feeds every registered statistic, so
    each statistic is a self-contained component and adding one does not touch the simulation loop).

    A statistic allocates its own per-replicate storage and updates from each tree (:meth:`process_tree`, called for
    every locus ``j`` of replicate ``i``) and/or each whole replicate (:meth:`process_replicate`, tree-sequence
    level). Both default to no-ops, so a statistic implements only the hook it needs.
    """

    def process_tree(self, i: int, j: int, tree, ts, ctx: dict) -> None:
        """Update from the locus-``j`` tree of replicate ``i`` (``ctx`` carries shared per-replicate data)."""

    def process_replicate(self, i: int, ts, ctx: dict, seed) -> None:
        """Update from the whole replicate ``i`` (the tree sequence ``ts``), with ``seed`` the replicate's msprime
        seed."""


class _TreeStatistics(_ReplicateStatistic):  # pragma: no cover
    """Tree height, total branch length and SFS read directly from each tree: the root time, the tree's total branch
    length, and per-node branch length binned by descendant count (population index 0; no migration recording)."""

    def __init__(self, n_loci: int, n_pops: int, num_replicates: int, sample_size: int) -> None:
        self.heights = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.total_branch_lengths = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.sfs = np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=float)

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        self.heights[j, 0, i] = tree.time(tree.roots[0])
        self.total_branch_lengths[j, 0, i] = tree.total_branch_length

        for node in tree.nodes():
            t = tree.get_branch_length(node)
            n = tree.get_num_leaves(node)

            self.sfs[j, 0, i, n] += t


class _MigrationTreeStatistics(_ReplicateStatistic):  # pragma: no cover
    """Tree height, total branch length and **per-deme** SFS reconstructed from the recorded migration history, so
    each quantity is attributed to the deme a lineage occupies through time (walking the coalescence and migration
    events). Only validated for relatively simple scenarios."""

    def __init__(
            self,
            n_loci: int,
            n_pops: int,
            num_replicates: int,
            sample_size: int,
            axis: np.ndarray
    ) -> None:
        self.heights = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.total_branch_lengths = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.sfs = np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=float)

        #: Deme axis of each msprime population id.
        self._axis = axis

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        axis = self._axis

        # the coalescences of this tree and the migrations of its lineages at locus ``j``, the unit interval
        # ``[j, j + 1)``, in time order. A tree may span several loci, whose lineages migrate independently, so the
        # migrations taken are those covering the locus midpoint
        point = j + 0.5
        migrations = ts.tables.migrations
        in_tree = (migrations.left <= point) & (migrations.right > point)
        order = np.argsort(migrations.time[in_tree], kind='stable')
        m_time = migrations.time[in_tree][order]
        m_node = migrations.node[in_tree][order]
        m_source = migrations.source[in_tree][order]
        m_dest = migrations.dest[in_tree][order]

        coalescences = sorted((u for u in tree.nodes() if tree.num_children(u) > 0), key=tree.time)

        # deme of each extant lineage, starting from the samples
        pop_states = {n: axis[tree.population(n)] for n in tree.samples()}
        lineages = np.bincount(list(pop_states.values()), minlength=len(axis)).astype(int)
        i_migration = 0
        time = 0.0

        def accumulate(delta: float) -> None:
            self.heights[j, :, i] += delta * lineages / sum(lineages)
            self.total_branch_lengths[j, :, i] += delta * lineages
            for n, pop in pop_states.items():
                self.sfs[j, pop, i, tree.num_samples(n)] += delta

        for u in coalescences:
            coal_time = tree.time(u)

            # migrations of extant lineages before this coalescence
            while i_migration < len(m_time) and m_time[i_migration] <= coal_time:
                if m_node[i_migration] in pop_states:
                    accumulate(m_time[i_migration] - time)
                    time = m_time[i_migration]
                    lineages[axis[m_source[i_migration]]] -= 1
                    lineages[axis[m_dest[i_migration]]] += 1
                    pop_states[m_node[i_migration]] = axis[m_dest[i_migration]]
                i_migration += 1

            accumulate(coal_time - time)
            time = coal_time

            # the children merge into the parent in the parent's deme; a unary node only relabels its lineage
            children = tree.children(u)
            lineages[axis[tree.population(u)]] -= len(children) - 1
            for c in children:
                pop_states.pop(c, None)
            pop_states[u] = axis[tree.population(u)]


class _JointSFSStatistics(_ReplicateStatistic):  # pragma: no cover
    """The joint (per-deme-of-origin) SFS branch lengths from the first locus' tree: the non-central moments (orders
    1..``max_order``) over all replicates plus a capped subset of per-replicate values for the within-tree joint
    ground truth. Single-locus only (accumulated from the ``j == 0`` tree)."""

    def __init__(self, num_replicates: int, max_order: int, shape: tuple, sample_cap: int) -> None:
        self.jsfs_acc = np.zeros((max_order,) + shape)
        self.jsfs_samples = np.zeros((min(num_replicates, sample_cap),) + shape)
        self._max_order = max_order
        self._shape = shape
        self._cap = sample_cap

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        if j != 0:  # accumulated from the first locus' tree only
            return

        pop_of_leaf = ctx['pop_of_leaf']
        jsfs_rep = np.zeros(self._shape)

        for node in tree.nodes():

            # the root subtends all samples (monomorphic) and is skipped
            if tree.parent(node) == -1:
                continue

            # count descendant samples by population (deme of origin)
            vec = [0] * len(self._shape)
            for leaf in tree.leaves(node):
                vec[pop_of_leaf[leaf]] += 1

            if sum(vec) > 0:
                jsfs_rep[tuple(vec)] += tree.get_branch_length(node)

        for order in range(self._max_order):
            self.jsfs_acc[order] += jsfs_rep ** (order + 1)

        if i < self._cap:
            self.jsfs_samples[i] = jsfs_rep


class _MutationStatistics(_ReplicateStatistic):  # pragma: no cover
    """The mutation-count SFS: drop mutations on the replicate's tree sequence at the configured rate and bin each by
    its locus, the unit interval containing its site, and by the number of leaves the carrying node subtends in the
    tree at that site (population index 0). Optionally also bin each by the descendant vector of the carrying node
    (the numbers of its leaves from each sampling population), and by the deme the carrying lineage resides in at the
    mutation time, read from the recorded migrations."""

    def __init__(
            self,
            n_loci: int,
            n_pops: int,
            num_replicates: int,
            sample_size: int,
            mutation_rate: float,
            jsfs_shape: tuple = None,
            axis: np.ndarray = None
    ) -> None:
        self.mutations = np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=int)
        self.joint = None if jsfs_shape is None else np.zeros((num_replicates,) + tuple(jsfs_shape), dtype=int)
        self.by_deme = None if axis is None else np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=int)
        self._rate = mutation_rate
        self._axis = axis

    def process_replicate(self, i, ts, ctx, seed) -> None:
        import msprime as ms

        mts = ms.sim_mutations(ts, rate=self._rate, random_seed=seed)

        # the mutations ordered by position, and the range of them on each tree
        positions = mts.sites_position[mts.mutations_site]
        nodes = mts.mutations_node
        bounds = np.searchsorted(positions, mts.breakpoints(as_array=True))

        leaves = np.empty(len(nodes), dtype=int)
        for tree in mts.trees():
            for k in range(bounds[tree.index], bounds[tree.index + 1]):
                leaves[k] = tree.get_num_leaves(nodes[k])

                if self.joint is not None:
                    vec = [0] * (self.joint.ndim - 1)
                    for leaf in tree.samples(nodes[k]):
                        vec[ctx['pop_of_leaf'][leaf]] += 1
                    self.joint[(i,) + tuple(vec)] += 1

        np.add.at(self.mutations, (positions.astype(int), 0, i, leaves), 1)

        if self.by_deme is not None:
            migrations = mts.tables.migrations
            times = mts.mutations_time

            for k, (u, t, x) in enumerate(zip(nodes, times, positions)):
                # the deme at time t is the destination of the latest migration of the lineage below t, or the deme
                # the node was born in
                hit = np.flatnonzero((migrations.node == u) & (migrations.left <= x) & (migrations.right > x)
                                     & (migrations.time <= t))
                pop = (migrations.dest[hit[np.argmax(migrations.time[hit])]] if hit.size
                       else mts.node(u).population)
                self.by_deme[int(x), self._axis[pop], i, leaves[k]] += 1


def _unlinked_initial_state(samples: dict, n_unlinked: int, demography) -> 'tskit.TableCollection':
    """
    The initial state of a two-locus simulation in which ``n_unlinked`` of the sampled lineages start unlinked
    between the loci.

    A sample node is ancestral over the whole sequence, so an unlinked sample cannot be expressed by sampling: it is
    given one parent per locus, which leaves it a child everywhere and its two parents as the extant lineages, one
    per locus. That is the configuration a recombination event produces, which is what "unlinked" means here. The
    unlinked lineages are taken from the demes in the order of ``samples``.

    :param samples: Number of samples per deme, by deme name.
    :param n_unlinked: Number of lineages starting unlinked between the loci.
    :param demography: The msprime demography, whose populations the tables must mirror.
    :return: Tables to start the simulation from.
    """
    import tskit

    # the two loci are the unit intervals of a length-two sequence
    tables = tskit.TableCollection(sequence_length=2)
    tables.time_units = 'generations'
    tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()

    index = {}
    for population in demography.populations:
        index[population.name] = tables.populations.add_row(metadata={'name': population.name})

    # the samples take node ids 0 to n - 1, which the statistics accumulators rely on
    unlinked = []
    for name, count in samples.items():
        for _ in range(count):
            node = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=index[name])
            if len(unlinked) < n_unlinked:
                unlinked.append((node, name))

    # the split parents sit just above the samples, msprime requiring a parent to postdate its child
    split_time = 1e-12
    for node, name in unlinked:
        locus_0 = tables.nodes.add_row(flags=0, time=split_time, population=index[name])
        locus_1 = tables.nodes.add_row(flags=0, time=split_time, population=index[name])
        tables.edges.add_row(left=0, right=1, parent=locus_0, child=node)
        tables.edges.add_row(left=1, right=2, parent=locus_1, child=node)

    tables.sort()

    return tables


class MsprimeCoalescent(AbstractCoalescent):
    """
    Coalescent whose statistics are estimated from ``msprime`` ancestry simulations, independently of the phase-type
    computation. :meth:`MsprimeCoalescent.simulate() <phasegen.distributions.MsprimeCoalescent.simulate>` splits the
    replicates into batches simulated in parallel, each seeded from its own child of
    :class:`numpy.random.SeedSequence` spawned from :attr:`seed`, and the statistics use the estimators of :class:`~phasegen.distributions.EmpiricalDistribution`.

    The following example estimates the mean tree height and site-frequency spectrum from 1000 simulated genealogies.

    ::

        ms = pg.distributions.MsprimeCoalescent(n=5, num_replicates=1000, parallelize=False, seed=1)

        height = ms.tree_height.mean
        sfs = ms.sfs.mean
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | LineageConfig | InitialDistribution,
            demography: Demography = None,
            model: CoalescentModel = StandardCoalescent(),
            loci: int | LocusConfig | InitialDistribution = 1,
            recombination_rate: float = None,
            mutation_rate: float = None,
            end_time: float = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            record_migration: bool = False,
            simulate_mutations: bool = False,
            seed: int = None
    ) -> None:
        """
        Configure an msprime simulation of the given scenario. The replicates are simulated on first access to a
        statistic.

        :param n: Number of lineages, lineage configuration, or initial distribution over lineage configurations,
            from which each replicate draws its starting configuration.
        :param demography: Demography.
        :param model: Coalescent model.
        :param loci: Number of loci, locus configuration, or initial distribution over locus configurations, from
            which each replicate draws its starting configuration.
        :param recombination_rate: Recombination rate.
        :param mutation_rate: Mutation rate.
        :param end_time: Time when to end the simulation.
        :param num_replicates: Number of replicates.
        :param n_threads: Number of threads.
        :param parallelize: Whether to parallelize. ``Settings.parallelize = False`` overrides it.
        :param record_migration: Whether to record migrations, which the per-deme statistics of more than one deme
            require.
        :param simulate_mutations: Whether to simulate mutations.
        :param seed: Non-negative integer random seed. ``None`` draws fresh entropy.
        :raises ValueError: If ``model`` is a Beta coalescent whose alpha exceeds 1.991, the largest that msprime
            accepts, or if ``simulate_mutations`` is set without a ``mutation_rate``.
        """
        super().__init__(
            n=n,
            model=model,
            loci=loci,
            recombination_rate=recombination_rate,
            demography=demography,
            end_time=end_time
        )

        if isinstance(self.model, BetaCoalescent) and self.model.alpha > _MSPRIME_BETA_ALPHA_MAX:
            raise ValueError(
                f"msprime accepts Beta coalescents with alpha up to {_MSPRIME_BETA_ALPHA_MAX}, got {self.model.alpha}."
            )

        if simulate_mutations and mutation_rate is None:
            raise ValueError("Simulating mutations requires a mutation rate.")

        if mutation_rate is not None and not simulate_mutations:
            self._logger.warning("Mutation rate is set but mutations are not simulated.")

        #: Site frequency spectrum counts per locus, deme and replicate.
        self.sfs_lengths: np.ndarray | None = None

        #: Total branch lengths per locus, deme and replicate.
        self.total_branch_lengths: np.ndarray | None = None

        #: Tree heights per locus, deme and replicate.
        self.heights: np.ndarray | None = None

        #: Mutations per locus, deme and replicate.
        self.mutations: np.ndarray | None = None

        #: Mutations per replicate and descendant vector, for multi-population single-locus scenarios.
        self.jsfs_mutations: np.ndarray | None = None

        #: Mutations per locus, deme of residence at the mutation time, replicate and frequency class, with migration
        #: recording.
        self.deme_mutations: np.ndarray | None = None

        #: Joint SFS (non-central) moments per descendant configuration, of orders 1, ..., ``_jsfs_max_order``.
        self.jsfs_moments: np.ndarray | None = None

        #: Per-replicate joint SFS branch lengths (capped subset) for the within-tree joint ground truth.
        self.jsfs_samples: np.ndarray | None = None

        #: Actual number of replicates simulated and averaged over (``num_replicates`` rounded down to a multiple of
        #: ``n_threads``), set by :meth:`simulate`.
        self.n_total: int | None = None

        #: Number of replicates.
        self.num_replicates: int = num_replicates

        #: Mutation rate.
        self.mutation_rate: float = mutation_rate

        #: Number of threads, capped at ``num_replicates`` so each thread simulates at least one replicate
        #: (``num_replicates // n_threads`` must not floor to zero, which would yield empty simulations).
        self.n_threads: int = max(1, min(n_threads, num_replicates))

        #: Whether to parallelize computations.
        self.parallelize: bool = parallelize

        #: Whether to record migrations.
        self.record_migration: bool = record_migration

        #: Whether to simulate mutations.
        self.simulate_mutations: bool = simulate_mutations

        #: Random seed.
        self.seed: int = seed

    def get_coalescent_model(self) -> 'msprime.AncestryModel':
        """
        Get the coalescent model.

        :return: msprime coalescent model.
        """
        import msprime as ms

        if isinstance(self.model, StandardCoalescent):
            return ms.StandardCoalescent()

        if isinstance(self.model, BetaCoalescent):
            return ms.BetaCoalescent(alpha=self.model.alpha)

        if isinstance(self.model, DiracCoalescent):
            return ms.DiracCoalescent(psi=self.model.psi, c=self.model.c)

    def _msprime_seed(self) -> Optional[int]:
        """
        The msprime seed of :attr:`seed`, wrapped into msprime's range :math:`[1, 2^{32} - 1]`, which it leaves
        unchanged. It seeds the statistics simulated outside the batches of :meth:`simulate`.

        :return: The msprime seed, ``None`` for fresh entropy.
        """
        return None if self.seed is None else (self.seed - 1) % (2 ** 32 - 1) + 1

    def _batch_seeds(self) -> List[Optional[np.random.SeedSequence]]:
        """
        The seed sequence of each of the :attr:`n_threads` batches, spawned from :attr:`seed`.

        :return: One seed sequence per batch, ``None`` for fresh entropy.
        """
        if self.seed is None:
            return [None] * self.n_threads

        return np.random.SeedSequence(self.seed).spawn(self.n_threads)

    @staticmethod
    def _msprime_seeds(seed: Optional[np.random.SeedSequence], n: int) -> List[Optional[int]]:
        """
        Draw ``n`` msprime seeds, in msprime's range :math:`[1, 2^{32} - 1]`, from a seed sequence.

        :param seed: Seed sequence, ``None`` for fresh entropy.
        :param n: Number of seeds.
        :return: The msprime seeds, ``None`` for fresh entropy.
        """
        if seed is None:
            return [None] * n

        return [int(s) % (2 ** 32 - 1) + 1 for s in seed.generate_state(n)]

    @property
    def _msprime_samples(self) -> Dict[str, int]:
        """The number of samples per population, keyed by its msprime name (see ``Demography._msprime_names``)."""
        names = self.demography._msprime_names
        return {names[pop]: n for pop, n in self.lineage_config.lineage_dict.items()}

    def _placements(self, demography: 'msprime.Demography') -> List[Tuple[float, dict]]:
        """
        The starting configurations of the replicates, one per component of the initial distribution. Lineages that
        start unlinked between the loci have no expression as samples, a sample being ancestral over the whole
        sequence, so they are set up as an initial state instead (see ``_unlinked_initial_state``).

        :param demography: The msprime demography.
        :return: Pairs of the weight and the keyword arguments of :func:`msprime.sim_ancestry` placing the samples.
        """
        names = self.demography._msprime_names
        lineages = [(1.0, self.lineage_config)] if self.lineage_distribution is None else self.lineage_distribution
        loci = [(1.0, self.locus_config)] if self.locus_distribution is None else self.locus_distribution

        placements = []
        for w_a, a in lineages:
            samples = {names[pop]: n for pop, n in a.lineage_dict.items()}

            for w_b, b in loci:
                if b.n == 2 and b.n_unlinked > 0:
                    placement = dict(initial_state=_unlinked_initial_state(samples, b.n_unlinked, demography))
                else:
                    placement = dict(samples=samples, sequence_length=b.n)

                placements.append((w_a * w_b, placement))

        return placements

    @staticmethod
    def _sim_ancestry(
            placements: List[Tuple[float, dict]],
            num_replicates: int,
            random_seed: Optional[int],
            **kwargs
    ) -> Iterator['tskit.TreeSequence']:
        """
        Simulate replicates whose starting configurations are drawn from the weights of ``placements``. The
        replicates are grouped by starting configuration.

        :param placements: Pairs of the weight and the keyword arguments placing the samples, see ``_placements``.
        :param num_replicates: Number of replicates.
        :param random_seed: msprime seed, which also draws the starting configurations. ``None`` draws fresh entropy.
        :param kwargs: Further keyword arguments of :func:`msprime.sim_ancestry`.
        :return: The tree sequences.
        """
        import msprime as ms

        if len(placements) == 1:
            yield from ms.sim_ancestry(
                num_replicates=num_replicates, random_seed=random_seed, **placements[0][1], **kwargs
            )
            return

        rng = np.random.default_rng(random_seed)
        counts = rng.multinomial(num_replicates, [w for w, _ in placements])
        seeds = rng.integers(1, 2 ** 32 - 1, size=len(placements))

        for (_, placement), count, seed in zip(placements, counts, seeds):
            if count > 0:
                yield from ms.sim_ancestry(num_replicates=int(count), random_seed=int(seed), **placement, **kwargs)

    def simulate(self) -> None:
        """
        Simulate data using msprime, once per instance, so that every statistic describes the same tree sequences.
        A subsequent call returns without simulating while the data are held.
        """
        if self.heights is not None:
            return

        # number of replicates for one thread
        num_replicates = self.num_replicates // self.n_threads
        demography = self.demography.to_msprime()
        placements = self._placements(demography)
        model = self.get_coalescent_model()
        end_time = self.end_time
        n_pops = self.demography.n_pops
        sample_size = self.lineage_config.n

        # joint SFS is accumulated from the same trees, but only for multi-population, single-locus scenarios where
        # it is meaningful (the descendant configuration is by deme of origin), with one sample configuration
        compute_jsfs = self.lineage_config.n_pops > 1 and self.locus_config.n == 1 and (
                self.lineage_distribution is None
                or all(c == self.lineage_config for c in self.lineage_distribution.configs)
        )
        jsfs_max_order = self._jsfs_max_order
        jsfs_shape = tuple(int(s) + 1 for s in self.lineage_config.lineages)

        # deme axes follow ``lineage_config.pop_names``, msprime population ids follow the demography
        names = self.demography._msprime_names
        name_to_index = {names[name]: i for i, name in enumerate(self.lineage_config.pop_names)}
        axis = np.array([name_to_index[pop.name] for pop in demography.populations])
        n_total = self.n_total = num_replicates * self.n_threads
        # retain a capped subset of per-replicate joint SFS branch lengths (the moments use all replicates; the
        # within-tree joint CDF / cross-moment ground truth needs only enough samples for a ~0.02 tolerance)
        jsfs_sample_cap = self._jsfs_sample_cap // self.n_threads

        def simulate_batch(seed: Optional[np.random.SeedSequence]) -> dict:
            """
            Simulate one batch of replicates, accumulating every requested statistic in a single pass over the tree
            sequences via self-contained per-statistic accumulators.

            :param seed: Seed sequence of the batch, from which the ancestry seed and one mutation seed per replicate
                are drawn.
            :return: Statistics.
            """
            import tskit

            ancestry_seed, *replicate_seeds = self._msprime_seeds(seed, num_replicates + 1)

            # simulate trees
            g: Iterator[tskit.TreeSequence] = self._sim_ancestry(
                placements,
                num_replicates,
                ancestry_seed,
                recombination_rate=self.locus_config.recombination_rate,
                record_migrations=self.record_migration,
                demography=demography,
                model=model,
                ploidy=1,
                end_time=end_time
            )

            # the per-statistic accumulators this scenario needs; the tree-height / total-branch-length / SFS triple
            # is recorded either directly from each tree or, with migration recording, from the migration history
            n_loci = self.locus_config.n
            tree_stats = (_MigrationTreeStatistics(n_loci, n_pops, num_replicates, sample_size, axis)
                          if self.record_migration
                          else _TreeStatistics(n_loci, n_pops, num_replicates, sample_size))
            jsfs_stats = (_JointSFSStatistics(num_replicates, jsfs_max_order, jsfs_shape, jsfs_sample_cap)
                          if compute_jsfs else None)
            mutation_stats = (_MutationStatistics(n_loci, n_pops, num_replicates, sample_size, self.mutation_rate,
                                                  jsfs_shape=jsfs_shape if compute_jsfs else None,
                                                  axis=axis if self.record_migration else None)
                              if self.simulate_mutations else None)
            stats = [s for s in (tree_stats, jsfs_stats, mutation_stats) if s is not None]

            # iterate over the tree sequences once, feeding every statistic
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):

                # map each sample to the index of its sampling population (deme of origin) for the joint SFS
                ctx = {}
                if compute_jsfs:
                    ctx['pop_of_leaf'] = {
                        u: name_to_index[ts.population(ts.node(u).population).metadata['name']]
                        for u in ts.samples()
                    }

                tree: tskit.Tree
                for j, tree in enumerate(self._expand_trees(ts)):
                    for stat in stats:
                        stat.process_tree(i, j, tree, ts, ctx)

                for stat in stats:
                    stat.process_replicate(i, ts, ctx, replicate_seeds[i])

            # the mutations / jSFS arrays default to zeros / None when not requested (kept in the return layout for
            # the cross-thread aggregation in ``simulate``)
            mutations = (mutation_stats.mutations if mutation_stats is not None
                         else np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=int))
            jsfs_acc = jsfs_stats.jsfs_acc if jsfs_stats is not None else np.zeros((jsfs_max_order,) + jsfs_shape)
            jsfs_samples = jsfs_stats.jsfs_samples if jsfs_stats is not None else None

            return dict(
                main=np.concatenate([[tree_stats.heights.T], [tree_stats.total_branch_lengths.T],
                                     tree_stats.sfs.T, mutations.T]),
                jsfs=jsfs_acc,
                jsfs_samples=jsfs_samples,
                jsfs_mutations=mutation_stats.joint if mutation_stats is not None else None,
                deme_mutations=mutation_stats.by_deme if mutation_stats is not None else None
            )

        # parallelize over threads
        batches = parallelize(
            func=simulate_batch,
            data=self._batch_seeds(),
            parallelize=self.parallelize,
            pbar=Settings.use_pbar,
            batch_size=num_replicates,
            desc="Simulating trees",
            dtype=object
        )

        # combine the per-replicate statistics across threads
        res = np.hstack([b['main'] for b in batches])

        # store results
        self.heights = res[0].T
        self.total_branch_lengths = res[1].T
        self.sfs_lengths = res[2:sample_size + 3].T
        self.mutations = res[sample_size + 3:].T.astype(int)

        # combine the joint SFS moments (summed over replicates) across threads and normalize to moments
        self.jsfs_moments = np.sum([b['jsfs'] for b in batches], axis=0) / n_total if compute_jsfs else None

        # combine the (capped) per-replicate joint SFS branch lengths across threads for the joint ground truth
        self.jsfs_samples = np.concatenate([b['jsfs_samples'] for b in batches]) if compute_jsfs else None

        # combine the per-replicate mutation counts by descendant vector and by deme of residence across threads
        if batches[0]['jsfs_mutations'] is not None:
            self.jsfs_mutations = np.concatenate([b['jsfs_mutations'] for b in batches])
        if batches[0]['deme_mutations'] is not None:
            self.deme_mutations = np.concatenate([b['deme_mutations'] for b in batches], axis=2)

    @staticmethod
    def _expand_trees(ts: 'tskit.TreeSequence') -> Iterator['tskit.Tree']:
        """
        Expand tree sequence to `n` trees where `n` is the number of loci.

        :param ts: Tree sequence.
        :return: List of trees.
        """
        for tree in ts.trees():
            for _ in range(int(tree.length)):
                yield tree

    @staticmethod
    def _get_cached_times(dist: 'EmpiricalPhaseTypeDistribution') -> np.ndarray:
        """
        The grid a distribution's curves are cached on: **its own** support, from 0 up to the largest value it
        sampled. Taken from the distribution's reduced (per-replicate) samples, not from the raw arrays, because the
        reduction differs per distribution -- the tree height takes the *maximum* over loci, the total branch length
        the *sum*.

        Each distribution must get its own grid rather than share the tree height's. They live on different scales
        (the total branch length exceeds the tree height by roughly ``2 H_{n-1}``), so a shared grid runs one of them
        off its own support, where both the sampled and the exact curve are zero and the comparison passes while
        asserting nothing.

        :param dist: The distribution whose curves are to be cached.
        :return: The grid.
        """
        return np.linspace(0, float(np.max(dist.samples)), 100)

    def _touch(self, **kwargs: dict) -> None:
        """
        Touch cached properties.

        :param kwargs: Additional keyword arguments.
        """
        self.simulate()

        # force-persist the statistics: _touch/_drop is the serialization contract and must hold even under
        # Settings.cache = False, where the getter would otherwise rebuild them without storing
        for name in ('tree_height', 'total_branch_length', 'sfs', 'fsfs'):
            dist = self.__dict__[name] = getattr(self, name)
            dist._touch(self._get_cached_times(dist))

        # cache the cross-locus joint surface ground truth (per-locus tree height / total branch length at the two
        # loci, separated by recombination) for two-locus scenarios, so it is serialized with the comparison and
        # survives the subsequent _drop(). The single pair (0, 1) over a full grid. The within-tree (single-locus and
        # multi-population) joint surfaces are cached separately by ``Comparison.cache_ground_truth`` from the
        # configured pairwise surface pairs.
        if self.locus_config.n == 2:
            for dist in (self.tree_height, self.total_branch_length):
                dist._cache_loci_joint_surface([(0, 1)])  # full-grid cross-locus surface ground truth

    def _drop(self) -> None:
        """
        Drop simulated data.
        """
        self.heights = None
        self.total_branch_lengths = None
        self.sfs_lengths = None
        self.mutations = None
        self.jsfs_mutations = None
        self.deme_mutations = None

        # the moments are retained by the cached jsfs distribution (referenced before _drop), so this only removes
        # the duplicate reference held on the coalescent
        self.jsfs_moments = None
        self.jsfs_samples = None

        self.tree_height._drop()
        self.total_branch_length._drop()
        self.sfs._drop()
        self.fsfs._drop()

        # caused problems when serializing
        self.demography = None

    @property
    def _resolves_demes(self) -> bool:
        """
        Whether the simulated statistics resolve the demes, which needs the migration history for more than one deme.
        """
        return self.record_migration or self.lineage_config.n_pops == 1

    @cached_property
    def tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """
        Tree height distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(
            self.heights,
            pops=self.lineage_config.pop_names,
            locus_agg=lambda x: x.max(axis=0),
            resolves_demes=self._resolves_demes
        )

    @cached_property
    def total_branch_length(self) -> EmpiricalPhaseTypeDistribution:
        """
        Total branch length distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(self.total_branch_lengths, pops=self.lineage_config.pop_names,
                                              resolves_demes=self._resolves_demes)

    @cached_property
    def sfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Unfolded site-frequency spectrum distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=self.sfs_lengths,
            mutations=self.mutations.T[1:-1].T if self.simulate_mutations else None,
            pops=self.lineage_config.pop_names,
            sfs_dist=UnfoldedSFSDistribution,
            resolves_demes=self._resolves_demes,
            lineage_config=self.lineage_config if self.lineage_distribution is None else self.lineage_distribution,
            locus_config=self.locus_config if self.locus_distribution is None else self.locus_distribution
        )

    @cached_property
    def fsfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Folded site-frequency spectrum distribution.
        """
        self.simulate()

        mid = (self.lineage_config.n + 1) // 2

        # fold SFS branch lengths
        lengths = self.sfs_lengths.copy().T
        lengths[:mid] += lengths[-mid:][::-1]
        lengths[-mid:] = 0

        # fold SFS mutations
        mutations = self.mutations.copy().T
        mutations[:mid] += mutations[-mid:][::-1]
        mutations = mutations[1:self.lineage_config.n // 2 + 1]

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=lengths.T,
            mutations=mutations.T if self.simulate_mutations else None,
            pops=self.lineage_config.pop_names,
            sfs_dist=FoldedSFSDistribution,
            resolves_demes=self._resolves_demes,
            lineage_config=self.lineage_config if self.lineage_distribution is None else self.lineage_distribution,
            locus_config=self.locus_config if self.locus_distribution is None else self.locus_distribution
        )

    #: Highest moment order computed for the empirical joint SFS ground truth.
    _jsfs_max_order: int = 3

    #: Max number of per-replicate joint SFS branch lengths retained for the within-tree joint ground truth
    _jsfs_sample_cap: int = 200000

    @cached_property
    def jsfs(self) -> 'EmpiricalJointSFSDistribution':
        """
        Joint (multi-population) site-frequency spectrum ground truth, accumulated from the same simulated trees as
        the other statistics (see :meth:`simulate`), returned as an :class:`EmpiricalJointSFSDistribution`. The
        descendant configuration of a branch is the number of its sample descendants from each population (its deme of
        origin). Only available for multi-population, single-locus scenarios with a single lineage configuration.

        :raises NotImplementedError: If there is one population or more than one locus.
        :raises ValueError: If the lineage configurations of an initial distribution differ.
        """
        if self.lineage_config.n_pops < 2 or self.locus_config.n != 1:
            raise NotImplementedError(
                "The joint SFS is only available for multi-population, single-locus scenarios."
            )

        self._assert_single_lineage_config("The joint SFS")

        self.simulate()

        return EmpiricalJointSFSDistribution(moments=self.jsfs_moments, samples=self.jsfs_samples,
                                             n_samples=self.n_total)

    @cached_property
    def sfs2(self) -> 'EmpiricalTwoLocusSFSDistribution':
        """
        Two-locus SFS estimated from msprime simulations of two loci separated by the recombination rate of the
        locus configuration, as the per-replicate product of the locus-0 and locus-1 branch lengths of each pair of
        frequency classes, averaged over replicates and returned as an
        :class:`~phasegen.distributions.EmpiricalTwoLocusSFSDistribution`.

        :raises NotImplementedError: If the scenario does not have exactly two loci.
        """
        if self.locus_config.n != 2:
            raise NotImplementedError("The two-locus SFS is only available for two-locus scenarios.")

        n = self.lineage_config.n
        demography = self.demography.to_msprime()
        model = self.get_coalescent_model()

        out = np.zeros((n + 1, n + 1))
        # per-replicate locus-0 / locus-1 SFS branch lengths, retained for the joint distribution / cross-moments
        lefts = np.zeros((self.num_replicates, n + 1))
        rights = np.zeros((self.num_replicates, n + 1))
        for rep, ts in enumerate(self._sim_ancestry(
                self._placements(demography),
                self.num_replicates,
                self._msprime_seed(),
                recombination_rate=self.locus_config.recombination_rate,
                demography=demography,
                model=model,
                ploidy=1,
                end_time=self.end_time
        )):
            t0, t1 = ts.at(0.5), ts.at(1.5)
            left = np.zeros(n + 1)
            right = np.zeros(n + 1)
            for nd in t0.nodes():
                if t0.parent(nd) != -1:
                    left[t0.num_samples(nd)] += t0.branch_length(nd)
            for nd in t1.nodes():
                if t1.parent(nd) != -1:
                    right[t1.num_samples(nd)] += t1.branch_length(nd)
            out += np.outer(left, right)
            lefts[rep] = left
            rights[rep] = right

        return EmpiricalTwoLocusSFSDistribution(out / self.num_replicates, left=lefts, right=rights)

    @cached_property
    def fst(self) -> float:
        r"""
        Hudson's :math:`F_{ST}` ground truth, simulated with msprime: ``1 - mean within-population branch diversity /
        mean between-population branch divergence``, averaged over replicate trees. The diversity averages over the
        populations with at least two sampled lineages and the divergence over the pairs of sampled populations, as
        :attr:`Coalescent.fst <phasegen.distributions.Coalescent.fst>` does.

        :raises ValueError: if fewer than two populations are sampled, none carries two sampled lineages, or the
            lineage configurations of an initial distribution differ.
        """
        import msprime as ms

        pops = self.demography.pop_names
        self._assert_single_lineage_config("F_ST")

        counts = self.lineage_config.lineage_dict
        sampled = [q for q in pops if counts.get(q, 0) >= 1]

        if len(sampled) < 2:
            raise ValueError(f"F_ST requires at least two sampled populations (got {len(sampled)}).")

        if not any(counts[q] >= 2 for q in sampled):
            raise ValueError("F_ST requires a population with at least two sampled lineages.")

        within = np.zeros(self.num_replicates)
        between = np.zeros(self.num_replicates)

        for k, ts in enumerate(ms.sim_ancestry(
                samples=self._msprime_samples,
                sequence_length=1,
                demography=self.demography.to_msprime(),
                model=self.get_coalescent_model(),
                ploidy=1,
                num_replicates=self.num_replicates,
                end_time=self.end_time,
                random_seed=self._msprime_seed(),
        )):
            sample_sets = [ts.samples(population=i) for i in range(len(pops))]

            # within-population diversity (only populations with at least two samples are informative)
            w = [ts.diversity(s, mode='branch') for s in sample_sets if len(s) >= 2]
            # between-population divergence over distinct population pairs
            b = [ts.divergence([sample_sets[i], sample_sets[j]], mode='branch')
                 for i in range(len(pops)) for j in range(i + 1, len(pops))
                 if len(sample_sets[i]) and len(sample_sets[j])]

            within[k] = np.mean(w)
            between[k] = np.mean(b)

        return float(1 - within.mean() / between.mean())

    def _pairwise_coalescence_time(self, pop_i: str, pop_j: str) -> float:
        """
        msprime estimate of the expected coalescence time of one lineage sampled in ``pop_i`` and one in ``pop_j``,
        two in ``pop_i`` when they coincide, simulated for that pair alone as :class:`Coalescent` computes it. It does
        not depend on the sample configuration of this coalescent. Memoized per pair.

        :param pop_i: Name of the first population.
        :param pop_j: Name of the second population.
        :return: The mean coalescence time.
        :raises ValueError: If a population is unknown.
        """
        import msprime as ms

        names = self.demography.pop_names
        for pop in (pop_i, pop_j):
            if pop not in names:
                raise ValueError(f"Unknown population '{pop}'. Available populations: {names}.")

        key = tuple(sorted((pop_i, pop_j)))
        cache = self.__dict__.setdefault('_pairwise_times', {})

        if key not in cache:
            names = self.demography._msprime_names
            samples = {names[pop_i]: 2} if pop_i == pop_j else {names[pop_i]: 1, names[pop_j]: 1}

            times = np.array([ts.first().time(ts.first().root) for ts in ms.sim_ancestry(
                samples=samples,
                sequence_length=1,
                demography=self.demography.to_msprime(),
                model=self.get_coalescent_model(),
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self._msprime_seed(),
            )])

            # an end time bounds the accumulation of the tree height, as in the exact computation
            cache[key] = float(np.mean(times if self.end_time is None else np.minimum(times, self.end_time)))

        return cache[key]

    def f2(self, pop_0: str, pop_1: str) -> float:
        """
        msprime ``f2`` ground truth from simulated pairwise coalescence times. Matches
        :meth:`Coalescent.f2() <phasegen.distributions.Coalescent.f2>`.
        """
        t = self._pairwise_coalescence_time
        return 2 * t(pop_0, pop_1) - t(pop_0, pop_0) - t(pop_1, pop_1)

    def f3(self, pop_target: str, pop_0: str, pop_1: str) -> float:
        """
        msprime ``f3`` ground truth from simulated pairwise coalescence times. Matches
        :meth:`Coalescent.f3() <phasegen.distributions.Coalescent.f3>`.
        """
        t = self._pairwise_coalescence_time
        return t(pop_target, pop_0) + t(pop_target, pop_1) - t(pop_0, pop_1) - t(pop_target, pop_target)

    def f4(self, pop_0: str, pop_1: str, pop_2: str, pop_3: str) -> float:
        """
        msprime ``f4`` ground truth from simulated pairwise coalescence times. Matches
        :meth:`Coalescent.f4() <phasegen.distributions.Coalescent.f4>`.
        """
        t = self._pairwise_coalescence_time
        return t(pop_0, pop_3) + t(pop_1, pop_2) - t(pop_0, pop_2) - t(pop_1, pop_3)

    def to_phasegen(self) -> Coalescent:
        """
        Convert to native phasegen coalescent.

        :return: phasegen coalescent.
        """
        return Coalescent(
            n=self._lineages,
            model=self.model,
            demography=self.demography,
            loci=self._loci,
            recombination_rate=self.locus_config.recombination_rate,
            end_time=self.end_time
        )


class SampledCoalescent(AbstractCoalescent):  # pragma: no cover
    """
    Coalescent whose statistics are estimated from :attr:`SampledCoalescent.n_samples
    <phasegen.distributions.SampledCoalescent.n_samples>` simulated trajectories each, built by
    :meth:`PhaseTypeDistribution.to_empirical() <phasegen.distributions.PhaseTypeDistribution.to_empirical>` on the
    wrapped :class:`~phasegen.distributions.Coalescent` at first access and cached.

    Each statistic is simulated separately from its own child of :class:`numpy.random.SeedSequence` spawned from
    :attr:`seed`, so its draws do not depend on the order of access. The entries of one statistic, such as the bins
    of a spectrum, share their trajectories and can be paired. Different statistics come from independent
    trajectories and cannot.

    The following example estimates the mean tree height and the 90% quantile of every bin of the site-frequency
    spectrum from 1000 trajectories each.

    ::

        sampled = pg.distributions.SampledCoalescent(pg.Coalescent(n=5), n_samples=1000, seed=1)

        height = sampled.tree_height.mean
        q = sampled.sfs.quantile(0.9)

    .. versionadded:: 2.0
    """

    #: Per-statistic spawn keys so each distribution is sampled reproducibly and independently of access order.
    _spawn_keys = dict(tree_height=0, total_branch_length=1, sfs=2, fsfs=3, jsfs=4, sfs2=5)

    def __init__(
            self,
            coalescent: Coalescent,
            n_samples: int = 100000,
            seed: int | np.random.Generator = None
    ) -> None:
        """
        :param coalescent: The exact coalescent to sample from.
        :param n_samples: Number of trajectories to simulate per statistic.
        :param seed: Non-negative integer seed, or a :class:`numpy.random.Generator` from which one is drawn at
            construction. ``None`` draws fresh entropy per statistic.
        :raises ValueError: If ``seed`` is a negative integer.
        """
        if seed is not None and not isinstance(seed, np.random.Generator) and seed < 0:
            raise ValueError(f"The seed must be non-negative, got {seed}.")

        # adopt the wrapped coalescent's configuration: this satisfies the AbstractCoalescent contract and retains
        # the config after the analytic coalescent is dropped (Comparison / serialization need it)
        super().__init__(
            n=coalescent._lineages,
            model=coalescent.model,
            demography=coalescent.demography,
            loci=coalescent._loci,
            end_time=coalescent.end_time
        )

        self._coalescent: Optional[Coalescent] = coalescent

        #: Number of trajectories sampled per statistic.
        self.n_samples: int = n_samples

        #: Random seed.
        self.seed: Optional[int] = int(seed.integers(2 ** 63)) if isinstance(seed, np.random.Generator) else seed

    def _to_empirical(self, name: str):
        """Sample the named analytic distribution into its empirical counterpart, seeded reproducibly."""
        seed = None if self.seed is None else np.random.default_rng(
            np.random.SeedSequence(self.seed, spawn_key=(self._spawn_keys[name],)))
        return getattr(self._coalescent, name).to_empirical(self.n_samples, seed=seed)

    @cached_property
    def tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """Sampled tree height distribution."""
        return self._to_empirical('tree_height')

    @cached_property
    def total_branch_length(self) -> EmpiricalPhaseTypeDistribution:
        """Sampled total branch length distribution."""
        return self._to_empirical('total_branch_length')

    @cached_property
    def sfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """Sampled unfolded site-frequency spectrum distribution."""
        return self._to_empirical('sfs')

    @cached_property
    def fsfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """Sampled folded site-frequency spectrum distribution."""
        return self._to_empirical('fsfs')

    @cached_property
    def jsfs(self) -> EmpiricalJointSFSDistribution:
        """Sampled joint (multi-population) site-frequency spectrum distribution."""
        return self._to_empirical('jsfs')

    @cached_property
    def sfs2(self) -> EmpiricalTwoLocusSFSDistribution:
        """Sampled two-locus site-frequency spectrum distribution."""
        return self._to_empirical('sfs2')

    def _touch(self, **kwargs: dict) -> None:
        """Build and cache the empirical distributions (so the cached stats/surfaces survive ``_drop`` and are
        serialized with the comparison)."""
        times = MsprimeCoalescent._get_cached_times

        self.tree_height._touch(times(self.tree_height))
        self.total_branch_length._touch(times(self.total_branch_length))

        # the single-locus site-frequency spectra (undefined for multiple loci, where ``sfs2`` is used instead)
        if self.locus_config.n == 1:
            self.sfs._touch(times(self.sfs))
            self.fsfs._touch(times(self.fsfs))

            # multi-population: the joint SFS
            if len(self.lineage_config.pop_names) > 1:
                _ = self.jsfs

        # two loci: the cross-locus joint surface ground truth, and the two-locus SFS (single-population only, as in
        # the analytic two-locus block-counting state space)
        if self.locus_config.n == 2:
            for dist in (self.tree_height, self.total_branch_length):
                dist._cache_loci_joint_surface([(0, 1)])
            if len(self.lineage_config.pop_names) == 1:
                _ = self.sfs2

    def _drop(self) -> None:
        """Drop the per-sample data and the analytic coalescent; the cached stats and surfaces are retained."""
        for name in ('tree_height', 'total_branch_length', 'sfs', 'fsfs', 'jsfs', 'sfs2'):
            if name in self.__dict__:
                self.__dict__[name]._drop()

        self._coalescent = None
        self.demography = None

