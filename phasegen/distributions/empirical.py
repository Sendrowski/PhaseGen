"""
Empirical distributions estimated from simulated genealogies -- via msprime (:class:`MsprimeCoalescent`) or
PhaseGen's own trajectory sampler (:class:`SampledCoalescent`) -- together with the containers that compute
statistics from the sampled realisations.
"""

import itertools
import logging
import math
from ..caching import cached_property, cache
from typing import Generator, List, Callable, Tuple, Dict, Iterable, Iterator, NoReturn, Optional, Sequence, Type, \
    TYPE_CHECKING, Union
import numpy as np
from ..coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from ..demography import Demography
from ..initial import InitialDistribution
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalTreeHeightReward, TotalBranchLengthReward, SFSReward, \
    TwoLocusSFSReward, SumReward, ProductReward, RestrictedReward, UnitReward, CombinedReward, JointSFSReward
from ..settings import Settings
from ..spectrum import AbstractSpectrum, SFS, TwoSFS, JointSFS, TwoLocusSFS
from ..utils import parallelize

from .base import DensityAwareDistribution, CumulativeDistributionFunction, DensityFunction, DistributionFunction, \
    QuantileFunction, CallableDistributionFunctions, JointCDF, JointDensity
from .spectra import FoldedSFSDistribution, SFSDistribution, _TajimaSFSMixin, UnfoldedSFSDistribution, \
    JointSFSDistribution, TwoLocusSFSDistribution
from .mutation_configs import MutationConfig, MutationLayout
from ._common import (
    _descendant_config, _frequency_class, _make_hashable, _polymorphic_class, _validate_order, _validate_reward,
    _validate_reward_count, _validate_rewards, _validate_start_time
)
from .coalescent import AbstractCoalescent, Coalescent
from .phase_type import PhaseTypeDistribution

if TYPE_CHECKING:
    import msprime
    import tskit
    from ..visualization import _CurveData
    from matplotlib import pyplot as plt

logger = logging.getLogger('phasegen')

#: The largest Beta-coalescent alpha msprime accepts.
_MSPRIME_BETA_ALPHA_MAX = 1.991

#: The largest number of grid points on which the empirical joint CDF counts the replicates below each point
#: directly. Larger grids bin all replicates against the grid.
_JOINT_CDF_DIRECT_MAX = 16

#: Multiple of the replicate count with which a pairwise coalescence time is simulated, for the f-statistics.
_PAIRWISE_REPLICATE_FACTOR = 10

#: Message of the error raised where the simulated genealogies or sampled trajectories are not held.
_NO_GENEALOGIES = (
    "The {statistic} requires the genealogies simulated by MsprimeCoalescent or the trajectories sampled by "
    "SampledCoalescent, which this {holder} does not hold."
)


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
        per_bin = samples.ndim >= 2
        keys = (list(self._distribution._polymorphic_bins()) if bins is None else list(np.atleast_1d(bins))) \
            if per_bin else []
        if samples.ndim == 2:
            keys = [int(i) for i in keys]
        elif per_bin:
            keys = [tuple(int(c) for c in key) for key in keys]
        columns = [
            int(np.ravel_multi_index(np.atleast_1d(key), samples.shape[1:])) for key in keys
        ] if per_bin else []

        included = samples.reshape(samples.shape[0], -1)[:, columns] if per_bin else samples
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

        y = values.reshape(values.shape[0], -1).T[columns] if per_bin else values[None]
        name = dict(pdf='PDF', cdf='CDF', quantile='quantile function')[self.kind]
        variable = self._distribution._variable

        return _CurveData(
            x=x,
            y=y,
            labels=[str(key) for key in keys] if per_bin else [''],
            xlabel='q' if self.kind == 'quantile' else variable,
            ylabel=dict(pdf=f'f({variable})', cdf=f'F({variable})', quantile='quantile')[self.kind],
            title=f'SFS bin {name}s' if per_bin else self._distribution._titled(name),
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
    """The empirical CDF of ``EmpiricalDistribution`` (see its class docstring), per entry for the samples of a
    spectrum, of shape ``t.shape + shape`` with ``shape`` the shape of one sample."""

    def __call__(self, t) -> 'np.ndarray':
        # sort along the replicate axis, never across the entries of one replicate
        samples = self._distribution.samples
        x = np.sort(samples.reshape(samples.shape[0], -1), axis=0)
        y = np.arange(1, len(samples) + 1) / len(samples)

        if samples.ndim == 1:
            return np.interp(t, x[:, 0], y, left=0.0)

        out = np.stack([np.interp(t, x_, y, left=0.0) for x_ in x.T], axis=-1)

        return out.reshape(np.shape(t) + samples.shape[1:])

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
    """The sample quantile of ``EmpiricalDistribution`` (see its class docstring), per entry for the samples of a
    spectrum."""

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
    """The cell-average density of ``EmpiricalDistribution`` (see its class docstring), per entry for the samples of a
    spectrum, of shape ``(len(t),) + shape`` with ``shape`` the shape of one sample. ``Comparison._cell_average``
    integrates the exact density over the same cells, so both sides estimate the same functional."""

    def __call__(self, t) -> 'np.ndarray':
        samples = self._distribution.samples
        t = np.atleast_1d(np.asarray(t, dtype=float))

        if t.size < 2:
            raise ValueError("The empirical density is a cell average, so it needs a grid of at least two points.")

        edges = np.append(t, 2 * t[-1] - t[-2])
        widths = np.diff(edges)

        if samples.ndim == 1:
            return self._cell_density(samples, edges, widths)

        out = np.stack([self._cell_density(s, edges, widths) for s in samples.reshape(samples.shape[0], -1).T], axis=-1)

        return out.reshape((len(t),) + samples.shape[1:])

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


class _EmpiricalJointSFSFunction:  # pragma: no cover
    """Mixin selecting the bins of an empirical joint spectrum by their descendant configurations, as the functions
    of :class:`~phasegen.distributions.JointSFSDistribution` do."""

    def _configs(self, configs: Sequence[Tuple[int, ...]] | None) -> List[Tuple[int, ...]] | None:
        """
        Validate the descendant configurations of the joint bins.

        :param configs: The descendant configurations, ``None`` for all polymorphic bins.
        :return: The configurations as tuples of integers, ``None`` for all polymorphic bins.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        return None if configs is None else [self._distribution._bin_config(c) for c in configs]

    def _plot_data(self, t: np.ndarray = None, configs: Sequence[Tuple[int, ...]] = None,
                   n_points: int = None) -> '_CurveData':
        """
        The curves :meth:`plot` draws, one per joint bin.

        :param t: Points to evaluate at, as for the function of a spectrum.
        :param configs: The joint bins (descendant configurations) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        return self._empirical_curves(t, self._configs(configs), n_points)

    def plot(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the empirical function of every joint bin at once, one curve per descendant configuration.

        :param ax: Axes to plot on.
        :param t: Points to evaluate at, as for the function of a spectrum.
        :param configs: The joint bins (descendant configurations) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(t=t, configs=configs, n_points=n_points),
                                         file=file, show=show, clear=clear, label=label, title=title, **kwargs)


class _EmpiricalJointSFSCDF(_EmpiricalJointSFSFunction, _EmpiricalCumulativeDistributionFunction):  # pragma: no cover
    """The empirical CDF of every bin of an empirical joint spectrum, with the bins selected by configuration."""


class _EmpiricalJointSFSDensity(_EmpiricalJointSFSFunction, _EmpiricalDensityFunction):  # pragma: no cover
    """The cell-average density of every bin of an empirical joint spectrum, with the bins selected by
    configuration."""


class _EmpiricalJointSFSQuantileFunction(_EmpiricalJointSFSFunction, _EmpiricalQuantileFunction):  # pragma: no cover
    """The sample quantile of every bin of an empirical joint spectrum, with the bins selected by configuration."""

    def _plot_data(self, q: np.ndarray = None, configs: Sequence[Tuple[int, ...]] = None,
                   n_points: int = None) -> '_CurveData':
        """
        The empirical quantile curves :meth:`plot` draws, one per joint bin.

        :param q: Probabilities to evaluate at, as for the quantile function of a spectrum.
        :param configs: The joint bins (descendant configurations) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        return self._empirical_curves(q, self._configs(configs), n_points)

    def plot(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the empirical quantile function of every joint bin at once (value versus probability).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate at, as for the quantile function of a spectrum.
        :param configs: The joint bins (descendant configurations) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, configs=configs, n_points=n_points),
                                         file=file, show=show, clear=clear, label=label, title=title, **kwargs)


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
      its diagonal, and the correlation derived from it, of shape ``shape + shape`` for samples whose entries have
      the shape ``shape``. Entries without variance are set to zero.

    The following example estimates the 90% quantile of the tree height and its CDF at 2 from 1000 sampled
    trajectories.

    ::

        emp = pg.Coalescent(n=5).tree_height.to_empirical(1000, seed=1)

        q = emp.quantile(0.9)
        p = emp.cdf(2.0)
    """
    # the cdf / pdf / quantile evaluation lives on these sample-based function objects; the distribution supplies the
    # ``samples`` they read (the per-bin spectrum case is handled by the same objects, on 2-D samples)
    _cdf_function = _EmpiricalCumulativeDistributionFunction
    _pdf_function = _EmpiricalDensityFunction
    _quantile_function = _EmpiricalQuantileFunction

    #: Label prefixed to the plot titles, such as ``"R_b | R_a = 1"`` for a conditional distribution.
    label: Optional[str] = None

    #: Variable on the x-axis of the curve plots, ``t`` for a time and ``x`` for any other reward.
    _variable: str = 'x'

    def _titled(self, base: str) -> str:
        """
        A plot title prefixed with :attr:`label` when one is set, and capitalized otherwise.

        :param base: The title without the label, such as ``"CDF"``.
        :return: The title.
        """
        return f"{self.label} {base}" if self.label else base[0].upper() + base[1:]

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
        samples = self.samples
        n_blocks = min(n_blocks, samples.shape[0] // 2)

        if n_blocks < 2:
            return

        blocks = samples[:samples.shape[0] // n_blocks * n_blocks]
        blocks = blocks.reshape(n_blocks, -1, *samples.shape[1:])

        # the base class' statistics are plain numpy; the subclasses only wrap the identical numerics in an SFS type
        stats = [EmpiricalDistribution(block) for block in blocks]

        self._standard_errors = {}
        for name in self._STANDARD_ERROR_STATISTICS:
            if name in ('cov', 'corr') and samples.ndim == 1:
                continue  # a 1-D sample has no covariance/correlation: corrcoef is the constant 1, SE a bogus 0
            values = np.array([np.asarray(getattr(s, name), dtype=float) for s in stats])
            self._standard_errors[name] = np.std(values, axis=0) / np.sqrt(n_blocks)

    def _drop(self) -> None:
        """
        Drop simulated samples.
        """
        self.samples = None

    def _cache_joint_surface(self, pairs: Sequence[tuple], n_grid: int = 25, q_max: float = 0.95) -> None:
        """
        Cache the empirical joint CDF and density surface of each pair of entries of a spectrum that provides
        ``joint``, for the full-grid surface comparison, as
        ``self._joint_surface = [(a, b, xs, ys, cdf_grid, pdf_grid), ...]``, serialized with the comparison.

        :param pairs: The pairs ``(a, b)`` of entries, as ``joint`` takes them.
        :param n_grid: Number of grid points per axis.
        :param q_max: Quantile of each entry up to which its axis extends.
        """
        def key(k):
            return tuple(int(c) for c in k) if isinstance(k, (tuple, list)) else int(k)

        self._joint_surface = [
            (key(a), key(b)) + self.joint(a, b)._surface(n_grid, q_max) for a, b in pairs
        ]

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

    @property
    def std(self) -> float | np.ndarray:
        """
        Sample standard deviation, the square root of :attr:`var`, per entry for a spectrum and of the type of
        :attr:`var`. A variance that rounding leaves marginally below zero is read as zero.
        """
        var = self.var

        if isinstance(var, AbstractSpectrum):
            return type(var)(np.maximum(np.asarray(var.data, dtype=float), 0.0) ** 0.5)

        if np.ndim(var) == 0:
            return max(float(var), 0.0) ** 0.5

        return np.maximum(np.asarray(var, dtype=float), 0.0) ** 0.5

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
        return self._pairwise(lambda x: np.cov(x, rowvar=False, bias=True))

    @cached_property
    def corr(self) -> float | np.ndarray:
        """
        Sample correlation matrix, see :class:`~phasegen.distributions.EmpiricalDistribution`.
        """
        return self._pairwise(lambda x: np.corrcoef(x, rowvar=False))

    def _pairwise(self, func: Callable[[np.ndarray], np.ndarray]) -> float | np.ndarray:
        """
        A matrix statistic of the entries, computed on the samples flattened to one row per replicate and reshaped
        to ``shape + shape``.

        :param func: The statistic of a matrix with one column per entry.
        :return: The statistic, with entries without variance set to zero.
        """
        samples = self.samples
        shape = samples.shape[1:]

        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.nan_to_num(func(samples.reshape(samples.shape[0], -1)))

        return out.reshape(shape + shape) if len(shape) > 1 else out

    def moment(self, k: int, center: bool = True) -> float | np.ndarray:
        r"""
        The :math:`k`-th sample moment of the realisations :math:`Y_1, \dots, Y_N` of
        :class:`~phasegen.distributions.EmpiricalDistribution`,

        .. math::

            \frac{1}{N} \sum_{m=1}^{N} (Y_m - \hat\mu)^k,

        with :math:`\hat\mu` the sample mean for a central moment (``center=True`` and :math:`k \ge 2`) and
        :math:`\hat\mu = 0` for a raw moment. :attr:`var` is the central moment of order two, and :attr:`m2`,
        :attr:`m3` and :attr:`m4` are raw moments.

        :param k: Order :math:`k \ge 0` of the moment.
        :param center: Whether to center the moment around the sample mean :math:`\hat\mu`.
        :return: The :math:`k`-th moment, per entry for a spectrum.
        :raises TypeError: If ``k`` is not a number.
        :raises ValueError: If ``k`` is not integral or is negative.
        """
        k = _validate_order(k)
        samples = self.samples
        if center and k > 1:
            samples = samples - np.mean(samples, axis=0)

        return np.mean(samples ** k, axis=0)


class _EmpiricalAccumulating:  # pragma: no cover
    """
    The accumulation over time of an empirical distribution, read from the genealogies simulated by
    :class:`~phasegen.distributions.MsprimeCoalescent` or the trajectories sampled by
    :class:`~phasegen.distributions.SampledCoalescent`, with the plotting code of
    :class:`~phasegen.distributions.PhaseTypeDistribution`.
    """

    #: Accumulation over time of the rewards of the simulated genealogies or sampled trajectories, set by
    #: :class:`MsprimeCoalescent` and :class:`SampledCoalescent` and ``None`` otherwise. Static for backward
    #: compatibility.
    _accumulator: Optional['_EmpiricalAccumulation'] = None

    def _require_accumulator(self) -> '_EmpiricalAccumulation':
        """
        :return: The accumulation over time of the simulated genealogies.
        :raises NotImplementedError: If the distribution does not hold simulated genealogies.
        """
        if self._accumulator is None:
            raise NotImplementedError(_NO_GENEALOGIES.format(statistic="accumulation over time", holder="distribution"))

        return self._accumulator

    def _plot_accumulation_data(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> '_CurveData':
        """
        The accumulation of a moment over time that :meth:`plot_accumulation` draws, one curve per polymorphic bin
        for a spectrum.

        :param k: The order of the moment.
        :param end_times: Times at which to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: Accepted for the signature of the exact distribution.
        :return: The curves.
        """
        return self._require_accumulator()._plot_accumulation_data(k, end_times, rewards, center, permute)

    plot_accumulation = PhaseTypeDistribution.plot_accumulation


class _EmpiricalSFSMixin(_TajimaSFSMixin):  # pragma: no cover
    """
    The bins, their pairwise statistics and the estimators of Tajima's :math:`D` of an empirical site-frequency
    spectrum, the sampled counterparts of those of :class:`~phasegen.distributions.UnfoldedSFSDistribution`.
    """

    #: Whether the spectrum is folded.
    _folded: bool = False

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
    def m3(self) -> SFS:
        """
        Third raw sample moment spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().m3)

    @cached_property
    def m4(self) -> SFS:
        """
        Fourth raw sample moment spectrum, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return SFS(super().m4)

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

    def moment(self, k: int, center: bool = True) -> SFS:
        r"""
        The :math:`k`-th sample moment of every frequency class, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.

        :param k: Order :math:`k \ge 0` of the moment.
        :param center: Whether to center the moment around the sample mean.
        :return: The :math:`k`-th moment spectrum.
        :raises TypeError: If ``k`` is not a number.
        :raises ValueError: If ``k`` is not integral or is negative.
        """
        return SFS(EmpiricalDistribution.moment(self, k, center))

    def _tajima_n(self) -> int:
        """Number of lineages, from the mean, which is retained when the samples are dropped."""
        return len(np.asarray(self.mean.data)) - 1

    def _tajima_mean(self) -> np.ndarray:
        """Mean branch length of the bins ``i = 1 .. n - 1``, zero at the bins this spectrum does not hold."""
        return np.asarray(self.mean.data)[1:self._tajima_n()]

    def _tajima_cov(self) -> np.ndarray:
        """Covariance of the bins ``i, j = 1 .. n - 1``, zero at the bins this spectrum does not hold."""
        n = self._tajima_n()
        return np.asarray(self.cov.data)[1:n, 1:n]

    def _polymorphic_bins(self) -> range:
        """
        The polymorphic frequency classes, ``1, ..., n // 2`` for a folded spectrum and ``1, ..., n - 1`` otherwise.

        :return: The frequency classes.
        """
        n = self._tajima_n()

        return range(1, n // 2 + 1) if self._folded else range(1, n)

    def _polymorphic_bin(self, i: int) -> int:
        """
        Validate a polymorphic frequency class, as ``SFSDistribution._polymorphic_bin``.

        :param i: The frequency class.
        :return: The frequency class as an integer.
        :raises ValueError: If ``i`` is not one of the polymorphic classes of this spectrum.
        """
        indices = self._polymorphic_bins()

        return _polymorphic_class(_frequency_class(i, self._tajima_n()), indices[0], indices[-1])

    def _bin_samples(self, i: int) -> np.ndarray:
        """
        The per-replicate branch lengths of a polymorphic frequency class.

        :param i: The frequency class.
        :return: The branch lengths.
        :raises ValueError: If ``i`` is not a polymorphic class, or the samples have been dropped.
        """
        i = self._polymorphic_bin(i)

        if self.samples is None:
            raise ValueError("The per-replicate samples have been dropped, and the bins need them.")

        return np.asarray(self.samples)[:, i]

    def get_cov(self, i: int, j: int) -> float:
        """
        The sample covariance between the branch lengths of the frequency classes ``i`` and ``j``, the sampled
        counterpart of :meth:`UnfoldedSFSDistribution.get_cov()
        <phasegen.distributions.UnfoldedSFSDistribution.get_cov>`, an entry of :attr:`cov`.

        :param i: The first frequency class.
        :param j: The second frequency class.
        :return: The covariance, zero if a class is not polymorphic in this spectrum.
        :raises ValueError: If ``i`` or ``j`` is not an integer from 0 to :math:`n`.
        """
        n = self._tajima_n()
        i, j = _frequency_class(i, n), _frequency_class(j, n)

        if i not in self._polymorphic_bins() or j not in self._polymorphic_bins():
            return 0

        return float(np.asarray(self.cov.data)[i, j])

    def get_corr(self, i: int, j: int) -> float:
        """
        The sample correlation between the branch lengths of the frequency classes ``i`` and ``j``, the sampled
        counterpart of :meth:`UnfoldedSFSDistribution.get_corr()
        <phasegen.distributions.UnfoldedSFSDistribution.get_corr>`, an entry of :attr:`corr`.

        :param i: The first frequency class.
        :param j: The second frequency class.
        :return: The correlation coefficient, zero if a class is not polymorphic in this spectrum or has no variance.
        :raises ValueError: If ``i`` or ``j`` is not an integer from 0 to :math:`n`.
        """
        n = self._tajima_n()
        i, j = _frequency_class(i, n), _frequency_class(j, n)

        if i not in self._polymorphic_bins() or j not in self._polymorphic_bins():
            return 0

        return float(np.asarray(self.corr.data)[i, j])

    def bin(self, i: int) -> EmpiricalDistribution:
        r"""
        The empirical distribution of the branch length :math:`L_i` of frequency class ``i``, the sampled counterpart
        of :meth:`UnfoldedSFSDistribution.bin() <phasegen.distributions.UnfoldedSFSDistribution.bin>`, whose
        estimators are those of the per-bin ``cdf``, ``pdf`` and ``quantile`` of this spectrum.

        :param i: The frequency class.
        :return: The empirical distribution of :math:`L_i`.
        :raises ValueError: If ``i`` is not a polymorphic frequency class of this spectrum, or the samples have been
            dropped.
        """
        return EmpiricalDistribution(self._bin_samples(i))

    def joint(self, i: int, j: int) -> 'EmpiricalJointDistribution':
        """
        The empirical joint distribution of the branch lengths of the frequency classes ``i`` and ``j``, the sampled
        counterpart of :meth:`UnfoldedSFSDistribution.joint() <phasegen.distributions.UnfoldedSFSDistribution.joint>`.

        :param i: The first frequency class.
        :param j: The second frequency class.
        :return: The empirical joint distribution of :math:`(L_i, L_j)`.
        :raises ValueError: If ``i`` or ``j`` is not a polymorphic frequency class of this spectrum, or the samples
            have been dropped.
        """
        jd = EmpiricalJointDistribution(self._bin_samples(i), self._bin_samples(j))
        jd.label = f"SFS bins ({int(i)}, {int(j)})"

        return jd


class EmpiricalSFSDistribution(_EmpiricalSFSMixin, EmpiricalDistribution):  # pragma: no cover
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

    @property
    def _folded(self) -> bool:
        """Whether the spectrum is folded."""
        return self.folded


class EmpiricalSpectrumDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    Base class for the empirical spectra, which hold the relative frequencies of the mutational configurations of
    any of their layouts among the simulated replicates.
    """

    #: Static for backward compatibility.
    _mutation_counts: Optional[np.ndarray] = None

    #: Static for backward compatibility.
    _count_frequencies: Optional[Dict[Tuple[int, ...], float]] = None

    #: The lineages the replicates start from, recorded by :meth:`mutation_layout`. Static for backward compatibility.
    _layout_lineages: LineageConfig | InitialDistribution | None = None

    #: The loci the replicates start from, recorded by :meth:`mutation_layout`. Static for backward compatibility.
    _layout_loci: LocusConfig | InitialDistribution | None = None

    #: Relative frequency yielded by the most recently started configuration iterator of ``get_mutation_configs()``.
    generated_mass: float = 0

    def get_mutation_configs(self, layout: MutationLayout = None) -> Iterator[Tuple[MutationConfig, float]]:
        """
        Unending iterator over the mutational configurations and their relative frequencies among the simulated
        replicates, in ascending order of the total number of mutations, the sampled counterpart of
        :meth:`UnfoldedSFSDistribution.get_mutation_configs()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs>` with ``order='count'``. The
        frequencies yielded so far sum to :attr:`generated_mass`.

        :param layout: The layout of the configurations, by default that of ``mutation_layout()``.
        :return: An iterator over pairs of configuration and relative frequency.
        :raises ValueError: If the spectrum carries no mutation counts, or ``layout`` is not a layout of this spectrum.
        """
        layout = self.mutation_layout() if layout is None else layout

        self.generated_mass = 0

        for k in itertools.count():
            for config in layout.configs(k):
                p = self.get_mutation_config(config)
                self.generated_mass += p
                yield config, p

    def _mutation_entries(self) -> List[tuple]:
        """
        The polymorphic entries of the spectrum, in the order of the stored counts.

        :return: The array indices of the entries.
        """
        raise NotImplementedError

    def _layout_axes(self) -> Tuple[Tuple[str, ...], ...]:
        """
        The axes of the spectrum arrays of the layouts this spectrum provides.

        :return: The axes of each kind of layout.
        """
        return self.mutation_layout().axes,

    def _entry_groups(self, layout: MutationLayout) -> List[List[int]]:
        """
        The positions in the stored counts of the entries each bin of a layout sums.

        :param layout: The layout.
        :return: One list of positions per bin.
        :raises ValueError: If ``layout`` is not a layout of this spectrum.
        """
        index = {tuple(e): k for k, e in enumerate(self._mutation_entries())}
        positions = [tuple(layout.positions[label]) for b in layout.bins for label in b]

        if layout.axes not in self._layout_axes() or any(p not in index for p in positions):
            raise ValueError(f"The layout {layout!r} does not belong to this spectrum, whose default layout is "
                             f"{self.mutation_layout()!r}.")

        return [[index[tuple(layout.positions[label])] for label in b] for b in layout.bins]

    def _frequencies_by_entry(self) -> Dict[Tuple[int, ...], float]:
        """
        Relative frequency of each tuple of counts over the polymorphic entries, computed once from the counts.

        :return: Dictionary from counts to relative frequency.
        :raises ValueError: If the spectrum carries no mutation counts.
        """
        if self._count_frequencies is None:
            if self._mutation_counts is None:
                raise ValueError("This spectrum carries no mutation counts, so mutational configuration frequencies "
                                 "are unavailable.")

            counts = np.stack([self._mutation_counts[(slice(None),) + tuple(e)] for e in self._mutation_entries()], 1)
            rows, n = np.unique(counts, axis=0, return_counts=True)
            self._count_frequencies = {tuple(int(x) for x in r): k / len(counts) for r, k in zip(rows, n)}

        return self._count_frequencies

    def _layout_frequencies(self, layout: MutationLayout) -> Dict[Tuple[int, ...], float]:
        """
        Relative frequency of each configuration of a layout shown by a replicate, memoized per layout.

        :param layout: The layout.
        :return: Dictionary from the counts of the bins to relative frequency.
        :raises ValueError: If the spectrum carries no mutation counts, or ``layout`` is not a layout of this spectrum.
        """
        frequencies = self._frequencies_by_entry()
        cache = self.__dict__.setdefault('_binned_frequencies', {})
        key = (layout.axes, layout.bins)

        if key not in cache:
            groups = self._entry_groups(layout)
            binned = {}
            for counts, p in frequencies.items():
                config = tuple(sum(counts[k] for k in g) for g in groups)
                binned[config] = binned.get(config, 0) + p
            cache[key] = binned

        return cache[key]

    @property
    def mutation_configs(self) -> Dict[MutationConfig, float]:
        """
        Relative frequency of each configuration of the default layout ``mutation_layout()`` shown by a replicate.

        :raises ValueError: If the spectrum carries no mutation counts.
        """
        layout = self.mutation_layout()

        return {layout.config(c): p for c, p in self._layout_frequencies(layout).items()}

    def get_mutation_config(self, config: Union[MutationConfig, Sequence[int], int]) -> float:
        """
        Relative frequency of a mutational configuration among the simulated replicates, the sampled counterpart of
        :meth:`UnfoldedSFSDistribution.get_mutation_config()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.

        :param config: A :class:`~phasegen.distributions.MutationConfig` of any layout of this spectrum, or one
            mutation count per bin of the default layout, a single count for one bin.
        :return: The fraction of replicates showing the configuration, 0 for a configuration no replicate shows.
        :raises ValueError: If the spectrum carries no mutation counts, ``config`` does not have one non-negative
            integer per bin of the default layout, or its layout is not one of this spectrum.
        """
        self._frequencies_by_entry()

        if not isinstance(config, MutationConfig):
            config = self.mutation_layout().config((config,) if np.isscalar(config) else config)

        return self._layout_frequencies(config.layout).get(tuple(config), 0)

    def _drop(self) -> None:
        """Drop the per-replicate mutation counts, retaining their configuration frequencies."""
        if self._mutation_counts is not None:
            self._frequencies_by_entry()
            self._mutation_counts = None

        self.__dict__.pop('_binned_frequencies', None)

        super()._drop()


class EmpiricalJointSFSDistribution(_EmpiricalAccumulating, EmpiricalSpectrumDistribution):  # pragma: no cover
    r"""
    Empirical joint site-frequency spectrum, built by
    :meth:`JointSFSDistribution.to_empirical() <phasegen.distributions.JointSFSDistribution.to_empirical>` or by
    :class:`~phasegen.distributions.MsprimeCoalescent`, the sampled counterpart of
    :class:`~phasegen.distributions.JointSFSDistribution`. With :math:`L_{m\mathbf{c}}` the branch length of
    replicate :math:`m = 1, \dots, N` whose descendants number :math:`c_p` in population :math:`p`, for the
    descendant vector :math:`\mathbf{c} = (c_0, \dots, c_{P-1})` over :math:`P` populations, it holds the raw moments

    .. math::

        \hat M_o(\mathbf{c}) = \frac{1}{N} \sum_{m=1}^{N} L_{m\mathbf{c}}^o, \qquad o = 1, 2, 3,

    over all replicates, which :attr:`mean`, :attr:`var`, :attr:`m2`, :attr:`m3` and :meth:`moment` up to order three
    return. The other estimators of :class:`~phasegen.distributions.EmpiricalDistribution` apply per descendant
    vector to the stored samples, which may be a capped subset of the replicates.

    The following example estimates the mean joint spectrum of two demes from 1000 sampled trajectories.

    ::

        coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
        ))

        mean = coal.jsfs.to_empirical(1000, seed=1).mean
    """
    _cdf_function = _EmpiricalJointSFSCDF
    _pdf_function = _EmpiricalJointSFSDensity
    _quantile_function = _EmpiricalJointSFSQuantileFunction

    #: Static for backward compatibility.
    _cache: Optional[dict] = None

    #: Static for backward compatibility.
    _standard_errors: dict = {}

    #: Static for backward compatibility.
    _joint_surface: list = []

    def __init__(
            self,
            moments: np.ndarray,
            samples: np.ndarray = None,
            n_samples: int = None,
            mutation_counts: np.ndarray = None,
            lineage_config: LineageConfig | InitialDistribution = None,
            locus_config: LocusConfig | InitialDistribution = None
    ) -> None:
        """
        Initialize the distribution from the raw moments and, optionally, per-replicate samples.

        :param moments: Raw moments per descendant vector of orders one to three, stacked along the first axis, of
            shape ``(3, n_0 + 1, ..., n_{P-1} + 1)``.
        :param samples: Optional per-replicate joint SFS branch lengths, of shape ``(N, n_0 + 1, ...)``, possibly a
            capped subset of the replicates the moments were averaged over.
        :param n_samples: The number of replicates :math:`N` the moments were averaged over. Defaults to the length
            of ``samples``, and must be given when ``samples`` is a capped subset.
        :param mutation_counts: Optional per-replicate mutation counts, of shape ``(N, n_0 + 1, ...)``, from which
            :meth:`get_mutation_config` takes the configuration frequencies.
        :param lineage_config: The lineages the replicates start from, recorded by :meth:`mutation_layout`. By default,
            ``n_p`` lineages in each deme ``pop_p``, with ``n_p + 1`` the extent of axis ``p`` of the moments.
        :param locus_config: The loci the replicates start from, recorded by :meth:`mutation_layout`. By default, one
            locus.
        """
        moments = np.asarray(moments)

        super().__init__(np.zeros((0,) + moments.shape[1:]) if samples is None else samples)

        if samples is None:
            self.samples = None

        #: Raw moments per descendant vector, indexed by order minus one.
        self._moments: np.ndarray = moments

        #: Number of replicates the moments were averaged over, which exceeds ``len(samples)`` for a capped subset.
        self.n_samples: Optional[int] = n_samples if n_samples is not None else (
            None if samples is None else self.samples.shape[0])

        #: Cached full-grid joint surface ground truth: ``[(config_a, config_b, xs, ys, cdf_grid, pdf_grid), ...]``.
        self._joint_surface = []

        #: Per-replicate mutation counts, ``None`` once dropped.
        self._mutation_counts = mutation_counts

        #: The lineages the replicates start from.
        self._layout_lineages = lineage_config

        #: The loci the replicates start from.
        self._layout_loci = locus_config

    def _mutation_entries(self) -> List[Tuple[int, ...]]:
        """
        The polymorphic descendant vectors.

        :return: The descendant vectors.
        """
        return self._polymorphic_bins()

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the joint SFS array, ``(n_0 + 1, ..., n_{P-1} + 1)``, as :attr:`JointSFSDistribution.shape
        <phasegen.distributions.JointSFSDistribution.shape>`.
        """
        return tuple(int(s) for s in self._moments.shape[1:])

    def mutation_layout(self, folded: bool = False) -> MutationLayout:
        r"""
        The layout of the mutational configurations, that of :meth:`JointSFSDistribution.mutation_layout()
        <phasegen.distributions.JointSFSDistribution.mutation_layout>`.

        :param folded: Whether to merge each descendant vector :math:`\mathbf{c}` with its complement.
        :return: The layout.
        """
        lineages = self._layout_lineages
        loci = LocusConfig() if self._layout_loci is None else self._layout_loci

        if lineages is None:
            lineages = LineageConfig({f'pop_{p}': s - 1 for p, s in enumerate(self.shape)})

        return JointSFSDistribution._layout_of(lineages, loci, folded)

    def _bin_config(self, config: Sequence[int]) -> Tuple[int, ...]:
        """
        Validate the descendant configuration of a polymorphic joint SFS bin, as
        ``JointSFSDistribution._bin_config``.

        :param config: The descendant configuration, one count per population.
        :return: The configuration as a tuple of integers.
        :raises ValueError: If ``config`` is not the descendant configuration of a polymorphic joint SFS bin.
        """
        return _descendant_config(config, tuple(s - 1 for s in self.shape))

    def get_cov(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> float:
        """
        The sample covariance between the branch lengths of two descendant configurations, the sampled counterpart of
        :meth:`JointSFSDistribution.get_cov() <phasegen.distributions.JointSFSDistribution.get_cov>`, an entry of
        :attr:`cov`.

        :param config_a: The first descendant configuration.
        :param config_b: The second descendant configuration.
        :return: The covariance.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        return float(np.asarray(self.cov)[self._bin_config(config_a) + self._bin_config(config_b)])

    def bin(self, *config: int) -> EmpiricalDistribution:
        """
        The empirical distribution of the branch length of the joint SFS bin with the given descendant counts per
        population, the sampled counterpart of :meth:`JointSFSDistribution.bin()
        <phasegen.distributions.JointSFSDistribution.bin>`, from the stored samples.

        :param config: The descendant configuration, one count per population.
        :return: The empirical distribution of the bin's branch length.
        :raises ValueError: If ``config`` is not the descendant configuration of a polymorphic joint SFS bin, or the
            per-replicate samples have been dropped.
        """
        config = self._bin_config(config)

        if self.samples is None:
            raise ValueError("The per-replicate samples have been dropped, and the bins need them.")

        return EmpiricalDistribution(self.samples[(slice(None),) + config])

    @property
    def demes(self) -> NoReturn:
        """
        Not available, as the simulation does not record in which deme the branch of a descendant configuration
        resides.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The empirical joint SFS records the branch length of each descendant configuration over all demes, not "
            "the deme the branch resides in, so it has no per-deme marginals."
        )

    @property
    def loci(self) -> Dict[int, 'EmpiricalJointSFSDistribution']:
        """
        The empirical joint spectrum of each locus, keyed by locus index, the sampled counterpart of
        :attr:`JointSFSDistribution.loci <phasegen.distributions.JointSFSDistribution.loci>`. The joint spectrum has
        one locus, whose spectrum is this one.
        """
        loci = _LocusContainer({0: self})
        loci._spectrum = JointSFS
        loci.cov = np.asarray(self.var.data, dtype=float)[None, None]

        with np.errstate(divide='ignore', invalid='ignore'):
            loci.corr = loci.cov / loci.cov

        return loci

    def _polymorphic_bins(self) -> List[Tuple[int, ...]]:
        """
        The polymorphic descendant vectors, all but the empty one and the one holding every lineage.

        :return: The descendant vectors.
        """
        shape = self._moments.shape[1:]
        full = tuple(s - 1 for s in shape)

        return [c for c in np.ndindex(*shape) if c != (0,) * len(shape) and c != full]

    @cached_property
    def mean(self) -> JointSFS:
        r"""
        Sample mean over all replicates, :math:`\hat M_1`.
        """
        return JointSFS(self._moments[0])

    @cached_property
    def var(self) -> JointSFS:
        r"""
        Sample variance over all replicates, :math:`\hat M_2 - \hat M_1^2`.
        """
        return JointSFS(self._moments[1] - self._moments[0] ** 2)

    @cached_property
    def m2(self) -> JointSFS:
        r"""
        Second raw sample moment over all replicates, :math:`\hat M_2`.
        """
        return JointSFS(self._moments[1])

    @cached_property
    def m3(self) -> JointSFS:
        r"""
        Third raw sample moment over all replicates, :math:`\hat M_3`.
        """
        return JointSFS(self._moments[2])

    @cached_property
    def m4(self) -> JointSFS:
        """
        Fourth raw sample moment of the stored samples, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return JointSFS(super().m4)

    def moment(self, k: int, center: bool = True) -> JointSFS:
        r"""
        The :math:`k`-th sample moment of :meth:`EmpiricalDistribution.moment()
        <phasegen.distributions.EmpiricalDistribution.moment>`, from the raw moments over all replicates up to order
        three, where the central moment is :math:`\sum_{o=0}^{k} \binom{k}{o} \hat M_o (-\hat M_1)^{k-o}` with
        :math:`\hat M_0 = 1`, and from the stored samples above.

        :param k: Order :math:`k \ge 0` of the moment.
        :param center: Whether to center the moment around the sample mean.
        :return: The :math:`k`-th moment per descendant vector.
        :raises TypeError: If ``k`` is not a number.
        :raises ValueError: If ``k`` is not integral or is negative, or if ``k`` exceeds three and the per-replicate
            samples have been dropped.
        """
        k = _validate_order(k)

        if k > 3:
            if self.samples is None:
                raise ValueError("Moments above order three need the per-replicate samples, which have been dropped.")

            return JointSFS(super().moment(k, center))

        raw = [np.ones(self._moments.shape[1:])] + list(self._moments)

        if not center or k < 2:
            return JointSFS(raw[k])

        return JointSFS(sum(math.comb(k, o) * raw[o] * (-raw[1]) ** (k - o) for o in range(k + 1)))

    @property
    def data(self) -> np.ndarray:
        """
        The mean joint site-frequency spectrum array.
        """
        return self._moments[0]

    def _drop(self) -> None:
        """Drop the per-replicate samples, retaining their covariance, correlation and block standard errors."""
        if self.samples is not None:
            for name in ('cov', 'corr'):
                self.__dict__[name] = getattr(self, name)

            self._cache_standard_errors()

        super()._drop()

    def _cache_standard_errors(self, n_blocks: int = 100) -> None:
        """
        Cache the block standard errors over the stored samples, scaling those of the moments over all replicates to
        the number of replicates.

        :param n_blocks: Number of blocks.
        """
        super()._cache_standard_errors(n_blocks)

        scale = np.sqrt(len(self.samples) / self.n_samples) if self.n_samples else 1.0

        for name in ('mean', 'var', 'm2', 'm3'):
            if name in self._standard_errors:
                self._standard_errors[name] = self._standard_errors[name] * scale

    def joint(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> 'EmpiricalJointDistribution':
        """
        The empirical joint distribution of the branch lengths of the descendant vectors ``config_a`` and
        ``config_b``, the sampled counterpart of :meth:`JointSFSDistribution.joint()
        <phasegen.distributions.JointSFSDistribution.joint>`.

        :param config_a: The first descendant vector.
        :param config_b: The second descendant vector.
        :return: The empirical joint distribution.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin, or
            the per-replicate samples have been dropped.
        """
        config_a, config_b = self._bin_config(config_a), self._bin_config(config_b)

        if self.samples is None:
            raise ValueError("The per-replicate samples have been dropped, and the joint distribution needs them.")

        jd = EmpiricalJointDistribution(
            self.samples[(slice(None),) + config_a], self.samples[(slice(None),) + config_b]
        )
        jd.label = f"jSFS bins {config_a} x {config_b}"

        return jd

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        """
        The :math:`k`-th sample moment of every bin accumulated from the start time to each end time, the sampled
        counterpart of :meth:`JointSFSDistribution.accumulate()
        <phasegen.distributions.JointSFSDistribution.accumulate>`, with the estimator of
        :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times at which to evaluate the moment.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :param start_time: The start time. By default, that of the coalescent, 0 for MsprimeCoalescent.
        :return: Array of shape ``(len(end_times),) +`` :attr:`shape` with the moment of each bin over time.
        :raises NotImplementedError: If the distribution does not hold simulated genealogies.
        """
        return self._require_accumulator().accumulate(k, end_times, center, permute, start_time)


class EmpiricalTwoLocusSFSDistribution(_EmpiricalAccumulating, EmpiricalSpectrumDistribution):  # pragma: no cover
    r"""
    Empirical two-locus site-frequency spectrum, built by
    :meth:`TwoLocusSFSDistribution.to_empirical() <phasegen.distributions.TwoLocusSFSDistribution.to_empirical>` or
    by :class:`~phasegen.distributions.MsprimeCoalescent`, the sampled counterpart of
    :class:`~phasegen.distributions.TwoLocusSFSDistribution`. Its samples are the products
    :math:`Y_{mij} = L^0_{mi} L^1_{mj}`, where :math:`L^\ell_{mi}` is the branch length subtending :math:`i` of the
    :math:`n` lineages at locus :math:`\ell \in \{0, 1\}` in replicate :math:`m = 1, \dots, N`, and the estimators
    of :class:`~phasegen.distributions.EmpiricalDistribution` apply per pair of classes :math:`(i, j)`, without
    symmetrizing over the two loci. :attr:`corr` is the correlation between :math:`L^0_i` and :math:`L^1_j`, as for
    :attr:`TwoLocusSFSDistribution.corr <phasegen.distributions.TwoLocusSFSDistribution.corr>`. The moment
    statistics are retained when the samples are freed.

    The following example estimates the two-locus spectrum and its cross-locus correlation from 1000 sampled
    trajectories.

    ::

        emp = pg.Coalescent(n=3, loci=2, recombination_rate=1).sfs2.to_empirical(1000, seed=1)

        mean, corr = emp.mean, emp.corr
    """

    #: Static for backward compatibility.
    _cache: Optional[dict] = None

    #: Static for backward compatibility.
    _standard_errors: dict = {}

    #: Static for backward compatibility.
    _joint_surface: list = []

    #: Statistics :meth:`_cache_standard_errors` estimates a standard error for.
    _STANDARD_ERROR_STATISTICS = ('mean', 'var', 'm2', 'm3', 'm4')

    #: Static for backward compatibility.
    _n_lineages: Optional[int] = None

    def __init__(
            self,
            left: np.ndarray,
            right: np.ndarray,
            mutation_counts: np.ndarray = None,
            lineage_config: LineageConfig | InitialDistribution = None,
            locus_config: LocusConfig | InitialDistribution = None
    ) -> None:
        """
        Initialize the distribution from the per-replicate branch lengths of the two loci.

        :param left: Locus-0 SFS branch lengths, of shape ``(N, n + 1)``.
        :param right: Locus-1 SFS branch lengths, of shape ``(N, n + 1)``.
        :param mutation_counts: Optional per-replicate mutation counts of the two loci, of shape ``(N, 2, n + 1)``,
            from which :meth:`get_mutation_config` takes the configuration frequencies.
        :param lineage_config: The lineages the replicates start from, recorded by :meth:`mutation_layout`. By default,
            ``n`` lineages in one deme.
        :param locus_config: The loci the replicates start from, recorded by :meth:`mutation_layout`. By default, two
            loci.
        """
        left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)

        super(EmpiricalDistribution, self).__init__()

        self._cache = None

        #: Per-replicate branch lengths of the two loci, ``None`` once freed for serialization.
        self._left: np.ndarray | None = left
        self._right: np.ndarray | None = right

        #: Number of samples, retained when the samples are freed so that it is recorded in a serialized comparison.
        self.n_samples: int = left.shape[0]

        #: Standard error of each moment statistic, estimated from blocks of the samples, retained when they are freed.
        self._standard_errors: dict = {}

        #: Number of lineages :math:`n`.
        self._n_lineages: int = left.shape[1] - 1

        #: Cached full-grid joint surface ground truth: ``[(i, j, xs, ys, cdf_grid, pdf_grid), ...]``.
        self._joint_surface = []

        #: Per-replicate mutation counts, ``None`` once dropped.
        self._mutation_counts = mutation_counts

        #: The lineages the replicates start from.
        self._layout_lineages = lineage_config

        #: The loci the replicates start from.
        self._layout_loci = locus_config

    def _mutation_entries(self) -> List[Tuple[int, int]]:
        """
        The pairs ``(locus, i)`` of polymorphic classes.

        :return: The pairs.
        """
        return [(locus, i) for locus in (0, 1) for i in range(1, self._n_lineages)]

    @property
    def samples(self) -> np.ndarray | None:
        r"""
        The products :math:`Y_{mij} = L^0_{mi} L^1_{mj}`, of shape ``(N, n + 1, n + 1)``, formed from the branch
        lengths of the two loci on every access, ``None`` once they are freed.
        """
        if self._left is None:
            return None

        return self._left[:, :, None] * self._right[:, None, :]

    @samples.setter
    def samples(self, value: None) -> None:
        """
        Free the branch lengths of the two loci.

        :param value: ``None``.
        :raises AttributeError: If ``value`` is not ``None``, as the samples are formed from the branch lengths.
        """
        if value is not None:
            raise AttributeError("The samples are formed from the branch lengths of the two loci and cannot be set.")

        self._left = None
        self._right = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the two-locus SFS array, ``(n + 1, n + 1)``, as :attr:`TwoLocusSFSDistribution.shape
        <phasegen.distributions.TwoLocusSFSDistribution.shape>`.
        """
        n = len(np.asarray(self.mean.data)) - 1 if self._n_lineages is None else self._n_lineages

        return n + 1, n + 1

    def mutation_layout(self, loci: Sequence[int] = (0, 1), folded: bool = False) -> MutationLayout:
        """
        The layout of the mutational configurations of the two loci, that of
        :meth:`TwoLocusSFSDistribution.mutation_layout()
        <phasegen.distributions.TwoLocusSFSDistribution.mutation_layout>`.

        :param loci: The loci whose mutations are counted.
        :param folded: Whether to merge the classes :math:`i` and :math:`n - i` of a locus into one bin.
        :return: The layout.
        :raises ValueError: If ``loci`` is empty, repeats a locus or holds a locus other than 0 and 1.
        """
        lineages = LineageConfig(self.shape[0] - 1) if self._layout_lineages is None else self._layout_lineages
        locus_config = LocusConfig(n=2) if self._layout_loci is None else self._layout_loci

        return TwoLocusSFSDistribution._layout_of(lineages, locus_config, loci, folded)

    cdf = pdf = quantile = property(TwoLocusSFSDistribution._no_univariate_distribution)
    plot_cdf = bin = TwoLocusSFSDistribution._no_univariate_distribution

    def _no_marginals(self, *args, **kwargs) -> NoReturn:
        """
        Not available for the two-locus spectrum, as for :class:`~phasegen.distributions.TwoLocusSFSDistribution`.
        The per-locus marginals are those of the single-locus spectrum.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no per-locus or per-deme marginals. The per-locus marginals are those of the "
            "single-locus spectrum, such as the empirical sfs of a single-locus coalescent."
        )

    loci = demes = property(_no_marginals)

    def _polymorphic_bins(self) -> List[Tuple[int, int]]:
        """
        The pairs of polymorphic classes.

        :return: The pairs ``(i, j)``.
        """
        n = self._n_lineages

        return [(i, j) for i in range(1, n) for j in range(1, n)]

    @cached_property
    def mean(self) -> TwoLocusSFS:
        """
        Sample mean, see :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return TwoLocusSFS(super().mean)

    @cached_property
    def var(self) -> TwoLocusSFS:
        """
        Sample variance, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return TwoLocusSFS(super().var)

    @cached_property
    def m2(self) -> TwoLocusSFS:
        """
        Second raw sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return TwoLocusSFS(super().m2)

    @cached_property
    def m3(self) -> TwoLocusSFS:
        """
        Third raw sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return TwoLocusSFS(super().m3)

    @cached_property
    def m4(self) -> TwoLocusSFS:
        """
        Fourth raw sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.
        """
        return TwoLocusSFS(super().m4)

    def moment(self, k: int, center: bool = True) -> TwoLocusSFS:
        r"""
        The :math:`k`-th sample moment, see
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>`.

        :param k: Order :math:`k \ge 0` of the moment.
        :param center: Whether to center the moment around the sample mean.
        :return: The :math:`k`-th moment per pair of classes.
        :raises TypeError: If ``k`` is not a number.
        :raises ValueError: If ``k`` is not integral or is negative, or if the samples have been dropped and the moment
            is not among those retained.
        """
        k = _validate_order(k)

        if self._left is None:
            retained = {1: 'mean', 2: 'var' if center else 'm2', 3: None if center else 'm3', 4: None if center else 'm4'}

            if retained.get(k) is None:
                raise ValueError(f"The moment of order {k} needs the per-replicate samples, which have been dropped.")

            return getattr(self, retained[k])

        return TwoLocusSFS(super().moment(k, center))

    @cached_property
    def corr(self) -> TwoLocusSFS:
        r"""
        Sample Pearson correlation between :math:`L^0_i` and :math:`L^1_j` for every pair of classes, the sampled
        counterpart of :attr:`TwoLocusSFSDistribution.corr <phasegen.distributions.TwoLocusSFSDistribution.corr>`.
        Pairs without variance are set to zero.
        """
        a = self._left - self._left.mean(axis=0)
        b = self._right - self._right.mean(axis=0)
        cov = a.T @ b / a.shape[0]

        with np.errstate(divide='ignore', invalid='ignore'):
            return TwoLocusSFS(np.nan_to_num(cov / np.outer(a.std(axis=0), b.std(axis=0))))

    def joint(self, i: int, j: int) -> 'EmpiricalJointDistribution':
        """
        The empirical joint distribution of :math:`L^0_i` and :math:`L^1_j`, the sampled counterpart of
        :meth:`TwoLocusSFSDistribution.joint()
        <phasegen.distributions.TwoLocusSFSDistribution.joint>`.

        :param i: The locus-0 frequency class.
        :param j: The locus-1 frequency class.
        :return: The empirical joint distribution.
        :raises ValueError: If ``i`` or ``j`` is not a polymorphic class from 1 to :math:`n - 1`, or the per-replicate
            samples have been dropped.
        """
        n = self.shape[0] - 1
        i, j = _polymorphic_class(i, 1, n - 1), _polymorphic_class(j, 1, n - 1)

        if self._left is None:
            raise ValueError("The per-replicate samples have been dropped, and the joint distribution needs them.")

        jd = EmpiricalJointDistribution(self._left[:, i], self._right[:, j])
        jd.label = f"locus-0 bin {i} x locus-1 bin {j}"

        return jd

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        r"""
        The :math:`k`-th sample moment of every bin :math:`L^0_i L^1_j`, with the branch lengths accumulated from the
        start time to each end time, the sampled counterpart of :meth:`TwoLocusSFSDistribution.accumulate()
        <phasegen.distributions.TwoLocusSFSDistribution.accumulate>`, with the estimator of
        :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times at which to evaluate the moment.
        :param center: Whether to return the central moment.
        :param start_time: The start time. By default, that of the coalescent, 0 for MsprimeCoalescent.
        :return: Array of shape ``(len(end_times), n + 1, n + 1)`` of the moments, symmetrized over the two loci.
        :raises NotImplementedError: If the distribution does not hold simulated genealogies.
        """
        return self._require_accumulator().accumulate(k, end_times, center, start_time)

    def _drop(self) -> None:
        """Drop the per-replicate samples, retaining the moment statistics and their block standard errors."""
        if self._left is None:
            return

        for name in ('mean', 'var', 'm2', 'm3', 'm4', 'cov', 'corr'):
            self.__dict__[name] = getattr(self, name)

        self._cache_standard_errors()

        super()._drop()


class DictContainer(dict):  # pragma: no cover
    """
    Empirical marginal distributions keyed by deme name or locus index, with their covariance and correlation matrices
    as ``cov`` and ``corr``, ordered as the keys.
    """

    #: Covariance matrix of the marginals.
    cov: Optional[np.ndarray] = None

    #: Correlation matrix of the marginals.
    corr: Optional[np.ndarray] = None

    #: The spectrum type of the per-class covariance and correlation of marginal spectra, ``None`` for scalar
    #: marginals.
    _spectrum: Optional[Type[AbstractSpectrum]] = None

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
        container._spectrum = SFS
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

    def _entry(self, matrix: np.ndarray, d1, d2) -> 'float | np.ndarray | AbstractSpectrum':
        """
        The entry of a matrix of the marginals for a pair of keys.

        :param matrix: The covariance or correlation matrix.
        :param d1: Deme name or locus index of the first marginal distribution.
        :param d2: Deme name or locus index of the second marginal distribution.
        :return: The entry, a spectrum of one value per frequency class for spectra.
        :raises ValueError: If there is no marginal of either key.
        """
        entry = np.atleast_2d(matrix)[self._index(d1), self._index(d2)]

        return entry if self._spectrum is None else self._spectrum(entry)

    def get_cov(self, d1, d2) -> 'float | AbstractSpectrum':
        """
        Get the covariance between two marginal distributions.

        :param d1: Deme name or locus index of the first marginal distribution.
        :param d2: Deme name or locus index of the second marginal distribution.
        :return: The covariance, a spectrum of one covariance per frequency class for spectra.
        :raises ValueError: If there is no marginal of either key.
        """
        return self._entry(self.cov, d1, d2)

    def get_corr(self, d1, d2) -> 'float | AbstractSpectrum':
        """
        Get the correlation coefficient between two marginal distributions.

        :param d1: Deme name or locus index of the first marginal distribution.
        :param d2: Deme name or locus index of the second marginal distribution.
        :return: The correlation coefficient, a spectrum of one coefficient per frequency class for spectra.
        :raises ValueError: If there is no marginal of either key.
        """
        return self._entry(self.corr, d1, d2)


class _DemeContainer(DictContainer):  # pragma: no cover
    """
    The empirical marginal distributions of the demes, keyed by deme name, the sampled counterpart of
    :class:`~phasegen.distributions.MarginalDemeDistributions`.
    """

    @property
    def demes(self) -> dict:
        """
        The empirical distribution of each deme, keyed by population name.
        """
        return dict(self)


class _LocusContainer(DictContainer):  # pragma: no cover
    """
    The empirical marginal distributions of the loci, keyed by locus index, the sampled counterpart of
    :class:`~phasegen.distributions.MarginalLocusDistributions`.
    """

    @property
    def loci(self) -> dict:
        """
        The empirical distribution of each locus, keyed by locus index.
        """
        return dict(self)

    def joint(self, locus1: int, locus2: int) -> 'EmpiricalJointDistribution':
        """
        The empirical joint distribution of the totals at ``locus1`` and at ``locus2``, the sampled counterpart of
        :meth:`MarginalLocusDistributions.joint() <phasegen.distributions.MarginalLocusDistributions.joint>`.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The empirical joint distribution across the two loci.
        :raises ValueError: If either locus does not exist, or the samples have been dropped.
        :raises NotImplementedError: If the marginals are spectra, whose joint distribution is taken per pair of
            frequency classes, as by :meth:`EmpiricalTwoLocusSFSDistribution.joint()
            <phasegen.distributions.EmpiricalTwoLocusSFSDistribution.joint>`.
        """
        locus1, locus2 = int(locus1), int(locus2)

        if locus1 not in self or locus2 not in self:
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        a, b = self[locus1].samples, self[locus2].samples

        if a is None or b is None:
            raise ValueError("The per-replicate samples have been dropped, and the joint distribution needs them.")

        if np.ndim(a) != 1:
            raise NotImplementedError("The joint distribution across loci is defined for a scalar total. For a "
                                      "spectrum, use the joint distribution of a pair of frequency classes.")

        return EmpiricalJointDistribution(a, b)


class EmpiricalPhaseTypeDistribution(_EmpiricalAccumulating, EmpiricalDistribution):  # pragma: no cover
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
        Initialize the distribution from the given realisations.

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

        demes = _DemeContainer(
            {pop: EmpiricalDistribution(self._samples.sum(axis=0)[i]) for i, pop in enumerate(self.pops)}
        )
        for dist in demes.values():
            dist._variable = self._variable

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
        loci = _LocusContainer(
            {i: EmpiricalDistribution(self._samples[i].sum(axis=0)) for i in range(self._samples.shape[0])}
        )
        for dist in loci.values():
            dist._variable = self._variable

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
            joint = EmpiricalJointDistribution(self._locus_samples(l1), self._locus_samples(l2))
            self._loci_joint_surface.append((int(l1), int(l2)) + joint._surface(n_grid, q_max))

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

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        """
        The :math:`k`-th sample moment accumulated from the start time to each end time, defined in
        :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`. A spectrum
        returns one column per site-frequency count.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the reward of the distribution for each factor.
        :param center: Whether to return the central moment.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :param start_time: The start time. By default, that of the coalescent, 0 for MsprimeCoalescent.
        :return: The moment at each end time.
        :raises NotImplementedError: If the distribution does not hold simulated genealogies, or a reward is not
            one they record.
        """
        return self._require_accumulator().accumulate(k, end_times, rewards, center, permute, start_time)

    def moment(
            self,
            k: int,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        r"""
        The :math:`k`-th sample moment, the sampled counterpart of :meth:`PhaseTypeDistribution.moment()
        <phasegen.distributions.PhaseTypeDistribution.moment>`: that of :meth:`EmpiricalDistribution.moment()
        <phasegen.distributions.EmpiricalDistribution.moment>` without rewards and times, and otherwise that of
        :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>` at the end time.

        :param k: The order :math:`k \ge 0` of the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the reward of the distribution for each factor.
        :param start_time: The start time. By default, that of the coalescent, 0 for MsprimeCoalescent.
        :param end_time: The end time. By default, the end time of the coalescent, or absorption.
        :param center: Whether to return the central moment.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :return: The :math:`k`-th moment.
        :raises TypeError: If ``k`` is not a number, or an entry of ``rewards`` is not a
            :class:`~phasegen.rewards.Reward`.
        :raises ValueError: If ``k`` is not integral or is negative, the number of rewards differs from it, the start
            time is negative, or the end time exceeds that of the coalescent.
        :raises NotImplementedError: If rewards or times are given and the distribution does not hold simulated
            genealogies, or a reward is not one they record.
        """
        k = _validate_order(k)

        if rewards is None and start_time is None and end_time is None:
            return EmpiricalDistribution.moment(self, k, center)

        return self._require_accumulator().moment(k, rewards, start_time, end_time, center, permute)

    def joint(self, reward_a: Reward, reward_b: Reward) -> 'EmpiricalJointDistribution':
        """
        The empirical joint distribution of two rewards accumulated by each replicate from the start time to the end
        time of the coalescent, the sampled counterpart of :meth:`PhaseTypeDistribution.joint()
        <phasegen.distributions.PhaseTypeDistribution.joint>`, for the rewards supported by
        :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param reward_a: The first reward.
        :param reward_b: The second reward.
        :return: The empirical joint distribution.
        :raises TypeError: If ``reward_a`` or ``reward_b`` is not a single :class:`~phasegen.rewards.Reward`.
        :raises NotImplementedError: If the distribution does not hold simulated genealogies, or a reward is not
            one they record.
        """
        _validate_reward(reward_a, "reward_a")
        _validate_reward(reward_b, "reward_b")

        accumulator = self._require_accumulator()

        return EmpiricalJointDistribution(accumulator.samples(reward_a), accumulator.samples(reward_b))


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


class _EmpiricalJointCDF(JointCDF):  # pragma: no cover
    r"""
    The empirical joint CDF of :class:`~phasegen.distributions.EmpiricalJointDistribution`, the fraction of
    replicates with :math:`R_a \le x` and :math:`R_b \le y`, NaN at a NaN threshold, drawn as
    :class:`~phasegen.distributions.JointCDF` is.
    """

    def _axis_end(self, axis: str) -> float:
        """
        The end of a plotting axis, the :attr:`Settings.plot_endpoint_quantile` sample quantile of the marginal.

        :param axis: The axis, ``'a'`` or ``'b'``.
        :return: The end of the axis.
        """
        return float(self._distribution.marginal(axis).quantile(Settings.plot_endpoint_quantile))

    def _grid_values(self, xs, ys) -> np.ndarray:
        d = self._distribution
        xs, ys = np.asarray(xs, dtype=float).ravel(), np.asarray(ys, dtype=float).ravel()

        if xs.size * ys.size <= _JOINT_CDF_DIRECT_MAX:
            counts = np.array([[np.count_nonzero((d._a <= x) & (d._b <= y)) for y in ys] for x in xs])
            cdf = counts.reshape(xs.size, ys.size) / len(d._a)
        else:
            ux, ix = np.unique(xs, return_inverse=True)
            uy, iy = np.unique(ys, return_inverse=True)

            # replicate m has R_a <= ux[j] exactly for j >= ja[m], so the cumulated counts over (ja, jb) give the CDF
            ja = np.searchsorted(ux, d._a, side='left')
            jb = np.searchsorted(uy, d._b, side='left')
            shape = (len(ux) + 1, len(uy) + 1)
            counts = np.bincount(ja * shape[1] + jb, minlength=shape[0] * shape[1]).reshape(shape)
            cdf = (counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] / len(d._a))[np.ix_(ix.ravel(), iy.ravel())]

        cdf[np.isnan(xs)] = np.nan
        cdf[:, np.isnan(ys)] = np.nan

        return cdf


class _EmpiricalJointDensity(JointDensity):  # pragma: no cover
    """
    The cell-average density of :class:`~phasegen.distributions.EmpiricalJointDistribution` described there, drawn as
    :class:`~phasegen.distributions.JointDensity` is, with the cells at their centres.
    """

    def __call__(self, x, y) -> np.ndarray:
        r"""
        Evaluate the cell-average density :math:`\hat f_{jl}` on the cells of the outer grid of the arguments.

        :param x: Left cell edges :math:`x_j` along :math:`R_a`, a 1D array of at least two increasing points.
        :param y: Left cell edges :math:`y_l` along :math:`R_b`, a 1D array of at least two increasing points.
        :return: An array of shape ``(len(x), len(y))``.
        :raises ValueError: If a grid has fewer than two points or is not increasing.
        """
        return self._grid_values(np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float))

    _axis_end = _EmpiricalJointCDF._axis_end

    def _grid_values(self, xs, ys) -> np.ndarray:
        d = self._distribution
        edges = []

        for grid in (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)):
            if grid.size < 2 or np.any(np.diff(grid) <= 0):
                raise ValueError("The empirical joint density is a cell average, so it needs increasing grids of at "
                                 "least two points.")
            edges.append(np.append(grid, 2 * grid[-1] - grid[-2]))

        positive = (d._a > 0) & (d._b > 0)
        counts, _, _ = np.histogram2d(d._a[positive], d._b[positive], bins=edges)

        return counts / len(d._a) / np.outer(np.diff(edges[0]), np.diff(edges[1]))

    def _plot_data(self, n_points: int = None, surface: bool = False) -> '_SurfaceData':
        """
        The cells and their densities that :meth:`plot` and :meth:`plot_surface` draw, at the cell centres. By
        default, the number of cells per axis is the fourth root of the sample size, between 10 and the default number
        of points of :class:`~phasegen.distributions.JointDensity`.

        :param n_points: Number of cells per axis, ``None`` for the default.
        :param surface: Whether the default resolution is that of a surface rather than a heatmap.
        :return: The grid and the values on it.
        """
        if n_points is None:
            n_max = Settings.plot_joint_pdf_surface_n_grid if surface else Settings.plot_joint_pdf_n_grid
            n_points = int(np.clip(len(self._distribution._a) ** 0.25, 10, max(n_max, 10)))

        data = super()._plot_data(n_points, surface)
        data.x = data.x + (data.x[1] - data.x[0]) / 2
        data.y = data.y + (data.y[1] - data.y[0]) / 2

        return data


class EmpiricalJointDistribution(CallableDistributionFunctions):  # pragma: no cover
    r"""
    Empirical joint distribution of two accumulated rewards :math:`R_a` and :math:`R_b` from paired realisations, the
    sampled counterpart of :class:`~phasegen.distributions.JointRewardDistribution`, with :math:`R_{am}` and
    :math:`R_{bm}` the rewards of replicate :math:`m = 1, \dots, N`. Its marginals, covariance and correlation are the
    sample estimators of :class:`~phasegen.distributions.EmpiricalDistribution`, with the normalisation :math:`1 / N`.
    A joint distribution has no quantile function.

    - :attr:`cdf`: the fraction of replicates with :math:`R_{am} \le x` and :math:`R_{bm} \le y`.
    - :attr:`pdf`: the fraction of replicates in each cell of the grids :math:`x_0 < x_1 < \dots` and
      :math:`y_0 < y_1 < \dots` of left cell edges, divided by its area, with the last cell as wide as the one before,

      .. math::

          \hat f_{jl} = \frac{\#\{m : R_{am}, R_{bm} > 0,\ x_j \le R_{am} < x_{j+1},\ y_l \le R_{bm} < y_{l+1}\}}
          {N\, (x_{j+1} - x_j)\, (y_{l+1} - y_l)}.

      Replicates with a zero reward are counted in :math:`N` only, so the cells carry the mass
      :math:`\mathbb{P}(R_a > 0,\ R_b > 0)`.

    .. warning::
        :meth:`EmpiricalJointDistribution.conditional()
        <phasegen.distributions.EmpiricalJointDistribution.conditional>` averages over a window of conditioning
        values and is therefore only an approximate check on the exact conditional distribution.

    The following example estimates the correlation of the branch lengths of the first two frequency classes from
    1000 sampled trajectories.

    ::

        emp = pg.Coalescent(n=5).sfs.to_empirical(1000, seed=1)

        corr = emp.joint(1, 2).corr
    """
    _cdf_function = _EmpiricalJointCDF
    _pdf_function = _EmpiricalJointDensity
    _quantile_function = None

    #: Static for backward compatibility.
    label: Optional[str] = None

    def __init__(self, samples_a: np.ndarray, samples_b: np.ndarray) -> None:
        """
        :param samples_a: Per-replicate realisations of the first reward.
        :param samples_b: Per-replicate realisations of the second reward.
        """
        #: Per-replicate realisations of the two rewards.
        self._a = np.asarray(samples_a, dtype=float)
        self._b = np.asarray(samples_b, dtype=float)

        #: Optional human-readable label used in plot titles, such as ``"SFS bins (1, 2)"``.
        self.label: Optional[str] = None

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

        dist = _WindowedConditional(other[mask], cond[mask] - value, float(window))
        dist.label = f"R_{'b' if on == 'a' else 'a'} | R_{on} = {value:g}"

        return dist

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

        dist = EmpiricalDistribution(other[empty])
        dist.label = f"R_{'b' if on == 'a' else 'a'} | R_{on} = 0"

        return float(empty.mean()), dist

    def window_average(self, statistic, on: str, value: float, half_width: float,
                       n_nodes: int = None) -> 'float | np.ndarray':
        r"""
        A statistic of the replicates whose conditioning reward :math:`R_c` lies in the window :math:`W` from
        ``value - half_width`` to ``value + half_width``, the sampled counterpart of
        :meth:`JointRewardDistribution.window_average()
        <phasegen.distributions.JointRewardDistribution.window_average>`. ``statistic`` is applied to the
        :class:`~phasegen.distributions.EmpiricalDistribution` of the other reward over these replicates, which
        estimates the average over :math:`W` weighted by the density of :math:`R_c`.

        :param statistic: Callable taking an :class:`~phasegen.distributions.EmpiricalDistribution` and returning a
            scalar or a 1D array, for example ``lambda c: c.mean`` or ``lambda c: c.cdf(ys)``.
        :param on: The conditioning reward, ``'a'`` or ``'b'``.
        :param value: Centre of the window.
        :param half_width: Half-width of the window, in units of the conditioning reward, positive.
        :param n_nodes: Unused, the average being over the replicates.
        :return: The window average, a float for a scalar statistic and an array of the statistic's shape otherwise.
        :raises ValueError: If ``on`` is not ``'a'`` or ``'b'``, if ``half_width`` is not positive and finite, if
            ``value`` is not finite, if the window reaches zero, where the conditioning reward may have an atom, or if
            no replicate falls in the window.
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        if not 0 < half_width < np.inf:
            raise ValueError(f"The half-width of the conditioning window must be positive and finite, got "
                             f"{half_width:g}.")
        if not np.isfinite(value):
            raise ValueError(f"The centre of the conditioning window must be finite, got {value:g}.")

        lo, hi = value - half_width, value + half_width

        if lo <= 0.0:
            raise ValueError(
                f"The conditioning window [{lo:g}, {hi:g}] reaches 0, where R_{on} has an atom. Narrow the window or "
                f"condition further from the origin."
            )

        cond, other = (self._a, self._b) if on == 'a' else (self._b, self._a)
        mask = np.abs(cond - value) <= half_width

        if not mask.any():
            raise ValueError(f"No replicate falls in the conditioning window [{lo:g}, {hi:g}].")

        average = np.asarray(statistic(EmpiricalDistribution(other[mask])), dtype=float)

        return float(average) if average.ndim == 0 else average

    @property
    def mean(self) -> np.ndarray:
        r"""The pair of sample means of :math:`R_a` and :math:`R_b`."""
        return np.array([self._a.mean(), self._b.mean()])

    def moment(self, order_a: int = 1, order_b: int = 1, center: bool = False) -> float:
        r"""
        The sample cross-moment :math:`N^{-1} \sum_{m=1}^{N} R_{am}^{j_a} R_{bm}^{j_b}` of orders
        :math:`j_a, j_b \ge 0`, uncentered by default, the sampled counterpart of
        :meth:`JointRewardDistribution.moment() <phasegen.distributions.JointRewardDistribution.moment>`.

        :param order_a: The order :math:`j_a` of :math:`R_a`.
        :param order_b: The order :math:`j_b` of :math:`R_b`.
        :param center: Whether to center around the sample means.
        :return: The cross-moment.
        """
        a = self._a - self._a.mean() if center else self._a
        b = self._b - self._b.mean() if center else self._b

        return float((a ** order_a * b ** order_b).mean())

    def _surface(self, n_grid: int, q_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The joint CDF on a grid spanning each reward up to its ``q_max`` quantile, and the density as its mixed
        second difference.

        :param n_grid: Number of grid points per axis.
        :param q_max: Quantile of each reward up to which its axis extends.
        :return: The grids ``xs`` and ``ys``, and the CDF and density on ``xs x ys``.
        """
        xs = np.linspace(0.0, float(np.quantile(self._a, q_max)), n_grid)
        ys = np.linspace(0.0, float(np.quantile(self._b, q_max)), n_grid)
        cdf = self.cdf._grid_values(xs, ys)
        pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)

        return xs, ys, cdf, pdf

    @property
    def cov(self) -> float:
        """The sample covariance of the two rewards, with the normalisation of
        :class:`~phasegen.distributions.EmpiricalDistribution`."""
        return float(np.cov(self._a, self._b, bias=True)[0, 1])

    @property
    def corr(self) -> float:
        """The empirical Pearson correlation of the two rewards."""
        return float(np.corrcoef(self._a, self._b)[0, 1])


class EmpiricalPhaseTypeSFSDistribution(_EmpiricalSFSMixin, EmpiricalPhaseTypeDistribution,
                                        EmpiricalSpectrumDistribution):  # pragma: no cover
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

    #: Whether the mutation counts are resolved by the deme in which each mutation occurs. Static for backward
    #: compatibility.
    _mutations_by_deme: bool = False

    def __new__(cls, *args, **kwargs) -> 'EmpiricalPhaseTypeSFSDistribution':
        """
        Create the spectrum, a folded one as an instance of the subclass with the layouts of
        :class:`~phasegen.distributions.FoldedSFSDistribution`.

        :param args: The arguments of :meth:`__init__`.
        :param kwargs: The keyword arguments of :meth:`__init__`.
        :return: The uninitialized spectrum.
        """
        sfs_dist = kwargs.get('sfs_dist', args[3] if len(args) > 3 else None)

        if cls is EmpiricalPhaseTypeSFSDistribution and isinstance(sfs_dist, type) and \
                issubclass(sfs_dist, FoldedSFSDistribution):
            cls = _EmpiricalFoldedSFSDistribution

        return super().__new__(cls)

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
        :param mutations: Unfolded mutation counts summed over loci, per replicate, deme in which the mutation occurs
            and frequency class, of shape ``(N, demes, n + 1)``, or per replicate and frequency class, of shape
            ``(N, n + 1)``, for counts that do not resolve the demes. ``None`` for a spectrum without mutations.
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

        #: Per-replicate mutation counts, ``None`` for a spectrum without mutations and once dropped.
        self._mutation_counts = mutations

        #: Whether the mutation counts are resolved by the deme in which each mutation occurs.
        self._mutations_by_deme = mutations is not None and np.ndim(mutations) == 3

        self.resolves_demes = resolves_demes

        #: The lineages the replicates start from.
        self._layout_lineages: LineageConfig | InitialDistribution = lineage_config

        #: The loci the replicates start from.
        self._layout_loci: LocusConfig | InitialDistribution = locus_config

        #: Atom-conditional ground truth: ``[(i, j, on, mass, dist), ...]``, see
        #: ``_cache_atom_conditional``. Survives ``_drop`` and is serialized with the comparison.
        self._atom_conditional: list = []

        #: Cached windowed-conditional ground truth, see ``_cache_windowed_conditional``.
        self._windowed_conditional: list = []

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

        return _DemeContainer._of_spectra(
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

        return _LocusContainer._of_spectra(
            {i: EmpiricalSFSDistribution(data[i], folded=self._folded) for i in range(data.shape[0])}, data
        )

    @property
    def _folded(self) -> bool:
        """Whether the spectrum is folded."""
        return issubclass(self._sfs_dist, FoldedSFSDistribution)

    def mutation_layout(self, folded: bool = False, demes: bool = False) -> MutationLayout:
        """
        The layout of the mutational configurations, that of :meth:`UnfoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>`.

        :param folded: Whether to merge the classes :math:`i` and :math:`n - i` into the bin of the smaller one.
        :param demes: Whether to resolve each bin by the deme in which the mutation occurs.
        :return: The layout.
        """
        return SFSDistribution._layout_of(
            LineageConfig(self.n) if self._layout_lineages is None else self._layout_lineages,
            LocusConfig() if self._layout_loci is None else self._layout_loci,
            folded=folded,
            demes=demes
        )

    def _mutation_entries(self) -> List[Tuple[int, ...]]:
        """
        The polymorphic entries of the stored counts, the unfolded classes ``(i,)``, or ``(p, i)`` per deme ``p`` for
        counts resolved by deme.

        :return: The entries.
        """
        classes = range(1, self.n)

        if self._mutations_by_deme:
            return [(p, i) for p in range(len(self.pops)) for i in classes]

        return [(i,) for i in classes]

    def _layout_axes(self) -> Tuple[Tuple[str, ...], ...]:
        """
        The axes of the spectrum arrays of the plain and the deme-resolved layouts.

        :return: The axes of each kind of layout.
        """
        return ('class',), ('deme', 'class')

    def _entry_groups(self, layout: MutationLayout) -> List[List[int]]:
        """
        The positions in the stored counts of the entries each bin of a layout sums. A class of a layout without demes
        sums the counts of all demes.

        :param layout: The layout.
        :return: One list of positions per bin.
        :raises ValueError: If ``layout`` is not a layout of this spectrum, or resolves the demes while the counts do
            not.
        """
        if layout.axes == ('deme', 'class') and not self._mutations_by_deme:
            raise ValueError(
                "The deme-resolved configurations need the deme in which each mutation occurs, which "
                "MsprimeCoalescent records with record_migration=True."
            )

        if layout.axes != ('class',) or not self._mutations_by_deme:
            return super()._entry_groups(layout)

        index = {e: k for k, e in enumerate(self._mutation_entries())}
        groups = [[(p, int(layout.positions[label][0])) for label in b for p in range(len(self.pops))]
                  for b in layout.bins]

        if any(e not in index for g in groups for e in g):
            raise ValueError(f"The layout {layout!r} does not belong to this spectrum, whose default layout is "
                             f"{self.mutation_layout()!r}.")

        return [[index[e] for e in g] for g in groups]

    def moment(
            self,
            k: int,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> SFS:
        r"""
        The :math:`k`-th sample moment of every frequency class, the sampled counterpart of
        :meth:`UnfoldedSFSDistribution.moment() <phasegen.distributions.UnfoldedSFSDistribution.moment>`: that of
        :meth:`EmpiricalDistribution.moment() <phasegen.distributions.EmpiricalDistribution.moment>` without rewards
        and times, and otherwise that of :meth:`EmpiricalPhaseTypeSFSDistribution.accumulate()
        <phasegen.distributions.EmpiricalPhaseTypeSFSDistribution.accumulate>` at the end time.

        :param k: The order :math:`k \ge 0` of the moment.
        :param rewards: Sequence of :math:`k` rewards, each combined with the reward of the bin. By default, the
            reward of the distribution for each factor.
        :param start_time: The start time. By default, that of the coalescent, 0 for MsprimeCoalescent.
        :param end_time: The end time. By default, the end time of the coalescent, or absorption.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :return: The :math:`k`-th moment spectrum.
        :raises TypeError: If ``k`` is not a number, or an entry of ``rewards`` is not a
            :class:`~phasegen.rewards.Reward`.
        :raises ValueError: If ``k`` is not integral or is negative, the number of rewards differs from it, the start
            time is negative, or the end time exceeds that of the coalescent.
        :raises NotImplementedError: If rewards or times are given and the distribution does not hold simulated
            genealogies, or a reward is not one they record.
        """
        k = _validate_order(k)

        if rewards is None and start_time is None and end_time is None:
            return _EmpiricalSFSMixin.moment(self, k, center)

        accumulator = self._require_accumulator()
        end = accumulator._end_time if end_time is None else end_time

        return SFS(accumulator.accumulate(k, [end], rewards, center, permute, start_time)[0])

    def get_accumulation(
            self,
            k: int,
            i: int,
            end_times: Iterable[float] | float,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray | float:
        """
        The accumulation of the :math:`k`-th sample moment of bin ``i``, column ``i`` of :meth:`accumulate`, the
        sampled counterpart of :meth:`UnfoldedSFSDistribution.get_accumulation()
        <phasegen.distributions.UnfoldedSFSDistribution.get_accumulation>`.

        :param k: The order of the moment.
        :param i: The site-frequency count.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Sequence of k rewards, each combined with the reward of the bin.
        :param center: Whether to center the moment around the mean.
        :param permute: Accepted for the signature of the exact distribution.
        :param start_time: The start time. By default, that of the coalescent, 0 for MsprimeCoalescent.
        :return: The moment, a float for a single time and an array for a sequence of times.
        :raises NotImplementedError: If the distribution does not hold simulated genealogies.
        """
        return self._require_accumulator().get_accumulation(k, i, end_times, rewards, center, permute, start_time)


class _EmpiricalFoldedSFSDistribution(EmpiricalPhaseTypeSFSDistribution):  # pragma: no cover
    """
    Empirical folded site-frequency spectrum, an :class:`~phasegen.distributions.EmpiricalPhaseTypeSFSDistribution`
    built for :class:`~phasegen.distributions.FoldedSFSDistribution`, with its layouts.
    """

    def mutation_layout(self, demes: bool = False) -> MutationLayout:
        """
        The layout of the mutational configurations, that of :meth:`FoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.FoldedSFSDistribution.mutation_layout>`.

        :param demes: Whether to resolve each bin by the deme in which the mutation occurs.
        :return: The layout.
        """
        return super().mutation_layout(folded=True, demes=demes)


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

    @staticmethod
    def _locus_migrations(j: int, ts, ctx: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The migrations covering the midpoint of locus ``j``, the unit interval ``[j, j + 1)``, in time order, with the
        migration table of the replicate read once into ``ctx``.

        :param j: The locus.
        :param ts: The tree sequence of the replicate.
        :param ctx: The shared per-replicate data.
        :return: The times, nodes, source and destination populations of the migrations.
        """
        if 'migrations' not in ctx:
            ctx['migrations'] = ts.tables.migrations

        migrations = ctx['migrations']
        point = j + 0.5
        hit = np.flatnonzero((migrations.left <= point) & (migrations.right > point))
        hit = hit[np.argsort(migrations.time[hit], kind='stable')]

        return migrations.time[hit], migrations.node[hit], migrations.source[hit], migrations.dest[hit]


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

        # the coalescences of this tree and the migrations of its lineages at locus ``j``, in time order
        m_time, m_node, m_source, m_dest = self._locus_migrations(j, ts, ctx)

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


class _TrajectoryStatistics(_ReplicateStatistic):  # pragma: no cover
    """
    The rows of each tree that ``_Trajectories`` reads: one per lineage and deme of residence, and one per occupied
    deme and interval between consecutive events with more than one lineage, holding the share of the lineages in it.
    """

    def __init__(self, n_pops: int, origin: np.ndarray, end_time: float = None, axis: np.ndarray = None) -> None:
        """
        :param n_pops: Number of demes.
        :param origin: Deme axis of each msprime population id, by which the leaves are counted.
        :param end_time: Time at which the simulation ends, ``None`` for absorption.
        :param axis: Deme axis of each msprime population id, ``None`` without migration recording.
        """
        self._n_pops = n_pops
        self._origin = origin
        self._end_time = np.inf if end_time is None else float(end_time)
        self._axis = axis

        #: Per tree, rows ``(replicate, locus, deme, start, end, leaves, c_0, ..., c_{P-1})`` of the lineages.
        self._lineages: List[np.ndarray] = []

        #: Per tree, rows ``(replicate, locus, deme, start, end, share)`` of the intervals between events.
        self._shares: List[np.ndarray] = []

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        axis = self._axis
        n_pops = self._n_pops

        # the migrations of each lineage at locus j, in time order
        moves = {}
        if axis is not None:
            for time, node, _, dest in zip(*self._locus_migrations(j, ts, ctx)):
                moves.setdefault(int(node), []).append((float(time), int(axis[dest])))

        # the descendant vector of each node, which is its number of leaves for one deme
        below = {}
        if n_pops > 1:
            for u in tree.nodes(order='postorder'):
                vec = [0] * n_pops
                if tree.is_sample(u):
                    vec[self._origin[tree.population(u)]] += 1
                for c in tree.children(u):
                    vec = [a + b for a, b in zip(vec, below[c])]
                below[u] = vec

        unfinished = tree.num_roots > 1
        rows = []

        for u in tree.nodes():
            parent = tree.parent(u)

            # a single root carries no branch
            if parent == -1 and not unfinished:
                continue

            start = tree.time(u)
            end = tree.time(parent) if parent != -1 else self._end_time
            leaves = tree.get_num_leaves(u)
            deme = 0 if axis is None else int(axis[tree.population(u)])
            vec = below[u] if n_pops > 1 else (leaves,)

            for time, dest in moves.get(u, ()):
                rows.append((i, j, deme, start, time, leaves, *vec))
                start, deme = time, dest

            rows.append((i, j, deme, start, end, leaves, *vec))

        lineages = np.array(rows, dtype=float).reshape(-1, 6 + n_pops)
        self._lineages.append(lineages)

        # the number of lineages per deme between consecutive events
        starts, ends, demes = lineages[:, 3], lineages[:, 4], lineages[:, 2].astype(int)
        grid = np.unique(np.concatenate([starts, ends]))
        counts = np.zeros((grid.size, self._n_pops))
        np.add.at(counts, (np.searchsorted(grid, starts), demes), 1)
        np.add.at(counts, (np.searchsorted(grid, ends), demes), -1)
        counts = np.cumsum(counts, axis=0)[:-1]
        total = counts.sum(axis=1)

        k, d = np.nonzero((counts > 0) & (total[:, None] > 1))
        self._shares.append(np.column_stack([
            np.full(k.size, i), np.full(k.size, j), d, grid[k], grid[k + 1], counts[k, d] / total[k]
        ]))

    def records(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: The rows of the lineages and of the intervals between events.
        """
        return (np.concatenate(self._lineages) if self._lineages else np.zeros((0, 6 + self._n_pops)),
                np.concatenate(self._shares) if self._shares else np.zeros((0, 6)))


class _Trajectories:  # pragma: no cover
    r"""
    Rewards accumulated over a time window :math:`[s, t]` by the replicates of :class:`MsprimeCoalescent`. Replicate
    :math:`m` accumulates

    .. math::

        R_m(s, t) = \sum_{\text{rows of } m} w \max\{0, \min(b, t) - \max(a, s)\},

    over the rows of ``_TrajectoryStatistics`` that the reward selects, each with interval :math:`[a, b]` and weight
    :math:`w`, the share of the lineages in its deme for the tree height and one for the branch lengths.
    """

    #: The rewards that are read from the simulated genealogies, for error messages.
    _supported = (
        "TreeHeightReward, TotalTreeHeightReward, TotalBranchLengthReward, UnfoldedSFSReward, FoldedSFSReward, "
        "TwoLocusSFSReward and JointSFSReward, their restrictions to a locus or a deme by RestrictedReward or "
        "CombinedReward, and sums of them by SumReward"
    )

    def __init__(
            self,
            lineages: np.ndarray,
            shares: np.ndarray,
            n_replicates: int,
            n_loci: int,
            n: int,
            pops: List[str],
            resolves_demes: bool,
            sizes: Tuple[int, ...] = None
    ) -> None:
        """
        :param lineages: Rows ``(replicate, locus, deme, start, end, leaves, c_0, ..., c_{P-1})`` of the lineages.
        :param shares: Rows ``(replicate, locus, deme, start, end, share)`` of the intervals between events.
        :param n_replicates: Number of replicates.
        :param n_loci: Number of loci.
        :param n: Number of sampled lineages.
        :param pops: Population names, in the order of the deme axis.
        :param resolves_demes: Whether the rows resolve the demes.
        :param sizes: Number of sampled lineages per deme, by default ``n`` in one deme.
        """
        self._lineages = lineages
        self._shares = shares
        self.n_replicates = n_replicates
        self.n_loci = n_loci
        self.n = n
        self.pops = pops
        self.resolves_demes = resolves_demes
        self.sizes = (n,) if sizes is None else tuple(sizes)

    def _terms(self, reward: Reward) -> List[tuple]:
        """
        The terms whose sum is the accumulated reward, each ``(kind, locus, deme, leaves, config)``, where ``kind``
        is ``'height'`` for the lineage shares, ``'length'`` for the lineages and ``'max_height'`` for the largest
        per-locus height, ``config`` is a descendant vector, and ``None`` selects all loci, demes, leaf counts or
        descendant vectors.

        :param reward: The reward.
        :return: The terms.
        :raises ValueError: If a locus, deme, frequency class or descendant vector does not exist, or the demes are
            not resolved.
        :raises NotImplementedError: If the reward is not read from the simulated genealogies.
        """
        if isinstance(reward, TreeHeightReward):
            return [('max_height' if self.n_loci > 1 else 'height', None, None, None, None)]

        if isinstance(reward, TotalTreeHeightReward):
            return [('height', None, None, None, None)]

        if isinstance(reward, TotalBranchLengthReward):
            return [('length', None, None, None, None)]

        if isinstance(reward, SFSReward):
            return [('length', None, None, tuple(reward._block_sizes(self.n)), None)]

        if isinstance(reward, TwoLocusSFSReward):
            _polymorphic_class(reward.count, 1, self.n - 1)
            self._check_locus(reward.locus)

            return [('length', reward.locus, None, (reward.count,), None)]

        if isinstance(reward, JointSFSReward):
            config = reward.config

            if len(config) != len(self.sizes) or not all(0 <= c <= s for c, s in zip(config, self.sizes)) \
                    or not any(config):
                raise ValueError(
                    f"The descendant vector must have one entry per population, the entry of population p lying in "
                    f"0, ..., n_p for the sample sizes {self.sizes}, and at least one non-zero entry, got {config}."
                )

            return [('length', None, None, None, config)]

        if isinstance(reward, SumReward):
            return [term for r in reward.rewards for term in self._terms(r)]

        # a product reward, combined rewards among them, is read when a single factor differs from the unit reward
        if isinstance(reward, ProductReward):
            factors = [r for r in reward.rewards if not isinstance(r, UnitReward)]

            if len(factors) == 1:
                return self._terms(factors[0])

        if isinstance(reward, RestrictedReward):
            terms = self._terms(reward.rewards[0])

            if reward.locus is not None:
                self._check_locus(reward.locus)
                terms = [('height' if kind == 'max_height' else kind, reward.locus, deme, leaves, config)
                         for kind, locus, deme, leaves, config in terms if locus in (None, reward.locus)]

            if reward.pop is not None:
                deme = self._deme(reward.pop)

                if any(kind == 'max_height' for kind, *_ in terms):
                    raise NotImplementedError(
                        f"TreeHeightReward does not decompose additively over the {self.n_loci} loci, so the "
                        f"restriction to a deme is ill-posed. Restrict to a single locus as well, or use a reward "
                        f"that is additive over loci such as TotalTreeHeightReward or TotalBranchLengthReward."
                    )

                terms = [(kind, locus, deme, leaves, config) for kind, locus, d, leaves, config in terms
                         if d in (None, deme)]

            return terms

        raise NotImplementedError(
            f"{type(reward).__name__} is not read from the simulated genealogies. Supported are {self._supported}."
        )

    def _check_locus(self, locus: int) -> None:
        """
        :param locus: The locus index.
        :raises ValueError: If the locus does not exist.
        """
        if not 0 <= locus < self.n_loci:
            raise ValueError(f"Locus {locus} does not exist.")

    def _deme(self, pop: str) -> int:
        """
        :param pop: The population name.
        :return: The deme index of the population.
        :raises ValueError: If the population does not exist or the demes are not resolved.
        """
        if pop not in self.pops:
            raise ValueError(f"Population {pop} does not exist.")

        if not self.resolves_demes:
            raise ValueError(
                "Per-deme statistics of MsprimeCoalescent require the migration history. "
                "Simulate with record_migration=True."
            )

        return self.pops.index(pop)

    def _term(
            self,
            kind: str,
            locus: Optional[int],
            deme: Optional[int],
            leaves: Optional[Tuple[int, ...]],
            config: Optional[Tuple[int, ...]],
            start_time: float,
            end_times: np.ndarray
    ) -> np.ndarray:
        """
        The accumulation of one term, see ``_terms``.

        :param kind: ``'height'``, ``'length'`` or ``'max_height'``.
        :param locus: The locus, ``None`` for all.
        :param deme: The deme, ``None`` for all.
        :param leaves: The numbers of leaves, ``None`` for all.
        :param config: The descendant vector, ``None`` for all.
        :param start_time: The start time of the window.
        :param end_times: The end times of the window.
        :return: The accumulation of shape ``(len(end_times), n_replicates)``.
        """
        if kind == 'max_height':
            return np.max([self._term('height', l, deme, None, None, start_time, end_times)
                           for l in range(self.n_loci)], axis=0)

        rows = self._shares if kind == 'height' else self._lineages
        mask = np.ones(len(rows), dtype=bool)

        if locus is not None:
            mask &= rows[:, 1] == locus

        if deme is not None:
            mask &= rows[:, 2] == deme

        if leaves is not None:
            mask &= np.isin(rows[:, 5], leaves)

        if config is not None:
            mask &= np.all(rows[:, 6:] == config, axis=1)

        rows = rows[mask]
        replicates = rows[:, 0].astype(int)
        weights = rows[:, 5] if kind == 'height' else np.ones(len(rows))
        lower = np.maximum(rows[:, 3], start_time)

        out = np.empty((len(end_times), self.n_replicates))
        for e, t in enumerate(end_times):
            overlap = np.clip(np.minimum(rows[:, 4], t) - lower, 0, None)
            out[e] = np.bincount(replicates, weights=weights * overlap, minlength=self.n_replicates)

        return out

    def accumulated(self, reward: Reward, start_time: float, end_times: np.ndarray) -> np.ndarray:
        """
        The reward accumulated by each replicate from the start time to each end time.

        :param reward: The reward.
        :param start_time: The start time.
        :param end_times: The end times.
        :return: The accumulated reward of shape ``(len(end_times), n_replicates)``.
        :raises ValueError: If a locus, deme or frequency class does not exist, or the demes are not resolved.
        :raises NotImplementedError: If the reward is not read from the simulated genealogies.
        """
        out = np.zeros((len(end_times), self.n_replicates))

        for term in self._terms(reward):
            out += self._term(*term, start_time, end_times)

        return out


class _SampledTrajectories:  # pragma: no cover
    r"""
    The sojourns of the trajectories that :class:`SampledCoalescent` samples, recorded on first use. Trajectory
    :math:`m` accumulates over the window :math:`[s, t]` the reward

    .. math::

        R_m(s, t) = \sum_{\text{sojourns of } m} r(x) \max\{0, \min(b, t) - \max(a, s)\},

    over its sojourns :math:`[a, b]` in the states :math:`x`, with :math:`r(x)` the reward rate of state :math:`x`.
    """

    def __init__(
            self,
            dist: PhaseTypeDistribution,
            n_samples: int,
            seed: np.random.SeedSequence,
            end_time: Optional[float],
            lineage_config: LineageConfig
    ) -> None:
        """
        :param dist: The distribution whose state space the trajectories visit.
        :param n_samples: Number of trajectories.
        :param seed: Seed sequence of the sampler.
        :param end_time: End time of the coalescent, ``None`` for absorption.
        :param lineage_config: Lineage configuration of the coalescent.
        """
        self._dist = dist
        self.n_replicates = n_samples
        self._seed = seed
        self.end_time = end_time
        self.lineage_config = lineage_config

        #: The trajectory, state, entry time and exit time of each sojourn, ``None`` until sampled.
        self._sojourns: Optional[Tuple[np.ndarray, ...]] = None

    def _trajectory_records(self) -> '_SampledTrajectories':
        """
        :return: These trajectories, sampled on first use.
        """
        if self._sojourns is None:
            path = []
            self._dist._sample(self.n_replicates, rewards=[self._dist.reward], rng=np.random.default_rng(self._seed),
                               path=path)
            self._sojourns = tuple(np.concatenate(c) for c in zip(*path)) if path else (
                np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0))

        return self

    def accumulated(self, reward: Reward, start_time: float, end_times: np.ndarray) -> np.ndarray:
        """
        The reward accumulated by each trajectory from the start time to each end time.

        :param reward: The reward.
        :param start_time: The start time.
        :param end_times: The end times.
        :return: The accumulated reward of shape ``(len(end_times), n_replicates)``.
        """
        trajectories, states, entries, exits = self._sojourns
        rates = np.asarray(reward._get(self._dist.state_space), dtype=float)[states]
        lower = np.maximum(entries, start_time)

        out = np.empty((len(end_times), self.n_replicates))
        for e, t in enumerate(end_times):
            overlap = np.clip(np.minimum(exits, t) - lower, 0, None)

            # a sojourn without end accrues nothing at a zero rate
            with np.errstate(invalid='ignore'):
                weights = np.where(rates != 0, rates * overlap, 0.0)

            out[e] = np.bincount(trajectories, weights=weights, minlength=self.n_replicates)

        return out


class _EmpiricalAccumulation:  # pragma: no cover
    """
    Sample moments of rewards accumulated over a time window by the replicates of :class:`MsprimeCoalescent` or the
    trajectories of :class:`SampledCoalescent`, with the plotting code of
    :class:`~phasegen.distributions.PhaseTypeDistribution`.
    """

    _reward_names = staticmethod(PhaseTypeDistribution._reward_names)

    _plot_accumulation_data = PhaseTypeDistribution._plot_accumulation_data

    plot_accumulation = PhaseTypeDistribution.plot_accumulation

    def __init__(
            self,
            coalescent: 'MsprimeCoalescent | _SampledTrajectories',
            reward: Reward,
            start_time: float = 0.0
    ) -> None:
        """
        :param coalescent: The coalescent, or the sampled trajectories, whose replicates accumulate the rewards.
        :param reward: The default reward.
        :param start_time: The default start time.
        """
        self._coalescent = coalescent
        self.reward = reward
        self.start_time = start_time

    @staticmethod
    def _rewards(k: int, rewards: Optional[Sequence[Reward]], default: Reward) -> Tuple[Reward, ...]:
        """
        Validate the rewards of a moment of order ``k``.

        :param k: The order of the moment.
        :param rewards: Sequence of ``k`` rewards, ``None`` for the default reward for each factor.
        :param default: The default reward.
        :return: The rewards.
        :raises ValueError: If a single reward is passed or the number of rewards differs from ``k``.
        :raises TypeError: If an entry is not a :class:`~phasegen.rewards.Reward`.
        """
        _validate_rewards(rewards, k)
        rewards = (default,) * k if rewards is None else tuple(rewards)
        _validate_reward_count(rewards, k)

        return rewards

    @staticmethod
    def _moment(values: List[np.ndarray], center: bool, size: int) -> np.ndarray:
        r"""
        The sample cross-moment :math:`\frac{1}{N} \sum_{m=1}^N \prod_{i=1}^k (R_{i,m} - \hat\mu_i)` of
        :math:`k` rewards over :math:`N` replicates, with :math:`\hat\mu_i` the sample mean of reward :math:`i` for
        a central moment (``center`` and :math:`k \ge 2`) and zero otherwise.

        :param values: The realisations :math:`R_{i,m}` of each reward, of shape ``(size, N)``.
        :param center: Whether to center the moment.
        :param size: The number of moments, one for order zero.
        :return: The moments, of shape ``(size,)``.
        """
        if not values:
            return np.ones(size)

        if center and len(values) > 1:
            values = [v - v.mean(axis=1, keepdims=True) for v in values]

        return np.mean(np.prod(values, axis=0), axis=1)

    @property
    def _end_time(self) -> float:
        """The end time of the coalescent, infinite for absorption."""
        return np.inf if self._coalescent.end_time is None else self._coalescent.end_time

    def _check_window(self, start_time: float, end_times: np.ndarray) -> None:
        """
        :param start_time: The start time.
        :param end_times: The end times.
        :raises ValueError: If the start time is negative, or an end time exceeds the end time of the coalescent, at
            which the simulation stops.
        """
        _validate_start_time(start_time)

        if np.any(end_times > self._end_time):
            raise ValueError(
                f"The end times must not exceed the end time of the coalescent ({self._end_time:g}), at which the "
                f"simulation stops, got {np.max(end_times):g}."
            )

    def _estimate(self, rewards: Tuple[Reward, ...], start_time: float, end_times: np.ndarray, center: bool) -> np.ndarray:
        """
        :param rewards: The rewards.
        :param start_time: The start time.
        :param end_times: The end times.
        :param center: Whether to center the moment.
        :return: The sample moment at each end time.
        :raises ValueError: If the start time is negative, or an end time exceeds the end time of the coalescent, at
            which the simulation stops.
        """
        self._check_window(start_time, end_times)

        if not rewards:
            return np.ones(len(end_times))

        records = self._coalescent._trajectory_records()
        values = {r: records.accumulated(r, start_time, end_times) for r in set(rewards)}

        return self._moment([values[r] for r in rewards], center, len(end_times))

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        """
        See :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param k: The order of the moment.
        :param end_times: The end times.
        :param rewards: Sequence of ``k`` rewards, by default :attr:`reward` for each factor.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :param start_time: The start time, by default :attr:`start_time`.
        :return: The moment at each end time.
        """
        k = _validate_order(k)
        rewards = self._rewards(k, rewards, self.reward)
        end_times = np.asarray(list(end_times), dtype=float)

        return self._estimate(rewards, self.start_time if start_time is None else start_time, end_times, center)

    def moment(
            self,
            k: int = 1,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        See :meth:`MsprimeCoalescent.moment() <phasegen.distributions.MsprimeCoalescent.moment>`.

        :param k: The order of the moment.
        :param rewards: Sequence of ``k`` rewards, by default :attr:`reward` for each factor.
        :param start_time: The start time, by default :attr:`start_time`.
        :param end_time: The end time, by default the end time of the coalescent.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :return: The moment.
        """
        k = _validate_order(k)
        rewards = self._rewards(k, rewards, self.reward)
        end = self._end_time if end_time is None else end_time
        start = self.start_time if start_time is None else start_time

        return float(self._estimate(rewards, start, np.array([end]), center)[0])

    def samples(self, reward: Reward) -> np.ndarray:
        """
        :param reward: The reward.
        :return: The reward accumulated by each replicate from :attr:`start_time` to the end time of the coalescent.
        """
        records = self._coalescent._trajectory_records()

        return records.accumulated(reward, self.start_time, np.array([self._end_time]))[0]

    def _default_end_times(self) -> np.ndarray:
        """
        Default times of moment accumulation plots: :attr:`Settings.plot_n_grid` points up to the
        :attr:`Settings.plot_endpoint_quantile` quantile of the sampled tree height, or up to the end time of the
        coalescent if it is finite.

        :return: The times.
        """
        if self._coalescent.end_time is not None:
            end = self._coalescent.end_time
        else:
            end = float(np.quantile(self.samples(TreeHeightReward()), Settings.plot_endpoint_quantile))

        return np.linspace(0, end, Settings.plot_n_grid)


class _EmpiricalSFSAccumulation(_EmpiricalAccumulation):  # pragma: no cover
    """
    The accumulation of every bin of a site-frequency spectrum, with the ``accumulate`` interface and the plotting
    code of :class:`~phasegen.distributions.SFSDistribution`. The bins are those of ``sfs_dist``.
    """

    _plot_accumulation_data = SFSDistribution._plot_accumulation_data

    def __init__(
            self,
            coalescent: 'MsprimeCoalescent | _SampledTrajectories',
            sfs_dist: Type[SFSDistribution],
            start_time: float = 0.0
    ) -> None:
        """
        :param coalescent: The coalescent, or the sampled trajectories, whose replicates accumulate the rewards.
        :param sfs_dist: The exact spectrum class whose bins are accumulated.
        :param start_time: The default start time.
        """
        super().__init__(coalescent, UnitReward(), start_time)

        self._sfs_dist = sfs_dist
        self.lineage_config = coalescent.lineage_config

    def _get_indices(self) -> np.ndarray:
        """The polymorphic bins of the spectrum."""
        return self._sfs_dist._get_indices(self)

    def _get_sfs_reward(self, i: int) -> SFSReward:
        """The reward of bin ``i``."""
        return self._sfs_dist._get_sfs_reward(self, i)

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        """
        The moment of every bin, each reward combined with the reward of the bin.

        :param k: The order of the moment.
        :param end_times: The end times.
        :param rewards: Sequence of ``k`` rewards, by default :attr:`reward` for each factor.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :param start_time: The start time, by default :attr:`start_time`.
        :return: Array of shape ``(len(end_times), n + 1)``, one column per site-frequency count.
        """
        k = _validate_order(k)
        rewards = self._rewards(k, rewards, self.reward)
        end_times = np.asarray(list(end_times), dtype=float)
        start_time = self.start_time if start_time is None else start_time

        out = np.zeros((len(end_times), self.lineage_config.n + 1))
        for i in self._get_indices():
            bin_rewards = tuple(CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards)
            out[:, i] = self._estimate(bin_rewards, start_time, end_times, center)

        return out

    def get_accumulation(
            self,
            k: int,
            i: int,
            end_times: Iterable[float] | float,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray | float:
        """
        The moment of bin ``i``, each reward combined with the reward of the bin.

        :param k: The order of the moment.
        :param i: The site-frequency count.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Sequence of ``k`` rewards, by default :attr:`reward` for each factor.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :param start_time: The start time, by default :attr:`start_time`.
        :return: The moment, a float for a single time and an array for a sequence of times.
        """
        k = _validate_order(k)
        rewards = self._rewards(k, rewards, self.reward)
        scalar = np.ndim(end_times) == 0
        times = np.asarray([end_times] if scalar else list(end_times), dtype=float)

        accumulation = self._estimate(
            tuple(CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards),
            self.start_time if start_time is None else start_time, times, center
        )

        return float(accumulation[0]) if scalar else accumulation


class _EmpiricalJointSFSAccumulation(_EmpiricalAccumulation):  # pragma: no cover
    """
    The accumulation of every bin of a joint site-frequency spectrum, with the ``accumulate`` interface and the
    plotting code of :class:`~phasegen.distributions.JointSFSDistribution`.
    """

    _plot_accumulation_data = JointSFSDistribution._plot_accumulation_data

    def __init__(self, coalescent: 'MsprimeCoalescent | _SampledTrajectories', start_time: float = 0.0) -> None:
        """
        :param coalescent: The coalescent, or the sampled trajectories, whose replicates accumulate the rewards.
        :param start_time: The default start time.
        """
        super().__init__(coalescent, UnitReward(), start_time)

        #: Shape of the joint SFS array, ``(n_0 + 1, ..., n_{P-1} + 1)``.
        self.shape = tuple(int(n) + 1 for n in coalescent.lineage_config.lineages)

    def _get_configs(self) -> List[Tuple[int, ...]]:
        """The descendant vectors of the polymorphic bins, all but the empty one and the one of every lineage."""
        full = tuple(s - 1 for s in self.shape)

        return [c for c in np.ndindex(*self.shape) if any(c) and c != full]

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        """
        The moment of every bin, of the reward of the bin.

        :param k: The order of the moment.
        :param end_times: The end times.
        :param center: Whether to return the central moment.
        :param permute: Accepted for the signature of the exact distribution.
        :param start_time: The start time, by default :attr:`start_time`.
        :return: Array of shape ``(len(end_times),) + shape``.
        """
        k = _validate_order(k)
        end_times = np.asarray(list(end_times), dtype=float)
        start_time = self.start_time if start_time is None else start_time

        out = np.zeros((len(end_times),) + self.shape)
        for config in self._get_configs():
            rewards = (CombinedReward([self.reward, JointSFSReward(config)]),) * k
            out[(slice(None),) + config] = self._estimate(rewards, start_time, end_times, center)

        return out


class _EmpiricalTwoLocusSFSAccumulation(_EmpiricalAccumulation):  # pragma: no cover
    """
    The accumulation of every bin :math:`L^0_i L^1_j` of a two-locus site-frequency spectrum, with the ``accumulate``
    interface and the plotting code of :class:`~phasegen.distributions.TwoLocusSFSDistribution`.
    """

    _plot_accumulation_data = TwoLocusSFSDistribution._plot_accumulation_data

    _get_indices = TwoLocusSFSDistribution._get_indices

    def __init__(self, coalescent: 'MsprimeCoalescent | _SampledTrajectories', start_time: float = 0.0) -> None:
        """
        :param coalescent: The coalescent, or the sampled trajectories, whose replicates accumulate the rewards.
        :param start_time: The default start time.
        """
        super().__init__(coalescent, UnitReward(), start_time)

        self.lineage_config = coalescent.lineage_config

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        """
        The moment of every bin, of the product of the per-locus branch lengths, symmetrized over the two loci.

        :param k: The order of the moment.
        :param end_times: The end times.
        :param center: Whether to return the central moment.
        :param start_time: The start time, by default :attr:`start_time`.
        :return: Array of shape ``(len(end_times), n + 1, n + 1)``.
        """
        k = _validate_order(k)
        end_times = np.asarray(list(end_times), dtype=float)
        start_time = self.start_time if start_time is None else start_time
        self._check_window(start_time, end_times)

        indices = self._get_indices()
        n = self.lineage_config.n
        out = np.zeros((len(end_times), n + 1, n + 1))

        records = self._coalescent._trajectory_records() if k > 0 else None
        lengths = {
            (locus, i): records.accumulated(CombinedReward([self.reward, TwoLocusSFSReward(locus, i)]),
                                            start_time, end_times)
            for locus in (0, 1) for i in indices
        } if k > 0 else {}

        for i in indices:
            for j in indices:
                products = [lengths[0, i] * lengths[1, j] for _ in range(k)]
                out[:, i, j] = self._moment(products, center, len(end_times))

        return (out + out.transpose(0, 2, 1)) / 2


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

    #: Lineage trajectories of the replicates, simulated on first use by ``_trajectory_records``. Static for
    #: backward compatibility.
    _trajectories: Optional[_Trajectories] = None

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
        :param seed: Non-negative integer random seed. ``None`` draws one from fresh entropy.
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

        #: Raw moments of the joint SFS per descendant configuration, of the orders one up to a fixed maximum order.
        self.jsfs_moments: np.ndarray | None = None

        #: Per-replicate joint SFS branch lengths of a capped subset of the replicates.
        self.jsfs_samples: np.ndarray | None = None

        #: Actual number of replicates simulated and averaged over (``num_replicates`` rounded down to a multiple of
        #: ``n_threads``), set by :meth:`simulate`.
        self.n_total: int | None = None

        #: Number of replicates.
        self.num_replicates: int = num_replicates

        #: Mutation rate.
        self.mutation_rate: float = mutation_rate

        #: Number of threads, at most ``num_replicates``.
        self.n_threads: int = max(1, min(n_threads, num_replicates))

        #: Whether to parallelize computations.
        self.parallelize: bool = parallelize

        #: Whether to record migrations.
        self.record_migration: bool = record_migration

        #: Whether to simulate mutations.
        self.simulate_mutations: bool = simulate_mutations

        #: Random seed, drawn from fresh entropy at construction if none is given.
        self.seed: int = int(np.random.default_rng().integers(2 ** 63)) if seed is None else seed

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

    def _msprime_seed(self) -> int:
        """
        The msprime seed of :attr:`seed`, wrapped into msprime's range :math:`[1, 2^{32} - 1]`, which it leaves
        unchanged. It seeds the statistics simulated outside the batches of :meth:`simulate`.

        :return: The msprime seed.
        """
        return (self.seed - 1) % (2 ** 32 - 1) + 1

    def _batch_seeds(self) -> List[np.random.SeedSequence]:
        """
        The seed sequence of each of the :attr:`n_threads` batches, spawned from :attr:`seed`.

        :return: One seed sequence per batch.
        """
        return np.random.SeedSequence(self.seed).spawn(self.n_threads)

    @staticmethod
    def _msprime_seeds(seed: np.random.SeedSequence, n: int) -> List[int]:
        """
        Draw ``n`` msprime seeds, in msprime's range :math:`[1, 2^{32} - 1]`, from a seed sequence.

        :param seed: Seed sequence.
        :param n: Number of seeds.
        :return: The msprime seeds.
        """
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
            random_seed: int,
            **kwargs
    ) -> Iterator['tskit.TreeSequence']:
        """
        Simulate replicates whose starting configurations are drawn from the weights of ``placements``. The
        replicates are grouped by starting configuration.

        :param placements: Pairs of the weight and the keyword arguments placing the samples, see ``_placements``.
        :param num_replicates: Number of replicates.
        :param random_seed: msprime seed, which also draws the starting configurations.
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

        self._simulate(main=True, trajectories=False)

    def _simulate(self, main: bool, trajectories: bool) -> None:
        """
        Simulate the replicates, from the seeds of :meth:`simulate`, and store the requested statistics.

        :param main: Whether to store the statistics of :meth:`simulate`.
        :param trajectories: Whether to store the lineage trajectories of ``_trajectory_records``.
        """
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

        def simulate_batch(seed: np.random.SeedSequence) -> dict:
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
            tree_stats = None
            if main:
                tree_stats = (_MigrationTreeStatistics(n_loci, n_pops, num_replicates, sample_size, axis)
                              if self.record_migration
                              else _TreeStatistics(n_loci, n_pops, num_replicates, sample_size))
            jsfs_stats = (_JointSFSStatistics(num_replicates, jsfs_max_order, jsfs_shape, jsfs_sample_cap)
                          if main and compute_jsfs else None)
            mutation_stats = (_MutationStatistics(n_loci, n_pops, num_replicates, sample_size, self.mutation_rate,
                                                  jsfs_shape=jsfs_shape if compute_jsfs else None,
                                                  axis=axis if self.record_migration else None)
                              if main and self.simulate_mutations else None)
            trajectory_stats = (_TrajectoryStatistics(n_pops, axis, end_time, axis if self.record_migration else None)
                                if trajectories else None)
            stats = [s for s in (tree_stats, jsfs_stats, mutation_stats, trajectory_stats) if s is not None]

            # iterate over the tree sequences once, feeding every statistic
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):

                # map each sample to the index of its sampling population (deme of origin) for the joint SFS
                ctx = {}
                if main and compute_jsfs:
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

            records = trajectory_stats.records() if trajectories else None

            if not main:
                return dict(trajectories=records)

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
                deme_mutations=mutation_stats.by_deme if mutation_stats is not None else None,
                trajectories=records
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

        # combine the lineage trajectories across threads, numbering the replicates of each thread after those of
        # the previous ones
        if trajectories:
            lineages, shares = [], []
            for b, batch in enumerate(batches):
                for rows, combined in zip(batch['trajectories'], (lineages, shares)):
                    combined.append(rows + np.eye(1, rows.shape[1])[0] * b * num_replicates)

            self._trajectories = _Trajectories(
                lineages=np.concatenate(lineages),
                shares=np.concatenate(shares),
                n_replicates=n_total,
                n_loci=self.locus_config.n,
                n=sample_size,
                pops=self.lineage_config.pop_names,
                resolves_demes=self._resolves_demes,
                sizes=tuple(int(s) for s in self.lineage_config.lineages)
            )

        if not main:
            return

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

    #: Names of the distributions that :meth:`_touch` persists.
    _distributions: Tuple[str, ...] = ('tree_height', 'total_branch_length', 'sfs', 'fsfs', 'jsfs', 'sfs2')

    def _touch(self, dists: Iterable[str] = None) -> None:
        """
        Simulate and persist the named distributions, so that their cached statistics survive :meth:`_drop` and are
        serialized. The tree height and total branch length of two loci also persist their cross-locus joint surface.

        :param dists: Names among :attr:`_distributions`. By default, every one the scenario defines: the joint SFS
            needs several demes, one locus and one lineage configuration, the two-locus SFS two loci.
        """
        if dists is not None:
            dists = list(dists)
            unknown = [name for name in dists if name not in self._distributions]
            if unknown:
                raise ValueError(f"Unknown distributions {unknown}, expected names among {self._distributions}.")

        self.simulate()

        if dists is None:
            dists = [name for name in self._distributions if name not in ('jsfs', 'sfs2')]

            if self.lineage_config.n_pops > 1 and self.locus_config.n == 1 and (
                    self.lineage_distribution is None
                    or all(c == self.lineage_config for c in self.lineage_distribution.configs)
            ):
                dists.append('jsfs')

            if self.locus_config.n == 2:
                dists.append('sfs2')

        for name in dists:
            # force-persist the statistics: _touch/_drop is the serialization contract and must hold even under
            # Settings.cache = False, where the getter would otherwise rebuild them without storing
            dist = self.__dict__[name] = getattr(self, name)

            if name in ('tree_height', 'total_branch_length', 'sfs', 'fsfs'):
                dist._touch(self._get_cached_times(dist))

            # the cross-locus joint surface ground truth (per-locus value at the two loci, separated by
            # recombination), the single pair (0, 1) over a full grid. The within-tree (single-locus and
            # multi-population) joint surfaces are cached separately by ``Comparison.cache_ground_truth`` from the
            # configured pairwise surface pairs.
            if name in ('tree_height', 'total_branch_length') and self.locus_config.n == 2:
                dist._cache_loci_joint_surface([(0, 1)])

    def _drop(self) -> None:
        """
        Drop the simulated data and the per-replicate samples of the persisted distributions.
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

        self._trajectories = None

        for name in self._distributions:
            if name in self.__dict__:
                self.__dict__[name]._drop()
                self.__dict__[name]._accumulator = None

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

        dist = EmpiricalPhaseTypeDistribution(
            self.heights,
            pops=self.lineage_config.pop_names,
            locus_agg=lambda x: x.max(axis=0),
            resolves_demes=self._resolves_demes
        )
        dist._accumulator = _EmpiricalAccumulation(self, TreeHeightReward())
        dist._variable = 't'

        return dist

    @cached_property
    def total_branch_length(self) -> EmpiricalPhaseTypeDistribution:
        """
        Total branch length distribution.
        """
        self.simulate()

        dist = EmpiricalPhaseTypeDistribution(self.total_branch_lengths, pops=self.lineage_config.pop_names,
                                              resolves_demes=self._resolves_demes)
        dist._accumulator = _EmpiricalAccumulation(self, TotalBranchLengthReward())

        return dist

    def _sfs_mutation_counts(self) -> Optional[np.ndarray]:
        """
        The unfolded mutation counts of the spectra, summed over loci, resolved by the deme in which each mutation
        occurs when the simulation resolves the demes.

        :return: Counts of shape ``(N, demes, n + 1)``, of shape ``(N, n + 1)`` when the demes are not resolved, or
            ``None`` without simulated mutations.
        """
        if not self.simulate_mutations:
            return None

        if not self._resolves_demes:
            return self.mutations.sum(axis=(0, 1))

        by_deme = self.mutations if self.deme_mutations is None else self.deme_mutations

        return np.moveaxis(by_deme.sum(axis=0), 1, 0)

    @cached_property
    def sfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Unfolded site-frequency spectrum distribution.
        """
        self.simulate()

        dist = EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=self.sfs_lengths,
            mutations=self._sfs_mutation_counts(),
            pops=self.lineage_config.pop_names,
            sfs_dist=UnfoldedSFSDistribution,
            resolves_demes=self._resolves_demes,
            lineage_config=self.lineage_config if self.lineage_distribution is None else self.lineage_distribution,
            locus_config=self.locus_config if self.locus_distribution is None else self.locus_distribution
        )
        dist._accumulator = _EmpiricalSFSAccumulation(self, UnfoldedSFSDistribution)

        return dist

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

        dist = EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=lengths.T,
            mutations=self._sfs_mutation_counts(),
            pops=self.lineage_config.pop_names,
            sfs_dist=FoldedSFSDistribution,
            resolves_demes=self._resolves_demes,
            lineage_config=self.lineage_config if self.lineage_distribution is None else self.lineage_distribution,
            locus_config=self.locus_config if self.locus_distribution is None else self.locus_distribution
        )
        dist._accumulator = _EmpiricalSFSAccumulation(self, FoldedSFSDistribution)

        return dist

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

        dist = EmpiricalJointSFSDistribution(
            moments=self.jsfs_moments,
            samples=self.jsfs_samples,
            n_samples=self.n_total,
            mutation_counts=self.jsfs_mutations,
            lineage_config=self.lineage_config if self.lineage_distribution is None else self.lineage_distribution,
            locus_config=self.locus_config if self.locus_distribution is None else self.locus_distribution
        )
        dist._accumulator = _EmpiricalJointSFSAccumulation(self)

        return dist

    @cached_property
    def sfs2(self) -> 'EmpiricalTwoLocusSFSDistribution':
        """
        Two-locus SFS of the simulated replicates, from the locus-0 and locus-1 branch lengths of each frequency class
        that :meth:`MsprimeCoalescent.simulate() <phasegen.distributions.MsprimeCoalescent.simulate>` records, as an
        :class:`~phasegen.distributions.EmpiricalTwoLocusSFSDistribution`. It is paired with the other statistics of
        the same replicates.

        :raises NotImplementedError: If the scenario does not have exactly two loci.
        """
        if self.locus_config.n != 2:
            raise NotImplementedError("The two-locus SFS is only available for two-locus scenarios.")

        self.simulate()

        # the branch lengths of a locus summed over the demes they reside in
        lengths = self.sfs_lengths.sum(axis=1)

        # the mutation counts of a locus summed over the demes they occur in, of shape (N, 2, n + 1)
        counts = np.moveaxis(self.mutations.sum(axis=1), 1, 0) if self.simulate_mutations else None

        dist = EmpiricalTwoLocusSFSDistribution(
            lengths[0],
            lengths[1],
            mutation_counts=counts,
            lineage_config=self.lineage_config if self.lineage_distribution is None else self.lineage_distribution,
            locus_config=self.locus_config if self.locus_distribution is None else self.locus_distribution
        )
        dist._accumulator = _EmpiricalTwoLocusSFSAccumulation(self)

        return dist

    @cached_property
    def fst(self) -> float:
        r"""
        Hudson's :math:`F_{ST}` ground truth, simulated with msprime: ``1 - mean within-population branch diversity /
        mean between-population branch divergence``, averaged over replicate trees. The diversity averages over the
        populations with at least two sampled lineages and the divergence over the pairs of sampled populations, as
        :attr:`Coalescent.fst <phasegen.distributions.Coalescent.fst>` does.

        :raises ValueError: if fewer than two populations are sampled, none carries two sampled lineages, or the
            lineage configurations of an initial distribution differ.
        :raises NotImplementedError: if the demography has been dropped, as for serialization.
        """
        import msprime as ms

        pops = self._require_demography().pop_names
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
        not depend on the sample configuration of this coalescent, and uses ten times its replicates. Memoized per pair.

        :param pop_i: Name of the first population.
        :param pop_j: Name of the second population.
        :return: The mean coalescence time.
        :raises ValueError: If a population is unknown.
        :raises NotImplementedError: If the demography has been dropped, as for serialization.
        """
        import msprime as ms

        names = self._require_demography().pop_names
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
                num_replicates=self.num_replicates * _PAIRWISE_REPLICATE_FACTOR,
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

    def _trajectory_records(self) -> _Trajectories:
        """
        The lineage trajectories of the replicates, simulated on first use from the seeds of :meth:`simulate`.

        :return: The trajectories.
        """
        if self._trajectories is None:
            self._simulate(main=self.heights is None, trajectories=True)

        return self._trajectories

    def _accumulator(self) -> _EmpiricalAccumulation:
        """
        :return: The accumulation of rewards over time, with the tree-height reward by default.
        :raises NotImplementedError: If the simulation setup has been dropped.
        """
        self._require_demography()

        return _EmpiricalAccumulation(self, TreeHeightReward())

    def _require_demography(self) -> Demography:
        """
        :return: The demography, from which the replicates are simulated.
        :raises NotImplementedError: If the demography has been dropped, as for serialization.
        """
        if self.demography is None:
            raise NotImplementedError(_NO_GENEALOGIES.format(statistic="statistic", holder="coalescent"))

        return self.demography

    @_make_hashable
    @cache
    def joint(self, reward_a: Reward, reward_b: Reward) -> EmpiricalJointDistribution:
        """
        Joint distribution of two rewards accumulated by each replicate until the end time of the coalescent, the
        sampled counterpart of :meth:`Coalescent.joint() <phasegen.distributions.Coalescent.joint>`, for the rewards
        supported by :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param reward_a: The first reward.
        :param reward_b: The second reward.
        :return: The joint distribution.
        :raises TypeError: if ``reward_a`` or ``reward_b`` is not a single :class:`~phasegen.rewards.Reward`.
        :raises NotImplementedError: if a reward is not read from the simulated genealogies, or the demography has
            been dropped, as for serialization.
        """
        _validate_reward(reward_a, "reward_a")
        _validate_reward(reward_b, "reward_b")

        accumulator = self._accumulator()

        return EmpiricalJointDistribution(accumulator.samples(reward_a), accumulator.samples(reward_b))

    @_make_hashable
    @cache
    def moment(
            self,
            k: int = 1,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        The :math:`k`-th sample moment of the rewards accumulated from the start time to the end time, the sampled
        counterpart of :meth:`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`, defined in
        :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param k: The order :math:`k` of the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the tree-height reward for each factor.
        :param start_time: The start time :math:`s`. By default, 0.
        :param end_time: The end time :math:`t`. By default, the end time of the coalescent, or absorption.
        :param center: Whether to return the central moment.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :return: The :math:`k`-th moment.
        :raises ValueError: if ``k`` is not a non-negative integer, the number of rewards differs from it, the start
            time is negative, or the end time exceeds that of the coalescent.
        :raises TypeError: if an entry of ``rewards`` is not a :class:`~phasegen.rewards.Reward`.
        :raises NotImplementedError: if a reward is not read from the simulated genealogies, or the demography has
            been dropped, as for serialization.
        """
        return self._accumulator().moment(k, rewards, start_time, end_time, center, permute)

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        r"""
        The :math:`k`-th sample moment of the rewards accumulated from the start time :math:`s` to each end time
        :math:`t` in ``end_times``, the sampled counterpart of :meth:`Coalescent.accumulate()
        <phasegen.distributions.Coalescent.accumulate>`. Replicate :math:`m` of the :math:`N` replicates accumulates
        the reward

        .. math::

            R_{i,m}(s, t) = \int_s^t r_{i,m}(u)\, \mathrm{d}u,

        where :math:`r_{i,m}(u)` is the rate of reward :math:`i` in the genealogy of replicate :math:`m` at time
        :math:`u`. For the tree height it is one while more than one lineage remains, for the total branch length the
        number of lineages, and for SFS bin :math:`j` the number of lineages subtending :math:`j` samples. The
        moment is

        .. math::

            \frac{1}{N} \sum_{m=1}^{N} \prod_{i=1}^{k} \big(R_{i,m}(s, t) - \hat\mu_i\big),

        with :math:`\hat\mu_i` the sample mean of :math:`R_{i,m}(s, t)` for a central moment (``center`` and
        :math:`k \ge 2`) and zero otherwise.

        The supported rewards are :class:`~phasegen.rewards.TreeHeightReward`,
        :class:`~phasegen.rewards.TotalTreeHeightReward`, :class:`~phasegen.rewards.TotalBranchLengthReward`,
        :class:`~phasegen.rewards.UnfoldedSFSReward`, :class:`~phasegen.rewards.FoldedSFSReward`,
        :class:`~phasegen.rewards.TwoLocusSFSReward` and :class:`~phasegen.rewards.JointSFSReward`, their restrictions
        to a locus or a deme by :class:`~phasegen.rewards.RestrictedReward` or
        :class:`~phasegen.rewards.CombinedReward`, and sums of them by :class:`~phasegen.rewards.SumReward`. A
        restriction to a deme of several demes requires ``record_migration``.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t` at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the tree-height reward for each factor.
        :param center: Whether to return the central moment.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :param start_time: The start time :math:`s`. By default, 0.
        :return: The moment at each end time.
        :raises ValueError: if ``k`` is not a non-negative integer, the number of rewards differs from it, the start
            time is negative, an end time exceeds that of the coalescent, or a reward refers to a locus, deme or
            frequency class that does not exist.
        :raises NotImplementedError: if a reward is not read from the simulated genealogies, or the demography has
            been dropped, as for serialization.
        """
        return self._accumulator().accumulate(k, end_times, rewards, center, permute, start_time)

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot the accumulation of the sample moments of :meth:`MsprimeCoalescent.accumulate()
        <phasegen.distributions.MsprimeCoalescent.accumulate>`, as :meth:`Coalescent.plot_accumulation()
        <phasegen.distributions.Coalescent.plot_accumulation>` does.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to the
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile of the sampled tree height, or up to
            the end time of the coalescent if it is finite.
        :param rewards: Sequence of k rewards. By default, the tree-height reward for each factor.
        :param center: Whether to center the moment around the mean.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        :raises ValueError: if ``k`` is not a non-negative integer, or if ``rewards`` is a single
            :class:`~phasegen.rewards.Reward` and not a sequence.
        :raises NotImplementedError: if a reward is not read from the simulated genealogies, or the demography has
            been dropped, as for serialization.
        """
        return self._accumulator().plot_accumulation(
            k=k,
            end_times=end_times,
            rewards=rewards,
            center=center,
            permute=permute,
            ax=ax,
            show=show,
            file=file,
            clear=clear,
            label=label,
            title=title
        )

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
    _spawn_keys = dict(tree_height=0, total_branch_length=1, sfs=2, fsfs=3, jsfs=4, sfs2=5, joint=6, moment=7)

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
            construction. ``None`` draws one from fresh entropy.
        :raises ValueError: If ``seed`` is a negative integer.
        """
        if seed is None:
            seed = np.random.default_rng()

        if not isinstance(seed, np.random.Generator) and seed < 0:
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

        #: Random seed, drawn at construction if none is given.
        self.seed: int = int(seed.integers(2 ** 63)) if isinstance(seed, np.random.Generator) else seed

    def _seed(self, name: str) -> np.random.SeedSequence:
        """
        :param name: The key of the spawned seed in ``_spawn_keys``.
        :return: The seed sequence of the statistic.
        """
        return np.random.SeedSequence(self.seed, spawn_key=(self._spawn_keys[name],))

    def _to_empirical(self, name: str):
        """Sample the named analytic distribution into its empirical counterpart, seeded reproducibly, with the
        accumulation over time of its trajectories."""
        exact = getattr(self._coalescent, name)
        dist = exact.to_empirical(self.n_samples, seed=np.random.default_rng(self._seed(name)))

        trajectories = self._trajectories(name, exact)
        start = self._coalescent.start_time

        if name in ('sfs', 'fsfs'):
            dist._accumulator = _EmpiricalSFSAccumulation(trajectories, type(exact), start)
        elif name == 'jsfs':
            dist._accumulator = _EmpiricalJointSFSAccumulation(trajectories, start)
        elif name == 'sfs2':
            dist._accumulator = _EmpiricalTwoLocusSFSAccumulation(trajectories, start)
        else:
            dist._accumulator = _EmpiricalAccumulation(trajectories, exact.reward, start)

        return dist

    def _trajectories(self, name: str, dist: PhaseTypeDistribution) -> _SampledTrajectories:
        """
        The trajectories of a statistic on the state space of a distribution, sampled on first use and cached.

        :param name: The key of the spawned seed in ``_spawn_keys``.
        :param dist: The distribution whose state space the trajectories visit.
        :return: The trajectories.
        """
        cache = self.__dict__.setdefault('_sampled_trajectories', {})
        key = (name, id(dist.state_space))

        if key not in cache:
            cache[key] = _SampledTrajectories(dist, self.n_samples, self._seed(name), self.end_time,
                                              self.lineage_config)

        return cache[key]

    def _sample_rewards(self, rewards: Tuple[Reward, ...], name: str) -> np.ndarray:
        """
        Sample the rewards from shared trajectories of the wrapped coalescent, on the smallest state space
        supporting all of them, seeded reproducibly.

        :param rewards: The rewards.
        :param name: The key of the spawned seed in ``_spawn_keys``.
        :return: The samples, of shape ``(n_samples, len(rewards))``.
        :raises NotImplementedError: If the wrapped coalescent has been dropped.
        """
        dist = self._require_coalescent()._get_dist(len(rewards), list(rewards))

        return dist._sample(self.n_samples, rewards=list(rewards), rng=np.random.default_rng(self._seed(name)))

    def _accumulator(self, k: int, rewards: Optional[Sequence[Reward]]) -> _EmpiricalAccumulation:
        """
        :param k: The order of the moment.
        :param rewards: Sequence of ``k`` rewards, ``None`` for the tree-height reward for each factor.
        :return: The accumulation of the rewards over time, from the trajectories of :meth:`moment` on the smallest
            state space supporting all of them.
        :raises NotImplementedError: If the wrapped coalescent has been dropped.
        """
        coalescent = self._require_coalescent()
        dist = coalescent._get_dist(k, rewards)

        return _EmpiricalAccumulation(self._trajectories('moment', dist), TreeHeightReward(), coalescent.start_time)

    def _require_coalescent(self) -> Coalescent:
        """
        :return: The wrapped coalescent.
        :raises NotImplementedError: If the wrapped coalescent has been dropped, as for serialization.
        """
        if self._coalescent is None:
            raise NotImplementedError(_NO_GENEALOGIES.format(statistic="statistic", holder="coalescent"))

        return self._coalescent

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        r"""
        The :math:`k`-th sample moment of the rewards accumulated from the start time to each end time, the sampled
        counterpart of :meth:`Coalescent.accumulate() <phasegen.distributions.Coalescent.accumulate>`, with the
        estimator of :meth:`MsprimeCoalescent.accumulate() <phasegen.distributions.MsprimeCoalescent.accumulate>`.
        Trajectory :math:`m` accumulates :math:`R_{i,m}(s, t) = \sum_j r_i(x_j) |[a_j, b_j] \cap [s, t]|` over its
        sojourns :math:`[a_j, b_j]` in the states :math:`x_j`, with :math:`r_i(x)` the rate of reward :math:`i` in
        state :math:`x`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t` at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the tree-height reward for each factor.
        :param center: Whether to return the central moment.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :param start_time: The start time :math:`s`. By default, the start time of the wrapped coalescent.
        :return: The moment at each end time.
        :raises ValueError: if ``k`` is not a non-negative integer, the number of rewards differs from it, the start
            time is negative, or an end time exceeds that of the coalescent.
        :raises TypeError: if an entry of ``rewards`` is not a :class:`~phasegen.rewards.Reward`.
        :raises NotImplementedError: if the wrapped coalescent has been dropped, as for serialization.
        """
        k = _validate_order(k)

        return self._accumulator(k, rewards).accumulate(k, end_times, rewards, center, permute, start_time)

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot the accumulation of the sample moments of :meth:`SampledCoalescent.accumulate()
        <phasegen.distributions.SampledCoalescent.accumulate>`, as :meth:`Coalescent.plot_accumulation()
        <phasegen.distributions.Coalescent.plot_accumulation>` does.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to the
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile of the sampled tree height, or up to
            the end time of the coalescent if it is finite.
        :param rewards: Sequence of k rewards. By default, the tree-height reward for each factor.
        :param center: Whether to center the moment around the mean.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        :raises ValueError: if ``k`` is not a non-negative integer, or if ``rewards`` is a single
            :class:`~phasegen.rewards.Reward` and not a sequence.
        :raises NotImplementedError: if the wrapped coalescent has been dropped, as for serialization.
        """
        k = _validate_order(k)

        return self._accumulator(k, rewards).plot_accumulation(
            k=k,
            end_times=end_times,
            rewards=rewards,
            center=center,
            permute=permute,
            ax=ax,
            show=show,
            file=file,
            clear=clear,
            label=label,
            title=title
        )

    @_make_hashable
    @cache
    def joint(self, reward_a: Reward, reward_b: Reward) -> EmpiricalJointDistribution:
        """
        Joint distribution of two accumulated rewards sampled from shared trajectories, the sampled counterpart of
        :meth:`Coalescent.joint() <phasegen.distributions.Coalescent.joint>`, cached per pair of rewards.

        :param reward_a: The first reward.
        :param reward_b: The second reward.
        :return: The joint distribution.
        :raises TypeError: if ``reward_a`` or ``reward_b`` is not a single :class:`~phasegen.rewards.Reward`.
        :raises NotImplementedError: if the wrapped coalescent has been dropped, as for serialization.
        """
        _validate_reward(reward_a, "reward_a")
        _validate_reward(reward_b, "reward_b")

        samples = self._sample_rewards((reward_a, reward_b), 'joint')

        return EmpiricalJointDistribution(samples[:, 0], samples[:, 1])

    @_make_hashable
    @cache
    def moment(
            self,
            k: int = 1,
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        The :math:`k`-th sample moment of rewards accumulated by shared trajectories over the window of the wrapped
        coalescent, the sampled counterpart of :meth:`Coalescent.moment() <phasegen.distributions.Coalescent.moment>`,
        with the estimator of :meth:`MsprimeCoalescent.accumulate()
        <phasegen.distributions.MsprimeCoalescent.accumulate>`.

        :param k: The order :math:`k` of the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the tree-height reward for each factor.
        :param start_time: The start time, which must be that of the wrapped coalescent.
        :param end_time: The end time, which must be that of the wrapped coalescent.
        :param center: Whether to return the central moment.
        :param permute: Ignored, as the sample moment does not depend on the order of the rewards.
        :return: The :math:`k`-th moment.
        :raises ValueError: if ``k`` is not a non-negative integer or the number of rewards differs from it.
        :raises TypeError: if an entry of ``rewards`` is not a :class:`~phasegen.rewards.Reward`.
        :raises NotImplementedError: if the start or end time differs from that of the wrapped coalescent, or the
            wrapped coalescent has been dropped, as for serialization.
        """
        k = _validate_order(k)
        rewards = _EmpiricalAccumulation._rewards(k, rewards, TreeHeightReward())
        coalescent = self._require_coalescent()
        window = (coalescent.start_time, coalescent.end_time)

        if (start_time not in (None, window[0])) or (end_time not in (None, window[1])):
            raise NotImplementedError(
                f"SampledCoalescent samples the rewards accumulated over the window of the wrapped coalescent, from "
                f"start_time={window[0]} to end_time={window[1]}."
            )

        if k == 0:
            return 1.0

        samples = self._sample_rewards(rewards, 'moment')

        return float(_EmpiricalAccumulation._moment([s[None, :] for s in samples.T], center, 1)[0])

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
                self.__dict__[name]._accumulator = None

        self.__dict__.pop('_sampled_trajectories', None)
        self._coalescent = None
        self.demography = None

