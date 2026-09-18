"""Distribution base classes and marginal (per-deme / per-locus) views."""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from ..caching import cached_property
from typing import Any, Callable, Iterator, Sequence, TYPE_CHECKING
import numpy as np
from ..expm import Backend
from ..rewards import RestrictedReward
from ..settings import Settings
from ..spectrum import AbstractSpectrum

if TYPE_CHECKING:
    from .reward import JointRewardDistribution
    from matplotlib import pyplot as plt
    from .phase_type import PhaseTypeDistribution
    from ..visualization import _CurveData, _SurfaceData

expm = Backend.expm
logger = logging.getLogger('phasegen')


class DistributionFunction:
    """
    Distribution function returned by the ``pdf``, ``cdf`` and ``quantile`` properties of a distribution.

    Calling the object evaluates the function, so that ``coal.sfs.pdf(x)`` returns the density of every bin at ``x``,
    and :meth:`DistributionFunction.plot() <phasegen.distributions.DistributionFunction.plot>` draws it, so that
    ``coal.sfs.pdf.plot()`` draws the density curve of every bin. A joint function also draws its surface with
    ``plot_surface()``.

    The properties return the subclasses :class:`~phasegen.distributions.DensityFunction`,
    :class:`~phasegen.distributions.CumulativeDistributionFunction` and
    :class:`~phasegen.distributions.QuantileFunction` in plain, marginal (one function per spectrum bin), joint and
    conditional variants. The methods by which they are evaluated are listed at
    :class:`~phasegen.distributions.CumulativeDistributionFunction`.

    :param distribution: The distribution this function belongs to.
    """
    #: Kind of the function: ``'pdf'``, ``'cdf'`` or ``'quantile'``.
    kind: str = ''

    def __init__(self, distribution: 'CallableDistributionFunctions') -> None:
        self._distribution = distribution

    def __call__(self, *args, **kwargs) -> 'Any':
        """Evaluate the function at a point or an array of points, with the arguments of the concrete subclass."""
        return getattr(self._distribution, '_' + self.kind)(*args, **kwargs)

    def _plot_data(self, *args, **kwargs) -> '_CurveData':
        """
        The curves :meth:`plot` draws: this function evaluated over a grid, with labels and title, built by the
        distribution's ``_plot_data_<kind>``. Also called by the R package.

        :return: The curves, one per bin for a spectrum.
        """
        return getattr(self._distribution, '_plot_data_' + self.kind)(*args, **kwargs)

    def plot(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the function over a grid. The curve is ``self(t)``, the function the caller evaluates.

        :param ax: Axes to plot on.
        :param t: Points to evaluate at. By default, :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            the :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curve, ``None`` for none.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curve, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(t=t, n_points=n_points), file=file, show=show,
                                         clear=clear, label=label, title=title, **kwargs)

    def _curve(self, grid: np.ndarray | None, n_points: int | None, variable: str, title: str) -> '_CurveData':
        """
        One unlabelled curve of this function, over ``grid`` or by default over :attr:`Settings.plot_n_grid` points
        up to the :attr:`Settings.plot_endpoint_quantile` quantile (the probability axis for a quantile function).

        :param grid: Points to evaluate at, ``None`` for the default grid.
        :param n_points: Number of points of the default grid, ``None`` for :attr:`Settings.plot_n_grid`.
        :param variable: Name of the variable on the x-axis.
        :param title: Plot title.
        :return: The curve.
        """
        from ..visualization import _CurveData

        grid = self._default_grid(self.kind, grid, n_points,
                                  lambda: self._distribution.quantile(Settings.plot_endpoint_quantile))

        return _CurveData(
            x=grid,
            y=np.atleast_2d(self(grid)),
            labels=[''],
            xlabel='q' if self.kind == 'quantile' else variable,
            ylabel=dict(pdf=f'f({variable})', cdf=f'F({variable})', quantile='quantile')[self.kind],
            title=title
        )

    @staticmethod
    def _default_grid(kind: str, grid: np.ndarray | None, n_points: int | None, end: Callable[[], float]) -> np.ndarray:
        """
        The plotting grid: ``grid`` itself if given, otherwise :attr:`Settings.plot_n_grid` points from 0 to ``end()``,
        or across the central :attr:`Settings.plot_endpoint_quantile` probability range for a quantile function.

        :param kind: The function kind, ``'pdf'``, ``'cdf'`` or ``'quantile'``.
        :param grid: Points to evaluate at, ``None`` for the default grid.
        :param n_points: Number of points of the default grid, ``None`` for :attr:`Settings.plot_n_grid`.
        :param end: Right end of the default grid of a density or CDF, called only when needed.
        :return: The grid.
        """
        if grid is not None:
            return np.asarray(grid, dtype=float)

        n_points = n_points or Settings.plot_n_grid
        q_end = Settings.plot_endpoint_quantile

        if kind == 'quantile':
            return np.linspace(1.0 - q_end, q_end, n_points)

        return np.linspace(0, float(end()), n_points)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: call to evaluate, .plot() to draw>"


class _SurfacePlottable:
    """Mixin adding :meth:`plot_surface` for bivariate (joint) distribution functions (a 3D surface in addition to the
    2D heatmap drawn by :meth:`plot`). Univariate function classes deliberately lack it."""

    def plot_surface(self, *args, **kwargs) -> 'plt.Axes':
        """Plot the joint distribution function as a 3D surface, with the arguments of the concrete subclass."""
        return getattr(self._distribution, '_plot_' + self.kind + '_surface')(*args, **kwargs)


# --- function kinds -------------------------------------------------------------------------------------------------

class DensityFunction(DistributionFunction):
    r"""Probability density function :math:`f(x) = F'(x)` of a distribution with CDF :math:`F`.

    Calling ``pdf(x)`` returns the density at ``x``, for a scalar or an array, evaluated as listed at
    :class:`~phasegen.distributions.CumulativeDistributionFunction`.
    """
    kind = 'pdf'


class CumulativeDistributionFunction(DistributionFunction):
    r"""Cumulative distribution function :math:`F(x) = \mathbb{P}(Y \le x)` of the random variable :math:`Y` of a
    distribution.

    Calling ``cdf(x)`` returns the probability at ``x``, for a scalar or an array. The evaluation is described at
    :class:`~phasegen.distributions.TreeHeightDistribution` for the tree height, at
    :class:`~phasegen.distributions.RewardDistribution` for any other accumulated reward and at
    :class:`~phasegen.distributions.EmpiricalDistribution` for a sample.
    """
    kind = 'cdf'


class QuantileFunction(DistributionFunction):
    r"""Quantile function :math:`F^{-1}(q) = \inf\{x : F(x) \ge q\}` of a distribution with CDF :math:`F`, for a
    probability level :math:`q \in [0, 1]`.

    Calling ``quantile(q)`` returns the quantile at ``q``, for a scalar or an array. An
    :class:`~phasegen.distributions.EmpiricalDistribution` uses the sample quantile.

    .. rubric:: Cumulative-hazard grid

    The analytic distributions carry the cumulative hazard :math:`H(x) = -\log(1 - F(x))` on a grid of nodes and
    interpolate it linearly, which is exact for an exponential tail. With :math:`\hat H` the interpolant,

    .. math::

        F^{-1}(q) = \hat H^{-1}\big(-\log(1 - q)\big).

    The tree height evaluates its CDF and density pointwise (see
    :class:`~phasegen.distributions.TreeHeightDistribution`). Any other accumulated reward reads them from the same
    grid (see :class:`~phasegen.distributions.RewardDistribution`), as :math:`F = 1 - e^{-\hat H}` and
    :math:`f = e^{-\hat H} \hat H'`, so that its quantile inverts its CDF exactly and its density is non-negative.

    .. rubric:: Implementation

    - The slope :math:`\hat H'` is interpolated from finite differences at the nodes.
    - Levels beyond the last node return the last node.
    """
    kind = 'quantile'

    def plot(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the quantile function (value versus probability). The curve is ``self(q)``, the function the caller
        evaluates.

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate at. By default, :attr:`~phasegen.settings.Settings.plot_n_grid` points
            from ``1 - Settings.plot_endpoint_quantile`` to :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curve, ``None`` for none.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curve, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, n_points=n_points), file=file, show=show,
                                         clear=clear, label=label, title=title, **kwargs)


# --- the shared CDF representation ----------------------------------------------------------------------------------

class _HazardGrid:
    """
    The cumulative-hazard grid described at ``QuantileFunction``. ``_interp_cdf``, ``_interp_quantile`` and
    ``_interp_pdf`` read it, and the subclasses ``_LSTFunction`` and ``_ExpmFunction`` supply the nodes through
    ``_cdf_grid``.
    """

    def _shared(self, key: str, build) -> 'Any':
        """Return a shared entry of the CDF representation, built once via ``build`` and cached on the distribution
        (so the cdf / pdf / quantile of one distribution reuse it). Honors :attr:`~phasegen.settings.Settings.cache`."""
        cache = self._distribution.__dict__.setdefault('_lst_curve_cache', {})
        # the grid is built for one de Hoog tail cut; if that setting was changed on this live distribution the cached
        # nodes no longer join the fit at the same place, so discard them and rebuild for the new cut.
        tail = Settings.dehoog_tail_quantile
        if cache.get('_tail_quantile', tail) != tail:
            cache.clear()
        cache['_tail_quantile'] = tail
        if key in cache:
            return cache[key]
        val = build()
        if Settings.cache:
            cache[key] = val
        return val

    def _cdf_grid(self, x_max: float = 0.0, q_max: float = 0.0) -> tuple:
        """
        The grid: its nodes and the cumulative hazard on them, both ascending.

        :param x_max: Largest point the caller will evaluate.
        :param q_max: Largest probability level the caller will invert.
        :return: The nodes and the cumulative hazard on them.
        """
        raise NotImplementedError

    @staticmethod
    def _hazard(cdf: 'np.ndarray | float') -> np.ndarray:
        r"""The cumulative hazard :math:`H = -\log(1 - F)`, the coordinate the grid is interpolated in. Capped, so a
        CDF that has saturated at 1 (as the cosine fit does at the end of its window) does not take it to infinity."""
        return -np.log1p(-np.minimum(np.asarray(cdf, dtype=float), 1.0 - 1e-16))

    def _interp_cdf(self, t: np.ndarray, nodes: np.ndarray, hazard: np.ndarray) -> np.ndarray:
        r"""
        The CDF between the grid's nodes: :math:`F(x) = 1 - e^{-H(x)}`, with the cumulative hazard ``H`` interpolated
        linearly in ``x``. A chord in ``F`` out in the tail would instead join the nodes underneath a concave curve,
        biasing the far-tail quantile 1e-3 long.

        :param t: Points to evaluate at.
        :param nodes: The grid's nodes.
        :param hazard: The cumulative hazard on them.
        :return: The CDF at ``t``.
        """
        # below the support (t < nodes[0] = 0) the CDF is 0, not the clamped first-node value np.interp would return
        return -np.expm1(-np.interp(t, nodes, hazard, left=0.0))

    def _interp_quantile(self, q: np.ndarray, nodes: np.ndarray, hazard: np.ndarray) -> np.ndarray:
        r"""The closed-form inverse of :meth:`_interp_cdf`'s map: the same relation between ``x`` and ``H``, read the
        other way. Levels at or below the atom :math:`\mathbb{P}(R = 0)` land on the first node, which is 0.

        :param q: Probability levels.
        :param nodes: The grid's nodes.
        :param hazard: The cumulative hazard on them.
        :return: The quantiles at ``q``.
        """
        return np.interp(self._hazard(q), hazard, nodes)

    def _interp_pdf(self, t: np.ndarray, nodes: np.ndarray, hazard: np.ndarray) -> np.ndarray:
        r"""The derivative of :meth:`_interp_cdf`'s map: on the segment :math:`[x_i, x_{i+1})` holding ``t``,
        :math:`f = e^{-H(x)}\,(H_{i+1} - H_i)/(x_{i+1} - x_i)`, so integrating the density over any segment gives
        exactly the CDF increment there. Non-negative since the hazard is non-decreasing, and zero outside the nodes.

        :param t: Points to evaluate at.
        :param nodes: The grid's nodes.
        :param hazard: The cumulative hazard on them.
        :return: The density at ``t``.
        """
        t = np.asarray(t, dtype=float)

        if len(nodes) < 2:
            # a grid of a single node (a near-total atom at 0) has no segment, so the continuous density is zero
            return np.zeros_like(t)

        widths = np.diff(nodes)
        slopes = np.divide(np.diff(hazard), widths, out=np.zeros_like(widths), where=widths > 0)

        # segment holding t, with the last node closing the last segment
        i = np.clip(np.searchsorted(nodes, t, side='right') - 1, 0, len(widths) - 1)
        inside = (t >= nodes[0]) & (t <= nodes[-1])

        return np.where(inside, np.exp(-np.interp(t, nodes, hazard, left=0.0)) * slopes[i], 0.0)


# --- the accumulated-reward (LST / de Hoog) inversion machinery, owned by the function objects -----------------------

class _LSTFunction(_HazardGrid):
    """
    The inversion of an accumulated-reward transform, described at ``RewardDistribution``, for the function objects
    of a ``RewardDistribution`` and its conditional subclasses. The transform and its scales come from
    ``self._distribution`` (``lst``, ``_invert``, ``_range``, ``_s_inf``). ``_cdf_point`` is the per-point de Hoog CDF
    behind the tail nodes and the conditional support bracket.
    """
    #: Equispaced nodes :math:`N` on which the expansion is evaluated.
    _cos_n_grid: int = 8192

    #: CDF level :math:`1 - \delta` at which the second-pass window ends. The expansion discards the mass above it.
    _cos_tail_target: float = 1.0 - 1e-5

    #: Scale factor :math:`\kappa` of the first-pass window :math:`[0, \hat\mu + \kappa \hat\sigma]`.
    _cos_rough_scale: float = 20.0

    #: Largest share of the CDF range the last half of the cosine terms may still move before the expansion is
    #: reported unresolved.
    _cos_truncation_tol: float = 1e-3

    @property
    def _cos_terms(self) -> int:
        """The number of cosine terms, :attr:`~phasegen.settings.Settings.cos_terms`."""
        return Settings.cos_terms

    @property
    def _cos_terms_rough(self) -> int:
        """The number of cosine terms of the locating pass, a third of the second pass."""
        return max(self._cos_terms // 3, 2)

    #: Step :math:`\eta_H` of the exact nodes in cumulative hazard.
    _hazard_step: float = 0.25

    #: Step :math:`\eta_F` of the exact nodes in probability, which resolves the body when the tail level is low.
    _cdf_step: float = 0.01

    #: Largest factor by which a step between exact nodes exceeds the previous one.
    _step_growth: float = 2.0

    #: CDF level :math:`1 - \epsilon` the exact nodes reach unless a query reaches further, and their maximum number.
    _tail_target: float = 1.0 - 1e-6
    _max_exact_nodes: int = 512

    # ---- distribution primitives (thin accessors) --------------------------------------------------------------
    def _range(self, scale: float = 12.0) -> float:
        return self._distribution._range(scale)

    def _cdf_point(self, t: float) -> float:
        """Per-point de Hoog CDF at ``t``, the atom at ``t = 0`` and 0 below it. Memoised per distribution. It supplies
        the tail nodes of the grid, the conditional support bracket and the exact reference of the tests."""
        if t < 0:
            return 0.0
        d = self._distribution
        if t == 0:
            # F(0) = P(R <= 0) = P(R = 0), the atom phi(inf) -- right-continuous at the point mass, matching the
            # de Hoog / cosine curves (which split the atom off and add it back). The inversion below is skipped
            # both to avoid the phi(s)/s singularity and because at t > 0 it already carries the atom.
            return max(d.lst(d._s_inf).real, 0.0)

        cache = self._shared('cdf_points', dict)
        if t not in cache:
            cache[t] = d._invert(lambda s: d.lst(s) / s, float(t))
        return cache[t]

    def _pdf_point(self, t: float) -> float:
        r"""Per-point de Hoog density (transform :math:`\mathcal{L}[f] = \varphi(s)`)."""
        d = self._distribution
        return d._invert(d.lst, float(t))

    # ---- shared CDF representation (cached on the distribution) -------------------------------------------------
    @property
    def _cos_coeffs(self) -> dict:
        return self._shared('cos_coeffs', self._build_cos_coeffs)

    @property
    def _cos_cdf_grid(self) -> tuple:
        return self._shared('cos_cdf_grid', self._build_cos_cdf_grid)

    def _build_cos_coeffs(self) -> dict:
        """The two-pass cosine expansion described at ``RewardDistribution``, with the window scale
        ``_cos_rough_scale`` and the tail level ``_cos_tail_target``."""
        rough = self._fit_cos(self._range(self._cos_rough_scale), self._cos_terms_rough)
        xs = np.linspace(0.0, rough['b'], 1024)
        cdf = np.maximum.accumulate(self._eval_cos_cdf(rough, xs))
        b = float(np.interp(self._cos_tail_target, cdf, xs))

        return self._fit_cos(max(b, rough['b'] * 1e-3), self._cos_terms)

    def _fit_cos(self, b: float, n_terms: int) -> dict:
        """
        One cosine expansion on ``[0, b]`` with ``n_terms`` terms, described at ``RewardDistribution``. The atom is
        split off above ``1e-9``, ``_warn_if_nonmonotone`` checks the raw continuous CDF for ringing and
        ``_warn_if_unresolved`` checks that the terms have converged.

        :param b: The window end.
        :param n_terms: The number of cosine terms.
        :return: The window end, frequencies, coefficients and atom.
        """
        d = self._distribution
        p0 = d.lst(d._s_inf).real
        w = np.arange(n_terms) * np.pi / b
        chi = np.array([d.lst(-1j * wk) for wk in w])
        if p0 > 1e-9:
            if 1.0 - p0 <= 1e-12:  # full atom at 0 (R = 0 almost surely): degenerate point mass, no continuous part
                return dict(b=b, w=w, fk=np.zeros(n_terms), p0=p0)
            chi = (chi - p0) / (1 - p0)  # continuous part only
        fk = (2.0 / b) * np.real(chi)  # a = 0, so exp(-i w a) = 1
        fk[0] *= 0.5

        # the largest backward step of the (continuous) CDF is the sensitive ringing detector (a visibly rippling CDF
        # can come from sub-percent density wiggles); the shared non-monotonicity guard surfaces a substantial one
        # (rtol 1e-2 of the [0, 1] CDF range -- a loose bar, the cosine series being coarse near a sharp feature)
        xd = np.linspace(0.0, b, max(512, 2 * n_terms))
        Fd = fk[0] * xd + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xd))
        d._warn_if_nonmonotone(Fd, d._titled('COS CDF (residual ripple)'), rtol=1e-2)

        half = max(n_terms // 2, 1)
        Fh = fk[0] * xd + (fk[1:half] / w[1:half]) @ np.sin(np.outer(w[1:half], xd))
        self._warn_if_unresolved(float(np.abs(Fd - Fh).max()) * (1 - p0 if p0 > 1e-9 else 1.0), n_terms)

        return dict(b=b, w=w, fk=fk, p0=p0)

    def _warn_if_unresolved(self, truncation: float, n_terms: int) -> None:
        """
        Warn when the second half of the terms still moves the CDF by more than ``_cos_truncation_tol``. The
        coefficients do not depend on how many of them are summed, so the difference between the expansion truncated
        at half the terms and at all of them estimates what the discarded terms would still contribute.

        :param truncation: The largest absolute difference between the two truncations, in probability.
        :param n_terms: The number of cosine terms summed.
        """
        if Settings.check_inversions and truncation > self._cos_truncation_tol:
            self._distribution._logger.warning(
                "%s: the cosine expansion is unresolved, the last %d of %d terms still move the CDF by %.2e (bar "
                "%.0e). The distribution spans scales the window cannot resolve at this many terms. Raise "
                "Settings.cos_terms, whose cost is linear in it.",
                self._distribution._titled('COS CDF (truncation)'), n_terms - n_terms // 2, n_terms,
                truncation, self._cos_truncation_tol
            )

    @staticmethod
    def _eval_cos_cdf(fit: dict, xs: np.ndarray) -> np.ndarray:
        """Evaluate the continuous COS CDF of ``fit`` (atom ``p0`` added back) at ``xs``, clipped to ``[0, 1]``."""
        w, fk, p0 = fit['w'], fit['fk'], fit['p0']
        cdf_c = fk[0] * xs + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xs))
        return np.clip(p0 + (1 - p0) * cdf_c if p0 > 1e-9 else cdf_c, 0.0, 1.0)

    def _build_cos_cdf_grid(self) -> tuple:
        """A fine, monotone CDF on the fit's window ``[0, b]``: the body of the shared grid of ``_cdf_grid``,
        computed once per distribution."""
        fit = self._cos_coeffs
        xs = np.linspace(0.0, fit['b'], self._cos_n_grid)
        return xs, np.maximum.accumulate(self._eval_cos_cdf(fit, xs))

    def _cos(self, x: np.ndarray, kind: str, n_terms: int = None, scale: float = 12.0) -> np.ndarray:
        """
        Evaluate the raw COS fit as a whole CDF/PDF curve over the grid ``x``. No caller reads its density: the
        published pdf differentiates the CDF grid instead, precisely because the raw cosine sum rings (and goes
        negative) at an atom. This is the handle the tests judging the fit itself need. The default window uses the
        cached two-pass fit; an explicit ``scale`` refits over ``[0, mean + scale*std]``. The CDF is clipped to
        ``[0, 1]`` and made monotone.
        """
        fit = self._cos_coeffs if scale == 12.0 else self._fit_cos(self._range(scale), n_terms or self._cos_terms)
        b, w, fk, p0 = fit['b'], fit['w'], fit['fk'], fit['p0']

        xa = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), 0.0, b)
        if kind == 'pdf':
            curve = fk @ np.cos(np.outer(w, xa))
            return (1 - p0) * curve if p0 > 1e-9 else curve

        cdf = self._eval_cos_cdf(fit, xa)
        order = np.argsort(xa)
        cdf[order] = np.maximum.accumulate(cdf[order])
        return cdf

    def _exact_step(self, nodes: list) -> float:
        """
        The step to the next exact node: the increment in probability, the finer of ``_hazard_step`` in cumulative
        hazard and ``_cdf_step``, divided by the local density, and at most a local limit. For the first step the
        density is the slope of the cosine grid at the anchor and the limit is the distance along that grid to the
        incremented level. Afterwards the density is the secant of the last two nodes and the limit is
        ``_step_growth`` times the last step. A density that is not positive takes the limit. The nodes do not depend on
        the queries.

        :param nodes: The ``(x, F)`` nodes so far, ascending, with ``F < 1`` at the last.
        :return: The step to the next node.
        """
        x, cdf = nodes[-1]
        increment = min(self._hazard_step * (1.0 - cdf), self._cdf_step)

        if len(nodes) == 1:
            xs, cs = self._cos_cdf_grid
            density = float(np.interp(x, xs, np.gradient(cs, xs)))
            limit = float(np.interp(cdf + increment, cs, xs)) - x
        else:
            x_prev, cdf_prev = nodes[-2]
            density = (cdf - cdf_prev) / (x - x_prev)
            limit = self._step_growth * (x - x_prev)

        return float(min(increment / density, limit)) if density > 0 else limit

    def _exact_nodes(self, x_cut: float, cut: float, x_max: float, q_max: float) -> list:
        """
        The ``(x, F)`` de Hoog nodes described at ``RewardDistribution``, marching outward from the anchor at the cut.
        They are cached on the distribution and extended when a query reaches past their end, never trimmed, so an
        answer does not depend on later queries. The march advances on exact values, because the cosine quantile
        saturates at the end of its window.

        :param x_cut: Where the CDF reaches the cut.
        :param cut: CDF value at or above which the exact inversion supplies the grid.
        :param x_max: Largest point the caller will evaluate.
        :param q_max: Largest probability level the caller will invert.
        :return: The nodes, ascending.
        """
        nodes = self._shared('cdf_exact', list)

        if not nodes:
            # the anchor carries the *fit's* value at the cut, so it sits exactly on the fit's own curve and joins the
            # two halves without a step. That value is usually the cut itself, but not always: an atom at 0 carries
            # the CDF straight past the cut in one jump, so ``x_cut`` is 0 and the value there is the atom, well above
            # the cut. Stamping the cut on it instead shifted the whole grid by the difference (2e-2 on a Dirac bin
            # whose atom is 0.99). Where the grid is exact throughout there is no fit to anchor to, so the value is
            # the exact one.
            xs, cdf = self._cos_cdf_grid
            nodes.append((x_cut, self._cdf_point(x_cut) if cut <= 0.0 else float(np.interp(x_cut, xs, cdf))))

        if x_max <= x_cut and q_max <= cut:
            return nodes  # the query stays in the fit's half, so the expensive nodes are left unbuilt

        target = min(max(q_max, self._tail_target), 1.0 - 1e-12)
        while len(nodes) < self._max_exact_nodes:
            x, cdf = nodes[-1]
            # stop once the ladder covers the query and holds the target mass; a CDF that has saturated at 1 says
            # nothing more about points beyond it either, so it ends the march regardless
            if cdf >= target and (x >= x_max or cdf >= 1.0 - 1e-12):
                break
            x = x + self._exact_step(nodes)
            nodes.append((x, self._cdf_point(x)))

        return nodes

    def _cdf_grid(self, x_max: float = 0.0, q_max: float = 0.0) -> tuple:
        """
        The grid described at ``RewardDistribution``: the cosine nodes below ``Settings.dehoog_tail_quantile``, whose
        saturated nodes are dropped, joined to the de Hoog nodes of ``_exact_nodes`` above it. A cut of 1 or ``None``
        uses the cosine nodes only, and a cut of 0 the de Hoog nodes only.

        :param x_max: Largest point the caller will evaluate.
        :param q_max: Largest probability level the caller will invert.
        :return: The nodes and the cumulative hazard on them, both ascending.
        """
        xs, cdf = self._cos_cdf_grid
        cut = Settings.dehoog_tail_quantile
        cut = 1.0 if cut is None else float(np.clip(cut, 0.0, 1.0))

        # the fit's own nodes, up to the cut. The saturated ones carry no information -- the fit force-normalises to 1
        # at the end of its window -- and would pin the hazard at its cap, so they go whatever the cut is.
        keep = (cdf < cut) & (cdf < 1.0 - 1e-12)
        nodes, values = xs[keep], cdf[keep]

        x_cut = float(np.interp(cut, cdf, xs)) if cut > 0.0 else 0.0

        if cut < 1.0:
            exact = self._exact_nodes(x_cut, cut, x_max, q_max)
            nodes = np.concatenate([nodes, [x for x, _ in exact]])
            values = np.concatenate([values, [c for _, c in exact]])

        order = np.argsort(nodes, kind='stable')

        return nodes[order], np.maximum.accumulate(self._hazard(values[order]))


class _LSTCumulativeDistributionFunction(_LSTFunction, CumulativeDistributionFunction):
    """The CDF of an accumulated reward, read from the grid of ``_LSTFunction._cdf_grid``."""

    def __call__(self, t) -> 'np.ndarray | float':
        r"""
        The CDF :math:`F(x) = \mathbb{P}(R \le x)`, see :class:`~phasegen.distributions.RewardDistribution`.

        :param t: Point or array of points :math:`x` at which to evaluate the CDF.
        :return: The CDF at ``t``, of the same shape.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        out = self._interp_cdf(ta, *self._cdf_grid(x_max=float(ta.max(initial=0.0))))

        return out if np.ndim(t) > 0 else float(out[0])

    def _plot_data(self, x: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The CDF curve :meth:`plot` draws.

        :param x: Points to evaluate at. By default, :attr:`Settings.plot_n_grid` points up to the
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._curve(x, n_points, 'x', self._distribution._titled('CDF'))

    def plot(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the function up to the configured plot-endpoint quantile. The curve is ``self(x)``, the function the
        caller evaluates.

        :param ax: Axes to plot on.
        :param x: Points to evaluate at. By default, :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            the :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curve, ``None`` for none.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curve, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(x=x, n_points=n_points), file=file, show=show,
                                         clear=clear, label=label, title=title, **kwargs)


class _LSTDensityFunction(_LSTFunction, DensityFunction):
    """The density of an accumulated reward, read from the grid of ``_LSTFunction._cdf_grid``."""

    def __call__(self, t, **kwargs) -> 'np.ndarray | float':
        r"""
        The density :math:`f(x)` of the continuous part of :math:`R`, see
        :class:`~phasegen.distributions.RewardDistribution`.

        :param t: Point or array of points :math:`x` at which to evaluate the density.
        :return: The density at ``t``, of the same shape.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        d = self._distribution
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        out = self._interp_pdf(ta, *self._cdf_grid(x_max=float(ta.max(initial=0.0))))
        out = d._warn_if_negative(out, d._titled('density'))
        return out if np.ndim(t) > 0 else float(out[0])

    def _plot_data(self, x: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The density curve :meth:`plot` draws.

        :param x: Points to evaluate at. By default, :attr:`Settings.plot_n_grid` points up to the
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._curve(x, n_points, 'x', self._distribution._titled('PDF'))

    plot = _LSTCumulativeDistributionFunction.plot


class _LSTQuantileFunction(_LSTFunction, QuantileFunction):
    """The quantile function of an accumulated reward, read from the grid of ``_LSTFunction._cdf_grid``."""

    def __call__(self, q) -> 'np.ndarray | float':
        r"""
        The quantile :math:`F^{-1}(q) = \inf\{x : F(x) \ge q\}` of the accumulated reward :math:`R` with CDF
        :math:`F`, evaluated as described at :class:`~phasegen.distributions.RewardDistribution`. Levels at or below
        the atom :math:`p_0` return 0.

        :param q: Probability level or array of levels :math:`q \in [0, 1]`.
        :return: The quantiles, of the same shape as ``q``.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        :raises ValueError: If any ``q`` lies outside :math:`[0, 1]`.
        """
        qa = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((qa < 0) | (qa > 1)):
            raise ValueError("Quantile must be between 0 and 1.")

        out = self._interp_quantile(qa, *self._cdf_grid(q_max=float(qa.max(initial=0.0))))

        return out if np.ndim(q) > 0 else float(out[0])

    def _plot_data(self, q: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The quantile curve :meth:`plot` draws (value versus probability).

        :param q: Probabilities to evaluate at. By default, :attr:`Settings.plot_n_grid` points from
            ``1 - Settings.plot_endpoint_quantile`` to :attr:`Settings.plot_endpoint_quantile`.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._curve(q, n_points, 'x', self._distribution._titled('quantile function'))


# --- direct grid evaluation (matrix-exponential tree height) ---------------------------------------------------------

class _GridCumulativeDistributionFunction(CumulativeDistributionFunction):
    """CDF whose distribution computes ``P(R <= t)`` *directly* (the exact matrix-exponential tree height) rather than
    by Laplace inversion. The evaluation lives in the subclass ``__call__``."""

    def _plot_data(self, t: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The CDF curve :meth:`plot` draws.

        :param t: Points to evaluate at. By default, :attr:`Settings.plot_n_grid` points up to the
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._curve(t, n_points, 't', 'CDF')


class _GridDensityFunction(DensityFunction):
    """Density whose distribution computes it directly (see :class:`_GridCumulativeDistributionFunction`)."""

    def _plot_data(self, t: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The density curve :meth:`plot` draws.

        :param t: Points to evaluate at. By default, :attr:`Settings.plot_n_grid` points up to the
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._curve(t, n_points, 't', 'PDF')


class _GridQuantileFunction(QuantileFunction):
    """Quantile function whose distribution computes it directly (see :class:`_GridCumulativeDistributionFunction`)."""

    def _plot_data(self, q: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The quantile curve :meth:`plot` draws (value versus probability).

        :param q: Probabilities to evaluate at. By default, :attr:`Settings.plot_n_grid` points from
            ``1 - Settings.plot_endpoint_quantile`` to :attr:`Settings.plot_endpoint_quantile`.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._curve(q, n_points, 't', 'Quantile function')


# --- marginal (per-bin spectrum) flavours ---------------------------------------------------------------------------

class MarginalDensity(DensityFunction):
    """Per-bin densities of a spectrum, each that of the bin's :class:`~phasegen.distributions.RewardDistribution`."""


class MarginalCDF(CumulativeDistributionFunction):
    """Per-bin CDFs of a spectrum, each that of the bin's :class:`~phasegen.distributions.RewardDistribution`."""


class MarginalQuantileFunction(QuantileFunction):
    """Per-bin quantile functions of a spectrum, each that of the bin's
    :class:`~phasegen.distributions.RewardDistribution`."""


# --- joint (bivariate) flavours -------------------------------------------------------------------------------------

class _JointFunction(_SurfacePlottable):
    """Plotting grid, heatmap and surface shared by ``JointCDF`` and ``JointDensity``. The 2D representation lives on
    the ``JointRewardDistribution`` the function belongs to."""

    def _grid_values(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The joint kind evaluated on the grid ``xs x ys`` (implemented per kind)."""
        raise NotImplementedError

    def _plot_data(self, n_points: int = None, surface: bool = False) -> '_SurfaceData':
        """
        The grid and values :meth:`plot` and :meth:`plot_surface` draw. Each axis runs from 0 to the marginal
        :attr:`Settings.plot_endpoint_quantile` quantile. A CDF has its value scale fixed to ``[0, 1]``.

        :param n_points: Number of grid points per axis. By default, :attr:`Settings.plot_joint_cdf_n_grid` for a
            CDF, and :attr:`Settings.plot_joint_pdf_surface_n_grid` or :attr:`Settings.plot_joint_pdf_n_grid` for a
            density drawn as a surface or a heatmap.
        :param surface: Whether the default resolution is that of a surface rather than a heatmap.
        :return: The grid and the values on it.
        """
        from ..visualization import _SurfaceData

        d = self._distribution
        is_cdf = self.kind == 'cdf'

        if n_points is None:
            if is_cdf:
                n_points = Settings.plot_joint_cdf_n_grid
            else:
                n_points = Settings.plot_joint_pdf_surface_n_grid if surface else Settings.plot_joint_pdf_n_grid

        # the axes are clipped to the cosine window the representation was built on
        q = Settings.plot_endpoint_quantile
        xs = np.linspace(0, min(d.marginal('a').quantile(q), d._cos2d['ba']), n_points)
        ys = np.linspace(0, min(d.marginal('b').quantile(q), d._cos2d['bb']), n_points)
        name = self.kind.upper()

        return _SurfaceData(
            x=xs,
            y=ys,
            z=self._grid_values(xs, ys),
            xlabel='$R_a$',
            ylabel='$R_b$',
            zlabel='F(R_a, R_b)' if is_cdf else 'f(R_a, R_b)',
            title=f"Joint {name} {d.label}" if d.label else f"Joint reward {name}",
            vmin=0.0 if is_cdf else None,
            vmax=1.0 if is_cdf else None
        )

    def plot(self, ax: 'plt.Axes' = None, n_points: int = None, show: bool = True, file: str = None,
             title: str = None) -> 'plt.Axes':
        """
        Heatmap of the joint function. Each axis runs from 0 to the marginal
        :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.

        :param ax: Axes to plot on.
        :param n_points: Number of grid points per axis. By default,
            :attr:`~phasegen.settings.Settings.plot_joint_cdf_n_grid` for a CDF and
            :attr:`~phasegen.settings.Settings.plot_joint_pdf_n_grid` for a density.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Plot title, ``None`` for the default title.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_surface(self._plot_data(n_points), surface=False, ax=ax, title=title, file=file,
                                          show=show)

    def plot_surface(self, ax: 'plt.Axes' = None, n_points: int = None, show: bool = True, file: str = None,
                     title: str = None) -> 'plt.Axes':
        """
        3D surface of the joint function. Each axis runs from 0 to the marginal
        :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.

        :param ax: Axes to plot on.
        :param n_points: Number of grid points per axis. By default,
            :attr:`~phasegen.settings.Settings.plot_joint_cdf_n_grid` for a CDF and
            :attr:`~phasegen.settings.Settings.plot_joint_pdf_surface_n_grid` for a density.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param title: Plot title, ``None`` for the default title.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_surface(self._plot_data(n_points, surface=True), surface=True, ax=ax, title=title,
                                          file=file, show=show)


class JointDensity(_JointFunction, DensityFunction):
    r"""
    Joint density :math:`f(x, y)` of the continuous part of a :class:`~phasegen.distributions.JointRewardDistribution`,
    for :math:`x, y > 0`. The atom and the mass on the axes described at :class:`~phasegen.distributions.JointCDF` have
    no density.

    The density is the mixed central difference of the continuous part :math:`C` of
    :class:`~phasegen.distributions.JointCDF` on a uniform grid with steps :math:`h_x` and :math:`h_y`,

    .. math::

        f(x, y) \approx \frac{1}{4 h_x h_y} \big[\, & C(x + h_x, y + h_y) - C(x + h_x, y - h_y) \\
        & - C(x - h_x, y + h_y) + C(x - h_x, y - h_y) \,\big],

    the density averaged over one cell, which smooths the residual oscillation of the cosine expansion near the origin.

    .. rubric:: Implementation

    - The grid spans the cosine window :math:`[0, b_a] \times [0, b_b]` of
      :class:`~phasegen.distributions.JointCDF`, so the steps are properties of the distribution and a value depends
      only on its own point. A bicubic spline interpolates between the nodes. The expansion holds no mass beyond its
      window, where the density is therefore zero.
    - Negative values are set to zero, with a warning under :attr:`Settings.check_inversions
      <phasegen.settings.Settings.check_inversions>`.
    """

    def __call__(self, x, y) -> 'np.ndarray | float':
        r"""
        Evaluate the joint density :math:`f(x_j, y_l)` on the outer grid of the arguments.

        :param x: Value(s) :math:`x_j` of :math:`R_a`, a scalar or a 1D array.
        :param y: Value(s) :math:`y_l` of :math:`R_b`, a scalar or a 1D array.
        :return: An array of shape ``(len(x), len(y))``, or a float for a single pair.
        :raises NotImplementedError: If one reward is a constant multiple of the other on every transient state, so
            that the law has no density on the plane, or if the coalescent has a bounded accumulation window.
        """
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        f = self._grid_values(xs, ys)
        return float(f.ravel()[0]) if f.size == 1 else f

    def _grid_values(self, xs, ys) -> 'np.ndarray':
        d = self._distribution
        # guarded here so that plot() and plot_surface(), which call _grid_values directly, refuse as well
        if d._ratio is not None:
            raise NotImplementedError("The joint density is singular when one reward is a constant multiple c of the "
                                      "other (R_a = c R_b almost surely): the law lives on a line and has no 2D "
                                      "density. Use cdf(x, y) = marginal CDF of R_a at min(x, c y), or the 1D marginal "
                                      "density.")
        raw = d._density(xs, ys)  # the cosine 2D density can dip negative near the origin edge (Gibbs)
        d._warn_if_negative(raw, 'joint density (cosine)')
        return np.clip(raw, 0.0, None)


class JointCDF(_JointFunction, CumulativeDistributionFunction):
    r"""
    Joint CDF :math:`F(x, y) = \mathbb{P}(R_a \le x,\ R_b \le y)` of a
    :class:`~phasegen.distributions.JointRewardDistribution`, zero when either threshold is negative. With
    :math:`p_{00} = \mathbb{P}(R_a = R_b = 0)`,

    .. math::

        F(x, y) = g_b(x) + g_a(y) - p_{00} + C(x, y),

    where :math:`g_b(x) = \mathbb{P}(R_a \le x,\ R_b = 0)` and :math:`g_a(y) = \mathbb{P}(R_a = 0,\ R_b \le y)` hold the
    mass on the axes, and :math:`C(x, y) = \mathbb{P}(0 < R_a \le x,\ 0 < R_b \le y)` is the continuous part.

    .. rubric:: Continuous part

    :math:`C` is a two-dimensional Fourier-cosine expansion (Ruijter and Oosterlee, 2012) on a window
    :math:`[0, L_a] \times [0, L_b]` that holds nearly all the mass. Removing the axes from the joint transform
    :math:`\Phi` gives the characteristic function of the continuous part,

    .. math::

        \chi(\omega_a, \omega_b) = \Phi(-\mathrm{i}\omega_a, -\mathrm{i}\omega_b)
        - \Phi(-\mathrm{i}\omega_a, \infty) - \Phi(\infty, -\mathrm{i}\omega_b) + p_{00}.

    With :math:`N` frequencies :math:`u_j = j\pi / L_a` and :math:`v_l = l\pi / L_b` per axis,

    .. math::

        C(x, y) = \sum_{j, l = 0}^{N - 1} A_{jl}\, \frac{\sin(u_j x)}{u_j}\, \frac{\sin(v_l y)}{v_l},

    .. math::

        A_{jl} = \frac{2}{L_a L_b} \operatorname{Re}\big[\chi(u_j, v_l) + \chi(u_j, -v_l)\big],

    for :math:`x \le L_a` and :math:`y \le L_b`. A zero frequency contributes :math:`x` or :math:`y` in place of the
    fraction and halves the coefficient.

    .. rubric:: Implementation

    - The window ends are the marginal means plus a fixed multiple of the standard deviations. Lanczos factors damp the
      ringing of the series without changing the total mass. :math:`N` is given by :attr:`Settings.cos_terms_2d
      <phasegen.settings.Settings.cos_terms_2d>`, at a cost quadratic in it.
    - For a single epoch on a dense state space, the transform values of one frequency :math:`u_j` form a shifted
      linear system in :math:`s_b`, so one generalized Schur (QZ) decomposition serves the whole row.
    - The axis terms are one-dimensional cosine series of :math:`\Phi(\cdot, \infty)` and :math:`\Phi(\infty, \cdot)`
      on wider windows, as for the marginal CDF of a :class:`~phasegen.distributions.RewardDistribution`.
    - When :math:`\mathbf{r}_a = c\,\mathbf{r}_b` on every transient state for a constant :math:`c > 0`,
      :math:`R_a = c R_b` almost surely and :math:`F(x, y) = \mathbb{P}(R_a \le \min(x, c y))`.
    - Under :attr:`Settings.check_inversions <phasegen.settings.Settings.check_inversions>`, a warning is logged when
      :math:`F(x, \infty)` departs from the marginal CDF near the origin.

    .. rubric:: References

    Ruijter, M. J. and Oosterlee, C. W. (2012). Two-dimensional Fourier cosine series expansion method for pricing
    financial options. SIAM Journal on Scientific Computing 34(5), B642-B671.
    """

    def __call__(self, x, y) -> 'np.ndarray | float':
        r"""
        Evaluate the joint CDF :math:`F(x_j, y_l)` on the outer grid of the thresholds.

        :param x: Threshold(s) :math:`x_j` for :math:`R_a`, a scalar or a 1D array.
        :param y: Threshold(s) :math:`y_l` for :math:`R_b`, a scalar or a 1D array.
        :return: An array of shape ``(len(x), len(y))``, or a float for a single pair.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        G = self._grid_values(xs, ys)
        return float(G.ravel()[0]) if G.size == 1 else G

    def _grid_values(self, xs, ys) -> 'np.ndarray':
        # reduced here so that plot() and plot_surface(), which call _grid_values directly, draw the same CDF
        d = self._distribution
        if d._ratio is not None:
            m = d.marginal('a')
            t = np.minimum.outer(xs, d._ratio * ys)
            # the CDF vanishes below 0 and equals the atom P(R = 0) at 0
            p0 = float(d._atoms['both0'])
            return np.array([[0.0 if tt < 0.0 else p0 if tt == 0.0 else float(m.cdf(tt)) for tt in row] for row in t])
        return d._cdf_grid(xs, ys)


# (a bivariate joint has no quantile flavour: a 2D quantile is not well-defined -- use a marginal or conditional)


# --- conditional flavours -------------------------------------------------------------------------------------------

class _ConditionalCosTerms:
    """Halves the cosine terms of a conditional expansion, every coefficient of which costs an inner inversion."""

    @property
    def _cos_terms(self) -> int:
        """The number of cosine terms, half of :attr:`~phasegen.settings.Settings.cos_terms`."""
        return max(Settings.cos_terms // 2, 2)

    @property
    def _cos_terms_rough(self) -> int:
        """The number of cosine terms of the locating pass, half of the second pass."""
        return max(self._cos_terms // 2, 2)


class ConditionalDensity(_ConditionalCosTerms, _LSTDensityFunction):
    """Density of a :class:`~phasegen.distributions.ConditionalRewardDistribution`, computed as described there."""


class ConditionalCDF(_ConditionalCosTerms, _LSTCumulativeDistributionFunction):
    """CDF of a :class:`~phasegen.distributions.ConditionalRewardDistribution`, computed as described there."""


class ConditionalQuantileFunction(_ConditionalCosTerms, _LSTQuantileFunction):
    """Quantile function of a :class:`~phasegen.distributions.ConditionalRewardDistribution`, computed as described
    there."""




class CallableDistributionFunctions:
    """
    Mixin exposing ``pdf`` / ``cdf`` / ``quantile`` as callable-and-plottable distribution-function properties. Each
    concrete distribution supplies the evaluation and plot data of these functions, or its function objects provide
    them. The class of the returned function objects depends on the distribution, for example the ``Marginal...``
    classes for a spectrum and the ``Conditional...`` classes for a conditional distribution.
    """
    #: The distribution-function classes returned by the properties; overridden by subclasses to select the flavour.
    #: ``_quantile_function = None`` marks a distribution without a quantile (e.g. a bivariate joint).
    _pdf_function = DensityFunction
    _cdf_function = CumulativeDistributionFunction
    _quantile_function = QuantileFunction

    def _function(self, kind: str, factory) -> 'Any':
        """Return the (cached) distribution-function object for ``kind``, built once via ``factory`` and stored on
        this distribution. Caching the object -- not just rebuilding a thin wrapper -- is what lets the function
        object's own cached cosine coefficients / CDF grid persist across ``.cdf`` / ``.pdf`` /
        ``.quantile`` accesses, since the three share the one distribution they hang off. Honors the global cache
        switch (:attr:`~phasegen.settings.Settings.cache`)."""
        cache = self.__dict__.setdefault('_function_cache', {})
        if kind in cache:
            return cache[kind]
        obj = factory(self)
        if Settings.cache:
            cache[kind] = obj
        return obj

    @property
    def cdf(self) -> CumulativeDistributionFunction:
        """Cumulative distribution function: callable (``cdf(t)``) and plottable (``cdf.plot()`` / -- joint --
        ``cdf.plot_surface()``)."""
        return self._function('cdf', self._cdf_function)

    @property
    def pdf(self) -> DensityFunction:
        """Probability density function: callable (``pdf(t)``) and plottable (``pdf.plot()`` / -- joint --
        ``pdf.plot_surface()``)."""
        return self._function('pdf', self._pdf_function)

    @property
    def quantile(self) -> QuantileFunction:
        """Quantile function: callable (``quantile(q)``) and plottable (``quantile.plot()``)."""
        if self._quantile_function is None:
            raise NotImplementedError(f"{type(self).__name__} has no quantile function "
                                      "(a bivariate joint quantile is not well-defined; use a marginal/conditional).")
        return self._function('quantile', self._quantile_function)

    def plot_cdf(self, *args, **kwargs) -> 'plt.Axes':
        """
        Plot the CDF curve.

        .. deprecated:: 2.0.0
            Use ``cdf.plot()``. ``plot_cdf`` will be removed in a future release.
        """
        warnings.warn("plot_cdf() is deprecated since 2.0.0 and will be removed in a future release; "
                      "use .cdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self.cdf.plot(*args, **kwargs)

    def plot_pdf(self, *args, **kwargs) -> 'plt.Axes':
        """
        Plot the density curve.

        .. deprecated:: 2.0.0
            Use ``pdf.plot()``. ``plot_pdf`` will be removed in a future release.
        """
        warnings.warn("plot_pdf() is deprecated since 2.0.0 and will be removed in a future release; "
                      "use .pdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self.pdf.plot(*args, **kwargs)

    def _warn_if_negative(self, values: np.ndarray, label: str, rtol: float = 1e-3) -> np.ndarray:
        """Warn (via this distribution's logger) if ``values`` has a substantial negative entry relative to its scale,
        then return it unchanged (the caller clips). A density / probability must be non-negative, so a real negative
        -- beyond the ``rtol`` numerical-noise band -- signals inversion ringing (Gibbs) worth surfacing rather than
        silently clipping. Gated by :attr:`~phasegen.settings.Settings.check_inversions`."""
        arr = np.asarray(values, dtype=float)
        if Settings.check_inversions and arr.size:
            scale = max(float(np.abs(arr).max()), 1e-300)
            mn = float(np.nanmin(arr))
            if mn < -rtol * scale:
                self._logger.warning(f"{label}: substantial negative value ({mn:.2e} vs scale {scale:.2e}); clipping "
                                     f"to 0 -- the numerical inversion may be imprecise here")
        return values

    def _warn_if_nonmonotone(self, cdf: np.ndarray, label: str, rtol: float = 1e-3) -> np.ndarray:
        """Warn if ``cdf`` has a downward step beyond ``rtol`` of its range, then return it unchanged (the caller
        enforces monotonicity). Logging, noise band and gating are those of ``_warn_if_negative``."""
        arr = np.asarray(cdf, dtype=float)
        if Settings.check_inversions and arr.size > 1:
            rng = max(float(np.nanmax(arr) - np.nanmin(arr)), 1e-300)
            drop = -float(np.nanmin(np.diff(arr)))
            if drop > rtol * rng:
                self._logger.warning(f"{label}: non-monotone CDF (downward step {drop:.2e} vs range {rng:.2e}); "
                                     f"enforcing monotonicity -- the numerical inversion may be imprecise here")
        return cdf


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability distributions.
    """

    def __init__(self) -> None:
        """
        Create object.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

    def _touch(self, **kwargs: dict) -> None:
        """
        Touch all cached properties.

        :param kwargs: Additional keyword arguments.
        """
        for cls in self.__class__.__mro__:
            for attr, value in cls.__dict__.items():
                if isinstance(value, cached_property):
                    # force-persist the value: _touch/_drop is the serialization contract and must hold even under
                    # Settings.cache = False, where the getter would otherwise recompute without storing
                    self.__dict__[attr] = getattr(self, attr)


class MomentAwareDistribution(ProbabilityDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    @abstractmethod
    @cached_property
    def mean(self) -> float:
        """
        First moment / mean.
        """
        pass

    @abstractmethod
    @cached_property
    def var(self) -> float:
        """
        Second central moment / variance.
        """
        pass

    @abstractmethod
    @cached_property
    def m2(self) -> float:
        """
        Second (non-central) moment.
        """
        pass


class MarginalDistributions(Mapping, ABC):
    """
    Base class for marginal distributions.
    """

    @staticmethod
    def _correlation(cov, scale):
        """
        The correlation coefficient ``cov / scale``, for a scalar statistic and for a spectrum alike. Where the scale
        vanishes the coefficient is undefined and ``nan`` is returned, matching the empirical estimator.

        :param cov: The covariance of the two marginals.
        :param scale: The product of their standard deviations.
        :return: The correlation coefficient, of the shape of ``cov``.
        """
        if isinstance(scale, AbstractSpectrum):
            s = np.asarray(scale.data, dtype=float)
            c = np.asarray(cov.data if isinstance(cov, AbstractSpectrum) else cov, dtype=float)

            return type(scale)(np.divide(c, s, out=np.full(s.shape, np.nan), where=s > 0))

        return cov / scale if scale > 0 else float('nan')

    @abstractmethod
    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix.
        """
        pass

    @abstractmethod
    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix.
        """
        pass

    @abstractmethod
    def get_cov(self, d1, d2) -> float:
        """
        Get the covariance between two marginal distributions.

        :param d1: The index of the first marginal distribution.
        :param d2: The index of the second marginal distribution.
        :return: covariance
        """
        pass

    @abstractmethod
    def get_corr(self, d1, d2) -> float:
        """
        Get the correlation coefficient between two marginal distributions.

        :param d1: The index of the first marginal distribution.
        :param d2: The index of the second marginal distribution.
        :return: correlation coefficient
        """
        pass


class MarginalLocusDistributions(MarginalDistributions):
    """
    Marginal locus distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution') -> None:
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item) -> 'Any':
        """
        Get the distribution for the given locus.

        :param item: Deme name.
        :return: Distribution.
        """
        return self.loci[item]

    def __iter__(self) -> Iterator:
        """
        Iterate over distributions.

        :return: Iterator.
        """
        return iter(self.loci)

    def __len__(self) -> int:
        """
        Get the number of distributions.

        :return: Number of distributions.
        """
        return len(self.loci)

    @cached_property
    def loci(self) -> dict:
        """
        Distributions marginalized over loci, keyed by locus index.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        from .phase_type import PhaseTypeDistribution, TreeHeightDistribution
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        loci = {}
        for locus in range(self.dist.locus_config.n):
            loci[locus] = cls(
                state_space=self.dist.state_space,
                tree_height=self.dist.tree_height,
                demography=self.dist.demography,
                reward=RestrictedReward(self.dist.reward, locus=locus)
            )

        return loci

    def get_cov(self, locus1: int, locus2: int) -> float:
        """
        Get the covariance between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The covariance.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        return self.dist.moment(
            k=2,
            rewards=(
                RestrictedReward(self.dist.reward, locus=locus1),
                RestrictedReward(self.dist.reward, locus=locus2)
            ),
            center=True
        )

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_cov(i, j) for i in range(n_loci)] for j in range(n_loci)])

    def get_corr(self, locus1: int, locus2: int) -> float:
        """
        Get the correlation coefficient between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The correlation coefficient, ``nan`` where either locus has zero variance and the coefficient is
            undefined, matching the empirical estimator.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        return self._correlation(self.get_cov(locus1, locus2), self.loci[locus1].std * self.loci[locus2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_corr(i, j) for i in range(n_loci)] for j in range(n_loci)])

    def joint_distribution(self, locus1: int, locus2: int) -> 'JointRewardDistribution':
        """
        Joint distribution of the distribution's reward accumulated at ``locus1`` and at ``locus2``, the pair behind
        :meth:`MarginalLocusDistributions.get_cov() <phasegen.distributions.MarginalLocusDistributions.get_cov>`, as a
        :class:`~phasegen.distributions.JointRewardDistribution`.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The joint distribution across the two loci.
        :raises ValueError: If either locus does not exist.
        """
        locus1, locus2 = int(locus1), int(locus2)

        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        return self.dist.joint_distribution(
            RestrictedReward(self.dist.reward, locus=locus1),
            RestrictedReward(self.dist.reward, locus=locus2)
        )


class MarginalDemeDistributions(MarginalDistributions):
    """
    Marginal deme distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution') -> None:
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item) -> 'Any':
        """
        Get the distribution for the given deme.

        :param item: Deme name.
        :return: Distribution.
        """
        return self.demes[item]

    def __iter__(self) -> Iterator:
        """
        Iterate over distributions.

        :return: Iterator.
        """
        return iter(self.demes)

    def __len__(self) -> int:
        """
        Get the number of distributions.

        :return: Number of distributions.
        """
        return len(self.demes)

    @cached_property
    def demes(self) -> dict:
        """
        Distributions marginalized over demes, keyed by population name.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        from .phase_type import PhaseTypeDistribution, TreeHeightDistribution
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        demes = {}
        for pop in self.dist.lineage_config.pop_names:
            demes[pop] = cls(
                state_space=self.dist.state_space,
                tree_height=self.dist.tree_height,
                demography=self.dist.demography,
                reward=RestrictedReward(self.dist.reward, pop=pop)
            )

        return demes

    def get_cov(self, pop1: str, pop2: str) -> float:
        """
        Get the covariance between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The covariance.
        """
        if pop1 not in self.dist.lineage_config.pop_names or pop2 not in self.dist.lineage_config.pop_names:
            raise ValueError(f"Population {pop1} or {pop2} does not exist.")

        return self.dist.moment(
            k=2,
            rewards=(
                RestrictedReward(self.dist.reward, pop=pop1),
                RestrictedReward(self.dist.reward, pop=pop2)
            ),
            center=True
        )

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix across demes.
        """
        pops = self.dist.lineage_config.pop_names

        return np.array([[self.get_cov(p1, p2) for p1 in pops] for p2 in pops])

    def get_corr(self, pop1: str, pop2: str) -> float:
        """
        Get the correlation coefficient between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The correlation coefficient, ``nan`` where either deme has zero variance and the coefficient is
            undefined, matching the empirical estimator.
        """
        return self._correlation(self.get_cov(pop1, pop2), self.demes[pop1].std * self.demes[pop2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across demes.
        """
        pops = self.dist.lineage_config.pop_names

        return np.array([[self.get_corr(p1, p2) for p1 in pops] for p2 in pops])


class DensityAwareDistribution(CallableDistributionFunctions, MomentAwareDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments and densities can be calculated. The
    ``cdf`` / ``pdf`` / ``quantile`` are exposed as callable-and-plottable
    :class:`~phasegen.distributions.DistributionFunction` objects, whose class depends on the distribution and which
    carry the evaluation and the plot data.
    """
