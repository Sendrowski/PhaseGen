"""Distribution base classes and marginal (per-deme / per-locus) views."""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from ..caching import cached_property
from typing import Callable, Iterator, Sequence, TYPE_CHECKING
import numpy as np
from ..expm import Backend
from ..rewards import DemeReward, LocusReward, CombinedReward
from ..settings import Settings

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .phase_type import PhaseTypeDistribution

expm = Backend.expm
logger = logging.getLogger('phasegen')


def adaptive_grid(f, a: float, b: float, n_init: int = 9, tol: float = None, max_points: int = None):
    """
    Adaptively sample a scalar function ``f`` on ``[a, b]``, concentrating evaluations where the curve bends.

    Starts from a coarse uniform grid and repeatedly bisects any interval whose midpoint value deviates from the
    straight chord between its endpoints by more than ``tol`` times the function's range. For an expensive ``f`` (the
    per-point de Hoog inversion) this hits a given visual accuracy with far fewer evaluations than a uniform grid --
    e.g. it resolves the near-zero atom spike of an SFS bin density that a uniform grid would miss.

    :param f: Scalar function to sample (called as ``f(x)`` for a float ``x``).
    :param a: Left endpoint.
    :param b: Right endpoint.
    :param n_init: Number of initial uniform points (>= 2).
    :param tol: Relative deviation tolerance; defaults to :attr:`Settings.plot_adaptive_tol`.
    :param max_points: Maximum number of evaluations; defaults to :attr:`Settings.plot_n_grid`.
    :return: Sorted ``(x, y)`` arrays.
    """
    from collections import deque

    tol = Settings.plot_adaptive_tol if tol is None else tol
    max_points = Settings.plot_n_grid if max_points is None else max_points

    xs = list(np.linspace(a, b, n_init))
    ys = [float(f(x)) for x in xs]
    thr = tol * max(max(ys) - min(ys), 1e-300)

    # work queue of intervals; bisect those whose midpoint departs from the chord by more than the threshold
    stack = deque((xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(xs) - 1))
    ex, ey = [], []
    while stack and len(xs) + len(ex) < max_points:
        xl, yl, xr, yr = stack.popleft()
        xm = 0.5 * (xl + xr)
        ym = float(f(xm))
        ex.append(xm)
        ey.append(ym)
        if abs(ym - 0.5 * (yl + yr)) > thr:
            stack.append((xl, yl, xm, ym))
            stack.append((xm, ym, xr, yr))

    x = np.array(xs + ex)
    y = np.array(ys + ey)
    order = np.argsort(x)
    return x[order], y[order]


class DistributionFunction:
    """
    A distribution function -- callable (evaluate) and plottable -- returned by a distribution's ``pdf`` / ``cdf`` /
    ``quantile`` property.

    Calling it evaluates the function (e.g. ``coal.sfs.pdf(t)`` returns the per-bin densities at ``t``), while
    :meth:`plot` draws it (e.g. ``coal.sfs.pdf.plot()`` overlays every bin's density curve).

    This base class is rarely used directly: each property returns one of the thin typed subclasses
    (:class:`DensityFunction` / :class:`CumulativeDistributionFunction` / :class:`QuantileFunction`, in plain,
    ``Marginal...``, ``Joint...`` and ``Conditional...`` flavours). They behave identically but carry distinct
    docstrings describing *what* the function is and *how* it is computed, and -- being real classes with real
    :meth:`plot` / :meth:`__call__` methods -- they let IDEs resolve ``.plot`` to a definition and surface those
    docstrings (unlike a dynamically bound attribute). Supersedes the former ``plot_pdf`` / ``plot_cdf`` methods (now
    deprecated aliases).

    A function object holds its owning ``distribution`` and dispatches by :attr:`kind` to the distribution's
    ``_<kind>`` (evaluate) and ``_plot_<kind>`` (plot) -- so it is fully defined by the distribution, not patched
    together from loose callbacks.

    :param distribution: The distribution this function belongs to.
    """
    #: Short kind label (``'pdf'`` / ``'cdf'`` / ``'quantile'``), set by the kind subclasses; selects the
    #: distribution's ``_<kind>`` / ``_plot_<kind>`` methods and is used in ``repr``.
    kind: str = ''

    def __init__(self, distribution: 'CallableDistributionFunctions'):
        self._distribution = distribution

    def __call__(self, *args, **kwargs):
        """Evaluate the distribution function at the given point(s) (the distribution's ``_<kind>``)."""
        return getattr(self._distribution, '_' + self.kind)(*args, **kwargs)

    def plot(self, *args, **kwargs) -> 'plt.Axes':
        """
        Plot the distribution function (the distribution's ``_plot_<kind>``). Accepted arguments depend on the
        distribution; common ones are ``exact`` (use the slower per-point de Hoog inversion instead of the fast COS
        curve), ``bins`` / ``configs`` (select which spectrum bins to draw), ``n_points`` (grid resolution),
        ``ax`` / ``show`` / ``file`` / ``title``.
        """
        return getattr(self._distribution, '_plot_' + self.kind)(*args, **kwargs)

    def __repr__(self):
        return f"<{type(self).__name__}: call to evaluate, .plot() to draw>"


class _SurfacePlottable:
    """Mixin adding :meth:`plot_surface` for bivariate (joint) distribution functions (a 3D surface in addition to the
    2D heatmap drawn by :meth:`plot`). Univariate function classes deliberately lack it."""

    def plot_surface(self, *args, **kwargs) -> 'plt.Axes':
        """Plot the joint distribution function as a 3D surface (the distribution's ``_plot_<kind>_surface``)."""
        return getattr(self._distribution, '_plot_' + self.kind + '_surface')(*args, **kwargs)


# --- function kinds -------------------------------------------------------------------------------------------------

class DensityFunction(DistributionFunction):
    """Probability density function.

    - **Callable** ``pdf(x)``: the exact pointwise density by per-point de Hoog Laplace inversion (the exact
      matrix-exponential density for the tree height; a histogram for empirical samples).
    - **Plot** ``pdf.plot()``: the fast cosine curve (the derivative of the cosine CDF), or the exact per-point
      density with ``exact=True``.
    """
    kind = 'pdf'


class CumulativeDistributionFunction(DistributionFunction):
    """Cumulative distribution function -- the probability of being at most ``x``.

    - **Callable** ``cdf(x)``: the exact value by per-point de Hoog inversion (the exact matrix exponential for the
      tree height; the empirical CDF for samples).
    - **Plot** ``cdf.plot()``: the fast cosine curve, or the exact per-point CDF with ``exact=True``.
    """
    kind = 'cdf'


class QuantileFunction(DistributionFunction):
    """Quantile function -- the inverse CDF.

    - **Callable** ``quantile(q)``: the value at which the CDF reaches ``q``, by bisection on the de Hoog CDF (or the
      sample quantile for empirical data).
    - **Plot** ``quantile.plot()``: inverts the fast cosine CDF curve (or the per-point bisection with ``exact=True``).
    """
    kind = 'quantile'


# --- the accumulated-reward (LST / de Hoog) inversion machinery, owned by the function objects -----------------------

class _LSTFunction:
    """
    Mixin owning the 1D accumulated-reward inversion machinery for the function objects of an LST distribution
    (:class:`~phasegen.distributions.reward.RewardDistribution` and its conditional flavours; a bare
    :class:`~phasegen.distributions.PhaseTypeDistribution` such as ``total_branch_length``). It pulls the transform
    and scale *primitives* (``lst`` / ``_invert`` / ``_cumulants`` / ``_range`` / ``_time_scale`` / ``_titled`` /
    the inversion guards) from ``self._distribution`` and turns them into the cdf / pdf / quantile.

    The **shared CDF representation** -- the de Hoog monotone spline and the two-pass Fourier-cosine coefficients --
    is cached on the *distribution* (the single object the cdf / pdf / quantile of one distribution hang off, see
    :meth:`CallableDistributionFunctions._function`), so all three reuse one fit without reaching into one another.
    """
    #: Cosine terms for the coarse support-locating pass and the fine accuracy pass of the two-pass COS fit.
    _cos_terms_rough: int = 128
    _cos_terms: int = 384

    # ---- distribution primitives (thin accessors) --------------------------------------------------------------
    def _range(self, scale: float = 12.0) -> float:
        return self._distribution._range(scale)

    def _cdf_point(self, t: float) -> float:
        """Per-point de Hoog CDF ``P(R <= t)`` (``L[CDF] = phi(s) / s``) -- the building block of both the exact
        :meth:`_LSTCumulativeDistributionFunction.__call__` and the de Hoog spline."""
        if t <= 0:
            return 0.0
        d = self._distribution
        return d._invert(lambda s: d.lst(s) / s, float(t))

    def _pdf_point(self, t: float) -> float:
        """Per-point de Hoog density (``L[pdf] = phi(s)``)."""
        d = self._distribution
        return d._invert(d.lst, float(t))

    # ---- shared CDF representation (cached on the distribution) -------------------------------------------------
    def _shared(self, key: str, build):
        """Return a shared CDF-representation entry, built once via ``build`` and cached on the distribution (so the
        cdf / pdf / quantile of one distribution reuse it). Honors :attr:`Settings.cache`."""
        cache = self._distribution.__dict__.setdefault('_lst_curve_cache', {})
        if key in cache:
            return cache[key]
        val = build()
        if Settings.cache:
            cache[key] = val
        return val

    @property
    def _cos_coeffs(self) -> dict:
        return self._shared('cos_coeffs', self._build_cos_coeffs)

    @property
    def _cos_cdf_grid(self) -> tuple:
        return self._shared('cos_cdf_grid', self._build_cos_cdf_grid)

    @property
    def _dehoog_spline(self) -> dict:
        return self._shared('dehoog_spline', self._build_dehoog_spline)

    def _build_cos_coeffs(self) -> dict:
        """
        COS coefficients, fit in **two passes**: a coarse pass over a generous window (``mean + 12*std``) locates the
        effective support, then the fit is redone over a window tightened to that support. Matching the window to
        where the mass actually is -- rather than ``mean + 12*std``, which a heavy tail blows far past the bulk --
        lets a few hundred cosine terms resolve the curve accurately, removing the ringing at the source.
        """
        rough = self._fit_cos(self._range(12.0), self._cos_terms_rough)
        xs = np.linspace(0.0, rough['b'], 1024)
        cdf = np.maximum.accumulate(self._eval_cos_cdf(rough, xs))
        b = float(np.interp(0.9995, cdf, xs))
        return self._fit_cos(max(b, rough['b'] * 1e-3), self._cos_terms)

    def _fit_cos(self, b: float, n_terms: int) -> dict:
        """
        Fit the COS (Fourier-cosine) inversion over ``[0, b]``: evaluate the characteristic function
        ``chi(w) = phi(-i w)`` on a fixed frequency grid and return the cosine coefficients. An atom at ``R = 0``
        (``p0 = phi(inf)``) is split off so the series sees only the smooth continuous part. Warns if a substantial
        CDF ripple remains (a sharp feature/atom the cosine series cannot resolve at this window/resolution).
        """
        d = self._distribution
        p0 = d.lst(1e8).real
        w = np.arange(n_terms) * np.pi / b
        chi = np.array([d.lst(-1j * wk) for wk in w])
        if p0 > 1e-9:
            chi = (chi - p0) / (1 - p0)  # continuous part only
        fk = (2.0 / b) * np.real(chi)  # a = 0, so exp(-i w a) = 1
        fk[0] *= 0.5

        # the largest backward step of the (continuous) CDF is the sensitive ringing detector (a visibly rippling CDF
        # can come from sub-percent density wiggles); the shared non-monotonicity guard surfaces a substantial one
        # (rtol 1e-2 of the [0,1] CDF range -- a looser bar than the de Hoog spline's, the cosine series being coarser)
        xd = np.linspace(0.0, b, max(512, 2 * n_terms))
        Fd = fk[0] * xd + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xd))
        d._warn_if_nonmonotone(Fd, d._titled('COS CDF (residual ripple)'), rtol=1e-2)

        return dict(b=b, w=w, fk=fk, p0=p0)

    @staticmethod
    def _eval_cos_cdf(fit: dict, xs: np.ndarray) -> np.ndarray:
        """Evaluate the continuous COS CDF of ``fit`` (atom ``p0`` added back) at ``xs``, clipped to ``[0, 1]``."""
        w, fk, p0 = fit['w'], fit['fk'], fit['p0']
        cdf_c = fk[0] * xs + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xs))
        return np.clip(p0 + (1 - p0) * cdf_c if p0 > 1e-9 else cdf_c, 0.0, 1.0)

    def _build_cos_cdf_grid(self) -> tuple:
        """A fine, monotone CDF on ``[0, b]`` underlying the COS plotting curves (the curve / quantile inversion
        interpolate it, so they are mutually consistent and computed once)."""
        fit = self._cos_coeffs
        xs = np.linspace(0.0, fit['b'], 2048)
        return xs, np.maximum.accumulate(self._eval_cos_cdf(fit, xs))

    def _cos(self, x: np.ndarray, kind: str, n_terms: int = None, scale: float = 12.0) -> np.ndarray:
        """
        Evaluate the COS fit as a whole CDF/PDF curve over the grid ``x`` (for plotting; the exact per-point
        cdf / pdf use de Hoog). The default window uses the cached two-pass fit; an explicit ``scale`` refits over
        ``[0, mean + scale*std]`` (used in tests). The CDF is clipped to ``[0, 1]`` and made monotone.
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

    def _build_dehoog_spline(self) -> dict:
        """
        Accurate, cheap-to-query cached curve: a monotone PCHIP spline of the *continuous* CDF through de Hoog
        Laplace-inversion values at adaptively placed points (the atom ``P(R = 0)`` split off first, as in the cosine
        path). Being de-Hoog-anchored it is accurate everywhere -- no Gibbs ringing on sharp / heavy-tailed features
        -- and the PDF is its analytic derivative. The default (``method='dehoog'``) backing of the curves.
        """
        from scipy.interpolate import PchipInterpolator

        d = self._distribution
        p0 = d.lst(1e8).real  # atom at R = 0
        b = self._range(scale=12.0)  # cheap cumulant-based support end (avoids depending on the quantile)

        def g(x):  # continuous CDF: (F(x) - p0) / (1 - p0); the de Hoog F includes the atom
            f = self._cdf_point(x)
            return (f - p0) / (1 - p0) if p0 > 1e-9 else f

        x, y = adaptive_grid(g, 0.0, b, tol=Settings.inversion_tol)
        # de Hoog CDF is monotone up to tiny inversion noise; warn if a real wiggle survives, then enforce monotonicity
        d._warn_if_nonmonotone(y, d._titled('CDF (de Hoog)'))
        y = np.clip(np.maximum.accumulate(y), 0.0, 1.0)
        return dict(spline=PchipInterpolator(x, y, extrapolate=True), p0=p0, b=b)

    def _cdf_curve(self, x, method: str = 'dehoog') -> np.ndarray:
        """The shared CDF curve over a whole grid ``x`` (the basis for the quantile inversion and the pdf derivative).
        ``method='dehoog'`` (default) uses the accurate de Hoog + monotone-spline representation; ``method='cos'``
        the faster two-pass COS grid."""
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        if method == 'cos':
            xs, cdf = self._cos_cdf_grid
            return np.interp(xa, xs, cdf)
        st = self._dehoog_spline
        g = np.clip(st['spline'](np.clip(xa, 0.0, st['b'])), 0.0, 1.0)
        return st['p0'] + (1 - st['p0']) * g if st['p0'] > 1e-9 else g


class _LSTCumulativeDistributionFunction(_LSTFunction, CumulativeDistributionFunction):
    """The CDF of a 1D accumulated-reward distribution: per-point de Hoog (``__call__``), the fast curve
    (:meth:`curve`) and the plot, on top of the shared :class:`_LSTFunction` machinery."""

    def __call__(self, t):
        """Exact CDF ``P(R <= t)`` by per-point de Hoog inversion. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self._cdf_point(float(x)) for x in np.asarray(t)])
        return self._cdf_point(float(t))

    def curve(self, x, method: str = 'dehoog') -> np.ndarray:
        """Fast CDF over a whole grid ``x`` (for plotting / many-query use). ``method='dehoog'`` (default) uses the
        accurate de Hoog + monotone-spline representation; ``method='cos'`` the faster two-pass COS grid."""
        return self._cdf_curve(x, method)

    def plot(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None,
             exact: bool = False) -> 'plt.Axes':
        """Plot the CDF up to the configured plot-endpoint quantile (fast COS inversion, or per-point de Hoog when
        ``exact=True``)."""
        from ..visualization import Visualization
        d = self._distribution
        if x is None and exact:
            # de Hoog is expensive per point -> place the points adaptively where the curve bends
            x, y = adaptive_grid(self._cdf_point, 0.0, d.quantile(Settings.plot_endpoint_quantile),
                                 max_points=n_points)
        else:
            if x is None:
                x = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), n_points or Settings.plot_n_grid)
            y = self(x) if exact else self.curve(x)
        ax = Visualization.plot(ax=ax, x=x, y=y, xlabel='x', ylabel='F(x)', label=label, file=file,
                                show=show, clear=clear, title=title or d._titled('CDF'))
        ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]
        return ax


class _LSTDensityFunction(_LSTFunction, DensityFunction):
    """The density of a 1D accumulated-reward distribution: per-point de Hoog (``__call__``), the curve as the
    derivative of the shared CDF representation (:meth:`curve`) and the plot."""

    def __call__(self, t, **kwargs):
        """Exact density by per-point de Hoog inversion. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self._pdf_point(float(x)) for x in np.asarray(t)])
        return self._pdf_point(float(t))

    def curve(self, x, method: str = 'dehoog') -> np.ndarray:
        """Fast PDF over a whole grid ``x``: the derivative of the shared CDF representation (``method='dehoog'`` ->
        the de Hoog + monotone-spline; ``method='cos'`` -> the two-pass COS grid). Deriving the PDF from the CDF
        keeps it clean and non-negative; use the per-point ``__call__`` (de Hoog) for the exact pointwise density."""
        d = self._distribution
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        if method == 'cos':
            xs, cdf = self._cos_cdf_grid
            der = np.interp(xa, xs, np.gradient(cdf, xs))
            return d._warn_if_negative(der, d._titled('density (cosine)'))
        st = self._dehoog_spline
        der = st['spline'].derivative()(np.clip(xa, 0.0, st['b']))
        d._warn_if_negative(der, d._titled('density (de Hoog)'))
        der = np.clip(der, 0.0, None)
        return (1 - st['p0']) * der if st['p0'] > 1e-9 else der

    def plot(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None,
             exact: bool = False) -> 'plt.Axes':
        """Plot the PDF up to the configured plot-endpoint quantile (derivative of the COS CDF, or per-point de Hoog
        when ``exact=True``)."""
        from ..visualization import Visualization
        d = self._distribution
        if x is None and exact:
            x, y = adaptive_grid(self._pdf_point, 0.0, d.quantile(Settings.plot_endpoint_quantile),
                                 max_points=n_points)
        else:
            if x is None:
                x = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), n_points or Settings.plot_n_grid)
            y = self(x) if exact else self.curve(x)
        return Visualization.plot(ax=ax, x=x, y=y, xlabel='x', ylabel='f(x)', label=label, file=file,
                                  show=show, clear=clear, title=title or d._titled('PDF'))


class _LSTQuantileFunction(_LSTFunction, QuantileFunction):
    """The quantile function of a 1D accumulated-reward distribution: bisection on the shared (monotone) CDF curve,
    with a per-point de Hoog bisection fallback for the far tail."""

    def __call__(self, q: float, precision: float = 1e-8, max_iter: int = 200, method: str = 'dehoog') -> float:
        """
        The ``q``-quantile ``inf{x : F(x) >= q}`` by bisection on the cached (monotone) CDF *curve* -- the accurate de
        Hoog spline by default (``method='dehoog'``), or the cosine curve (``method='cos'``). The curve is built once,
        so each bisection step is a cheap evaluation. For ``q`` in the far tail beyond the curve's support it falls
        back to the per-point de Hoog bisection.
        """
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1.")

        # at or below the atom mass P(R = 0) the quantile is exactly 0; return it directly rather than letting the
        # bisection converge to a few-1e-9 residue (which makes a relative comparison against an exact 0 blow up)
        if q <= float(self._cdf_curve(0.0, method=method)[0]):
            return 0.0

        b = self._range(scale=12.0)
        if float(self._cdf_curve(b, method=method)[0]) < q:  # beyond the curve's support -> exact de Hoog bisection
            return self._quantile_dehoog(q, precision, max_iter)

        lo, hi = 0.0, b
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if float(self._cdf_curve(mid, method=method)[0]) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < precision:
                break

        return 0.5 * (lo + hi)

    def _quantile_dehoog(self, q: float, precision: float = 1e-8, max_iter: int = 200) -> float:
        """Exact ``q``-quantile by bisection on the per-point de Hoog CDF (a full inversion per step). The robust
        fallback for the far tail, where the cached CDF curve does not reach ``q``."""
        d = self._distribution
        # bracket: grow the upper bound until its CDF exceeds q (seed from the reward's mean via the LST,
        # E[R] = -phi'(0)). The step is scaled by ``1/tau`` so the seed evaluation does not overflow for large-N.
        h = 1e-3 / d._time_scale
        mean = (1.0 - d.lst(h).real) / h
        lo, hi = 0.0, max(mean, 1.0)
        for _ in range(max_iter):
            if self._cdf_point(hi) >= q:
                break
            hi *= 2
        else:
            raise RuntimeError("Failed to bracket the quantile.")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self._cdf_point(mid) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < precision:
                break

        return 0.5 * (lo + hi)

    def plot(self, ax: 'plt.Axes' = None, q: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None,
             exact: bool = False) -> 'plt.Axes':
        """Plot the quantile function (value versus probability), inverting the fast COS CDF curve (or the per-point
        de Hoog bisection when ``exact=True``)."""
        from ..visualization import Visualization
        d = self._distribution
        qe = Settings.plot_endpoint_quantile
        if q is None:
            q = np.linspace(1.0 - qe, qe, n_points or Settings.plot_n_grid)
        if exact:
            y = np.array([self(float(p)) for p in np.atleast_1d(q)])
        else:
            grid = np.linspace(0, self._range(), 512)
            y = np.interp(q, self._cdf_curve(grid), grid)
        return Visualization.plot(ax=ax, x=q, y=y, xlabel='q', ylabel='quantile', label=label, file=file, show=show,
                                  clear=clear, title=title or d._titled('quantile function'))


# --- marginal (per-bin spectrum) flavours ---------------------------------------------------------------------------

class MarginalDensity(DensityFunction):
    """Per-bin marginal densities of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``pdf(x)``: every bin's density at ``x`` by per-point de Hoog inversion.
    - **Plot** ``pdf.plot()``: overlays every bin's fast cosine density curve (or per-point de Hoog with ``exact=True``).
    """


class MarginalCDF(CumulativeDistributionFunction):
    """Per-bin marginal CDFs of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``cdf(x)``: every bin's probability of being at most ``x`` by per-point de Hoog inversion.
    - **Plot** ``cdf.plot()``: overlays every bin's fast cosine CDF curve (or per-point de Hoog with ``exact=True``).
    """


class MarginalQuantileFunction(QuantileFunction):
    """Per-bin marginal quantile functions of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``quantile(q)``: each bin's quantile by bisection on its de Hoog CDF.
    - **Plot** ``quantile.plot()``: overlays every bin's quantile, inverting the fast cosine CDF curve.
    """


# --- joint (bivariate) flavours -------------------------------------------------------------------------------------

class _JointFunction(_SurfacePlottable):
    """Shared machinery for the bivariate joint function objects (:class:`JointCDF` / :class:`JointDensity`): builds
    the plotting grid, evaluates the joint kind on it, and hands the heatmap / 3D surface to :class:`Visualization`.
    The bivariate *representation* (the joint LST grid, the 2D Fourier-cosine expansion, the axis/origin atoms, the
    nested inversion) lives on the :class:`~phasegen.distributions.reward.JointRewardDistribution` this hangs off; the
    subclasses own only the user-facing :meth:`__call__` and the plots."""

    def _joint_grid(self, n_points: int) -> tuple:
        """The plotting grid: each axis runs to the configured marginal quantile (like the 1D plots, so a heavy
        upper tail does not stretch the view), clipped to the cosine window the representation was built on."""
        d = self._distribution
        st = d._cos2d
        q = Settings.plot_endpoint_quantile
        xs = np.linspace(0, min(d.marginal('a').quantile(q), st['ba']), n_points)
        ys = np.linspace(0, min(d.marginal('b').quantile(q), st['bb']), n_points)
        return xs, ys

    def _grid_values(self, xs: np.ndarray, ys: np.ndarray, dehoog: bool) -> np.ndarray:
        """The joint kind evaluated on the grid ``xs x ys`` (implemented per kind)."""
        raise NotImplementedError

    def _default_n_points(self, dehoog: bool, surface: bool) -> int:
        """Default grid resolution (implemented per kind; the slow nested de Hoog uses a coarser grid)."""
        raise NotImplementedError

    def _joint_title(self) -> str:
        d = self._distribution
        return f"Joint {self.kind.upper()} {d.label}" if d.label else f"Joint reward {self.kind.upper()}"

    def _draw(self, surface: bool, ax, n_points, show, file, title, method) -> 'plt.Axes':
        from ..visualization import Visualization
        d = self._distribution
        dehoog = d._use_dehoog(method)
        n_points = n_points or self._default_n_points(dehoog, surface)
        xs, ys = self._joint_grid(n_points)
        if dehoog:
            d._logger.info("Computing the joint %s by direct nested de Hoog inversion on a %dx%d grid; this is slow.",
                           self.kind.upper(), len(xs), len(ys))
        Z = self._grid_values(xs, ys, dehoog)
        is_cdf = self.kind == 'cdf'  # a CDF is a probability -> fix its scale to [0, 1]
        return Visualization.plot_surface(
            xs, ys, Z, surface=surface, ax=ax, xlabel='$R_a$', ylabel='$R_b$',
            zlabel='F(R_a, R_b)' if is_cdf else 'f(R_a, R_b)', title=title or self._joint_title(),
            vmin=0.0 if is_cdf else None, vmax=1.0 if is_cdf else None, file=file, show=show,
        )

    def plot(self, ax: 'plt.Axes' = None, n_points: int = None, show: bool = True, file: str = None,
             title: str = None, method: str = 'cos') -> 'plt.Axes':
        """Heatmap of the joint function. ``method='dehoog'`` uses the accurate nested de Hoog inversion (a coarser
        default grid); the default ``'cos'`` uses the fast cosine reconstruction."""
        return self._draw(False, ax, n_points, show, file, title, method)

    def plot_surface(self, ax: 'plt.Axes' = None, n_points: int = None, show: bool = True, file: str = None,
                     title: str = None, method: str = 'cos') -> 'plt.Axes':
        """3D surface of the joint function (see :meth:`plot` for ``method``)."""
        return self._draw(True, ax, n_points, show, file, title, method)


class JointDensity(_JointFunction, DensityFunction):
    """Joint density of two rewards / bins (the within-tree pair of branch lengths).

    - **Callable** ``pdf(x, y)``: the continuous part of the joint law -- the accurate nested de Hoog inversion by
      default, or the fast 2D cosine expansion with ``method='cos'``. Accepts scalars or arrays.
    - **Plot** ``pdf.plot()`` / ``pdf.plot_surface()``: heatmap / 3D surface of the fast cosine density, or the
      nested de Hoog with ``method='dehoog'``.
    """

    def __call__(self, x, y, method: str = None):
        """Joint probability density of ``(R_a, R_b)`` (the continuous, both-positive part). The distribution also has
        atom mass on the axes where a reward is zero (a non-empty SFS bin pair has none there). ``method`` selects the
        inversion: ``'cos'`` the fast cosine expansion, ``'dehoog'`` / ``None`` the accurate nested de Hoog."""
        d = self._distribution
        if d._is_diagonal:
            raise NotImplementedError("The joint density is singular when both rewards are identical (R_a = R_b "
                                      "almost surely): the law lives on the diagonal and has no 2D density. Use "
                                      "cdf(x, y) = marginal CDF at min(x, y), or the 1D marginal density.")
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        if d._use_dehoog(method):
            f = d._density_nested(xs, ys)
        else:
            raw = d._density(xs, ys)  # the cosine 2D density can dip negative near the origin edge (Gibbs)
            d._warn_if_negative(raw, 'joint density (cosine)')
            f = np.clip(raw, 0.0, None)
        return float(f.ravel()[0]) if f.size == 1 else f

    def _grid_values(self, xs, ys, dehoog):
        d = self._distribution
        return d._density_nested(xs, ys) if dehoog else np.clip(d._density(xs, ys), 0.0, None)

    def _default_n_points(self, dehoog, surface):
        return 25 if dehoog else (80 if surface else 120)


class JointCDF(_JointFunction, CumulativeDistributionFunction):
    """Joint CDF of two rewards / bins -- the probability both are at most their thresholds.

    - **Callable** ``cdf(x, y)``: the axis atoms (where a reward is zero) plus the continuous part -- the accurate
      nested de Hoog box by default (no near-origin bias for skewed multi-epoch rewards), or the fast 2D cosine box
      with ``method='cos'``. Accepts scalars or arrays.
    - **Plot** ``cdf.plot()`` / ``cdf.plot_surface()``: heatmap / 3D surface of the fast cosine box, or the nested de
      Hoog box with ``method='dehoog'``.
    """

    def __call__(self, x, y, method: str = None):
        """Joint CDF ``P(R_a <= x, R_b <= y)``: the axis atoms plus the continuous box integral. ``method`` selects
        the box method: ``'cos'`` the fast cosine box, ``'dehoog'`` / ``None`` the accurate nested de Hoog. When both
        rewards are identical the law is singular on the diagonal and the CDF reduces to ``P(R <= min(x, y))``."""
        d = self._distribution
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        if d._is_diagonal:
            m = d.marginal('a')
            # at t = 0 the marginal CDF is the atom P(R = 0) (the de Hoog inversion misses the jump there)
            G = np.array([[float(d._atoms['both0'] if min(xx, yy) <= 0.0 else m.cdf(min(xx, yy)))
                           for yy in ys] for xx in xs])
        else:
            G = d._cdf_grid(xs, ys, dehoog=d._use_dehoog(method))
        return float(G.ravel()[0]) if G.size == 1 else G

    def _grid_values(self, xs, ys, dehoog):
        return self._distribution._cdf_grid(xs, ys, dehoog=dehoog)

    def _default_n_points(self, dehoog, surface):
        return 25 if dehoog else 60


# (a bivariate joint has no quantile flavour: a 2D quantile is not well-defined -- use a marginal or conditional)


# --- conditional flavours -------------------------------------------------------------------------------------------

class ConditionalDensity(_LSTDensityFunction):
    """Density of one reward conditional on another being held at a value (e.g. one bin's length given another's).

    - **Callable** ``pdf(y)``: the exact pointwise conditional density by per-point de Hoog of the nested-inversion
      conditional transform (an inner inversion along the conditioned axis, then the outer one).
    - **Plot** ``pdf.plot()``: the de Hoog spline density curve; pass ``exact=True`` for the per-point density.
    """


class ConditionalCDF(_LSTCumulativeDistributionFunction):
    """CDF of one reward conditional on another being held at a value.

    - **Callable** ``cdf(y)``: the exact pointwise conditional CDF by per-point de Hoog of the nested-inversion
      conditional transform.
    - **Plot** ``cdf.plot()``: the de Hoog spline CDF curve; pass ``exact=True`` for the per-point CDF.
    """


class ConditionalQuantileFunction(_LSTQuantileFunction):
    """Quantile function of one reward conditional on another being held at a value.

    - **Callable** ``quantile(q)``: bisection on the conditional de Hoog CDF.
    - **Plot** ``quantile.plot()``: the conditional quantile over a probability grid.
    """


class CallableDistributionFunctions:
    """
    Mixin exposing ``pdf`` / ``cdf`` / ``quantile`` as callable-and-plottable distribution-function properties. Each
    concrete distribution supplies the evaluators ``_pdf`` / ``_cdf`` / ``_quantile`` and the plotters ``_plot_pdf`` /
    ``_plot_cdf`` / ``_plot_quantile``; this mixin wires them together. Subclasses pick the *flavour* of the returned
    function objects by overriding :attr:`_pdf_function` / :attr:`_cdf_function` / :attr:`_quantile_function` (e.g. a
    spectrum returns the ``Marginal...`` flavours, a conditional the ``Conditional...`` flavours), which is what gives
    IDEs the right type and docstring. Also keeps the former ``plot_pdf`` / ``plot_cdf`` methods as deprecated aliases.
    """
    #: The distribution-function classes returned by the properties; overridden by subclasses to select the flavour.
    #: ``_quantile_function = None`` marks a distribution without a quantile (e.g. a bivariate joint).
    _pdf_function = DensityFunction
    _cdf_function = CumulativeDistributionFunction
    _quantile_function = QuantileFunction

    def _function(self, kind: str, factory):
        """Return the (cached) distribution-function object for ``kind``, built once via ``factory`` and stored on
        this distribution. Caching the object -- not just rebuilding a thin wrapper -- is what lets the function
        object's own cached curves (the de Hoog spline, the COS coefficients) persist across ``.cdf`` / ``.pdf`` /
        ``.quantile`` accesses, since the three share the one distribution they hang off. Honors the global cache
        switch (:attr:`Settings.cache`)."""
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

    def plot_cdf(self, *args, **kwargs):
        """Deprecated: use :attr:`cdf`.plot() instead."""
        warnings.warn("plot_cdf() is deprecated; use .cdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self.cdf.plot(*args, **kwargs)

    def plot_pdf(self, *args, **kwargs):
        """Deprecated: use :attr:`pdf`.plot() instead."""
        warnings.warn("plot_pdf() is deprecated; use .pdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self.pdf.plot(*args, **kwargs)

    def _warn_if_negative(self, values: np.ndarray, label: str, rtol: float = 1e-3) -> np.ndarray:
        """Warn (via this distribution's logger) if ``values`` has a substantial negative entry relative to its scale,
        then return it unchanged (the caller clips). A density / probability must be non-negative, so a real negative
        -- beyond the ``rtol`` numerical-noise band -- signals inversion ringing (Gibbs) worth surfacing rather than
        silently clipping. Gated by :attr:`Settings.check_inversions`."""
        arr = np.asarray(values, dtype=float)
        if Settings.check_inversions and arr.size:
            scale = max(float(np.abs(arr).max()), 1e-300)
            mn = float(np.nanmin(arr))
            if mn < -rtol * scale:
                self._logger.warning(f"{label}: substantial negative value ({mn:.2e} vs scale {scale:.2e}); clipping "
                                     f"to 0 -- the numerical inversion may be imprecise here")
        return values

    def _warn_if_nonmonotone(self, cdf: np.ndarray, label: str, rtol: float = 1e-3) -> np.ndarray:
        """Warn (via this distribution's logger) if ``cdf`` has a substantial downward step relative to its range, then
        return it unchanged (the caller enforces monotonicity). A CDF must be non-decreasing, so a real drop -- beyond
        the ``rtol`` numerical-noise band -- signals inversion ringing (a wiggle). Gated by
        :attr:`Settings.check_inversions`."""
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
    Abstract base class for probability distributions for which moments can be calculated.
    """

    def __init__(self):
        """
        Create object.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

    def touch(self, **kwargs: dict):
        """
        Touch all cached properties.

        :param kwargs: Additional keyword arguments.
        """
        for cls in self.__class__.__mro__:
            for attr, value in cls.__dict__.items():
                if isinstance(value, cached_property):
                    getattr(self, attr)


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

    def __init__(self, dist: 'PhaseTypeDistribution'):
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item):
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
    def loci(self) -> 'MarginalLocusDistributions':
        """
        Distributions marginalized over loci.
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
                reward=CombinedReward([self.dist.reward, LocusReward(locus)])
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
                CombinedReward([self.dist.reward, LocusReward(locus1)]),
                CombinedReward([self.dist.reward, LocusReward(locus2)])
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
        :return: The correlation coefficient.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        return self.get_cov(locus1, locus2) / (self.loci[locus1].std * self.loci[locus2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_corr(i, j) for i in range(n_loci)] for j in range(n_loci)])

    def joint_distribution(self, locus1: int, locus2: int) -> 'JointRewardDistribution':
        """
        Joint distribution of the accumulated reward at ``locus1`` and at ``locus2`` — e.g. the per-locus tree height
        or total branch length at two loci separated by recombination. This is the distributional object behind the
        cross-locus covariance :meth:`get_cov`/:meth:`get_corr`: it is built from the host distribution's own reward
        restricted to each locus (``CombinedReward([reward, LocusReward(locus)])``, exactly the cross-moment reward
        pair used there), so its marginals are the per-locus distributions :attr:`loci` and it shares the joint LST /
        2D inversion machinery with the within-tree SFS pairwise joint.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The joint accumulated-reward distribution across the two loci.
        """
        locus1, locus2 = int(locus1), int(locus2)

        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        return self.dist.joint_distribution(
            CombinedReward([self.dist.reward, LocusReward(locus1)]),
            CombinedReward([self.dist.reward, LocusReward(locus2)])
        )


class MarginalDemeDistributions(MarginalDistributions):
    """
    Marginal deme distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution'):
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item):
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
    def demes(self) -> 'MarginalDemeDistributions':
        """
        Distributions marginalized over demes.
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
                reward=CombinedReward([self.dist.reward, DemeReward(pop)])
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
                CombinedReward([self.dist.reward, DemeReward(pop1)]),
                CombinedReward([self.dist.reward, DemeReward(pop2)])
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
        :return: The correlation coefficient.
        """
        return self.get_cov(pop1, pop2) / (self.demes[pop1].std * self.demes[pop2].std)

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
    ``cdf`` / ``pdf`` / ``quantile`` are exposed as callable-and-plottable :class:`DistributionFunction`s (see
    :class:`CallableDistributionFunctions`); subclasses implement the ``_cdf`` / ``_pdf`` / ``_quantile`` evaluators.
    """

    @abstractmethod
    def _cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Value or values to evaluate the CDF at.
        :return: CDF.
        """
        pass

    @abstractmethod
    def _quantile(self, q: float) -> float:
        """
        Get the qth quantile.
        """
        pass

    @abstractmethod
    def _pdf(self, t: float | Sequence[float], **kwargs: dict) -> float | np.ndarray:
        """
        Density function.

        :param t: Value or values to evaluate the density function at.
        :param kwargs: Additional keyword arguments.
        :return: Density.
        """
        pass

    def _plot_quantile(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Quantile function'
    ) -> 'plt.Axes':
        """
        Plot the quantile function (value versus probability ``q``).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate the quantile at. By default, 99 evenly spaced values in ``(0, 1)``.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if q is None:
            q = np.linspace(1.0 - Settings.plot_endpoint_quantile, Settings.plot_endpoint_quantile, Settings.plot_n_grid)

        return Visualization.plot(
            ax=ax,
            x=q,
            y=np.array([self._quantile(float(p)) for p in q]),
            xlabel='q',
            ylabel='quantile',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

    def _plot_cdf(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height CDF'
    ) -> 'plt.Axes':
        """
        Plot cumulative distribution function.

        :param ax: Axes to plot on.
        :param t: Values to evaluate the CDF at. By default, 200 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if t is None:
            t = np.linspace(0, self._quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

        ax = Visualization.plot(
            ax=ax,
            x=t,
            y=self._cdf(t),
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )
        ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]
        return ax

    def _plot_pdf(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height PDF',
            dx: float = None
    ) -> 'plt.Axes':
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param t: Values to evaluate the density function at.
            By default, 200 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :param dx: Step size for numerical differentiation. By default, the 99th percentile divided by 1e10.
        :return: Axes.
        """
        from ..visualization import Visualization

        if dx is None:
            dx = self._quantile(Settings.plot_endpoint_quantile) / 1e10

        if t is None:
            t = np.linspace(0, self._quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self._pdf(t, dx=dx),
            xlabel='t',
            ylabel='f(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

