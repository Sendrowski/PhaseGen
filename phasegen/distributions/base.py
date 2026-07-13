"""Distribution base classes and marginal (per-deme / per-locus) views."""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from ..caching import cached_property
from typing import Any, Callable, Iterator, Sequence, TYPE_CHECKING
import numpy as np
from ..expm import Backend
from ..rewards import DemeReward, LocusReward, CombinedReward
from ..settings import Settings

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .phase_type import PhaseTypeDistribution

expm = Backend.expm
logger = logging.getLogger('phasegen')


def adaptive_grid(f, a: float, b: float, n_init: int = 9, tol: float = None, max_points: int = None) -> 'Tuple[np.ndarray, np.ndarray]':
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

    Each property returns one of the typed subclasses (:class:`DensityFunction` /
    :class:`CumulativeDistributionFunction` / :class:`QuantileFunction`, in plain, ``Marginal...``, ``Joint...`` and
    ``Conditional...`` flavours), whose docstrings describe *what* that function is and *how* it is computed.

    The function holds its owning ``distribution`` and dispatches by :attr:`kind` to the distribution's ``_<kind>``
    (evaluate) and ``_plot_<kind>`` (plot).

    :param distribution: The distribution this function belongs to.
    """
    #: Short kind label (``'pdf'`` / ``'cdf'`` / ``'quantile'``), set by the kind subclasses; selects the
    #: distribution's ``_<kind>`` / ``_plot_<kind>`` methods and is used in ``repr``.
    kind: str = ''

    def __init__(self, distribution: 'CallableDistributionFunctions') -> None:
        self._distribution = distribution

    def __call__(self, *args, **kwargs) -> 'Any':
        """Evaluate the distribution function at the given point(s) (the distribution's ``_<kind>``)."""
        return getattr(self._distribution, '_' + self.kind)(*args, **kwargs)

    def plot(self, *args, **kwargs) -> 'plt.Axes':
        """
        Plot the distribution function (the distribution's ``_plot_<kind>``). Accepted arguments depend on the
        distribution; common ones are ``exact`` (use the slower per-point de Hoog inversion instead of the default
        cosine one), ``bins`` / ``configs`` (select which spectrum bins to draw), ``n_points`` (grid resolution),
        ``ax`` / ``show`` / ``file`` / ``title``.
        """
        return getattr(self._distribution, '_plot_' + self.kind)(*args, **kwargs)

    def __repr__(self) -> str:
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

    - **Callable** ``pdf(x)``: the density at ``x`` (scalar or array). For an accumulated reward this is the
      derivative of the cosine CDF; ``method='dehoog'`` gives the exact per-point Laplace inversion instead. The tree
      height uses the exact matrix exponential, empirical samples a histogram.
    - **Plot** ``pdf.plot()``: the same, or the per-point inversion with ``exact=True``.
    """
    kind = 'pdf'


class CumulativeDistributionFunction(DistributionFunction):
    """Cumulative distribution function -- the probability of being at most ``x``.

    - **Callable** ``cdf(x)``: the probability at ``x`` (scalar or array). For an accumulated reward this is the
      Fourier-cosine inversion; ``method='dehoog'`` gives the exact per-point Laplace inversion instead. The tree
      height uses the exact matrix exponential, samples the empirical CDF.
    - **Plot** ``cdf.plot()``: the same, or the per-point inversion with ``exact=True``.
    """
    kind = 'cdf'


class QuantileFunction(DistributionFunction):
    """Quantile function -- the inverse CDF.

    - **Callable** ``quantile(q)``: the value at which the CDF reaches ``q`` (scalar or array). For an accumulated
      reward this inverts the same cosine CDF the :class:`CumulativeDistributionFunction` reads, so the two are
      mutually consistent; in the far tail it falls back to a bisection on the exact per-point inversion, which
      ``method='dehoog'`` selects throughout. Empirical data uses the sample quantile.
    - **Plot** ``quantile.plot()``: the same, or the per-point bisection with ``exact=True``.
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

    There are two representations, and the cdf / pdf / quantile all go through both:

    - the **two-pass Fourier-cosine grid** (``method='cos'``, the default): one fit answers a whole array, so it is
      the vectorised route. Cached on the *distribution* (the single object the cdf / pdf / quantile of one
      distribution hang off, see :meth:`CallableDistributionFunctions._function`), so all three reuse one fit.
    - the **per-point de Hoog inversion** (``method='dehoog'``): one Laplace inversion per point, exact. It is the
      reference the cosine grid is validated against, and the far tail of the quantile falls back to it.

    Cosine is the default because it is what the scenario suite can actually test at scale: against msprime the two
    agree to within sampling noise except on kinked densities, where cosine's Gibbs ringing costs it (a 2-epoch rapid
    decline: 2e-3 vs 3e-4 against the per-point reference), while costing 5-40x less.
    """
    #: Above this quantile the cosine grid is not trusted, and the quantile falls back to the exact per-point de Hoog
    #: bisection. The cosine CDF force-normalises to 1 at the window end, so the usual ``cdf(b) < q`` tail-fallback
    #: test can never fire; and where the density is low a small CDF ripple is a large quantile error (``q = 0.999``
    #: drifts 1.5-3%, against <=0.1% at ``q = 0.99``). ``None`` disables the fallback.
    _cos_quantile_max: float = 0.99

    #: Cosine terms for the coarse support-locating pass and the fine accuracy pass of the two-pass COS fit.
    _cos_terms_rough: int = 128
    _cos_terms: int = 384

    # ---- distribution primitives (thin accessors) --------------------------------------------------------------
    def _range(self, scale: float = 12.0) -> float:
        return self._distribution._range(scale)

    def _cdf_point(self, t: float) -> float:
        """Per-point de Hoog CDF ``P(R <= t)`` (``L[CDF] = phi(s) / s``) -- the building block of both the exact
        :meth:`_LSTCumulativeDistributionFunction.__call__` (``method='dehoog'``) and the far-tail quantile."""
        if t < 0:
            return 0.0
        d = self._distribution
        if t == 0:
            # F(0) = P(R <= 0) = P(R = 0), the atom phi(inf) -- right-continuous at the point mass, matching the
            # de Hoog / cosine curves (which split the atom off and add it back). The inversion below is skipped
            # both to avoid the phi(s)/s singularity and because at t > 0 it already carries the atom.
            return max(d.lst(d._s_inf).real, 0.0)
        return d._invert(lambda s: d.lst(s) / s, float(t))

    def _pdf_point(self, t: float) -> float:
        """Per-point de Hoog density (``L[pdf] = phi(s)``)."""
        d = self._distribution
        return d._invert(d.lst, float(t))

    # ---- shared CDF representation (cached on the distribution) -------------------------------------------------
    def _shared(self, key: str, build) -> 'Any':
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
        p0 = d.lst(d._s_inf).real
        w = np.arange(n_terms) * np.pi / b
        chi = np.array([d.lst(-1j * wk) for wk in w])
        if p0 > 1e-9:
            chi = (chi - p0) / (1 - p0)  # continuous part only
        fk = (2.0 / b) * np.real(chi)  # a = 0, so exp(-i w a) = 1
        fk[0] *= 0.5

        # the largest backward step of the (continuous) CDF is the sensitive ringing detector (a visibly rippling CDF
        # can come from sub-percent density wiggles); the shared non-monotonicity guard surfaces a substantial one
        # (rtol 1e-2 of the [0, 1] CDF range -- a loose bar, the cosine series being coarse near a sharp feature)
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

    def _cos_cdf(self, x) -> np.ndarray:
        """The cosine CDF over ``x``, read off the cached monotone grid (so cdf / pdf / quantile stay consistent)."""
        xs, cdf = self._cos_cdf_grid
        return np.interp(np.atleast_1d(np.asarray(x, dtype=float)), xs, cdf)


class _LSTCumulativeDistributionFunction(_LSTFunction, CumulativeDistributionFunction):
    """The CDF of a 1D accumulated-reward distribution, on top of the shared :class:`_LSTFunction` machinery."""

    def __call__(self, t, method: str = 'cos') -> 'np.ndarray | float':
        """
        CDF ``P(R <= t)``, for a scalar or an array of ``t``. ``method='cos'`` (default) reads the cached cosine grid,
        so a whole array costs one fit; ``method='dehoog'`` inverts per point (exact, one Laplace inversion each).

        :param t: Point(s) at which to evaluate the CDF.
        :param method: Inversion route, ``'cos'`` or ``'dehoog'``.
        :return: The CDF at ``t``, of the same shape.
        """
        if method == 'cos':
            out = self._cos_cdf(t)
        else:
            out = np.array([self._cdf_point(float(x)) for x in np.atleast_1d(np.asarray(t, dtype=float))])
        return out if np.ndim(t) > 0 else float(out[0])

    def plot(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None,
             exact: bool = False, **kwargs) -> 'plt.Axes':
        """Plot the CDF up to the configured plot-endpoint quantile (the cosine grid, or per-point de Hoog when
        ``exact=True``). Extra keyword arguments (``alpha``, ``lw``, ...) are forwarded to the line."""
        from ..visualization import Visualization
        d = self._distribution
        if x is None and exact:
            # de Hoog is expensive per point -> place the points adaptively where the curve bends
            x, y = adaptive_grid(self._cdf_point, 0.0, d.quantile(Settings.plot_endpoint_quantile),
                                 max_points=n_points)
        else:
            if x is None:
                x = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), n_points or Settings.plot_n_grid)
            y = self(x, method='dehoog') if exact else self(x)
        ax = Visualization.plot(ax=ax, x=x, y=y, xlabel='x', ylabel='F(x)', label=label, file=file,
                                show=show, clear=clear, title=title or d._titled('CDF'), **kwargs)
        ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]
        return ax


class _LSTDensityFunction(_LSTFunction, DensityFunction):
    """The density of a 1D accumulated-reward distribution."""

    def __call__(self, t, method: str = 'cos', **kwargs) -> 'np.ndarray | float':
        """
        Density, for a scalar or an array of ``t``. ``method='cos'`` (default) differentiates the cached cosine CDF,
        which keeps the density consistent with it and free of the raw cosine sum's Gibbs negativity;
        ``method='dehoog'`` inverts per point (exact).

        :param t: Point(s) at which to evaluate the density.
        :param method: Inversion route, ``'cos'`` or ``'dehoog'``.
        :return: The density at ``t``, of the same shape.
        """
        d = self._distribution
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        if method == 'cos':
            xs, cdf = self._cos_cdf_grid
            out = np.interp(ta, xs, np.gradient(cdf, xs))
            out = d._warn_if_negative(out, d._titled('density (cosine)'))
        else:
            out = np.array([self._pdf_point(float(x)) for x in ta])
        return out if np.ndim(t) > 0 else float(out[0])

    def plot(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None,
             exact: bool = False, **kwargs) -> 'plt.Axes':
        """Plot the PDF up to the configured plot-endpoint quantile (derivative of the cosine CDF, or per-point de Hoog
        when ``exact=True``). Extra keyword arguments (``alpha``, ``lw``, ...) are forwarded to the line."""
        from ..visualization import Visualization
        d = self._distribution
        if x is None and exact:
            x, y = adaptive_grid(self._pdf_point, 0.0, d.quantile(Settings.plot_endpoint_quantile),
                                 max_points=n_points)
        else:
            if x is None:
                x = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), n_points or Settings.plot_n_grid)
            y = self(x, method='dehoog') if exact else self(x)
        return Visualization.plot(ax=ax, x=x, y=y, xlabel='x', ylabel='f(x)', label=label, file=file,
                                  show=show, clear=clear, title=title or d._titled('PDF'), **kwargs)


class _LSTQuantileFunction(_LSTFunction, QuantileFunction):
    """The quantile function of a 1D accumulated-reward distribution: inverse interpolation of the cosine CDF grid,
    with a per-point de Hoog bisection for the far tail."""

    def __call__(self, q, precision: float = 1e-8, max_iter: int = 200, method: str = 'cos') -> 'np.ndarray | float':
        """
        The ``q``-quantile ``inf{x : F(x) >= q}``, for a scalar or an array of ``q``. ``method='cos'`` (default)
        inverts the cached monotone cosine CDF grid by interpolation, so a whole array is one vectorised pass;
        ``method='dehoog'`` bisects on the per-point inversion (a full Laplace inversion per step).

        Beyond :attr:`_cos_quantile_max` the cosine grid is not trusted and the de Hoog bisection takes over: the
        cosine CDF force-normalises to 1 at the window end, and where the density is low a small CDF ripple is a large
        quantile error. At or below the atom mass ``P(R = 0)`` the quantile is exactly 0.

        Just *above* a large atom the cosine quantile is accurate in absolute terms but loses relative precision,
        because the quantile is itself near zero there: for a bin empty with probability 0.44, ``q = 0.5`` lands at
        0.041 against the exact 0.036 (16% relative, but 0.006 absolute against a 0.95-quantile of 13.3). This is the
        cosine series' Gibbs artifact at the jump, and it decays away from the atom (4% at ``q = 0.6``, 0.04% at
        ``q = 0.95``). Use ``method='dehoog'`` where a small quantile's relative precision matters.

        :param q: Probability level(s) in ``[0, 1]``.
        :param precision: Absolute convergence tolerance of the de Hoog bisection.
        :param max_iter: Maximum bisection / bracketing iterations.
        :param method: Inversion route, ``'cos'`` or ``'dehoog'``.
        :return: The quantile(s), of the same shape as ``q``.
        :raises ValueError: If any ``q`` lies outside ``[0, 1]``.
        """
        qa = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((qa < 0) | (qa > 1)):
            raise ValueError("Quantile must be between 0 and 1.")

        if method == 'cos':
            xs, cdf = self._cos_cdf_grid
            out = np.interp(qa, cdf, xs)  # the grid is monotone, so the inverse is an interpolation
            tail = qa > self._cos_quantile_max if self._cos_quantile_max is not None else np.zeros_like(qa, bool)
        else:
            out, tail = np.zeros_like(qa), np.ones_like(qa, dtype=bool)

        for i in np.flatnonzero(tail):
            out[i] = self._quantile_dehoog(float(qa[i]), precision, max_iter)

        return out if np.ndim(q) > 0 else float(out[0])

    def _quantile_dehoog(self, q: float, precision: float = 1e-8, max_iter: int = 200) -> float:
        """Exact ``q``-quantile by bisection on the per-point de Hoog CDF (a full inversion per step). The robust
        route for the far tail, where the cosine grid is not trusted."""
        d = self._distribution

        # at or below the atom mass P(R = 0) the quantile is exactly 0; return it directly rather than letting the
        # bisection converge to a few-1e-9 residue (which makes a relative comparison against an exact 0 blow up)
        if q <= self._cdf_point(0.0):
            return 0.0
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
             exact: bool = False, **kwargs) -> 'plt.Axes':
        """Plot the quantile function (value versus probability), inverting the fast COS CDF curve (or the per-point
        de Hoog bisection when ``exact=True``). Extra keyword arguments (``alpha``, ``lw``, ...) are forwarded to the
        line."""
        from ..visualization import Visualization
        d = self._distribution
        qe = Settings.plot_endpoint_quantile
        if q is None:
            q = np.linspace(1.0 - qe, qe, n_points or Settings.plot_n_grid)
        y = self(q, method='dehoog') if exact else self(q)
        return Visualization.plot(ax=ax, x=q, y=y, xlabel='q', ylabel='quantile', label=label, file=file, show=show,
                                  clear=clear, title=title or d._titled('quantile function'), **kwargs)


# --- direct grid evaluation (matrix-exponential tree height, empirical samples) --------------------------------------

class _GridCumulativeDistributionFunction(CumulativeDistributionFunction):
    """CDF whose distribution computes ``P(R <= t)`` *directly* (the exact matrix-exponential tree height, the
    empirical sample estimate) rather than by Laplace inversion. The evaluation lives in the subclass ``__call__``
    (reaching into ``self._distribution`` for the state space / demography / samples); :meth:`plot` draws it on a
    uniform grid up to the configured endpoint quantile, via :class:`Visualization`."""

    def plot(self, ax: 'plt.Axes' = None, t: np.ndarray = None, show: bool = True, file: str = None,
             clear: bool = True, label: str = None, title: str = 'CDF') -> 'plt.Axes':
        from ..visualization import Visualization
        if t is None:
            t = np.linspace(0, self._distribution.quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)
        ax = Visualization.plot(ax=ax, x=t, y=self(t), xlabel='t', ylabel='F(t)', label=label, file=file,
                                show=show, clear=clear, title=title)
        ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]
        return ax


class _GridDensityFunction(DensityFunction):
    """Density whose distribution computes it directly (see :class:`_GridCumulativeDistributionFunction`)."""

    def plot(self, ax: 'plt.Axes' = None, t: np.ndarray = None, show: bool = True, file: str = None,
             clear: bool = True, label: str = None, title: str = 'PDF', dx: float = None) -> 'plt.Axes':
        from ..visualization import Visualization
        d = self._distribution
        if dx is None:
            dx = d.quantile(Settings.plot_endpoint_quantile) / 1e10
        if t is None:
            t = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)
        return Visualization.plot(ax=ax, x=t, y=self(t, dx=dx), xlabel='t', ylabel='f(t)', label=label, file=file,
                                  show=show, clear=clear, title=title)


class _GridQuantileFunction(QuantileFunction):
    """Quantile function whose distribution computes it directly (see :class:`_GridCumulativeDistributionFunction`)."""

    def plot(self, ax: 'plt.Axes' = None, q: np.ndarray = None, show: bool = True, file: str = None,
             clear: bool = True, label: str = None, title: str = 'Quantile function') -> 'plt.Axes':
        from ..visualization import Visualization
        if q is None:
            q = np.linspace(1.0 - Settings.plot_endpoint_quantile, Settings.plot_endpoint_quantile, Settings.plot_n_grid)
        return Visualization.plot(ax=ax, x=q, y=np.array([self(float(p)) for p in q]), xlabel='q', ylabel='quantile',
                                  label=label, file=file, show=show, clear=clear, title=title)


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

    def _grid_values(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The joint kind evaluated on the grid ``xs x ys`` (implemented per kind)."""
        raise NotImplementedError

    def _default_n_points(self, surface: bool) -> int:
        """Default grid resolution (implemented per kind)."""
        raise NotImplementedError

    def _joint_title(self) -> str:
        d = self._distribution
        return f"Joint {self.kind.upper()} {d.label}" if d.label else f"Joint reward {self.kind.upper()}"

    def _draw(self, surface: bool, ax, n_points, show, file, title) -> 'plt.Axes':
        from ..visualization import Visualization
        n_points = n_points or self._default_n_points(surface)
        xs, ys = self._joint_grid(n_points)
        Z = self._grid_values(xs, ys)
        is_cdf = self.kind == 'cdf'  # a CDF is a probability -> fix its scale to [0, 1]
        return Visualization.plot_surface(
            xs, ys, Z, surface=surface, ax=ax, xlabel='$R_a$', ylabel='$R_b$',
            zlabel='F(R_a, R_b)' if is_cdf else 'f(R_a, R_b)', title=title or self._joint_title(),
            vmin=0.0 if is_cdf else None, vmax=1.0 if is_cdf else None, file=file, show=show,
        )

    def plot(self, ax: 'plt.Axes' = None, n_points: int = None, show: bool = True, file: str = None,
             title: str = None) -> 'plt.Axes':
        """Heatmap of the joint function."""
        return self._draw(False, ax, n_points, show, file, title)

    def plot_surface(self, ax: 'plt.Axes' = None, n_points: int = None, show: bool = True, file: str = None,
                     title: str = None) -> 'plt.Axes':
        """3D surface of the joint function."""
        return self._draw(True, ax, n_points, show, file, title)


class JointDensity(_JointFunction, DensityFunction):
    """Joint density of two rewards / bins (the within-tree pair of branch lengths).

    - **Callable** ``pdf(x, y)``: the continuous part of the joint law, by 2D cosine expansion. Accepts scalars or
      arrays.
    - **Plot** ``pdf.plot()`` / ``pdf.plot_surface()``: heatmap / 3D surface of the density.
    """

    def __call__(self, x, y) -> 'np.ndarray | float':
        """Joint probability density of ``(R_a, R_b)`` (the continuous, both-positive part). The distribution also has
        atom mass on the axes where a reward is zero (a non-empty SFS bin pair has none there)."""
        d = self._distribution
        if d._is_diagonal:
            raise NotImplementedError("The joint density is singular when both rewards are identical (R_a = R_b "
                                      "almost surely): the law lives on the diagonal and has no 2D density. Use "
                                      "cdf(x, y) = marginal CDF at min(x, y), or the 1D marginal density.")
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        f = self._grid_values(xs, ys)
        return float(f.ravel()[0]) if f.size == 1 else f

    def _grid_values(self, xs, ys) -> 'np.ndarray':
        d = self._distribution
        raw = d._density(xs, ys)  # the cosine 2D density can dip negative near the origin edge (Gibbs)
        d._warn_if_negative(raw, 'joint density (cosine)')
        return np.clip(raw, 0.0, None)

    def _default_n_points(self, surface) -> int:
        return 80 if surface else 120


class JointCDF(_JointFunction, CumulativeDistributionFunction):
    """Joint CDF of two rewards / bins -- the probability both are at most their thresholds.

    - **Callable** ``cdf(x, y)``: the axis atoms (where a reward is zero) plus the continuous cosine box integral.
      Accepts scalars or arrays.
    - **Plot** ``cdf.plot()`` / ``cdf.plot_surface()``: heatmap / 3D surface of the box CDF.
    """

    def __call__(self, x, y) -> 'np.ndarray | float':
        """Joint CDF ``P(R_a <= x, R_b <= y)``: the axis atoms plus the continuous box integral. When both rewards are
        identical the law is singular on the diagonal and the CDF reduces to ``P(R <= min(x, y))``."""
        d = self._distribution
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        if d._is_diagonal:
            m = d.marginal('a')
            # at t = 0 the marginal CDF is the atom P(R = 0) (the de Hoog inversion misses the jump there)
            G = np.array([[float(d._atoms['both0'] if min(xx, yy) <= 0.0 else m.cdf(min(xx, yy)))
                           for yy in ys] for xx in xs])
        else:
            G = d._cdf_grid(xs, ys)
        return float(G.ravel()[0]) if G.size == 1 else G

    def _grid_values(self, xs, ys) -> 'np.ndarray':
        return self._distribution._cdf_grid(xs, ys)

    def _default_n_points(self, surface) -> int:
        return 60


# (a bivariate joint has no quantile flavour: a 2D quantile is not well-defined -- use a marginal or conditional)


# --- conditional flavours -------------------------------------------------------------------------------------------

class ConditionalDensity(_LSTDensityFunction):
    """Density of one reward conditional on another being held at a value (e.g. one bin's length given another's).

    The conditional transform is itself a *nested* inversion (an inner inversion along the conditioned axis, then the
    outer one), so a single de Hoog node costs an entire inner inversion and ``method='dehoog'`` is ~1e4x dearer here
    than for a marginal (293 s against a shared cosine grid on a 3-epoch bottleneck) while agreeing with it to 2e-4.
    """

    #: COS terms for the conditionals. Fewer than the marginals' 384: for a *nested* inversion each cosine frequency
    #: costs an entire inner inversion, so the fit is ~145x dearer and the count is re-tuned. 192 costs 1.5x less and
    #: shifts the CDF by <=1e-4 (against a 0.5-1.9% method error); 128 is too few -- it degrades sawtooth measurably.
    _cos_terms: int = 192
    _cos_terms_rough: int = 96




class ConditionalCDF(_LSTCumulativeDistributionFunction):
    """CDF of one reward conditional on another being held at a value (see :class:`ConditionalDensity` on why the
    per-point route is prohibitive for a nested transform)."""

    #: COS terms for the conditionals. Fewer than the marginals' 384: for a *nested* inversion each cosine frequency
    #: costs an entire inner inversion, so the fit is ~145x dearer and the count is re-tuned. 192 costs 1.5x less and
    #: shifts the CDF by <=1e-4 (against a 0.5-1.9% method error); 128 is too few -- it degrades sawtooth measurably.
    _cos_terms: int = 192
    _cos_terms_rough: int = 96




class ConditionalQuantileFunction(_LSTQuantileFunction):
    """Quantile function of one reward conditional on another being held at a value (see :class:`ConditionalDensity`
    on why the per-point route is prohibitive for a nested transform)."""

    #: COS terms for the conditionals. Fewer than the marginals' 384: for a *nested* inversion each cosine frequency
    #: costs an entire inner inversion, so the fit is ~145x dearer and the count is re-tuned. 192 costs 1.5x less and
    #: shifts the CDF by <=1e-4 (against a 0.5-1.9% method error); 128 is too few -- it degrades sawtooth measurably.
    _cos_terms: int = 192
    _cos_terms_rough: int = 96




class CallableDistributionFunctions:
    """
    Mixin exposing ``pdf`` / ``cdf`` / ``quantile`` as callable-and-plottable distribution-function properties. Each
    concrete distribution supplies the evaluators ``_pdf`` / ``_cdf`` / ``_quantile`` and the plotters ``_plot_pdf`` /
    ``_plot_cdf`` / ``_plot_quantile``; this mixin wires them together. Subclasses pick the *flavour* of the returned
    function objects by overriding :attr:`_pdf_function` / :attr:`_cdf_function` / :attr:`_quantile_function` (e.g. a
    spectrum returns the ``Marginal...`` flavours, a conditional the ``Conditional...`` flavours).
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

    def plot_cdf(self, *args, **kwargs) -> 'plt.Axes':
        """Deprecated: use :attr:`cdf`.plot() instead."""
        warnings.warn("plot_cdf() is deprecated; use .cdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self.cdf.plot(*args, **kwargs)

    def plot_pdf(self, *args, **kwargs) -> 'plt.Axes':
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

    def __init__(self) -> None:
        """
        Create object.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

    def touch(self, **kwargs: dict) -> None:
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
    :class:`CallableDistributionFunctions`); the evaluation lives on those function objects (subclasses select the
    flavour via :attr:`_cdf_function` / :attr:`_pdf_function` / :attr:`_quantile_function`). The generic grid
    :meth:`_plot_cdf` / :meth:`_plot_pdf` / :meth:`_plot_quantile` below back the direct-evaluation flavours (the
    empirical sample estimates) that do not bring their own plot.
    """

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
            y=np.array([self.quantile(float(p)) for p in q]),
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
            t = np.linspace(0, self.quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

        ax = Visualization.plot(
            ax=ax,
            x=t,
            y=self.cdf(t),
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
            dx = self.quantile(Settings.plot_endpoint_quantile) / 1e10

        if t is None:
            t = np.linspace(0, self.quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self.pdf(t, dx=dx),
            xlabel='t',
            ylabel='f(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

