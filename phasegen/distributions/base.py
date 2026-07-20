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
        Plot the distribution function (the distribution's ``_plot_<kind>``) -- the same function :meth:`__call__`
        evaluates, over a grid. Accepted arguments depend on the distribution; common ones are ``bins`` /
        ``configs`` (select which spectrum bins to draw), ``n_points`` (grid resolution), ``ax`` / ``show`` /
        ``file`` / ``title``.
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
      derivative of the cosine CDF. The tree height uses the exact matrix exponential, empirical samples a
      histogram.
    - **Plot** ``pdf.plot()``: the same function, over a grid.
    """
    kind = 'pdf'


class CumulativeDistributionFunction(DistributionFunction):
    """Cumulative distribution function -- the probability of being at most ``x``.

    - **Callable** ``cdf(x)``: the probability at ``x`` (scalar or array). For an accumulated reward this is the
      Fourier-cosine inversion. The tree height uses the exact matrix exponential, samples the empirical CDF.
    - **Plot** ``cdf.plot()``: the same function, over a grid.
    """
    kind = 'cdf'


class QuantileFunction(DistributionFunction):
    """Quantile function -- the inverse CDF.

    - **Callable** ``quantile(q)``: the value at which the CDF reaches ``q`` (scalar or array). For an accumulated
      reward this inverts the very CDF grid the :class:`CumulativeDistributionFunction` reads, so the two are exact
      mutual inverses. Empirical data uses the sample quantile.
    - **Plot** ``quantile.plot()``: the same function, over a grid of probabilities.
    """
    kind = 'quantile'


# --- the shared CDF representation ----------------------------------------------------------------------------------

class _HazardGrid:
    """
    The one representation the cdf, pdf and quantile of a continuous distribution are read off: a grid of nodes and
    the **cumulative hazard** ``H = -log(1 - F)`` on them, interpolated linearly in ``x``.

    The map is the whole definition: ``F(x) = 1 - exp(-H(x))``, so the cdf reads it forwards
    (:meth:`_interp_cdf`), the quantile backwards (:meth:`_interp_quantile`) and the pdf differentiates it
    (:meth:`_interp_pdf`). No root-find, no finite difference, and the three are exact mutual inverses of one
    another rather than agreeing to a tolerance.

    ``H`` is the coordinate because it is the one in which both halves of the curve are near-straight: near the
    origin ``H ~ F``, so a chord in ``H`` is the obvious linear interpolation of the CDF; out in the tail ``H`` is
    ``-log S``, which an (asymptotically exponential) survival traces almost exactly. Linear in ``H`` is a
    piecewise-constant *hazard*, the natural interpolant of a survival function.

    Where the nodes come from is the subclass's business, and the two sources differ because their point evaluators
    do: :class:`_LSTFunction` inverts the Laplace transform, which is dear enough (by some three orders of magnitude)
    that it fits a cosine series for the body and pays for exact nodes only in the tail, while the tree height's
    :class:`~phasegen.distributions.phase_type._ExpmFunction` exponentiates the rate matrix, cheap enough that every
    node is exact.
    """

    def _shared(self, key: str, build) -> 'Any':
        """Return a shared entry of the CDF representation, built once via ``build`` and cached on the distribution
        (so the cdf / pdf / quantile of one distribution reuse it). Honors :attr:`Settings.cache`."""
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
        """The cumulative hazard ``H = -log(1 - F)``, the coordinate the grid is interpolated in. Capped, so a CDF
        that has saturated at 1 (as the cosine fit does at the end of its window) does not take it to infinity."""
        return -np.log1p(-np.minimum(np.asarray(cdf, dtype=float), 1.0 - 1e-16))

    def _interp_cdf(self, t: np.ndarray, nodes: np.ndarray, hazard: np.ndarray) -> np.ndarray:
        """
        The CDF between the grid's nodes: ``F(x) = 1 - exp(-H(x))``, with the cumulative hazard ``H`` interpolated
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
        """The closed-form inverse of :meth:`_interp_cdf`'s map: the same relation between ``x`` and ``H``, read the
        other way. Levels at or below the atom ``P(R = 0)`` land on the first node, which is 0.

        :param q: Probability levels.
        :param nodes: The grid's nodes.
        :param hazard: The cumulative hazard on them.
        :return: The quantiles at ``q``.
        """
        return np.interp(self._hazard(q), hazard, nodes)

    def _interp_pdf(self, t: np.ndarray, nodes: np.ndarray, hazard: np.ndarray) -> np.ndarray:
        """The derivative of :meth:`_interp_cdf`'s map: ``f = dF/dx = S * dH/dx``, the survival times the hazard rate.
        Non-negative by construction, so it cannot inherit the raw cosine sum's Gibbs negativity.

        :param t: Points to evaluate at.
        :param nodes: The grid's nodes.
        :param hazard: The cumulative hazard on them.
        :return: The density at ``t``.
        """
        if len(nodes) < 2:
            # a degenerate grid (a near-total atom at 0, whose mass sits above the tail cut) leaves no interval to
            # differentiate the hazard over: np.gradient needs at least two nodes. The continuous density is
            # negligible there (the mass is in the atom), so it is zero.
            return np.zeros_like(np.asarray(t, dtype=float))

        # below the support (t < nodes[0] = 0) the density is 0; np.interp would otherwise clamp to the first node
        h = np.interp(t, nodes, hazard, left=0.0)

        return np.exp(-h) * np.interp(t, nodes, np.gradient(hazard, nodes), left=0.0)


# --- the accumulated-reward (LST / de Hoog) inversion machinery, owned by the function objects -----------------------

class _LSTFunction(_HazardGrid):
    """
    Mixin owning the 1D accumulated-reward inversion machinery for the function objects of an LST distribution
    (:class:`~phasegen.distributions.reward.RewardDistribution` and its conditional flavours; a bare
    :class:`~phasegen.distributions.PhaseTypeDistribution` such as ``total_branch_length``). It pulls the transform
    and scale *primitives* (``lst`` / ``_invert`` / ``_cumulants`` / ``_range`` / ``_time_scale`` / ``_titled`` /
    the inversion guards) from ``self._distribution`` and turns them into the cdf / pdf / quantile.

    One representation serves the cdf / pdf / quantile: the **CDF grid** of :meth:`_cdf_grid`, a two-pass
    Fourier-cosine fit carrying exact de Hoog nodes above :attr:`~phasegen.settings.Settings.dehoog_tail_quantile`,
    where the fit force-normalises to 1 and so loses the tail outright. A single fit answers a whole array, and the
    grid is cached on the *distribution* (the one object the cdf / pdf / quantile of a distribution hang off, see
    :meth:`CallableDistributionFunctions._function`), so all three read it and are mutually consistent by construction:
    the pdf is its derivative and the quantile its inverse interpolation, making ``cdf(quantile(q)) == q`` exact.

    The **per-point de Hoog inversion** (:meth:`_cdf_point` / :meth:`_pdf_point`) is exact but costs one Laplace
    inversion (~19 ms) per point, so it is never a route the caller selects -- there is no ``exact=`` switch, and every
    plotted curve is the very function the caller evaluates. It is memoised per distribution and used to build:

    - the grid's own **far-tail nodes**, materialised on first use and only as far as the query reaches, so a plot
      (whose endpoint quantile sits below the cut) never pays for them;
    - a **conditional's support window** (``_Conditional._range_via_cdf``), which brackets the exact CDF because the
      nested transform's finite-difference variance is unusable;
    - the **joint's near-origin wiggle check** (``JointRewardDistribution._cos2d_wiggle_check``), cached per joint;
    - the exactness pins of the test suite, which need a reference the grid cannot be its own judge of.
    """
    #: Cosine terms for the coarse support-locating pass and the fine accuracy pass of the two-pass COS fit.
    _cos_terms_rough: int = 128
    _cos_terms: int = 384

    #: Nodes the fit is sampled on to become the grid's body. The fit is analytic, so these cost only its evaluation
    #: (~6 ms) against the hundreds of ms of transform evaluations behind it, and the interpolation error between them
    #: falls as their spacing squared. At 2048 that error reached 9.4e-4 on a heavy-tailed bin, whose density spikes
    #: in a window stretched long by its tail -- *larger* than the fit's own ~3e-4, so the grid, not the fit, was the
    #: dominant error in the body. 8192 puts it at 5.9e-5, comfortably back under the fit's.
    _cos_n_grid: int = 8192

    #: Support scale (``mean + scale * std``) of the coarse pass, which bounds where the fine pass may put its window.
    #: At 12 the coarse window can fall *short* of :attr:`_cos_tail_target` for a heavy-tailed bin, and the fine window
    #: is then pinned to a support end that is too small however tight the target is (an n = 10 mid-frequency bin
    #: saturates at a 0.6% error in the mean and 4.4% in the second moment).
    _cos_rough_scale: float = 20.0

    #: CDF mass the fine pass's window must contain. Everything above it is discarded: the fit force-normalises to 1
    #: at the window end, so the target *is* the tail that the grid keeps. The window trades against near-origin
    #: resolution (``b / n_terms``), but only weakly, and the tail is by far the more expensive side to get wrong: at
    #: the old 0.9995 the cut cost 0.05-0.6% of the mean, 0.3-4.4% of the second moment, and put a systematic 2.5e-4
    #: error in the CDF itself, all of which this removes at no cost in terms or transform evaluations.
    _cos_tail_target: float = 1.0 - 1e-5

    #: Spacing of the exact (de Hoog) nodes, as a decrement of the cumulative hazard ``H = -log(1 - F)`` -- the
    #: coordinate the whole grid is interpolated in (see :meth:`_interp_cdf`). One spacing resolves body and tail
    #: alike because ``H`` is both: near the origin ``H ~ F``, so a step in ``H`` is a step in probability; near
    #: ``F -> 1`` it is ``-log S``, so a step is a fixed factor of survival. A ladder in ``F`` alone cannot resolve a
    #: survival of 1e-6, and one in ``log S`` alone takes enormous steps through the body, where ``S`` barely moves.
    _hazard_step: float = 0.25

    #: Probability spacing of the exact nodes, applied alongside :attr:`_hazard_step`. Redundant at the default cut
    #: (in the far tail ``H`` is the finer of the two), it is what resolves the body when the cut is set low.
    _cdf_step: float = 0.01

    #: Survival the exact nodes are carried down to (unless a query asks for more), and the node budget bounding them.
    _tail_target: float = 1.0 - 1e-6
    _max_exact_nodes: int = 512

    # ---- distribution primitives (thin accessors) --------------------------------------------------------------
    def _range(self, scale: float = 12.0) -> float:
        return self._distribution._range(scale)

    def _cdf_point(self, t: float) -> float:
        """Per-point de Hoog CDF ``P(R <= t)`` (``L[CDF] = phi(s) / s``) -- the exact reference the cosine grid is
        checked against, the nodes of its far-tail extension, the conditional support bracket and the joint wiggle
        check. Memoised per distribution: one inversion costs ~19 ms, so it is the *points* that are worth caching,
        not any grid assembled from them."""
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
        """Per-point de Hoog density (``L[pdf] = phi(s)``)."""
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
        """
        COS coefficients, fit in **two passes**: a coarse pass over a generous window
        (:attr:`_cos_rough_scale` standard deviations) locates the effective support, then the fit is redone over a
        window tightened to the support that holds :attr:`_cos_tail_target` of the mass. Matching the window to where
        the mass actually is -- rather than ``mean + scale*std``, which a heavy tail blows far past the bulk -- lets a
        few hundred cosine terms resolve the curve accurately, removing the ringing at the source.
        """
        rough = self._fit_cos(self._range(self._cos_rough_scale), self._cos_terms_rough)
        xs = np.linspace(0.0, rough['b'], 1024)
        cdf = np.maximum.accumulate(self._eval_cos_cdf(rough, xs))
        b = float(np.interp(self._cos_tail_target, cdf, xs))
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
        """A fine, monotone CDF on the fit's window ``[0, b]``: the body of the shared grid of :meth:`_cdf_grid`,
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
        The distance from the last exact node to the next: whichever of a step in the cumulative hazard and a step in
        the probability is the *finer* there, converted to a distance by the local density (``dx = dF / f``, and
        ``dx = dH * S / f`` since ``dH/dx = f / S``).

        Neither spacing suffices alone. A ladder in ``F`` cannot reach a survival of 1e-6 -- it would need a million
        steps -- while a ladder in ``H`` takes enormous strides through the body, where the survival barely moves; on
        a rapid decline the second put a 4.5e-3 error in the CDF. Taking the finer of the two makes one rule resolve
        the whole curve, so the cut is free to sit anywhere, including 0.

        The step is set by the exact values already in hand (and, for the first, by the fit's density), never by what
        was queried, so the nodes land in the same places however the caller arrives at them.

        :param nodes: The ``(x, F)`` nodes so far, ascending.
        :return: The step to the next node.
        """
        x, cdf = nodes[-1]
        survival = 1.0 - cdf

        if len(nodes) == 1:
            xs, cs = self._cos_cdf_grid
            density = float(np.interp(x, xs, np.gradient(cs, xs)))
        else:
            x_prev, cdf_prev = nodes[-2]
            density = (cdf - cdf_prev) / (x - x_prev) if x > x_prev else 0.0

        if density <= 0 or survival <= 0:
            return float(self._cos_coeffs['b'])

        return float(min(min(self._hazard_step * survival, self._cdf_step) / density, self._cos_coeffs['b']))

    def _exact_nodes(self, x_cut: float, cut: float, x_max: float, q_max: float) -> list:
        """
        The ``(x, F)`` nodes whose values come from the exact inversion, marching outward from the cut. Cached on the
        distribution and *extended* when a query reaches past their end -- never rebuilt, and never trimmed to the
        span that happens to be asked for. Every node ever computed stays in the grid, so an answer cannot change
        because a later call asked for something further out, and the ~19 ms an exact node costs is paid once.

        The march only *starts* when a query enters this half of the curve. A plot, whose endpoint quantile sits below
        the default cut, therefore leaves the ladder at its anchor and pays nothing.

        The nodes cannot be placed by the fit's own quantile, tempting as that is: the fit force-normalises to 1 at
        the end of its window, so its quantile saturates there and a node asked for at a far level lands where the
        *fit* believes that level is -- inside the window, at a point whose true CDF is far lower. The ladder then
        tops out below the level being asked for and the quantile runs off the end of it. Marching outward on the
        exact values instead, the nodes go wherever the distribution actually is, including past the fit's window.

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
        The :class:`_HazardGrid` of an LST distribution: one grid of nodes, carrying the cosine fit's values below
        :attr:`~phasegen.settings.Settings.dehoog_tail_quantile` and the exact de Hoog inversion's above it. The cut
        is a plain probability, so it is a knob over the whole range: at 1 the grid is entirely the (cheap,
        vectorised) fit, at 0 entirely the (exact, ~19 ms a node) inversion, and in between each node takes the value
        of whichever is trusted at its own level. Nothing else about the grid depends on it -- in particular not the
        interpolation rule, which is :meth:`~_HazardGrid._interp_cdf`'s hazard rule everywhere.

        The cosine fit has to be corrected above *some* level because it force-normalises to 1 at the end of its
        window, so beyond that an interpolation of it reports a survival of exactly zero: the CDF came back as exactly
        1.0 and the density as exactly 0 for a bin whose true survival there is 1e-3.

        The exact nodes are far too expensive to build eagerly, so they are materialised on first use and only as far
        as the query reaches. A plot, whose endpoint quantile sits below the default cut, never builds one.

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
    """The CDF of a 1D accumulated-reward distribution, on top of the shared :class:`_LSTFunction` machinery."""

    def __call__(self, t) -> 'np.ndarray | float':
        """
        CDF ``P(R <= t)``, for a scalar or an array of ``t``, interpolated on the shared CDF grid
        (:meth:`_LSTFunction._cdf_grid`), so a whole array costs one fit.

        :param t: Point(s) at which to evaluate the CDF.
        :return: The CDF at ``t``, of the same shape.
        """
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        out = self._interp_cdf(ta, *self._cdf_grid(x_max=float(ta.max(initial=0.0))))

        return out if np.ndim(t) > 0 else float(out[0])

    def plot(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None, **kwargs) -> 'plt.Axes':
        """Plot the CDF up to the configured plot-endpoint quantile. The curve is ``self(x)``, i.e. exactly the
        function the caller evaluates. Extra keyword arguments (``alpha``, ``lw``, ...) are forwarded to the line."""
        from ..visualization import Visualization
        d = self._distribution
        if x is None:
            x = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), n_points or Settings.plot_n_grid)
        y = self(x)
        ax = Visualization.plot(ax=ax, x=x, y=y, xlabel='x', ylabel='F(x)', label=label, file=file,
                                show=show, clear=clear, title=title or d._titled('CDF'), **kwargs)
        ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]
        return ax


class _LSTDensityFunction(_LSTFunction, DensityFunction):
    """The density of a 1D accumulated-reward distribution."""

    def __call__(self, t, **kwargs) -> 'np.ndarray | float':
        """
        Density, for a scalar or an array of ``t``, by differentiating the shared CDF grid
        (:meth:`_LSTFunction._cdf_grid`) -- which keeps it consistent with the CDF, free of the raw cosine sum's Gibbs
        negativity, and non-zero in the far tail, where the cosine window alone ends and its derivative is flat zero.

        :param t: Point(s) at which to evaluate the density.
        :return: The density at ``t``, of the same shape.
        """
        d = self._distribution
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        out = self._interp_pdf(ta, *self._cdf_grid(x_max=float(ta.max(initial=0.0))))
        out = d._warn_if_negative(out, d._titled('density'))
        return out if np.ndim(t) > 0 else float(out[0])

    def plot(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None, **kwargs) -> 'plt.Axes':
        """Plot the PDF up to the configured plot-endpoint quantile (the derivative of the cosine CDF grid). The curve
        is ``self(x)``, i.e. exactly the function the caller evaluates. Extra keyword arguments (``alpha``, ``lw``,
        ...) are forwarded to the line."""
        from ..visualization import Visualization
        d = self._distribution
        if x is None:
            x = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), n_points or Settings.plot_n_grid)
        y = self(x)
        return Visualization.plot(ax=ax, x=x, y=y, xlabel='x', ylabel='f(x)', label=label, file=file,
                                  show=show, clear=clear, title=title or d._titled('PDF'), **kwargs)


class _LSTQuantileFunction(_LSTFunction, QuantileFunction):
    """The quantile function of a 1D accumulated-reward distribution: inverse interpolation of the shared CDF grid."""

    def __call__(self, q) -> 'np.ndarray | float':
        """
        The ``q``-quantile ``inf{x : F(x) >= q}``, for a scalar or an array of ``q``.

        The shared CDF grid (:meth:`_LSTFunction._cdf_grid`) is monotone, so the quantile is its inverse
        *interpolation* -- a whole array in one vectorised pass. There is no Laplace inversion that returns a quantile
        directly (the transform gives ``F``, so a quantile is always a root of it), but reading the same piecewise
        linear ``F`` the CDF reads makes the two exact mutual inverses, ``cdf(quantile(q)) == q``. At or below the
        atom mass ``P(R = 0)`` the quantile is exactly 0.

        Just *above* a large atom the quantile is accurate in absolute terms but loses relative precision, because it
        is itself near zero there: for a bin empty with probability 0.44, ``q = 0.5`` lands at 0.041 against the exact
        0.036 (16% relative, but 0.006 absolute against a 0.95-quantile of 13.3). This is the cosine series' Gibbs
        artifact at the jump, and it decays away from the atom (4% at ``q = 0.6``, 0.04% at ``q = 0.95``).

        :param q: Probability level(s) in ``[0, 1]``.
        :return: The quantile(s), of the same shape as ``q``.
        :raises ValueError: If any ``q`` lies outside ``[0, 1]``.
        """
        qa = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((qa < 0) | (qa > 1)):
            raise ValueError("Quantile must be between 0 and 1.")

        out = self._interp_quantile(qa, *self._cdf_grid(q_max=float(qa.max(initial=0.0))))

        return out if np.ndim(q) > 0 else float(out[0])

    def plot(self, ax: 'plt.Axes' = None, q: np.ndarray = None, n_points: int = None, show: bool = True,
             file: str = None, clear: bool = True, label: str = None, title: str = None, **kwargs) -> 'plt.Axes':
        """Plot the quantile function (value versus probability). The curve is ``self(q)``, i.e. exactly the function
        the caller evaluates. Extra keyword arguments (``alpha``, ``lw``, ...) are forwarded to the line."""
        from ..visualization import Visualization
        d = self._distribution
        qe = Settings.plot_endpoint_quantile
        if q is None:
            q = np.linspace(1.0 - qe, qe, n_points or Settings.plot_n_grid)
        y = self(q)
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
             clear: bool = True, label: str = None, title: str = 'PDF') -> 'plt.Axes':
        from ..visualization import Visualization
        d = self._distribution
        if t is None:
            t = np.linspace(0, d.quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)
        return Visualization.plot(ax=ax, x=t, y=self(t), xlabel='t', ylabel='f(t)', label=label, file=file,
                                  show=show, clear=clear, title=title)


class _GridQuantileFunction(QuantileFunction):
    """Quantile function whose distribution computes it directly (see :class:`_GridCumulativeDistributionFunction`)."""

    def plot(self, ax: 'plt.Axes' = None, q: np.ndarray = None, show: bool = True, file: str = None,
             clear: bool = True, label: str = None, title: str = 'Quantile function') -> 'plt.Axes':
        from ..visualization import Visualization
        if q is None:
            q = np.linspace(1.0 - Settings.plot_endpoint_quantile, Settings.plot_endpoint_quantile,
                            Settings.plot_n_grid)
        return Visualization.plot(ax=ax, x=q, y=self(q), xlabel='q', ylabel='quantile',
                                  label=label, file=file, show=show, clear=clear, title=title)


# --- marginal (per-bin spectrum) flavours ---------------------------------------------------------------------------

class MarginalDensity(DensityFunction):
    """Per-bin marginal densities of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``pdf(x)``: every bin's ``pdf(x)``, the derivative of that bin's cosine CDF grid.
    - **Plot** ``pdf.plot()``: overlays those same curves, one per bin.
    """


class MarginalCDF(CumulativeDistributionFunction):
    """Per-bin marginal CDFs of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``cdf(x)``: every bin's ``cdf(x)``, read off that bin's cosine CDF grid.
    - **Plot** ``cdf.plot()``: overlays those same curves, one per bin.
    """


class MarginalQuantileFunction(QuantileFunction):
    """Per-bin marginal quantile functions of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``quantile(q)``: every bin's ``quantile(q)`` -- the inverse interpolation of that bin's cosine
      CDF grid.
    - **Plot** ``quantile.plot()``: overlays those same curves, one per bin.
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
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        f = self._grid_values(xs, ys)
        return float(f.ravel()[0]) if f.size == 1 else f

    def _grid_values(self, xs, ys) -> 'np.ndarray':
        d = self._distribution
        # the guard lives here, not in __call__, so plot() / plot_surface() (which reach _grid_values directly) also
        # refuse a diagonal-singular law rather than drawing a 2D surface for a distribution that has no 2D density
        if d._is_diagonal:
            raise NotImplementedError("The joint density is singular when both rewards are identical (R_a = R_b "
                                      "almost surely): the law lives on the diagonal and has no 2D density. Use "
                                      "cdf(x, y) = marginal CDF at min(x, y), or the 1D marginal density.")
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
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        G = self._grid_values(xs, ys)
        return float(G.ravel()[0]) if G.size == 1 else G

    def _grid_values(self, xs, ys) -> 'np.ndarray':
        # the diagonal reduction lives here, not in __call__, so plot() / plot_surface() (which reach _grid_values
        # directly) draw the same singular-on-the-diagonal CDF the callable returns rather than the 2D cosine box
        # expansion of a measure that has no 2D density
        d = self._distribution
        if d._is_diagonal:
            m = d.marginal('a')
            # at t = 0 the marginal CDF is the atom P(R = 0) (the de Hoog inversion misses the jump there)
            return np.array([[float(d._atoms['both0'] if min(xx, yy) <= 0.0 else m.cdf(min(xx, yy)))
                              for yy in ys] for xx in xs])
        return d._cdf_grid(xs, ys)

    def _default_n_points(self, surface) -> int:
        return 60


# (a bivariate joint has no quantile flavour: a 2D quantile is not well-defined -- use a marginal or conditional)


# --- conditional flavours -------------------------------------------------------------------------------------------

class ConditionalDensity(_LSTDensityFunction):
    """Density of one reward conditional on another being held at a value (e.g. one bin's length given another's).

    The conditional transform is itself a *nested* inversion (an inner inversion along the conditioned axis, then the
    outer one), so a single de Hoog node costs an entire inner inversion and the per-point route is ~1e4x dearer here
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
    Abstract base class for probability distributions.
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
        :param q: Probabilities to evaluate the quantile at. Defaults to
            :attr:`~phasegen.settings.Settings.plot_n_grid` evenly spaced values from
            ``1 - Settings.plot_endpoint_quantile`` to ``Settings.plot_endpoint_quantile``.
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
        :param t: Values to evaluate the CDF at. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
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
    ) -> 'plt.Axes':
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param t: Values to evaluate the density function at.
            Defaults to a grid over :attr:`~phasegen.settings.Settings.plot_n_grid`
            points up to :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
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

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self.pdf(t),
            xlabel='t',
            ylabel='f(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

