"""
Distribution of an accumulated reward.

For a reward ``r`` over the states, the accumulated reward to absorption is ``R = int_0^tau_abs r(X_s) ds``
(e.g. tree height for the unit reward, total branch length for the lineage-count reward, an SFS bin for the
size-``i`` block-count reward). Unlike :meth:`MomentEvaluator.moment`, which returns only the moments of ``R``,
this gives the full distribution (CDF / PDF / quantiles) for an *arbitrary* reward and an *arbitrary*
piecewise time-homogeneous demography, via the Laplace-Stieltjes transform and its numerical inversion.

The transform tracks, in real time, the row vector ``a(t)_i = E[e^{-s R_t}; X_t = i, not absorbed]``. While in
state ``i`` the reward grows at rate ``r(i)``, so the weight ``e^{-s R_t}`` decays at rate ``s r(i)`` — i.e. the
reward enters as a *shift* of the generator:

    da/dt = a (T - s diag(r)).

Chaining the (real-time) epochs and augmenting with the absorbing state (reward rate 0 there, so absorbed mass
keeps its frozen weight) gives the accumulated-reward LST

    phi(s) = E[e^{-s R}] = c + a (s diag(r) - T_m)^{-1} (-T_m 1),
    [a, c] = [alpha, 0] prod_{finite epochs e} exp((Q_e - s diag(r)_aug) tau_e),

with ``Q_e`` the full (incl. absorbing) generator of epoch ``e`` and ``T_m`` the final unbounded epoch's
transient sub-generator. The CDF has Laplace transform ``phi(s) / s``; both are inverted with the de Hoog
quotient-difference method, which (unlike Talbot) is robust to a double-precision transform and (unlike the
Euler method) stays accurate on steep multi-epoch CDFs.

Zero-reward states need no special handling: ``s diag(r)`` simply has zeros there, so ``e^{-s R}`` does not decay
while the chain passes through them. An atom at ``R = 0`` (e.g. an SFS bin that may be empty) is recovered
automatically by the inversion.
"""
import logging
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import mpmath as mp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..rewards import Reward
from ..settings import Settings
from .base import CallableDistributionFunctions, DistributionFunction
from ._moments import MomentEvaluator

if TYPE_CHECKING:
    from .phase_type import PhaseTypeDistribution

logger = logging.getLogger('phasegen')


class RewardDistribution(CallableDistributionFunctions):
    """
    Full distribution of the accumulated reward ``R = int_0^tau_abs r(X_s) ds`` to absorption, via the
    Laplace-Stieltjes transform and numerical inversion. Handles arbitrary (non-negative) rewards, zero-reward
    states, and arbitrary piecewise time-homogeneous demographies (so it is multi-epoch-native).

    Its ``cdf`` / ``pdf`` / ``quantile`` are callable-and-plottable :class:`DistributionFunction`s (see
    :class:`CallableDistributionFunctions`); this is the 1D object returned by ``SFSDistribution.bin`` etc.
    """

    def __init__(self, dist: 'PhaseTypeDistribution', reward: Reward = None):
        """
        :param dist: The phase-type distribution providing the state space, demography and epoch machinery.
        :param reward: The reward whose accumulation defines ``R``. Defaults to ``dist``'s own reward.
        :raises NotImplementedError: if the reward is not a scalar (one value per state) reward.
        """
        self._host = dist
        self.state_space = dist.state_space
        self.demography = dist.demography
        self.reward = reward if reward is not None else dist.reward
        self._logger = logger.getChild(self.__class__.__name__)

    @cached_property
    def _setup(self):
        """Bind the reward vector to the host's (reward-independent, shared) per-epoch transient generators."""
        ss = self.state_space

        r_full = np.asarray(self.reward._get(ss))
        if r_full.ndim != 1:
            raise NotImplementedError(
                "RewardDistribution requires a scalar reward (one value per state); got a reward of shape "
                f"{r_full.shape}. For a spectrum, take the distribution of a single bin's reward."
            )

        # the transient states, initial vector and per-epoch generators do not depend on the reward, so they are
        # built once on the host and shared across all bins of a spectrum (see ``_reward_epoch_data``)
        data = self._host._reward_epoch_data
        r = r_full[data['idx']].astype(float)

        if np.any(r < 0):
            raise ValueError("RewardDistribution requires a non-negative reward.")

        return dict(r=r, **data)

    def lst(self, s: complex) -> complex:
        """The accumulated-reward Laplace-Stieltjes transform ``phi(s) = E[e^{-s R}]`` at (complex) ``s``."""
        st = self._setup
        return _lst_from_shift(s * st['r'], st['alpha'], st['T_epochs'], st['sparse'])

    def _invert(self, transform, t: float) -> float:
        """Numerical Laplace inversion (de Hoog) of ``transform`` evaluated at ``t``."""
        if t <= 0:
            return 0.0

        def F(s):
            val = transform(complex(s))
            return mp.mpc(val.real, val.imag)

        return float(mp.invertlaplace(F, t, method='dehoog'))

    def _cdf(self, t):
        """Cumulative distribution function ``P(R <= t)``. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self._cdf(float(x)) for x in np.asarray(t)])
        if t < 0:
            raise ValueError("Negative values are not allowed.")
        return self._invert(lambda s: self.lst(s) / s, float(t))  # L[CDF] = phi(s) / s

    def _pdf(self, t, **kwargs):
        """Probability density function. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self._pdf(float(x)) for x in np.asarray(t)])
        return self._invert(self.lst, float(t))  # L[pdf] = phi(s)

    def _quantile(self, q: float, precision: float = 1e-8, max_iter: int = 200) -> float:
        """The ``q``-quantile ``inf{x : F(x) >= q}`` via bisection on the (monotone) CDF."""
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1.")

        # bracket: grow the upper bound until its CDF exceeds q (seed from the reward's mean via the LST,
        # E[R] = -phi'(0), so we start near the right scale and only double a few times)
        h = 1e-3
        mean = (1.0 - self.lst(h).real) / h
        lo, hi = 0.0, max(mean, 1.0)
        for _ in range(max_iter):
            if self._cdf(hi) >= q:
                break
            hi *= 2
        else:
            raise RuntimeError("Failed to bracket the quantile.")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self._cdf(mid) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < precision:
                break

        return 0.5 * (lo + hi)

    def _plot_cdf(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = 200, show: bool = True,
                  file: str = None, clear: bool = True, label: str = None, title: str = 'CDF') -> 'plt.Axes':
        """Plot the CDF curve (fast COS inversion) up to the configured plot-endpoint quantile."""
        from ..visualization import Visualization
        if x is None:
            x = np.linspace(0, self._quantile(Settings.plot_endpoint_quantile), n_points)
        return Visualization.plot(ax=ax, x=x, y=self.cdf_curve(x), xlabel='x', ylabel='F(x)', label=label, file=file,
                                  show=show, clear=clear, title=title)

    def _plot_pdf(self, ax: 'plt.Axes' = None, x: np.ndarray = None, n_points: int = 200, show: bool = True,
                  file: str = None, clear: bool = True, label: str = None, title: str = 'PDF') -> 'plt.Axes':
        """Plot the PDF curve (derivative of the COS CDF) up to the configured plot-endpoint quantile."""
        from ..visualization import Visualization
        if x is None:
            x = np.linspace(0, self._quantile(Settings.plot_endpoint_quantile), n_points)
        return Visualization.plot(ax=ax, x=x, y=self.pdf_curve(x), xlabel='x', ylabel='f(x)', label=label, file=file,
                                  show=show, clear=clear, title=title)

    def _plot_quantile(self, ax: 'plt.Axes' = None, q: np.ndarray = None, n_points: int = 99, show: bool = True,
                       file: str = None, clear: bool = True, label: str = None,
                       title: str = 'Quantile function') -> 'plt.Axes':
        """Plot the quantile function (value versus probability), inverting the fast COS CDF curve."""
        from ..visualization import Visualization
        qe = Settings.plot_endpoint_quantile
        if q is None:
            q = np.linspace(1.0 - qe, qe, n_points)
        grid = np.linspace(0, self._range(), 512)
        return Visualization.plot(ax=ax, x=q, y=np.interp(q, self.cdf_curve(grid), grid), xlabel='q',
                                  ylabel='quantile', label=label, file=file, show=show, clear=clear, title=title)

    # ------------------------------------------------------------------------------------------------------------
    # fast curve evaluation (whole CDF/PDF curve from one fixed set of transform evaluations)
    # ------------------------------------------------------------------------------------------------------------
    #: Maximum number of COS terms the auto-refinement (see :meth:`_fit_cos`) grows to before giving up and warning.
    _cos_max_terms: int = 768

    @cached_property
    def _cos_coeffs(self) -> dict:
        """The cached COS coefficients (with auto-refined term count) for this distribution, computed once and shared
        across :meth:`cdf_curve` / :meth:`pdf_curve` / the plot endpoint (which all evaluate the same fit at
        different points). See :meth:`_fit_cos`."""
        return self._fit_cos(12.0)

    def _fit_cos(self, scale: float, n_terms: int = 192) -> dict:
        """
        Fit the COS (Fourier-cosine) inversion: evaluate the characteristic function ``chi(w) = phi(-i w)`` on a
        fixed frequency grid over ``[0, b]`` (``b = mean + scale*std``) and return the cosine coefficients. An atom
        at ``R = 0`` (``p0 = phi(inf)``) is split off so the series sees only the smooth continuous part.

        When the reconstruction rings (a wide window under-resolved by too few terms — common for skewed / heavy-
        tailed distributions) the term count is auto-refined (quadrupled, up to :attr:`_cos_max_terms`). The CDF's
        non-monotonicity (largest backward step) is the sensitive ringing detector; the density amplitude alone is
        weak. A *substantial* residual ripple at the cap, or a density still appreciable at the window edge (window
        too small for the tail), is warned about — pointing to the exact per-point ``cdf()`` / ``pdf()`` (de Hoog).
        """
        a, b = 0.0, self._range(scale)
        p0 = self.lst(1e8).real
        w = np.arange(n_terms) * np.pi / (b - a)
        chi = np.array([self.lst(-1j * wk) for wk in w])
        if p0 > 1e-9:
            chi = (chi - p0) / (1 - p0)  # continuous part only
        fk = (2.0 / (b - a)) * np.real(chi)  # a = 0, so exp(-i w a) = 1
        fk[0] *= 0.5

        xd = np.linspace(0.0, b - a, max(256, 2 * n_terms))
        fd = fk @ np.cos(np.outer(w, xd))                                   # continuous density
        Fd = fk[0] * xd + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xd))    # its analytic integral (continuous CDF)
        peak = max(float(fd.max()), 1e-12)
        dip = -float(np.diff(Fd).min())                                     # largest backward step of the CDF (>= 0)

        # under-resolved and still resolvable -> quadruple the terms and refit (the monotonicity clamp in _cos keeps
        # the residual harmless, so the cap can be modest)
        if (dip > 1e-3 or float(fd.min()) < -0.005 * peak) and n_terms < self._cos_max_terms:
            return self._fit_cos(scale, min(4 * n_terms, self._cos_max_terms))

        if dip > 1e-2:
            warnings.warn(
                f"COS inversion still rings substantially at {n_terms} terms (a sharp feature or atom the cosine "
                f"series cannot resolve); the plotted curve may be imprecise. Prefer the per-point cdf()/pdf() "
                f"(de Hoog).", stacklevel=4
            )
        elif float(fd[-1]) > 0.05 * peak:
            warnings.warn(
                "COS inversion looks imprecise: the density is still appreciable at the support window edge, so the "
                "window is too small to capture the tail. Prefer the per-point cdf()/pdf() (de Hoog).", stacklevel=4
            )

        return dict(b=b, w=w, fk=fk, p0=p0)

    def _cos(self, x: np.ndarray, kind: str, n_terms: int = 192, scale: float = 12.0) -> np.ndarray:
        """
        Evaluate the (cached, auto-refined) COS fit as a whole CDF/PDF curve over the grid ``x`` (for plotting; the
        exact per-point ``cdf()`` / ``pdf()`` use de Hoog). The CDF is clipped to ``[0, 1]`` and made monotone (it is
        monotone by definition, so clamp the small residual ripple — this also makes :meth:`pdf_curve`, its numerical
        derivative, non-negative).
        """
        fit = self._cos_coeffs if scale == 12.0 and n_terms == 192 else self._fit_cos(scale, n_terms)
        b, w, fk, p0 = fit['b'], fit['w'], fit['fk'], fit['p0']

        xa = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), 0.0, b)
        if kind == 'pdf':
            curve = fk @ np.cos(np.outer(w, xa))
            return (1 - p0) * curve if p0 > 1e-9 else curve

        cdf_c = fk[0] * xa + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xa))
        cdf = np.clip(p0 + (1 - p0) * cdf_c if p0 > 1e-9 else cdf_c, 0.0, 1.0)
        order = np.argsort(xa)
        cdf[order] = np.maximum.accumulate(cdf[order])
        return cdf

    def _cumulants(self) -> tuple:
        """Mean and variance of the accumulated reward from the LST near 0 (``phi(0) = 1``): ``c1 = -phi'(0)``,
        ``c2 = phi''(0) - phi'(0)^2``. Cheap (three transform evaluations); used to set the COS / plot range."""
        h = 1e-4
        d1 = (self.lst(h).real - self.lst(-h).real) / (2 * h)
        d2 = (self.lst(h).real - 2.0 + self.lst(-h).real) / h ** 2
        return -d1, max(d2 - d1 ** 2, 1e-12)

    def _range(self, scale: float = 12.0) -> float:
        """An upper end for the support (``mean + scale * std``), for the COS interval and default plot grids."""
        c1, c2 = self._cumulants()
        return float(c1 + scale * np.sqrt(c2))

    def cdf_curve(self, x, n_terms: int = 192) -> np.ndarray:
        """Fast CDF over a whole grid ``x`` via COS inversion (for plotting; see :meth:`_cos`)."""
        return self._cos(x, 'cdf', n_terms=n_terms)

    def pdf_curve(self, x, n_terms: int = 192) -> np.ndarray:
        """
        Fast PDF over a whole grid ``x`` (for plotting). Computed as the numerical derivative of the COS *CDF* (which
        is refined and clamped monotone, so it differentiates to a clean non-negative density) rather than the raw
        cosine density sum, which rings for skewed / heavy-tailed distributions. The differentiation uses a fine
        internal grid and interpolates to ``x``, so the result is independent of the (possibly coarse) plotting grid.
        Use the per-point :meth:`pdf` (de Hoog) for exact values.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.size < 2:
            return self._cos(x, 'pdf', n_terms=n_terms)
        # differentiate on a fine grid from 0 so every requested point is interior (central differences), then
        # interpolate -- independent of the (possibly coarse) plotting grid and accurate near the origin / an atom
        fine = np.linspace(0.0, float(x.max()), max(1024, 4 * x.size))
        return np.interp(x, fine, np.gradient(self._cos(fine, 'cdf', n_terms=n_terms), fine))


def _build_epoch_data(host) -> dict:
    """
    The reward-independent ingredients of the accumulated-reward transform: the transient states, the initial
    vector and the per-epoch transient sub-generators. Shared across all bins of a spectrum (the generators depend
    only on the state space and demography, not on which reward is accumulated), so it is built once on the host.
    """
    ss = host.state_space
    idx = np.where(~ss.absorbing)[0]
    alpha = np.asarray(ss.alpha)[idx].astype(float)
    nt = len(idx)
    sparse = nt >= Settings.closed_form_sparse_min_states

    T_epochs = []
    for epoch in host._get_epochs_until_unbounded():
        ss.update_epoch(epoch)
        host._check_numerical_stability(ss.S, 0)
        T_epochs.append((host._transient_block(idx, sparse=sparse), epoch.start_time, epoch.end_time))

    return dict(idx=idx, alpha=alpha, nt=nt, sparse=sparse, T_epochs=T_epochs)


def _exit_rates(T) -> np.ndarray:
    """Per-state rate of (direct) absorption = row deficit of the transient sub-generator, ``-T 1``."""
    return -np.asarray(T @ np.ones(T.shape[0])).ravel()


def _lst_from_shift(shift: np.ndarray, alpha: np.ndarray, T_epochs, sparse: bool) -> complex:
    """
    Accumulated-reward LST evaluated with an arbitrary diagonal *shift vector* ``shift`` (the reward enters the
    generator only as ``-diag(shift)``). For one reward ``shift = s diag(r)``; for two rewards (the joint
    transform) ``shift = s_a r_a + s_b r_b`` — so the univariate and joint cases share this one routine.

    ``E[e^{-<shift-as-accumulated>}] = c + a (diag(shift) - T_m)^{-1} (-T_m 1)``, with ``[a, c]`` the transient /
    absorbed mass pushed through the finite epochs (augmented with the absorbing state, reward 0, so absorbed mass
    keeps its frozen weight).
    """
    nt = len(alpha)
    vec = np.concatenate([alpha, [0.0]]).astype(complex)

    for T, t0, t1 in T_epochs[:-1]:
        exit_col = _exit_rates(T)
        tau = t1 - t0
        if sparse:
            Q = sp.bmat([
                [sp.csc_matrix(T) - sp.diags(shift), sp.csc_matrix(exit_col.reshape(-1, 1))],
                [None, sp.csc_matrix((1, 1))],
            ], format='csc')
            vec = spla.expm_multiply((Q * tau).T.tocsc(), vec)  # vec @ exp(Q tau)
        else:
            Q = np.zeros((nt + 1, nt + 1), dtype=complex)
            Q[:nt, :nt] = np.asarray(T) - np.diag(shift)
            Q[:nt, nt] = exit_col
            vec = vec @ sla.expm(Q * tau)

    a, c = vec[:nt], vec[nt]
    Tm = T_epochs[-1][0]
    A = (sp.diags(shift) if sparse else np.diag(shift)) - Tm
    solve = MomentEvaluator._lu_solver(A, sparse)
    return complex(c + a @ solve(_exit_rates(Tm)))


class JointRewardDistribution:
    """
    Joint distribution of two accumulated rewards ``R_a = int r_a(X_s) ds`` and ``R_b = int r_b(X_s) ds`` to
    absorption, the distributional object behind a cross-moment ``E[R_a R_b]`` (the within-tree 2-SFS / the
    two-locus SFS). The joint Laplace-Stieltjes transform uses the *combined* generator shift,

        Phi(s_a, s_b) = E[e^{-s_a R_a - s_b R_b}] = _lst_from_shift(s_a r_a + s_b r_b, ...),

    so it is the same machinery as the univariate :class:`RewardDistribution` with a two-parameter shift, and is
    multi-epoch-native. Setting one argument to zero recovers a marginal; mixed derivatives at the origin recover
    the cross-moments. The joint CDF/PDF (2D inversion) and the product distribution are views built on top.
    """

    def __init__(self, dist: 'PhaseTypeDistribution', reward_a: Reward, reward_b: Reward):
        """
        :param dist: The phase-type distribution providing the state space, demography and epoch machinery.
        :param reward_a: The first reward.
        :param reward_b: The second reward.
        """
        self._host = dist
        self.reward_a = reward_a
        self.reward_b = reward_b
        self._logger = logger.getChild(self.__class__.__name__)

    @cached_property
    def _setup(self):
        """Bind both reward vectors to the host's shared (reward-independent) per-epoch generators."""
        ss = self._host.state_space
        data = self._host._reward_epoch_data
        out = dict(**data)
        for name, reward in (('ra', self.reward_a), ('rb', self.reward_b)):
            r_full = np.asarray(reward._get(ss))
            if r_full.ndim != 1:
                raise NotImplementedError("JointRewardDistribution requires scalar (per-state) rewards.")
            r = r_full[data['idx']].astype(float)
            if np.any(r < 0):
                raise ValueError("JointRewardDistribution requires non-negative rewards.")
            out[name] = r
        return out

    def lst(self, s_a: complex, s_b: complex) -> complex:
        """The joint LST ``Phi(s_a, s_b) = E[e^{-s_a R_a - s_b R_b}]`` via the combined generator shift."""
        st = self._setup
        return _lst_from_shift(s_a * st['ra'] + s_b * st['rb'], st['alpha'], st['T_epochs'], st['sparse'])

    def marginal(self, which: str = 'a') -> RewardDistribution:
        """The marginal accumulated-reward distribution of ``R_a`` (``which='a'``) or ``R_b`` (``which='b'``)."""
        return RewardDistribution(self._host, self.reward_a if which == 'a' else self.reward_b)

    @cached_property
    def _is_diagonal(self) -> bool:
        """Whether the two rewards are identical, so ``R_a = R_b`` almost surely. The joint law then lives on the
        diagonal -- a singular measure the smooth 2D representation cannot resolve -- but the joint CDF reduces in
        closed form to the marginal at ``min(x, y)`` (see :meth:`cdf`)."""
        st = self._setup
        return np.array_equal(st['ra'], st['rb'])

    def moment(self, order_a: int = 1, order_b: int = 1, center: bool = False) -> float:
        """
        The cross-moment ``E[R_a^{order_a} R_b^{order_b}]`` (uncentered by default), via the exact moment engine —
        i.e. ``Phi``'s mixed derivative ``(-1)^{a+b} d^{a+b}/ds_a^a ds_b^b`` at the origin, but computed exactly.

        :param order_a: Power of ``R_a``.
        :param order_b: Power of ``R_b``.
        :param center: Whether to center around the means.
        :return: The cross-moment.
        """
        rewards = (self.reward_a,) * order_a + (self.reward_b,) * order_b
        return float(MomentEvaluator.moment(
            self._host, k=order_a + order_b, rewards=rewards, center=center, permute=True
        ))

    def cov(self) -> float:
        """The covariance ``E[R_a R_b] - E[R_a] E[R_b]``."""
        return self.moment(1, 1, center=False) - self.moment(1, 0) * self.moment(0, 1)

    def corr(self) -> float:
        """The Pearson correlation between ``R_a`` and ``R_b``."""
        va = self.marginal('a')._cumulants()[1]
        vb = self.marginal('b')._cumulants()[1]
        return self.cov() / np.sqrt(va * vb)

    # ------------------------------------------------------------------------------------------------------------
    # joint distribution (2D) and the product distribution
    # ------------------------------------------------------------------------------------------------------------
    @cached_property
    def _atoms(self) -> dict:
        """Atom probabilities from the ``s -> inf`` limits of ``Phi``: ``P(R_a = 0)``, ``P(R_b = 0)``,
        ``P(R_a = 0, R_b = 0)`` (an SFS bin is empty with positive probability)."""
        big = 1e8
        return dict(a0=self.lst(big, 0.0).real, b0=self.lst(0.0, big).real, both0=self.lst(big, big).real)

    @cached_property
    def _cos2d(self) -> dict:
        """
        The *continuous-continuous* joint density (both rewards ``> 0``) as a 2D Fourier-cosine (COS) expansion on
        ``[0, b_a] x [0, b_b]``. The marginal atoms are removed by inclusion-exclusion so the cosine series only
        sees the smooth part: ``cf_cc(w_a, w_b) = Phi(-i w_a, -i w_b) - Phi(-i w_a, inf) - Phi(inf, -i w_b) +
        P(both = 0)``. Returns the coefficient matrix and the (zero-based) ranges/frequencies.
        """
        n_terms, scale, big = 64, 10.0, 1e8
        p00 = self._atoms['both0']
        ca, va = self.marginal('a')._cumulants()
        cb, vb = self.marginal('b')._cumulants()
        ba, bb = ca + scale * np.sqrt(va), cb + scale * np.sqrt(vb)
        ua = np.arange(n_terms) * np.pi / ba
        ub = np.arange(n_terms) * np.pi / bb

        phi_a_inf = np.array([self.lst(-1j * w, big) for w in ua])    # Phi(-i w_a, inf), reused across w_b
        phi_inf_b_p = np.array([self.lst(big, -1j * w) for w in ub])  # Phi(inf, -i w_b)
        phi_inf_b_m = np.array([self.lst(big, 1j * w) for w in ub])   # Phi(inf, +i w_b)

        A = np.zeros((n_terms, n_terms))
        for i, wa in enumerate(ua):
            pp = np.array([self.lst(-1j * wa, -1j * wb) for wb in ub]) - phi_a_inf[i] - phi_inf_b_p + p00
            pm = np.array([self.lst(-1j * wa, 1j * wb) for wb in ub]) - phi_a_inf[i] - phi_inf_b_m + p00
            A[i] = (2.0 / ba) * (2.0 / bb) * 0.5 * np.real(pp + pm)  # lower limits are 0, so exp(-i w a) = 1
        A[0, :] *= 0.5
        A[:, 0] *= 0.5
        return dict(ba=ba, bb=bb, ua=ua, ub=ub, A=A)

    def _density(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The continuous joint density on the outer grid ``xs x ys`` (shape ``(len(xs), len(ys))``), as the raw
        cosine sum (may be slightly negative from Gibbs near edges; callers clip only for display, never for
        integration, which would bias the mass/moments)."""
        st = self._cos2d
        cx = np.cos(np.outer(st['ua'], xs))
        cy = np.cos(np.outer(st['ub'], ys))
        return cx.T @ st['A'] @ cy

    def _pdf(self, x, y):
        """
        Joint probability density of ``(R_a, R_b)`` (the continuous, both-positive part). The distribution also has
        atom mass on the axes where a reward is zero (see :attr:`_atoms`); a non-empty SFS bin pair has none there.

        :param x: ``R_a`` value(s).
        :param y: ``R_b`` value(s).
        :return: Density, scalar or a ``(len(x), len(y))`` grid.
        """
        if self._is_diagonal:
            raise NotImplementedError("The joint density is singular when both rewards are identical (R_a = R_b "
                                      "almost surely): the law lives on the diagonal and has no 2D density. Use "
                                      "cdf(x, y) = marginal CDF at min(x, y), or the 1D marginal density.")
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        f = np.clip(self._density(xs, ys), 0.0, None)  # clip for display only
        return float(f.ravel()[0]) if f.size == 1 else f

    @staticmethod
    def _cos_antideriv(u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``int_0^x cos(u_k t) dt`` for each frequency ``u_k`` and point ``x`` (``sin(u_k x)/u_k``, and ``x`` for
        ``u_0 = 0``); returns shape ``(len(x), len(u))``. Used to integrate the cosine density in closed form."""
        x = np.atleast_1d(x).astype(float)
        safe = np.where(u == 0, 1.0, u)
        out = np.sin(np.outer(x, u)) / safe
        out[:, u == 0] = x[:, None]
        return out

    def _cc_box(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The continuous-continuous box probabilities ``int_0^x int_0^y f_cc`` on the grid ``xs x ys``, evaluated
        *analytically* from the cosine coefficients (exact, unlike a grid quadrature of the oscillatory density)."""
        st = self._cos2d
        Ix = self._cos_antideriv(st['ua'], np.minimum(xs, st['ba']))   # (len_x, N)
        Iy = self._cos_antideriv(st['ub'], np.minimum(ys, st['bb']))   # (len_y, N)
        return Ix @ st['A'] @ Iy.T                                     # (len_x, len_y)

    def _cdf_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Joint CDF on the grid ``xs x ys``: the axis atoms (marginal sub-transform inversions, one per grid line)
        plus the analytic continuous box integral."""
        big = 1e8
        # P(R_a=0, R_b<=y) and P(R_b=0, R_a<=x) from inverting the marginal sub-transforms; at the atom edge (t=0)
        # the de Hoog inversion is unreliable, so use the exact limit P(R_a=0, R_b<=0) = P(R_b=0, R_a<=0) = P(both=0)
        both0 = self._atoms['both0']
        g_a = np.array([both0 if y == 0 else self.marginal('b')._invert(lambda s: self.lst(big, s) / s, float(y)) for y in ys])
        g_b = np.array([both0 if x == 0 else self.marginal('a')._invert(lambda s: self.lst(s, big) / s, float(x)) for x in xs])
        cc = self._cc_box(np.asarray(xs, float), np.asarray(ys, float))
        return g_b[:, None] + g_a[None, :] - both0 + cc

    def _cdf(self, x, y):
        """
        Joint CDF ``P(R_a <= x, R_b <= y)``: the axis atoms ``P(R_a = 0, R_b <= y)`` and ``P(0 < R_a <= x,
        R_b = 0)`` (from inverting the marginal sub-transforms ``Phi(inf, .)`` / ``Phi(., inf)``) plus the
        continuous box integral ``P(0 < R_a <= x, 0 < R_b <= y)`` (the cosine density integrated in closed form).

        When both rewards are identical (``R_a = R_b`` almost surely, e.g. a bin paired with itself) the joint law
        is singular on the diagonal; the CDF then reduces exactly to ``P(R <= min(x, y))`` of the shared marginal.
        """
        if self._is_diagonal:
            t = min(float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0]))
            # at t = 0 the marginal CDF is the atom P(R = 0) (the de Hoog inversion misses the jump there)
            return float(self._atoms['both0'] if t <= 0.0 else self.marginal('a').cdf(t))
        return float(self._cdf_grid(np.atleast_1d(x), np.atleast_1d(y))[0, 0])

    @property
    def pdf(self) -> DistributionFunction:
        """Joint density of ``(R_a, R_b)``: callable (``pdf(x, y)``) and plottable as a heatmap (``pdf.plot()``)."""
        return DistributionFunction(self._pdf, self._plot_pdf, 'pdf')

    @property
    def cdf(self) -> DistributionFunction:
        """Joint CDF of ``(R_a, R_b)``: callable (``cdf(x, y)``) and plottable as a heatmap (``cdf.plot()``)."""
        return DistributionFunction(self._cdf, self._plot_cdf, 'cdf')

    def _plot_pdf(self, ax: 'plt.Axes' = None, n_points: int = 120, show: bool = True, file: str = None,
                  title: str = 'Joint reward PDF') -> 'plt.Axes':
        """Heatmap of the joint (continuous) density of ``(R_a, R_b)``."""
        return self._plot_joint('pdf', ax, n_points, show, file, title)

    def _plot_cdf(self, ax: 'plt.Axes' = None, n_points: int = 60, show: bool = True, file: str = None,
                  title: str = 'Joint reward CDF') -> 'plt.Axes':
        """Heatmap of the joint CDF of ``(R_a, R_b)``."""
        return self._plot_joint('cdf', ax, n_points, show, file, title)

    def plot_pdf(self, *args, **kwargs) -> 'plt.Axes':
        """Deprecated: use ``.pdf.plot()`` instead."""
        warnings.warn("plot_pdf() is deprecated; use .pdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self._plot_pdf(*args, **kwargs)

    def plot_cdf(self, *args, **kwargs) -> 'plt.Axes':
        """Deprecated: use ``.cdf.plot()`` instead."""
        warnings.warn("plot_cdf() is deprecated; use .cdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self._plot_cdf(*args, **kwargs)

    def _plot_joint(self, kind, ax, n_points, show, file, title):
        import matplotlib.pyplot as plt

        st = self._cos2d
        xs = np.linspace(0, st['ba'], n_points)
        ys = np.linspace(0, st['bb'], n_points)
        Z = np.clip(self._density(xs, ys), 0.0, None) if kind == 'pdf' else self._cdf_grid(xs, ys)

        if ax is None:
            ax = plt.gca()
        mesh = ax.pcolormesh(xs, ys, Z.T, shading='auto', cmap='viridis')
        ax.figure.colorbar(mesh, ax=ax)
        ax.set_xlabel('$R_a$')
        ax.set_ylabel('$R_b$')
        ax.set_title(title)
        if file is not None:
            plt.savefig(file)
        if show:
            plt.show()
        return ax

    def conditional(self, on: str = 'a', value: float = 0.0) -> 'ConditionalRewardDistribution':
        """
        The 1D conditional distribution of the *other* reward given ``R_{on} = value``.

        For ``value > 0`` this is the continuous conditional density -- a slice of the 2D cosine density at the
        conditioning value, normalized by the marginal density there -- plus the atom ``P(R_other = 0 | R_{on} =
        value)`` (non-zero only when the other bin can be empty). For ``value = 0`` it conditions on the atom event
        ``{R_{on} = 0}`` (which must have positive probability). The returned object is a callable-and-plottable 1D
        distribution like any other (``cdf`` / ``pdf`` / ``quantile``).

        :param on: Which reward to condition on, ``'a'`` or ``'b'``.
        :param value: The conditioning value (``>= 0``).
        :return: The conditional distribution of the other reward.
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        if value is None or value < 0:
            raise ValueError("`value` must be non-negative.")
        if self._is_diagonal:
            raise NotImplementedError("The conditional of a self-pair is a point mass at `value` (R_a = R_b a.s.).")

        big = 1e8
        other_name = 'b' if on == 'a' else 'a'

        # condition on the atom {R_on = 0}: the sub-distribution of the other reward there, normalized by its mass
        if value == 0:
            atom = self._atoms['a0' if on == 'a' else 'b0']
            if atom < 1e-9:
                raise ValueError(f"Cannot condition on R_{on} = 0: it has (near) zero probability.")
            other = self.marginal(other_name)
            sub = (lambda s: self.lst(big, s)) if on == 'a' else (lambda s: self.lst(s, big))
            return ConditionalRewardDistribution(
                cdf=lambda y: other._invert(lambda s: sub(s) / s, float(y)) / atom,
                pdf=lambda y: other._invert(sub, float(y)) / atom,
                x_max=other._range(),
                atom0=self._atoms['both0'] / atom,
                label=f"R_{other_name} | R_{on} = 0"
            )

        # condition on R_on = value > 0: the conditional continuous density is a 1D cosine series (the slice of the
        # 2D cosine coefficients at ``value``) divided by the marginal density there; the leftover mass is the atom
        st = self._cos2d
        if on == 'a':
            marg, freqs, support = self.marginal('a'), st['ub'], st['bb']
            coeffs = np.cos(st['ua'] * value) @ st['A']
        else:
            marg, freqs, support = self.marginal('b'), st['ua'], st['ba']
            coeffs = st['A'] @ np.cos(st['ub'] * value)

        f_marg = marg.pdf(value)
        if f_marg <= 0:
            raise ValueError(f"The marginal density at R_{on} = {value} is zero; cannot condition there.")

        coeffs = coeffs / f_marg
        # continuous mass on (0, support]; the remainder is the atom P(R_other = 0 | R_on = value)
        mass = coeffs[0] * support + (coeffs[1:] / freqs[1:]) @ np.sin(freqs[1:] * support)
        atom0 = float(np.clip(1.0 - mass, 0.0, 1.0))

        def pdf(y: float) -> float:
            return float(coeffs @ np.cos(freqs * min(max(y, 0.0), support)))

        def cdf(y: float) -> float:
            yy = min(max(y, 0.0), support)
            return float(atom0 + coeffs[0] * yy + (coeffs[1:] / freqs[1:]) @ np.sin(freqs[1:] * yy))

        return ConditionalRewardDistribution(cdf, pdf, support, atom0, f"R_{other_name} | R_{on} = {value:g}")


class ConditionalRewardDistribution(CallableDistributionFunctions):
    """
    A 1D conditional distribution of one reward given a fixed value of the other (see
    :meth:`JointRewardDistribution.conditional`). It may carry an atom at 0 (the conditioned bin can still be
    empty). Its ``cdf`` / ``pdf`` / ``quantile`` are callable and plottable like any other distribution function.
    """

    def __init__(self, cdf, pdf, x_max: float, atom0: float = 0.0, label: str = ''):
        """
        :param cdf: Scalar CDF callback ``F(y)``.
        :param pdf: Scalar (continuous) density callback ``f(y)``.
        :param x_max: Upper end of the support (for plotting and quantile bracketing).
        :param atom0: Probability mass at ``0``.
        :param label: A human-readable label (e.g. ``"R_b | R_a = 1.2"``).
        """
        self._cdf_fn = cdf
        self._pdf_fn = pdf
        self._x_max = float(x_max)
        #: Probability mass at 0.
        self.atom0 = float(atom0)
        #: Human-readable label.
        self.label = label

    def _cdf(self, t):
        """Cumulative distribution function ``P(R <= t)``. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self._cdf(float(x)) for x in np.asarray(t)])
        if t < 0:
            raise ValueError("Negative values are not allowed.")
        return float(np.clip(self._cdf_fn(float(t)), 0.0, 1.0))

    def _pdf(self, t, **kwargs):
        """Probability density of the continuous part. Scalar or array-valued (the atom at 0 is in :attr:`atom0`)."""
        if np.ndim(t) > 0:
            return np.array([self._pdf(float(x)) for x in np.asarray(t)])
        return float(max(self._pdf_fn(float(t)), 0.0))

    def _quantile(self, q: float, precision: float = 1e-8, max_iter: int = 200) -> float:
        """The ``q``-quantile via bisection on the (monotone) CDF; values ``q <= atom0`` map to the atom at 0."""
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1.")
        if q <= self.atom0:
            return 0.0

        lo, hi = 0.0, self._x_max
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self._cdf(mid) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < precision:
                break
        return 0.5 * (lo + hi)

    def _plot_cdf(self, ax: 'plt.Axes' = None, t: np.ndarray = None, n_points: int = 200, show: bool = True,
                  file: str = None, clear: bool = True, label: str = None,
                  title: str = 'Conditional CDF') -> 'plt.Axes':
        """Plot the conditional CDF."""
        from ..visualization import Visualization
        if t is None:
            t = np.linspace(0, self._x_max, n_points)
        return Visualization.plot(ax=ax, x=t, y=self._cdf(t), xlabel='x', ylabel='F(x)', label=label or self.label,
                                  file=file, show=show, clear=clear, title=title)

    def _plot_pdf(self, ax: 'plt.Axes' = None, t: np.ndarray = None, n_points: int = 200, show: bool = True,
                  file: str = None, clear: bool = True, label: str = None,
                  title: str = 'Conditional PDF') -> 'plt.Axes':
        """Plot the conditional (continuous-part) PDF."""
        from ..visualization import Visualization
        if t is None:
            t = np.linspace(0, self._x_max, n_points)
        return Visualization.plot(ax=ax, x=t, y=self._pdf(t), xlabel='x', ylabel='f(x)', label=label or self.label,
                                  file=file, show=show, clear=clear, title=title)

    def _plot_quantile(self, ax: 'plt.Axes' = None, q: np.ndarray = None, n_points: int = 99, show: bool = True,
                       file: str = None, clear: bool = True, label: str = None,
                       title: str = 'Conditional quantile function') -> 'plt.Axes':
        """Plot the conditional quantile function (value versus probability ``q``)."""
        from ..visualization import Visualization
        if q is None:
            q = np.linspace(0.01, 0.99, n_points)
        return Visualization.plot(ax=ax, x=q, y=np.array([self._quantile(float(p)) for p in q]), xlabel='q',
                                  ylabel='quantile', label=label or self.label, file=file, show=show, clear=clear,
                                  title=title)
