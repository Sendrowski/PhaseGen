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
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import mpmath as mp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..rewards import Reward
from ..settings import Settings
from ._moments import MomentEvaluator

if TYPE_CHECKING:
    from .phase_type import PhaseTypeDistribution

logger = logging.getLogger('phasegen')


class RewardDistribution:
    """
    Full distribution of the accumulated reward ``R = int_0^tau_abs r(X_s) ds`` to absorption, via the
    Laplace-Stieltjes transform and numerical inversion. Handles arbitrary (non-negative) rewards, zero-reward
    states, and arbitrary piecewise time-homogeneous demographies (so it is multi-epoch-native).
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

    def cdf(self, t):
        """Cumulative distribution function ``P(R <= t)``. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self.cdf(float(x)) for x in np.asarray(t)])
        if t < 0:
            raise ValueError("Negative values are not allowed.")
        return self._invert(lambda s: self.lst(s) / s, float(t))  # L[CDF] = phi(s) / s

    def pdf(self, t):
        """Probability density function. Scalar or array-valued."""
        if np.ndim(t) > 0:
            return np.array([self.pdf(float(x)) for x in np.asarray(t)])
        return self._invert(self.lst, float(t))  # L[pdf] = phi(s)

    def quantile(self, q: float, precision: float = 1e-8, max_iter: int = 200) -> float:
        """The ``q``-quantile ``inf{x : F(x) >= q}`` via bisection on the (monotone) CDF."""
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1.")

        # bracket: grow the upper bound until its CDF exceeds q (seed from the reward's mean via the LST,
        # E[R] = -phi'(0), so we start near the right scale and only double a few times)
        h = 1e-3
        mean = (1.0 - self.lst(h).real) / h
        lo, hi = 0.0, max(mean, 1.0)
        for _ in range(max_iter):
            if self.cdf(hi) >= q:
                break
            hi *= 2
        else:
            raise RuntimeError("Failed to bracket the quantile.")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) < q:
                lo = mid
            else:
                hi = mid
            if hi - lo < precision:
                break

        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------------------------------------------------
    # fast curve evaluation (whole CDF/PDF curve from one fixed set of transform evaluations)
    # ------------------------------------------------------------------------------------------------------------
    def _cos(self, x: np.ndarray, kind: str, n_terms: int = 192, scale: float = 12.0):
        """
        Whole-curve CDF/PDF via the COS (Fourier-cosine) inversion. Unlike the per-point de Hoog inversion (which
        re-evaluates the transform at ``t``-dependent nodes for every point), COS evaluates the characteristic
        function ``chi(w) = phi(-i w)`` on a *single* fixed frequency grid and reconstructs the curve at every ``x``
        by cheap cosine sums — so a whole curve (and, with the shared per-epoch generators, a whole spectrum) costs
        one set of transform evaluations. Used for plotting; ``cdf`` / ``pdf`` keep the exact per-point de Hoog.

        An atom at ``R = 0`` (e.g. an empty SFS bin), ``p0 = phi(inf)``, is split off so the cosine series only sees
        the smooth continuous part (avoiding Gibbs ringing); the CDF then steps from 0 to ``p0`` at the origin.
        """
        x = np.asarray(x, dtype=float)
        a, b = 0.0, self._range(scale)

        p0 = self.lst(1e8).real  # atom at 0
        k = np.arange(n_terms)
        w = k * np.pi / (b - a)
        chi = np.array([self.lst(-1j * wk) for wk in w])  # characteristic function chi(w) = phi(-i w)
        if p0 > 1e-9:
            chi = (chi - p0) / (1 - p0)  # continuous part only

        fk = (2.0 / (b - a)) * np.real(chi * np.exp(-1j * w * a))
        fk[0] *= 0.5
        xa = np.clip(x, a, b) - a

        if kind == 'pdf':
            curve = fk @ np.cos(np.outer(w, xa))
            return (1 - p0) * curve if p0 > 1e-9 else curve

        # CDF: analytic integral of the cosine series
        cdf_c = fk[0] * xa + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xa))
        cdf = p0 + (1 - p0) * cdf_c if p0 > 1e-9 else cdf_c
        return np.clip(cdf, 0.0, 1.0)

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
        """Fast PDF over a whole grid ``x`` via COS inversion (for plotting; see :meth:`_cos`)."""
        return self._cos(x, 'pdf', n_terms=n_terms)


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
