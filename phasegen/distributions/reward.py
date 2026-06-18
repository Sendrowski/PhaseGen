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
import functools
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import mpmath as mp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..rewards import Reward
from ..settings import Settings
from .base import CallableDistributionFunctions, JointDensity, JointCDF, \
    ConditionalDensity, ConditionalCDF, ConditionalQuantileFunction, \
    _LSTCumulativeDistributionFunction, _LSTDensityFunction, _LSTQuantileFunction
from ._moments import MomentEvaluator, _AUTO_PERM

if TYPE_CHECKING:
    from .phase_type import PhaseTypeDistribution

logger = logging.getLogger('phasegen')


class RewardDistribution(CallableDistributionFunctions):
    """
    Full distribution of the accumulated reward ``R = int_0^tau_abs r(X_s) ds`` to absorption, via the
    Laplace-Stieltjes transform and numerical inversion. Handles arbitrary (non-negative) rewards, zero-reward
    states, and arbitrary piecewise time-homogeneous demographies (so it is multi-epoch-native).

    Its ``cdf`` / ``pdf`` / ``quantile`` are callable-and-plottable :class:`DistributionFunction`s (see
    :class:`CallableDistributionFunctions`); this is the 1D object returned by ``SFSDistribution.bin`` etc. The de
    Hoog / Fourier-cosine inversion that turns the transform into those functions lives on the function objects (the
    :class:`~phasegen.distributions.base._LSTFunction` family); this distribution supplies the transform and scale
    primitives (:meth:`lst`, :meth:`_invert`, :meth:`_cumulants`, :meth:`_range`, :attr:`_time_scale`).
    """
    #: the 1D LST function-object flavours owning the de Hoog / cosine inversion machinery
    _cdf_function = _LSTCumulativeDistributionFunction
    _pdf_function = _LSTDensityFunction
    _quantile_function = _LSTQuantileFunction

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
        #: Optional human-readable label (e.g. ``"SFS bin 3"``) used in plot titles; set by ``bin()`` etc.
        self.label: Optional[str] = None

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
        # built once on the host and shared across all bins of a spectrum (see ``_reward_epoch_data``). The generators
        # are time-rescaled (``tau``) for large-N conditioning; the LST compensates with ``s tau`` (see ``lst``).
        data = self._host._reward_epoch_data_scaled
        r = r_full[data['idx']].astype(float)

        if np.any(r < 0):
            raise ValueError("RewardDistribution requires a non-negative reward.")

        return dict(r=r, tau=self._host._time_scale, **data)

    @property
    def _time_scale(self) -> float:
        """The inversion time-scale (average Ne at ``t = 0``; ``1.0`` outside the large-N regime), read straight from
        the host. Decoupled from :attr:`_setup` so the conditional flavours -- whose ``lst`` is a nested transform
        with no state-space reward to bind -- can scale their cumulant/quantile step without invoking ``_setup``."""
        return getattr(getattr(self, '_host', None), '_time_scale', 1.0)

    def lst(self, s: complex) -> complex:
        """The accumulated-reward Laplace-Stieltjes transform ``phi(s) = E[e^{-s R}]`` at (complex) ``s``."""
        st = self._setup
        # evaluate against the tau-scaled generators at s*tau (R -> R/tau); the result equals the unscaled phi(s)
        # exactly but stays well-conditioned for large N (see ``time_scale``)
        return _lst_from_shift((s * st['tau']) * st['r'], st['alpha'], st['T_epochs'], st['sparse'], st['lu_perm'])

    def _invert(self, transform, t: float) -> float:
        """
        Numerical Laplace inversion (de Hoog) of ``transform`` evaluated at ``t``.

        de Hoog evaluates ``transform`` at ``2 * degree + 1`` contour nodes (each a linear solve over the transient
        sub-generator -- the dominant cost for large state spaces) and combines them with an ill-conditioned QD
        recurrence that needs mpmath's extended precision. The degree is :attr:`Settings.dehoog_degree` (cost is
        linear in it; accuracy is non-monotonic, peaking near 15). (Parallelising the independent node solves across
        threads was tried and did not pay off -- the per-node matrix assembly holds the GIL, so the solves do not
        parallelise; whole-curve speed comes from the Fourier-cosine plotting path instead.)
        """
        if t <= 0:
            return 0.0

        def F(s):
            val = transform(complex(s))
            return mp.mpc(val.real, val.imag)

        return float(mp.invertlaplace(F, t, method='dehoog', degree=Settings.dehoog_degree))

    def _titled(self, base: str) -> str:
        """A plot title incorporating :attr:`label` (e.g. ``"SFS bin 3 CDF"``) when one has been set. Used by the
        function objects (the :class:`~phasegen.distributions.base._LSTFunction` family) for their plot titles."""
        return f"{self.label} {base}" if self.label else base

    def _cumulants(self) -> tuple:
        """Mean and variance of the accumulated reward from the LST near 0 (``phi(0) = 1``): ``c1 = -phi'(0)``,
        ``c2 = phi''(0) - phi'(0)^2``. Cheap (three transform evaluations); used to set the COS / plot range. The
        finite-difference step is scaled by ``1/tau`` (``tau ~`` the reward scale for large N) so that ``h * R`` stays
        small and ``phi(-h) = E[e^{h R}]`` does not overflow for large-N demographies."""
        h = 1e-4 / self._time_scale
        d1 = (self.lst(h).real - self.lst(-h).real) / (2 * h)
        d2 = (self.lst(h).real - 2.0 + self.lst(-h).real) / h ** 2
        return -d1, max(d2 - d1 ** 2, 1e-12)

    def _range(self, scale: float = 12.0) -> float:
        """An upper end for the support (``mean + scale * std``), for the COS interval and default plot grids."""
        c1, c2 = self._cumulants()
        return float(c1 + scale * np.sqrt(c2))


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
    # ``sparse`` gates the sparse matrix build and the sparse block-triangular LU of the final-epoch solve (which is
    # the large-space win and handles the s->inf atom shift directly). The finite-epoch matrix-exponential is always
    # dense (densifying a sparse block): the only alternative, the expm_multiply *action*, is norm-driven and cannot
    # evaluate the ``s = inf`` (1e8) atom shifts that every inversion needs -- so it has no usable role here (unlike
    # the moment path, which never inverts an atom and gates its action on ``Settings.expm_action_min_dim``).
    sparse = nt >= Settings.closed_form_sparse_min_states

    T_epochs = []
    for epoch in host._get_epochs_until_unbounded():
        ss.update_epoch(epoch)
        host._check_numerical_stability(ss.S, 0)
        T_epochs.append((host._transient_block(idx, sparse=sparse), epoch.start_time, epoch.end_time))

    # the block-triangular ordering of the final (unbounded) epoch's sub-generator depends only on its sparsity
    # pattern, which is fixed across the many shifted solves of the de Hoog inversion; compute it once here so the
    # per-node factorization can reuse it (the SCC analysis is the dominant cost of the sparse solve)
    lu_perm = MomentEvaluator._block_triangular_order(T_epochs[-1][0]) if sparse else None

    return dict(idx=idx, alpha=alpha, nt=nt, sparse=sparse, T_epochs=T_epochs, lu_perm=lu_perm)


def _avg_ne_at_zero(host) -> float:
    """Average effective population size across populations at ``t = 0`` (the demography's first epoch)."""
    try:
        sizes = [float(v) for v in host.demography.get_epochs([0.0])[0].pop_sizes.values()]
        return float(np.mean(sizes)) if sizes else 1.0
    except Exception:
        return 1.0


def time_scale(host) -> float:
    """
    Time-rescaling factor for the accumulated-reward LST inversion.

    For a large-N demography the LST is evaluated at tiny arguments (the COS frequencies / de Hoog nodes scale like
    ``1/N``) against a generator whose rates also scale like ``1/N``, so the reward-shifted sub-generator
    ``diag(s r) - T`` is ``~1/N``-scaled and its LU factorization loses precision and can overflow to inf/NaN.
    Rescaling time by the average Ne at ``t = 0`` (``R -> R/tau``, ``T -> tau T``, epoch durations ``-> /tau``)
    makes that solve ``O(1)``-conditioned. The transform value is *exactly invariant* under this rescaling -- the
    ``tau`` factors cancel in both the ``expm(Q tau)`` of the bounded epochs and the final-epoch solve (see
    :func:`_scale_epoch_data` and :meth:`RewardDistribution.lst`) -- so nothing downstream changes. A no-op (returns
    ``1.0``) in the normal range so existing single-N fixtures are byte-identical.
    """
    tau = _avg_ne_at_zero(host)
    return tau if (tau > 1e2 or tau < 1e-2) else 1.0


def _scale_epoch_data(data: dict, tau: float) -> dict:
    """Return a copy of :func:`_build_epoch_data` output with the per-epoch sub-generators scaled by ``tau`` and the
    epoch boundaries by ``1/tau`` (the reward stays unscaled; the LST is evaluated at ``s tau`` -- see
    :meth:`RewardDistribution.lst`). The block-triangular ordering is pattern-fixed under the positive scaling, so it
    is reused. A no-op when ``tau == 1``."""
    if tau == 1.0:
        return data
    scaled = dict(data)
    scaled['T_epochs'] = [(T * tau, t0 / tau, t1 / tau) for (T, t0, t1) in data['T_epochs']]
    return scaled


def _exit_rates(T) -> np.ndarray:
    """Per-state rate of (direct) absorption = row deficit of the transient sub-generator, ``-T 1``."""
    return -np.asarray(T @ np.ones(T.shape[0])).ravel()


def _lst_from_shift(shift: np.ndarray, alpha: np.ndarray, T_epochs, sparse: bool, perm=_AUTO_PERM) -> complex:
    """
    Accumulated-reward LST evaluated with an arbitrary diagonal *shift vector* ``shift`` (the reward enters the
    generator only as ``-diag(shift)``). For one reward ``shift = s diag(r)``; for two rewards (the joint
    transform) ``shift = s_a r_a + s_b r_b`` — so the univariate and joint cases share this one routine.

    ``E[e^{-<shift-as-accumulated>}] = c + a (diag(shift) - T_m)^{-1} (-T_m 1)``, with ``[a, c]`` the transient /
    absorbed mass pushed through the finite epochs (augmented with the absorbing state, reward 0, so absorbed mass
    keeps its frozen weight).

    ``perm`` is the precomputed block-triangular ordering of the final epoch's sub-generator (pattern-fixed across
    shifts), passed through to :meth:`MomentEvaluator._lu_solver` so repeated evaluations (the de Hoog nodes) skip the
    per-solve SCC analysis.
    """
    nt = len(alpha)
    vec = np.concatenate([alpha, [0.0]]).astype(complex)

    for T, t0, t1 in T_epochs[:-1]:
        exit_col = _exit_rates(T)
        tau = t1 - t0
        # finite-epoch propagation by a dense matrix exponential. A sparse transient block (large-space build) is
        # densified here: the expm_multiply *action* alternative is norm-driven and cannot evaluate the s->inf atom
        # shifts the inversion needs, so it has no usable role on this path (see ``_build_epoch_data``).
        Q = np.zeros((nt + 1, nt + 1), dtype=complex)
        Q[:nt, :nt] = (T.toarray() if sp.issparse(T) else np.asarray(T)) - np.diag(shift)
        Q[:nt, nt] = exit_col
        vec = vec @ sla.expm(Q * tau)

    a, c = vec[:nt], vec[nt]
    Tm = T_epochs[-1][0]
    A = (sp.diags(shift) if sparse else np.diag(shift)) - Tm
    solve = MomentEvaluator._lu_solver(A, sparse, perm)
    return complex(c + a @ solve(_exit_rates(Tm)))


class JointRewardDistribution(CallableDistributionFunctions):
    """
    Joint distribution of two accumulated rewards ``R_a = int r_a(X_s) ds`` and ``R_b = int r_b(X_s) ds`` to
    absorption, the distributional object behind a cross-moment ``E[R_a R_b]`` (the within-tree 2-SFS / the
    two-locus SFS). The joint Laplace-Stieltjes transform uses the *combined* generator shift,

        Phi(s_a, s_b) = E[e^{-s_a R_a - s_b R_b}] = _lst_from_shift(s_a r_a + s_b r_b, ...),

    so it is the same machinery as the univariate :class:`RewardDistribution` with a two-parameter shift, and is
    multi-epoch-native. Setting one argument to zero recovers a marginal; mixed derivatives at the origin recover
    the cross-moments. The joint CDF/PDF (2D inversion) and the product distribution are views built on top.
    """
    #: bivariate function-object flavours (built by the :class:`CallableDistributionFunctions` mixin, passing the
    #: ``plot_surface`` callback); a joint has no quantile (a 2D quantile is not well-defined)
    _pdf_function = JointDensity
    _cdf_function = JointCDF
    _quantile_function = None

    #: Number of cosine terms per axis for the 2D Fourier-cosine joint density (:attr:`_cos2d`). The cost is the square
    #: of this (an ``n x n`` coefficient matrix of joint-LST evaluations), so it is smaller than the 1D term count; at
    #: 128 the heatmap/surface ringing is ~0.5% of the peak (vs ~2.5% at 64). For a de-Hoog-accurate (non-ringing) but
    #: far slower density, use ``pdf.plot_surface(method='dehoog')`` (nested de Hoog inversion).
    _cos2d_terms: int = 128

    #: Window half-width for the 2D Fourier-cosine expansion, in std-units past the mean per axis (``b = mean + scale *
    #: std``). The cosine resolves at the single scale ``b / n_terms``, so an over-wide window wastes resolution on a
    #: near-empty tail and *under-resolves the near-origin* rise (the source of the near-axis CDF bias). Validated
    #: against msprime across scenarios, ``scale = 5`` is the robust optimum -- it covers the query range (~the 0.95-
    #: 0.99 marginal quantile) with margin yet keeps the resolution fine: ~2-3.5x more accurate near the origin than
    #: the old ``10`` and close to de Hoog, at no extra cost. Too small (<=3) truncates real tail mass and aliases.
    _cos2d_window_scale: float = 5.0

    #: Per-axis node count of the uniform grid on which the cosine 2D **density** is taken as a mixed *finite
    #: difference* of the cosine box CDF (:meth:`_cc_box`), then cubic-interpolated to the query points -- mirroring the
    #: 1D cosine ``pdf_curve`` (a finite difference of its CDF), so the PDF is consistently the numerical derivative of
    #: the CDF and free of the raw cosine sum's residual Gibbs negativity. The grid spans the queried range, so the
    #: step adapts to it; ``_cc_box`` is analytic and cheap, so this is essentially free.
    _cos2d_pdf_grid: int = 50

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
        #: Optional human-readable label (e.g. ``"bins (1, 2)"``) used in plot titles; set by ``joint_distribution()``.
        self.label: Optional[str] = None

    @cached_property
    def _setup(self):
        """Bind both reward vectors to the host's shared (reward-independent) per-epoch generators (time-rescaled by
        ``tau`` for large-N conditioning; the LST compensates with ``s tau`` -- see :meth:`lst`)."""
        ss = self._host.state_space
        data = self._host._reward_epoch_data_scaled
        out = dict(tau=self._host._time_scale, **data)
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
        # both rewards share the time-scale tau (R -> R/tau): evaluate at s*tau against the tau-scaled generators;
        # the value equals the unscaled joint LST exactly but stays well-conditioned for large N (see ``time_scale``)
        tau = st['tau']
        return _lst_from_shift((s_a * tau) * st['ra'] + (s_b * tau) * st['rb'], st['alpha'], st['T_epochs'],
                               st['sparse'], st['lu_perm'])

    def _lst_grid(self, s_a_vals: np.ndarray, s_b_vals: np.ndarray) -> np.ndarray:
        """The joint LST ``Phi(s_a, s_b)`` on the full outer grid ``s_a_vals x s_b_vals`` (shape
        ``(len(s_a_vals), len(s_b_vals))``), the batched form of :meth:`lst` used to build the 2D cosine coefficients.

        For a single (homogeneous) epoch with a dense generator, ``Phi = alpha (diag(s_a tau r_a + s_b tau r_b) -
        T)^{-1}(-T 1)``. Fixing ``s_a``, the column family ``(M + s_b tau diag(r_b)) x = -T 1`` with ``M = diag(s_a
        tau r_a) - T`` is a *shifted* linear system in ``s_b``: one generalized Schur (QZ) factorization of the pencil
        ``(M, tau diag(r_b))`` then solves every ``s_b`` by triangular back-substitution (O(n^2)). This replaces the
        naive grid's ``len(s_a) x len(s_b)`` dense LU factorizations (O(n^3) each) with ``len(s_a)`` QZ factorizations
        -- a ~n-fold cut in the cubic work, the dominant cost of :attr:`_cos2d`. ``diag(r_b)`` is generally singular
        (rewards have zero entries); the QZ handles the singular pencil. The multi-epoch / sparse case -- where the
        per-shift cost is dominated by the cross-epoch ``expm``, not the final solve -- falls back to per-element
        :meth:`lst`, giving an identical result."""
        st = self._setup
        s_a_vals, s_b_vals = np.asarray(s_a_vals, dtype=complex), np.asarray(s_b_vals, dtype=complex)

        if st['sparse'] or len(st['T_epochs']) != 1:
            return np.array([[self.lst(sa, sb) for sb in s_b_vals] for sa in s_a_vals], dtype=complex)

        tau = st['tau']
        Tm = np.asarray(st['T_epochs'][-1][0], dtype=float)
        n = Tm.shape[0]
        exit_col = (-Tm @ np.ones(n)).astype(complex)  # -T 1
        alpha = np.asarray(st['alpha'], dtype=float)
        ra_t, rb_t = st['ra'] * tau, st['rb'] * tau

        # one QZ per *outer* node, every *inner* node by triangular back-substitution -- so QZ over the shorter axis
        # (the same matrix ``A = diag(s_a r_a + s_b r_b) tau - T`` factors either way, by symmetry of the two rewards)
        transpose = len(s_a_vals) > len(s_b_vals)
        outer, r_out = (s_b_vals, rb_t) if transpose else (s_a_vals, ra_t)
        inner, r_in = (s_a_vals, ra_t) if transpose else (s_b_vals, rb_t)
        N = np.diag(r_in).astype(complex)  # pencil B: the inner variable multiplies this

        out = np.empty((len(outer), len(inner)), dtype=complex)
        for i, so in enumerate(outer):
            M = np.diag(so * r_out) - Tm  # pencil A; A(s_inner) = M + s_inner N = diag(shift) - T
            S, T, Q, Z = sla.qz(M, N, output='complex')  # M = Q S Z^H, N = Q T Z^H, S/T upper-triangular
            c = Q.conj().T @ exit_col  # (S + s_inner T) y = c, with y = Z^H x
            aZ = alpha @ Z             # Phi = alpha @ x = alpha @ Z @ y = aZ @ y
            for j, si in enumerate(inner):
                y = sla.solve_triangular(S + si * T, c, check_finite=False)
                out[i, j] = aZ @ y
        return out.T if transpose else out

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
    def _cos_axis_coeffs(self) -> dict:
        """Fourier-cosine coefficients for the two axis-atom sub-CDFs, the cosine replacement for the per-point de Hoog
        atom inversions in :meth:`_cdf_grid`. Validated across scenarios to match the de Hoog atoms (and msprime)
        identically, so the cosine CDF path uses these and avoids de Hoog entirely (the de Hoog mixing bought no
        accuracy -- both are equally imperfect near 0). Each is a defective 1D distribution: ``g_b(x) = P(R_a <= x,
        R_b = 0)`` (key ``'b'``, sub-transform ``lst(., inf)``) and ``g_a(y) = P(R_a = 0, R_b <= y)`` (key ``'a'``,
        ``lst(inf, .)``), with atom ``P(both = 0)`` at 0 and continuous mass up to ``P(R_b = 0)`` / ``P(R_a = 0)``,
        COS-inverted over the corresponding marginal's support window."""
        big = 1e8
        both0 = self._atoms['both0']
        out = {}
        for key, total, marg in (('b', self._atoms['b0'], self.marginal('a')),
                                 ('a', self._atoms['a0'], self.marginal('b'))):
            b = marg._range(12.0)
            w = np.arange(marg.cdf._cos_terms) * np.pi / b
            # chi(w) = phi(-i w) of the sub-transform: for 'b' it is lst(., inf) (sweep s_a), for 'a' lst(inf, .)
            # (sweep s_b); the batched _lst_grid does the whole sweep with a single QZ (one fixed inf-coordinate)
            chi = (self._lst_grid(-1j * w, np.array([big]))[:, 0] if key == 'b'
                   else self._lst_grid(np.array([big]), -1j * w)[0, :])
            cont = total - both0
            chi_c = (chi - both0) / cont if cont > 1e-12 else chi  # remove the R=0 atom, normalize the continuous part
            fk = (2.0 / b) * np.real(chi_c)
            fk[0] *= 0.5
            out[key] = dict(b=b, w=w, fk=fk, atom=both0, cont=cont)
        return out

    def _cos_axis(self, which: str, xs: np.ndarray) -> np.ndarray:
        """The axis-atom sub-CDF on ``xs`` via the cached 1D Fourier-cosine fit (:attr:`_cos_axis_coeffs`) -- the
        cosine analogue of the de Hoog atom inversion: ``which='b'`` -> ``g_b(x) = P(R_a <= x, R_b = 0)``,
        ``which='a'`` -> ``g_a(y) = P(R_a = 0, R_b <= y)``. At ``x = 0`` it returns the atom ``P(both = 0)`` exactly."""
        c = self._cos_axis_coeffs[which]
        w, fk = c['w'], c['fk']
        xa = np.clip(np.asarray(xs, dtype=float), 0.0, c['b'])
        Fc = fk[0] * xa + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xa))
        return c['atom'] + c['cont'] * np.clip(Fc, 0.0, 1.0)

    @cached_property
    def _cos2d(self) -> dict:
        """
        The *continuous-continuous* joint density (both rewards ``> 0``) as a 2D Fourier-cosine (COS) expansion on
        ``[0, b_a] x [0, b_b]``. The marginal atoms are removed by inclusion-exclusion so the cosine series only
        sees the smooth part: ``cf_cc(w_a, w_b) = Phi(-i w_a, -i w_b) - Phi(-i w_a, inf) - Phi(inf, -i w_b) +
        P(both = 0)``. Returns the coefficient matrix and the (zero-based) ranges/frequencies.
        """
        n_terms, scale, big = self._cos2d_terms, self._cos2d_window_scale, 1e8
        p00 = self._atoms['both0']
        ca, va = self.marginal('a')._cumulants()
        cb, vb = self.marginal('b')._cumulants()
        ba, bb = ca + scale * np.sqrt(va), cb + scale * np.sqrt(vb)
        ua = np.arange(n_terms) * np.pi / ba
        ub = np.arange(n_terms) * np.pi / bb

        # all joint-LST evaluations on one batched grid (rows ``s_a in {-i u_a} u {inf}``, columns
        # ``s_b in {-i u_b} u {+i u_b} u {inf}``) via the shifted-system QZ solve (see :meth:`_lst_grid`) -- the
        # dominant cost of this expansion, an n_terms x n_terms coefficient matrix of LST values
        sa, sb = -1j * ua, np.concatenate([-1j * ub, 1j * ub])
        G = self._lst_grid(np.concatenate([sa, [big]]), np.concatenate([sb, [big]]))
        phi_a_inf = G[:n_terms, 2 * n_terms]      # Phi(-i w_a, inf), reused across w_b
        phi_inf_b_p = G[n_terms, :n_terms]        # Phi(inf, -i w_b)
        phi_inf_b_m = G[n_terms, n_terms:2 * n_terms]  # Phi(inf, +i w_b)
        pp = G[:n_terms, :n_terms] - phi_a_inf[:, None] - phi_inf_b_p[None, :] + p00       # Phi(-i w_a, -i w_b)
        pm = G[:n_terms, n_terms:2 * n_terms] - phi_a_inf[:, None] - phi_inf_b_m[None, :] + p00  # Phi(-i w_a, +i w_b)

        A = (2.0 / ba) * (2.0 / bb) * 0.5 * np.real(pp + pm)  # lower limits are 0, so exp(-i w a) = 1
        A[0, :] *= 0.5
        A[:, 0] *= 0.5

        # Lanczos sigma-filter: scale each coefficient by sigma_k sigma_l with sigma_k = sinc(k / n_terms). This damps
        # the high-frequency terms responsible for Gibbs ringing near the origin edge (the residual negative wiggle in
        # the density and the surface/heatmap), empirically the only robust fix -- window tightening trades one bin's
        # error for another's and grid-averaging two windows does not cancel the ring (it is pinned to the origin, not
        # a shiftable interference pattern). The k=0 coefficient is untouched (sinc(0)=1), and the total box mass
        # depends only on it (the cos antiderivatives vanish at the window edge for k>=1), so the filter is
        # mass-preserving: it leaves :meth:`_cc_box` (the CDF) essentially unchanged while removing the density ringing.
        sigma = np.sinc(np.arange(n_terms) / n_terms)
        A *= np.outer(sigma, sigma)
        return dict(ba=ba, bb=bb, ua=ua, ub=ub, A=A)

    @cached_property
    def _cos2d_wiggle_check(self) -> float:
        """One-time accuracy / wiggle check for the 2D Fourier-cosine inversion (the analogue of the 1D cosine ripple
        warning in :meth:`_fit_cos`). The cosine error is concentrated **near the axes / origin** (empirically 5-10x
        the interior across non-homogeneous demographies; the Lanczos filter keeps the *bulk* density essentially
        non-negative, so interior ringing is not a reliable separate signal). The robust detector is therefore a
        *margin* one: the cosine joint CDF must reduce to the **accurate 1D marginal CDF** as the other coordinate
        grows, so a large near-origin discrepancy means the series cannot resolve a sharp / skewed near-origin feature
        (e.g. a heavy-tailed multi-epoch reward) and is biased there. This fires for the value-inaccurate cases and
        stays quiet when the CDF / quantiles are accurate. Computed from the local cosine coefficients (no ``_cc_box``
        / ``_density`` call -> no recursion) plus a few de Hoog marginal solves. Logs once (cached); returns the worst
        absolute near-origin CDF discrepancy.
        """
        st = self._cos2d
        big = 1e8
        ma = self.marginal('a')
        xs = np.linspace(0.0, float(ma.quantile(0.4)), 5)[1:]  # near-origin small-x points, where the bias concentrates
        # cosine full CDF F(x, inf) = axis atoms (de Hoog) + the cosine continuous box integrated to the window edge
        box = self._cos_antideriv(st['ua'], np.minimum(xs, st['ba'])) @ st['A'] @ self._cos_antideriv(st['ub'], np.array([st['bb']])).T
        g_b = np.array([ma._invert(lambda s: self.lst(s, big) / s, float(x)) for x in xs])
        cos_cdf = g_b + self._atoms['a0'] - self._atoms['both0'] + box[:, 0]
        true_cdf = np.array([float(ma.cdf(float(x))) for x in xs])
        err = float(np.abs(cos_cdf - true_cdf).max())
        if Settings.check_inversions and err > 0.03:
            self._logger.warning(
                "The 2D Fourier-cosine joint inversion under-resolves near the origin: its CDF deviates from the "
                "exact 1D marginal by up to %.1f%% there (a sharp / skewed near-origin feature the cosine series "
                "cannot capture, so the heatmap/surface rings and is biased). Use pdf(...) / cdf(...) with "
                "method='dehoog' for accurate values.", err * 100,
            )
        return err

    def _density(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The continuous joint density on the outer grid ``xs x ys`` (shape ``(len(xs), len(ys))``), as the **mixed
        finite difference of the cosine box CDF** (:meth:`_cc_box`) on a uniform grid spanning the queried range, then
        cubic-interpolated to ``xs x ys``. This mirrors the 1D cosine ``pdf_curve`` (a finite difference of its CDF):
        the PDF is consistently the numerical derivative of the cosine CDF, which avoids the raw cosine sum's residual
        Gibbs negativity. ``_cc_box`` is analytic (closed-form cosine antiderivatives), so this is cheap."""
        from scipy.interpolate import RectBivariateSpline

        _ = self._cos2d_wiggle_check  # one-time ringing/under-resolution warning (cached)
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        n = max(6, self._cos2d_pdf_grid)
        gx = np.linspace(0.0, max(float(xs.max()), 1e-9) * 1.1, n)
        gy = np.linspace(0.0, max(float(ys.max()), 1e-9) * 1.1, n)
        hx, hy = gx[1] - gx[0], gy[1] - gy[0]
        # box CDF on the grid (vanishes on the axes), then the mixed central second difference at the interior nodes
        F = np.zeros((n, n))
        F[1:, 1:] = self._cc_box(gx[1:], gy[1:])
        dens = (F[2:, 2:] - F[2:, :-2] - F[:-2, 2:] + F[:-2, :-2]) / (4.0 * hx * hy)
        k = min(3, dens.shape[0] - 1)
        return RectBivariateSpline(gx[1:-1], gy[1:-1], dens, kx=k, ky=k)(xs, ys)

    def _nested_invert(self, xs: np.ndarray, ys: np.ndarray, kind: str, M: int = 8) -> np.ndarray:
        """
        A continuous-continuous joint quantity on the grid ``xs x ys`` by **direct nested Laplace inversion** of the
        joint transform (no cosine expansion). The marginal atoms are first removed by inclusion-exclusion
        (``cc(s_a, s_b) = Phi(s_a, s_b) - Phi(s_a, inf) - Phi(inf, s_b) + P(both=0)``), so the transform decays and the
        inversion sees a genuine 2D density transform. ``kind`` selects what is inverted:

        - ``'pdf'``: ``cc`` -> the density ``f(x, y)``;
        - ``'cdf'``: ``cc / (s_a s_b)`` -> the box integral ``int_0^x int_0^y f`` (dividing a transform by ``s`` is
          integration).

        For each ``y`` the inner inversion in ``s_b`` is Gaver-Stehfest (it must accept the *complex* ``s_a`` the outer
        inversion probes it at); the outer inversion in ``s_a`` is de Hoog (far more tolerant of the inner result's
        double-precision noise than a second Stehfest, which would amplify it catastrophically). A full inner Stehfest
        per outer de Hoog node per grid point makes this much slower than the cosine path, so use a coarse grid.

        The two inclusion-exclusion marginal terms depend on a single argument each, and the node sets repeat across
        the grid (de Hoog ``s_a`` down each column, Stehfest ``s_b`` across each row), so they are memoised grid-wide
        (turning the 3 LST solves per ``cc`` into ~1); the full 2D term is distinct per node pair and is not cached.
        """
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        big, both0 = 1e8, self._atoms['both0']
        integrate = kind == 'cdf'  # divide the transform by s on each axis to integrate the density into a box CDF

        marg_a = functools.lru_cache(maxsize=None)(lambda s_a: self.lst(s_a, big))
        marg_b = functools.lru_cache(maxsize=None)(lambda s_b: self.lst(big, s_b))

        def cc(s_a: complex, s_b: complex) -> complex:
            """The continuous-continuous transform (marginal atoms removed by inclusion-exclusion)."""
            return self.lst(s_a, s_b) - marg_a(s_a) - marg_b(s_b) + both0

        out = np.zeros((xs.size, ys.size))
        for j, y in enumerate(ys):
            if y <= 0:
                continue  # both the density and the box vanish on the axis
            # inner Stehfest inversion of cc(s_a, .) in s_b at y; for the CDF the extra 1/s_b (inner) and 1/s_a (outer)
            # integrate the density up to (x, y). The outer de Hoog evaluates it at complex s_a.
            def G(s_a: complex, _y: float = float(y)) -> complex:
                inner = _stehfest_invert(lambda s_b: cc(s_a, s_b) / s_b if integrate else cc(s_a, s_b), _y, M)
                return inner / s_a if integrate else inner

            def F(s, _G=G):  # de Hoog wants the transform as an mpmath complex
                v = _G(complex(s))
                return mp.mpc(v.real, v.imag)

            for i, x in enumerate(xs):
                if x <= 0:
                    continue
                # mpmath's default degree (20) is kept here rather than Settings.dehoog_degree: the outer inversion
                # sees the noisy inner Stehfest result, and a lower degree spikes near the origin (a higher one in the
                # far tail) -- the default is the more stable middle ground for this nested, noise-carrying input.
                out[i, j] = float(mp.invertlaplace(F, float(x), method='dehoog'))
        return out

    def _density_nested(self, xs: np.ndarray, ys: np.ndarray, M: int = 8) -> np.ndarray:
        """The continuous joint density by direct nested inversion -- the accurate (slow) ``exact`` counterpart of the
        cosine :meth:`_density`, and the de Hoog ``pdf``. Clipped to non-negative. See :meth:`_nested_invert`."""
        raw = self._nested_invert(xs, ys, 'pdf', M)
        self._warn_if_negative(raw, 'joint density (de Hoog)')
        return np.clip(raw, 0.0, None)

    @staticmethod
    def _use_dehoog(method: str) -> bool:
        """Resolve a 2D inversion ``method`` to a de-Hoog/cosine choice: ``'dehoog'`` -> True, ``'cos'`` -> False, and
        ``None`` (the default) -> the accurate de Hoog. Pass ``method='cos'`` for the fast cosine inversion (e.g. plots)."""
        m = 'dehoog' if method is None else method
        if m not in ('dehoog', 'cos'):
            raise ValueError(f"method must be 'dehoog', 'cos', or None (the de Hoog default); got {m!r}.")
        return m == 'dehoog'

    def _pdf(self, x, y, method: str = None):
        """
        Joint probability density of ``(R_a, R_b)`` (the continuous, both-positive part). The distribution also has
        atom mass on the axes where a reward is zero (see :attr:`_atoms`); a non-empty SFS bin pair has none there.

        ``method`` selects the inversion: ``'cos'`` the fast cosine expansion, ``'dehoog'`` the accurate nested de Hoog;
        ``None`` (the default) uses de Hoog (the plots pass ``method='cos'`` for speed). The de Hoog density is the mixed
        derivative of a spline through the clean nested-de-Hoog box CDF (see :meth:`_density_nested`).

        :param x: ``R_a`` value(s).
        :param y: ``R_b`` value(s).
        :param method: ``'dehoog'`` / ``'cos'``, or ``None`` for the de Hoog default.
        :return: Density, scalar or a ``(len(x), len(y))`` grid.
        """
        if self._is_diagonal:
            raise NotImplementedError("The joint density is singular when both rewards are identical (R_a = R_b "
                                      "almost surely): the law lives on the diagonal and has no 2D density. Use "
                                      "cdf(x, y) = marginal CDF at min(x, y), or the 1D marginal density.")
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        if self._use_dehoog(method):
            f = self._density_nested(xs, ys)
        else:
            raw = self._density(xs, ys)  # the cosine 2D density can dip negative near the origin edge (Gibbs)
            self._warn_if_negative(raw, 'joint density (cosine)')
            f = np.clip(raw, 0.0, None)
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
        _ = self._cos2d_wiggle_check  # one-time ringing/under-resolution warning (cached)
        st = self._cos2d
        Ix = self._cos_antideriv(st['ua'], np.minimum(xs, st['ba']))   # (len_x, N)
        Iy = self._cos_antideriv(st['ub'], np.minimum(ys, st['bb']))   # (len_y, N)
        return Ix @ st['A'] @ Iy.T                                     # (len_x, len_y)

    def _cc_box_dehoog(self, xs: np.ndarray, ys: np.ndarray, M: int = 8) -> np.ndarray:
        """The continuous-continuous box CDF ``int_0^x int_0^y f_cc`` by direct nested inversion -- the accurate
        counterpart of the cosine :meth:`_cc_box`. The cosine box reconstructs the density on a single fixed window
        ``[0, b_a] x [0, b_b]``; for a heavily skewed reward (e.g. a multi-epoch demography whose std greatly exceeds
        its mean, so the window spans 0 to many tens) a fixed number of cosine terms cannot resolve the sharp
        near-origin rise, and the box is badly wrong at small ``x``/``y`` (it does not even reduce to the marginal as
        the other coordinate grows). De Hoog adapts its contour per point, so it has no such limit. See
        :meth:`_nested_invert`."""
        return self._nested_invert(xs, ys, 'cdf', M)

    def _cdf_grid(self, xs: np.ndarray, ys: np.ndarray, dehoog: bool = True) -> np.ndarray:
        """Joint CDF on the grid ``xs x ys``: the axis atoms (marginal sub-transform inversions, one per grid line,
        always per-point de Hoog) plus the continuous box integral. ``dehoog`` selects the box method: the accurate
        nested de Hoog (:meth:`_cc_box_dehoog`, the default -- correct even for skewed multi-epoch rewards) or the fast
        cosine box (:meth:`_cc_box`, for dense plotting grids where the near-origin bias is an acceptable tradeoff)."""
        big = 1e8
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        both0 = self._atoms['both0']
        # axis atoms P(R_a=0, R_b<=y) and P(R_b=0, R_a<=x). The cosine path inverts them with the fast 1D Fourier-
        # cosine (:meth:`_cos_axis`, validated to match de Hoog/msprime -- no de Hoog mixing); the de Hoog box path
        # inverts them per point with de Hoog, using the exact limit P(both=0) at the unreliable atom edge (t=0).
        if dehoog:
            g_a = np.array([both0 if y == 0 else self.marginal('b')._invert(lambda s: self.lst(big, s) / s, float(y)) for y in ys])
            g_b = np.array([both0 if x == 0 else self.marginal('a')._invert(lambda s: self.lst(s, big) / s, float(x)) for x in xs])
            cc = self._cc_box_dehoog(xs, ys)
        else:
            g_a, g_b = self._cos_axis('a', ys), self._cos_axis('b', xs)
            cc = self._cc_box(xs, ys)
        return g_b[:, None] + g_a[None, :] - both0 + cc

    def _cdf(self, x, y, method: str = None):
        """
        Joint CDF ``P(R_a <= x, R_b <= y)``: the axis atoms ``P(R_a = 0, R_b <= y)`` and ``P(0 < R_a <= x,
        R_b = 0)`` (from inverting the marginal sub-transforms ``Phi(inf, .)`` / ``Phi(., inf)``) plus the
        continuous box integral ``P(0 < R_a <= x, 0 < R_b <= y)``. ``method`` selects the box method: ``'cos'`` the fast
        cosine box, ``'dehoog'`` the accurate nested de Hoog (no near-origin bias for skewed multi-epoch rewards);
        ``None`` (the default) uses de Hoog (the plots pass ``method='cos'`` for speed). Accepts scalars or arrays,
        returning a scalar or the ``(len(x), len(y))`` grid.

        When both rewards are identical (``R_a = R_b`` almost surely, e.g. a bin paired with itself) the joint law
        is singular on the diagonal; the CDF then reduces exactly to ``P(R <= min(x, y))`` of the shared marginal.
        """
        xs, ys = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        if self._is_diagonal:
            m = self.marginal('a')
            # at t = 0 the marginal CDF is the atom P(R = 0) (the de Hoog inversion misses the jump there)
            G = np.array([[float(self._atoms['both0'] if min(xx, yy) <= 0.0 else m.cdf(min(xx, yy)))
                           for yy in ys] for xx in xs])
        else:
            G = self._cdf_grid(xs, ys, dehoog=self._use_dehoog(method))
        return float(G.ravel()[0]) if G.size == 1 else G

    def _title(self, kind: str) -> str:
        """Joint plot title incorporating the bin :attr:`label` when set."""
        return f"Joint {kind.upper()} {self.label}" if self.label else f"Joint reward {kind.upper()}"

    def _plot_pdf(self, ax=None, n_points: int = None, show: bool = True, file: str = None, title: str = None,
                  method: str = 'cos'):
        """Heatmap of the joint (continuous) density of ``(R_a, R_b)``. ``method='dehoog'`` uses the accurate nested
        de Hoog inversion (a coarser default grid); the default ``'cos'`` uses the fast cosine reconstruction."""
        n_points = n_points or (25 if self._use_dehoog(method) else 120)
        return self._plot_joint('pdf', ax, n_points, show, file, title or self._title('pdf'), surface=False, method=method)

    def _plot_cdf(self, ax=None, n_points: int = None, show: bool = True, file: str = None, title: str = None,
                  method: str = 'cos'):
        """Heatmap of the joint CDF of ``(R_a, R_b)``. A coarser default grid than the density: each grid node is an
        analytically integrated 2D box (cosine) or a nested de Hoog box, so the CDF surface is costlier per point."""
        n_points = n_points or (25 if self._use_dehoog(method) else 60)
        return self._plot_joint('cdf', ax, n_points, show, file, title or self._title('cdf'), surface=False, method=method)

    def _plot_pdf_surface(self, ax=None, n_points: int = None, show: bool = True, file: str = None, title: str = None,
                          method: str = 'cos'):
        """3D surface of the joint (continuous) density of ``(R_a, R_b)``. ``method='dehoog'`` uses the accurate nested
        de Hoog inversion (a coarser default grid); the default ``'cos'`` uses the fast cosine reconstruction."""
        n_points = n_points or (25 if self._use_dehoog(method) else 80)
        return self._plot_joint('pdf', ax, n_points, show, file, title or self._title('pdf'), surface=True, method=method)

    def _plot_cdf_surface(self, ax=None, n_points: int = None, show: bool = True, file: str = None, title: str = None,
                          method: str = 'cos'):
        """3D surface of the joint CDF of ``(R_a, R_b)`` (coarser default grid than the density -- see :meth:`_plot_cdf`)."""
        n_points = n_points or (25 if self._use_dehoog(method) else 60)
        return self._plot_joint('cdf', ax, n_points, show, file, title or self._title('cdf'), surface=True, method=method)

    def _plot_joint(self, kind, ax, n_points, show, file, title, surface=False, method='cos'):
        import matplotlib.pyplot as plt

        dehoog = self._use_dehoog(method)
        st = self._cos2d
        # end each axis at the configured marginal quantile (like the 1D plots) so a heavy upper tail does not
        # stretch the view to mean + many std; clip to the cosine window the density was reconstructed on
        q = Settings.plot_endpoint_quantile
        xs = np.linspace(0, min(self.marginal('a').quantile(q), st['ba']), n_points)
        ys = np.linspace(0, min(self.marginal('b').quantile(q), st['bb']), n_points)
        if kind == 'cdf':
            # the axis atoms are always inverted per point (de Hoog); ``method`` selects the continuous box method --
            # the fast cosine box for the dense default grid, or the accurate (but per-point, slow) nested de Hoog box
            if dehoog:
                self._logger.info("Computing the joint CDF box by direct nested inversion on a %dx%d grid; this is "
                                  "slow.", len(xs), len(ys))
            Z = self._cdf_grid(xs, ys, dehoog=dehoog)
        elif dehoog:
            # nested de Hoog density (mixed derivative of a spline through the box CDF); slow, hence the coarse grid
            self._logger.info("Computing the joint density by nested de Hoog inversion on a %dx%d grid; this is slow.",
                              len(xs), len(ys))
            Z = self._density_nested(xs, ys)
        else:
            Z = np.clip(self._density(xs, ys), 0.0, None)
        zlim = dict(vmin=0.0, vmax=1.0) if kind == 'cdf' else {}  # a CDF is a probability -> fix its scale to [0, 1]

        if surface:
            if ax is None:
                ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(*np.meshgrid(xs, ys), Z.T, cmap='viridis', **zlim)
            ax.set_zlabel('F(R_a, R_b)' if kind == 'cdf' else 'f(R_a, R_b)')
            if kind == 'cdf':
                ax.set_zlim(0.0, 1.0)  # a CDF spans [0, 1]
        else:
            if ax is None:
                ax = plt.gca()
            mesh = ax.pcolormesh(xs, ys, Z.T, shading='auto', cmap='viridis', **zlim)
            ax.figure.colorbar(mesh, ax=ax)
        ax.set_xlabel('$R_a$')
        ax.set_ylabel('$R_b$')
        ax.set_title(title)
        if file is not None:
            plt.savefig(file)
        if show:
            plt.show()
        return ax

    def conditional(self, on: str = 'a', value: float = 0.0) -> RewardDistribution:
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

        other_name = 'b' if on == 'a' else 'a'

        # condition on the atom {R_on = 0}: the sub-distribution of the other reward there, normalised by its mass.
        # _AtomConditional reuses the full RewardDistribution machinery (de Hoog, adaptive-spline curves, quantile,
        # plotting), like _NestedConditional does for value > 0 -- so both conditional cases share one accurate path.
        if value == 0:
            return _AtomConditional(self, on, f"R_{other_name} | R_{on} = 0")

        # condition on R_on = value > 0: a proper 1D distribution via nested inversion (see _NestedConditional),
        # which reuses the full RewardDistribution machinery (de Hoog, two-pass COS curves, atom, plotting)
        return _NestedConditional(self, on, float(value), f"R_{other_name} | R_{on} = {value:g}")

    def check_total_expectation(self, n_points: int = 8, tol: float = 0.1) -> dict:
        """
        Self-consistency tripwire for the (nested-inversion) conditional path: verify the **law of total expectation**
        ``E[R_other] = E_{R_on}[ E[R_other | R_on ] ]`` for conditioning on each axis, and **log a warning** (per axis)
        when the relative error exceeds ``tol``.

        The conditioning marginal's expectation is split into its atom at 0 (``P(R_on = 0) E[R_other | R_on = 0]``)
        plus an equal-probability midpoint quadrature over its continuous part (``n_points`` conditional means at the
        marginal's ``(p0, 1)`` quantiles -- so the quadrature weights by the marginal density automatically). Uses only
        the conditional **mean** (the reliable first-difference cumulant), so it is bounded but not free (a handful of
        nested inversions per point); call it explicitly rather than on every construction.

        :param n_points: Equal-probability quadrature points over the continuous part of each conditioning marginal.
        :param tol: Relative-error threshold above which a violation is logged.
        :return: ``{'a': rel_err_conditioning_on_a, 'b': rel_err_conditioning_on_b}`` (empty for a self-pair).
        """
        if self._is_diagonal:
            return {}  # R_a == R_b a.s.; the conditional is a point mass, nothing to integrate

        out = {}
        for on, other in (('a', 'b'), ('b', 'a')):
            marg_on, marg_other = self.marginal(on), self.marginal(other)
            lhs = float(marg_other._cumulants()[0])  # E[R_other] from the (reliable) ordinary marginal
            p0 = float(self._atoms['a0' if on == 'a' else 'b0'])

            # continuous part: midpoint rule in probability space over (p0, 1) -> conditional means at those quantiles
            qs = p0 + (1.0 - p0) * (np.arange(n_points) + 0.5) / n_points
            cond_means = [self.conditional(on, float(marg_on.quantile(float(q))))._cumulants()[0] for q in qs]
            rhs = (1.0 - p0) * float(np.mean(cond_means))
            if p0 > 1e-6:  # atom term P(R_on = 0) E[R_other | R_on = 0]
                rhs += p0 * float(self.conditional(on, 0.0)._cumulants()[0])

            rel = abs(rhs - lhs) / max(abs(lhs), 1e-12)
            out[on] = rel
            if Settings.check_inversions and rel > tol:
                self._logger.warning(
                    f"law of total expectation violated conditioning on R_{on}: E[R_{other}]={lhs:.4g} vs "
                    f"E[E[R_{other}|R_{on}]]={rhs:.4g} (rel {rel:.1%} > {tol:.0%}); the conditional inversion may be "
                    f"imprecise here"
                )
        return out


class _Conditional(RewardDistribution):
    """
    Shared base for the 1D conditional distributions (:class:`_AtomConditional`, :class:`_NestedConditional`). Their
    ``lst`` is a nested inversion, so the finite-difference cumulants are unreliable -- the second difference of the
    noisy nested transform makes :meth:`_cumulants` collapse the **variance** to its floor, which would shrink the
    support window (:meth:`_range`) to a point near the mean and truncate the distribution. So a conditional sizes its
    support window by **bracketing the exact CDF** instead: a handful of (exact de Hoog) evaluations, memoised per
    scale. Everything else (de Hoog ``cdf``/``pdf``, the adaptive-grid + monotone-spline curve, quantile, plotting) is
    inherited unchanged.
    """

    def _range(self, scale: float = 12.0) -> float:
        """Support upper end, found by bracketing the exact CDF (overrides the cumulant-based estimate, whose variance
        is unreliable for the nested transform). Memoised per ``scale`` so the repeated callers (quantile, curve fit)
        do not re-bracket."""
        cache = self.__dict__.setdefault('_range_cache', {})
        if scale not in cache:
            cache[scale] = self._range_via_cdf(scale)
        return cache[scale]

    def _range_via_cdf(self, scale: float = 12.0, n_iter: int = 80) -> float:
        """Double ``b`` from a robust seed until the exact CDF ``cdf(b)`` exceeds a generous target probability (more
        generous for larger ``scale``, matching the cumulant-based ``mean + scale*std`` it replaces). The mean
        (first-difference ``_cumulants()[0]``) is reliable and used only as the seed; the variance is not."""
        target = min(1.0 - float(np.exp(-scale)), 1.0 - 1e-6)  # scale=12 -> ~1-1e-6 (full support); scale=4 -> ~0.98
        cdf = self.cdf  # the (per-point de Hoog) CDF function object
        c1 = float(self._cumulants()[0])
        b = max(c1, 1.0 / self._time_scale, 1e-3)
        for _ in range(n_iter):
            if float(cdf(b)) >= target:
                break
            b *= 1.6
        return b


class _AtomConditional(_Conditional):
    """
    The 1D conditional of the *other* reward given ``R_{on} = 0`` -- conditioning on the **atom** ``{R_on = 0}``: the
    sub-distribution of the other reward there, normalised by the atom mass ``P(R_on = 0)``. Its LST is the marginal
    sub-transform restricted to that atom (``Phi(inf, .)`` for ``on='a'``, ``Phi(., inf)`` for ``on='b'``) divided by
    the atom mass, so it plugs straight into the full :class:`RewardDistribution` machinery -- de Hoog ``cdf``/``pdf``,
    the adaptive-grid + monotone-spline curves, quantile and plotting -- exactly like :class:`_NestedConditional` does
    for the ``value > 0`` case, so both conditional cases share one accurate path. The residual atom at 0
    (``P(both = 0) / P(R_on = 0)``) surfaces automatically as ``p0 = lst(inf)``.
    """
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, joint: 'JointRewardDistribution', on: str, label: str = ''):
        atom = joint._atoms['a0' if on == 'a' else 'b0']
        if atom < 1e-9:
            raise ValueError(f"Cannot condition on R_{on} = 0: it has (near) zero probability.")
        big = 1e8
        self._joint = joint
        # the conditional reuses the host's state space / time-scale (its ``lst`` is the marginal sub-transform; there
        # is no own reward to bind, so ``_setup`` is never invoked -- see ``_time_scale``)
        self._host = joint._host
        self.state_space = joint._host.state_space
        self._on = on
        self._atom = atom
        self._sub = (lambda s: joint.lst(big, s)) if on == 'a' else (lambda s: joint.lst(s, big))
        self._logger = logger.getChild(self.__class__.__name__)
        self.label = label

    def lst(self, s: complex) -> complex:
        """Conditional LST: the marginal sub-transform on the atom ``{R_on = 0}``, normalised by the atom mass."""
        return self._sub(s) / self._atom


_STEHFEST_WEIGHTS: dict = {}


def _stehfest_weights(M: int) -> list:
    """Gaver-Stehfest weights ``V_k`` (``k = 1..2M``), cached per order ``M``."""
    if M not in _STEHFEST_WEIGHTS:
        V = []
        for k in range(1, 2 * M + 1):
            s = mp.mpf(0)
            for j in range((k + 1) // 2, min(k, M) + 1):
                s += (mp.mpf(j) ** M * mp.factorial(2 * j) /
                      (mp.factorial(M - j) * mp.factorial(j) * mp.factorial(j - 1) *
                       mp.factorial(k - j) * mp.factorial(2 * j - k)))
            V.append((-1) ** (k + M) * s)
        _STEHFEST_WEIGHTS[M] = V
    return _STEHFEST_WEIGHTS[M]


def _stehfest_invert(transform, t: float, M: int = 8) -> complex:
    """
    Gaver-Stehfest numerical Laplace inversion of ``transform`` at ``t``, evaluated in extended precision. Unlike
    mpmath's de Hoog / Talbot (which assume a real-valued transform), this uses purely *real* nodes
    ``s = k ln2 / t`` with real weights, so it inverts a **complex-valued** transform to a complex result — needed
    for the nested conditional, whose inner transform is complex when the outer argument is (the COS frequencies).
    """
    a = mp.log(2) / t
    with mp.workdps(max(mp.mp.dps, 3 * M + 10)):
        return complex(a * mp.fsum(Vk * mp.mpc(transform(complex(k * a)))
                                   for k, Vk in enumerate(_stehfest_weights(M), start=1)))


class _NestedConditional(_Conditional):
    """
    The 1D conditional distribution of one reward given ``R_{on} = value`` (``value > 0``), built by **nested
    inversion**: invert the conditioned dimension at ``value`` to get the other reward's conditional Laplace
    transform, then reuse the full :class:`RewardDistribution` machinery (de Hoog ``cdf``/``pdf``, two-pass COS
    plotting curves, atom handling, quantile, plotting).

    The conditional LST is ``phi(s) = G(s) / G(0)`` where ``G(s) = L^{-1}_{u -> value}[ u -> Phi(s, u) ]`` is the
    inner (Gaver-Stehfest) inversion of the joint transform ``Phi`` along the conditioned axis at ``value``, and
    ``G(0)`` is the marginal density of the conditioning reward there (the normaliser). This resolves the conditional
    exactly, unlike the coarse 2D-cosine slice.
    """
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, joint: 'JointRewardDistribution', on: str, value: float, label: str = ''):
        self._joint = joint
        # the conditional reuses the host's state space / time-scale (its ``lst`` is the nested transform; there is no
        # own reward to bind, so ``_setup`` is never invoked -- see ``_time_scale``)
        self._host = joint._host
        self.state_space = joint._host.state_space
        self._on = on
        self._value = float(value)
        self._stehfest_M = 8
        self._logger = logger.getChild(self.__class__.__name__)
        self.label = label

        g0 = self._G(0.0).real  # = marginal density of the conditioning reward at ``value`` (the normaliser)
        if g0 <= 1e-300:
            raise ValueError(f"The marginal density at R_{on} = {value} is zero; cannot condition there.")
        self._G0 = g0

    def _G(self, s: complex) -> complex:
        """``G(s) = L^{-1}_{u -> value}[ u -> Phi(s, u) ](value)`` (inner inversion along the conditioned axis)."""
        if self._on == 'b':
            f = lambda u: self._joint.lst(s, u)
        else:
            f = lambda u: self._joint.lst(u, s)
        return _stehfest_invert(f, self._value, self._stehfest_M)

    def lst(self, s: complex) -> complex:
        """The conditional Laplace-Stieltjes transform ``phi(s) = G(s) / G(0)``."""
        return self._G(complex(s)) / self._G0
