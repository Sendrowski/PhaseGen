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
from math import comb, factorial
from typing import TYPE_CHECKING, Optional

import mpmath as mp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.integrate import simpson

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

    def __init__(self, dist: 'PhaseTypeDistribution', reward: Reward = None) -> None:
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
    def _setup(self) -> dict:
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
        with no state-space reward to bind -- can scale their cumulant/quantile step without invoking ``_setup``.

        Deliberately *not* defaulted. Every flavour binds ``_host`` in its constructor, so a missing one means the
        attribute is being read too early -- and a default of 1.0 would answer that with a plausible number rather
        than an error, silently unscaling every rate-scaled quantity downstream (the ``s -> inf`` atom probe, the
        cumulant step, the inversion contour). ``_AtomConditional`` did exactly this and was wrong by 0.75% on a
        small-N demography for as long as it existed.
        """
        return self._host._time_scale

    @property
    def _s_inf(self) -> float:
        """
        The ``s -> inf`` probe used for the atom ``P(R = 0) = phi(inf)`` (and the axis atoms of a joint).

        Scaled by the inversion time scale, *not* a fixed number: the transform decays on the scale of the rates,
        which go like ``1 / tau``, so a hard-coded ``s`` is only large in the ``tau ~ 1`` regime. On a small-N
        demography (``tau = 1e-6``) ``phi(1e8)`` has not decayed at all and reports a 1.9% atom for a doubleton bin
        whose atom is exactly 0 (every binary tree has a cherry); it needs ``s ~ 1e12`` to converge. Probing at
        ``1e8 / tau`` keeps ``s`` the same large multiple of the rate scale in every regime.
        """
        return 1e8 / self._time_scale

    @cached_property
    def mean(self) -> float:
        """Mean ``E[R]`` of the accumulated reward (the exact first moment from the moment engine). The conditional
        flavours carry no state-space reward, so they fall back to the (reliable) LST cumulant mean."""
        reward = getattr(self, 'reward', None)
        if reward is None:
            return float(self._cumulants()[0])
        # go through the moment engine directly: a spectrum host overrides ``moment`` to return a whole SFS, which
        # would break ``float()`` for a single-bin reward
        return float(MomentEvaluator.moment(self._host, k=1, rewards=(reward,), center=False))

    @cached_property
    def var(self) -> float:
        """Variance of the accumulated reward (the exact second central moment, from the moment engine). A conditional
        carries no state-space reward and takes a different route (see :attr:`ConditionalRewardDistribution.var`)."""
        return float(MomentEvaluator.moment(self._host, k=2, rewards=(self.reward, self.reward), center=True))

    @cached_property
    def std(self) -> float:
        """Standard deviation of the accumulated reward."""
        return self.var ** 0.5

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

        The inversion runs in **tau-scaled time**, which makes it scale-invariant. For any transform ``G`` with
        inverse ``g`` (a density or a CDF alike),

            g(t) = (1 / tau) * L^-1[ sigma -> G(sigma / tau) ](t / tau),

        so the contour nodes and the inverted value both stay O(1) however large the demography's rewards are.
        Inverting at the raw ``t`` instead lets de Hoog's accuracy drift with the units: on a large-N demography
        (rewards ~1e7) it put the *reference* conditional mean 16% off at the 0.75 conditioning quantile, while the
        same demography rescaled to N ~ 1e-6 -- the identical model in different units -- was accurate to 1%.
        """
        if t <= 0:
            return 0.0

        tau = self._time_scale

        def F(s) -> 'mp.mpc':
            val = transform(complex(s) / tau)
            return mp.mpc(val.real, val.imag)

        return float(mp.invertlaplace(F, t / tau, method='dehoog', degree=Settings.dehoog_degree)) / tau

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


def _lst_taylor_from_shift(shift: np.ndarray, deriv: np.ndarray, alpha: np.ndarray, T_epochs, sparse: bool,
                           perm=_AUTO_PERM, order: int = 2) -> list:
    """
    Taylor coefficients in ``eps`` of :func:`_lst_from_shift` evaluated at the shift ``shift + eps * deriv``, i.e.
    ``[Phi_0, Phi_1, ..., Phi_order]`` with ``Phi(eps) = sum_k Phi_k eps^k``. So the ``k``-th derivative in ``eps`` is
    ``k! Phi_k`` — **exactly**, with no finite difference.

    Every quantity is evaluated in the truncated polynomial ring ``R[eps] / (eps^{order+1})``, whose elements are
    represented by their coefficient tuples. The matrix exponential and the linear solve both commute with the ring
    homomorphism, so a matrix over the ring is the block-Toeplitz upper-triangular matrix with the coefficient blocks
    on its diagonals, and:

    * a finite epoch propagates the coefficient vector by ``expm`` of that block matrix (its blocks *are* the Taylor
      coefficients of the epoch's propagator);
    * the final epoch's ``x(eps) = (A + eps diag(deriv))^{-1} e`` is obtained by back-substitution in the ring,
      ``x_0 = A^{-1} e`` and ``x_k = -A^{-1} (deriv * x_{k-1})`` — the same LU, reused.

    Differencing ``Phi`` instead is not viable: the derivative is fed to a de Hoog inversion whose QD recurrence
    amplifies the ``eps_mach / h`` residue of the difference, which made the inverted value swing by 15% between
    neighbouring steps ``h``.
    """
    nt, k = len(alpha), order + 1
    n_aug = nt + 1

    # row vector over the ring: blocks [v_0, ..., v_order]
    vec = np.zeros((k, n_aug), dtype=complex)
    vec[0] = np.concatenate([alpha, [0.0]])

    for T, t0, t1 in T_epochs[:-1]:
        dt = t1 - t0
        Q = np.zeros((n_aug, n_aug), dtype=complex)
        Q[:nt, :nt] = (T.toarray() if sp.issparse(T) else np.asarray(T)) - np.diag(shift)
        Q[:nt, nt] = _exit_rates(T)
        D = np.zeros((n_aug, n_aug), dtype=complex)
        D[:nt, :nt] = -np.diag(deriv)  # d/deps of the generator: the shift enters as -diag(.)

        M = np.zeros((k * n_aug, k * n_aug), dtype=complex)
        for i in range(k):
            M[i * n_aug:(i + 1) * n_aug, i * n_aug:(i + 1) * n_aug] = Q * dt
            if i + 1 < k:
                M[i * n_aug:(i + 1) * n_aug, (i + 1) * n_aug:(i + 2) * n_aug] = D * dt
        vec = (vec.reshape(1, -1) @ sla.expm(M)).reshape(k, n_aug)

    Tm = T_epochs[-1][0]
    A = (sp.diags(shift) if sparse else np.diag(shift)) - Tm
    solve = MomentEvaluator._lu_solver(A, sparse, perm)

    xs = [solve(_exit_rates(Tm))]
    for _ in range(1, k):
        xs.append(-solve(deriv * xs[-1]))

    a, c = vec[:, :nt], vec[:, nt]
    return [complex(c[i] + sum(a[j] @ xs[i - j] for j in range(i + 1))) for i in range(k)]


class JointRewardDistribution(CallableDistributionFunctions):
    """
    Joint distribution of two accumulated rewards ``R_a = int r_a(X_s) ds`` and ``R_b = int r_b(X_s) ds`` to
    absorption, the distributional object behind a cross-moment ``E[R_a R_b]`` (the within-tree 2-SFS / the
    two-locus SFS). The joint Laplace-Stieltjes transform uses the *combined* generator shift,

        Phi(s_a, s_b) = E[e^{-s_a R_a - s_b R_b}] = _lst_from_shift(s_a r_a + s_b r_b, ...),

    so it is the same machinery as the univariate :class:`RewardDistribution` with a two-parameter shift, and is
    multi-epoch-native. Setting one argument to zero recovers a marginal; mixed derivatives at the origin recover
    the cross-moments. The joint CDF/PDF (2D inversion) and the product distribution are views built on top.

    The 2D inversion is the **Fourier-cosine expansion** (:meth:`_cc_box`), which is the only 2D method: it inverts
    both axes at once from a single coefficient matrix, so a whole grid costs one solve set. The nested per-point
    alternatives were dropped. Nested **de Hoog** was no more accurate than the cosine grid on realistic demographies
    (both within a few percent of msprime at the quantile corners) while costing ~1 s per point, which put it out of
    reach of the scenario suite. Nested **Euler** (which the 1D conditional does need, see :class:`_NestedConditional`)
    is accurate but likewise per-point, and equally untestable at grid scale. Cosine's known weakness is the
    steep near-origin rise, mitigated by the window scale (:attr:`_cos2d_window_scale`).
    """
    @property
    def _time_scale(self) -> float:
        """The inversion time scale of the host distribution. Not defaulted, for the reason given on
        :attr:`RewardDistribution._time_scale`: a fallback of 1.0 hides an unset host behind a plausible number."""
        return self._host._time_scale

    @property
    def _s_inf(self) -> float:
        """
        The ``s -> inf`` probe used for the atom ``P(R = 0) = phi(inf)`` (and the axis atoms of a joint).

        Scaled by the inversion time scale, *not* a fixed number: the transform decays on the scale of the rates,
        which go like ``1 / tau``, so a hard-coded ``s`` is only large in the ``tau ~ 1`` regime. On a small-N
        demography (``tau = 1e-6``) ``phi(1e8)`` has not decayed at all and reports a 1.9% atom for a doubleton bin
        whose atom is exactly 0 (every binary tree has a cherry); it needs ``s ~ 1e12`` to converge. Probing at
        ``1e8 / tau`` keeps ``s`` the same large multiple of the rate scale in every regime.
        """
        return 1e8 / self._time_scale

    #: bivariate function-object flavours (built by the :class:`CallableDistributionFunctions` mixin, passing the
    #: ``plot_surface`` callback); a joint has no quantile (a 2D quantile is not well-defined)
    _pdf_function = JointDensity
    _cdf_function = JointCDF
    _quantile_function = None

    #: Number of cosine terms per axis for the 2D Fourier-cosine joint density (:attr:`_cos2d`). The cost is the square
    #: of this (an ``n x n`` coefficient matrix of joint-LST evaluations), so it is smaller than the 1D term count; at
    #: 128 the heatmap/surface ringing is ~0.5% of the peak (vs ~2.5% at 64).
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

    def __init__(self, dist: 'PhaseTypeDistribution', reward_a: Reward, reward_b: Reward) -> None:
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
    def _setup(self) -> dict:
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

    def lst_taylor(self, s: complex, on: str = 'a', order: int = 2) -> list:
        """
        The Taylor coefficients of the joint LST in *one* argument at 0, with the other held at ``s``:
        ``[Phi_k]`` such that ``Phi(s, eps) = sum_k Phi_k eps^k`` (for ``on = 'a'``; mirrored for ``'b'``). The
        ``k``-th derivative in the free argument is therefore ``k! Phi_k``, exactly (see
        :func:`_lst_taylor_from_shift`).

        :param s: The value at which the conditioned argument (``on``) is held.
        :param on: Which argument is held at ``s``, ``'a'`` or ``'b'``; the coefficients are in the other one.
        :param order: Highest coefficient to return.
        :return: ``[Phi_0, ..., Phi_order]``.
        """
        st = self._setup
        tau = st['tau']
        r_on, r_other = (st['ra'], st['rb']) if on == 'a' else (st['rb'], st['ra'])

        # differentiate in the *tau-scaled* free argument and put the tau^j back afterwards, which is exact (it is a
        # constant factor per order). Differentiating in the unscaled one instead puts blocks of magnitude 1, tau and
        # tau^2 into the same augmented matrix -- 1, 1e7 and 1e14 on a large-N demography -- and ``expm``'s
        # scaling-and-squaring, driven by the largest of them, then costs the O(1) block its precision
        coeffs = _lst_taylor_from_shift((s * tau) * r_on, r_other, st['alpha'], st['T_epochs'], st['sparse'],
                                        st['lu_perm'], order)
        return [c * tau ** j for j, c in enumerate(coeffs)]

    def lst_batch(self, s_a, s_b) -> np.ndarray:
        """The joint LST over a *vector* of nodes in one argument (the other held scalar), evaluated as one batch.

        The inner inversion of :class:`_NestedConditional` always has its whole node set to hand, and batching the
        per-epoch assembly and matrix exponential across it is ~2x (see :func:`_lst_from_shift_batch`)."""
        st = self._setup
        tau = st['tau']
        s_a, s_b = np.atleast_1d(s_a), np.atleast_1d(s_b)
        shifts = (np.outer(s_a * tau, st['ra']) + np.outer(s_b * tau, st['rb'])).astype(complex)
        return _lst_from_shift_batch(shifts, st['alpha'], st['T_epochs'], st['sparse'], st['lu_perm'])

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

    @cached_property
    def mean(self) -> np.ndarray:
        """The pair of marginal means ``(E[R_a], E[R_b])``."""
        return np.array([self.moment(1, 0), self.moment(0, 1)])

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
        big = self._s_inf
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
        big = self._s_inf
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
        n_terms, scale, big = self._cos2d_terms, self._cos2d_window_scale, self._s_inf
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
        big = self._s_inf
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
                "a finer grid or more terms may be needed.", err * 100,
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

    def _cdf_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Joint CDF on the grid ``xs x ys``: the axis atoms ``P(R_a = 0, R_b <= y)`` and ``P(R_b = 0, R_a <= x)``
        (1D Fourier-cosine inversions, one per grid line) plus the continuous box integral (:meth:`_cc_box`)."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        g_a, g_b = self._cos_axis('a', ys), self._cos_axis('b', xs)
        return g_b[:, None] + g_a[None, :] - self._atoms['both0'] + self._cc_box(xs, ys)

    def conditional(self, on: str = 'a', value: float = 0.0) -> 'ConditionalRewardDistribution':
        """
        The 1D conditional distribution of the *other* reward given ``R_{on} = value``.

        For ``value > 0`` this is the continuous conditional density -- a slice of the 2D density at the conditioning
        value, normalized by the marginal density there -- plus the atom ``P(R_other = 0 | R_{on} = value)`` (non-zero
        only when the other bin can be empty). For ``value = 0`` it conditions on the atom event ``{R_{on} = 0}``
        (which must have positive probability).

        The result is a :class:`ConditionalRewardDistribution`: callable and plottable like any other 1D distribution
        (``cdf`` / ``pdf`` / ``quantile`` / ``mean`` / ``var``), but obtained differently, and -- for ``value > 0`` --
        resting on a nested inversion. See that class for the caveats.

        :param on: Which reward to condition on, ``'a'`` or ``'b'``.
        :param value: The conditioning value (``>= 0``).
        :return: The conditional distribution of the other reward.
        :raises NotImplementedError: If the pair is a self-pair (``R_a == R_b`` a.s., so the conditional is a point
            mass at ``value``).
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        if value is None or value < 0:
            raise ValueError("`value` must be non-negative.")
        if self._is_diagonal:
            raise NotImplementedError("The conditional of a self-pair is a point mass at `value` (R_a = R_b a.s.).")

        other_name = 'b' if on == 'a' else 'a'

        # condition on the atom {R_on = 0}: the sub-distribution of the other reward there, normalised by its mass.
        # _AtomConditional reuses the full RewardDistribution machinery (the cosine cdf/pdf/quantile, de Hoog,
        # plotting), like _NestedConditional does for value > 0 -- so both conditional cases share one accurate path.
        if value == 0:
            return _AtomConditional(self, on, f"R_{other_name} | R_{on} = 0")

        # condition on R_on = value > 0: a proper 1D distribution via nested inversion (see _NestedConditional),
        # which reuses the full RewardDistribution machinery (de Hoog, two-pass COS curves, atom, plotting)
        return _NestedConditional(self, on, float(value), f"R_{other_name} | R_{on} = {value:g}")

    def check_total_expectation(self, n_points: int = 32, tol: float = 0.01) -> dict:
        """
        Self-consistency tripwire for the (nested-inversion) conditional path: verify the **law of total expectation**
        ``E[R_other] = E_{R_on}[ E[R_other | R_on ] ]`` for conditioning on each axis, and **log a warning** (per axis)
        when the relative error exceeds ``tol``.

        The conditioning marginal's expectation is split into its atom at 0 (``P(R_on = 0) E[R_other | R_on = 0]``)
        plus a Gauss-Legendre integral over its continuous part, taken in probability space: substituting
        ``u = F_on(v)`` turns ``INT E[R_other|v] f_on(v) dv`` into ``INT_0^1 E[R_other | F_on^{-1}(u)] du``, so the
        marginal density is absorbed into the measure and only quantiles are needed.

        ``n_points`` must not be cut: the integrand grows without bound as ``u -> 1``, so it is the *quadrature*, not
        the conditional, that limits this check (with a perfect integrand it is still 8% off at 8 nodes, 0.2% at 32).
        Cost is a handful of nested inversions per node; call it explicitly rather than on every construction.

        :param n_points: Gauss-Legendre nodes for the integral over the continuous part of each conditioning marginal.
        :param tol: Relative-error threshold above which a violation is logged.
        :return: ``{'a': rel_err_conditioning_on_a, 'b': rel_err_conditioning_on_b}`` (empty for a self-pair).
        """
        if self._is_diagonal:
            return {}  # R_a == R_b a.s.; the conditional is a point mass, nothing to integrate

        x, w = np.polynomial.legendre.leggauss(n_points)
        us, ws = 0.5 * (x + 1.0), 0.5 * w  # map [-1, 1] -> [0, 1]

        out = {}
        for on, other in (('a', 'b'), ('b', 'a')):
            marg_on, marg_other = self.marginal(on), self.marginal(other)
            lhs = float(marg_other._cumulants()[0])  # E[R_other] from the (reliable) ordinary marginal
            p0 = float(self._atoms['a0' if on == 'a' else 'b0'])

            rhs, n_refused = 0.0, 0
            for u, weight in zip(us, ws):
                v = float(marg_on.quantile(p0 + (1.0 - p0) * float(u)))  # continuous part spans quantiles (p0, 1)
                try:
                    rhs += (1.0 - p0) * weight * float(self.conditional(on, v)._cumulants()[0])
                except ValueError:
                    n_refused += 1  # dropping this node's contribution shows up as a deficit in ``rhs``
            if p0 > 1e-6:  # atom term P(R_on = 0) E[R_other | R_on = 0]
                rhs += p0 * float(self.conditional(on, 0.0)._cumulants()[0])

            rel = abs(rhs - lhs) / max(abs(lhs), 1e-12)
            out[on] = rel
            if Settings.check_inversions and (rel > tol or n_refused):
                refused = f", and {n_refused}/{n_points} conditionals could not be built" if n_refused else ""
                self._logger.warning(
                    f"law of total expectation violated conditioning on R_{on}: E[R_{other}]={lhs:.4g} vs "
                    f"E[E[R_{other}|R_{on}]]={rhs:.4g} (rel {rel:.1%} > {tol:.0%}){refused}; the conditional inversion "
                    f"may be imprecise here"
                )
        return out

    def check_total_probability(self, n_points: int = 8, n_y: int = 15, tol: float = 0.01) -> dict:
        """
        Self-consistency tripwire for the (nested-inversion) conditional path: verify the **law of total probability**
        ``F_other(y) = E_{R_on}[ P(R_other <= y | R_on) ]`` for conditioning on each axis, and **log a warning** (per
        axis) when the sup-norm deviation over ``y`` exceeds ``tol``.

        **Not wired into the scenario suite, and not recommended as a tripwire:** unlike :meth:`check_total_expectation`
        it does not converge under refinement (it plateaus near 0.01 and then drifts up: 0.037 / 0.0096 / 0.0099 /
        0.0126 at 8 / 16 / 32 / 64 nodes) while costing 20-160 s, so a reading cannot be attributed to the conditional
        rather than to the check itself. Use :meth:`check_conditional_moments`, which is exact, pointwise and ~1 s.

        The companion :meth:`check_total_expectation` tests only the first moment, and moments cancel: a conditional
        can be several percent wrong pointwise and still integrate to the right mean, because the errors above and
        below the mean offset. This tests the whole law instead, at every ``y``, and so catches shape errors that the
        moment check cannot see.

        The conditioning marginal is integrated out in **probability space**: substituting ``u = F_on(v)`` turns
        ``INT F(y|v) f_on(v) dv`` into ``INT_0^1 F(y | F_on^{-1}(u)) du``, so the marginal density is absorbed into the
        measure and only quantiles are needed. The atom at 0 (``P(R_on = 0) F(y | R_on = 0)``) is added separately.
        Both sides are CDFs, so each already carries its own atom at ``R_other = 0`` and the identity holds without a
        correction. A conditional that *refuses* to build (a non-positive normaliser at an interior quantile) is itself
        a violation and is reported as such.

        The ``u``-integrand is smooth, so the nodes are **Gauss-Legendre** rather than the equal-probability midpoints
        of :meth:`check_total_expectation`. This matters: the midpoint rule converges as ``O(n^-2)``, and against a
        closed-form copula its discretisation error at ``n_points = 8`` is 0.004 to 0.020 (rising with the coupling
        between the two rewards) *for an exact conditional* -- the same size as the inversion error being hunted, which
        would make the check measure its own quadrature and read strongly-coupled reward pairs as broken. Gauss-Legendre
        at ``n_points = 12`` holds that floor below 5e-4, roughly ``tol / 20``.

        Note that the quadrature weights each conditional by ``f_on(v)``, so a conditional that is only wrong far out
        in the tail contributes little and may pass. This bounds bulk error; it is not a tail certificate.

        **This is expensive**: ``n_points`` nested inversions per axis, each building a CDF curve. On a multi-epoch
        demography a single conditional costs tens of seconds, so the whole check runs in minutes -- and it is slow
        precisely *because* it is useful, since the outermost quadrature nodes sit at the ~99th percentile of the
        conditioning marginal, exactly where the inner inversion has to refine hardest (see
        :meth:`_NestedConditional._calibrate`). Call it deliberately when a conditional looks suspect, never on every
        construction. ``n_points = 8`` holds the Gauss-Legendre quadrature floor at ~2.6e-3, comfortably inside
        ``tol``, while keeping the node count down.

        :param n_points: Gauss-Legendre nodes for the integral over the continuous part of each conditioning marginal.
        :param n_y: Evaluation points for the sup-norm, at evenly spaced quantiles of the other reward's marginal.
        :param tol: Sup-norm deviation (an absolute probability) above which a violation is logged.
        :return: ``{'a': sup_dev_conditioning_on_a, 'b': sup_dev_conditioning_on_b}`` (empty for a self-pair).
        """
        if self._is_diagonal:
            return {}  # R_a == R_b a.s.; the conditional is a point mass, nothing to integrate

        x, w = np.polynomial.legendre.leggauss(n_points)
        us, ws = 0.5 * (x + 1.0), 0.5 * w  # map [-1, 1] -> [0, 1]

        out = {}
        for on, other in (('a', 'b'), ('b', 'a')):
            marg_on, marg_other = self.marginal(on), self.marginal(other)
            p0 = float(self._atoms['a0' if on == 'a' else 'b0'])

            # evaluate the sup-norm where the other reward actually lives, not on an arbitrary grid
            ys = np.array([float(marg_other.quantile(float(p))) for p in np.linspace(0.05, 0.95, n_y)])
            lhs = np.asarray(marg_other.cdf(ys), dtype=float)  # from the (reliable) ordinary marginal

            rhs, n_refused = np.zeros(n_y), 0
            for u, weight in zip(us, ws):
                v = float(marg_on.quantile(p0 + (1.0 - p0) * float(u)))  # continuous part spans quantiles (p0, 1)
                try:
                    cond = self.conditional(on, v)
                except ValueError:
                    n_refused += 1  # dropping this node's contribution shows up as a deficit in ``rhs``
                    continue
                # the CURVE, not ``cond.cdf(ys)``: a per-point de Hoog costs (2 dehoog_degree + 1) outer times
                # (2 (N0 + m) + 1) inner ``lst`` solves for EVERY y, while the COS curve shares one set of phi
                # evaluations across the whole grid -- ~11x here, at no measurable accuracy cost
                rhs += (1.0 - p0) * weight * np.asarray(cond.cdf(ys), dtype=float)
            if p0 > 1e-6:  # atom term P(R_on = 0) F(y | R_on = 0)
                rhs = rhs + p0 * np.asarray(self.conditional(on, 0.0).cdf(ys), dtype=float)

            dev = float(np.max(np.abs(rhs - lhs)))
            out[on] = dev
            if Settings.check_inversions and (dev > tol or n_refused):
                refused = f", and {n_refused}/{n_points} conditionals could not be built" if n_refused else ""
                self._logger.warning(
                    f"law of total probability violated conditioning on R_{on}: "
                    f"sup_y |F_{other}(y) - E[F_{other}(y|R_{on})]| = {dev:.3g} > {tol:.3g}{refused}; the conditional "
                    f"inversion may be imprecise here"
                )
        return out

    #: Gauss-Legendre nodes used to average a conditional over a window (:meth:`window_average`). The conditional
    #: varies smoothly with the conditioning value, so a handful resolves the average; each node costs a conditional.
    _WINDOW_QUAD_NODES = 8

    def window_average(self, statistic, on: str = 'a', value: float = 0.0, half_width: float = 0.0,
                       n_nodes: int = None) -> np.ndarray:
        """
        A statistic of the conditional, averaged over the conditioning window ``[value - half_width, value +
        half_width]`` and weighted by the conditioning marginal's density::

            E[ g(R_other) | R_on in W ]  =  int_W f_on(u) E[ g(R_other) | R_on = u ] du  /  int_W f_on(u) du

        This is what a **sample** of the conditional actually estimates. No replicate lands exactly on ``value``, so
        an empirical conditional keeps those in a window around it, and what it then measures is not the conditional
        at ``value`` but this window average. Comparing a sampled conditional against the *pointwise* one therefore
        carries an ``O(half_width)`` discrepancy that is a property of the window and no part of phasegen's error,
        and that no number of replicates removes.

        Averaging phasegen's own conditional over the same window puts both sides on the same functional, so the
        window bias cancels identically and the sample becomes a genuine ground truth for the conditional -- the only
        one it has away from the atom. It also *gains* power: with the bias gone the window may be widened to keep
        more replicates, which lowers the standard error instead of trading it against bias.

        :param statistic: Callable taking a :class:`ConditionalRewardDistribution` and returning a scalar or an array
            (e.g. ``lambda c: c.mean``, or ``lambda c: c.cdf(ys)``).
        :param on: Axis to condition on, ``'a'`` or ``'b'``.
        :param value: Centre of the conditioning window.
        :param half_width: Half-width of the window, in the conditioning reward's units.
        :param n_nodes: Quadrature nodes; defaults to :attr:`_WINDOW_QUAD_NODES`.
        :return: The window average of ``statistic``, shaped as ``statistic`` returns.
        :raises ValueError: If the window reaches ``0``, where it would take in the conditioning atom and stop being
            a purely continuous average.
        """
        n_nodes = self._WINDOW_QUAD_NODES if n_nodes is None else n_nodes
        lo, hi = value - half_width, value + half_width

        # a window reaching 0 would include the atom {R_on = 0}, a positive-probability event that no density weight
        # can represent: the sample's window mean would then mix the atom conditional in, and the two sides would
        # again be measuring different things
        if lo <= 0.0:
            raise ValueError(
                f"The conditioning window [{lo:g}, {hi:g}] reaches 0, where R_{on} has an atom. Narrow the window or "
                f"condition further from the origin."
            )

        x, w = np.polynomial.legendre.leggauss(n_nodes)
        u = 0.5 * (hi - lo) * (x + 1.0) + lo

        marg = self.marginal(on)
        weights = w * np.array([float(marg.pdf(float(ui))) for ui in u])
        values = np.array([np.atleast_1d(np.asarray(statistic(self.conditional(on, float(ui))), dtype=float))
                           for ui in u])

        return (weights[:, None] * values).sum(axis=0) / weights.sum()

    #: Quantile span of the conditioning marginal over which :meth:`check_conditional_moments` places its points.
    #: Runs nearly the whole law: the conditional is resolved just as well at ``u = 0.01`` and ``u = 0.99`` as in the
    #: middle, and a span confined to the bulk tests only the region where nothing can go wrong.
    _COND_CHECK_SPAN = (0.01, 0.99)

    #: Floor of the relative-error denominator in :meth:`check_conditional_moments`, as a fraction of the *unconditional*
    #: mean ``E[R_other]``. Deep in the conditioning tail the conditional mean goes to ~0 (for ``n = 4``, conditioning on
    #: a huge doubleton length forces the balanced topology, which has no 3-leaf clade at all, so ``E[R_3 | .] -> 0``), and
    #: a plain relative error there divides two ~1e-6 numbers and reports pure noise as a 4% failure. Scaling against the
    #: unconditional mean asks the honest question -- is the discrepancy *material* -- and is what lets the span reach the
    #: tail at all.
    _COND_CHECK_FLOOR = 0.01

    def check_conditional_moments(self, n_points: int = 5, tol: float = 0.02, quantiles: 'Sequence[float]' = None,
                                  curves: int = 0) -> dict:
        """
        Self-consistency tripwire for the (nested-inversion) conditional path: compare each conditional's **mean**
        against the exact one from :meth:`ConditionalRewardDistribution._raw_moments`, and **log a warning** (per
        axis) when the worst scaled error over the conditioning points exceeds ``tol``.

        Where :meth:`check_total_expectation` integrates the conditional back out and so only tests it *on average*
        over the conditioning axis (errors at different ``v`` can cancel, and its own quadrature sets the floor), this
        pins it **pointwise**, at each ``v`` separately, against a reference that shares no code with the nested
        inversion and no quadrature. It is both stronger and several times cheaper, so it is the check to reach for --
        but only where the reference is sound: :meth:`ConditionalRewardDistribution._raw_moments` is exact on a single
        epoch and carries de Hoog's ~1% error beyond it, so a multi-epoch tolerance has to be set against the
        *reference's* floor, and on an extreme bottleneck (where the reference is off by tens of percent) the tower
        check is the only option.

        The mean under test is the conditional's own (the cumulant of its nested transform), so this asserts the
        transform, not the cosine ``cdf`` / ``pdf`` grid built on top of it; ``curves`` draws those for inspection.

        The conditioning points span :attr:`_COND_CHECK_SPAN` in quantile space of the conditioning marginal (so they
        mean the same thing across demographies), and the error is scaled by :attr:`_COND_CHECK_FLOOR`. Pass
        ``quantiles`` to target specific conditioning values instead.

        :param n_points: Conditioning values per axis, spread over :attr:`_COND_CHECK_SPAN` by quantile. Ignored when
            ``quantiles`` is given.
        :param tol: Error threshold above which a violation is logged.
        :param quantiles: Explicit conditioning quantiles of the conditioning marginal, in ``(0, 1)``.
        :param curves: Number of conditioning values per axis at which to also evaluate the conditional **density**,
            for :attr:`conditional_densities` (and so the plots). Nothing is asserted on them; each one builds a
            cosine grid the mean does not need (~1 s), which is why they are off by default.
        :return: ``{'a': worst_err_conditioning_on_a, 'b': ...}`` (empty for a self-pair).
        """
        us = np.linspace(*self._COND_CHECK_SPAN, n_points) if quantiles is None else np.asarray(quantiles, float)
        if self._is_diagonal:
            return {}  # R_a == R_b a.s.; the conditional is a point mass

        out = {}
        #: per-axis ``(quantiles, exact, nested, errors)`` series of the last run, for the comparison plots. The
        #: errors are the *scaled* ones asserted on, so a plot shows the same metric the result line reports.
        self.conditional_moment_curves = {}

        #: per-axis ``[(quantile, value, ys, density)]`` of the last run, when ``curves`` asked for them.
        self.conditional_densities = {}

        for on, other in (('a', 'b'), ('b', 'a')):
            marg_on = self.marginal(on)
            p0 = float(self._atoms['a0' if on == 'a' else 'b0'])
            floor = self._COND_CHECK_FLOOR * abs(float(self.marginal(other).mean))

            errs, refused = [], []
            kept, exacts, nesteds, conds = [], [], [], []
            for u in us:
                v = float(marg_on.quantile(p0 + (1.0 - p0) * float(u)))
                try:
                    cond = self.conditional(on, v)
                    exact = cond._raw_moments(k=1)[0]
                    got = float(cond.mean)
                except ValueError:
                    refused.append(float(u))
                    continue
                errs.append(abs(got - exact) / max(abs(exact), floor, 1e-12))
                kept.append(float(u))
                exacts.append(exact)
                nesteds.append(got)
                conds.append((v, cond))
            self.conditional_moment_curves[on] = (np.array(kept), np.array(exacts), np.array(nesteds),
                                                  np.array(errs))
            if curves and kept:
                # spread the drawn densities over the kept conditioning points rather than taking the first few
                idx = np.unique(np.linspace(0, len(kept) - 1, min(curves, len(kept))).round().astype(int))
                self.conditional_densities[on] = [
                    (kept[i], conds[i][0], *self._density_curve(conds[i][1])) for i in idx
                ]

            out[on] = self._verdict(on, errs, refused, tol, 'conditional mean', 'the exact derivative identity')
        return out

    def _verdict(self, on: str, errs: list, refused: list, tol: float, what: str, against: str) -> float:
        """
        The worst error of one conditioning axis, treating a conditioning point that could **not** be built as a
        failure (``inf``) rather than dropping it.

        A check must not thin its own question set: silently skipping the points that refuse, and reporting the worst
        of whatever survived, means a check that resolved *nothing* reports a perfect score (``max`` of an empty set),
        which is exactly what every large-N scenario did. Where a demography genuinely cannot resolve a conditioning
        value, the config has to say so by narrowing ``quantiles`` -- visibly, in the config.
        """
        if refused:
            if Settings.check_inversions:
                self._logger.warning(
                    f"{len(refused)}/{len(errs) + len(refused)} conditionals on R_{on} could not be built (at the "
                    f"conditioning quantiles {[f'{u:.2f}' for u in refused]}), so {what} cannot be checked there; "
                    f"narrow the checked quantiles if this demography truly cannot resolve them"
                )
            return float('inf')

        if not errs:
            return float('inf')

        rel = float(np.max(errs))
        if Settings.check_inversions and rel > tol:
            self._logger.warning(
                f"{what} disagrees with {against} conditioning on R_{on}: worst relative error {rel:.2%} over "
                f"{len(errs)} points > {tol:.2%}; the conditional may be imprecise here"
            )
        return rel

    def check_conditional_grid_moments(self, n_points: int = 3, tol: float = 0.02, k: int = 2,
                                       quantiles: 'Sequence[float]' = None) -> dict:
        """
        The distribution-level counterpart of :meth:`check_conditional_moments`: compare the raw moments obtained by
        **integrating each conditional's CDF grid** against the exact ones from
        :meth:`ConditionalRewardDistribution._raw_moments`, and **log a warning** (per axis) when the worst scaled
        error over the conditioning points and orders exceeds ``tol``.

        This is the only check that exercises the cosine ``cdf`` / ``pdf`` layer. :meth:`check_conditional_moments`
        cannot: the mean it tests is a cumulant of the *transform*, so a truncation or a residual ripple in the grid
        would sail straight past it. Here the grid is the thing being integrated, and order 2 sees the tail long
        before the mean does.

        The moments come from the survival function, ``E[R^j] = int_0^b j y^(j-1) (1 - F(y)) dy``, rather than from
        ``int y^j f(y) dy``: the CDF is what the cosine fit actually produces (the density is its gradient), and the
        atom at 0 -- which a conditional generally has -- contributes nothing to a raw moment of order ``j >= 1`` but
        would dominate a density quadrature near the origin.

        Errors are scaled against the *unconditional* raw moments, as in :meth:`check_conditional_moments` and for the
        same reason: deep in the conditioning tail the conditional collapses onto its atom at 0, and an unscaled
        relative error there divides two vanishing numbers and reports noise as a total failure.

        ``k`` stops at 2 by design. The cosine CDF is accurate to ~1e-6 *absolutely*, so where the survival function
        is itself ~1e-5 that is a large relative error, and a third moment weights exactly there by ``y^3``: order 3
        reports several percent on a grid that is sound, which measures the amplification and not the grid.

        :param n_points: Conditioning values per axis, spread over :attr:`_COND_CHECK_SPAN` by quantile. Ignored when
            ``quantiles`` is given. Fewer than for the mean check by default: each point builds a cosine grid.
        :param tol: Error threshold above which a violation is logged.
        :param k: Highest raw moment to compare.
        :param quantiles: Explicit conditioning quantiles of the conditioning marginal, in ``(0, 1)``.
        :return: ``{'a': worst_err_conditioning_on_a, 'b': ...}`` (empty for a self-pair).
        """
        us = np.linspace(*self._COND_CHECK_SPAN, n_points) if quantiles is None else np.asarray(quantiles, float)
        if self._is_diagonal:
            return {}

        out = {}
        #: per-axis ``(quantiles, orders, errors)`` of the last run, for the comparison plots; ``errors[i, j]`` is the
        #: scaled error of the order-``j+1`` moment at conditioning point ``i``.
        self.conditional_grid_moment_errors = {}

        for on, other in (('a', 'b'), ('b', 'a')):
            marg_on = self.marginal(on)
            p0 = float(self._atoms['a0' if on == 'a' else 'b0'])
            floors = [self._COND_CHECK_FLOOR * abs(m) for m in self._uncond_raw_moments(other, k)]

            errs, kept, refused = [], [], []
            for u in us:
                v = float(marg_on.quantile(p0 + (1.0 - p0) * float(u)))
                try:
                    cond = self.conditional(on, v)
                    exact = cond._raw_moments(k=k)
                    got = self._grid_raw_moments(cond, k=k)
                except ValueError:
                    refused.append(float(u))
                    continue
                errs.append([abs(g - e) / max(abs(e), f, 1e-12) for g, e, f in zip(got, exact, floors)])
                kept.append(float(u))

            self.conditional_grid_moment_errors[on] = (np.array(kept), np.arange(1, k + 1),
                                                       np.array(errs).reshape(len(kept), k))
            flat = [e for row in errs for e in row]
            out[on] = self._verdict(on, flat, refused, tol, "the conditional's CDF grid", 'the exact raw moments')
        return out

    def _uncond_raw_moments(self, which: str, k: int) -> list:
        """The exact *unconditional* raw moments ``E[R^j]``, ``j = 1..k``, of one of the two rewards, straight from the
        moment engine (no inversion anywhere). They set the error floor of :meth:`check_conditional_grid_moments`."""
        reward = self.reward_a if which == 'a' else self.reward_b
        return [float(MomentEvaluator.moment(self._host, k=j, rewards=(reward,) * j, center=False))
                for j in range(1, k + 1)]

    @staticmethod
    def _grid_raw_moments(cond: RewardDistribution, k: int = 2, n: int = 4001) -> list:
        """
        The raw moments ``E[R^j]``, ``j = 1..k``, of a conditional **as its CDF grid represents it**: from the
        survival function, ``E[R^j] = int_0^b j y^(j-1) (1 - F(y)) dy``.

        The window ``b`` is the **cosine fit's own**, not ``_range``: the fit force-normalises to 1 there, so the
        survival is zero beyond it by construction and a wider window adds nothing but wasted resolution. On a small-N
        demography the two differ by ten orders of magnitude (``_range`` returns 1e5 where the conditional's mean is
        1e-5), which leaves every quadrature node past the support and makes the integral meaningless.

        Simpson on a uniform grid: the integrand ``j y^(j-1) (1 - F(y))`` is smooth and bounded (the atom at 0 sits in
        ``F(0)``, not in the integrand), unlike the density, which spikes at the origin.
        """
        b = float(cond.cdf._cos_coeffs['b'])
        ys = np.linspace(0.0, b, n)
        surv = 1.0 - np.asarray(cond.cdf(ys), float)
        return [float(simpson(j * ys ** (j - 1) * surv, x=ys)) for j in range(1, k + 1)]

    @staticmethod
    def _density_curve(cond: RewardDistribution) -> tuple:
        """The density of one conditional over the same endpoint every 1D distribution plot uses
        (:attr:`Settings.plot_endpoint_quantile`), for :attr:`conditional_densities`. Log-spaced, unlike those: the
        conditionals drawn together span orders of magnitude in scale (conditioning at ``u = 0.01`` against ``u =
        0.99``), and a linear grid would leave the small-reward ones with a handful of points."""
        # a conditional can be almost entirely the atom at 0 (conditioning on a huge R_on can leave the other bin empty
        # outright), and its endpoint quantile is then 0; fall back to the bracketed support so the curve has an axis
        top = float(cond.quantile(Settings.plot_endpoint_quantile)) or cond._range(scale=4.0)
        ys = np.geomspace(top * 1e-4, top, Settings.plot_n_grid)
        return ys, np.asarray(cond.pdf(ys), float)


class ConditionalRewardDistribution(RewardDistribution):
    """
    The distribution of one accumulated reward given a value of another, as returned by
    :meth:`JointRewardDistribution.conditional`. A :class:`RewardDistribution` like any other in how it is *used*
    (``cdf`` / ``pdf`` / ``quantile`` / ``mean``, callable and plottable), but not in how it is *obtained*, and the
    difference is visible in two places.

    A conditional carries no state-space reward, so its moments do not come from the moment engine: :attr:`var` and
    :meth:`moment` take the conditional's own transform on the atom and the derivative identity away from it. Its
    support is sized by bracketing the exact CDF rather than from ``mean + scale * std``, whose variance the nested
    transform under-estimates.

    .. warning::
        Conditioning on a *value* (``value > 0``) makes the transform a **nested inversion**: one numerical inversion
        divided by another. A loss of precision is possible, most of all far out in the conditioning tail and on
        demographies with many epochs. Expect a few correct digits rather than machine precision. Conditioning on the
        atom (``value = 0``) is not affected -- that transform is exact.
    """
    #: The conditioning value. Overridden by :class:`_NestedConditional`; the atom conditions on ``R_on = 0``.
    _value: float = 0.0

    def _raw_moments(self, k: int = 2) -> list:
        """
        The raw moments ``E[R_other^j | R_on = value]``, ``j = 1..k``, by the derivative identity.

        Differentiating the joint transform in the *other* argument at 0 turns a conditional moment into a single
        **1D** Laplace inversion, with no nested inversion anywhere::

            d^j/ds_o^j Phi(s_on, s_o) |_{s_o = 0}  =  (-1)^j L_{s_on}[ E[R_other^j | R_on = v] f_on(v) ]

        because the right-hand side is just the transform of ``v -> E[R_other^j | R_on = v] f_on(v)``. So

            E[R_other^j | R_on = v]  =  L^-1[ (-1)^j d^j Phi ](v)  /  f_on(v),

        with ``f_on(v) = L^-1[Phi(., 0)](v)`` the conditioning marginal's density (the ``j = 0`` case). The derivatives
        come from :meth:`~JointRewardDistribution.lst_taylor`, which evaluates the transform in a truncated polynomial
        ring and so returns them **exactly**: ``d^j Phi = j! Phi_j``. They are *not* finite differences. Differencing
        here is not viable, even though the transform is analytic: the difference's ``eps_mach / h`` residue is then
        fed to de Hoog's QD recurrence, which amplifies it enough to swing the inverted value by 15% between
        neighbouring steps ``h``.

        Costs ``k + 1`` 1D inversions over one shared node set, so it shares no code and no quadrature with the nested
        transform whose mean :meth:`JointRewardDistribution.check_conditional_moments` checks against it.

        The inversion is the accuracy floor, and on a **multi-epoch** demography de Hoog's own error is what that
        floor is. It is not a smooth bias but a pointwise noise band: de Hoog places its nodes by ``value``, so a
        relative change of 1e-6 in the conditioning value reshuffles the roundoff its QD recurrence amplifies and
        swings the answer by a percent or so. It is non-monotone in :attr:`Settings.dehoog_degree`, and averaging over
        a neighbourhood of ``value`` does not remove it. So treat this as exact on a single epoch, and as a
        gross-regression tripwire with a few-percent floor beyond that.

        :param k: Highest moment to return.
        :return: ``[E[R_other], ..., E[R_other^k]]`` given ``R_on = value``.
        :raises NotImplementedError: On the atom conditional (``value = 0``), where the identity divides by the
            *continuous* density and so cannot be evaluated.
        :raises ValueError: If the conditioning marginal's density at ``value`` is not resolvable, so the conditional
            cannot be normalised.
        """
        if self._value == 0.0:
            raise NotImplementedError(
                "The derivative identity divides by the conditioning marginal's continuous density, which is not the "
                "atom mass at 0, so it cannot give the moments of the atom conditional."
            )

        marg = self._joint.marginal(self._on)

        # the k+1 inversions share one de Hoog node set (same ``value``, same degree), and one Taylor evaluation yields
        # every coefficient at a node, so cache on ``s`` rather than paying for the transform once per moment
        cache = {}

        def taylor(s) -> list:
            if s not in cache:
                cache[s] = self._joint.lst_taylor(s, self._on, order=k)
            return cache[s]

        f_on = marg._invert(lambda s: taylor(s)[0], self._value)

        # a density has units of 1 / reward, so the floor below which the inversion cannot resolve it scales like
        # 1 / E[R_on], NOT like E[R_on]: a large-N demography carries rewards of ~1e7 and so healthy densities of
        # ~1e-7, every one of which a floor proportional to the mean would reject as unresolvable
        if not f_on > 1e-12 / max(abs(float(marg.mean)), 1e-300):
            raise ValueError(
                f"The marginal density at R_{self._on} = {self._value:g} inverts to {f_on:.3g}, so the conditional "
                f"moments there cannot be normalised. The density is below the float64 resolution of the inversion, "
                f"not necessarily zero -- condition closer to the bulk."
            )

        # d^j Phi = j! Phi_j, and the identity carries (-1)^j; both signs cancel into factorial(j) * (-1)^j * Phi_j
        return [factorial(j) * (-1) ** j * marg._invert(lambda s, j=j: taylor(s)[j], self._value) / f_on
                for j in range(1, k + 1)]

    @cached_property
    def var(self) -> float:
        """
        Variance of the conditional.

        On the atom (``value = 0``) this is the second cumulant of the conditional's own (exact, closed-form)
        transform; :meth:`_raw_moments` cannot answer there. Away from the atom the nested transform's second
        difference is biased low, so the variance comes from :meth:`_raw_moments` instead.

        Accurate to a percent or so away from the atom, and nothing cross-checks it: only the
        :attr:`~RewardDistribution.mean` has two independent routes.
        """
        if self._value == 0.0:
            return float(self._cumulants()[1])

        m1, m2 = self._raw_moments(k=2)

        return max(float(m2) - float(m1) ** 2, 0.0)

    def moment(self, k: int) -> float:
        """
        The ``k``-th raw moment ``E[R_other^k | R_on = value]``.

        Away from the atom any ``k`` is available, from :meth:`_raw_moments`. On the atom only the first two are: the
        identity cannot be evaluated there, and the higher cumulant differences of the transform are too noisy.

        :param k: Order of the moment.
        :return: The ``k``-th raw moment.
        :raises ValueError: If ``k`` is below 1.
        :raises NotImplementedError: If ``k > 2`` on the atom conditional.
        """
        if k < 1:
            raise ValueError("k must be at least 1.")

        if k == 1:
            return float(self.mean)

        if self._value == 0.0:
            if k > 2:
                raise NotImplementedError(
                    "Only the first two moments are available when conditioning on the atom (value = 0): the "
                    "derivative identity cannot be evaluated there, and the higher cumulant differences of the "
                    "transform are too noisy to trust."
                )
            return float(self.var) + float(self.mean) ** 2

        return float(self._raw_moments(k=k)[k - 1])

    def _range(self, scale: float = 12.0) -> float:
        """Support upper end, found by bracketing the exact CDF (overrides the cumulant-based estimate, whose variance
        is unreliable for the nested transform). Memoised per ``scale`` so the repeated callers (quantile, curve fit)
        do not re-bracket."""
        cache = self.__dict__.setdefault('_range_cache', {})
        if scale not in cache:
            cache[scale] = self._range_via_cdf(scale)
        return cache[scale]

    def _range_via_cdf(self, scale: float = 12.0, n_iter: int = 80) -> float:
        """Grow ``b`` from a seed until the exact CDF ``cdf(b)`` exceeds a generous target probability (more generous
        for larger ``scale``, matching the cumulant-based ``mean + scale*std`` it replaces). The mean (first-difference
        ``_cumulants()[0]``) is reliable and used as the seed; the variance is not.

        The bracket only ever grows, so an over-large seed is never walked back: it has to start *below* the support,
        not above it. Seeding it at ``1 / time_scale`` (as this once did) is fine for a large-N demography, where that
        is negligible against the mean, and catastrophic for a small one -- at ``N = 1e-5`` it seeds at ``1e5`` for a
        conditional whose mass lies below ``1e-4``, the exact CDF is already 1 there, and the support comes out ten
        million times too wide, leaving the cosine fit unable to resolve the distribution at all."""
        target = min(1.0 - float(np.exp(-scale)), 1.0 - 1e-6)  # scale=12 -> ~1-1e-6 (full support); scale=4 -> ~0.98
        # the per-point de Hoog inversion, NOT the ``self.cdf`` function object: a conditional's ``cdf`` answers from
        # the COS grid (the default), whose fit needs this very support window -- going through it recurses
        cdf_point = self.cdf._cdf_point

        # the conditional's own mean, falling back to the *unconditional* mean of the same reward (exact, from the
        # moment engine) when the conditional is essentially the atom at 0 and its mean vanishes
        b = float(self._cumulants()[0])
        if not b > 0:
            other = 'b' if self._on == 'a' else 'a'
            b = abs(float(self._joint.marginal(other).mean))
        if not b > 0:
            return 1e-3

        for _ in range(n_iter):
            if float(cdf_point(b)) >= target:
                break
            b *= 1.6
        return b


class _AtomConditional(ConditionalRewardDistribution):
    """
    The 1D conditional of the *other* reward given ``R_{on} = 0`` -- conditioning on the **atom** ``{R_on = 0}``: the
    sub-distribution of the other reward there, normalised by the atom mass ``P(R_on = 0)``. Its LST is the marginal
    sub-transform restricted to that atom (``Phi(inf, .)`` for ``on='a'``, ``Phi(., inf)`` for ``on='b'``) divided by
    the atom mass, so it plugs straight into the full :class:`RewardDistribution` machinery -- de Hoog ``cdf``/``pdf``,
    the cosine ``cdf`` / ``pdf``, quantile and plotting -- exactly like :class:`_NestedConditional` does
    for the ``value > 0`` case, so both conditional cases share one accurate path. The residual atom at 0
    (``P(both = 0) / P(R_on = 0)``) surfaces automatically as ``p0 = lst(inf)``.
    """
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, joint: 'JointRewardDistribution', on: str, label: str = '') -> None:
        atom = joint._atoms['a0' if on == 'a' else 'b0']
        if atom < 1e-9:
            raise ValueError(f"Cannot condition on R_{on} = 0: it has (near) zero probability.")
        self._joint = joint
        # the conditional reuses the host's state space / time-scale (its ``lst`` is the marginal sub-transform; there
        # is no own reward to bind, so ``_setup`` is never invoked -- see ``_time_scale``). This must be bound BEFORE
        # ``_s_inf`` is read: ``_time_scale`` reaches the host through ``getattr(self, '_host', None)``, so probing
        # first silently falls through to tau = 1 and hard-codes the probe at 1e8 -- exactly what ``_s_inf`` exists to
        # prevent (at tau = 1e-6 the transform has not decayed at all by 1e8 and the sub-transform is simply wrong)
        self._host = joint._host
        big = self._s_inf
        self.state_space = joint._host.state_space
        self._on = on
        self._atom = atom
        self._sub = (lambda s: joint.lst(big, s)) if on == 'a' else (lambda s: joint.lst(s, big))
        self._logger = logger.getChild(self.__class__.__name__)
        self.label = label

    def lst(self, s: complex) -> complex:
        """Conditional LST: the marginal sub-transform on the atom ``{R_on = 0}``, normalised by the atom mass."""
        return self._sub(s) / self._atom




_PADE13 = np.array([64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
                    129060195264000., 10559470521600., 670442572800., 33522128640.,
                    1323241920., 40840800., 960960., 16380., 182., 1.])


def _expm_batch(A: np.ndarray) -> np.ndarray:
    """
    Matrix exponential of a *stack* of matrices ``(k, n, n)``, by Pade-13 with scaling-and-squaring, vectorised over
    the leading axis.

    Used for the inner-inversion node batch (:func:`_lst_from_shift_batch`), where the same generator is exponentiated
    against a hundred-odd different diagonal shifts. ``scipy.linalg.expm`` is ~100x the FLOPs a 22x22 Pade-13 needs
    (~13 matmuls, ~300 kflop): the time goes on its per-call norm estimation, Pade-order selection and allocations,
    and its own batched mode does not amortise them either. Hoisting that analysis out of the loop -- one scaling
    choice, stacked matmuls -- is 2.2x here, and matches ``scipy.linalg.expm`` to ~3e-15.
    """
    n = A.shape[-1]
    nrm = np.abs(A).sum(-2).max(-1)
    sq = np.maximum(0, np.ceil(np.log2(np.maximum(nrm / 5.37, 1e-300))).astype(int))
    As = A / (2.0 ** sq)[:, None, None]
    I = np.broadcast_to(np.eye(n, dtype=A.dtype), A.shape)
    A2 = As @ As
    A4 = A2 @ A2
    A6 = A2 @ A4
    U = As @ (A6 @ (_PADE13[13] * A6 + _PADE13[11] * A4 + _PADE13[9] * A2)
              + _PADE13[7] * A6 + _PADE13[5] * A4 + _PADE13[3] * A2 + _PADE13[1] * I)
    V = (A6 @ (_PADE13[12] * A6 + _PADE13[10] * A4 + _PADE13[8] * A2)
         + _PADE13[6] * A6 + _PADE13[4] * A4 + _PADE13[2] * A2 + _PADE13[0] * I)
    R = np.linalg.solve(V - U, V + U)
    for k in np.nonzero(sq)[0]:  # square each back up to its own scaling
        for _ in range(int(sq[k])):
            R[k] = R[k] @ R[k]
    return R


def _lst_from_shift_batch(shifts: np.ndarray, alpha, T_epochs, sparse: bool, perm=_AUTO_PERM) -> np.ndarray:
    """
    :func:`_lst_from_shift` over a *stack* of shift vectors ``(k, nt)``, sharing the per-epoch matrix assembly and
    exponentiating the whole batch at once (:func:`_expm_batch`). The inner inversion evaluates the transform at a
    fixed node set, so it always has a batch to hand; the scalar routine is left untouched for every other caller.
    """
    nt = len(alpha)
    k = len(shifts)
    vec = np.zeros((k, nt + 1), dtype=complex)
    vec[:, :nt] = alpha

    for T, t0, t1 in T_epochs[:-1]:
        Td = T.toarray() if sp.issparse(T) else np.asarray(T)
        Q = np.zeros((k, nt + 1, nt + 1), dtype=complex)
        Q[:, :nt, :nt] = Td
        Q[:, np.arange(nt), np.arange(nt)] -= shifts  # only the diagonal varies across the batch
        Q[:, :nt, nt] = _exit_rates(T)
        vec = np.einsum('ki,kij->kj', vec, _expm_batch(Q * (t1 - t0)))

    a, c = vec[:, :nt], vec[:, nt]
    Tm = T_epochs[-1][0]
    exit_m = _exit_rates(Tm)
    out = np.empty(k, dtype=complex)
    for i in range(k):  # the final-epoch solve keeps the (block-triangular, sparse-capable) LU of the scalar path
        A = (sp.diags(shifts[i]) if sparse else np.diag(shifts[i])) - Tm
        out[i] = c[i] + a[i] @ MomentEvaluator._lu_solver(A, sparse, perm)(exit_m)
    return out


#: Starting Fourier truncation for the inner Euler inversion; :meth:`_NestedConditional._calibrate` refines from here.
_EULER_N0 = 30


def _euler_invert(transform, t: float, A: float = 16.0, N0: int = _EULER_N0, m: int = 12) -> complex:
    """
    Euler-accelerated Fourier-series (Abate-Whitt) Laplace inversion of ``transform`` at ``t``.

    With period ``T = 2t`` the Bromwich integral becomes the Fourier series

        f(t) ~ (e^{A/2} / 2t) sum_{k=-inf}^{inf} (-1)^k F( (A + 2 pi i k) / (2t) ),

    whose ``2 pi / T`` node spacing makes it genuinely *alternating*, which is what Euler summation requires (a
    ``pi / T`` spacing gives weights ``i^k``, a period-4 rotation, and the acceleration is then invalid). The tail
    beyond ``N0`` is Euler-averaged with binomial weights over the partial sums ``S_{N0..N0+m}``.

    Summed **two-sided**, so the inverse may be complex-valued -- no conjugate symmetry is assumed (unlike de Hoog
    and Talbot, which evaluate only the upper half-plane and take a real part).

    The nodes and weights are *fixed*, so the result is a plain linear combination ``sum_k w_k F(u_k)``: for a
    two-argument transform this keeps the inversion exactly as analytic in the *other* argument as the transform
    itself, which is what lets it be nested inside the outer de Hoog (see :meth:`_NestedConditional._G`).

    ``A`` trades aliasing error (``~ e^{-A}``) against amplification of roundoff and of the near-cancelling
    alternating sum (``~ e^{A/2}``). Larger is *not* better: the aliasing term is negligible here, while by
    ``A = 36`` the amplification alone puts the bottleneck normaliser 94% off. The residual error is Fourier-series
    truncation, which ``N0`` (not ``A``) buys down.
    """
    ks = np.arange(-(N0 + m), N0 + m + 1)
    u = (A + 2.0j * np.pi * ks) / (2.0 * t)
    binom = np.array([comb(m, j) for j in range(m + 1)], dtype=float) / 2.0 ** m
    # Euler weight of node k: the binomial-averaged fraction of the partial sums S_{N0+j} that include it
    frac = np.array([binom[max(0, abs(k) - N0):].sum() if abs(k) > N0 else 1.0 for k in ks])
    w = (np.exp(A / 2.0) / (2.0 * t)) * ((-1.0) ** ks) * frac
    vals = transform(u)  # the whole node set at once -- see _lst_from_shift_batch
    return complex(np.sum(w * np.asarray(vals)))


class _NestedConditional(ConditionalRewardDistribution):
    """
    The 1D conditional distribution of one reward given ``R_{on} = value`` (``value > 0``), built by **nested
    inversion**: invert the conditioned dimension at ``value`` to get the other reward's conditional Laplace
    transform, then reuse the full :class:`RewardDistribution` machinery (de Hoog ``cdf``/``pdf``, two-pass COS
    plotting curves, atom handling, quantile, plotting).

    The conditional LST is ``phi(s) = G(s) / G(0)`` where ``G(s) = L^{-1}_{u -> value}[ u -> Phi(s, u) ]`` is the
    inner inversion of the joint transform ``Phi`` along the conditioned axis at ``value`` (see :meth:`_G`), and
    ``G(0)`` is the marginal density of the conditioning reward there (the normaliser). This resolves the conditional
    exactly, unlike the coarse 2D-cosine slice.

    The inner inversion's resolution (``N0``, its Fourier-series truncation) is calibrated **once** in
    :meth:`__init__`, by refining until the normaliser ``G(0)`` stops moving, and then held **fixed for every** ``s``.
    Letting it vary with ``s`` would make ``G`` a *different* linear functional at each ``s`` and so destroy its
    analyticity in ``s`` -- the one property the outer inversion depends on (see :meth:`_G`).
    """
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, joint: 'JointRewardDistribution', on: str, value: float, label: str = '') -> None:
        self._joint = joint
        # the conditional reuses the host's state space / time-scale (its ``lst`` is the nested transform; there is no
        # own reward to bind, so ``_setup`` is never invoked -- see ``_time_scale``)
        self._host = joint._host
        self.state_space = joint._host.state_space
        self._on = on
        self._value = float(value)
        self._logger = logger.getChild(self.__class__.__name__)
        self.label = label

        self._N0, self._G0 = self._calibrate()

    def _calibrate(self, tol: float = 2e-2, n_max: int = 480) -> tuple:
        """
        Pick the inner inversion's Fourier truncation ``N0`` and evaluate the normaliser ``G(0)`` (the marginal
        density of the conditioning reward at ``value``) with it.

        ``N0`` is refined by doubling until ``G(0)`` stops moving by more than ``tol`` in relative terms. A fixed
        ``N0`` cannot serve every demography: a sharply peaked reward density has slowly-decaying Fourier
        coefficients, and where ``N0 = 30`` leaves the normaliser 27% off on a strong bottleneck, ``N0 = 240`` gets it
        to 4e-5. Refining is also what makes the *easy* cases cheap -- they converge at the first step and never pay
        for the hard ones.

        ``tol`` is deliberately loose. The Euler sum converges slowly enough in ``N0`` that a tight criterion keeps
        doubling long after the *conditional* has stopped improving: at ``tol = 1e-3`` a cell whose conditional mean
        was already within 1.0% of an independent sampler refined to 0.8% and paid 7x for it. The normaliser only has
        to be good enough not to distort ``phi = G(s) / G(0)``.

        The ``cur > 0`` guard is load-bearing, not a redundant sanity check: a *resolved* density stays positive at
        every truncation, so a **sign flip along the refinement sequence** is proof that the inversion is returning
        noise rather than a small number. That is what a deep bottleneck does out in the tail
        (``+4.6e-4, -1.0e-3, -5.0e-4, +1.4e-4`` as ``N0`` doubles), and it is what makes the refusal robust
        independently of where ``n_max`` happens to land.

        Non-convergence is the honest failure signal, and it is *not* the same as "the density is zero": for a
        sufficiently deep bottleneck the true density out in the tail (~1e-7, and smaller) drops below what any
        float64 Laplace inversion can resolve -- PhaseGen's own de Hoog marginal returns a *negative* density there
        too. The distribution is perfectly well defined (the reward-bridge sampler recovers a conditional mean of 0.22
        on exactly the cells refused here); it is the *inversion* that cannot see it. So refuse, and say so, rather
        than return a confidently wrong number -- the old Gaver-Stehfest path returned 0.0009 for that 0.22.

        :param tol: Relative change in ``G(0)`` below which the truncation is deemed converged.
        :param n_max: Largest truncation to try before giving up.
        :return: ``(N0, G(0))``.
        """
        n0 = _EULER_N0
        prev = _euler_invert(self._phi, self._value, N0=n0).real
        while n0 < n_max:
            n0 *= 2
            cur = _euler_invert(self._phi, self._value, N0=n0).real
            if abs(cur - prev) <= tol * abs(cur) and cur > 0:
                return n0, cur
            prev = cur

        if prev <= 1e-300:
            raise ValueError(
                f"The marginal density at R_{self._on} = {self._value:g} is not resolvable: the numerical Laplace "
                f"inversion returns {prev:.3g} there, so the conditional cannot be normalised. The density is far out "
                f"in the tail and below the float64 resolution of the inversion, not necessarily zero -- conditioning "
                f"closer to the bulk, or sampling, will work."
            )
        raise ValueError(
            f"The marginal density at R_{self._on} = {self._value:g} did not converge under refinement of the inner "
            f"inversion (still moving by more than {tol:.0%} at N0 = {n_max}); the conditional there would be "
            f"unreliable. Condition closer to the bulk, or sample."
        )

    def _phi(self, u: np.ndarray) -> np.ndarray:
        """``u -> Phi(., u)`` along the conditioned axis, with the *other* argument held at 0 (the normaliser)."""
        z = np.zeros(len(u))
        return self._joint.lst_batch(z, u) if self._on == 'b' else self._joint.lst_batch(u, z)

    def _G(self, s: complex) -> complex:
        """
        ``G(s) = L^{-1}_{u -> value}[ u -> Phi(s, u) ](value)`` (inner inversion along the conditioned axis), by the
        **Euler-accelerated Fourier series** (Abate-Whitt) -- see :func:`_euler_invert`.

        The inner inversion has to satisfy three constraints at once, and each rules out an obvious method:

        * **accurate.** Rules out Gaver-Stehfest (the previous choice), which samples the transform only on the
          *real* axis and extrapolates with huge alternating weights. It has no viable operating point on coalescent
          reward densities: on a bottleneck it still overstates the normaliser by 15x at ``M = 10`` (truncation --
          it cannot resolve a sharply peaked, multi-scale density from the real axis alone), while ``sum |V_k|``
          reaches 4e15 by ``M = 12``, so the 1e-16 error of a float64 ``lst`` swamps the answer before the
          truncation error dies. It returned *negative* densities 3x to 32x too large, which is what produced the
          spurious "density is zero" refusals in :meth:`__init__`, the negative conditional branch lengths, and the
          tail errors.
        * **a fixed linear functional of Phi**, i.e. ``G(s) = sum_k w_k Phi(u_k, s)`` at *fixed* nodes, so that ``G``
          inherits ``Phi``'s analyticity in ``s``. The **outer** inversion (:meth:`RewardDistribution._invert`) is an
          ill-conditioned QD recurrence and needs a smooth, analytic ``phi(s)``. This rules out using de Hoog for the
          inner inversion too: de Hoog is *nonlinear* (Pade/QD acceleration), so ``G(s)`` is accurate pointwise but
          not smooth in ``s``, and nesting one ill-conditioned recurrence inside another produces a non-monotone CDF
          wrong by tens of percent even on Kingman.
        * **nodes on a vertical contour.** Rules out Talbot, whose contour deforms into ``Re(s) -> -inf``, where the
          per-epoch ``exp((T - s diag(r)) dt)`` overflows.

        Euler satisfies all three. It is summed two-sided, so no conjugate symmetry is assumed and a complex-valued
        inverse (which ``G_s`` is, whenever ``s`` is complex) needs no special handling.
        """
        if self._on == 'b':
            return _euler_invert(lambda u: self._joint.lst_batch(np.full(len(u), s), u), self._value, N0=self._N0)
        return _euler_invert(lambda u: self._joint.lst_batch(u, np.full(len(u), s)), self._value, N0=self._N0)

    def lst(self, s: complex) -> complex:
        """The conditional Laplace-Stieltjes transform ``phi(s) = G(s) / G(0)``."""
        return self._G(complex(s)) / self._G0
