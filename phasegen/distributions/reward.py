"""
Distributions of accumulated rewards obtained from their Laplace transforms: the 1D
:class:`~phasegen.distributions.RewardDistribution`, the bivariate
:class:`~phasegen.distributions.JointRewardDistribution` and the
:class:`~phasegen.distributions.ConditionalRewardDistribution`.
"""
import logging
from math import comb, factorial
from typing import TYPE_CHECKING, Optional, Sequence

import mpmath as mp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.integrate import simpson

from ..caching import cached_property
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
    r"""
    Distribution of the accumulated reward :math:`R` from time 0 to absorption, with the notation of
    :class:`~phasegen.distributions.PhaseTypeDistribution`. It is returned by
    :meth:`Coalescent.distribution() <phasegen.distributions.Coalescent.distribution>`,
    :meth:`PhaseTypeDistribution.distribution() <phasegen.distributions.PhaseTypeDistribution.distribution>` and the
    ``bin()`` methods of the spectra, and it supplies the ``cdf``, ``pdf`` and ``quantile`` of every phase-type
    distribution except :class:`~phasegen.distributions.TreeHeightDistribution`. The mean and variance are exact
    moments. The ``cdf``, ``pdf`` and ``quantile`` numerically invert the transform :math:`\varphi` of
    :meth:`RewardDistribution.lst() <phasegen.distributions.RewardDistribution.lst>` as follows.

    The atom :math:`p_0 = \mathbb{P}(R = 0) = \lim_{s \to \infty} \varphi(s)` is positive when the reward can be
    zero at absorption, as for an SFS bin that may be empty. It is evaluated as :math:`\varphi(s_\infty)` at a real
    :math:`s_\infty` that is a large fixed multiple of :math:`1/\zeta`, with :math:`\zeta` the time unit of
    :meth:`RewardDistribution.lst() <phasegen.distributions.RewardDistribution.lst>`. A negligible atom is not split
    off.

    Below the CDF level :math:`c \in [0, 1]` set by :attr:`Settings.dehoog_tail_quantile
    <phasegen.settings.Settings.dehoog_tail_quantile>`, the CDF is the Fourier-cosine expansion of the continuous part
    on a window :math:`[0, \beta]` (Fang and Oosterlee, 2008),

    .. math::

        F(x) = p_0 + (1 - p_0) \Big[\frac{x}{\beta} + \sum_{j=1}^{K-1} \frac{\beta A_j}{j \pi}
        \sin\Big(\frac{j \pi x}{\beta}\Big)\Big], \qquad
        A_j = \frac{2}{\beta}\,\mathrm{Re}\,\frac{\varphi(-\mathrm{i} j \pi / \beta) - p_0}{1 - p_0},

    for :math:`0 \le x \le \beta`, where :math:`K` is the number of cosine terms and :math:`A_j` is the :math:`j`-th
    cosine coefficient of the continuous density, a sample of its characteristic function at frequency :math:`j \pi /
    \beta`. The result is clipped to :math:`[0, 1]` and made non-decreasing. The expansion satisfies :math:`F(\beta) =
    1`, so the mass beyond :math:`\beta` is lost, and the window is chosen in two passes. A first expansion on
    :math:`[0, \hat\mu + \kappa \hat\sigma]` locates the support, with :math:`\hat\mu` and :math:`\hat\sigma^2` the mean
    and variance from central differences of :math:`\varphi` at 0 and :math:`\kappa > 0` a scale factor. The second
    expansion uses the window whose end :math:`\beta` is the :math:`1 - \delta` quantile of the first, with
    :math:`\delta > 0` the tail mass it may discard, and it is evaluated on :math:`N` equispaced nodes in :math:`[0,
    \beta]`.

    Above :math:`c` the nodes carry exact values of :math:`F`, the inverse transform of :math:`\varphi(s)/s`, from the
    method of de Hoog et al. (1982),

    .. math::

        F(x) \approx \frac{e^{\gamma x}}{L}\,\mathrm{Re}\Big[\frac{1}{2} \frac{\varphi(z_0)}{z_0}
        + \sum_{\ell=1}^{2D} \frac{\varphi(z_\ell)}{z_\ell}\, e^{\mathrm{i} \ell \pi x / L}\Big],
        \qquad z_\ell = \gamma + \frac{\mathrm{i} \ell \pi}{L},

    where the series is summed by a quotient-difference continued fraction, :math:`D` is
    :attr:`Settings.dehoog_degree <phasegen.settings.Settings.dehoog_degree>`, :math:`L = 2x` is the period
    parameter and :math:`\gamma > 0` the abscissa of the integration contour. The inversion runs in the time unit
    :math:`\zeta` and is evaluated in extended precision by ``mpmath``. Starting at the point where the expansion
    reaches :math:`c`, each exact node lies a step :math:`\min\{\eta_H (1 - F),\, \eta_F\} / \hat f` beyond the
    previous one, where :math:`\hat f` is the local density and :math:`\eta_H, \eta_F > 0` are the step sizes in
    cumulative hazard and in probability. The nodes extend until they cover the largest queried point and level and a
    CDF of :math:`1 - \epsilon`, with :math:`\epsilon > 0` a small tail mass, up to a fixed number of nodes. They are
    computed only when a query reaches beyond :math:`c`, and they are kept for later queries.

    The grid consists of the equispaced cosine nodes whose CDF lies below :math:`c` and the exact nodes. The ``cdf``,
    ``pdf`` and ``quantile`` are all read from it by the cumulative-hazard interpolation described at
    :class:`~phasegen.distributions.QuantileFunction`, so the quantile inverts the CDF exactly, the density is
    non-negative, and the quantile of a level :math:`q \le p_0` is 0. The grid is built once per distribution, shared
    by the three functions, and rebuilt when :attr:`Settings.dehoog_tail_quantile
    <phasegen.settings.Settings.dehoog_tail_quantile>` changes. If :attr:`Settings.check_inversions
    <phasegen.settings.Settings.check_inversions>` is set, a warning is logged when the raw cosine CDF decreases by
    more than a small fraction of its range, which indicates a feature the expansion cannot resolve.

    .. rubric:: References

    de Hoog, F. R., Knight, J. H. and Stokes, A. N. (1982). An improved method for numerical inversion of Laplace
    transforms. SIAM Journal on Scientific and Statistical Computing 3(3), 357-366.

    Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series
    expansions. SIAM Journal on Scientific Computing 31(2), 826-848.

    .. versionadded:: 2.0
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
        r"""
        The :math:`s \to \infty` probe used for the atom :math:`\Pr(R = 0) = \varphi(\infty)` (and the axis atoms
        of a joint).

        Scaled by the inversion time scale, *not* a fixed number: the transform decays on the scale of the rates,
        which go like :math:`1/\tau`, so a hard-coded :math:`s` is only large in the :math:`\tau \sim 1` regime. On
        a small-N demography (:math:`\tau = 10^{-6}`) :math:`\varphi(10^8)` has not decayed at all and reports a 1.9%
        atom for a doubleton bin whose atom is exactly 0 (every binary tree has a cherry); it needs
        :math:`s \sim 10^{12}` to converge. Probing at :math:`10^8/\tau` keeps :math:`s` the same large multiple of
        the rate scale in every regime.
        """
        return 1e8 / self._time_scale

    @cached_property
    def mean(self) -> float:
        r"""Mean :math:`\mathbb{E}[R]` of the accumulated reward, evaluated by
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`."""
        reward = getattr(self, 'reward', None)
        if reward is None:
            return float(self._cumulants()[0])
        # go through the moment engine directly: a spectrum host overrides ``moment`` to return a whole SFS, which
        # would break ``float()`` for a single-bin reward
        return float(MomentEvaluator.moment(self._host, k=1, rewards=(reward,), center=False))

    @cached_property
    def var(self) -> float:
        r"""Variance :math:`\operatorname{Var}(R) = \mathbb{E}[R^2] - \mathbb{E}[R]^2` of the accumulated reward,
        evaluated by :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`."""
        return float(MomentEvaluator.moment(self._host, k=2, rewards=(self.reward, self.reward), center=True))

    @cached_property
    def std(self) -> float:
        """Standard deviation of the accumulated reward."""
        return self.var ** 0.5

    def lst(self, s: complex) -> complex:
        r"""
        The Laplace-Stieltjes transform :math:`\varphi(s) = \mathbb{E}[e^{-sR}]` of the accumulated reward, with the
        notation of :class:`~phasegen.distributions.PhaseTypeDistribution`.

        Let :math:`\mathbf{r}_T` be the restriction of :math:`\mathbf{r}` to the transient states. While the process
        occupies state :math:`x`, the weight :math:`e^{-sR}` decays at rate :math:`s\,r(x)`, so the reward enters as the
        diagonal shift :math:`-s \operatorname{diag}(\mathbf{r}_T)` of the sub-intensity matrix. The bounded epochs
        propagate the row vector :math:`\mathbf{a}(s)` of weighted transient probabilities together with the absorbed
        weight :math:`c(s)`,

        .. math::

            [\mathbf{a}(s),\; c(s)] = [\boldsymbol{\alpha}_T,\; 0] \prod_{i=1}^{M-1}
            \exp\left(\begin{bmatrix} \mathbf{T}_i - s \operatorname{diag}(\mathbf{r}_T) & \mathbf{q}_i \\
            \mathbf{0} & 0 \end{bmatrix} \Delta_i\right),

        and the unbounded last epoch contributes in closed form,

        .. math::

            \varphi(s) = c(s) + \mathbf{a}(s) \big(s \operatorname{diag}(\mathbf{r}_T) - \mathbf{T}_M\big)^{-1}
            \mathbf{q}_M.

        With a single epoch this is the transform :math:`\boldsymbol{\alpha}_T (s \operatorname{diag}(\mathbf{r}_T) -
        \mathbf{T}_1)^{-1} \mathbf{q}_1` of the reward-transformed phase-type distribution (Hobolth et al., 2019).
        The exponentials of the bounded epochs are formed densely. The last-epoch system is solved by a sparse LU
        factorisation from :attr:`Settings.closed_form_sparse_min_states
        <phasegen.settings.Settings.closed_form_sparse_min_states>` transient states on, and by a dense one below.

        The transform is evaluated in a time unit :math:`\zeta > 0`, which is the mean population size
        :math:`\bar N` at time 0 when :math:`\bar N` lies outside a fixed interval around 1, and 1 otherwise. The
        substitution :math:`\mathbf{T}_i \mapsto \zeta \mathbf{T}_i`, :math:`t_i \mapsto t_i / \zeta`,
        :math:`s \mapsto \zeta s` leaves :math:`\varphi(s)` unchanged and keeps the shifted matrices well scaled for
        very large or very small populations.

        .. rubric:: References

        Hobolth, A., Siri-Jégousse, A. and Bladt, M. (2019). Phase-type distributions in population genetics.
        Theoretical Population Biology 127, 16-32.

        :param s: The argument, with non-negative real part, or purely imaginary for the characteristic function
            :math:`\varphi(-\mathrm{i}\omega) = \mathbb{E}[e^{\mathrm{i}\omega R}]` at the frequency
            :math:`\omega \in \mathbb{R}`.
        :return: The transform at ``s``.
        :raises NotImplementedError: If the reward does not assign one value per state, or if the coalescent has a
            bounded accumulation window.
        :raises ValueError: If the reward is negative.
        """
        self._host._assert_not_windowed()
        st = self._setup
        # evaluate against the tau-scaled generators at s*tau (R -> R/tau); the result equals the unscaled phi(s)
        # exactly but stays well-conditioned for large N (see ``time_scale``)
        return _lst_from_shift((s * st['tau']) * st['r'], st['alpha'], st['T_epochs'], st['sparse'], st['lu_perm'])

    def _invert(self, transform, t: float) -> float:
        r"""
        De Hoog inversion (``mpmath.invertlaplace``) of ``transform`` at ``t``, described at ``RewardDistribution``.
        It runs in the time unit :math:`\zeta` of ``_time_scale``, using
        :math:`g(t) = \zeta^{-1} \mathcal{L}^{-1}[\sigma \mapsto G(\sigma / \zeta)](t / \zeta)` for a transform
        :math:`G` with inverse :math:`g`, so the contour nodes stay of order one.

        :param transform: The transform to invert, a function of a complex argument.
        :param t: The point at which to evaluate the inverse.
        :return: The inverse at ``t``, 0 for ``t <= 0``.
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
        r"""Mean and variance of the accumulated reward from the LST near 0 (:math:`\varphi(0) = 1`):
        :math:`c_1 = -\varphi'(0)`, :math:`c_2 = \varphi''(0) - \varphi'(0)^2`. Cheap (three transform evaluations);
        used to set the COS / plot range. The
        finite-difference step is scaled by ``1/tau`` (``tau ~`` the reward scale for large N) so that ``h * R`` stays
        small and ``phi(-h) = E[e^{h R}]`` does not overflow for large-N demographies."""
        h = 1e-4 / self._time_scale
        d1 = (self.lst(h).real - self.lst(-h).real) / (2 * h)
        d2 = (self.lst(h).real - 2.0 + self.lst(-h).real) / h ** 2
        return -d1, max(d2 - d1 ** 2, 1e-12)

    def _range(self, scale: float = 12.0) -> float:
        r"""An upper end for the support (:math:`\mathbb{E}[R] + \text{scale}\cdot\operatorname{std}(R)`), for the COS
        interval and default plot grids."""
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
    The time unit of the transform, described at :meth:`RewardDistribution.lst()
    <phasegen.distributions.RewardDistribution.lst>`: the mean population size at time 0 when it lies outside
    ``[1e-2, 1e2]``, and 1 otherwise.

    :param host: The phase-type distribution whose demography sets the unit.
    :return: The time unit.
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
    r"""Per-state rate of (direct) absorption = row deficit of the transient sub-generator,
    :math:`-\mathbf{T}\mathbf{e}`."""
    return -np.asarray(T @ np.ones(T.shape[0])).ravel()


def _lst_from_shift(shift: np.ndarray, alpha: np.ndarray, T_epochs, sparse: bool, perm=_AUTO_PERM) -> complex:
    r"""
    The transform of ``RewardDistribution.lst`` with the diagonal shift :math:`s \mathbf{r}_T` replaced by an
    arbitrary vector ``shift``, which is :math:`s_a \mathbf{r}_a + s_b \mathbf{r}_b` for the joint transform.
    ``perm`` is the block-triangular ordering of the last-epoch sub-intensity matrix, which depends only on its
    sparsity pattern and is passed to ``MomentEvaluator._lu_solver``.
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
    """The Taylor coefficients ``[Phi_0, ..., Phi_order]`` of ``_lst_from_shift`` at the shift ``shift + eps * deriv``,
    described at ``JointRewardDistribution.lst_taylor``: bounded epochs exponentiate the block-bidiagonal matrix over
    the truncated polynomial ring, the last epoch back-substitutes with one LU."""
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
    r"""
    Joint distribution of two accumulated rewards :math:`R_a` and :math:`R_b` with reward vectors :math:`\mathbf{r}_a`
    and :math:`\mathbf{r}_b`, accumulated over :math:`[0, \infty)`, with the notation of
    :class:`~phasegen.distributions.PhaseTypeDistribution`. It is returned by
    :meth:`PhaseTypeDistribution.joint_distribution() <phasegen.distributions.PhaseTypeDistribution.joint_distribution>`
    and the accessors built on it.

    The distribution is determined by the bivariate Laplace-Stieltjes transform
    :math:`\Phi(s_a, s_b) = \mathbb{E}[e^{-s_a R_a - s_b R_b}]` with complex arguments :math:`s_a` and :math:`s_b`. The
    rewards enter only through the diagonal matrix
    :math:`\mathbf{D} = \operatorname{diag}(s_a \mathbf{r}_{a,T} + s_b \mathbf{r}_{b,T})`, with
    :math:`\mathbf{r}_{a,T}` and :math:`\mathbf{r}_{b,T}` the reward vectors restricted to the transient states, which
    shifts every sub-intensity matrix. With the absorbing states lumped into one,

    .. math::

        [\mathbf{a},\ c] = [\boldsymbol{\alpha}_T,\ 0] \prod_{i=1}^{M-1}
        \exp\!\left( \begin{bmatrix} \mathbf{T}_i - \mathbf{D} & \mathbf{q}_i \\ \mathbf{0} & 0 \end{bmatrix}
        \Delta_i \right),
        \qquad
        \Phi(s_a, s_b) = c + \mathbf{a}\,(\mathbf{D} - \mathbf{T}_M)^{-1}\,\mathbf{q}_M,

    where the product runs forward in time, and the row vector :math:`\mathbf{a}` and the scalar :math:`c` are the
    transient and the absorbed probability mass at time :math:`t_{M-1}`, each weighted by
    :math:`e^{-s_a R_a - s_b R_b}` with the rewards accumulated up to that time. The last, unbounded epoch is solved in
    closed form. This is the transform of :meth:`RewardDistribution.lst()
    <phasegen.distributions.RewardDistribution.lst>` with its diagonal shift replaced by :math:`\mathbf{D}`, evaluated
    with the same time rescaling.

    Setting one argument to zero gives a marginal transform, for example :math:`\Phi(s, 0) = \mathbb{E}[e^{-sR_a}]`.
    Infinite arguments give the atoms

    .. math::

        p_a = \mathbb{P}(R_a = 0) = \Phi(\infty, 0), \qquad p_b = \mathbb{P}(R_b = 0) = \Phi(0, \infty), \qquad
        p_{00} = \mathbb{P}(R_a = 0,\ R_b = 0) = \Phi(\infty, \infty),

    evaluated at the same large finite argument as the atom :math:`p_0` of a
    :class:`~phasegen.distributions.RewardDistribution`. The mixed moments
    :math:`\mathbb{E}[R_a^j R_b^l] = (-1)^{j+l}\,\partial_{s_a}^j \partial_{s_b}^l \Phi(0, 0)`, for integers
    :math:`j, l \ge 0`, are evaluated exactly by :meth:`JointRewardDistribution.moment()
    <phasegen.distributions.JointRewardDistribution.moment>`.

    The joint CDF and density are described at :class:`~phasegen.distributions.JointCDF` and
    :class:`~phasegen.distributions.JointDensity`, the Taylor coefficients of :math:`\Phi` at
    :meth:`JointRewardDistribution.lst_taylor() <phasegen.distributions.JointRewardDistribution.lst_taylor>`, and the
    conditionals at :class:`~phasegen.distributions.ConditionalRewardDistribution`. A joint distribution has no quantile
    function. On a windowed coalescent the transform and every distribution function raise
    :class:`NotImplementedError`.

    .. versionadded:: 2.0
    """
    @property
    def _time_scale(self) -> float:
        """The inversion time scale of the host, not defaulted (see ``RewardDistribution._time_scale``)."""
        return self._host._time_scale

    @property
    def _s_inf(self) -> float:
        r"""The atom probe :math:`10^8/\tau`, scaled like ``RewardDistribution._s_inf``."""
        return 1e8 / self._time_scale

    #: Bivariate function objects. A joint has no quantile function.
    _pdf_function = JointDensity
    _cdf_function = JointCDF
    _quantile_function = None

    #: Number of cosine terms per axis of the 2D Fourier-cosine expansion (:math:`N` in ``JointCDF``).
    _cos2d_terms: int = 128

    #: Window scale of the 2D expansion, ``mean + scale * std`` per axis (:math:`\kappa` in ``JointCDF``). A wider
    #: window coarsens the resolution ``b / n_terms`` near the origin, a narrower one truncates tail mass.
    _cos2d_window_scale: float = 5.0

    #: Per-axis node count of the finite-difference grid of the density (:math:`m` in ``JointDensity``).
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
        #: Optional human-readable label used in plot titles, such as ``"SFS bins (1, 2)"``.
        self.label: Optional[str] = None

    @cached_property
    def _setup(self) -> dict:
        """Both reward vectors on the transient states, bound to the host's shared time-rescaled epoch data. Every
        transform of the joint and its conditionals passes through here, so the window guard sits here."""
        self._host._assert_not_windowed()
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
        r"""
        The joint transform :math:`\Phi(s_a, s_b)` defined at :class:`~phasegen.distributions.JointRewardDistribution`.

        :param s_a: Argument of :math:`R_a`.
        :param s_b: Argument of :math:`R_b`.
        :return: The transform value.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        st = self._setup
        # both rewards share the time-scale tau (R -> R/tau): evaluate at s*tau against the tau-scaled generators;
        # the value equals the unscaled joint LST exactly but stays well-conditioned for large N (see ``time_scale``)
        tau = st['tau']
        return _lst_from_shift((s_a * tau) * st['ra'] + (s_b * tau) * st['rb'], st['alpha'], st['T_epochs'],
                               st['sparse'], st['lu_perm'])

    def lst_taylor(self, s: complex, on: str = 'a', order: int = 2) -> list:
        r"""
        Taylor coefficients of the joint transform in one argument about zero, with the other argument held at
        :math:`s`. For ``on='a'``,

        .. math::

            \Phi(s, \epsilon) = \sum_{j=0}^{J} \Phi_j(s)\,\epsilon^j + O(\epsilon^{J+1}),

        with :math:`\Phi` as in :class:`~phasegen.distributions.JointRewardDistribution`, :math:`J \ge 0` the ``order``
        and :math:`\epsilon` the argument of :math:`R_b`. For ``on='b'`` the two arguments exchange roles. The
        :math:`j`-th derivative in :math:`\epsilon` at zero is :math:`j!\,\Phi_j(s)`.

        The coefficients carry no truncation or differencing error. Write :math:`\mathbf{r}_h` and :math:`\mathbf{r}_f`
        for the reward vectors of the held and the free argument restricted to the transient states,
        :math:`\mathbf{C}_i` for the matrix of epoch
        :math:`i` inside the exponential of :class:`~phasegen.distributions.JointRewardDistribution` at
        :math:`\epsilon = 0`, and :math:`\mathbf{W}` for :math:`-\operatorname{diag}(\mathbf{r}_f)` padded with a zero
        row and column for the lumped absorbing state. The coefficients :math:`\mathbf{P}_0, \ldots, \mathbf{P}_J` of
        :math:`\exp((\mathbf{C}_i + \epsilon \mathbf{W})\Delta_i) = \sum_j \mathbf{P}_j \epsilon^j + O(\epsilon^{J+1})`
        are the blocks of a single matrix exponential (Van Loan, 1978),

        .. math::

            \exp\begin{pmatrix} \mathbf{C}_i \Delta_i & \mathbf{W} \Delta_i & & \\
            & \ddots & \ddots & \\ & & \mathbf{C}_i \Delta_i & \mathbf{W} \Delta_i \\ & & & \mathbf{C}_i \Delta_i
            \end{pmatrix}
            = \begin{pmatrix} \mathbf{P}_0 & \mathbf{P}_1 & \cdots & \mathbf{P}_J \\ & \mathbf{P}_0 & \ddots & \vdots \\
            & & \ddots & \mathbf{P}_1 \\ & & & \mathbf{P}_0 \end{pmatrix},

        with :math:`J + 1` blocks along the diagonal. These blocks propagate the Taylor coefficients
        :math:`\mathbf{a}_j` and :math:`c_j` of the weighted masses :math:`\mathbf{a}` and :math:`c` through the bounded
        epochs. In the last epoch, with :math:`\mathbf{H} = \operatorname{diag}(s\,\mathbf{r}_h) - \mathbf{T}_M`, the
        solution :math:`\mathbf{z}(\epsilon) = (\mathbf{H} + \epsilon
        \operatorname{diag}(\mathbf{r}_f))^{-1}\mathbf{q}_M` has the coefficients

        .. math::

            \mathbf{z}_0 = \mathbf{H}^{-1}\mathbf{q}_M, \qquad
            \mathbf{z}_j = -\mathbf{H}^{-1}\operatorname{diag}(\mathbf{r}_f)\,\mathbf{z}_{j-1}, \qquad
            \Phi_j(s) = c_j + \sum_{l=0}^{j} \mathbf{a}_l\,\mathbf{z}_{j-l},

        which share one LU factorization of :math:`\mathbf{H}`.

        .. rubric:: References

        Van Loan, C. F. (1978). Computing integrals involving the matrix exponential. IEEE Transactions on Automatic
        Control 23(3), 395-404.

        :param s: Value of the held argument.
        :param on: The held argument, ``'a'`` or ``'b'``. The coefficients are in the other argument.
        :param order: Highest order :math:`J`.
        :return: The coefficients :math:`[\Phi_0(s), \ldots, \Phi_J(s)]`.
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
        r"""
        The joint transform :math:`\Phi` of :class:`~phasegen.distributions.JointRewardDistribution` at a batch of
        argument pairs, sharing the per-epoch matrix assembly and exponentiation across the batch.

        :param s_a: Arguments of :math:`R_a`, a scalar or a 1D array.
        :param s_b: Arguments of :math:`R_b`, a scalar or a 1D array of the same length as ``s_a`` or of length one.
        :return: The transform values, one per argument pair.
        """
        st = self._setup
        tau = st['tau']
        s_a, s_b = np.atleast_1d(s_a), np.atleast_1d(s_b)
        shifts = (np.outer(s_a * tau, st['ra']) + np.outer(s_b * tau, st['rb'])).astype(complex)
        return _lst_from_shift_batch(shifts, st['alpha'], st['T_epochs'], st['sparse'], st['lu_perm'])

    def _lst_grid(self, s_a_vals: np.ndarray, s_b_vals: np.ndarray) -> np.ndarray:
        """``Phi`` on the outer grid ``s_a_vals x s_b_vals``. For one dense epoch, one QZ decomposition of the pencil
        ``(diag(s r_outer) - T, diag(r_inner))`` per node of the shorter axis solves every node of the other axis by
        triangular back-substitution. The pencil may be singular. Several epochs or a sparse space use ``lst`` per
        element."""
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
        r"""
        The marginal distribution of :math:`R_a` or :math:`R_b`, whose transform is :math:`\Phi(s, 0)` or
        :math:`\Phi(0, s)` with :math:`\Phi` as in :class:`~phasegen.distributions.JointRewardDistribution`.

        :param which: The reward, ``'a'`` or ``'b'``.
        :return: The marginal distribution.
        :raises ValueError: If ``which`` is not ``'a'`` or ``'b'``.

        .. versionadded:: 2.0
        """
        if which not in ('a', 'b'):
            raise ValueError("`which` must be 'a' or 'b'.")

        return RewardDistribution(self._host, self.reward_a if which == 'a' else self.reward_b)

    @cached_property
    def _is_diagonal(self) -> bool:
        """Whether both reward vectors agree on every transient state, so ``R_a = R_b`` almost surely and the law has
        no density on the plane."""
        st = self._setup
        return np.array_equal(st['ra'], st['rb'])

    def moment(self, order_a: int = 1, order_b: int = 1, center: bool = False) -> float:
        r"""
        The cross-moment :math:`\mathbb{E}[R_a^{j_a} R_b^{j_b}]` of orders :math:`j_a, j_b \ge 0`, uncentered by
        default, which equals :math:`(-1)^{j_a + j_b}\, \partial^{j_a + j_b} \Phi / \partial s_a^{j_a}
        \partial s_b^{j_b}` at the origin. It is evaluated exactly by
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>` with
        :math:`j_a` copies of :math:`R_a` and :math:`j_b` copies of :math:`R_b`.

        :param order_a: The order :math:`j_a` of ``R_a``.
        :param order_b: The order :math:`j_b` of ``R_b``.
        :param center: Whether to center around the means.
        :return: The cross-moment.
        """
        rewards = (self.reward_a,) * order_a + (self.reward_b,) * order_b
        return float(MomentEvaluator.moment(
            self._host, k=order_a + order_b, rewards=rewards, center=center, permute=True
        ))

    @cached_property
    def mean(self) -> np.ndarray:
        r"""The pair of marginal means :math:`(\mathbb{E}[R_a], \mathbb{E}[R_b])`."""
        return np.array([self.moment(1, 0), self.moment(0, 1)])

    def cov(self) -> float:
        r"""The covariance :math:`\operatorname{Cov}(R_a, R_b) = \mathbb{E}[R_a R_b] - \mathbb{E}[R_a]\,
        \mathbb{E}[R_b]`."""
        return float(self.moment(1, 1, center=False) - self.moment(1, 0) * self.moment(0, 1))

    def corr(self) -> float:
        r"""The Pearson correlation
        :math:`\operatorname{corr}(R_a, R_b) = \operatorname{Cov}(R_a, R_b)/\sqrt{\operatorname{Var}(R_a)\,
        \operatorname{Var}(R_b)}` between :math:`R_a` and :math:`R_b`."""
        va = self.marginal('a')._cumulants()[1]
        vb = self.marginal('b')._cumulants()[1]
        return float(self.cov() / np.sqrt(va * vb))

    # ------------------------------------------------------------------------------------------------------------
    # joint CDF and density (2D Fourier-cosine), documented at JointCDF and JointDensity
    # ------------------------------------------------------------------------------------------------------------
    @cached_property
    def _atoms(self) -> dict:
        """The atoms ``a0 = P(R_a = 0)``, ``b0 = P(R_b = 0)`` and ``both0 = P(R_a = 0, R_b = 0)``, probed at
        ``_s_inf``."""
        big = self._s_inf
        return dict(a0=self.lst(big, 0.0).real, b0=self.lst(0.0, big).real, both0=self.lst(big, big).real)

    @cached_property
    def _cos_axis_coeffs(self) -> dict:
        """Cosine coefficients of the axis sub-distributions ``g_b`` (key ``'b'``, transform ``Phi(., inf)``) and
        ``g_a`` (key ``'a'``, transform ``Phi(inf, .)``) of ``JointCDF``, on the marginal's cumulant window with the
        marginal's term count."""
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
        """The axis sub-distribution ``g_b`` (``which='b'``) or ``g_a`` (``which='a'``) of ``JointCDF`` at ``xs``, equal
        to ``P(R_a = 0, R_b = 0)`` at 0."""
        c = self._cos_axis_coeffs[which]
        w, fk = c['w'], c['fk']
        xa = np.clip(np.asarray(xs, dtype=float), 0.0, c['b'])
        Fc = fk[0] * xa + (fk[1:] / w[1:]) @ np.sin(np.outer(w[1:], xa))
        return c['atom'] + c['cont'] * np.clip(Fc, 0.0, 1.0)

    @cached_property
    def _cos2d(self) -> dict:
        """The coefficient matrix ``A``, windows ``ba``, ``bb`` and frequencies ``ua``, ``ub`` of the 2D cosine
        expansion of ``JointCDF``, with the atoms removed by inclusion-exclusion and the Lanczos factors applied."""
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

        # Lanczos factors sinc(k / n_terms) damp the Gibbs ringing pinned to the origin edge. The k = 0 factor is 1 and
        # the antiderivatives of the k >= 1 terms vanish at the window edge, so the total box mass is unchanged.
        sigma = np.sinc(np.arange(n_terms) / n_terms)
        A *= np.outer(sigma, sigma)
        return dict(ba=ba, bb=bb, ua=ua, ub=ub, A=A)

    @cached_property
    def _cos2d_wiggle_check(self) -> float:
        """The largest near-origin gap between ``F(x, inf)`` of the cosine expansion and the marginal CDF of ``R_a``,
        logged as a warning above 0.03. The cosine error concentrates near the axes, so this margin comparison detects
        an under-resolved near-origin rise. It reads the coefficients directly, so it does not recurse into
        ``_cc_box``."""
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
                "marginal CDF by up to %.3g there. The joint CDF and density may be biased near the axes, where the "
                "cosine series cannot capture a sharp or skewed rise.", err,
            )
        return err

    def _density(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The continuous density of ``JointDensity`` on the outer grid ``xs x ys``: the mixed central difference of
        ``_cc_box`` on a coarse uniform grid spanning the queried range, interpolated by a bicubic spline. A spline's
        analytic mixed derivative or a fine grid overshoots near the origin, the coarse cell average does not."""
        from scipy.interpolate import RectBivariateSpline

        if Settings.check_inversions:
            _ = self._cos2d_wiggle_check
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
        out = RectBivariateSpline(gx[1:-1], gy[1:-1], dens, kx=k, ky=k)(xs, ys)
        out[xs < 0, :] = 0.0
        out[:, ys < 0] = 0.0
        return out

    @staticmethod
    def _cos_antideriv(u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``sin(u x) / u`` for every point and frequency, ``x`` where ``u = 0``, shape ``(len(x), len(u))``."""
        x = np.atleast_1d(x).astype(float)
        safe = np.where(u == 0, 1.0, u)
        out = np.sin(np.outer(x, u)) / safe
        out[:, u == 0] = x[:, None]
        return out

    def _cc_box(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The box probability ``C`` of ``JointCDF`` on the outer grid ``xs x ys``, integrated in closed form from the
        cosine coefficients."""
        if Settings.check_inversions:
            _ = self._cos2d_wiggle_check
        st = self._cos2d
        Ix = self._cos_antideriv(st['ua'], np.clip(xs, 0.0, st['ba']))   # (len_x, N)
        Iy = self._cos_antideriv(st['ub'], np.clip(ys, 0.0, st['bb']))   # (len_y, N)
        return Ix @ st['A'] @ Iy.T                                       # (len_x, len_y)

    def _cdf_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The joint CDF of ``JointCDF`` on the outer grid ``xs x ys``, zero where either threshold is negative."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        g_a, g_b = self._cos_axis('a', ys), self._cos_axis('b', xs)
        out = g_b[:, None] + g_a[None, :] - self._atoms['both0'] + self._cc_box(xs, ys)
        out[xs < 0, :] = 0.0
        out[:, ys < 0] = 0.0
        return out

    def conditional(self, on: str = 'a', value: float = 0.0) -> 'ConditionalRewardDistribution':
        r"""
        The distribution of the other reward given that the reward ``on`` equals ``value``, as a
        :class:`~phasegen.distributions.ConditionalRewardDistribution`, whose docstring describes its transform and
        inversion. A ``value`` of zero conditions on the atom of the reward ``on``, a positive ``value`` on its
        continuous part.

        :param on: The conditioning reward, ``'a'`` or ``'b'``.
        :param value: The conditioning value, non-negative.
        :return: The conditional distribution of the other reward.
        :raises ValueError: If ``on`` is not ``'a'`` or ``'b'``, if ``value`` is negative, if ``value`` is zero and the
            conditioning reward has a negligible atom, or if the density of the conditioning reward at ``value`` is
            below the resolution of the inversion.
        :raises NotImplementedError: If both rewards agree on every transient state, so that the conditional is a point
            mass at ``value``, or on a windowed coalescent.

        .. versionadded:: 2.0
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        if value is None or value < 0:
            raise ValueError("`value` must be non-negative.")
        if self._is_diagonal:
            raise NotImplementedError("The conditional of a self-pair is a point mass at `value` (R_a = R_b a.s.).")

        other_name = 'b' if on == 'a' else 'a'

        if value == 0:
            return _AtomConditional(self, on, f"R_{other_name} | R_{on} = 0")

        return _NestedConditional(self, on, float(value), f"R_{other_name} | R_{on} = {value:g}")

    def check_total_expectation(self, n_points: int = 32, tol: float = 0.01) -> dict:
        r"""
        Test the law of total expectation over the conditionals of
        :class:`~phasegen.distributions.ConditionalRewardDistribution`, and log a warning per conditioning reward when
        the relative error exceeds ``tol``.

        For each conditioning reward :math:`R_c \in \{R_a, R_b\}`, with :math:`R_o` the other reward,
        :math:`p_c = \mathbb{P}(R_c = 0)` its atom, :math:`F_c` its CDF and :math:`F_c^{-1}` its quantile function,

        .. math::

            \mathbb{E}[R_o] = p_c\,\mathbb{E}[R_o \mid R_c = 0]
            + (1 - p_c) \int_0^1 \mathbb{E}\big[R_o \mid R_c = F_c^{-1}(p_c + (1 - p_c)\,\xi)\big]\,\mathrm{d}\xi.

        The substitution :math:`F_c(v) = p_c + (1 - p_c)\,\xi` maps the continuous part of :math:`R_c`, with values
        :math:`v > 0`, onto the levels :math:`\xi \in (0, 1)` and absorbs its density into the measure, so only
        quantiles of :math:`R_c` are needed. The other checks place their conditioning values at these levels as well.
        The integral uses Gauss-Legendre quadrature with ``n_points`` nodes, and the atom term is omitted when
        :math:`p_c` is negligible. The integrand may vary sharply as :math:`\xi \to 1`, so the quadrature error can
        dominate for few nodes.

        All conditional checks treat a conditioning level whose conditional cannot be constructed alike, typically a
        level so far in the tail of :math:`R_c` that its density lies below the resolution of the inversion. The level
        is skipped, one warning per conditioning reward names the skipped levels and the reason, and the error is
        computed over the remaining levels. A quadrature is renormalised to the total weight of the remaining nodes.
        The error is infinite only when no conditional could be constructed.

        :param n_points: Number of Gauss-Legendre nodes per conditioning reward.
        :param tol: Relative error above which a warning is logged.
        :return: The relative error between the two sides of the identity per conditioning reward, keyed ``'a'`` and
            ``'b'``, infinite when no conditional could be constructed, or an empty dictionary when both rewards agree
            on every transient state.
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

            values, weights, skipped = [], [], []
            for u, weight in zip(us, ws):
                v = float(marg_on.quantile(p0 + (1.0 - p0) * float(u)))  # continuous part spans quantiles (p0, 1)
                try:
                    values.append(float(self.conditional(on, v)._cumulants()[0]))
                    weights.append(weight)
                except ValueError as e:
                    skipped.append((float(u), str(e)))

            self._warn_skipped(on, skipped, n_points, 'the law of total expectation')
            if not values:
                out[on] = float('inf')
                continue

            # the quadrature over the constructed nodes, renormalised to their total weight
            rhs = (1.0 - p0) * float(np.dot(weights, values) / np.sum(weights))
            if p0 > 1e-6:  # atom term P(R_on = 0) E[R_other | R_on = 0]
                rhs += p0 * float(self.conditional(on, 0.0)._cumulants()[0])

            rel = abs(rhs - lhs) / max(abs(lhs), 1e-12)
            out[on] = rel
            if Settings.check_inversions and rel > tol:
                self._logger.warning(
                    "Law of total expectation violated conditioning on R_%s: E[R_%s] = %.4g against "
                    "E[E[R_%s | R_%s]] = %.4g, a relative error of %.3g above the tolerance %.3g. The conditional "
                    "inversion may be imprecise here.", on, other, lhs, other, on, rhs, rel, tol
                )
        return out

    def _warn_skipped(self, on: str, skipped: list, n: int, what: str) -> None:
        """Log one warning for the conditioning levels whose conditional could not be constructed, with the first
        reason. The checks skip these nodes and compute their error over the others."""
        if skipped and Settings.check_inversions:
            self._logger.warning(
                "%d of %d conditionals on R_%s could not be constructed, at the conditioning levels %s, and are skipped "
                "in the check of %s. Reason: %s", len(skipped), n, on, [round(u, 4) for u, _ in skipped], what,
                skipped[0][1]
            )

    def check_total_probability(self, n_points: int = 8, n_y: int = 15, tol: float = 0.01) -> dict:
        r"""
        Test the law of total probability over the conditionals, and log a warning per conditioning reward when the
        largest deviation exceeds ``tol``.

        With :math:`R_c`, :math:`R_o`, :math:`p_c`, :math:`F_c^{-1}` and the levels :math:`\xi` as in
        :meth:`JointRewardDistribution.check_total_expectation()
        <phasegen.distributions.JointRewardDistribution.check_total_expectation>`, and :math:`F_o` the CDF of
        :math:`R_o`, the deviation is

        .. math::

            \max_{y} \Big| F_o(y) - p_c\,\mathbb{P}(R_o \le y \mid R_c = 0)
            - (1 - p_c) \sum_{j} w_j\,\mathbb{P}\big(R_o \le y \mid R_c = F_c^{-1}(p_c + (1 - p_c)\,\xi_j)\big) \Big|,

        where :math:`\xi_j` and :math:`w_j` are the ``n_points`` Gauss-Legendre nodes and weights on :math:`[0, 1]`, and
        :math:`y` runs over the quantiles of :math:`R_o` at ``n_y`` evenly spaced levels in the bulk. The whole
        conditional law enters, so errors that cancel in the mean remain visible. Each node builds the CDF of one
        conditional, and a node whose conditional cannot be constructed is handled as described at the expectation
        check.

        :param n_points: Number of Gauss-Legendre nodes per conditioning reward.
        :param n_y: Number of evaluation points :math:`y`.
        :param tol: Deviation, an absolute probability, above which a warning is logged.
        :return: The deviation per conditioning reward, keyed ``'a'`` and ``'b'``, infinite when no conditional could be
            constructed, or an empty dictionary when both rewards agree on every transient state.
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

            cdfs, weights, skipped = [], [], []
            for u, weight in zip(us, ws):
                v = float(marg_on.quantile(p0 + (1.0 - p0) * float(u)))  # continuous part spans quantiles (p0, 1)
                try:
                    cdfs.append(np.asarray(self.conditional(on, v).cdf(ys), dtype=float))
                    weights.append(weight)
                except ValueError as e:
                    skipped.append((float(u), str(e)))

            self._warn_skipped(on, skipped, n_points, 'the law of total probability')
            if not cdfs:
                out[on] = float('inf')
                continue

            # the quadrature over the constructed nodes, renormalised to their total weight
            rhs = (1.0 - p0) * (np.asarray(weights) @ np.asarray(cdfs)) / np.sum(weights)
            if p0 > 1e-6:  # atom term P(R_on = 0) F(y | R_on = 0)
                rhs = rhs + p0 * np.asarray(self.conditional(on, 0.0).cdf(ys), dtype=float)

            dev = float(np.max(np.abs(rhs - lhs)))
            out[on] = dev
            if Settings.check_inversions and dev > tol:
                self._logger.warning(
                    "Law of total probability violated conditioning on R_%s: the largest deviation of the mixed "
                    "conditional CDFs of R_%s from its marginal CDF is %.3g, above the tolerance %.3g. The conditional "
                    "inversion may be imprecise here.", on, other, dev, tol
                )
        return out

    #: Gauss-Legendre nodes used to average a conditional over a window (:meth:`window_average`). The conditional
    #: varies smoothly with the conditioning value, so a handful resolves the average; each node costs a conditional.
    _WINDOW_QUAD_NODES = 8

    def window_average(self, statistic, on: str = 'a', value: float = 0.0, half_width: float = 0.0,
                       n_nodes: int = None) -> 'float | np.ndarray':
        r"""
        A statistic of the conditionals averaged over a window of conditioning values, weighted by the density of the
        conditioning reward,

        .. math::

            \bar{g} = \frac{\int_W f_c(v)\, g(v)\,\mathrm{d}v}{\int_W f_c(v)\,\mathrm{d}v}, \qquad
            W = [v_0 - h,\ v_0 + h],

        where :math:`R_c` is the reward ``on`` with density :math:`f_c` on :math:`(0, \infty)`, :math:`g(v)` is
        ``statistic`` applied to the conditional distribution given :math:`R_c = v`, :math:`v_0` is ``value`` and
        :math:`h \ge 0` is ``half_width``, with :math:`v_0 - h > 0`. The integrals use Gauss-Legendre quadrature. For a
        statistic linear in the conditional law, such as the mean or the CDF at fixed points, :math:`\bar{g}` is that
        statistic of the other reward given :math:`R_c \in W`, the quantity estimated by
        :meth:`EmpiricalJointDistribution.conditional()
        <phasegen.distributions.EmpiricalJointDistribution.conditional>` with the same window.

        :param statistic: Callable taking a :class:`~phasegen.distributions.ConditionalRewardDistribution` and returning
            a scalar or a 1D array, for example ``lambda c: c.mean`` or ``lambda c: c.cdf(ys)``.
        :param on: The conditioning reward, ``'a'`` or ``'b'``.
        :param value: Centre :math:`v_0` of the window.
        :param half_width: Half-width :math:`h` of the window, in units of the conditioning reward.
        :param n_nodes: Number of Gauss-Legendre nodes, ``None`` for the default.
        :return: The window average, a float for a scalar statistic and an array of the statistic's shape otherwise.
        :raises ValueError: If the window reaches zero, where the conditioning reward may have an atom.
        :raises NotImplementedError: If both rewards agree on every transient state.
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
        values = np.array([np.asarray(statistic(self.conditional(on, float(ui))), dtype=float) for ui in u])
        average = np.tensordot(weights, values, axes=1) / weights.sum()

        return float(average) if average.ndim == 0 else average

    #: Level span of the conditioning points of the conditional checks, covering nearly the whole law.
    _COND_CHECK_SPAN = (0.01, 0.99)

    #: Error floor of the conditional checks as a fraction of the unconditional moment. Deep in the conditioning tail a
    #: conditional mean can vanish (for ``n = 4`` a long doubleton branch forces a tree without a three-leaf clade), and
    #: an unscaled relative error would divide two vanishing numbers.
    _COND_CHECK_FLOOR = 0.01

    def check_conditional_moments(self, n_points: int = 5, tol: float = 0.02, quantiles: 'Sequence[float]' = None,
                                  curves: int = 0) -> dict:
        r"""
        Compare the mean of each conditional with the derivative identity at a set of conditioning values, and log a
        warning per conditioning reward when the largest scaled error exceeds ``tol``.

        With :math:`R_c`, :math:`R_o`, :math:`p_c` and :math:`F_c^{-1}` as in
        :meth:`JointRewardDistribution.check_total_expectation()
        <phasegen.distributions.JointRewardDistribution.check_total_expectation>`, the conditioning values are
        :math:`v_j = F_c^{-1}(p_c + (1 - p_c)\,\xi_j)` for levels :math:`\xi_j` spread evenly over nearly all of
        :math:`(0, 1)`, or given by ``quantiles``. At each :math:`v_j` the mean :math:`\hat{m}_j` of the conditional
        transform, described at :class:`~phasegen.distributions.ConditionalRewardDistribution`, is compared with
        :math:`m_j = \mathbb{E}[R_o \mid R_c = v_j]` from the identity of :meth:`ConditionalRewardDistribution.moment()
        <phasegen.distributions.ConditionalRewardDistribution.moment>`, which involves no nested inversion. The scaled
        error is

        .. math::

            \delta_j = \frac{|\hat{m}_j - m_j|}{\max\big(|m_j|,\ \epsilon_f\,|\mathbb{E}[R_o]|\big)},

        with a small floor fraction :math:`\epsilon_f > 0` that keeps the error meaningful where the conditional mean
        vanishes, deep in the tail of :math:`R_c`. The identity inherits the error of the de Hoog inversion, which is
        largest on demographies with many epochs, so ``tol`` must exceed that error. The check tests the transform, not
        the distribution functions built from it (see :meth:`JointRewardDistribution.check_conditional_grid_moments()
        <phasegen.distributions.JointRewardDistribution.check_conditional_grid_moments>`). A value whose conditional
        cannot be constructed is handled as described at the expectation check.

        :param n_points: Number of conditioning values per conditioning reward. Ignored when ``quantiles`` is given.
        :param tol: Scaled error above which a warning is logged.
        :param quantiles: Levels :math:`\xi_j \in (0, 1)` within the continuous part of the conditioning reward.
        :param curves: Number of conditioning values per conditioning reward at which the conditional density is also
            evaluated and stored in ``conditional_densities``, for inspection only.
        :return: The largest scaled error per conditioning reward, keyed ``'a'`` and ``'b'``, infinite when no
            conditional could be constructed, or an empty dictionary when both rewards agree on every transient state.
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
                except ValueError as e:
                    refused.append((float(u), str(e)))
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

            out[on] = self._verdict(on, errs, refused, len(us), tol, 'the conditional mean',
                                    'the derivative identity')
        return out

    def _verdict(self, on: str, errs: list, refused: list, n: int, tol: float, what: str, against: str) -> float:
        """The worst error of one conditioning reward over the constructed conditioning points, with the skipped points
        warned about by ``_warn_skipped``. ``inf`` when no point could be constructed, so a check that resolved nothing
        cannot report a perfect score."""
        self._warn_skipped(on, refused, n, what)

        if not errs:
            return float('inf')

        rel = float(np.max(errs))
        if Settings.check_inversions and rel > tol:
            self._logger.warning(
                "%s disagrees with %s conditioning on R_%s: the largest scaled error over %d values is %.3g, above the "
                "tolerance %.3g. The conditional may be imprecise here.", what, against, on, len(errs), rel, tol
            )
        return rel

    def check_conditional_grid_moments(self, n_points: int = 3, tol: float = 0.02, k: int = 2,
                                       quantiles: 'Sequence[float]' = None) -> dict:
        r"""
        Integrate the evaluated CDF of each conditional into raw moments, and log a warning per conditioning reward when
        they deviate from the derivative identity by more than ``tol``.

        At the conditioning values :math:`v_j` of :meth:`JointRewardDistribution.check_conditional_moments()
        <phasegen.distributions.JointRewardDistribution.check_conditional_moments>` and for orders
        :math:`i = 1, \ldots, k`, the moments of the conditional CDF :math:`F_j`, as evaluated by its ``cdf``, are

        .. math::

            \hat{m}_{ji} = \int_0^{L_j} i\,y^{i-1}\big(1 - F_j(y)\big)\,\mathrm{d}y,

        by Simpson's rule on a uniform grid, where :math:`L_j` is the upper end of the support window of that CDF, and
        the atom at zero contributes nothing. They are compared with
        :math:`\mathbb{E}[R_o^i \mid R_c = v_j]` from :meth:`ConditionalRewardDistribution.moment()
        <phasegen.distributions.ConditionalRewardDistribution.moment>`, scaled as in the mean check with the floor
        :math:`\epsilon_f\,\mathbb{E}[R_o^i]`. Unlike the mean check, this reaches the distribution function itself.
        Orders above two weight the far tail, where a small absolute error of the CDF is a large relative one. A value
        whose conditional cannot be constructed is handled as described at the expectation check.

        :param n_points: Number of conditioning values per conditioning reward. Ignored when ``quantiles`` is given.
        :param tol: Scaled error above which a warning is logged.
        :param k: Highest moment order :math:`k`.
        :param quantiles: Levels within the continuous part of the conditioning reward, in :math:`(0, 1)`.
        :return: The largest scaled error over conditioning values and orders per conditioning reward, keyed ``'a'``
            and ``'b'``, infinite when no conditional could be constructed, or an empty dictionary when both rewards
            agree on every transient state.
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
                except ValueError as e:
                    refused.append((float(u), str(e)))
                    continue
                errs.append([abs(g - e) / max(abs(e), f, 1e-12) for g, e, f in zip(got, exact, floors)])
                kept.append(float(u))

            self.conditional_grid_moment_errors[on] = (np.array(kept), np.arange(1, k + 1),
                                                       np.array(errs).reshape(len(kept), k))
            flat = [e for row in errs for e in row]
            out[on] = self._verdict(on, flat, refused, len(us), tol, "the moments of the conditional's CDF",
                                    'the derivative identity')
        return out

    def _uncond_raw_moments(self, which: str, k: int) -> list:
        """The exact raw moments of orders ``1..k`` of one reward, setting the floor of
        ``check_conditional_grid_moments``."""
        reward = self.reward_a if which == 'a' else self.reward_b
        return [float(MomentEvaluator.moment(self._host, k=j, rewards=(reward,) * j, center=False))
                for j in range(1, k + 1)]

    @staticmethod
    def _grid_raw_moments(cond: RewardDistribution, k: int = 2, n: int = 4001) -> list:
        """The raw moments of orders ``1..k`` of a conditional's CDF by Simpson's rule on the survival function over the
        cosine fit's own window, beyond which the fit's survival is zero by construction."""
        b = float(cond.cdf._cos_coeffs['b'])
        ys = np.linspace(0.0, b, n)
        surv = 1.0 - np.asarray(cond.cdf(ys), float)
        return [float(simpson(j * ys ** (j - 1) * surv, x=ys)) for j in range(1, k + 1)]

    @staticmethod
    def _density_curve(cond: RewardDistribution) -> tuple:
        """The density of one conditional on a log-spaced grid up to ``Settings.plot_endpoint_quantile``, for
        ``conditional_densities``. Conditionals drawn together span orders of magnitude in scale."""
        # a conditional can be almost entirely the atom at 0 (conditioning on a huge R_on can leave the other bin empty
        # outright), and its endpoint quantile is then 0; fall back to the bracketed support so the curve has an axis
        top = float(cond.quantile(Settings.plot_endpoint_quantile)) or cond._range(scale=4.0)
        ys = np.geomspace(top * 1e-4, top, Settings.plot_n_grid)
        return ys, np.asarray(cond.pdf(ys), float)


class ConditionalRewardDistribution(RewardDistribution):
    r"""
    Distribution of one accumulated reward given the value of another, returned by
    :meth:`JointRewardDistribution.conditional() <phasegen.distributions.JointRewardDistribution.conditional>`.

    Write :math:`R_c` for the conditioning reward, :math:`R_o` for the other reward, :math:`v \ge 0` for the
    conditioning value, :math:`p_c = \mathbb{P}(R_c = 0)`, and :math:`\Phi(s_o, s_c) = \mathbb{E}[e^{-s_o R_o - s_c
    R_c}]` for the joint transform of :class:`~phasegen.distributions.JointRewardDistribution` with its arguments in
    this order. The conditional distribution is given by its Laplace-Stieltjes transform

    .. math::

        \varphi(s) = \mathbb{E}\big[e^{-sR_o} \mid R_c = v\big],

    and ``cdf``, ``pdf`` and ``quantile`` invert :math:`\varphi` by the construction described at
    :class:`~phasegen.distributions.RewardDistribution` and :class:`~phasegen.distributions.QuantileFunction`, including
    the atom
    :math:`p_0 = \varphi(\infty) = \mathbb{P}(R_o = 0 \mid R_c = v)`.

    For :math:`v = 0` the condition is the atom of :math:`R_c`, which requires :math:`p_c > 0`, and

    .. math::

        \varphi(s) = \frac{\Phi(s, \infty)}{\Phi(0, \infty)} = \frac{\mathbb{E}[e^{-sR_o};\ R_c = 0]}{p_c},

    evaluated at a large finite second argument as for the atoms of the joint distribution.

    For :math:`v > 0`, let :math:`f_c` be the density of :math:`R_c` on :math:`(0, \infty)`. Splitting :math:`\Phi` over
    the atom and the continuous part of :math:`R_c`,

    .. math::

        \Phi(s_o, s_c) = \mathbb{E}\big[e^{-s_o R_o};\ R_c = 0\big]
        + \int_0^\infty e^{-s_c y}\, \mathbb{E}\big[e^{-s_o R_o} \mid R_c = y\big]\, f_c(y)\,\mathrm{d}y,

    with :math:`y` the integration variable. The first term does not depend on :math:`s_c` and contributes nothing to
    the inverse Laplace transform in :math:`s_c` at :math:`v > 0`, so

    .. math::

        G(s) = \mathcal{L}^{-1}_{s_c}\big[\Phi(s, s_c)\big](v) = \mathbb{E}\big[e^{-sR_o} \mid R_c = v\big]\, f_c(v),
        \qquad \varphi(s) = \frac{G(s)}{G(0)},

    the transform form of :math:`f_{o \mid c}(x \mid v) = f_{oc}(x, v)/f_c(v)` for :math:`x > 0`, where
    :math:`f_{oc}` is the joint density of :math:`(R_o, R_c)` and :math:`G(0) = f_c(v)`.

    The inverse Laplace transform in :math:`s_c` is the Fourier-series method with Euler summation of Abate and Whitt
    (1995),

    .. math::

        G(s) \approx \frac{e^{\eta/2}}{2v} \sum_{j=-(N+m)}^{N+m} (-1)^j\, w_j\,
        \Phi\!\left(s, \frac{\eta + 2\pi \mathrm{i} j}{2v}\right),
        \qquad
        w_j = \begin{cases} 1, & |j| \le N, \\ 2^{-m} \sum_{l=|j|-N}^{m} \binom{m}{l}, & N < |j| \le N + m, \end{cases}

    where :math:`\eta > 0` sets the discretization error, of order :math:`e^{-\eta}`, :math:`N` truncates the series,
    and the last :math:`m + 1` partial sums are averaged with binomial weights. The nodes and weights do not depend on
    :math:`s`, so :math:`G` is a fixed linear combination of transform values and inherits the analyticity of
    :math:`\Phi` in :math:`s`, on which the outer inversion of :math:`\varphi` relies. :math:`N` is chosen once per
    conditional and held for every :math:`s`. It is doubled from a starting value until :math:`G(0)` is positive and
    changes by less than a fixed relative tolerance. If no truncation up to a maximum qualifies, the density of
    :math:`R_c` at :math:`v` lies below the resolution of the inversion and construction raises :class:`ValueError`.

    The upper end :math:`L` of the support window of the Fourier-cosine fit is found by bracketing: starting from the
    conditional mean, :math:`L` is multiplied by a fixed factor until the CDF, inverted pointwise by the de Hoog method,
    reaches a target probability close to one. The mean is :math:`-\varphi'(0)`, evaluated by a central difference of
    :math:`\varphi`, and higher moments are described at :meth:`ConditionalRewardDistribution.moment()
    <phasegen.distributions.ConditionalRewardDistribution.moment>`.

    .. warning::
        For :math:`v > 0` the transform is itself a numerical inversion, so the results carry a few correct digits
        rather than machine precision, least of all far in the tail of :math:`R_c` and on demographies with many
        epochs.

    .. rubric:: References

    Abate, J. and Whitt, W. (1995). Numerical inversion of Laplace transforms of probability distributions. ORSA
    Journal on Computing 7(1), 36-43.

    .. versionadded:: 2.0
    """
    #: The conditioning value. Overridden by ``_NestedConditional``, the atom conditions on ``R_on = 0``.
    _value: float = 0.0

    def lst(self, s: complex) -> complex:
        r"""
        The conditional transform :math:`\varphi(s)` defined at
        :class:`~phasegen.distributions.ConditionalRewardDistribution`.

        :param s: The argument.
        :return: The transform at ``s``.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        raise NotImplementedError

    @cached_property
    def mean(self) -> float:
        r"""The mean :math:`\mathbb{E}[R_o \mid R_c = v] = -\varphi'(0)`, with the notation of
        :class:`~phasegen.distributions.ConditionalRewardDistribution`, by a central difference of the conditional
        transform."""
        return float(self._cumulants()[0])

    def _raw_moments(self, k: int = 2) -> list:
        """
        The raw moments of orders ``1..k`` by the derivative identity of ``ConditionalRewardDistribution.moment``, with
        ``k + 1`` de Hoog inversions of the Taylor coefficients of ``JointRewardDistribution.lst_taylor`` sharing one
        node set. Differencing the transform instead would feed roundoff into the de Hoog recurrence. On several epochs
        the de Hoog error is a pointwise noise band in the conditioning value, not a smooth bias.

        :param k: Highest order.
        :return: ``[E[R_o | R_c = v], ..., E[R_o^k | R_c = v]]``.
        :raises NotImplementedError: For the atom conditional, where the identity has no continuous density to divide
            by.
        :raises ValueError: If the density of the conditioning reward at the value is not resolvable.
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
        r"""
        The variance :math:`\operatorname{Var}(R_o \mid R_c = v) = \mathbb{E}[R_o^2 \mid R_c = v] - \mathbb{E}[R_o \mid
        R_c = v]^2`, with the notation of :class:`~phasegen.distributions.ConditionalRewardDistribution`, from the
        mean and the second moment of :meth:`ConditionalRewardDistribution.moment()
        <phasegen.distributions.ConditionalRewardDistribution.moment>`, truncated at zero.
        """
        if self._value == 0.0:
            return float(self._cumulants()[1])

        return max(self.moment(2) - float(self.mean) ** 2, 0.0)

    def moment(self, k: int) -> float:
        r"""
        The raw moment :math:`\mathbb{E}[R_o^k \mid R_c = v]` of order :math:`k \ge 1`, with the notation of
        :class:`~phasegen.distributions.ConditionalRewardDistribution`.

        The first moment is the mean :math:`-\varphi'(0)`. For :math:`v > 0` and :math:`k \ge 2`, differentiating the
        decomposition of :math:`\Phi` over the atom and the continuous part of :math:`R_c` gives

        .. math::

            \frac{\partial^k \Phi}{\partial s_o^k}(0, s_c) = (-1)^k \Big( \mathbb{E}\big[R_o^k;\ R_c = 0\big]
            + \int_0^\infty e^{-s_c y}\, \mathbb{E}\big[R_o^k \mid R_c = y\big]\, f_c(y)\,\mathrm{d}y \Big),

        where the first term does not depend on :math:`s_c`. Inverting in :math:`s_c` at :math:`v > 0` therefore gives

        .. math::

            \mathbb{E}[R_o^k \mid R_c = v] = \frac{k!\,(-1)^k\,\mathcal{L}^{-1}\big[\Phi_k\big](v)}
            {\mathcal{L}^{-1}\big[\Phi_0\big](v)},

        where :math:`\Phi_j(s_c)` is the coefficient of :math:`s_o^j` in the Taylor expansion of :math:`\Phi(s_o, s_c)`
        about :math:`s_o = 0`, computed without truncation error by :meth:`JointRewardDistribution.lst_taylor()
        <phasegen.distributions.JointRewardDistribution.lst_taylor>`, and the denominator equals :math:`f_c(v)`. Both
        inverse transforms use the de Hoog method of :class:`~phasegen.distributions.RewardDistribution` once, without
        nesting, so this identity is independent of the conditional transform. Its accuracy is
        that of the de Hoog method, which degrades on demographies with many epochs.

        For :math:`v = 0` the identity has no density to divide by, and the second moment is
        :math:`\varphi''(0)`, evaluated by central differences. Higher orders are not available there.

        :param k: Order :math:`k` of the moment.
        :return: The raw moment of order ``k``.
        :raises ValueError: If ``k`` is below 1, or if the density of the conditioning reward at :math:`v` is not
            resolvable.
        :raises NotImplementedError: If ``k`` exceeds 2 for :math:`v = 0`.
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
        """Support upper end by bracketing the de Hoog CDF (``_range_via_cdf``), memoised per ``scale``. The cumulant
        variance of the nested transform is unusable, its second difference collapses to the floor."""
        cache = self.__dict__.setdefault('_range_cache', {})
        if scale not in cache:
            cache[scale] = self._range_via_cdf(scale)
        return cache[scale]

    def _range_via_cdf(self, scale: float = 12.0, n_iter: int = 80) -> float:
        """Grow ``b`` by a fixed factor from the conditional mean, or the unconditional mean of the same reward when the
        conditional is essentially its atom, until the de Hoog CDF reaches ``min(1 - exp(-scale), 1 - 1e-6)``. The
        bracket only grows, so the seed must lie below the support, which a mean does in every unit of time."""
        target = min(1.0 - float(np.exp(-scale)), 1.0 - 1e-6)
        # the per-point de Hoog CDF, not ``self.cdf``, whose cosine fit needs this very window and would recurse
        cdf_point = self.cdf._cdf_point

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
    """The conditional on the atom ``R_on = 0`` of ``ConditionalRewardDistribution``: the sub-transform at an infinite
    conditioning argument, divided by the atom mass. Its own atom is ``P(R_a = 0, R_b = 0) / P(R_on = 0)``."""
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, joint: 'JointRewardDistribution', on: str, label: str = '') -> None:
        atom = joint._atoms['a0' if on == 'a' else 'b0']
        if atom < 1e-9:
            raise ValueError(f"Cannot condition on R_{on} = 0: it has (near) zero probability.")
        self._joint = joint
        # bound before ``_s_inf`` is read, which scales the probe by the host's time scale
        self._host = joint._host
        big = self._s_inf
        self.state_space = joint._host.state_space
        self._on = on
        self._atom = atom
        self._sub = (lambda s: joint.lst(big, s)) if on == 'a' else (lambda s: joint.lst(s, big))
        self._logger = logger.getChild(self.__class__.__name__)
        self.label = label

    def lst(self, s: complex) -> complex:
        """The conditional transform on the atom, see ``ConditionalRewardDistribution``."""
        return self._sub(s) / self._atom




_PADE13 = np.array([64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
                    129060195264000., 10559470521600., 670442572800., 33522128640.,
                    1323241920., 40840800., 960960., 16380., 182., 1.])


def _expm_batch(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a stack ``(k, n, n)`` by Pade-13 with scaling and squaring, vectorised over the leading
    axis, for the Euler node batch of ``_lst_from_shift_batch``. The per-call analysis of ``scipy.linalg.expm``
    dominates at the small sizes of this batch."""
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
    """``_lst_from_shift`` over a stack of shift vectors ``(k, nt)``, sharing the per-epoch assembly and exponentiating
    the batch with ``_expm_batch``. The last-epoch solve uses the LU of the scalar path per shift."""
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


#: Starting Fourier truncation of the Euler inversion, refined by ``_NestedConditional._calibrate``.
_EULER_N0 = 30


def _euler_invert(transform, t: float, A: float = 16.0, N0: int = _EULER_N0, m: int = 12) -> complex:
    """Euler-summed Fourier-series inversion of ``transform`` at ``t``, the inner inversion of
    ``ConditionalRewardDistribution`` (``A`` is its damping, ``N0`` its truncation, ``m`` its Euler terms). The node
    spacing ``2 pi / (2t)`` makes the series alternating, as Euler summation requires. It is summed two-sided, so a
    complex-valued inverse needs no conjugate symmetry. A larger ``A`` amplifies roundoff by ``exp(A / 2)``, the
    remaining error is truncation, reduced by ``N0``."""
    ks = np.arange(-(N0 + m), N0 + m + 1)
    u = (A + 2.0j * np.pi * ks) / (2.0 * t)
    binom = np.array([comb(m, j) for j in range(m + 1)], dtype=float) / 2.0 ** m
    # Euler weight of node k: the binomial-averaged fraction of the partial sums S_{N0+j} that include it
    frac = np.array([binom[max(0, abs(k) - N0):].sum() if abs(k) > N0 else 1.0 for k in ks])
    w = (np.exp(A / 2.0) / (2.0 * t)) * ((-1.0) ** ks) * frac
    vals = transform(u)  # the whole node set at once -- see _lst_from_shift_batch
    return complex(np.sum(w * np.asarray(vals)))


class _NestedConditional(ConditionalRewardDistribution):
    """The conditional on a value ``R_on = value > 0`` of ``ConditionalRewardDistribution``: ``phi(s) = G(s) / G(0)``
    with ``G`` the Euler inversion along the conditioning axis. The truncation ``N0`` is calibrated once at
    construction and held for every ``s``, since a truncation varying with ``s`` would break the analyticity of ``G``
    in ``s`` that the outer inversion needs."""
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, joint: 'JointRewardDistribution', on: str, value: float, label: str = '') -> None:
        self._joint = joint
        self._host = joint._host
        self.state_space = joint._host.state_space
        self._on = on
        self._value = float(value)
        self._logger = logger.getChild(self.__class__.__name__)
        self.label = label

        self._N0, self._G0 = self._calibrate()

    def _calibrate(self, tol: float = 2e-2, n_max: int = 480) -> tuple:
        """
        The Euler truncation ``N0``, doubled from ``_EULER_N0`` until ``G(0)`` moves by at most ``tol`` relatively,
        and ``G(0)`` at it. A sharply peaked density needs a large truncation, an easy case converges at the first
        step, and a loose ``tol`` suffices because ``G(0)`` only normalises. The ``cur > 0`` condition is essential: a
        resolved density stays positive under refinement, so a sign change shows the inversion returns noise, as deep
        in the tail of a strong bottleneck, where the density falls below float64 inversion resolution. Refusing
        there is correct, the distribution itself is well defined.

        :param tol: Relative change of ``G(0)`` below which the truncation counts as converged.
        :param n_max: Largest truncation tried.
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
        """``Phi`` along the conditioning axis with the other argument at 0, the transform behind ``G(0)``."""
        z = np.zeros(len(u))
        return self._joint.lst_batch(z, u) if self._on == 'b' else self._joint.lst_batch(u, z)

    def _G(self, s: complex) -> complex:
        """``G(s)``, the Euler inversion of ``Phi`` along the conditioning axis at the value. The inner method must be
        accurate on peaked coalescent densities (Gaver-Stehfest is not), a fixed linear functional so that ``G`` stays
        analytic in ``s`` for the outer de Hoog recurrence (a nested de Hoog is not), and use a vertical contour, since
        the epoch exponentials overflow as the real part tends to minus infinity (Talbot's contour does)."""
        if self._on == 'b':
            return _euler_invert(lambda u: self._joint.lst_batch(np.full(len(u), s), u), self._value, N0=self._N0)
        return _euler_invert(lambda u: self._joint.lst_batch(u, np.full(len(u), s)), self._value, N0=self._N0)

    def lst(self, s: complex) -> complex:
        """The conditional transform ``G(s) / G(0)``, see ``ConditionalRewardDistribution``."""
        return self._G(complex(s)) / self._G0
