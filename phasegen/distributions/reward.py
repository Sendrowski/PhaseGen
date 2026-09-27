"""
Distributions of accumulated rewards obtained from their Laplace transforms: the 1D
:class:`~phasegen.distributions.RewardDistribution`, the bivariate
:class:`~phasegen.distributions.JointRewardDistribution` and the
:class:`~phasegen.distributions.ConditionalRewardDistribution`.
"""
import logging
from math import comb, factorial
from typing import Any, TYPE_CHECKING, Optional, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.integrate import simpson

from ..caching import cached_property
from ..errors import ModelError
from ..rewards import Reward
from ..settings import Settings
from .base import CallableDistributionFunctions, JointDensity, JointCDF, \
    ConditionalDensity, ConditionalCDF, ConditionalQuantileFunction, \
    _LSTCumulativeDistributionFunction, _LSTDensityFunction, _LSTQuantileFunction
from ._moments import MomentEvaluator, _AUTO_PERM

if TYPE_CHECKING:
    from .phase_type import PhaseTypeDistribution

logger = logging.getLogger('phasegen')

#: Mass beyond the window of the 2D cosine expansion of a joint distribution above which a warning is logged.
_COS2D_TAIL_WARN = 1e-2

#: Relative distance from an epoch time within which a conditioning value is warned about, the density of a reward
#: accrued at one rate jumping there.
_JUMP_WARN = 0.05

#: Smallest atom treated as a positive probability. The probe at ``_s_inf`` exceeds a zero atom by up to about 1e-8.
_ATOM_FLOOR = 1e-6

#: Starting Fourier truncation of the Euler inversion, refined by ``_NestedConditional._calibrate`` and
#: ``_NestedConditional._refine``.
_EULER_N0 = 30

#: Largest Fourier truncation of the Euler inversion tried by ``_NestedConditional``.
_EULER_N0_MAX = 480


class RewardDistribution(CallableDistributionFunctions):
    r"""
    Distribution of the accumulated reward :math:`R` from time 0 to absorption, with the notation of
    :class:`~phasegen.distributions.PhaseTypeDistribution`. It is returned by :meth:`Coalescent.distribution()
    <phasegen.distributions.Coalescent.distribution>`, :meth:`PhaseTypeDistribution.distribution()
    <phasegen.distributions.PhaseTypeDistribution.distribution>` and the ``bin()`` methods of the spectra. The mean and
    variance of :math:`R` are exact moments, but across several epochs its distribution has no closed matrix form. The
    transform :math:`\varphi` given by :meth:`RewardDistribution.lst() <phasegen.distributions.RewardDistribution.lst>`
    is exact, requires a single linear solve and determines the distribution uniquely, so the ``cdf``, ``pdf`` and
    ``quantile`` are obtained by inverting it numerically. The tree height is the exception, as its distribution
    follows directly from matrix exponentials (see :class:`~phasegen.distributions.TreeHeightDistribution`).

    .. rubric:: Fourier-cosine expansion

    The reward is zero with probability :math:`p_0 = \lim_{s \to \infty} \varphi(s)`, for example when an SFS bin is
    empty. The CDF of the remaining continuous part is expanded in :math:`K` cosine terms on a window
    :math:`[0, \beta]` (Fang and Oosterlee, 2008),

    .. math::

        F(x) = p_0 + (1 - p_0) \Big[\frac{x}{\beta}
        + \sum_{j=1}^{K-1} \frac{\beta A_j}{j \pi} \sin\frac{j \pi x}{\beta}\Big].

    The coefficients evaluate the transform on the imaginary axis, :math:`s = -\mathrm{i}\omega`, where
    :math:`\varphi(-\mathrm{i}\omega) = \mathbb{E}[e^{\mathrm{i}\omega R}]` is the characteristic function at the
    frequency :math:`\omega`, here at the frequencies :math:`\omega = j \pi / \beta`,

    .. math::

        A_j = \frac{2}{\beta}\, \mathrm{Re}\, \frac{\varphi(-\mathrm{i} j \pi / \beta) - p_0}{1 - p_0}.

    These :math:`K` transform evaluations give the whole curve, and :math:`K` is given by
    :attr:`Settings.cos_terms <phasegen.settings.Settings.cos_terms>`. The expansion reaches 1 at :math:`\beta`, so
    the mass beyond the window is lost, and it resolves features no narrower than :math:`\beta / K`.

    .. rubric:: Tail

    Above the CDF level set by :attr:`Settings.dehoog_tail_quantile
    <phasegen.settings.Settings.dehoog_tail_quantile>`, the CDF is evaluated pointwise as the inverse transform of
    :math:`\varphi(s)/s` by the method of de Hoog et al. (1982),

    .. math::

        F(x) \approx \frac{e^{\gamma x}}{2x}\, \mathrm{Re} \sum_{l=0}^{2D} w_l\,
        \frac{\varphi(z_l)}{z_l}\, e^{\mathrm{i} l \pi / 2},

    where the transform is evaluated at the nodes :math:`s = z_l = \gamma + \mathrm{i} l \pi / (2x)`. They lie on a
    vertical line in the complex plane, and their real part :math:`\gamma > 0` damps the function being inverted by
    :math:`e^{-\gamma x}`, which keeps the series convergent. The weights are :math:`w_0 = 1/2` and :math:`w_l = 1` otherwise, and :math:`D` is given by
    :attr:`Settings.dehoog_degree <phasegen.settings.Settings.dehoog_degree>`. Read as a power series in
    :math:`e^{\mathrm{i} \pi / 2}`, the sum converges slowly, so it is replaced by its Padé approximant: a continued
    fraction that matches its first :math:`2D + 1` terms, with coefficients from the quotient-difference algorithm.

    .. rubric:: Implementation

    - The window is chosen in two passes. A first expansion over several standard deviations of :math:`R` locates the
      support, and the second window ends where the first expansion comes close to 1. The window width is what the
      expansion resolves, no feature narrower than :math:`\beta / K`.
    - The ``cdf``, ``pdf`` and ``quantile`` are read from one cumulative-hazard grid, described at
      :class:`~phasegen.distributions.QuantileFunction`, of expansion nodes below the tail level and de Hoog nodes
      above it. The de Hoog nodes are computed only when a query reaches the tail, and they are kept.
    - The atom is evaluated at a large real :math:`s` scaled with the time unit of the transform.
    - With :attr:`Settings.check_inversions <phasegen.settings.Settings.check_inversions>`, a warning is logged when
      the expansion is not monotone, and when its truncation error, estimated from how much the last :math:`K/2`
      terms move the CDF and from the decay of the coefficients, exceeds :math:`10^{-3}`, which a distribution whose
      body is narrow against its window does. Raising :attr:`Settings.cos_terms
      <phasegen.settings.Settings.cos_terms>` resolves it, at a cost linear in :math:`K`.

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
        """The inversion time scale of :func:`time_scale`, read straight from the host. Decoupled from :attr:`_setup` so
        the conditional flavours -- whose ``lst`` is a nested transform with no state-space reward to bind -- can scale
        their atom probe and inversion contour without invoking ``_setup``.

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

        The argument :math:`s` is conjugate to :math:`R`, so the probe is
        :math:`10^8/\min(\tau, \sqrt{\mathbb{E}[R^2]})`, with :math:`\tau` the inversion time scale of
        ``_time_scale`` and :math:`\sqrt{\mathbb{E}[R^2]}` the root mean square of ``_rms``. This keeps :math:`sR` at
        the same large multiple in every regime. A reward that is zero almost surely takes the time scale in place of
        the root mean square.
        """
        return 1e8 / min(self._time_scale, self._rms or self._time_scale)

    @cached_property
    def mean(self) -> float:
        r"""Mean :math:`\mathbb{E}[R]` of the accumulated reward, evaluated by
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`."""
        # go through the moment engine directly: a spectrum host overrides ``moment`` to return a whole SFS, which
        # would break ``float()`` for a single-bin reward
        return float(MomentEvaluator.moment(self._host, k=1, rewards=(self.reward,), center=False))

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

        .. rubric:: Single epoch

        While the process occupies state :math:`x`, the weight :math:`e^{-sR}` decays at rate :math:`s\, r(x)`, so the
        reward enters as a diagonal shift of the sub-intensity matrix. With the reward vector restricted to the
        transient states,

        .. math::

            \varphi(s) = \boldsymbol{\alpha}_T \big(s \operatorname{diag}(\mathbf{r}) - \mathbf{T}\big)^{-1} \mathbf{q},

        the transform of the reward-transformed phase-type distribution (Hobolth et al., 2019).

        .. rubric:: Several epochs

        Each bounded epoch propagates the weighted transient probabilities together with the absorbed weight, which
        stays constant after absorption. With the block matrix

        .. math::

            \mathbf{G}_i(s) =
            \begin{pmatrix} \mathbf{T}_i - s \operatorname{diag}(\mathbf{r}) & \mathbf{q}_i \\ \mathbf{0} & 0
            \end{pmatrix},

        the bounded epochs are chained and the unbounded last epoch is closed with the single-epoch formula,

        .. math::

            \varphi(s) = (\boldsymbol{\alpha}_T, 0) \prod_{i=1}^{M-1} e^{\mathbf{G}_i(s) \Delta_i}
            \begin{pmatrix} (s \operatorname{diag}(\mathbf{r}) - \mathbf{T}_M)^{-1} \mathbf{q}_M \\ 1 \end{pmatrix}.

        .. rubric:: Implementation

        - The exponentials are formed densely. The last-epoch system is solved by a sparse LU factorization from
          :attr:`Settings.closed_form_sparse_min_states <phasegen.settings.Settings.closed_form_sparse_min_states>`
          transient states on, and by a dense one below.
        - Transient states that the initial vector cannot reach in any epoch carry no mass and are left out.
        - When the largest transition rate is far from 1, time is measured in units of its inverse, which leaves
          :math:`\varphi` unchanged and keeps the shifted matrices well scaled.

        .. rubric:: References

        Hobolth, A., Siri-Jégousse, A. and Bladt, M. (2019). Phase-type distributions in population genetics.
        Theoretical Population Biology 127, 16-32.

        :param s: The argument, with non-negative real part, or purely imaginary for the characteristic function
            :math:`\varphi(-\mathrm{i}\omega) = \mathbb{E}[e^{\mathrm{i}\omega R}]` at the frequency
            :math:`\omega \in \mathbb{R}`.
        :return: The transform at ``s``.
        :raises NotImplementedError: If the reward does not assign one value per state, or if the coalescent has a
            bounded accumulation window.
        :raises ValueError: If the reward is negative, or if some state carrying mass can never reach a common
            ancestor in the final epoch.
        """
        self._host._assert_not_windowed()
        st = self._setup
        _assert_lst_absorbs(self._host)
        # evaluate against the tau-scaled generators at s*tau (R -> R/tau); the result equals the unscaled phi(s)
        # exactly but stays well-conditioned for large N (see ``time_scale``)
        return _lst_from_shift((s * st['tau']) * st['r'], st['alpha'], st['T_epochs'], st['sparse'], st['lu_perm'])

    def _invert(self, transform, t: float) -> float:
        r"""
        De Hoog inversion of ``transform`` at ``t``, described at ``RewardDistribution``. It runs in the time unit
        :math:`\zeta` of ``_time_scale``, using
        :math:`g(t) = \zeta^{-1} \mathcal{L}^{-1}[\sigma \mapsto G(\sigma / \zeta)](t / \zeta)` for a transform
        :math:`G` with inverse :math:`g`, so the contour nodes stay of order one.

        :param transform: The transform to invert, a function of a complex argument.
        :param t: The point at which to evaluate the inverse.
        :return: The inverse at ``t``, 0 for ``t <= 0``.
        """
        if t <= 0:
            return 0.0

        tau = self._time_scale

        return _dehoog_invert(lambda s: transform(s / tau), t / tau, Settings.dehoog_degree) / tau

    def _titled(self, base: str) -> str:
        """A plot title incorporating :attr:`label` (e.g. ``"SFS bin 3 CDF"``) when one has been set. Used by the
        function objects (the :class:`~phasegen.distributions.base._LSTFunction` family) for their plot titles."""
        return f"{self.label} {base}" if self.label else base

    @property
    def _rms(self) -> float:
        r"""The root mean square :math:`\sqrt{\mathbb{E}[R^2]}` from the exact moments, the scale of the step of
        :meth:`_cumulants`."""
        return float(np.sqrt(self.var + self.mean ** 2))

    #: Step of :meth:`_cumulants` relative to the root mean square, set by the precision of the transform.
    _cumulant_step: float = 1e-4

    def _cumulants(self) -> tuple:
        r"""Mean and variance of the accumulated reward from the transform near 0 (:math:`\varphi(0) = 1`):
        :math:`c_1 = -\varphi'(0)`, :math:`c_2 = \varphi''(0) - \varphi'(0)^2`, by central differences with the step
        :math:`h = \epsilon / \sqrt{\mathbb{E}[R^2]}` for the relative step :math:`\epsilon` of ``_cumulant_step``.
        The step keeps ``phi(-h) = E[e^{h R}]`` inside the region of convergence unless a tail slower than ``1/h``
        carries a negligible mass. A reward that is zero almost surely takes the time scale in place of the root mean
        square."""
        h = self._cumulant_step / (self._rms or self._time_scale)
        d1 = (self.lst(h).real - self.lst(-h).real) / (2 * h)
        d2 = (self.lst(h).real - 2.0 + self.lst(-h).real) / h ** 2
        return -d1, max(d2 - d1 ** 2, 0.0)

    def _range(self, scale: float = 12.0) -> float:
        r"""An upper end for the support, :math:`\mathbb{E}[R] + \text{scale}\cdot\operatorname{std}(R)` from the
        exact moments, for the cosine window and default plot grids. A reward that is zero almost surely takes the
        time scale.

        :raises NotImplementedError: If the reward is not a scalar reward.
        :raises ValueError: If the reward is negative.
        """
        _ = self._setup  # validates the reward before the moment engine reads it
        return float(self.mean + scale * self.std) or self._time_scale


def _build_epoch_data(host) -> dict:
    """
    The reward-independent ingredients of the accumulated-reward transform: the transient states reachable from the
    initial vector, the initial vector and the per-epoch transient sub-generators on those states. Shared across all
    bins of a spectrum (the generators depend only on the state space and demography, not on which reward is
    accumulated), so it is built once on the host.
    """
    ss = host.state_space
    transient = np.where(~ss.absorbing)[0]

    blocks = []
    for i_epoch, epoch in enumerate(host._get_epochs_until_unbounded()):
        ss.update_epoch(epoch)
        host._check_numerical_stability(ss.S, i_epoch)
        blocks.append((host._transient_block(transient, sparse=True), epoch.start_time, epoch.end_time))

    # the states that carry mass in some epoch: the closure of the initial support under each epoch's transitions in
    # turn. The others, such as migration targets of a zero rate, may never absorb and would make the last-epoch
    # system singular at s = 0.
    reach = np.asarray(ss.alpha)[transient] > 0
    for T, _, _ in blocks:
        adj = (T != 0).T.tocsr()
        while True:
            nxt = reach | (adj @ reach > 0)
            if np.array_equal(nxt, reach):
                break
            reach = nxt

    idx = transient[reach]
    alpha = np.asarray(ss.alpha)[idx].astype(float)
    nt = len(idx)
    # ``sparse`` gates the sparse matrix build and the sparse block-triangular LU of the final-epoch solve (which is
    # the large-space win and handles the s->inf atom shift directly). The finite-epoch matrix-exponential is always
    # dense (densifying a sparse block): the only alternative, the expm_multiply *action*, is norm-driven and cannot
    # evaluate the ``s = inf`` (1e8) atom shifts that every inversion needs -- so it has no usable role here (unlike
    # the moment path, which never inverts an atom).
    sparse = MomentEvaluator._solve_sparse(nt)

    T_epochs = []
    for T, t0, t1 in blocks:
        sub = T[reach][:, reach]
        T_epochs.append((sub.tocsc() if sparse else sub.toarray(), t0, t1))

    # the block-triangular ordering of the final (unbounded) epoch's sub-generator depends only on its sparsity
    # pattern, which is fixed across the many shifted solves of the de Hoog inversion; compute it once here so the
    # per-node factorization can reuse it (the SCC analysis is the dominant cost of the sparse solve)
    lu_perm = MomentEvaluator._block_triangular_order(T_epochs[-1][0]) if sparse else None

    return dict(idx=idx, alpha=alpha, nt=nt, sparse=sparse, T_epochs=T_epochs, lu_perm=lu_perm)


def _assert_lst_absorbs(host) -> None:
    """
    Raise unless every state carrying mass can still reach absorption in the final epoch, which is what makes the
    shifted final-epoch system invertible at ``s = 0``. The state selection of :func:`_build_epoch_data` is a forward
    closure of the initial vector and so cannot establish this; the backward reachability of the moment engine can.
    Without it the final-epoch solve is singular and the transform is ``nan``, surfacing much later as an opaque
    failure of the inversion.

    :param host: The phase-type distribution whose state space and demography are checked.
    :raises ModelError: if some state carrying mass can never reach a common ancestor.
    """
    # one backward reachability per host, memoized: an SFS evaluates many bins through the same state space
    absorbs = getattr(host, '_lst_absorbs_cache', None)

    if absorbs is None:
        idx = host._reward_epoch_data['idx']
        host.state_space.update_epoch(host._get_epochs_until_unbounded()[-1])
        _, reach_absorption = MomentEvaluator._reaches_absorption(host)
        absorbs = host._lst_absorbs_cache = bool(reach_absorption[idx].all())

    if absorbs:
        return

    raise ModelError(
        "The demography does not absorb: some states carrying probability mass can never reach a common "
        "ancestor in the final (unbounded) epoch, so the accumulated reward has no proper distribution. This "
        "typically means a deme is isolated or migration is one-way/blocked in the last epoch, leaving lineages "
        "that can never coalesce. Check the migration structure of the last epoch."
    )


def time_scale(host) -> float:
    """
    The time unit of the transform, described at :meth:`RewardDistribution.lst()
    <phasegen.distributions.RewardDistribution.lst>`: the inverse of the largest total transition rate of a transient
    state reachable from the initial vector, over all epochs, when it lies outside ``[1e-2, 1e2]``, and 1 otherwise.

    :param host: The phase-type distribution whose state space and demography set the unit.
    :return: The time unit.
    """
    rate = max(float(np.max(-np.asarray(T.diagonal()), initial=0.0)) for T, _, _ in host._reward_epoch_data['T_epochs'])
    tau = 1.0 / rate if rate > 0 else 1.0
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
    Joint distribution of two rewards :math:`R_a` and :math:`R_b` accumulated until absorption, with the notation of
    :class:`~phasegen.distributions.PhaseTypeDistribution`. It is returned by
    :meth:`PhaseTypeDistribution.joint_distribution() <phasegen.distributions.PhaseTypeDistribution.joint_distribution>`
    and the accessors built on it.

    The distribution is determined by the bivariate Laplace-Stieltjes transform
    :math:`\Phi(s_a, s_b) = \mathbb{E}[e^{-s_a R_a - s_b R_b}]`. For a single epoch, with the reward vectors
    :math:`\mathbf{r}_a` and :math:`\mathbf{r}_b` restricted to the transient states,

    .. math::

        \Phi(s_a, s_b) = \boldsymbol{\alpha}_T
        \big(\operatorname{diag}(s_a \mathbf{r}_a + s_b \mathbf{r}_b) - \mathbf{T}\big)^{-1} \mathbf{q}.

    Both rewards enter as one diagonal shift, so this is the transform of :meth:`RewardDistribution.lst()
    <phasegen.distributions.RewardDistribution.lst>` with :math:`s\,\mathbf{r}` replaced by
    :math:`s_a \mathbf{r}_a + s_b \mathbf{r}_b`, and several epochs are chained as described there.

    .. rubric:: Atoms and moments

    A reward that can be zero puts mass on an axis. Infinite arguments give the atoms

    .. math::

        \mathbb{P}(R_a = 0) = \Phi(\infty, 0), \qquad
        \mathbb{P}(R_b = 0) = \Phi(0, \infty), \qquad
        \mathbb{P}(R_a = R_b = 0) = \Phi(\infty, \infty).

    Mixed moments are derivatives of :math:`\Phi` at the origin and are computed exactly by
    :meth:`JointRewardDistribution.moment() <phasegen.distributions.JointRewardDistribution.moment>`.

    .. rubric:: Implementation

    - The joint CDF and density are described at :class:`~phasegen.distributions.JointCDF` and
      :class:`~phasegen.distributions.JointDensity`, the conditionals at
      :class:`~phasegen.distributions.ConditionalRewardDistribution`.
    - An infinite argument is represented by a large finite one, as for the atom of a
      :class:`~phasegen.distributions.RewardDistribution`.
    - A joint distribution has no quantile function. On a coalescent with a start or end time, the transform and the
      distribution functions raise :class:`NotImplementedError`.

    .. versionadded:: 2.0
    """
    @property
    def _time_scale(self) -> float:
        """The inversion time scale of the host, not defaulted (see ``RewardDistribution._time_scale``)."""
        return self._host._time_scale

    @property
    def _s_inf(self) -> float:
        r"""The atom probe :math:`10^8/\min(\tau, \sqrt{\mathbb{E}[R_a^2]}, \sqrt{\mathbb{E}[R_b^2]})`, scaled like
        ``RewardDistribution._s_inf``. One probe serves both axes."""
        scales = [v for v in (self._time_scale, self._rms['a'], self._rms['b']) if v > 0]
        return 1e8 / min(scales)

    #: Bivariate function objects. A joint has no quantile function.
    _pdf_function = JointDensity
    _cdf_function = JointCDF
    _quantile_function = None

    #: Window scale of the 2D expansion, ``mean + scale * std`` per axis (:math:`\kappa` in ``JointCDF``). A wider
    #: window coarsens the resolution ``b / n_terms`` near the origin, a narrower one truncates tail mass.
    _cos2d_window_scale: float = 5.0

    #: Per-axis node count of the finite-difference grid of the density (:math:`m` in ``JointDensity``), spread over
    #: the window of the expansion. The step ``b / (m - 1)`` sets the width of the cell the density is averaged over.
    _cos2d_pdf_grid: int = 800

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
        :raises ValueError: If some state carrying mass can never reach a common ancestor in the final epoch.
        """
        st = self._setup
        _assert_lst_absorbs(self._host)
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

        For a single epoch, with :math:`\mathbf{A} = \operatorname{diag}(s\,\mathbf{r}_a) - \mathbf{T}` and the reward
        vectors restricted to the transient states, expanding the inverse gives the coefficients exactly,

        .. math::

            \Phi_j(s) = (-1)^j\, \boldsymbol{\alpha}_T
            \big(\mathbf{A}^{-1} \operatorname{diag}(\mathbf{r}_b)\big)^j \mathbf{A}^{-1} \mathbf{q},

        so no derivative is approximated by a difference, and one LU factorization of :math:`\mathbf{A}` serves all
        orders. For several epochs, the coefficients of each epoch's matrix exponential are the blocks of a single
        exponential of a block upper-bidiagonal matrix (Van Loan, 1978), with the shifted generator on the diagonal and
        :math:`-\operatorname{diag}(\mathbf{r}_b)` above it.

        .. rubric:: References

        Van Loan, C. F. (1978). Computing integrals involving the matrix exponential. IEEE Transactions on Automatic
        Control 23(3), 395-404.

        :param s: Value of the held argument.
        :param on: The held argument, ``'a'`` or ``'b'``. The coefficients are in the other argument.
        :param order: Highest order :math:`J`.
        :return: The coefficients :math:`[\Phi_0(s), \ldots, \Phi_J(s)]`.
        :raises ValueError: If some state carrying mass can never reach a common ancestor in the final epoch.
        """
        st = self._setup
        _assert_lst_absorbs(self._host)
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
        :raises ValueError: If some state carrying mass can never reach a common ancestor in the final epoch.
        """
        st = self._setup
        _assert_lst_absorbs(self._host)
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
    def _ratio(self) -> Optional[float]:
        """The constant ``c > 0`` with ``r_a = c r_b`` on every transient state, so that ``R_a = c R_b`` almost surely
        and the law has no density on the plane, or ``None`` when the reward vectors are not proportional or one of
        them vanishes."""
        st = self._setup
        na, nb = st['ra'].max(initial=0.0), st['rb'].max(initial=0.0)
        if na > 0 and nb > 0 and np.allclose(st['ra'] * nb, st['rb'] * na, rtol=1e-12, atol=0.0):
            return float(na / nb)
        return None

    def _line_lst_batch(self, c: float, on: str, u) -> np.ndarray:
        r"""
        The transform :math:`\Phi_c(u) = \mathbb{E}\big[e^{-u R_{on}};\ R_a = c R_b\big]` of the reward ``on`` on the
        paths that absorb without leaving the states :math:`E_c` where :math:`r_a = c\,r_b`, at a batch of arguments.
        Only those paths give :math:`R_a = c R_b` with positive probability, since time spent where the rewards are not
        in that ratio adds a term with a continuous law. A state outside :math:`E_c` is killed by a shift of
        ``_s_inf`` times :math:`|r_a - c\,r_b|` there, as the atom probe kills a positive reward.

        :param c: The slope :math:`c > 0` of the line.
        :param on: The reward the transform variable acts on, ``'a'`` or ``'b'``.
        :param u: The arguments, a scalar or a 1D array.
        :return: The transform values.
        """
        st = self._setup
        tau = st['tau']
        kill = (self._s_inf * tau) * np.abs(st['ra'] - c * st['rb'])
        shifts = (np.outer(np.atleast_1d(u) * tau, st['ra' if on == 'a' else 'rb']) + kill).astype(complex)
        return _lst_from_shift_batch(shifts, st['alpha'], st['T_epochs'], st['sparse'], st['lu_perm'])

    def _line_cdf(self, c: float, on: str, ys: np.ndarray) -> np.ndarray:
        """``P(0 < R_on <= y, R_a = c R_b)`` at each ``y``, the Euler inversion of the line transform less its mass at
        zero, divided by the transform variable."""
        at_zero = self._line_lst_batch(c, on, self._s_inf)[0]
        return np.array([_euler_invert(lambda u: (self._line_lst_batch(c, on, u) - at_zero) / u, float(y)).real
                         for y in ys])

    @cached_property
    def _lines(self) -> tuple:
        r"""
        The slopes :math:`c` of the lines :math:`R_a = c R_b` on which the joint law places a positive probability
        :math:`\Pr(R_a = c R_b > 0)`, ascending. The candidates are the ratios :math:`r_a / r_b` over the transient
        states where both rewards are positive. For two loci of one statistic only :math:`c = 1` occurs, for two
        different rewards other slopes do as well.
        """
        st = self._setup
        ra, rb = st['ra'], st['rb']
        both = (ra > 0) & (rb > 0)
        ratios = ra[both] / rb[both]
        _, first = np.unique(np.round(ratios, 12), return_index=True)

        lines = []
        for c in ratios[np.sort(first)]:
            mass = (self._line_lst_batch(c, 'a', 0.0)[0] - self._line_lst_batch(c, 'a', self._s_inf)[0]).real
            if mass > _ATOM_FLOOR:
                lines.append(float(c))
        return tuple(sorted(lines))

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
    def _rms(self) -> dict:
        r"""The exact root mean squares :math:`\sqrt{\mathbb{E}[R_a^2]}` and :math:`\sqrt{\mathbb{E}[R_b^2]}`, keyed
        ``'a'`` and ``'b'``."""
        return dict(a=float(np.sqrt(self.moment(2, 0))), b=float(np.sqrt(self.moment(0, 2))))

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
        return float(self.cov() / np.sqrt(self.marginal('a').var * self.marginal('b').var))

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

    def _cos2d_memo(self, name: str, build) -> Any:
        """
        A value derived from the 2D cosine expansion, built once per :attr:`Settings.cos_terms_2d
        <phasegen.settings.Settings.cos_terms_2d>`: changing the term count on a live distribution discards every such
        value. Honours :attr:`Settings.cache <phasegen.settings.Settings.cache>`.

        :param name: Name of the value.
        :param build: Builds the value.
        :return: The value.
        """
        memo = self.__dict__.setdefault('_cos2d_cache', {})

        if memo.get('_terms') != Settings.cos_terms_2d:
            memo.clear()
            memo['_terms'] = Settings.cos_terms_2d

        if name in memo:
            return memo[name]

        value = build()
        if Settings.cache:
            memo[name] = value

        return value

    @property
    def _cos2d(self) -> dict:
        """The coefficient matrix ``A``, windows ``ba``, ``bb`` and frequencies ``ua``, ``ub`` of the 2D cosine
        expansion of ``JointCDF``, with the atoms removed by inclusion-exclusion and the Lanczos factors applied."""
        return self._cos2d_memo('cos2d', self._build_cos2d)

    def _build_cos2d(self) -> dict:
        """Build ``_cos2d``."""
        n_terms, scale, big = Settings.cos_terms_2d, self._cos2d_window_scale, self._s_inf
        p00 = self._atoms['both0']
        ba, bb = self.marginal('a')._range(scale), self.marginal('b')._range(scale)

        # the expansion folds the mass beyond its window back into it, so the joint CDF near the window end is too
        # high by up to that mass
        if Settings.check_inversions:
            for axis, b in (('a', ba), ('b', bb)):
                tail = 1.0 - float(self.marginal(axis).cdf._cdf_point(b))
                if tail > _COS2D_TAIL_WARN:
                    self._logger.warning(
                        "The 2D Fourier-cosine window of R_%s ends at %.3g and leaves out a mass of %.2g, which the "
                        "expansion folds back into it. The joint CDF may be too high by up to that amount towards the "
                        "window end, and its margin reaches 1 there.", axis, b, tail
                    )
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

    @property
    def _cos2d_wiggle_check(self) -> float:
        """The largest near-origin gap between ``F(x, inf)`` of the cosine expansion and the marginal CDF of ``R_a``,
        logged as a warning above 0.03. The cosine error concentrates near the axes, so this margin comparison detects
        an under-resolved near-origin rise. It reads the coefficients directly, so it does not recurse into
        ``_cc_box``."""
        return self._cos2d_memo('wiggle', self._build_cos2d_wiggle_check)

    def _build_cos2d_wiggle_check(self) -> float:
        """Build ``_cos2d_wiggle_check``."""
        st = self._cos2d
        big = self._s_inf
        ma = self.marginal('a')
        # near-origin points of the continuous part, where the bias concentrates. They lie above the atom at 0, where
        # the inversion of the continuous part is defined
        a0 = float(self._atoms['a0'])
        xs = np.linspace(0.0, float(ma.quantile(a0 + 0.4 * (1.0 - a0))), 5)[1:]
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

    @property
    def _density_grid(self) -> dict:
        """The bicubic spline of ``JointDensity`` through the mixed central difference of ``_cc_box``, and the interior
        nodes it is built on. The grid is uniform over the cosine window ``ba`` x ``bb``, so a density value does not
        depend on the queried grid."""
        return self._cos2d_memo('density_grid', self._build_density_grid)

    def _build_density_grid(self) -> dict:
        """Build ``_density_grid``."""
        from scipy.interpolate import RectBivariateSpline

        st = self._cos2d
        n = max(6, self._cos2d_pdf_grid)
        gx, gy = np.linspace(0.0, st['ba'], n), np.linspace(0.0, st['bb'], n)
        hx, hy = gx[1] - gx[0], gy[1] - gy[0]
        # box CDF on the grid (vanishes on the axes), then the mixed central second difference at the interior nodes
        F = np.zeros((n, n))
        F[1:, 1:] = self._cc_box(gx[1:], gy[1:])
        dens = (F[2:, 2:] - F[2:, :-2] - F[:-2, 2:] + F[:-2, :-2]) / (4.0 * hx * hy)
        k = min(3, dens.shape[0] - 1)
        return dict(spline=RectBivariateSpline(gx[1:-1], gy[1:-1], dens, kx=k, ky=k), gx=gx[1:-1], gy=gy[1:-1])

    def _density(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The continuous density of ``JointDensity`` on the outer grid ``xs x ys``, read off the spline of
        ``_density_grid``. The cosine expansion holds no mass beyond its window, so the density vanishes there."""
        if Settings.check_inversions:
            _ = self._cos2d_wiggle_check
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        # the spline evaluates strictly increasing points, mapped back to the caller's order and repetitions
        ux, ix = np.unique(xs, return_inverse=True)
        uy, iy = np.unique(ys, return_inverse=True)
        g, st = self._density_grid, self._cos2d
        gx, gy = g['gx'], g['gy']
        out = g['spline'](np.clip(ux, gx[0], gx[-1]), np.clip(uy, gy[0], gy[-1]))[np.ix_(ix, iy)]
        out[(xs < 0) | (xs > st['ba']), :] = 0.0
        out[:, (ys < 0) | (ys > st['bb'])] = 0.0
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
        """The joint CDF of ``JointCDF`` on the outer grid ``xs x ys``, zero where either threshold is negative and
        within ``[0, 1]`` everywhere."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        g_a, g_b = self._cos_axis('a', ys), self._cos_axis('b', xs)
        out = np.clip(g_b[:, None] + g_a[None, :] - self._atoms['both0'] + self._cc_box(xs, ys), 0.0, 1.0)
        out[xs < 0, :] = 0.0
        out[:, ys < 0] = 0.0
        return out

    def conditional(self, on: str = 'a', value: float = 0.0) -> 'ConditionalRewardDistribution':
        r"""
        The distribution of the other reward given that the reward ``on`` equals ``value``, as a
        :class:`~phasegen.distributions.ConditionalRewardDistribution`, whose docstring describes its transform and
        inversion. A ``value`` of zero conditions on the atom of the reward ``on``, a positive ``value`` on its
        continuous part. Mass on a line :math:`R_a = c R_b` gives the conditional on a ``value`` :math:`v > 0` an atom
        at :math:`v / c` given :math:`R_a`, or :math:`c v` given :math:`R_b`, which its density excludes.

        :param on: The conditioning reward, ``'a'`` or ``'b'``.
        :param value: The conditioning value, non-negative.
        :return: The conditional distribution of the other reward.
        :raises ValueError: If ``on`` is not ``'a'`` or ``'b'``, if ``value`` is negative, if ``value`` is zero and the
            conditioning reward has a negligible atom, or if the density of the conditioning reward at ``value`` is
            below the resolution of the inversion or too close to one of its jumps at an epoch time.
        :raises NotImplementedError: If one reward is a constant multiple of the other on every transient state, so
            that the conditional is a point mass, or on a windowed coalescent.

        .. versionadded:: 2.0
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        if value is None or value < 0:
            raise ValueError("`value` must be non-negative.")
        if self._ratio is not None:
            raise NotImplementedError("The conditional of a pair of proportional rewards is a point mass (R_a = c R_b "
                                      "almost surely).")

        other_name = 'b' if on == 'a' else 'a'

        if value == 0:
            return _AtomConditional(self, on, f"R_{other_name} | R_{on} = 0")

        nested = _NestedConditional(self, on, float(value), f"R_{other_name} | R_{on} = {value:g}")

        # each line R_a = c R_b of positive probability gives the conditional an atom where it crosses the value
        atoms = []
        for c in self._lines:
            f = _euler_invert(lambda u, c=c: self._line_lst_batch(c, on, u), float(value), N0=nested._N0).real
            if f / nested._G0 > _ATOM_FLOOR:
                atoms.append((float(value) / c if on == 'a' else c * float(value), c))

        return _LineConditional(nested, atoms) if atoms else nested

    def check_total_expectation(self, n_points: int = 32, tol: float = 0.01) -> dict:
        r"""
        Test the law of total expectation over the conditionals of
        :class:`~phasegen.distributions.ConditionalRewardDistribution`, and log a warning per conditioning reward when
        the relative error exceeds ``tol``.

        For each conditioning reward :math:`R_c`, with :math:`R_o` the other reward and :math:`p_c = \mathbb{P}(R_c = 0)`,

        .. math::

            \mathbb{E}[R_o] = p_c\,\mathbb{E}[R_o \mid R_c = 0]
            + (1 - p_c) \int_0^1 \mathbb{E}\big[R_o \mid R_c = v(\xi)\big]\,\mathrm{d}\xi,

        where :math:`v(\xi)` is the quantile of :math:`R_c` at the level :math:`p_c + (1 - p_c)\,\xi`, so that the levels
        :math:`\xi \in (0, 1)` cover the continuous part of :math:`R_c`. All conditional checks place their conditioning
        values at such levels. The integral uses Gauss-Legendre quadrature with ``n_points`` nodes.

        A level whose conditional cannot be constructed, typically one so far in the tail that the density of
        :math:`R_c` is below the resolution of the inversion, is skipped by every conditional check with one warning
        per conditioning reward. The error is computed over the remaining levels, with quadrature weights renormalised,
        and is infinite only when no conditional could be constructed.

        :param n_points: Number of Gauss-Legendre nodes per conditioning reward.
        :param tol: Relative error above which a warning is logged.
        :return: The relative error between the two sides of the identity per conditioning reward, keyed ``'a'`` and
            ``'b'``, infinite when no conditional could be constructed, or an empty dictionary when one reward is a
            constant multiple of the other on every transient state.
        """
        if self._ratio is not None:
            return {}  # R_a = c R_b a.s., the conditional is a point mass

        x, w = np.polynomial.legendre.leggauss(n_points)
        us, ws = 0.5 * (x + 1.0), 0.5 * w  # map [-1, 1] -> [0, 1]

        out = {}
        for on, other in (('a', 'b'), ('b', 'a')):
            marg_on, marg_other = self.marginal(on), self.marginal(other)
            lhs = float(marg_other.mean)
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
            if p0 >= _ATOM_FLOOR:  # atom term P(R_on = 0) E[R_other | R_on = 0]
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

    def check_total_probability(self, n_points: int = 32, n_y: int = 15, tol: float = 0.01) -> dict:
        r"""
        Test the law of total probability over the conditionals, and log a warning per conditioning reward when the
        largest deviation exceeds ``tol``.

        With the notation of :meth:`JointRewardDistribution.check_total_expectation()
        <phasegen.distributions.JointRewardDistribution.check_total_expectation>`, the identity

        .. math::

            \mathbb{P}(R_o \le y) = p_c\,\mathbb{P}(R_o \le y \mid R_c = 0)
            + (1 - p_c) \int_0^1 \mathbb{P}\big(R_o \le y \mid R_c = v(\xi)\big)\,\mathrm{d}\xi

        is tested at ``n_y`` quantiles :math:`y` of :math:`R_o`, and the largest absolute deviation is reported. The
        whole conditional law enters, so errors that cancel in the mean remain visible. Where the joint law places
        positive probability on a line :math:`R_a = c R_b`, each conditional carries an atom where the line crosses its
        conditioning value, whose step in :math:`y` the quadrature cannot resolve. That part enters exactly instead, as
        :math:`\mathbb{P}(0 < R_o \le y,\, R_a = c R_b)`. Unbuildable levels are handled as described there.

        :param n_points: Number of Gauss-Legendre nodes per conditioning reward.
        :param n_y: Number of evaluation points :math:`y`.
        :param tol: Deviation, an absolute probability, above which a warning is logged.
        :return: The deviation per conditioning reward, keyed ``'a'`` and ``'b'``, infinite when no conditional could be
            constructed, or an empty dictionary when one reward is a constant multiple of the other on every transient
            state.
        """
        if self._ratio is not None:
            return {}  # R_a = c R_b a.s., the conditional is a point mass

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
                    cond = self.conditional(on, v)
                    if isinstance(cond, _LineConditional):  # its atoms enter through the line terms below
                        cdfs.append((1.0 - cond._p) * np.asarray(cond._continuous.cdf(ys), dtype=float))
                    else:
                        cdfs.append(np.asarray(cond.cdf(ys), dtype=float))
                    weights.append(weight)
                except ValueError as e:
                    skipped.append((float(u), str(e)))

            self._warn_skipped(on, skipped, n_points, 'the law of total probability')
            if not cdfs:
                out[on] = float('inf')
                continue

            # the quadrature over the constructed nodes, renormalised to their total weight
            rhs = (1.0 - p0) * (np.asarray(weights) @ np.asarray(cdfs)) / np.sum(weights)
            if p0 >= _ATOM_FLOOR:  # atom term P(R_on = 0) F(y | R_on = 0)
                rhs = rhs + p0 * np.asarray(self.conditional(on, 0.0).cdf(ys), dtype=float)
            for c in self._lines:  # line term P(0 < R_other <= y, R_a = c R_b)
                rhs = rhs + self._line_cdf(c, other, ys)

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
        A statistic of the conditionals averaged over the window :math:`W` of conditioning values from
        ``value - half_width`` to ``value + half_width``, weighted by the density :math:`f_c` of the conditioning reward
        :math:`R_c`,

        .. math::

            \bar{g} = \frac{\int_W f_c(v)\, g(v)\,\mathrm{d}v}{\int_W f_c(v)\,\mathrm{d}v},

        where :math:`g(v)` is ``statistic`` applied to the conditional given :math:`R_c = v`, and the integrals use
        Gauss-Legendre quadrature. For the mean or the CDF this is the quantity that a sample restricted to the window
        estimates, as in :meth:`EmpiricalJointDistribution.conditional()
        <phasegen.distributions.EmpiricalJointDistribution.conditional>`.

        :param statistic: Callable taking a :class:`~phasegen.distributions.ConditionalRewardDistribution` and returning
            a scalar or a 1D array, for example ``lambda c: c.mean`` or ``lambda c: c.cdf(ys)``.
        :param on: The conditioning reward, ``'a'`` or ``'b'``.
        :param value: Centre of the window.
        :param half_width: Half-width of the window, in units of the conditioning reward.
        :param n_nodes: Number of Gauss-Legendre nodes, ``None`` for the default.
        :return: The window average, a float for a scalar statistic and an array of the statistic's shape otherwise.
        :raises ValueError: If the window reaches zero, where the conditioning reward may have an atom.
        :raises NotImplementedError: If one reward is a constant multiple of the other on every transient state.
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

        The conditioning values :math:`v(\xi)` are placed as described at
        :meth:`JointRewardDistribution.check_total_expectation()
        <phasegen.distributions.JointRewardDistribution.check_total_expectation>`, at levels :math:`\xi` spread over
        nearly all of :math:`(0, 1)` or given by ``quantiles``. At each, the mean :math:`\hat{m}` of the conditional
        transform is compared with :math:`m = \mathbb{E}[R_o \mid R_c = v(\xi)]` from
        :meth:`ConditionalRewardDistribution.moment() <phasegen.distributions.ConditionalRewardDistribution.moment>`,
        which involves no nested inversion, with the scaled error

        .. math::

            \frac{|\hat{m} - m|}{\max\big(|m|,\ \epsilon\,\mathbb{E}[R_o]\big)}.

        The small fraction :math:`\epsilon` of the unconditional mean keeps the error meaningful where the conditional
        mean vanishes. The identity carries the de Hoog error, which grows with the number of epochs, so ``tol`` must
        exceed it.

        :param n_points: Number of conditioning values per conditioning reward. Ignored when ``quantiles`` is given.
        :param tol: Scaled error above which a warning is logged.
        :param quantiles: Levels :math:`\xi \in (0, 1)` within the continuous part of the conditioning reward.
        :param curves: Number of conditioning values per conditioning reward at which the conditional density is also
            evaluated and stored in ``conditional_densities``, for inspection only.
        :return: The largest scaled error per conditioning reward, keyed ``'a'`` and ``'b'``, infinite when no
            conditional could be constructed, or an empty dictionary when one reward is a constant multiple of the
            other on every transient state.
        """
        us = np.linspace(*self._COND_CHECK_SPAN, n_points) if quantiles is None else np.asarray(quantiles, float)
        if self._ratio is not None:
            return {}  # R_a = c R_b a.s., the conditional is a point mass

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

        At the conditioning values of :meth:`JointRewardDistribution.check_conditional_moments()
        <phasegen.distributions.JointRewardDistribution.check_conditional_moments>`, the moment of order
        :math:`i \le k` of the evaluated conditional CDF :math:`F` is

        .. math::

            \hat{m}_i = \int_0^{\infty} i\,y^{i-1}\big(1 - F(y)\big)\,\mathrm{d}y,

        computed by Simpson's rule over the support window of :math:`F`. It is compared with
        :meth:`ConditionalRewardDistribution.moment() <phasegen.distributions.ConditionalRewardDistribution.moment>` and
        scaled as in the mean check. Unlike the mean check, this reaches the distribution function itself. Orders above
        two weight the far tail, where a small absolute error of the CDF becomes a large relative one.

        :param n_points: Number of conditioning values per conditioning reward. Ignored when ``quantiles`` is given.
        :param tol: Scaled error above which a warning is logged.
        :param k: Highest moment order :math:`k`.
        :param quantiles: Levels within the continuous part of the conditioning reward, in :math:`(0, 1)`.
        :return: The largest scaled error over conditioning values and orders per conditioning reward, keyed ``'a'``
            and ``'b'``, infinite when no conditional could be constructed, or an empty dictionary when one reward is a
            constant multiple of the other on every transient state.
        """
        us = np.linspace(*self._COND_CHECK_SPAN, n_points) if quantiles is None else np.asarray(quantiles, float)
        if self._ratio is not None:
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
    conditioning value, and :math:`\Phi(s_o, s_c)` for the joint transform of
    :class:`~phasegen.distributions.JointRewardDistribution` with its arguments in this order. The conditional law is
    given by its transform :math:`\varphi(s) = \mathbb{E}[e^{-sR_o} \mid R_c = v]`, which ``cdf``, ``pdf`` and
    ``quantile`` invert as for any :class:`~phasegen.distributions.RewardDistribution`.

    For :math:`v > 0` the conditional density is :math:`f(x \mid v) = f(x, v) / f_c(v)`, with :math:`f(x, v)` the joint
    density of :math:`(R_o, R_c)` and :math:`f_c` the density of :math:`R_c`. In transform form,

    .. math::

        \varphi(s) = \frac{G(s)}{G(0)}, \qquad
        G(s) = \mathcal{L}^{-1}_{s_c}\big[\Phi(s, s_c)\big](v) = \mathbb{E}\big[e^{-sR_o} \mid R_c = v\big]\, f_c(v),

    where :math:`\mathcal{L}^{-1}_{s_c}` inverts the Laplace transform in :math:`s_c` and :math:`G(0) = f_c(v)`. The
    atom of :math:`R_c` at zero does not contribute at :math:`v > 0`.

    .. rubric:: Inner inversion

    :math:`G` is computed by the Fourier-series method with Euler summation (Abate and Whitt, 1995),

    .. math::

        G(s) \approx \frac{e^{\eta/2}}{2v} \sum_{j} (-1)^j\, w_j\,
        \Phi\Big(s,\ \frac{\eta + 2\pi \mathrm{i} j}{2v}\Big),

    where the damping :math:`\eta > 0` bounds the discretization error by about :math:`e^{-\eta}` and the weights
    :math:`w_j` are 1 up to a truncation :math:`N` and then taper binomially over a few more terms. The nodes and weights
    do not depend on :math:`s`, so :math:`\varphi` stays analytic in :math:`s`, as the outer inversion requires.

    .. rubric:: Conditioning on the atom

    For :math:`v = 0` the condition is the event :math:`R_c = 0`, which must have positive probability, and
    :math:`\varphi(s) = \Phi(s, \infty) / \Phi(0, \infty)` needs no inner inversion.

    .. rubric:: Implementation

    - :math:`N` is doubled until :math:`G(0)` is positive and stable. If it does not stabilize, construction raises
      :class:`ValueError`. This happens where the density of :math:`R_c` at :math:`v` is below the resolution of the
      inversion, and near a jump of that density, which a reward accrued at one rate has at every epoch time and across
      which the series converges slowly. Within 5% of such a jump a warning is logged, since a conditional that does
      stabilize may still be off by several percent there.
    - The support window of the cosine fit grows from the conditional mean until the de Hoog CDF reaches a probability
      close to one.
    - Before the first cosine expansion, :math:`N` is doubled further until the CDF of the locating pass of the
      expansion on that window moves by at most :math:`10^{-3}` when :math:`N` is halved, and held for all :math:`s`.
      Both truncations weight the same nodes, so the check needs no transform evaluations beyond the pass. A CDF still
      moving at the largest truncation is reported by a warning. The mean is taken at the truncation on which
      :math:`G(0)` stabilizes, which resolves :math:`G` near :math:`s = 0`.
    - The mean is :math:`-\varphi'(0)` by a central difference, and higher moments are described at
      :meth:`ConditionalRewardDistribution.moment() <phasegen.distributions.ConditionalRewardDistribution.moment>`.
    - For :math:`v > 0` the transform is itself a numerical inversion, so results carry a few correct digits, fewest
      far in the tail of :math:`R_c` and on demographies with many epochs.

    .. rubric:: References

    Abate, J. and Whitt, W. (1995). Numerical inversion of Laplace transforms of probability distributions. ORSA
    Journal on Computing 7(1), 36-43.

    .. versionadded:: 2.0
    """
    #: The conditioning value. Overridden by ``_NestedConditional``, the atom conditions on ``R_on = 0``.
    _value: float = 0.0

    #: Relative step of the cumulant differences, larger than for an exact transform because the conditional transform
    #: is a numerical inversion with fewer correct digits.
    _cumulant_step: float = 1e-3

    @property
    def _rms(self) -> float:
        r"""The exact root mean square :math:`\sqrt{\mathbb{E}[R_o^2]}` of the unconditional other reward, the scale of
        the step of :meth:`_cumulants`."""
        return self._joint._rms['b' if self._on == 'a' else 'a']

    def lst(self, s: complex) -> complex:
        r"""
        The conditional transform :math:`\varphi(s)` defined at
        :class:`~phasegen.distributions.ConditionalRewardDistribution`.

        :param s: The argument.
        :return: The transform at ``s``.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        raise NotImplementedError

    def _refine(self) -> None:
        """Refine the inner inversion before the first cosine expansion, nothing for a transform without one."""

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

        The first moment is the mean :math:`-\varphi'(0)`. For :math:`v > 0` and :math:`k \ge 2`,

        .. math::

            \mathbb{E}[R_o^k \mid R_c = v]
            = \frac{k!\,(-1)^k\,\mathcal{L}^{-1}[\Phi_k](v)}{\mathcal{L}^{-1}[\Phi_0](v)},

        where :math:`\Phi_j(s_c)` is the coefficient of :math:`s_o^j` in the expansion of :math:`\Phi(s_o, s_c)` about
        :math:`s_o = 0`, from :meth:`JointRewardDistribution.lst_taylor()
        <phasegen.distributions.JointRewardDistribution.lst_taylor>`. The denominator is :math:`f_c(v)`. Both inversions
        are single de Hoog inversions, so the identity is independent of the conditional transform, with the accuracy
        of the de Hoog method. For :math:`v = 0` only the second moment :math:`\varphi''(0)` is available, by central
        differences.

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
        if atom < _ATOM_FLOOR:
            raise ValueError(f"Cannot condition on R_{on} = 0: it has (near) zero probability.")
        # the sub-transform is probed where the atom dividing it was measured, so that ``lst(0) = 1`` exactly
        big = joint._s_inf
        self._joint = joint
        self._host = joint._host
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
    the batch with ``_expm_batch``. A dense last epoch is solved as one stacked ``np.linalg.solve`` over the batch, a
    sparse one by the block-triangular sparse LU of the scalar path per shift."""
    nt = len(alpha)
    k = len(shifts)
    diag = np.arange(nt)
    vec = np.zeros((k, nt + 1), dtype=complex)
    vec[:, :nt] = alpha

    for T, t0, t1 in T_epochs[:-1]:
        Td = T.toarray() if sp.issparse(T) else np.asarray(T)
        Q = np.zeros((k, nt + 1, nt + 1), dtype=complex)
        Q[:, :nt, :nt] = Td
        Q[:, diag, diag] -= shifts  # only the diagonal varies across the batch
        Q[:, :nt, nt] = _exit_rates(T)
        vec = np.einsum('ki,kij->kj', vec, _expm_batch(Q * (t1 - t0)))

    a, c = vec[:, :nt], vec[:, nt]
    Tm = T_epochs[-1][0]
    exit_m = _exit_rates(Tm)

    if sparse:
        return c + np.array([a[i] @ MomentEvaluator._lu_solver(sp.diags(shifts[i]) - Tm, True, perm)(exit_m)
                             for i in range(k)])

    A = np.repeat(-np.asarray(Tm, dtype=complex)[None], k, axis=0)
    A[:, diag, diag] += shifts
    return c + np.einsum('ki,ki->k', a, np.linalg.solve(A, np.broadcast_to(exit_m[:, None], (k, nt, 1)))[..., 0])


def _dehoog_invert(transform, t: float, degree: int) -> float:
    """
    Inverse Laplace transform of ``transform`` at ``t > 0`` by the method of de Hoog, Knight and Stokes (1982), the
    tail inversion of ``RewardDistribution``. The Fourier series on the ``2 * degree + 1`` contour nodes
    ``gamma + i k pi / T``, ``T = 2t``, is accelerated by a continued fraction whose coefficients come from the
    quotient-difference algorithm, with the improved remainder of the last term. The contour abscissa follows the
    degree through ``alpha = 10 ** -int(1.38 * degree)``, ``tol = 10 * alpha`` and ``gamma = alpha - log(tol) / (2T)``.

    :param transform: The transform, a function of a complex argument.
    :param t: The point at which to evaluate the inverse.
    :param degree: The degree ``M``.
    :return: The inverse at ``t``.
    """
    M = degree
    n = 2 * M + 1
    alpha = 10.0 ** -int(1.38 * degree)
    T = 2.0 * t
    gamma = alpha - np.log(10.0 * alpha) / (2.0 * T)
    fp = np.array([complex(transform(gamma + 1j * np.pi * k / T)) for k in range(n)])

    # quotient-difference table, filled by the rhombus rule
    e = np.zeros((n, M + 1), dtype=complex)
    q = np.zeros((2 * M, M), dtype=complex)
    q[0, 0] = fp[1] / (fp[0] / 2)
    q[1:, 0] = fp[2:2 * M + 1] / fp[1:2 * M]

    for r in range(1, M + 1):
        mr = 2 * (M - r) + 1
        e[:mr, r] = q[1:mr + 1, r - 1] - q[:mr, r - 1] + e[1:mr + 1, r - 1]

        if r < M:
            mq = 2 * (M - r) + 1
            q[:mq, r] = q[1:mq + 1, r - 1] * e[1:mq + 1, r] / e[:mq, r]

    # continued-fraction coefficients
    d = np.empty(n, dtype=complex)
    d[0] = fp[0] / 2
    d[1:2 * M:2] = -q[0, :M]
    d[2:2 * M + 1:2] = -e[0, 1:M + 1]

    # three-term recurrence of the numerator and denominator of the Pade approximant
    z = np.exp(1j * np.pi * t / T)
    A = np.zeros(n + 1, dtype=complex)
    B = np.ones(n + 1, dtype=complex)
    A[1] = d[0]

    for i in range(1, 2 * M):
        A[i + 1] = A[i] + d[i] * A[i - 1] * z
        B[i + 1] = B[i] + d[i] * B[i - 1] * z

    # improved remainder of the continued fraction: the period-2 tail u = d_e z / (1 + v), v = d_o z / (1 + u)
    # solves u^2 + u (1 + (d_o - d_e) z) - d_e z = 0, so u = h (sqrt(1 + d_e z / h^2) - 1)
    h = (1 + (d[2 * M - 1] - d[2 * M]) * z) / 2
    rem = h * np.expm1(0.5 * np.log1p(d[2 * M] * z / h ** 2))
    A[n] = A[2 * M] + rem * A[2 * M - 1]
    B[n] = B[2 * M] + rem * B[2 * M - 1]

    return float(np.exp(gamma * t) / T * (A[n] / B[n]).real)


def _euler_series(t: float, truncations: Sequence[int], A: float = 16.0, m: int = 12) -> tuple:
    """
    Nodes and weights of the Euler-summed Fourier series of ``_euler_invert`` at ``t``, one row of weights per
    truncation. The nodes are those of the largest truncation, on which a smaller one weights a subset and puts zero
    elsewhere, so several truncations cost the transform evaluations of the largest.

    :param t: The point of inversion.
    :param truncations: The truncations ``N0``.
    :param A: The damping.
    :param m: The number of Euler terms.
    :return: The nodes ``u`` and the weights, of shape ``(len(truncations), len(u))``.
    """
    n = max(truncations) + m
    ks = np.arange(-n, n + 1)
    u = (A + 2.0j * np.pi * ks) / (2.0 * t)
    binom = np.array([comb(m, j) for j in range(m + 1)], dtype=float) / 2.0 ** m
    # Euler weight of node k: the binomial-averaged fraction of the partial sums S_{N0+j} that include it, the sum of
    # binom[j] over j >= |k| - N0, which is 1 up to N0 and 0 beyond N0 + m
    tail = np.concatenate([np.cumsum(binom[::-1])[::-1], [0.0]])
    frac = np.array([tail[np.clip(np.abs(ks) - N0, 0, m + 1)] for N0 in truncations])
    return u, (np.exp(A / 2.0) / (2.0 * t)) * ((-1.0) ** ks) * frac


def _euler_invert(transform, t: float, A: float = 16.0, N0: int = _EULER_N0, m: int = 12) -> complex:
    """Euler-summed Fourier-series inversion of ``transform`` at ``t``, the inner inversion of
    ``ConditionalRewardDistribution`` (``A`` is its damping, ``N0`` its truncation, ``m`` its Euler terms). The node
    spacing ``2 pi / (2t)`` makes the series alternating, as Euler summation requires. It is summed two-sided, so a
    complex-valued inverse needs no conjugate symmetry. A larger ``A`` amplifies roundoff by ``exp(A / 2)``, the
    remaining error is truncation, reduced by ``N0``."""
    u, w = _euler_series(t, (N0,), A, m)
    vals = transform(u)  # the whole node set at once -- see _lst_from_shift_batch
    return complex(np.sum(w[0] * np.asarray(vals)))


class _NestedConditional(ConditionalRewardDistribution):
    """The conditional on a value ``R_on = value > 0`` of ``ConditionalRewardDistribution``: ``phi(s) = G(s) / G(0)``
    with ``G`` the Euler inversion along the conditioning axis. The truncation ``N0`` is calibrated on ``G(0)`` at
    construction by ``_calibrate``, and on the CDF by ``_refine`` before the first cosine expansion, and held for every
    ``s`` in between, since a truncation varying with ``s`` would break the analyticity of ``G`` in ``s`` that the
    outer inversion needs."""
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

        #: ``G`` at the arguments of the locating pass of the cosine expansion, keyed by the argument, once ``_refine``
        #: has run.
        self._G_rough = None

        #: Mean and variance of ``RewardDistribution._cumulants``, from the transform at the small real arguments of its
        #: differences, where the truncation of ``_calibrate`` resolves ``G``. They are held so that the moments do not
        #: depend on whether ``_refine`` has run.
        self._cumulants_calibrated = super()._cumulants()

        jump = self._nearby_jump()
        if jump is not None:
            self._logger.warning(
                "The density of R_%s jumps at %g, where an epoch begins, within %d%% of the conditioning value %g. The "
                "inner inversion converges slowly across a jump, so the conditional may be off by several percent.",
                self._on, jump, round(100 * _JUMP_WARN), self._value
            )

    def _calibrate(self, tol: float = 2e-2, n_max: int = _EULER_N0_MAX) -> tuple:
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
        jump = self._nearby_jump()
        cause = (f"The density of R_{self._on} jumps at {jump:g}, where an epoch begins, and the inner inversion "
                 f"converges slowly across a jump. Condition further from it, or sample." if jump is not None else
                 "Condition closer to the bulk, or sample.")

        raise ValueError(
            f"The marginal density at R_{self._on} = {self._value:g} did not converge under refinement of the inner "
            f"inversion (still moving by more than {tol:.0%} at N0 = {n_max}), so the conditional there would be "
            f"unreliable. {cause}"
        )

    def _cumulants(self) -> tuple:
        """The mean and variance at the truncation of ``_calibrate``, held at construction."""
        return self._cumulants_calibrated

    def _refine(self, n_max: int = _EULER_N0_MAX, target: Optional[ConditionalRewardDistribution] = None) -> None:
        r"""
        Double the truncation ``N0`` until the CDF of the locating pass of the cosine expansion of ``target``
        (``_LSTFunction._build_cos_coeffs``) moves by at most ``_cos_truncation_tol`` between the truncations
        :math:`N_0 / 2` and :math:`N_0`, and log a warning if it still moves by more at ``n_max``. The target is this
        conditional, or the continuous part of a line-atom conditional (``_LineContinuous``), whose transform is a
        function of ``G`` at the same truncation (``_lst_from_G``). ``G(0)``, on which ``_calibrate`` settles,
        converges at a smaller truncation than ``G`` at the frequencies of the expansion, which on several epochs
        oscillates in the conditioning variable. Both truncations weight the nodes of the larger one, so the difference
        costs no transform evaluations beyond the pass, whose values at the accepted truncation the expansion reuses. A
        doubling that makes ``G(0)`` non-positive is rejected, as in ``_calibrate``. The window of the pass is located
        at the truncation of ``_calibrate``. Runs once, and is called by the function objects before their first
        expansion, so a conditional whose moments alone are read never pays for it.

        :param n_max: Largest truncation tried.
        :param target: The distribution whose expansion is resolved, this one if ``None``.
        """
        if self._G_rough is not None:
            return

        target = self if target is None else target
        cdf = target.cdf
        b = cdf._range(cdf._cos_rough_scale)
        w = np.arange(cdf._cos_terms_rough) * np.pi / b
        args = [complex(-1j * wk) for wk in w] + [complex(self._s_inf)]
        xs = np.linspace(0.0, b, max(512, 2 * len(w)))

        def locating_pass(n0: int) -> tuple:
            """``G`` at ``args`` at the truncation ``n0``, and the largest difference of the locating CDF between
            the truncations ``n0 // 2`` and ``n0``, each from the transform of the target at its own truncation."""
            truncations = (n0, n0 // 2)
            G = np.array([self._inner(s, truncations) for s in args])  # (len(args), 2)
            phi = target._lst_from_G(np.array(args), G, truncations)
            curves = [cdf._eval_cos_cdf(cdf._cos_fit_from(b, w, f[:-1], f[-1].real), xs) for f in phi.T]
            return G[:, 0], float(np.abs(curves[0] - curves[1]).max())

        n0 = self._N0
        G, move = locating_pass(n0)
        while move > cdf._cos_truncation_tol and n0 < n_max:
            G_next, move_next = locating_pass(2 * n0)
            if not G_next[0].real > 0:
                break
            n0, G, move = 2 * n0, G_next, move_next

        if n0 != self._N0:
            self._N0, self._G0 = n0, float(G[0].real)
            # the per-point CDF values of the window search were taken at the smaller truncation
            for d in (self, target):
                d.__dict__.get('_lst_curve_cache', {}).pop('cdf_points', None)

        self._G_rough = dict(zip(args, G))

        if move > cdf._cos_truncation_tol:
            self._logger.warning(
                "%s: the inner inversion is unresolved at the frequencies of the cosine expansion, its CDF moving by "
                "%.2e (bar %.0e) when the truncation is halved from N0 = %d. The conditional CDF may be off by about "
                "that much. Conditioning closer to the bulk may help, or sample.",
                self.label, move, cdf._cos_truncation_tol, n0
            )

    @staticmethod
    def _lst_from_G(args: np.ndarray, G: np.ndarray, truncations: Sequence[int]) -> np.ndarray:
        """
        The transform ``G(s) / G(0)`` from ``G``, per truncation.

        :param args: The arguments, the first of which is 0.
        :param G: ``G`` at ``args``, one column per truncation.
        :param truncations: The truncations of the columns.
        :return: The transform, of the shape of ``G``.
        """
        return G / G[0]

    def _nearby_jump(self) -> Optional[float]:
        """
        The value within ``_JUMP_WARN`` of the conditioning value at which the density of the conditioning reward
        jumps, or ``None``. A reward accrued at one rate :math:`c` wherever it accrues, as a tree height, has a density
        that jumps at :math:`c t_0` for every epoch time :math:`t_0`.

        :return: The value of the jump, or ``None``.
        """
        r = self._joint._setup['ra' if self._on == 'a' else 'rb']
        rates = np.unique(r[r > 0])

        if len(rates) != 1:
            return None

        for epoch in self._host._get_epochs_until_unbounded()[1:]:
            jump = float(rates[0]) * epoch.start_time
            if abs(self._value - jump) <= _JUMP_WARN * self._value:
                return jump

        return None

    def _phi(self, u: np.ndarray) -> np.ndarray:
        """``Phi`` along the conditioning axis with the other argument at 0, the transform behind ``G(0)``."""
        z = np.zeros(len(u))
        return self._joint.lst_batch(z, u) if self._on == 'b' else self._joint.lst_batch(u, z)

    def _inner(self, s: complex, truncations: Sequence[int]) -> np.ndarray:
        """
        The Euler inversions of ``Phi`` along the conditioning axis at the value, with the other argument at ``s``, at
        each truncation from the nodes of the largest (``_euler_series``).

        :param s: The argument of the other reward.
        :param truncations: The truncations ``N0``.
        :return: ``G(s)`` per truncation.
        """
        u, weights = _euler_series(self._value, truncations)
        other = np.full(len(u), s)
        vals = self._joint.lst_batch(other, u) if self._on == 'b' else self._joint.lst_batch(u, other)
        return np.sum(weights * np.asarray(vals), axis=1)

    def _G(self, s: complex) -> complex:
        """``G(s)``, the Euler inversion of ``Phi`` along the conditioning axis at the value. The inner method must be
        accurate on peaked coalescent densities (Gaver-Stehfest is not), a fixed linear functional so that ``G`` stays
        analytic in ``s`` for the outer de Hoog recurrence (a nested de Hoog is not), and use a vertical contour, since
        the epoch exponentials overflow as the real part tends to minus infinity (Talbot's contour does)."""
        if self._G_rough is not None and s in self._G_rough:
            return complex(self._G_rough[s])
        return complex(self._inner(s, (self._N0,))[0])

    def lst(self, s: complex) -> complex:
        """The conditional transform ``G(s) / G(0)``, see ``ConditionalRewardDistribution``."""
        return self._G(complex(s)) / self._G0


class _LineContinuous(ConditionalRewardDistribution):
    """The continuous part of ``_LineConditional``: the conditional with its atoms removed,
    ``(G(s) - sum_k f_k e^{-s y_k}) / (G(0) - sum_k f_k)`` in the notation of ``_NestedConditional``, inverted by the
    ordinary conditional machinery on its own window. The densities ``f_k`` of the atoms are Euler inversions at the
    truncation of ``G``, so that the atoms cancel from the transform when ``_NestedConditional._refine`` raises it."""
    _pdf_function = ConditionalDensity
    _cdf_function = ConditionalCDF
    _quantile_function = ConditionalQuantileFunction

    def __init__(self, nested: '_NestedConditional', atoms: list) -> None:
        """
        :param nested: The conditional with the atoms included in its transform.
        :param atoms: ``(location, slope)`` per atom, the slope :math:`c` of its line :math:`R_a = c R_b`.
        """
        self._nested = nested
        self._joint = nested._joint
        self._host = nested._host
        self.state_space = nested.state_space
        self._on = nested._on
        self._value = nested._value
        self._logger = nested._logger
        self.label = nested.label
        self._y = np.array([y for y, _ in atoms], dtype=float)
        self._c = np.array([c for _, c in atoms], dtype=float)

        #: The densities of the atoms, keyed by the truncation.
        self._f_cache = {}

    def _densities(self, truncations: Sequence[int]) -> np.ndarray:
        """
        The densities :math:`f_c(v)` of the atoms, the Euler inversions of ``JointRewardDistribution._line_lst_batch``
        at the value, at each truncation from the nodes of the largest (``_euler_series``).

        :param truncations: The truncations ``N0``.
        :return: The densities, of shape ``(len(truncations), len(atoms))``.
        """
        u, weights = _euler_series(self._value, truncations)
        vals = np.array([self._joint._line_lst_batch(c, self._on, u) for c in self._c])  # (len(atoms), len(u))
        return (weights @ vals.T).real

    @property
    def _f(self) -> np.ndarray:
        """The densities of the atoms at the truncation of ``G``."""
        n0 = self._nested._N0
        if n0 not in self._f_cache:
            self._f_cache[n0] = self._densities((n0,))[0]
        return self._f_cache[n0]

    def _refine(self) -> None:
        """Refine the inner inversion on the expansion of this continuous part, see ``_NestedConditional._refine``."""
        self._nested._refine(target=self)

    def _lst_from_G(self, args: np.ndarray, G: np.ndarray, truncations: Sequence[int]) -> np.ndarray:
        """
        The transform of the continuous part from ``G``, per truncation, with the densities of the atoms at the same
        truncation.

        :param args: The arguments, the first of which is 0.
        :param G: ``G`` at ``args``, one column per truncation.
        :param truncations: The truncations of the columns.
        :return: The transform, of the shape of ``G``.
        """
        f = self._densities(truncations)  # (len(truncations), len(atoms))
        rest = G - np.exp(-np.outer(args, self._y)) @ f.T
        return rest / rest[0]

    def lst(self, s: complex) -> complex:
        """The transform of the continuous part."""
        s = complex(s)
        nested, f = self._nested, self._f
        return (nested._G(s) - np.sum(f * np.exp(-s * self._y))) / (nested._G0 - np.sum(f))


class _LineCDF(ConditionalCDF):
    """The CDF of ``_LineConditional``: the continuous part weighted by ``1 - P`` plus the steps of the atoms."""

    @property
    def _cos_coeffs(self) -> dict:
        """The cosine expansion of the continuous part, whose window is the support the mixture is read over."""
        return self._distribution._continuous.cdf._cos_coeffs

    def __call__(self, t) -> 'np.ndarray | float':
        """
        :param t: Point or array of points.
        :return: The CDF, of the same shape.
        """
        d = self._distribution
        d._warn_if_line_unresolved()
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        steps = (ta[:, None] >= d._atom_values[None, :]) @ d._atom_masses
        out = (1.0 - d._p) * np.atleast_1d(d._continuous.cdf(ta)) + steps
        return out if np.ndim(t) > 0 else float(out[0])


class _LineDensity(ConditionalDensity):
    """The density of ``_LineConditional``, that of its continuous part weighted by ``1 - P``."""

    def __call__(self, t, **kwargs) -> 'np.ndarray | float':
        """
        :param t: Point or array of points.
        :return: The density of the continuous part, of the same shape.
        """
        d = self._distribution
        d._warn_if_line_unresolved()
        out = (1.0 - d._p) * np.atleast_1d(d._continuous.pdf(np.atleast_1d(np.asarray(t, dtype=float))))
        return out if np.ndim(t) > 0 else float(out[0])


class _LineQuantile(ConditionalQuantileFunction):
    """The quantile of ``_LineConditional``: the levels an atom covers return its location, the others the quantile
    of the continuous part at the level less the atoms below it, rescaled by ``1 - P``."""

    def __call__(self, q) -> 'np.ndarray | float':
        """
        :param q: Level or array of levels in ``[0, 1]``.
        :return: The quantiles, of the same shape.
        :raises ValueError: If any ``q`` lies outside ``[0, 1]``.
        """
        d = self._distribution
        d._warn_if_line_unresolved()
        qa = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((qa < 0) | (qa > 1)):
            raise ValueError("Quantile must be between 0 and 1.")

        ys, ps = d._atom_values, d._atom_masses
        # the CDF just below and at each atom
        lo = (1.0 - d._p) * np.atleast_1d(d._continuous.cdf(ys)) + np.concatenate([[0.0], np.cumsum(ps)[:-1]])
        hi = lo + ps

        passed = (qa[:, None] > hi[None, :]) @ ps
        out = np.atleast_1d(d._continuous.quantile(np.clip((qa - passed) / (1.0 - d._p), 0.0, 1.0)))
        for y, a, b in zip(ys, lo, hi):
            out = np.where((qa >= a) & (qa <= b), y, out)
        return out if np.ndim(q) > 0 else float(out[0])


class _LineConditional(ConditionalRewardDistribution):
    r"""
    The conditional on a value ``R_on = v > 0`` when the joint law places positive probability on lines
    :math:`R_a = c R_b`, as the equal rewards of linked loci do on :math:`c = 1`. It is a mixture of one atom per line,
    at the value of the other reward on that line, of mass :math:`f_c(v) / f(v)`, where :math:`f_c` is the density of
    the conditioning reward on the paths that keep the rewards in ratio :math:`c`
    (``JointRewardDistribution._line_lst_batch``) and :math:`f` its marginal density, and a continuous part
    (``_LineContinuous``). The transform is the full one of ``_NestedConditional``, so the mean and moments include
    the atoms.
    """
    _pdf_function = _LineDensity
    _cdf_function = _LineCDF
    _quantile_function = _LineQuantile

    def __init__(self, nested: '_NestedConditional', atoms: list) -> None:
        """
        :param nested: The conditional with the atoms included in its transform.
        :param atoms: ``(location, slope)`` per atom, the slope :math:`c` of its line.
        """
        self._nested = nested
        self._joint = nested._joint
        self._host = nested._host
        self.state_space = nested.state_space
        self._on = nested._on
        self._value = nested._value
        self._logger = nested._logger
        self.label = nested.label

        atoms = sorted(atoms)

        #: Locations of the atoms, ascending.
        self._atom_values = np.array([y for y, _ in atoms], dtype=float)

        #: The continuous part.
        self._continuous = _LineContinuous(nested, atoms)

    @property
    def _atom_masses(self) -> np.ndarray:
        """Masses of the atoms, :math:`f_c(v) / G(0)` at the truncation of the expansion of the continuous part."""
        self._continuous._refine()
        return self._continuous._f / self._nested._G0

    @property
    def _p(self) -> float:
        """Total mass of the atoms."""
        return float(self._atom_masses.sum())

    def lst(self, s: complex) -> complex:
        """The conditional transform, atoms included."""
        return self._nested.lst(s)

    def _cumulants(self) -> tuple:
        """The mean and variance of the conditional with the atoms, whose transform this is."""
        return self._nested._cumulants()

    def _warn_if_line_unresolved(self) -> None:
        r"""
        Warn when the cosine expansion of the continuous part reaches frequencies at which the inner inversion no
        longer resolves the line part of the transform. The line part of an atom at :math:`y` oscillates at the
        frequency :math:`\omega` of the outer variable, and the inner Euler series, whose weights taper beyond its
        truncation :math:`N_0`, resolves it only while :math:`\omega y / \pi \le N_0`. Above that the subtracted
        atom fades from the transform, and the continuous part carries its negative.

        """
        if not Settings.check_inversions or self.__dict__.get('_line_warned'):
            return

        w_max = self._continuous.cdf._cos_coeffs['w'][-1]

        cutoff = np.pi * self._nested._N0 / self._atom_values

        if np.any(w_max > cutoff):
            self.__dict__['_line_warned'] = True
            self._logger.warning(
                "%s: the cosine expansion reaches the frequency %.3g, above %.3g, up to which the inner inversion "
                "resolves the atom at %.3g. The continuous part then carries the negative of that atom, so its CDF and "
                "density may be wrong, the more so the more cosine terms are used.",
                self.label, w_max, float(cutoff.min()), float(self._atom_values[np.argmin(cutoff)])
            )
