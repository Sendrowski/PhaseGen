"""Phase-type distribution (moment engine) and the tree-height distribution."""

import logging
import warnings
from ..caching import cached_property
from typing import Tuple, Iterable, Sequence, Union, TYPE_CHECKING
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from ..demography import Demography, Epoch
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalBranchLengthReward
from ..settings import Settings
from ..spectrum import SFS, AbstractSpectrum
from ..state_space import LineageCountingStateSpace, StateSpace

from .base import CallableDistributionFunctions, DensityAwareDistribution, DistributionFunction, \
    MarginalDemeDistributions, MarginalLocusDistributions, MomentAwareDistribution, _HazardGrid, \
    _GridCumulativeDistributionFunction, _GridDensityFunction, _GridQuantileFunction
from ._moments import MomentEvaluator

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .reward import RewardDistribution, JointRewardDistribution
    from .empirical import EmpiricalPhaseTypeDistribution
    from ..visualization import _CurveData

expm = Backend.expm
logger = logging.getLogger('phasegen')


class PhaseTypeDistribution(CallableDistributionFunctions, MomentEvaluator, MomentAwareDistribution):
    r"""
    Phase-type distribution of rewards accumulated by a piecewise time-homogeneous Markov jump process. The notation
    introduced here is shared by the method descriptions throughout the API reference.

    The process :math:`\{X_u\}_{u \ge 0}` is in state :math:`X_u` at time :math:`u \ge 0` and moves on the finite
    state set :math:`E` of the :class:`~phasegen.state_space.StateSpace`, which has :math:`|E|` states. The absorbing
    states form a closed set :math:`B \subset E`. The :math:`n_T` states in :math:`E \setminus B` are transient, and
    :math:`\tau = \inf\{u \ge 0 : X_u \in B\}` is the absorption time.

    The :class:`~phasegen.demography.Demography` divides time into :math:`M` epochs with boundaries
    :math:`0 = t_0 < t_1 < \dots < t_M = \infty`. Epoch :math:`i \in \{1, \dots, M\}` spans :math:`[t_{i-1}, t_i)`
    and has duration :math:`\Delta_i = t_i - t_{i-1}`. Within epoch :math:`i` the process is time-homogeneous with
    intensity matrix :math:`\mathbf{S}_i \in \mathbb{R}^{|E| \times |E|}`, whose off-diagonal entries are transition
    rates and whose rows sum to zero. Its restriction to the transient states is the sub-intensity matrix
    :math:`\mathbf{T}_i`, and :math:`\mathbf{q}_i = -\mathbf{T}_i \mathbf{e}_T` is the vector of absorption rates,
    where :math:`\mathbf{e}` and :math:`\mathbf{e}_T` denote the all-ones column vectors on :math:`E` and on the
    transient states. The initial distribution is the row vector :math:`\boldsymbol{\alpha}` on :math:`E`, and
    :math:`\boldsymbol{\alpha}_T` is its restriction to the transient states.

    A :class:`~phasegen.rewards.Reward` assigns the non-negative reward vector :math:`\mathbf{r}`, which is zero on
    :math:`B` and has entry :math:`r(x)` at state :math:`x \in E`, and :math:`\operatorname{diag}(\mathbf{r})` is the
    diagonal matrix with diagonal :math:`\mathbf{r}`. The reward accumulated over the window
    :math:`[t_\mathrm{start}, t_\mathrm{end}]` is

    .. math::

        R = \int_{t_\mathrm{start}}^{t_\mathrm{end}} r(X_u)\, \mathrm{d}u,

    where the start time :math:`t_\mathrm{start} \ge 0` and the end time :math:`t_\mathrm{end} \le \infty` default to
    :math:`t_\mathrm{start} = 0` and :math:`t_\mathrm{end} = \infty`, the accumulation until absorption. Several
    accumulated rewards are written
    :math:`R_1, \dots, R_k` for a moment of order :math:`k \ge 1`, and :math:`R_a, R_b` for a joint distribution. The
    number of sampled lineages is :math:`n`.

    The distribution of :math:`R` is characterized by its Laplace transform :math:`\varphi(s) = \mathbb{E}[e^{-sR}]`
    with complex argument :math:`s`, and a pair of accumulated rewards by the joint transform
    :math:`\Phi(s_a, s_b) = \mathbb{E}[e^{-s_a R_a - s_b R_b}]`. The atom :math:`p_0 = \mathbb{P}(R = 0)` is the
    probability that no reward accumulates. Distribution functions are evaluated at :math:`x`, and a quantile at the
    probability level :math:`q \in (0, 1)`.

    Moments are evaluated by
    :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
    """

    def __init__(
            self,
            state_space: StateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography = None,
            reward: Reward = None
    ) -> None:
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward. By default, the tree height reward.
        """
        if demography is None:
            demography = Demography()

        if reward is None:
            reward = TreeHeightReward()

        super().__init__()

        #: Population configuration (:class:`~phasegen.lineage.LineageConfig`).
        self.lineage_config: LineageConfig = state_space.lineage_config

        #: Locus configuration (:class:`~phasegen.locus.LocusConfig`).
        self.locus_config: LocusConfig = state_space.locus_config

        #: The accumulated :class:`~phasegen.rewards.Reward`.
        self.reward: Reward = reward

        #: The :class:`~phasegen.state_space.StateSpace`.
        self.state_space: StateSpace = state_space

        #: The :class:`~phasegen.demography.Demography`.
        self.demography: Demography = demography

        #: The :class:`~phasegen.distributions.TreeHeightDistribution`.
        self.tree_height: TreeHeightDistribution = tree_height

    @cached_property
    def mean(self) -> float | SFS:
        """
        First moment / mean.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> float | SFS:
        """
        Second central moment / variance.
        """
        return self.moment(k=2, center=True)

    @cached_property
    def std(self) -> float | SFS:
        """
        Standard deviation. A variance that cancellation in the moment solve leaves marginally below zero is read as
        zero, so the result stays real.
        """
        var = self.var

        if isinstance(var, AbstractSpectrum):
            return type(var)(np.maximum(np.asarray(var.data, dtype=float), 0.0) ** 0.5)

        return max(float(var), 0.0) ** 0.5

    @cached_property
    def m2(self) -> float | SFS:
        """
        Second (non-central) moment.
        """
        return self.moment(k=2, center=False)

    def distribution(self, reward: Reward = None) -> 'RewardDistribution':
        r"""
        The distribution of the accumulated reward :math:`R`, as a :class:`~phasegen.distributions.RewardDistribution`
        whose evaluation is described there. The reward must assign one value per state, so a spectrum takes the
        reward of a single bin. The ``cdf``, ``pdf`` and ``quantile`` of this distribution are those of the
        distribution of its own reward, except for :class:`~phasegen.distributions.TreeHeightDistribution`.

        :param reward: The reward whose accumulation is distributed. Defaults to this distribution's own reward.
        :return: The accumulated-reward distribution.
        """
        from .reward import RewardDistribution

        return RewardDistribution(self, reward)

    def joint_distribution(self, reward_a: Reward, reward_b: Reward) -> 'JointRewardDistribution':
        """
        Joint distribution of two accumulated rewards, as a :class:`~phasegen.distributions.JointRewardDistribution`.

        :param reward_a: The reward of :math:`R_a`.
        :param reward_b: The reward of :math:`R_b`.
        :return: The joint distribution.

        .. versionadded:: 2.0
        """
        from .reward import JointRewardDistribution

        return JointRewardDistribution(self, reward_a, reward_b)

    @property
    def _windowed(self) -> bool:
        """Whether the coalescent accumulates over a bounded window, ``start_time > 0`` or a finite ``end_time``. The
        window is stored on ``tree_height`` and bounds the moments and the sampler only."""
        end = self.tree_height.end_time

        return self.tree_height.start_time > 0 or (end is not None and end < np.inf)

    def _assert_not_windowed(self) -> None:
        """
        Raise if the coalescent accumulates over a bounded window (see ``_windowed``). Every pdf, cdf and quantile
        describes the reward accumulated from time 0 to absorption, so the distribution functions of the tree height,
        of a 1D reward and of a joint or conditional reward call this before evaluating.

        :raises NotImplementedError: If ``start_time > 0`` or ``end_time`` is finite.
        """
        if self._windowed:
            start, end = self.tree_height.start_time, self.tree_height.end_time
            raise NotImplementedError(
                "pdf, cdf and quantile are not implemented for a coalescent with a bounded accumulation window "
                f"(start_time={start}, end_time={end}). The window applies to the moments and the sampler, while the "
                "distribution functions describe the reward accumulated from time 0 to absorption."
            )

    @cached_property
    def _reward_distribution(self) -> 'RewardDistribution':
        """The accumulated-reward distribution of this distribution's own reward (cached for repeated CDF/PDF)."""
        return self.distribution()

    @cached_property
    def _reward_epoch_data(self) -> dict:
        """Reward-independent per-epoch transient generators for the accumulated-reward transform, built once and
        shared across all rewards on this state space (e.g. every bin of a spectrum)."""
        from .reward import _build_epoch_data

        return _build_epoch_data(self)

    @cached_property
    def _time_scale(self) -> float:
        """The accumulated-reward inversion time scale, the inverse of the largest transition rate or ``1.0`` near it.
        Rescales the LST inversion to keep the reward-shifted generator well-conditioned for large-N demographies; the
        transform value is invariant (see :func:`~phasegen.distributions.reward.time_scale`)."""
        from .reward import time_scale

        return time_scale(self)

    @property
    def _s_inf(self) -> float:
        r"""
        The :math:`s \to \infty` probe used for the atom :math:`\mathbb{P}(Y = 0) = \varphi(\infty)`, scaled by the
        inversion time scale as explained at ``RewardDistribution._s_inf``.
        """
        return 1e8 / self._time_scale

    @cached_property
    def _reward_epoch_data_scaled(self) -> dict:
        """:attr:`_reward_epoch_data` rescaled by :attr:`_time_scale` for the LST inversion (shared across all rewards
        on this state space). Identical to the unscaled data in the normal-N regime (``tau == 1``)."""
        from .reward import _scale_epoch_data

        return _scale_epoch_data(self._reward_epoch_data, self._time_scale)

    def _cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        r"""
        Cumulative distribution function of the accumulated reward, :math:`\mathbb{P}(Y \le t)`, via the
        Laplace-Stieltjes transform and its numerical inversion (see
        :class:`~phasegen.distributions.reward.RewardDistribution`).

        :param t: Value or values to evaluate the CDF at.
        :return: Cumulative probability.
        """
        return self._reward_distribution.cdf(t)

    def _pdf(self, t: float | Sequence[float], **kwargs) -> float | np.ndarray:
        """
        Probability density function of the accumulated reward.

        :param t: Value or values to evaluate the PDF at.
        :return: Density.
        """
        return self._reward_distribution.pdf(t)

    def _quantile(self, q: float) -> float:
        """
        The ``q``-quantile of the accumulated reward.

        :param q: Quantile in ``[0, 1]``.
        :return: The quantile.
        """
        return self._reward_distribution.quantile(q)

    def _plot_data_cdf(self, t: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The CDF curve of the accumulated reward (see :meth:`_reward_curves`).

        :param t: Points to evaluate at, ``None`` for the default grid.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._reward_curves('cdf', [('cdf', self._reward_distribution)], t, n_points, 'CDF')

    def _plot_data_pdf(self, t: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The density curve of the accumulated reward (see :meth:`_reward_curves`).

        :param t: Points to evaluate at, ``None`` for the default grid.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._reward_curves('pdf', [('pdf', self._reward_distribution)], t, n_points, 'PDF')

    def _plot_data_quantile(self, q: np.ndarray = None, n_points: int = None) -> '_CurveData':
        """
        The quantile curve of the accumulated reward (see :meth:`_reward_curves`).

        :param q: Probabilities to evaluate at, ``None`` for the default grid.
        :param n_points: Number of points of the default grid.
        :return: The curve.
        """
        return self._reward_curves('quantile', [('quantile', self._reward_distribution)], q, n_points,
                                   'Quantile function')

    @staticmethod
    def _reward_curves(
            kind: str,
            items: Sequence[Tuple[object, 'RewardDistribution']],
            grid: np.ndarray | None,
            n_points: int | None,
            title: str,
            legend_title: str = None
    ) -> '_CurveData':
        """
        The CDF, density or quantile curve of each ``(label, distribution)`` in ``items``, each evaluated through that
        distribution's own function. The default grid of a density or CDF ends at the largest
        :attr:`Settings.plot_endpoint_quantile` quantile, read off each distribution's CDF on a coarse grid.

        :param kind: The function kind, ``'pdf'``, ``'cdf'`` or ``'quantile'``.
        :param items: Label and distribution of each curve.
        :param grid: Points or probabilities to evaluate at, ``None`` for the default grid.
        :param n_points: Number of points of the default grid.
        :param title: Plot title.
        :param legend_title: Legend title.
        :return: The curves.
        """
        from ..visualization import _CurveData

        def end() -> float:
            q_end = Settings.plot_endpoint_quantile
            return max(float(np.interp(q_end, d.cdf(xs := np.linspace(0, d._range(), 256)), xs)) for _, d in items)

        x = DistributionFunction._default_grid(kind, grid, n_points, end)

        return _CurveData(
            x=x,
            y=np.array([getattr(d, kind)(x) for _, d in items]).reshape(len(items), len(x)),
            labels=[str(label) for label, _ in items],
            xlabel='q' if kind == 'quantile' else 'accumulated branch length',
            ylabel=dict(pdf='f(x)', cdf='F(x)', quantile='quantile')[kind],
            title=title,
            legend_title=legend_title
        )

    @cached_property
    def demes(self) -> MarginalDemeDistributions:
        """
        Marginal distributions over each deme.
        """
        return MarginalDemeDistributions(self)

    @cached_property
    def loci(self) -> MarginalLocusDistributions:
        """
        Marginal distributions over each locus.
        """
        return MarginalLocusDistributions(self)

    def sample(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> np.ndarray:
        r"""
        Draw independent samples of the accumulated reward :math:`R` by simulating trajectories of the Markov jump
        process, with the notation of :class:`~phasegen.distributions.PhaseTypeDistribution`.

        .. rubric:: Single epoch

        A trajectory starts in a state drawn from :math:`\boldsymbol{\alpha}`. In state :math:`x` it stays for an
        exponential holding time with rate :math:`\lambda(x) = -(\mathbf{S}_1)_{xx}` and then jumps to :math:`y \ne x`
        with probability :math:`(\mathbf{S}_1)_{xy} / \lambda(x)`. The sample collects the reward of every visited
        state,

        .. math::

            R = \sum_{j} r(x_j)\, D_j,

        where :math:`x_1, x_2, \dots` are the states visited before absorption and :math:`D_j` their holding times.

        .. rubric:: Several epochs

        On entering a state at time :math:`u_0`, a trajectory draws a hazard budget :math:`H \sim \mathrm{Exp}(1)`
        and leaves when the rates of the epochs it passes through have used it up,

        .. math::

            \int_{u_0}^{u_0 + D} \lambda_{i(u)}(x)\, \mathrm{d}u = H,

        with :math:`i(u)` the epoch containing :math:`u`. The unused budget is carried across epoch boundaries, and a
        jump follows the probabilities of the epoch in which it occurs. Only time within
        :math:`[t_\mathrm{start}, t_\mathrm{end}]` contributes to :math:`R`. A transient state with zero exit rate in
        the last epoch is never left, which gives an infinite sample for :math:`t_\mathrm{end} = \infty`.

        .. rubric:: Implementation

        - All trajectories advance together, one jump per step.
        - The jump probabilities of all epochs are stored as sparse rows of cumulative probabilities, so one sorted
          search draws the next state of every trajectory.
        - The state space and the intensity matrix of every epoch are built as for the exact computation.
        - Trajectories are simulated in batches of at most
          :attr:`Settings.sample_batch_size <phasegen.settings.Settings.sample_batch_size>`. Several batches draw
          from generators spawned from ``seed``, so a fixed seed yields different draws for different batch sizes.

        :param n_samples: Number of samples :math:`N`.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: The samples, of shape ``(n_samples,)``.
        """
        return self._sample(n_samples, rng=np.random.default_rng(seed)).reshape(n_samples)

    @staticmethod
    def _empirical_locus_agg(x: np.ndarray) -> np.ndarray:
        """Aggregation over the locus axis used when building the empirical distribution (sum by default; tree
        height overrides this with the maximum). Mirrors :class:`~phasegen.distributions.empirical.MsprimeCoalescent`."""
        return x.sum(axis=0)

    def to_empirical(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> 'EmpiricalPhaseTypeDistribution':
        r"""
        Build an empirical counterpart of this distribution from trajectories simulated as in
        :meth:`PhaseTypeDistribution.sample() <phasegen.distributions.PhaseTypeDistribution.sample>`, with the
        estimators of :class:`~phasegen.distributions.EmpiricalDistribution`.

        Each trajectory accumulates the reward :math:`R_{\ell d}` of every locus :math:`\ell` and deme :math:`d`, as
        restricted by the marginals :attr:`loci` and :attr:`demes`. The per-locus and per-deme marginals sum these
        over demes and over loci. The total sums over demes and over loci, except for the tree height, which is the
        maximum over loci of the per-locus heights, each summed over demes. All breakdowns share the trajectories, so their covariances are estimated jointly.

        :param n_samples: Number of trajectories.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: The empirical distribution.

        .. versionadded:: 2.0
        """
        from .empirical import EmpiricalPhaseTypeDistribution

        pops = self.lineage_config.pop_names
        n_loci = self.locus_config.n

        # stacked rewards over (locus, deme); one sampling pass yields the full (loci, demes) breakdown
        rewards = [self.loci[locus].demes[pop].reward for locus in range(n_loci) for pop in pops]
        sampled = self._sample(n_samples, rewards=rewards, rng=np.random.default_rng(seed))  # (n_samples, n_loci * n_demes)

        # (n_samples, n_loci, n_demes) -> (n_loci, n_demes, n_samples), the layout the empirical container expects
        samples = sampled.reshape(n_samples, n_loci, len(pops)).transpose(1, 2, 0)

        return EmpiricalPhaseTypeDistribution(samples, pops=pops, locus_agg=self._empirical_locus_agg)

    def _sample(
            self,
            n_samples: int,
            rewards: Sequence[Reward] = None,
            record_visits: bool = False,
            rng: np.random.Generator = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample the given rewards from shared trajectories, in batches with spawned child generators, as described in
        ``PhaseTypeDistribution.sample``.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is this distribution's reward.
        :param record_visits: Whether to also return the mean number of visits per trajectory to each state.
        :param rng: Generator to draw from, ``None`` for fresh entropy.
        :return: Array of sampled rewards of shape ``(n_samples, len(rewards))``, and optionally the visit counts.
        """
        if rewards is None:
            rewards = [self.reward]

        if rng is None:
            rng = np.random.default_rng()

        batch = Settings.sample_batch_size
        if batch is None or n_samples <= batch:
            return self._sample_vectorized(n_samples, rewards, record_visits, rng=rng)

        # bound peak memory by simulating the ensemble in batches and concatenating the per-trajectory results.
        # each batch draws from its own independent child generator, so the memory-batching does not couple the
        # batches' draw streams (a batch's samples do not depend on the preceding batches' sizes)
        sizes = [batch] * (n_samples // batch)
        if n_samples % batch:
            sizes.append(n_samples % batch)

        mass_parts, visits = [], None
        for size, child in zip(sizes, rng.spawn(len(sizes))):
            out = self._sample_vectorized(size, rewards, record_visits, rng=child)
            if record_visits:
                part, visited = out
                visits = visited * size if visits is None else visits + visited * size  # visit counts, re-averaged below
            else:
                part = out
            mass_parts.append(part)

        mass = np.concatenate(mass_parts, axis=0)

        if record_visits:
            return mass, visits / n_samples

        return mass

    def _sample_vectorized(
            self,
            n_samples: int,
            rewards: Sequence[Reward],
            record_visits: bool = False,
            rng: np.random.Generator = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate one batch of trajectories, as described in :meth:`sample`.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from.
        :param record_visits: Whether to also return the per-state visit frequencies.
        :return: Array of sampled rewards of shape ``(n_samples, len(rewards))`` (and visit frequencies if requested).
        """
        if rng is None:
            rng = np.random.default_rng()

        n_rewards = len(rewards)
        k = self.state_space.k
        absorbing = self.state_space.absorbing
        alpha = self.state_space.alpha
        R = np.array([r._get(self.state_space) for r in rewards])  # (n_rewards, k), epoch-invariant

        # accumulation window [t_a, t_b): reward accrues only for time within it (default [0, inf), i.e. to
        # absorption, in which case the clips below are no-ops)
        t_a = self.tree_height.start_time
        t_b = self.tree_height.end_time if self.tree_height.end_time is not None else np.inf

        # materialize the per-epoch generators once: exit rates and a sparse cumulative jump distribution. States,
        # rewards, absorption and the initial distribution are epoch-invariant, so only the rates differ across
        # epochs. The jump distribution is stored as a global flat CSR keyed by (epoch, state): each row holds its
        # neighbour states and the within-row cumulative jump probabilities, band-shifted by the global row id so a
        # single searchsorted draws the next state for every walker at once. This is O(nnz) rather than O(E * k^2)
        # in both memory and the categorical draw, lifting the ceiling on large (sparse) state spaces.
        end_times, lam_epochs = [], []
        indptr_list, neighbours_list, cum_list = [], [], []
        nnz = 0
        for ei, epoch in enumerate(self._get_epochs_until_unbounded()):
            self.state_space.update_epoch(epoch)
            S = sp.csr_matrix(self.state_space.S, dtype=float)
            lam = -S.diagonal()  # (k,) exit rates
            coo = S.tocoo()
            off = coo.row != coo.col  # off-diagonal jump rates only
            order = np.lexsort((coo.col[off], coo.row[off]))  # row-major, ascending destination within each source
            rows, cols, vals = coo.row[off][order], coo.col[off][order], coo.data[off][order]
            with np.errstate(divide='ignore', invalid='ignore'):
                probs = vals / lam[rows]  # row-normalized jump probabilities (lam > 0 for any sampled state)
            indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=k))))  # (k + 1,) per-epoch pointers
            csum = np.concatenate(([0.0], np.cumsum(probs)))
            within = csum[1:] - csum[indptr[rows]]  # cumulative probability within each source row
            end_times.append(epoch.end_time)
            lam_epochs.append(lam)
            indptr_list.append(indptr[:-1] + nnz)  # shift the row pointers into the global flat arrays
            neighbours_list.append(cols)
            cum_list.append(within + (ei * k + rows))  # band-shift by global row id -> globally sorted
            nnz += cols.size

        end_times = np.array(end_times)  # (E,)
        lam_epochs = np.array(lam_epochs)  # (E, k)
        last = len(end_times) - 1
        # global flat CSR over the E * k rows: pointers, destination states and band-shifted cumulative probabilities
        cum_indptr = np.concatenate(indptr_list + [[nnz]])  # (E * k + 1,)
        cum_neighbours = np.concatenate(neighbours_list)  # (nnz,)
        cum_offsets = np.concatenate(cum_list)  # (nnz,) globally sorted

        # ensemble state
        state = rng.choice(k, size=n_samples, p=alpha)
        t = np.zeros(n_samples)
        e = np.zeros(n_samples, dtype=int)
        H = rng.exponential(size=n_samples)  # remaining hazard budget ~ Exp(1)
        mass = np.zeros((n_samples, n_rewards))
        states_visited = np.zeros(k) if record_visits else None
        if record_visits:
            np.add.at(states_visited, state, 1)  # each walker visits its initial state (drawn from alpha)
        active = ~absorbing[state]

        with np.errstate(over='ignore', invalid='ignore'):
            while active.any():

                # advance active walkers across whole epochs until their next event fits in the current epoch
                # (skipped entirely for a single (unbounded) epoch, where no boundary can be crossed)
                while last > 0:
                    a = np.where(active)[0]
                    if a.size == 0:
                        break
                    dur = end_times[e[a]] - t[a]  # time to the current epoch boundary (inf in the last epoch)
                    haz_to_boundary = lam_epochs[e[a], state[a]] * dur  # 0 for isolated states (lambda == 0)
                    cross = (e[a] < last) & (H[a] > haz_to_boundary)
                    if not cross.any():
                        break
                    ca = a[cross]
                    dca = end_times[e[ca]] - t[ca]
                    ov = np.clip(np.minimum(t[ca] + dca, t_b) - np.maximum(t[ca], t_a), 0.0, None)
                    mass[ca] += R[:, state[ca]].T * ov[:, None]
                    H[ca] -= lam_epochs[e[ca], state[ca]] * dca
                    t[ca] = end_times[e[ca]]
                    e[ca] += 1

                # every active walker now fires its event within its current epoch
                a = np.where(active)[0]
                lam = lam_epochs[e[a], state[a]]

                # degenerate non-absorption: a transient state with zero exit rate in the unbounded epoch never
                # absorbs (e.g. permanently isolated demes), giving an infinite reward
                stuck = lam == 0
                if stuck.any():
                    sa = a[stuck]
                    if t_b == np.inf:
                        # a stuck walker waits forever; only reward components with a positive rate in the stuck
                        # state diverge, a zero-rate component keeps its finite accumulated value
                        mass[sa] = np.where(R[:, state[sa]].T > 0, np.inf, mass[sa])
                    else:
                        # a finite window caps the wait, so even a stuck walker accrues a finite reward
                        ov = np.clip(t_b - np.maximum(t[sa], t_a), 0.0, None)
                        mass[sa] += R[:, state[sa]].T * ov[:, None]
                    active[sa] = False
                    keep = ~stuck
                    a, lam = a[keep], lam[keep]
                    if a.size == 0:
                        break

                dt = H[a] / lam
                ov = np.clip(np.minimum(t[a] + dt, t_b) - np.maximum(t[a], t_a), 0.0, None)
                mass[a] += R[:, state[a]].T * ov[:, None]
                t[a] += dt
                if t_b != np.inf:
                    active[a[t[a] >= t_b]] = False  # past the window end: done accruing

                # sample the next state via inverse-CDF on the sparse cumulative jump distribution: one global
                # searchsorted over the band-shifted cumulative probabilities, clipped to each walker's own row
                row = e[a] * k + state[a]
                q = rng.random(a.size) + row  # band-shifted uniform draw lands in this row's band
                pos = np.clip(np.searchsorted(cum_offsets, q, side='left'), cum_indptr[row], cum_indptr[row + 1] - 1)
                nxt = cum_neighbours[pos]
                state[a] = nxt
                if record_visits:
                    np.add.at(states_visited, nxt, 1)

                # resample the hazard budget for survivors; absorbed walkers leave the ensemble
                H[a] = rng.exponential(size=a.size)
                active[a[absorbing[nxt]]] = False

        if record_visits:
            states_visited /= n_samples
            return mass, states_visited

        return mass

    def _default_end_times(self) -> np.ndarray:
        """
        Default times of moment accumulation plots: :attr:`Settings.plot_n_grid` points up to the
        :attr:`Settings.plot_endpoint_quantile` quantile of the tree height, or up to ``tree_height.t_max`` on a
        windowed coalescent, whose tree height has no quantile function.

        :return: The times.
        """
        if self._windowed:
            end = self.tree_height.t_max
        else:
            end = self.tree_height.quantile(Settings.plot_endpoint_quantile)

        return np.linspace(0, end, Settings.plot_n_grid)

    @staticmethod
    def _reward_names(rewards: Sequence[Reward]) -> str:
        """
        Names of the reward classes, for plot titles.

        :param rewards: The rewards.
        :return: The comma-separated names without the ``Reward`` suffix.
        """
        return ', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)

    def _plot_accumulation_data(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> '_CurveData':
        """
        The accumulation of a moment over time that :meth:`plot_accumulation` draws.

        :param k: The order of the moment.
        :param end_times: Times at which to evaluate the moment. By default, :attr:`Settings.plot_n_grid` points up to
            the :attr:`Settings.plot_endpoint_quantile` quantile of the tree height.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the orderings of the rewards.
        :return: The curve, titled by the reward classes.
        """
        from ..visualization import _CurveData

        k = int(k)
        end_times = self._default_end_times() if end_times is None else np.asarray(list(end_times), dtype=float)
        rewards = (self.reward,) * k if rewards is None else rewards

        return _CurveData(
            x=end_times,
            y=np.atleast_2d(self.accumulate(k, end_times, rewards, center, permute)),
            labels=[''],
            xlabel='t',
            ylabel='moment',
            title=f"Moment accumulation ({self._reward_names(rewards)})"
        )

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot accumulation of moments at different times, one curve per polymorphic bin for a spectrum.

        .. note:: This differs from a CDF: it shows the accumulation of moments, not the probability of having reached
            absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, :attr:`~phasegen.settings.Settings.plot_n_grid`
            points up to the :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile of the tree height.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :return: Axes.
        """
        from ..visualization import Visualization

        data = self._plot_accumulation_data(k, end_times, rewards, center, permute)

        return Visualization.plot_curves(ax=ax, data=data, file=file, show=show, clear=clear, label=label, title=title)


class _ExpmFunction(_HazardGrid):
    """
    Grid of the tree-height quantile, described at ``TreeHeightDistribution``. The cdf and pdf evaluate
    ``TreeHeightDistribution._sweep`` pointwise and do not read the grid.
    """
    #: Number of grid nodes :math:`K`.
    _n_grid: int = 8192

    #: Number of octaves :math:`J` of the locating pass below ``t_max``, and its nodes per octave.
    _n_probe_octaves: int = 30
    _n_probe_per_octave: int = 16

    #: Step between the segment bounds of the second pass, in cumulative hazard above one and in its logarithm below.
    _segment_hazard_step: float = 1.0

    #: Smallest cumulative hazard that bounds a segment of the second pass.
    _min_segment_hazard: float = float(np.finfo(float).eps)

    #: Number of probe intervals of the locating pass that are additionally bounded, the steepest in cumulative
    #: hazard, and the share of ``_segment_hazard_step`` an interval must gain to qualify.
    _n_refine: int = 64
    _refine_hazard_share: float = 0.125

    def _cdf_grid(self, x_max: float = 0.0, q_max: float = 0.0) -> tuple:
        """The grid over ``[0, t_max]``, built once. The arguments are ignored, the grid always spans the support."""
        return self._shared('expm_cdf_grid', self._build_cdf_grid)

    def _steep_bounds(self, nodes: np.ndarray, hazard: np.ndarray) -> set:
        """
        The ends of the ``_n_refine`` intervals of ``nodes`` across which the cumulative hazard rises most, among
        those gaining more than ``_refine_hazard_share`` of ``_segment_hazard_step``.

        :param nodes: Ascending times.
        :param hazard: Cumulative hazard on them, ascending.
        :return: Times to bound a segment at.
        """
        jumps = np.diff(hazard)
        steep = np.argsort(jumps)[-self._n_refine:]
        steep = steep[jumps[steep] > self._segment_hazard_step * self._refine_hazard_share]

        return set(nodes[steep]) | set(nodes[steep + 1])

    def _build_cdf_grid(self) -> tuple:
        """
        Build the two-pass grid described at ``TreeHeightDistribution``.

        :return: The nodes and the cumulative hazard on them.
        """
        d = self._distribution
        t_max = float(d.t_max)

        # pass 1 (locate): octaves down from t_max, uniform within each, so one exponential covers each octave
        octaves = [0.0] + [t_max * 2.0 ** -k for k in range(self._n_probe_octaves, -1, -1)]
        x_probe, cdf_probe, _ = d._sweep_uniform(octaves, self._n_probe_per_octave * len(octaves))
        h_probe = np.maximum.accumulate(self._hazard(cdf_probe))

        # pass 2 (resolve): segment bounds at equal steps of that hazard above one and at equal steps of its logarithm
        # below, down to the smallest resolved hazard, plus the epoch kinks
        step = self._segment_hazard_step
        h_min = max(h_probe[h_probe > 0][0], self._min_segment_hazard)
        levels = np.concatenate([np.exp(np.arange(np.log(h_min), 0.0, step)), np.arange(1.0, h_probe[-1], step)])
        bounds = set(np.interp(levels, h_probe, x_probe))
        bounds |= {e.start_time for e in d._get_epochs_until_unbounded() if 0.0 < e.start_time < t_max}

        # bound the probe intervals across which the hazard rises most: a rise that is narrow in time and gains less
        # than one step of hazard attracts no level, and the nodes within a segment are uniform in time, so the
        # interpolated hazard would run straight across it
        bounds |= self._steep_bounds(x_probe, h_probe)

        nodes, cdf, _ = d._sweep_uniform(sorted(bounds | {0.0, t_max}), self._n_grid)

        return nodes, np.maximum.accumulate(self._hazard(cdf))


class _ExpmCumulativeDistributionFunction(_ExpmFunction, _GridCumulativeDistributionFunction):
    """The tree-height CDF, evaluated pointwise by ``TreeHeightDistribution._sweep``."""

    def __call__(self, t) -> 'np.ndarray | float':
        """
        The CDF of the tree height, described at :class:`~phasegen.distributions.TreeHeightDistribution`.

        :param t: Point or array of points at which to evaluate the CDF.
        :return: The CDF at ``t``, of the same shape.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        :raises ValueError: If any ``t`` is negative.
        """
        d = self._distribution
        d._assert_not_windowed()

        if not isinstance(d.reward, TreeHeightReward):
            raise NotImplementedError("CDF not implemented for non-default rewards.")

        ta = np.asarray(t, dtype=float)

        if np.any(ta < 0):
            raise ValueError("Negative values are not allowed.")

        # the sweep is monotone in time, so evaluate the flattened points in sorted order and restore the caller's
        # order and shape afterwards
        # NaN points are passed through, the sweep taking the others
        flat = ta.ravel()
        finite = d._sweepable(flat)
        order = np.argsort(flat[finite])
        probs = np.full_like(flat, np.nan)
        probs[np.flatnonzero(finite)[order]] = d._sweep(flat[finite][order])[0]
        probs[np.isinf(flat) & ~finite] = 1.0

        if np.isnan(probs[finite]).any():
            d._logger.critical("NaN values in CDF. This is likely due to an ill-conditioned rate matrix.")

        return probs.reshape(ta.shape) if ta.ndim > 0 else float(probs[0])


class _ExpmQuantileFunction(_ExpmFunction, _GridQuantileFunction):
    """The tree-height quantile, read from the grid of ``_ExpmFunction._cdf_grid``."""

    def __call__(self, q) -> 'np.ndarray | float':
        """
        The quantile function of the tree height, described at
        :class:`~phasegen.distributions.TreeHeightDistribution`.

        :param q: Probability level or array of levels in :math:`[0, 1]`.
        :return: The quantiles, of the same shape as ``q``.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        :raises ValueError: If any ``q`` lies outside :math:`[0, 1]`.
        """
        self._distribution._assert_not_windowed()

        qa = np.atleast_1d(np.asarray(q, dtype=float))

        if np.any((qa < 0) | (qa > 1)):
            raise ValueError("Specified quantile must be between 0 and 1.")

        out = self._interp_quantile(qa, *self._cdf_grid())

        return out if np.ndim(q) > 0 else float(out[0])


class _ExpmDensityFunction(_ExpmFunction, _GridDensityFunction):
    """The tree-height density, evaluated pointwise by ``TreeHeightDistribution._sweep``."""

    def __call__(self, t) -> 'np.ndarray | float':
        """
        The density of the tree height, described at :class:`~phasegen.distributions.TreeHeightDistribution`.

        :param t: Point or array of points at which to evaluate the density.
        :return: The density at ``t``, of the same shape.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        :raises ValueError: If any ``t`` is negative.
        """
        d = self._distribution
        d._assert_not_windowed()

        if not isinstance(d.reward, TreeHeightReward):
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        ta = np.asarray(t, dtype=float)

        if np.any(ta < 0):
            raise ValueError("Negative values are not allowed.")

        flat = ta.ravel()
        finite = d._sweepable(flat)
        order = np.argsort(flat[finite])
        dens = np.full_like(flat, np.nan)
        dens[np.flatnonzero(finite)[order]] = d._sweep(flat[finite][order])[1]
        dens[np.isinf(flat) & ~finite] = 0.0

        return dens.reshape(ta.shape) if ta.ndim > 0 else float(dens[0])


class TreeHeightDistribution(PhaseTypeDistribution, DensityAwareDistribution):
    r"""
    Distribution of the tree height, the absorption time :math:`\tau`, with the notation of
    :class:`~phasegen.distributions.PhaseTypeDistribution`. Its ``cdf``, ``pdf`` and ``quantile`` are evaluated by
    matrix exponentiation.

    .. rubric:: Single epoch

    For a time-homogeneous process, :math:`\tau` is phase-type distributed with CDF and density

    .. math::

        F(x) = 1 - \boldsymbol{\alpha}_T\, e^{\mathbf{T} x}\, \mathbf{e}_T,
        \qquad
        f(x) = \boldsymbol{\alpha}_T\, e^{\mathbf{T} x}\, \mathbf{q},

    where :math:`\boldsymbol{\alpha}_T e^{\mathbf{T} x}` holds the probabilities of the transient states at time
    :math:`x` and :math:`\mathbf{q}` their absorption rates (Bladt and Nielsen, 2017). The density is read off the
    same vector as the CDF, without differencing.

    .. rubric:: Several epochs

    For :math:`x` in epoch :math:`\ell`, the exponential is replaced by the product of the epoch exponentials up to
    :math:`x`,

    .. math::

        e^{\mathbf{T}_1 \Delta_1} \cdots e^{\mathbf{T}_{\ell - 1} \Delta_{\ell - 1}}\,
        e^{\mathbf{T}_\ell (x - t_{\ell - 1})},

    and the density uses the absorption rates :math:`\mathbf{q}_\ell` of that epoch. At a boundary
    :math:`x = t_\ell` the density uses :math:`\mathbf{q}_\ell`, the absorption rates of the epoch ending there.

    .. rubric:: Implementation

    - The state probabilities are propagated from point to point in ascending order. The exponential is formed
      densely by the active :class:`~phasegen.expm.Backend` below :attr:`Settings.expm_action_min_dim
      <phasegen.settings.Settings.expm_action_min_dim>` states, and is otherwise applied as a sparse action (Al-Mohy
      and Higham, 2011).
    - The quantile is read from the cumulative-hazard grid of :class:`~phasegen.distributions.QuantileFunction` on
      :math:`[0, t_\mathrm{max}]`, with :math:`t_\mathrm{max}` given by :attr:`TreeHeightDistribution.t_max
      <phasegen.distributions.TreeHeightDistribution.t_max>`. A first pass over octaves below :math:`t_\mathrm{max}`
      locates the rise of the CDF. The grid nodes are then spread over segments whose cumulative hazard grows by a
      constant factor below one and by a constant step above, with the epoch boundaries as nodes.
    - A coalescent with a start time above 0 or a finite end time raises :class:`NotImplementedError`.

    .. rubric:: References

    Al-Mohy, A. H. and Higham, N. J. (2011). Computing the action of the matrix exponential, with an application to
    exponential integrators. SIAM Journal on Scientific Computing 33(2), 488-511.

    Bladt, M. and Nielsen, B. F. (2017). Matrix-Exponential Distributions in Applied Probability. Springer, New York.
    """
    _cdf_function = _ExpmCumulativeDistributionFunction
    _pdf_function = _ExpmDensityFunction
    _quantile_function = _ExpmQuantileFunction

    @staticmethod
    def _empirical_locus_agg(x: np.ndarray) -> np.ndarray:
        """The (total) tree height across loci is the deepest per-locus height, so aggregate by the maximum over
        the locus axis (matching :class:`~phasegen.distributions.empirical.MsprimeCoalescent.tree_height`)."""
        return x.max(axis=0)

    @cached_property
    def demes(self) -> MarginalDemeDistributions:
        """
        Marginal tree-height distributions over each deme. Defined for a single locus only: the multi-locus tree
        height is the maximum over loci, which has no additive per-deme decomposition (unlike the total branch
        length), so the per-deme breakdown is ill-posed under recombination.
        """
        if self.locus_config.n > 1:
            raise NotImplementedError(
                "Per-deme tree height is not defined for multiple loci: the two-locus tree height is the maximum "
                "over loci, which has no additive per-deme decomposition. Use total_branch_length.demes (additive) "
                "for the per-deme breakdown under recombination, or restrict to a single locus."
            )

        return MarginalDemeDistributions(self)

    #: Maximum number of time we double the end time when determining time to almost sure absorption.
    max_iter: int = 64

    #: Probability of almost sure absorption.
    p_absorption: float = 1 - 1e-15

    def __init__(
            self,
            state_space: LineageCountingStateSpace,
            demography: Demography = None,
            start_time: float = 0,
            end_time: float = None
    ) -> None:
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param demography: The demography.
        :param start_time: Time when to start accumulating moments.
        :param end_time: Time when to end accumulation of moments. By default, the time until almost sure absorption.
        """
        if not start_time >= 0:
            raise ValueError(f"Start time must be greater than or equal to 0, got {start_time}.")

        if end_time is not None and not end_time >= 0:
            raise ValueError(f"End time must be greater than or equal to 0, got {end_time}.")

        if end_time is not None and end_time < start_time:
            raise ValueError("End time must be greater than equal start time.")

        super().__init__(
            state_space=state_space,
            tree_height=self,
            demography=demography,
            reward=TreeHeightReward()
        )

        #: State space
        self.state_space: LineageCountingStateSpace = state_space

        #: Start time
        self.start_time: float = start_time

        #: End time
        self.end_time: float | None = end_time

    #: Largest row-sum norm of ``S tau`` exponentiated in one step by ``_propagate``.
    _max_step_norm: float = 1e3

    def _propagate(self, w: np.ndarray, tau: float) -> np.ndarray:
        """
        Advance the state distribution ``w`` by ``tau`` within the current epoch, by the dense exponential below
        ``Settings.expm_action_min_dim`` states and by the sparse action at or above it. A step whose exponent
        exceeds ``_max_step_norm`` is split into steps of doubling length, and propagation ends once a step leaves the
        transient entries unchanged. The absorbing states never feed the transient ones, so those entries are then
        stationary, and the CDF and density they carry are final. This makes ``tau`` of any size, infinity included,
        a finite number of exponentials.

        :param w: The row vector to advance.
        :param tau: Time to advance by, within the current epoch.
        :return: The advanced row vector.
        """
        if tau <= 0:
            return w

        S = self.state_space.S
        norm = float(abs(S).sum(axis=1).max())
        h = self._max_step_norm / norm if norm > 0 else tau

        while tau > 0:
            step = min(tau, h)

            # ``expm_multiply`` computes ``exp(a) @ b``, so the left action ``w @ exp(S tau)`` is ``exp(S^T tau) @ w``
            if self.state_space.k >= Settings.expm_action_min_dim:
                v = Backend.expm_multiply((sp.csr_matrix(S) * step).T.tocsr(), w)
            else:
                v = w @ expm(self._dense_rate_matrix() * step)

            stationary = np.array_equal(v * self._e, w * self._e)
            w, tau, h = v, tau - step, 2 * h

            if stationary:
                break

        return w

    @cached_property
    def _e(self) -> np.ndarray:
        """
        Indicator of the transient states, one there and zero on the absorbing states.
        """
        return self.reward._get(self.state_space)

    def _cum(self, w: np.ndarray) -> float:
        """
        The CDF carried by the propagated state distribution ``w``, the absorbed share of its mass, see ``_sweep``.
        Summing the absorbed entries keeps the relative precision of a small CDF.

        :param w: The propagated row vector.
        :return: Cumulative probability.
        """
        return float(w @ (1 - self._e) / w.sum())

    def _sweep_to(self, w: np.ndarray, u_prev: float, u: float, epoch: 'Epoch') -> np.ndarray:
        """
        Advance the row vector from ``u_prev`` to ``u``, crossing whatever epoch boundaries lie between (the rate
        matrix changes at each, so the exponential is taken piecewise). Leaves the state space updated to the epoch
        whose interval ends at or after ``u``, whose rate matrix the caller needs to read off the density, so that a
        ``u`` on a boundary takes the epoch ending there.

        :param w: The row vector at ``u_prev``.
        :param u_prev: Time the vector is currently at.
        :param u: Time to advance to.
        :param epoch: Epoch containing ``u_prev``.
        :return: The row vector at ``u``.
        """
        self.state_space.update_epoch(epoch)

        while u > epoch.end_time:
            self._check_numerical_stability(self.state_space.S, epoch.index)
            w = self._propagate(w, epoch.end_time - u_prev)

            u_prev = epoch.end_time
            epoch = self.demography.get_epoch(epoch.end_time)
            self.state_space.update_epoch(epoch)

        self._check_numerical_stability(self.state_space.S, epoch.index)

        return self._propagate(w, u - u_prev)

    def _sweepable(self, t: np.ndarray) -> np.ndarray:
        """
        The times ``_sweep`` evaluates: those not NaN, and not infinite on a demography with infinitely many epochs,
        where the sweep would never reach infinity. There the CDF is 1 and the density 0, the demography absorbing
        almost surely, which ``t_max`` asserts.

        :param t: Times.
        :return: Mask of the times the sweep evaluates.
        """
        mask = ~np.isnan(t)

        if np.isinf(t).any() and not self.demography._has_finitely_many_epochs:
            _ = self.t_max
            mask &= ~np.isinf(t)

        return mask

    def _exit_rates(self) -> np.ndarray:
        """The per-state absorption rates of the current epoch, see ``_sweep``."""
        return -(self.state_space.S @ self._e)

    def _sweep(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        The exact CDF and density at the ascending times ``t``, in one pass. The state distribution
        :math:`\mathbf{p}(x)` is propagated through the epochs, and at each time
        :math:`F = \mathbf{p}\,(\mathbf{1} - \mathbf{h}) / \mathbf{p}\,\mathbf{1}` and
        :math:`f = \mathbf{p}\,(-\mathbf{S}_\ell\,\mathbf{h})` are read off, with :math:`\mathbf{h}` the indicator of
        the transient states (``_e``), :math:`\mathbf{1}` the vector of ones and :math:`\ell` the epoch
        ``_sweep_to`` leaves the state space in (``TreeHeightDistribution``).

        :param t: Ascending times to evaluate at.
        :return: The CDF and the density at ``t``.
        """
        w = np.asarray(self.state_space.alpha, dtype=float)
        epoch = self.demography.get_epoch(0)
        u_prev = 0.0

        cdf, pdf = np.zeros(len(t)), np.zeros(len(t))

        for i, u in enumerate(t):
            w = self._sweep_to(w, u_prev, float(u), epoch)
            epoch = self.state_space.epoch

            cdf[i] = self._cum(w)
            pdf[i] = float(w @ self._exit_rates())
            u_prev = float(u)

        return cdf, pdf

    def _sweep_uniform(self, bounds: Sequence[float], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The exact CDF and density on roughly ``n`` nodes, spread uniformly *within* each segment of ``bounds``, with
        every bound landing exactly on a node. The node budget is split equally between segments, so the segmentation
        is what grades the nodes (see ``_ExpmFunction._build_cdf_grid``, which chooses the segments by cumulative
        hazard).

        Uniform within a segment is what makes this affordable at large ``n``: when the segment lies within a single
        epoch the propagator ``exp(S dt)`` is the same for every step, so the dense path forms one exponential per
        segment and applies it repeatedly rather than one per node. A segment that straddles an epoch boundary cannot
        share one propagator (the rate matrix changes at the boundary), so each of its steps is taken piecewise via
        :meth:`_sweep_to`. The segment boundaries (set by cumulative hazard) do not align with the epoch boundaries, so
        this case does arise; it is rare, so the per-segment fast path is kept for the common one.

        :param bounds: Ascending segment boundaries, the first of which the propagation starts from.
        :param n: Approximate total number of nodes.
        :return: The nodes, the CDF and the density on them.
        """
        w = np.asarray(self.state_space.alpha, dtype=float)
        self.state_space.update_epoch(self.demography.get_epoch(bounds[0]))

        nodes, cdf, pdf = [bounds[0]], [self._cum(w)], [float(w @ self._exit_rates())]
        n_seg = max(1, int(round(n / (len(bounds) - 1))))

        for start, end in zip(bounds[:-1], bounds[1:]):
            dt = (end - start) / n_seg
            start_epoch = self.demography.get_epoch(start)

            if start_epoch.end_time >= end:
                # the whole segment lies within one epoch: exponentiate once and reuse the propagator for every step
                self.state_space.update_epoch(start_epoch)
                self._check_numerical_stability(self.state_space.S, start_epoch.index)

                dense = self.state_space.k < Settings.expm_action_min_dim
                P = expm(self._dense_rate_matrix() * dt) if dense else None
                s = self._exit_rates()

                for j in range(n_seg):
                    w = w @ P if dense else self._propagate(w, dt)
                    nodes.append(start + (j + 1) * dt)
                    cdf.append(self._cum(w))
                    pdf.append(float(w @ s))

            else:
                # an epoch boundary crosses this segment: propagate each step piecewise, switching the rate matrix at
                # the boundary, and read the density off the epoch active at the node ``_sweep_to`` leaves us in
                for j in range(n_seg):
                    u_prev, u = start + j * dt, start + (j + 1) * dt
                    w = self._sweep_to(w, u_prev, u, self.demography.get_epoch(u_prev))
                    nodes.append(u)
                    cdf.append(self._cum(w))
                    pdf.append(float(w @ self._exit_rates()))

        return np.array(nodes), np.array(cdf), np.array(pdf)

    @cached_property
    def t_max(self) -> float:
        """
        Time until which computations are performed. This is either the end time specified when initializing
        the distribution or the time until almost sure absorption.
        """
        if self.end_time is not None:
            return self.end_time

        t_abs = self._get_absorption_time()

        if t_abs < self.start_time:
            raise ValueError(
                f"Determined time of almost sure absorption ({t_abs:.1f}) "
                f"is smaller than start time ({self.start_time:.1f}). "
                "The start time may be too large or the demography not well-defined."
            )

        return t_abs

    def _get_absorption_scale(self) -> float:
        r"""
        The time scale of the coalescent, the mean time to absorption under the first epoch held constant,

        .. math::
            \mathbb{E}[\tau_1] = \boldsymbol{\alpha}_T (-\mathbf{S}_T)^{-1} \mathbf{1},

        with :math:`\boldsymbol{\alpha}_T` the initial distribution on the transient states :math:`T` of the first
        epoch, :math:`\mathbf{S}_T` its transient sub-generator and :math:`\mathbf{1}` a vector of ones. It is where
        the doubling search of :meth:`_get_absorption_time` starts and the scale that
        :meth:`_extension_is_negligible` measures against, and it carries the time scale of the coalescent model
        itself, which the population size alone does not: the Dirac and Beta rates are divided by :math:`N^2` and by
        a multiple of :math:`N^{\alpha - 1}` where the Kingman rates are divided by :math:`N`. The mean population
        size of the first epoch is used instead where that system has no positive solution, as when no state of the
        first epoch can reach absorption.

        :return: A positive time.
        """
        epoch = self.demography.get_epoch(0)
        times = self._mean_absorption_times(epoch)

        if times.size:
            transient = np.where(self._e > 0)[0]
            seed = float(np.asarray(self.state_space.alpha, dtype=float)[transient] @ times)

            if np.isfinite(seed) and seed > 0:
                return seed

        return float(np.mean(list(epoch.pop_sizes.values())))

    #: Share of the first epoch's absorption scale that holding the final epoch until absorption may misplace, and
    #: the most epochs consumed past almost sure absorption in search of one that may be held.
    _extension_tol: float = 1e-8
    _max_extension_epochs: int = 1000

    def _mean_absorption_times(self, epoch: Epoch) -> np.ndarray:
        r"""
        The mean time to absorption from each transient state under ``epoch`` held constant,
        :math:`(-\mathbf{S}_T)^{-1}\mathbf{1}`, with :math:`\mathbf{S}_T` the transient sub-generator of that epoch
        and :math:`\mathbf{1}` the vector of ones on the transient states. The state space is left on ``epoch``.

        :param epoch: The epoch whose rates are held.
        :return: The mean absorption time per transient state, empty where the system has no positive solution.
        """
        self.state_space.update_epoch(epoch)

        transient = np.where(self._e > 0)[0]
        sparse = self._solve_sparse(len(transient))

        try:
            with warnings.catch_warnings():
                # an exactly singular ``-S_T``, a transient state without a path to absorption, raises on both paths
                warnings.simplefilter('error', sla.LinAlgWarning)
                solve = self._lu_solver(-self._transient_block(transient, sparse=sparse), sparse)
            times = np.asarray(solve(np.ones(len(transient))), dtype=float)
        except (np.linalg.LinAlgError, sla.LinAlgWarning, RuntimeError, ValueError):
            return np.empty(0)

        return times if np.all(np.isfinite(times)) and np.all(times > 0) else np.empty(0)

    def _survival(self, t: float) -> float:
        """
        The probability still transient at ``t``, read off the propagated state distribution. This is the quantity
        the accumulation weighs against the time scale of a later epoch, and ``1 - cdf(t)`` cannot stand in for it:
        a survival below the double-precision resolution of a number near one reads as exactly zero there while it
        is still represented here.

        :param t: Time at which to read the survival.
        :return: The transient probability mass.
        """
        epoch = self.demography.get_epoch(0)
        w = self._sweep_to(np.asarray(self.state_space.alpha, dtype=float), 0.0, float(t), epoch)

        return float(w @ self._e / w.sum())

    def _extension_is_negligible(self, epoch: Epoch, survival: float, scale: float) -> bool:
        """
        Whether holding ``epoch`` until absorption misplaces at most ``_extension_tol`` of ``scale``. The reward the
        extension attributes to the wrong epoch is bounded by the probability still transient at its start times the
        longest mean absorption time under its own rates, so an epoch whose own time scale dwarfs that probability
        is not a valid stand-in for the epochs after it.

        :param epoch: The epoch that would be held until absorption.
        :param survival: Transient probability at the time the extension starts from.
        :param scale: Time scale the tolerance is taken relative to.
        :return: Whether the extension is within tolerance.
        """
        times = self._mean_absorption_times(epoch)

        return not times.size or survival * float(times.max()) <= self._extension_tol * scale

    def _get_absorption_time(self) -> float:
        """
        Get a time estimate for when we have reached absorption almost surely.
        We base this computation on the transition matrix rather than the moments, because here
        we have a good idea about how likely absorption is, and can warn the user if necessary.
        Stopping the computation when no more rewards are accumulated is not a good idea, as this
        can happen before almost sure absorption (exponential runaway growth, temporary isolation in different demes).
        """
        i = 0
        epoch = self.demography.get_epoch(0)

        self._check_demography_conditioning()

        t = self._get_absorption_scale()
        expansion_factor = 2

        w = self._sweep_to(np.asarray(self.state_space.alpha, dtype=float), 0.0, t, epoch)
        p = self._cum(w)

        # the states that reach absorption in the unbounded epoch, computed on first entering it
        reach = None

        # multiple time by expansion_factor until we reach p_absorption
        while p < self.p_absorption and i < self.max_iter:
            w = self._sweep_to(w, t, t * expansion_factor, self.demography.get_epoch(t))
            t = t * expansion_factor
            p = self._cum(w)

            if self.demography.get_epoch(t).end_time == np.inf:
                if reach is None:
                    _, reach = self._reaches_absorption()

                self._assert_absorbs(w, reach)

            if np.isnan(p):
                self._logger.critical(
                    "Could not reliably find time of almost sure absorption "
                    "as probability of absorption is NaN. "
                    "This is likely due to an ill-conditioned rate matrix. "
                    f"Using time {t:.1f}. "
                )

            i += 1

        if i == self.max_iter and p < self.p_absorption:
            self._logger.warning(
                "Could not reliably find time of almost sure absorption after maximum number of iterations. "
                f"Using time {t:.1f} with probability of absorption 1 - {1 - p:.1e}. "
                "This could be due to numerical imprecision, unreachable states or very large or small "
                "absorption times. You can set the end time manually (see `Coalescent.end_time`) or increase "
                "the maximum number of iterations (`TreeHeightDistribution.max_iter`)."
            )

        return t

    def _empirical_cdf(self, n_samples: int, reward: Reward = None, t: float | Sequence[float] = None) -> np.ndarray:
        """
        Generate an empirical cumulative distribution function (CDF) by sampling from the distribution.

        :param n_samples: Number of samples to generate.
        :param reward: Reward function to use for sampling. If not specified,
            the default reward of the distribution is used.
        :param t: Values at which to evaluate the CDF. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :return: Sorted array of sampled total rewards.
        """
        if t is None:
            t = self._default_end_times()

        samples = self._sample(n_samples, [reward] if reward is not None else None).reshape(n_samples)

        x = np.sort(samples)
        y = np.arange(1, n_samples + 1) / n_samples

        if x.ndim == 1:
            return np.interp(t, x, y, left=0.0)

    def _plot_empirical_cdf(
            self,
            n_samples: int = 1000,
            reward: Reward = None,
            t: float | Sequence[float] = None,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Empirical CDF'
    ) -> 'plt.Axes':
        """
        Plot the empirical cumulative distribution function (CDF).

        :param n_samples: Number of samples to generate.
        :param reward: Reward function to use for sampling. If not specified,
            the default reward of the distribution is used.
        :param t: Values at which to evaluate the CDF. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import _CurveData, Visualization

        if t is None:
            t = self._default_end_times()

        data = _CurveData(x=np.asarray(t, dtype=float), y=np.atleast_2d(self._empirical_cdf(n_samples, reward, t)),
                         labels=[''], xlabel='t', ylabel='F(t)', title=title)

        return Visualization.plot_curves(ax=ax, data=data, file=file, show=show, clear=clear, label=label)


class TotalBranchLengthDistribution(PhaseTypeDistribution):
    """
    Distribution of the total branch length of the coalescent tree, the accumulated reward that counts the lineages
    in each state, returned by :attr:`Coalescent.total_branch_length
    <phasegen.distributions.Coalescent.total_branch_length>`. Its moments are those of
    :class:`~phasegen.distributions.PhaseTypeDistribution`, and its ``cdf``, ``pdf`` and ``quantile`` are those of
    the :class:`~phasegen.distributions.RewardDistribution` of the same reward.
    """

    def __init__(
            self,
            state_space: StateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography = None,
            reward: Reward = None
    ) -> None:
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward. Defaults to the total-branch-length reward. The marginal views pass the total
            branch length restricted to one locus or deme.
        """
        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward if reward is not None else TotalBranchLengthReward()
        )

