"""Phase-type distribution (moment engine) and the tree-height distribution."""

import logging
from ..caching import cached_property
from typing import Tuple, Iterable, Sequence, Union, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp
from ..demography import Demography, Epoch
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalBranchLengthReward
from ..settings import Settings
from ..spectrum import SFS
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
    """
    Phase-type distribution for a piecewise time-homogeneous process.
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
        Standard deviation.
        """
        return self.var ** 0.5

    @cached_property
    def m2(self) -> float | SFS:
        """
        Second (non-central) moment.
        """
        return self.moment(k=2, center=False)

    def distribution(self, reward: Reward = None) -> 'RewardDistribution':
        r"""
        The distribution of the accumulated reward :math:`Y = \int_0^\tau r(X_s)\,\mathrm{d}s` to absorption, as a
        callable and plottable :class:`~phasegen.distributions.reward.RewardDistribution` (see there for how its
        CDF / PDF / quantiles are computed). The reward must be scalar (one value per state); for a spectrum, pass a
        single bin's reward.

        :param reward: The reward whose accumulation is distributed. Defaults to this distribution's own reward.
        :return: The accumulated-reward distribution.
        """
        from .reward import RewardDistribution

        return RewardDistribution(self, reward)

    def joint_distribution(self, reward_a: Reward, reward_b: Reward) -> 'JointRewardDistribution':
        """
        Joint distribution of two accumulated rewards (e.g. a pair of SFS bins within a tree, or a two-locus SFS
        entry across loci), as a callable :class:`~phasegen.distributions.reward.JointRewardDistribution` (see there
        for the joint transform, marginals, cross-moments and joint CDF / PDF).

        :param reward_a: The first reward.
        :param reward_b: The second reward.
        :return: The joint accumulated-reward distribution.

        .. versionadded:: 2.0
        """
        from .reward import JointRewardDistribution

        return JointRewardDistribution(self, reward_a, reward_b)

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
        """The accumulated-reward inversion time-scale (average Ne at ``t = 0``; ``1.0`` outside the large-N regime).
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
        Draw samples of the accumulated reward :math:`R = \int_0^{\tau} r(X_u)\,\mathrm{d}u` by forward-simulating
        trajectories of the Markov jump process, with :math:`X_u` its state at time :math:`u`, :math:`\tau` its
        absorption time and :math:`r` the reward.

        All trajectories are advanced together, one jump at a time. Each starts in a state drawn from the initial
        distribution :math:`\boldsymbol{\alpha}` and carries a hazard budget :math:`H \sim \mathrm{Exp}(1)`, redrawn
        after every jump. In a state with exit rate :math:`\lambda` the holding time is :math:`H / \lambda`, over which
        the reward :math:`r(X)` accrues, and the next state is drawn from the embedded jump chain. A trajectory whose
        budget outlasts the current epoch moves to the epoch boundary and spends :math:`\lambda` times the remaining
        epoch duration of its budget, so an epoch with :math:`\lambda = 0` adds reward without a jump. The cost scales
        with the number of samples rather than the number of states, and the samples are simulated in batches of
        :attr:`Settings.sample_batch_size <phasegen.settings.Settings.sample_batch_size>`.

        :param n_samples: Number of samples to draw.
        :param seed: Random seed, or a random number generator to draw from. ``None`` draws fresh entropy.
        :return: Array of sampled rewards of shape ``(n_samples,)``.
        """
        return self._sample(n_samples, rng=np.random.default_rng(seed)).reshape(n_samples)

    @staticmethod
    def _empirical_locus_agg(x: np.ndarray) -> np.ndarray:
        """Aggregation over the locus axis used when building the empirical distribution (sum by default; tree
        height overrides this with the maximum). Mirrors :class:`~phasegen.distributions.empirical.MsprimeCoalescent`."""
        return x.sum(axis=0)

    def to_empirical(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> 'EmpiricalPhaseTypeDistribution':
        """
        Build an empirical (sample-based) counterpart of this distribution by simulating ``n_samples`` trajectories.
        The returned object exposes the same statistic interface (``mean``/``var``/``pdf``/``cdf``/...) computed from
        the samples, broken down per deme (:attr:`demes`) and per locus (:attr:`loci`), and is directly comparable to
        the analytic distribution. The per-(locus, deme) breakdown is obtained by sampling the matching marginal
        rewards (``DemeReward``/``LocusReward``), exactly the rewards the analytic marginals use.

        :param n_samples: Number of trajectories to simulate.
        :param seed: Random seed, or a random number generator to draw from. ``None`` draws fresh entropy.
        :return: An :class:`~phasegen.distributions.empirical.EmpiricalPhaseTypeDistribution`.

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
        Sample the given rewards in batches of ``Settings.sample_batch_size``, by the simulation described in
        :meth:`sample`.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is the tree height reward.
        :param record_visits: Whether to record which states were visited during the sampling.
        :return: Array of sampled rewards of size (n_samples, len(rewards)),
                 and optionally an array of probabilities of visiting each state.
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
        for ei, epoch in enumerate(self.demography.epochs):
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
            if epoch.end_time == np.inf:
                break

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
        :attr:`Settings.plot_endpoint_quantile` quantile of the tree height.

        :return: The times.
        """
        return np.linspace(0, self.tree_height.quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

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
        :param permute: For cross-moments, whether to average over all permutations of rewards.
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

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, :attr:`~phasegen.settings.Settings.plot_n_grid`
            points up to the :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile of the tree height.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
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
    r"""
    Mixin owning the tree-height's matrix-exponential machinery for its function objects. The point evaluator is
    ``d._sweep`` (see :meth:`~phasegen.distributions.phase_type.TreeHeightDistribution._sweep`), which propagates
    :math:`\mathbf{w} = \boldsymbol{\alpha} \prod_e \exp(\mathbf{S}_e \tau_e)` through the epochs and reads off both
    :math:`F(t) = 1 - \mathbf{w}\,\mathbf{e}` and :math:`f(t) = -\mathbf{w}\,\mathbf{S}\,\mathbf{e}`, the CDF *and*
    its exact derivative, so nothing here differences the CDF numerically.

    An expm point is orders of magnitude cheaper than the de Hoog inversion of an LST distribution
    (:class:`~.base._LSTFunction`), so the cdf and pdf simply evaluate it and are exact at every point asked for. The
    quantile is the one function the transform cannot hand back directly, and it reads the shared
    :class:`~.base._HazardGrid` -- whose nodes are exact expm values throughout -- by inverse interpolation, in one
    vectorised pass.
    """
    #: Nodes of the grid the quantile inverts. Matches the LST grid's :attr:`~.base._LSTFunction._cos_n_grid`, and
    #: like it costs only an evaluation each (here a matrix-vector product) against the exponentials behind it.
    _n_grid: int = 8192

    #: Octaves the locating pass spans below ``t_max``, and its nodes per octave. Doubling down from ``t_max``, rather
    #: than stepping uniformly, is what makes the pass indifferent to *where* the mass sits: ``t_max`` is fixed by the
    #: slowest rate in the demography and the bulk by the fastest, so on a bottleneck the two are orders of magnitude
    #: apart.
    _n_probe_octaves: int = 30
    _n_probe_per_octave: int = 16

    #: Cumulative hazard per segment of the resolving pass. The segments are the level sets of the locating pass's
    #: hazard, so the nodes end up graded *by the curve*: dense where it turns, sparse through the long flat tail.
    _segment_hazard_step: float = 1.0

    def _cdf_grid(self, x_max: float = 0.0, q_max: float = 0.0) -> tuple:
        """The grid, built once and spanning the whole support (unlike the LST grid, there is no expensive far tail
        to extend into lazily: the nodes are exact everywhere and cost a matrix-vector product each)."""
        return self._shared('expm_cdf_grid', self._build_cdf_grid)

    def _build_cdf_grid(self) -> tuple:
        """
        The grid, in **two passes**, mirroring the two-pass cosine fit of an LST distribution: a coarse pass over
        octaves doubling down from ``t_max`` locates where the CDF actually rises, then the nodes are laid down
        uniformly *within* segments of equal cumulative hazard, read off that pass.

        Grading the nodes by the hazard rather than spreading them along the axis is what makes this accurate. A
        uniform grid over ``[0, t_max]`` spends its nodes where nothing happens: ``t_max`` is set by the slowest rate
        in the demography and the bulk by the fastest, so under a bottleneck almost every node lands in the empty tail
        and the few left across the bulk leave cells straddling percents of the mass.

        The epoch boundaries are forced in as segment bounds: the rates change there, so the CDF has a kink, and a
        node on the kink is what keeps the piecewise-linear hazard from cutting the corner.
        """
        d = self._distribution
        t_max = float(d.t_max)

        # pass 1 (locate): octaves down from t_max, uniform within each, so one exponential covers each octave
        octaves = [0.0] + [t_max * 2.0 ** -k for k in range(self._n_probe_octaves, -1, -1)]
        x_probe, cdf_probe, _ = d._sweep_uniform(octaves, self._n_probe_per_octave * len(octaves))
        h_probe = np.maximum.accumulate(self._hazard(cdf_probe))

        # pass 2 (resolve): segment bounds at equal steps of that hazard, plus the epoch kinks
        levels = np.arange(0.0, h_probe[-1], self._segment_hazard_step)
        bounds = set(np.interp(levels, h_probe, x_probe))
        bounds |= {e.start_time for e in d._get_epochs_until_unbounded() if 0.0 < e.start_time < t_max}
        bounds = sorted(bounds | {0.0, t_max})

        nodes, cdf, _ = d._sweep_uniform(bounds, self._n_grid)

        return nodes, np.maximum.accumulate(self._hazard(cdf))


class _ExpmCumulativeDistributionFunction(_ExpmFunction, _GridCumulativeDistributionFunction):
    r"""The tree-height CDF by direct matrix exponentiation,
    :math:`\mathbb{P}(\tau \le t) = 1 - \boldsymbol{\alpha} \prod_e \exp(\mathbf{S}_e \tau_e)\,\mathbf{e}`, exact at
    every point asked for."""

    def __call__(self, t) -> 'np.ndarray | float':
        """
        CDF ``P(R <= t)``, for a scalar or an array of ``t``.

        :param t: Point(s) at which to evaluate the CDF.
        :return: The CDF at ``t``, of the same shape.
        :raises NotImplementedError: If the distribution's reward is not the tree height.
        """
        d = self._distribution

        if not isinstance(d.reward, TreeHeightReward):
            raise NotImplementedError("CDF not implemented for non-default rewards.")

        ta = np.atleast_1d(np.asarray(t, dtype=float))

        if np.any(ta < 0):
            raise ValueError("Negative values are not allowed.")

        # the sweep is monotone in time, so evaluate in sorted order and restore the caller's order afterwards
        order = np.argsort(ta)
        probs = np.empty_like(ta)
        probs[order] = d._sweep(ta[order])[0]

        if np.isnan(probs).any():
            d._logger.critical("NaN values in CDF. This is likely due to an ill-conditioned rate matrix.")

        return probs if np.ndim(t) > 0 else float(probs[0])


class _ExpmQuantileFunction(_ExpmFunction, _GridQuantileFunction):
    """The tree-height quantile by inverse interpolation of the shared hazard grid (``_ExpmFunction._cdf_grid``),
    whose nodes carry exact matrix-exponential CDF values."""

    def __call__(self, q) -> 'np.ndarray | float':
        """
        The ``q``-quantile ``inf{t : F(t) >= q}``, for a scalar or an array of ``q``, in one vectorised pass.

        Levels beyond the grid's last node clamp to it: that node is the time of almost-sure absorption (or the
        user-supplied end time), so there is nothing above it to resolve.

        :param q: Probability level(s) in ``[0, 1]``.
        :return: The quantile(s), of the same shape as ``q``.
        :raises ValueError: If any ``q`` lies outside ``[0, 1]``.
        """
        qa = np.atleast_1d(np.asarray(q, dtype=float))

        if np.any((qa < 0) | (qa > 1)):
            raise ValueError("Specified quantile must be between 0 and 1.")

        out = self._interp_quantile(qa, *self._cdf_grid())

        return out if np.ndim(q) > 0 else float(out[0])


class _ExpmDensityFunction(_ExpmFunction, _GridDensityFunction):
    r"""The tree-height density by direct matrix exponentiation,
    :math:`f(t) = -\boldsymbol{\alpha} \prod_e \exp(\mathbf{S}_e \tau_e)\,\mathbf{S}\,\mathbf{e}`, the exit-rate
    reading of the same propagated vector the CDF is read off. Exact, and in particular not a finite difference of
    the CDF: subtracting CDF values a step ``quantile(0.99) / 1e10`` apart would throw away most of their
    significant digits."""

    def __call__(self, t) -> 'np.ndarray | float':
        """
        Density, for a scalar or an array of ``t``.

        :param t: Point(s) at which to evaluate the density.
        :return: The density at ``t``, of the same shape.
        :raises NotImplementedError: If the distribution's reward is not the tree height.
        """
        d = self._distribution

        if not isinstance(d.reward, TreeHeightReward):
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        ta = np.atleast_1d(np.asarray(t, dtype=float))

        if np.any(ta < 0):
            raise ValueError("Negative values are not allowed.")

        order = np.argsort(ta)
        dens = np.empty_like(ta)
        dens[order] = d._sweep(ta[order])[1]

        return dens if np.ndim(t) > 0 else float(dens[0])


class TreeHeightDistribution(PhaseTypeDistribution, DensityAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process that allows the computation of the
    density function. This is currently only possible with default rewards.

    The exact (matrix-exponential) cdf / pdf / quantile evaluation lives on the function objects
    (``_ExpmCumulativeDistributionFunction`` / ``_ExpmDensityFunction`` / ``_ExpmQuantileFunction``);
    this distribution supplies the state space, demography, epoch machinery and the exit vector they reach into.
    """
    #: the exact matrix-exponential function-object flavours (selected over the inherited LST/COS ones, whose CDF is
    #: itself a de Hoog inversion -- the expm path stays robust on ill-conditioned demographies)
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
    max_iter: int = 20

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
        if start_time < 0:
            raise ValueError("Start time must be greater than or equal to 0.")

        if end_time is not None and end_time < 0:
            raise ValueError("End time must be greater than or equal to 0.")

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

    def _propagate(self, w: np.ndarray, tau: float) -> np.ndarray:
        r"""
        Advance the row vector :math:`\mathbf{w} = \boldsymbol{\alpha} \prod \exp(\mathbf{S}\tau)` by ``tau`` in the
        *current* epoch: :math:`\mathbf{w} \mapsto \mathbf{w}\,\exp(\mathbf{S}\tau)`.

        Only :math:`\boldsymbol{\alpha}\,\mathbf{T}\,\mathbf{e}` is ever read off the propagator, so the row vector
        is carried rather than the :math:`k \times k` matrix, and the choice of how to apply the exponential follows
        the same configuration as the moment engine
        (:meth:`~._moments.MomentEvaluator._accumulate`): above :attr:`~phasegen.settings.Settings.expm_action_min_dim`
        the (sparse) matrix-exponential *action* is applied to the vector, below it the dense exponential is formed.
        A dense ``k x k`` exponential of a state space this machinery is asked for at large ``n`` is precisely what
        :attr:`~phasegen.settings.Settings.dense_rate_matrix_max_states` exists to avoid.

        :param w: The row vector to advance.
        :param tau: Time to advance by, within the current epoch.
        :return: The advanced row vector.
        """
        if tau <= 0:
            return w

        S = self.state_space.S

        # ``expm_multiply`` computes ``exp(a) @ b``, so the left action ``w @ exp(S tau)`` is ``exp(S^T tau) @ w``
        if self.state_space.k >= Settings.expm_action_min_dim:
            return Backend.expm_multiply((sp.csr_matrix(S) * tau).T.tocsr(), w)

        return w @ expm(self._dense_rate_matrix() * tau)

    @cached_property
    def _e(self) -> np.ndarray:
        """
        Exit vector.
        """
        return self.reward._get(self.state_space)

    def _cum(self, w: np.ndarray) -> float:
        r"""
        The cumulative probability carried by a propagated row vector: :math:`F(t) = 1 - \mathbf{w}\,\mathbf{e}`.

        :param w: The propagated row vector :math:`\boldsymbol{\alpha}\,\mathbf{T}`.
        :return: Cumulative probability.
        """
        return float(1 - w @ self._e)

    def _sweep_to(self, w: np.ndarray, u_prev: float, u: float, epoch: 'Epoch') -> np.ndarray:
        """
        Advance the row vector from ``u_prev`` to ``u``, crossing whatever epoch boundaries lie between (the rate
        matrix changes at each, so the exponential is taken piecewise). Leaves the state space updated to the epoch
        containing ``u``, whose rate matrix the caller needs to read off the density.

        :param w: The row vector at ``u_prev``.
        :param u_prev: Time the vector is currently at.
        :param u: Time to advance to.
        :param epoch: Epoch containing ``u_prev``.
        :return: The row vector at ``u``.
        """
        self.state_space.update_epoch(epoch)

        while u > epoch.end_time:
            self._check_numerical_stability(self.state_space.S, 0)
            w = self._propagate(w, epoch.end_time - u_prev)

            u_prev = epoch.end_time
            epoch = self.demography.get_epoch(epoch.end_time)
            self.state_space.update_epoch(epoch)

        self._check_numerical_stability(self.state_space.S, 0)

        return self._propagate(w, u - u_prev)

    def _exit_rates(self) -> np.ndarray:
        r"""The exit-rate vector :math:`-\mathbf{S}\,\mathbf{e}` of the *current* epoch: the density is
        :math:`f(t) = \mathbf{w}\,(-\mathbf{S}\,\mathbf{e})` for the propagated :math:`\mathbf{w}`, since
        :math:`F = 1 - \mathbf{w}\,\mathbf{e}` and
        :math:`\frac{\mathrm{d}}{\mathrm{d}t}(\mathbf{w}\,\mathbf{e}) = \mathbf{w}\,\mathbf{S}\,\mathbf{e}`."""
        return -(self.state_space.S @ self._e)

    def _sweep(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        The exact CDF *and* density at the ascending times ``t``, in one pass: propagate
        :math:`\mathbf{w} = \boldsymbol{\alpha}\,\mathbf{T}(u)` through the epochs, reading off
        :math:`F = 1 - \mathbf{w}\,\mathbf{e}` and :math:`f = \mathbf{w}\,(-\mathbf{S}\,\mathbf{e})` at each. The
        density is the exit-rate reading of the very same vector, so it costs one matrix-vector product and needs no
        finite difference.

        :param t: Ascending times to evaluate at.
        :return: The CDF and the density at ``t``.
        """
        w = np.asarray(self.state_space.alpha, dtype=float)
        epoch = self.demography.get_epoch(0)
        u_prev = 0.0

        cdf, pdf = np.zeros(len(t)), np.zeros(len(t))

        for i, u in enumerate(t):
            w = self._sweep_to(w, u_prev, float(u), epoch)
            epoch = self.demography.get_epoch(float(u))

            cdf[i] = self._cum(w)
            pdf[i] = float(w @ self._exit_rates())
            u_prev = float(u)

        return cdf, pdf

    def _sweep_uniform(self, bounds: Sequence[float], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The exact CDF and density on roughly ``n`` nodes, spread uniformly *within* each segment of ``bounds``, with
        every bound landing exactly on a node. The node budget is split equally between segments, so the segmentation
        is what grades the nodes (see ``_ExpmFunction._build_cdf_grid``, which chooses segments of equal
        cumulative hazard).

        Uniform within a segment is what makes this affordable at large ``n``: when the segment lies within a single
        epoch the propagator ``exp(S dt)`` is the same for every step, so the dense path forms one exponential per
        segment and applies it repeatedly rather than one per node. A segment that straddles an epoch boundary cannot
        share one propagator (the rate matrix changes at the boundary), so each of its steps is taken piecewise via
        :meth:`_sweep_to`. The segment boundaries (equal cumulative hazard) do not align with the epoch boundaries, so
        this case does arise; it is rare, so the per-segment fast path is kept for the common one.

        :param bounds: Ascending segment boundaries, the first of which the propagation starts from.
        :param n: Approximate total number of nodes.
        :return: The nodes, the CDF and the density on them.
        """
        w = np.asarray(self.state_space.alpha, dtype=float)
        self.state_space.update_epoch(self.demography.get_epoch(bounds[0]))

        nodes, cdf, pdf = [bounds[0]], [self._cum(w)], [float(w @ self._exit_rates())]
        n_seg = max(1, int(round(n / (len(bounds) - 1))))

        for i_seg, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            dt = (end - start) / n_seg
            start_epoch = self.demography.get_epoch(start)

            if start_epoch.end_time >= end:
                # the whole segment lies within one epoch: exponentiate once and reuse the propagator for every step
                self.state_space.update_epoch(start_epoch)
                self._check_numerical_stability(self.state_space.S, i_seg)

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

        t = 2 ** int(np.log2(np.mean(list(epoch.pop_sizes.values()))))
        expansion_factor = 2

        w = self._sweep_to(np.asarray(self.state_space.alpha, dtype=float), 0.0, t, epoch)
        p = self._cum(w)

        # multiple time by expansion_factor until we reach p_absorption
        while p < self.p_absorption and i < self.max_iter:
            w = self._sweep_to(w, t, t * expansion_factor, self.demography.get_epoch(t))
            t = t * expansion_factor
            p = self._cum(w)

            if np.isnan(p):
                self._logger.critical(
                    "Could not reliably find time of almost sure absorption "
                    "as probability of absorption is NaN. "
                    "This is likely due to an ill-conditioned rate matrix. "
                    f"Using time {t:.1f}. "
                )

            i += 1

        # if absorption was not reached, fail loudly for a demography that *never* absorbs rather than returning the
        # doubling ceiling (see :meth:`_assert_absorbs`).
        if p < self.p_absorption and not np.isnan(p):
            self._assert_absorbs(w)

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
            return np.interp(t, x, y)

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
    Distribution of the total branch length of the coalescent tree -- the accumulated lineage-counting reward (the
    sum of all branch lengths) to absorption. An explicitly named, thin container around
    :class:`PhaseTypeDistribution` carrying the total-branch-length reward, returned by
    :attr:`~phasegen.distributions.Coalescent.total_branch_length`; its moments and its callable-and-plottable
    ``cdf`` / ``pdf`` / ``quantile`` work like any accumulated-reward distribution.
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
        :param reward: The reward. Defaults to the total-branch-length reward; an explicit reward (e.g. the
            total branch length restricted to one locus / deme, as built by the marginal-distribution views) is
            accepted so generic ``cls(reward=...)`` construction works.
        """
        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward if reward is not None else TotalBranchLengthReward()
        )

