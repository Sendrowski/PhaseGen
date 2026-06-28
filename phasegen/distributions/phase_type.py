"""Phase-type distribution (moment engine) and the tree-height distribution."""

import logging
from ..caching import cached_property, cache
from typing import Tuple, Collection, Iterable, Sequence, Union, TYPE_CHECKING
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

from .base import CallableDistributionFunctions, DensityAwareDistribution, MarginalDemeDistributions, \
    MarginalLocusDistributions, MomentAwareDistribution, \
    _GridCumulativeDistributionFunction, _GridDensityFunction, _GridQuantileFunction
from ._moments import MomentEvaluator

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .reward import RewardDistribution, JointRewardDistribution

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

        #: Population configuration
        self.lineage_config: LineageConfig = state_space.lineage_config

        #: Locus configuration
        self.locus_config: LocusConfig = state_space.locus_config

        #: Reward
        self.reward: Reward = reward

        #: State space
        self.state_space: StateSpace = state_space

        #: Demography
        self.demography: Demography = demography

        #: Tree height distribution
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
        """
        Full distribution (CDF / PDF / quantiles) of the accumulated reward to absorption, for an arbitrary
        reward and demography, via the Laplace-Stieltjes transform and its numerical inversion. Where
        :attr:`mean` / :meth:`moment` give only the moments of the accumulated reward, this gives its
        distribution. The reward must be scalar (one value per state); for a spectrum, pass a single bin's reward.

        :param reward: The reward whose accumulation is distributed. Defaults to this distribution's own reward.
        :return: The accumulated-reward distribution.
        """
        from .reward import RewardDistribution

        return RewardDistribution(self, reward)

    def joint_distribution(self, reward_a: Reward, reward_b: Reward) -> 'JointRewardDistribution':
        """
        Joint distribution of two accumulated rewards — the distributional object behind a cross-moment
        ``E[R_a R_b]`` (e.g. a pair of SFS bins within a tree, or a two-locus SFS entry across loci). Provides the
        joint LST, the marginals, and the cross-moments/covariance/correlation; the joint CDF/PDF builds on it.

        :param reward_a: The first reward.
        :param reward_b: The second reward.
        :return: The joint accumulated-reward distribution.
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

    @cached_property
    def _reward_epoch_data_scaled(self) -> dict:
        """:attr:`_reward_epoch_data` rescaled by :attr:`_time_scale` for the LST inversion (shared across all rewards
        on this state space). Identical to the unscaled data in the normal-N regime (``tau == 1``)."""
        from .reward import _scale_epoch_data

        return _scale_epoch_data(self._reward_epoch_data, self._time_scale)

    def _cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function of the accumulated reward, ``P(R <= t)``, via the Laplace-Stieltjes
        transform and its numerical inversion (see :class:`RewardDistribution`).

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

    def _plot_cdf(self, ax: 'plt.Axes' = None, t: np.ndarray = None, n_points: int = None, show: bool = True,
                  file: str = None, clear: bool = True, label: str = None, title: str = 'CDF',
                  exact: bool = False) -> 'plt.Axes':
        """Plot the CDF curve of the accumulated reward (see :meth:`_plot_reward_curves`)."""
        return self._plot_reward_curves('cdf', [(label or 'cdf', self.reward)], ax, t, n_points, show, file, clear,
                                        title, exact)

    def _plot_pdf(self, ax: 'plt.Axes' = None, t: np.ndarray = None, n_points: int = None, show: bool = True,
                  file: str = None, clear: bool = True, label: str = None, title: str = 'PDF',
                  exact: bool = False) -> 'plt.Axes':
        """Plot the PDF curve of the accumulated reward (see :meth:`_plot_reward_curves`)."""
        return self._plot_reward_curves('pdf', [(label or 'pdf', self.reward)], ax, t, n_points, show, file, clear,
                                        title, exact)

    def _plot_quantile(self, ax: 'plt.Axes' = None, q: np.ndarray = None, n_points: int = None, show: bool = True,
                       file: str = None, clear: bool = True, label: str = None,
                       title: str = 'Quantile function', exact: bool = False) -> 'plt.Axes':
        """Plot the quantile function (accumulated reward versus probability ``q``)."""
        return self._plot_reward_curves('quantile', [(label or 'quantile', self.reward)], ax, q, n_points, show, file,
                                        clear, title, exact)

    def _plot_reward_curves(
            self,
            kind: str,
            items: Sequence[Tuple[object, Reward]],
            ax: 'plt.Axes',
            x: np.ndarray,
            n_points: int,
            show: bool,
            file: str,
            clear: bool,
            title: str,
            exact: bool = False
    ) -> 'plt.Axes':
        """
        Plot the CDF or PDF curve of each ``(label, reward)`` in ``items`` on one axes. By default this uses the fast
        COS inversion (sharing the per-epoch generators across all rewards on this state space); pass ``exact=True``
        to evaluate each point with the slower but more accurate per-point de Hoog inversion instead.
        """
        import matplotlib.pyplot as plt
        from ..visualization import Visualization

        if ax is None:
            ax = plt.gca()
            if clear:
                ax.clear()

        from .base import adaptive_grid

        dists = [(label, self.distribution(reward=reward)) for label, reward in items]
        user_x = x is not None

        if x is None:
            n_points = n_points or Settings.plot_n_grid
            q_end = Settings.plot_endpoint_quantile
            if kind == 'quantile':
                # the quantile function lives on the probability axis q in (0, 1)
                x = np.linspace(1.0 - q_end, q_end, n_points)
            else:
                # right end = the configured upper quantile, so a heavy upper tail does not stretch the view (mean +
                # many std can extend far past the mass). Derived cheaply from the COS CDF (one curve per bin) rather
                # than the per-point de Hoog quantile.
                end = max(
                    float(np.interp(q_end, d.cdf.curve(grid := np.linspace(0, d._range(), 256)), grid))
                    for _, d in dists
                )
                x = np.linspace(0, end, n_points)
        else:
            x = np.asarray(x, dtype=float)

        ylabel = {'cdf': 'F(x)', 'pdf': 'f(x)', 'quantile': 'quantile'}[kind]
        xlabel = 'q' if kind == 'quantile' else 'accumulated branch length'

        # for the expensive exact (de Hoog) cdf/pdf, place each bin's points adaptively where its own curve bends
        # (resolving e.g. the near-zero atom spike), unless the caller supplied an explicit grid
        adaptive = exact and not user_x and kind in ('cdf', 'pdf')

        for k, (label, d) in enumerate(dists):
            if adaptive:
                xk, y = adaptive_grid(d.cdf if kind == 'cdf' else d.pdf, 0.0, float(x[-1]), max_points=n_points)
            else:
                xk = x
                if kind == 'cdf':
                    y = d.cdf(x) if exact else d.cdf.curve(x)
                elif kind == 'pdf':
                    y = d.pdf(x) if exact else d.pdf.curve(x)
                elif exact:
                    # quantile function via the per-point de Hoog bisection
                    y = np.array([d.quantile(float(p)) for p in x])
                else:
                    # quantile function: invert the (fast) COS CDF curve by interpolation rather than a per-point
                    # bisection (which would re-run the de Hoog inversion at every probability and bin)
                    xx = np.linspace(0, d._range(), 512)
                    y = np.interp(x, d.cdf.curve(xx), xx)

            Visualization.plot(
                ax=ax,
                x=xk,
                y=y,
                xlabel=xlabel,
                ylabel=ylabel,
                label=str(label),
                file=file,
                show=(k == len(dists) - 1 and show),
                clear=clear,
                title=title
            )

        if kind == 'cdf':
            ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]

        return ax

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

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Draw samples of the accumulated reward by simulating trajectories through the underlying Markov chain.

        :param n_samples: Number of samples to draw.
        :return: Array of sampled rewards of shape ``(n_samples,)``.
        """
        return self._sample(n_samples).reshape(n_samples)

    @staticmethod
    def _empirical_locus_agg(x: np.ndarray) -> np.ndarray:
        """Aggregation over the locus axis used when building the empirical distribution (sum by default; tree
        height overrides this with the maximum). Mirrors :class:`~phasegen.distributions.empirical.MsprimeCoalescent`."""
        return x.sum(axis=0)

    def to_empirical(self, n_samples: int) -> 'EmpiricalPhaseTypeDistribution':
        """
        Build an empirical (sample-based) counterpart of this distribution by simulating ``n_samples`` trajectories.
        The returned object exposes the same statistic interface (``mean``/``var``/``pdf``/``cdf``/...) computed from
        the samples, broken down per deme (:attr:`demes`) and per locus (:attr:`loci`), and is directly comparable to
        the analytic distribution. The per-(locus, deme) breakdown is obtained by sampling the matching marginal
        rewards (``DemeReward``/``LocusReward``), exactly the rewards the analytic marginals use.

        :param n_samples: Number of trajectories to simulate.
        :return: An :class:`~phasegen.distributions.empirical.EmpiricalPhaseTypeDistribution`.
        """
        from .empirical import EmpiricalPhaseTypeDistribution

        pops = self.lineage_config.pop_names
        n_loci = self.locus_config.n

        # stacked rewards over (locus, deme); one sampling pass yields the full (loci, demes) breakdown
        rewards = [self.loci[locus].demes[pop].reward for locus in range(n_loci) for pop in pops]
        sampled = self._sample(n_samples, rewards=rewards)  # (n_samples, n_loci * n_demes)

        # (n_samples, n_loci, n_demes) -> (n_loci, n_demes, n_samples), the layout the empirical container expects
        samples = sampled.reshape(n_samples, n_loci, len(pops)).transpose(1, 2, 0)

        return EmpiricalPhaseTypeDistribution(samples, pops=pops, locus_agg=self._empirical_locus_agg)

    def _sample(
            self,
            n_samples: int,
            rewards: Sequence[Reward] = None,
            record_visits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate samples from the mean reward distribution by simulating CTMC trajectories with the vectorized
        ensemble sampler. Its memory scales with the number of trajectories (not the state count), so requests
        larger than :attr:`~phasegen.settings.Settings.sample_batch_size` are simulated in batches and concatenated
        to bound peak memory.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is the tree height reward.
        :param record_visits: Whether to record which states were visited during the sampling.
        :return: Array of sampled rewards of size (n_samples, len(rewards)),
                 and optionally an array of probabilities of visiting each state.
        """
        if rewards is None:
            rewards = [self.reward]

        batch = Settings.sample_batch_size
        if batch is None or n_samples <= batch:
            return self._sample_vectorized(n_samples, rewards, record_visits)

        # bound peak memory by simulating the ensemble in batches and concatenating the per-trajectory results
        sizes = [batch] * (n_samples // batch)
        if n_samples % batch:
            sizes.append(n_samples % batch)

        mass_parts, visits = [], None
        for size in sizes:
            out = self._sample_vectorized(size, rewards, record_visits)
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
            record_visits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Vectorized trajectory sampler: advance all ``n_samples`` walkers through the CTMC in lockstep, one wave per
        jump, instead of looping in Python.

        Each walker carries a remaining hazard budget ``H ~ Exp(1)``, resampled after every jump. The time to its
        next event in the current epoch is ``H / lambda`` (``lambda`` the exit rate); a walker whose budget outlasts
        the epoch is advanced to the boundary (accruing reward and consuming ``lambda * duration`` of hazard) and
        steps into the next epoch. This hazard-budget form handles zero-rate epochs (temporarily isolated demes)
        uniformly: ``lambda = 0`` consumes no hazard, so the walker simply waits out the epoch accruing reward.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from.
        :param record_visits: Whether to also return the per-state visit frequencies.
        :return: Array of sampled rewards of shape ``(n_samples, len(rewards))`` (and visit frequencies if requested).
        """
        n_rewards = len(rewards)
        k = self.state_space.k
        absorbing = self.state_space.absorbing
        alpha = self.state_space.alpha
        R = np.array([r._get(self.state_space) for r in rewards])  # (n_rewards, k), epoch-invariant

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
        state = np.random.choice(k, size=n_samples, p=alpha)
        t = np.zeros(n_samples)
        e = np.zeros(n_samples, dtype=int)
        H = np.random.exponential(size=n_samples)  # remaining hazard budget ~ Exp(1)
        mass = np.zeros((n_samples, n_rewards))
        states_visited = np.zeros(k) if record_visits else None
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
                    mass[ca] += R[:, state[ca]].T * dca[:, None]
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
                    mass[a[stuck]] = np.inf
                    active[a[stuck]] = False
                    keep = ~stuck
                    a, lam = a[keep], lam[keep]
                    if a.size == 0:
                        break

                dt = H[a] / lam
                mass[a] += R[:, state[a]].T * dt[:, None]
                t[a] += dt

                # sample the next state via inverse-CDF on the sparse cumulative jump distribution: one global
                # searchsorted over the band-shifted cumulative probabilities, clipped to each walker's own row
                row = e[a] * k + state[a]
                q = np.random.random(a.size) + row  # band-shifted uniform draw lands in this row's band
                pos = np.clip(np.searchsorted(cum_offsets, q, side='left'), cum_indptr[row], cum_indptr[row + 1] - 1)
                nxt = cum_neighbours[pos]
                state[a] = nxt
                if record_visits:
                    np.add.at(states_visited, nxt, 1)

                # resample the hazard budget for survivors; absorbed walkers leave the ensemble
                H[a] = np.random.exponential(size=a.size)
                active[a[absorbing[nxt]]] = False

        if record_visits:
            states_visited /= n_samples
            return mass, states_visited

        return mass

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
        Plot accumulation of (non-central) moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and the
            99th percentile.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        k = int(k)

        from ..visualization import Visualization

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = f"Moment accumulation ({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})"

        y = self.accumulate(k, end_times, rewards, center, permute)

        Visualization.plot(
            ax=ax,
            x=end_times,
            y=y,
            xlabel='t',
            ylabel='moment',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )


class _ExpmCumulativeDistributionFunction(_GridCumulativeDistributionFunction):
    """The tree-height CDF by direct matrix exponentiation: ``P(R <= t) = 1 - alpha @ prod_e exp(Q_e tau_e) @ e``,
    accumulating the per-epoch transition matrices up to ``t``. Reaches into the
    :class:`TreeHeightDistribution` for the state space, demography, epochs and reward (the exit vector)."""

    def __call__(self, t) -> 'np.ndarray | float':
        """Cumulative distribution function. Scalar or array-valued; raises for non-default rewards."""
        d = self._distribution

        # raise error if rewards are not default
        if not isinstance(d.reward, TreeHeightReward):
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        # assume scalar if not array
        if not isinstance(t, Iterable):
            return self(np.array([t]))[0]

        # check for negative values
        if np.any(t < 0):
            raise ValueError("Negative values are not allowed.")

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(t).astype(float)

        epochs = enumerate(d.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get the transition matrix for the first epoch
        d.state_space.update_epoch(epoch)

        # initialize transition matrix
        T = np.eye(d.state_space.k)
        u_prev = 0

        # initialize probabilities
        probs = np.zeros_like(t_sorted)

        # take reward vector as exit vector
        e = d.reward._get(d.state_space)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                d._check_numerical_stability(d.state_space.S, i_epoch)

                # update transition matrix with remaining time in current epoch
                T @= expm(d._dense_rate_matrix() * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                d.state_space.update_epoch(epoch)

            d._check_numerical_stability(d.state_space.S, i_epoch)

            # update transition matrix with remaining time in current epoch
            T @= expm(d._dense_rate_matrix() * (u - u_prev))

            probs[i] = 1 - d.state_space.alpha @ T @ e

            u_prev = u

        # sort probabilities back to original order (inverse of the sorting permutation)
        probs = probs[np.argsort(np.argsort(t))]

        if np.isnan(probs).any():
            d._logger.critical(
                "NaN values in CDF. This is likely due to an ill-conditioned rate matrix."
            )

        return probs


class _ExpmQuantileFunction(_GridQuantileFunction):
    """The tree-height quantile by adaptive bisection on the exact (matrix-exponential) CDF, bounded by the time of
    almost-sure absorption. Reaches into the :class:`TreeHeightDistribution` for the epoch machinery."""

    @cache
    def __call__(
            self,
            q: float,
            expansion_factor: float = 2,
            precision: float = 1e-5,
            max_iter: int = 1000
    ) -> 'np.ndarray | float':
        """Find the specified quantile of the CDF using an adaptive bisection method."""
        d = self._distribution

        if q < 0 or q > 1:
            raise ValueError("Specified quantile must be between 0 and 1.")

        if expansion_factor <= 1:
            raise ValueError("Expansion factor must be greater than 1.")

        # finite upper bound for the search: the time of almost-sure absorption (any quantile q < 1 lies below it).
        # This also guards against a demography that never absorbs — ``_get_absorption_time`` raises in that case —
        # and keeps the expansion below from doubling ``b`` to an overflow-inducing ceiling. A user-supplied end
        # time bounds the (necessarily proper) distribution instead.
        b_max = d.end_time if d.end_time is not None else d._get_absorption_time()

        # initialize bounds
        a, b = 0, 1

        T_a = np.eye(d.state_space.k)
        epoch_a, epoch_b = d.demography.get_epoch(0), d.demography.get_epoch(0)
        b, T_b, epoch_b = d._update(min(b, b_max), a, T_a, epoch_b)

        i = 0

        # expand the upper bound until its CDF reaches q (bounded by the absorption time, so it always terminates)
        while d._cum(T_b) < q and b < b_max and i < max_iter:
            b, T_b, epoch_b = d._update(min(b * expansion_factor, b_max), b, T_b, epoch_b)

            i += 1

        # use bisection method within the determined bounds
        while d._cum(T_b) - d._cum(T_a) > precision and i < max_iter:
            m, T_m, epoch_m = d._update((a + b) / 2, a, T_a, epoch_a)

            if d._cum(T_m) < q:
                a, T_a, epoch_a = m, T_m, epoch_m
            else:
                b, T_b, epoch_b = m, T_m, epoch_m

            i += 1

        # warn if maximum number of iterations reached
        if i - 1 == max_iter:
            raise RuntimeError("Maximum number of iterations reached when determining quantile.")

        return (a + b) / 2


class _ExpmDensityFunction(_GridDensityFunction):
    """The tree-height density by numerical differentiation of the exact (matrix-exponential) CDF (which is exact
    and continuous, so a central difference is accurate)."""

    def __call__(self, t: float | Sequence[float], dx: float = None) -> float | np.ndarray:
        """Density function (central difference of the CDF)."""
        d = self._distribution

        if dx is None:
            dx = d.quantile(0.99) / 1e10

        if isinstance(t, Iterable):
            t = np.array(t)

        # determine (non-negative) evaluation points
        x1 = np.max([t - dx / 2, np.zeros_like(t)], axis=0)
        x2 = x1 + dx

        return (d.cdf(x2) - d.cdf(x1)) / dx


class TreeHeightDistribution(PhaseTypeDistribution, DensityAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process that allows the computation of the
    density function. This is currently only possible with default rewards.

    The exact (matrix-exponential) cdf / pdf / quantile evaluation lives on the function objects
    (:class:`_ExpmCumulativeDistributionFunction` / :class:`_ExpmDensityFunction` / :class:`_ExpmQuantileFunction`);
    this distribution supplies the state space, demography, epoch machinery and the exit vector they reach into.
    """
    #: the exact matrix-exponential function-object flavours (selected over the inherited LST/COS ones, whose CDF is
    #: itself a de Hoog inversion -- the expm path stays robust on ill-conditioned demographies)
    _cdf_function = _ExpmCumulativeDistributionFunction
    _pdf_function = _ExpmDensityFunction
    _quantile_function = _ExpmQuantileFunction
    #: Maximum number of epochs to consider when determining time to almost sure absorption.
    max_epochs: int = 10000

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

    def _update(
            self,
            u: float,
            u_prev: float,
            T: np.ndarray,
            epoch: 'Epoch'
    ) -> Tuple[float, np.ndarray, 'Epoch']:
        """
        Update transition matrix and time.

        :param u: Time to update to.
        :param u_prev: Previous time.
        :param T: Transition matrix.
        :param epoch: Current epoch.
        :return: Updated time, transition matrix, and epoch.
        """
        self.state_space.update_epoch(epoch)

        while u > epoch.end_time:

            # update transition matrix with remaining time in current epoch
            tau = epoch.end_time - u_prev
            T = T @ expm(self._dense_rate_matrix() * tau)
            u_prev = epoch.end_time

            # fetch and update for next epoch
            epoch = self.demography.get_epoch(epoch.end_time)
            self.state_space.update_epoch(epoch)
        else:
            # update transition matrix
            T = T @ expm(self._dense_rate_matrix() * (u - u_prev))

        return u, T, epoch

    @cached_property
    def _e(self) -> np.ndarray:
        """
        Exit vector.
        """
        return self.reward._get(self.state_space)

    def _cum(self, T: np.ndarray) -> float:
        """
        Get cumulative probability for given transition matrix.

        :param T: Transition matrix.
        :return: Cumulative probability.
        """
        return float(1 - self.state_space.alpha @ T @ self._e)

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
        T = np.eye(self.state_space.k)
        epoch = self.demography.get_epoch(0)

        self._check_demography_conditioning()

        t = 2 ** int(np.log2(np.mean(list(epoch.pop_sizes.values()))))
        expansion_factor = 2

        t, T, epoch = self._update(t, 0, T, epoch)
        p = self._cum(T)

        # multiple time by expansion_factor until we reach p_absorption
        while p < self.p_absorption and i < self.max_iter:
            t, T, epoch = self._update(t * expansion_factor, t, T, epoch)
            p = self._cum(T)

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
            self._assert_absorbs(T)

        if i - 1 == self.max_iter:
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
        :param t: Values at which to evaluate the CDF. Default to 100 evenly spaced values
            between 0 and the 99th percentile.
        :return: Sorted array of sampled total rewards.
        """
        if t is None:
            t = np.linspace(0, self.tree_height.quantile(0.99), 100)

        samples = self._sample(n_samples, reward).reshape(n_samples)

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
        :param t: Values at which to evaluate the CDF. Default to 100 evenly spaced values
            between 0 and the 99th percentile.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if t is None:
            t = np.linspace(0, self.tree_height.quantile(0.99), 100)

        y = self._empirical_cdf(n_samples, reward, t)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=y,
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )


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

