"""Phase-type distribution (moment engine) and the tree-height distribution."""

import logging
from ..caching import cached_property, cache
from typing import Tuple, Collection, Iterable, Sequence, Union, TYPE_CHECKING
import numpy as np
from tqdm import tqdm
from ..demography import Demography, Epoch
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..rewards import Reward, TreeHeightReward, TotalBranchLengthReward
from ..settings import Settings
from ..spectrum import SFS
from ..state_space import LineageCountingStateSpace, StateSpace

from .base import CallableDistributionFunctions, DensityAwareDistribution, MarginalDemeDistributions, \
    MarginalLocusDistributions, MomentAwareDistribution
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
    ):
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
                    float(np.interp(q_end, d.cdf_curve(grid := np.linspace(0, d._range(), 256)), grid))
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
                xk, y = adaptive_grid(d._cdf if kind == 'cdf' else d._pdf, 0.0, float(x[-1]), max_points=n_points)
            else:
                xk = x
                if kind == 'cdf':
                    y = d._cdf(x) if exact else d.cdf_curve(x)
                elif kind == 'pdf':
                    y = d._pdf(x) if exact else d.pdf_curve(x)
                elif exact:
                    # quantile function via the per-point de Hoog bisection
                    y = np.array([d._quantile(float(p)) for p in x])
                else:
                    # quantile function: invert the (fast) COS CDF curve by interpolation rather than a per-point
                    # bisection (which would re-run the de Hoog inversion at every probability and bin)
                    xx = np.linspace(0, d._range(), 512)
                    y = np.interp(x, d.cdf_curve(xx), xx)

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

    def _sample(
            self,
            n_samples: int,
            rewards: Sequence[Reward] = None,
            record_visits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate samples from the mean reward distribution by simulating trajectories.

        :param n_samples: Number of trajectories to simulate.
        :param rewards: Rewards to sample from. Default is the tree height reward.
        :param record_visits: Whether to record which states were visited during the sampling.
        :return: Array of sampled rewards of size (n_samples, len(rewards)),
                 and optionally an array of probabilities of visiting each state.
        """
        if rewards is None:
            rewards = [self.reward]

        n_rewards = len(rewards)
        samples = np.zeros((n_samples, n_rewards))
        absorbing = self.state_space.absorbing
        alpha = self.state_space.alpha
        states_visited = np.zeros_like(alpha)
        R = np.array([r._get(self.state_space) for r in rewards])

        # iterate over samples
        for i in tqdm(range(n_samples), disable=not Settings.use_pbar):
            mass = np.zeros(n_rewards)
            t = 0
            rate = 0
            state = np.random.choice(len(alpha), p=alpha)
            epochs = self.demography.epochs

            try:
                # find first non-zero rate epoch
                while rate == 0:
                    epoch = next(epochs)
                    self.state_space.update_epoch(epoch)
                    rate = -self.state_space.S[state, state]

                # sample next time step
                dt = np.random.exponential(1 / rate)

                # iterate over transitions
                while True:

                    # iterate over epochs
                    while t + dt >= epoch.end_time:
                        # reward until epoch boundary
                        mass += R[:, state] * (epoch.end_time - t)
                        dt -= (epoch.end_time - t)
                        t = epoch.end_time

                        # advance epoch
                        epoch = next(epochs)
                        self.state_space.update_epoch(epoch)

                        new_rate = -self.state_space.S[state, state]
                        if new_rate == 0:
                            t = epoch.end_time
                            continue

                        # rescale remaining time
                        dt *= rate / new_rate
                        rate = new_rate

                    # step completes in current epoch
                    mass += R[:, state] * dt
                    t += dt

                    # sample next state
                    probs = self.state_space.S[state].copy()
                    probs[state] = 0
                    state = np.random.choice(len(probs), p=probs / rate)

                    states_visited[state] += 1

                    if absorbing[state]:
                        raise StopIteration

                    rate = -self.state_space.S[state, state]

                    # if rate is zero, we skip to the next epoch
                    if rate == 0:
                        t = epoch.end_time
                        continue

                    # sample next time step
                    dt = np.random.exponential(1 / rate)

            except StopIteration:
                pass

            samples[i] = mass

        # normalize states visited
        states_visited /= n_samples

        return (samples, states_visited) if record_visits else samples

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


class TreeHeightDistribution(PhaseTypeDistribution, DensityAwareDistribution):
    """
    Phase-type distribution for a piecewise time-homogeneous process that allows the computation of the
    density function. This is currently only possible with default rewards.
    """
    #: Maximum number of epochs to consider when determining time to almost sure absorption.
    max_epochs: int = 10000

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
    ):
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

    def _cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Value or values to evaluate the CDF at.
        :return: Cumulative probability
        :raises NotImplementedError: if rewards are not default
        """
        # raise error if rewards are not default
        if not isinstance(self.reward, TreeHeightReward):
            raise NotImplementedError("PDF not implemented for non-default rewards.")

        # assume scalar if not array
        if not isinstance(t, Iterable):
            return self._cdf(np.array([t]))[0]

        # check for negative values
        if np.any(t < 0):
            raise ValueError("Negative values are not allowed.")

        # sort array in ascending order but keep track of original indices
        t_sorted: Collection[float] = np.sort(t).astype(float)

        epochs = enumerate(self.demography.epochs)
        i_epoch, epoch = next(epochs)

        # get the transition matrix for the first epoch
        self.state_space.update_epoch(epoch)

        # initialize transition matrix
        T = np.eye(self.state_space.k)
        u_prev = 0

        # initialize probabilities
        probs = np.zeros_like(t_sorted)

        # take reward vector as exit vector
        e = self.reward._get(self.state_space)

        # iterate through sorted values
        for i, u in enumerate(t_sorted):

            # iterate over epochs between u_prev and u
            while u > epoch.end_time:
                self._check_numerical_stability(self.state_space.S, i_epoch)

                # update transition matrix with remaining time in current epoch
                T @= expm(self._dense_rate_matrix() * (epoch.end_time - u_prev))

                # fetch and update for next epoch
                u_prev = epoch.end_time
                i_epoch, epoch = next(epochs)
                self.state_space.update_epoch(epoch)

            self._check_numerical_stability(self.state_space.S, i_epoch)

            # update transition matrix with remaining time in current epoch
            T @= expm(self._dense_rate_matrix() * (u - u_prev))

            probs[i] = 1 - self.state_space.alpha @ T @ e

            u_prev = u

        # sort probabilities back to original order
        probs = probs[np.argsort(t)]

        if np.isnan(probs).any():
            self._logger.critical(
                "NaN values in CDF. This is likely due to an ill-conditioned rate matrix."
            )

        return probs

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

    @cache
    def _quantile(
            self,
            q: float,
            expansion_factor: float = 2,
            precision: float = 1e-5,
            max_iter: int = 1000
    ):
        """
        Find the specified quantile of a CDF using an adaptive bisection method.

        :param q: The desired quantile (between 0 and 1).
        :param expansion_factor: Factor by which to multiply the upper bound that does not yet contain the quantile.
        :param precision: Precision for quantile, i.e. ``F(b) - F(a) < precision``.
        :param max_iter: Maximum number of iterations.
        :return: The quantile.
        """
        if q < 0 or q > 1:
            raise ValueError("Specified quantile must be between 0 and 1.")

        if expansion_factor <= 1:
            raise ValueError("Expansion factor must be greater than 1.")

        # finite upper bound for the search: the time of almost-sure absorption (any quantile q < 1 lies below it).
        # This also guards against a demography that never absorbs — ``_get_absorption_time`` raises in that case —
        # and keeps the expansion below from doubling ``b`` to an overflow-inducing ceiling. A user-supplied end
        # time bounds the (necessarily proper) distribution instead.
        b_max = self.end_time if self.end_time is not None else self._get_absorption_time()

        # initialize bounds
        a, b = 0, 1

        T_a = np.eye(self.state_space.k)
        epoch_a, epoch_b = self.demography.get_epoch(0), self.demography.get_epoch(0)
        b, T_b, epoch_b = self._update(min(b, b_max), a, T_a, epoch_b)

        i = 0

        # expand the upper bound until its CDF reaches q (bounded by the absorption time, so it always terminates)
        while self._cum(T_b) < q and b < b_max and i < max_iter:
            b, T_b, epoch_b = self._update(min(b * expansion_factor, b_max), b, T_b, epoch_b)

            i += 1

        # use bisection method within the determined bounds
        while self._cum(T_b) - self._cum(T_a) > precision and i < max_iter:
            m, T_m, epoch_m = self._update((a + b) / 2, a, T_a, epoch_a)

            if self._cum(T_m) < q:
                a, T_a, epoch_a = m, T_m, epoch_m
            else:
                b, T_b, epoch_b = m, T_m, epoch_m

            i += 1

        # warn if maximum number of iterations reached
        if i - 1 == max_iter:
            raise RuntimeError("Maximum number of iterations reached when determining quantile.")

        return (a + b) / 2

    def _pdf(self, t: float | Sequence[float], dx: float = None) -> float | np.ndarray:
        """
        Density function. We use numerical differentiation of the CDF to calculate the density. This provides good
        results as the CDF is exact and continuous.

        :param t: Value or values to evaluate the density function at.
        :param dx: Step size for numerical differentiation. By default, the 99th percentile divided by 1e10.
        :return: Density
        """
        if dx is None:
            dx = self._quantile(0.99) / 1e10

        if isinstance(t, Iterable):
            t = np.array(t)

        # determine (non-negative) evaluation points
        x1 = np.max([t - dx / 2, np.zeros_like(t)], axis=0)
        x2 = x1 + dx

        return (self._cdf(x2) - self._cdf(x1)) / dx

    # the tree height has an exact (expm-based) CDF that stays robust on ill-conditioned demographies, so plot via
    # the exact path (DensityAwareDistribution) rather than the COS/LST reward-curve path used for the other
    # phase-type distributions (whose CDF is itself the de Hoog inversion)
    _plot_cdf = DensityAwareDistribution._plot_cdf
    _plot_pdf = DensityAwareDistribution._plot_pdf
    _plot_quantile = DensityAwareDistribution._plot_quantile

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
            demography: Demography = None
    ):
        """
        Initialize the distribution.

        :param state_space: The state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        """
        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=TotalBranchLengthReward()
        )

