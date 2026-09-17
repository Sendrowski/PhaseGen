"""Site-frequency-spectrum distributions (SFS, folded, joint, two-locus)."""

import heapq
import itertools
import logging
from abc import ABC, abstractmethod
from ..caching import cached_property, cache
from typing import List, Tuple, Iterable, Iterator, Optional, Sequence, Set, Union, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp
from ..demography import Demography
from ..expm import Backend
from ..rewards import Reward, TreeHeightReward, UnfoldedSFSReward, UnitReward, CombinedReward, FoldedSFSReward, SFSReward, JointSFSReward, TwoLocusSFSReward
from ..settings import Settings
from ..spectrum import SFS, TwoSFS, JointSFS, TwoLocusSFS
from ..state_space import BlockCountingStateSpace, StateSpace, JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace
from ..utils import multiset_permutations

from ._common import _make_hashable
from .base import MarginalDensity, MarginalCDF, MarginalQuantileFunction
from .phase_type import PhaseTypeDistribution, TreeHeightDistribution

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from ..visualization import _CurveData
    from .reward import JointRewardDistribution, RewardDistribution
    from .empirical import (
        EmpiricalPhaseTypeSFSDistribution,
        EmpiricalJointSFSDistribution,
        EmpiricalTwoLocusSFSDistribution,
    )

expm = Backend.expm
logger = logging.getLogger('phasegen')


class _SFSAggregateFunction:
    """Per-bin SFS function, looping ``SFSDistribution._bin_distribution`` over the polymorphic bins."""

    def __call__(self, t) -> 'SFS | np.ndarray':
        """
        Evaluate the function of every polymorphic bin, each that of the bin's
        :class:`~phasegen.distributions.RewardDistribution` under the spectrum's reward.

        :param t: A point or an array of points, or probability levels for a quantile function.
        :return: For a scalar ``t``, a spectrum with one value per bin and zeros at the monomorphic bins. For an
            array, an array of shape ``(len(t), n + 1)``.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        d = self._distribution
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros((t_arr.size, d.lineage_config.n + 1))
        for i in d._get_indices():
            out[:, i] = getattr(d._bin_distribution(i), self.kind)(t_arr)
        return SFS(out[0]) if np.ndim(t) == 0 else out

    def plot(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the function of every SFS bin at once, one curve per bin.

        :param ax: Axes to plot on.
        :param x: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.
        :param bins: The bins (frequency classes) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(x=x, bins=bins, n_points=n_points), file=file,
                                         show=show, clear=clear, label=label, title=title, **kwargs)


class SFSDensity(_SFSAggregateFunction, MarginalDensity):
    """Per-bin densities of the SFS, one per frequency class, each that of the bin's
    :class:`~phasegen.distributions.RewardDistribution`."""


class SFSCDF(_SFSAggregateFunction, MarginalCDF):
    """Per-bin CDFs of the SFS, one per frequency class, each that of the bin's
    :class:`~phasegen.distributions.RewardDistribution`."""


class SFSQuantileFunction(_SFSAggregateFunction, MarginalQuantileFunction):
    """Per-bin quantile functions of the SFS, one per frequency class, each that of the bin's
    :class:`~phasegen.distributions.RewardDistribution`."""

    def plot(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the quantile function of every SFS bin at once (bin branch length versus probability ``q``).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate at. By default, :attr:`~phasegen.settings.Settings.plot_n_grid` points
            from ``1 - Settings.plot_endpoint_quantile`` to :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param bins: The bins (frequency classes) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, bins=bins, n_points=n_points), file=file,
                                         show=show, clear=clear, label=label, title=title, **kwargs)


class SFSDistribution(PhaseTypeDistribution, ABC):
    r"""
    Base class for site-frequency spectrum distributions. Bin :math:`i` accumulates the total branch length
    :math:`L_i` subtending :math:`i` of the :math:`n` samples. The spectrum mean is the vector of expected bin branch
    lengths :math:`\mathbb{E}[L_i]`, and :attr:`cov` its within-tree covariance :math:`\operatorname{Cov}[L_i, L_j]`.

    The evaluation of spectrum-wide moments is described in
    :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
    """
    # the spectrum's pdf/cdf/quantile are per-bin (one curve per frequency class) -> SFS-specific flavours
    _pdf_function = SFSDensity
    _cdf_function = SFSCDF
    _quantile_function = SFSQuantileFunction

    @property
    def pdf(self) -> SFSDensity:
        """Per-bin SFS probability density functions (one per frequency class): callable (``pdf(t)``) and plottable."""
        return super().pdf

    @property
    def cdf(self) -> SFSCDF:
        """Per-bin SFS cumulative distribution functions (one per frequency class): callable and plottable."""
        return super().cdf

    @property
    def quantile(self) -> SFSQuantileFunction:
        """Per-bin SFS quantile functions (one per frequency class): callable (``quantile(q)``) and plottable."""
        return super().quantile

    def __init__(
            self,
            state_space: BlockCountingStateSpace,
            tree_height: TreeHeightDistribution,
            demography: Demography,
            reward: Reward = None
    ) -> None:
        """
        Initialize the distribution.

        :param state_space: Block-counting state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward to multiply the SFS reward with. By default, the unit reward is used, which
            has no effect.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward
        )

        #: Probability mass yielded by the most recently started configuration iterator, see
        #: :meth:`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.
        self.generated_mass = 0

    @abstractmethod
    def _get_sfs_reward(self, i: int) -> SFSReward:
        """
        Get the reward for the ith site-frequency count.

        :param i: The ith site-frequency count.
        :return: The reward.
        """
        pass

    @abstractmethod
    def _get_indices(self) -> np.ndarray:
        """
        Get the indices for the site-frequency spectrum.

        :return: The indices.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_configs(n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Get all possible mutational configurations for a given number of mutations.

        :param n: The number of lineages.
        :param k: The number of mutations.
        :return: An iterator over all possible mutational configurations.
        """
        pass

    def _bin_distribution(self, i: int) -> 'RewardDistribution':
        """The reward distribution of SFS bin ``i`` under this spectrum's reward, cached so the expensive cosine / LST
        fit behind its cdf / pdf / quantile is built once and reused across repeated calls and across the three
        curves, rather than rebuilt on every ``sfs.cdf(t)``. Honors :attr:`Settings.cache`."""
        i = int(i)

        if not Settings.cache:
            return self.distribution(reward=CombinedReward([self.reward, self._get_sfs_reward(i)]))

        cache = self.__dict__.setdefault('_bin_distributions', {})
        if i not in cache:
            cache[i] = self.distribution(reward=CombinedReward([self.reward, self._get_sfs_reward(i)]))
        return cache[i]

    @_make_hashable
    @cache
    def moment(
            self,
            k: int,
            rewards: Sequence[SFSReward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> SFS:
        r"""
        The :math:`k`-th moment of every polymorphic bin of the site-frequency spectrum, central by default, as
        described in :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param rewards: Sequence of :math:`k` rewards, each multiplied with the bin reward. By default, the reward of
            the distribution for each factor.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :param end_time: The end time :math:`t_\mathrm{end}`. By default, the end time of the distribution, or
            absorption.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: A site-frequency spectrum of :math:`k`-th moments.
        """
        if rewards is None:
            rewards = (self.reward,) * k

        effective_start = self.tree_height.start_time if start_time is None else start_time

        # batched mean: every bin's mean is ``occupation . r_bin`` with the same occupation-time vector, so the whole
        # spectrum is one contraction instead of a per-bin solve. This is the closed form's spectrum path (it shares
        # the transient solve across bins); only for the plain mean (k=1, default reward, no custom end time) and when
        # flattening does not apply (flattening reduces the state space and wins). A non-zero start time is handled by
        # subtracting the occupation up to it (occupation is additive in time). Other cases fall through to the
        # per-bin path.
        if (
                Settings.closed_form_last_epoch and
                not self._flattening_applies(k) and
                k == 1 and
                end_time is None and
                self.tree_height.end_time is None and
                rewards == (self.reward,)
        ):
            occupation = self._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
                if effective_start > 0:
                    m = m - self._occupation_times(cap=effective_start)[0]
                base = np.asarray(self.reward._get(self.state_space), dtype=float)
                R = np.column_stack([
                    (base * np.asarray(self._get_sfs_reward(i)._get(self.state_space), dtype=float))[idx_t]
                    for i in self._get_indices()
                ])
                moments = m @ R
                return SFS([0] + list(moments) + [0] * (self.lineage_config.n - len(moments)))

        # moment of each SFS bin (serial; performance-critical paths use the batched closed form above)
        moments = np.array([
            self._moment(k, i, rewards, start_time, end_time, center, permute)
            for i in self._get_indices()
        ])

        return SFS([0] + list(moments) + [0] * (self.lineage_config.n - len(moments)))

    def _moment(
            self,
            k: int,
            i: int,
            rewards: Sequence[SFSReward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> float:
        """
        Get the kth moment for the ith site-frequency count.

        :param k: The order of the moment
        :param i: The ith site-frequency count
        :param rewards: Sequence of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: The kth SFS (cross)-moment at the ith site-frequency count
        """
        return PhaseTypeDistribution.moment(
            self,
            k=k,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards]),
            start_time=start_time,
            end_time=end_time,
            center=center,
            permute=permute
        )

    def sample(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> np.ndarray:
        r"""
        Draw samples of the site-frequency spectrum, one trajectory per sample, simulated as in
        :meth:`PhaseTypeDistribution.sample() <phasegen.distributions.PhaseTypeDistribution.sample>` with the notation
        of :class:`~phasegen.distributions.PhaseTypeDistribution`. Entry :math:`j` of a sample is the branch length
        :math:`L_j = \int_{t_\mathrm{start}}^{t_\mathrm{end}} r(X_u)\, a_j(X_u)\, \mathrm{d}u`, where :math:`a_j(x)` is
        the number of lineages of state
        :math:`x` subtending :math:`j` of the :math:`n` lineages, or :math:`a_j(x) + a_{n-j}(x)` for :math:`j < n - j`
        in the folded spectrum. All entries of a sample share one trajectory, and entries outside the polymorphic
        classes are zero.

        :param n_samples: Number of spectra.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: Array of shape ``(n_samples, n + 1)``, whose column means estimate :attr:`mean`.
        """
        indices = self._get_indices()
        rewards = [CombinedReward([self.reward, self._get_sfs_reward(i)]) for i in indices]
        sampled = self._sample(n_samples, rewards=rewards, rng=np.random.default_rng(seed))

        out = np.zeros((n_samples, self.lineage_config.n + 1))
        out[:, 1:1 + len(indices)] = sampled

        return out

    def to_empirical(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> 'EmpiricalPhaseTypeSFSDistribution':
        """
        Build an empirical spectrum from ``n_samples`` trajectories, with the per-deme breakdown of
        :meth:`PhaseTypeDistribution.to_empirical() <phasegen.distributions.PhaseTypeDistribution.to_empirical>`
        applied to every frequency class of
        :meth:`UnfoldedSFSDistribution.sample() <phasegen.distributions.UnfoldedSFSDistribution.sample>`. The
        spectrum carries branch lengths only, so it provides no mutational configurations.

        :param n_samples: Number of trajectories.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: The empirical spectrum.
        :raises NotImplementedError: For more than one locus.
        """
        from .empirical import EmpiricalPhaseTypeSFSDistribution

        if self.locus_config.n != 1:
            raise NotImplementedError("Sampled SFS is only available for single-locus scenarios.")

        pops = self.lineage_config.pop_names
        n = self.lineage_config.n
        indices = self._get_indices()

        # stacked rewards over (deme, polymorphic bin); one sampling pass yields the full per-deme spectrum
        rewards = [CombinedReward([self.demes[pop].reward, self._get_sfs_reward(i)]) for pop in pops for i in indices]
        sampled = self._sample(n_samples, rewards=rewards, rng=np.random.default_rng(seed)).reshape(n_samples, len(pops), len(indices))

        # (loci=1, demes, samples, n + 1); the polymorphic bins scatter into their index positions
        branch_lengths = np.zeros((1, len(pops), n_samples, n + 1))
        for bi, i in enumerate(indices):
            branch_lengths[0, :, :, i] = sampled[:, :, bi].T

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=branch_lengths,
            mutations=None,
            pops=pops,
            sfs_dist=type(self)
        )

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        r"""
        The :math:`k`-th moment of every bin of the site-frequency spectrum accumulated up to each end time
        :math:`t_\mathrm{end}` in ``end_times``, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t_\mathrm{end}` at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the reward of the distribution for each factor.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: Array of moments accumulated at the specified times, one for each site-frequency count.
        """
        k = int(k)
        indices = self._get_indices()
        end_times = np.array(list(end_times))

        accumulation = self._accumulate_batched(k, indices, end_times, rewards)
        if accumulation is None:
            accumulation = np.array([
                self.get_accumulation(k, i, end_times, rewards, center, permute) for i in indices
            ])

        # pad with zeros
        return np.concatenate([
            np.zeros((1, len(end_times))),
            accumulation,
            np.zeros((self.lineage_config.n - len(indices), len(end_times)))
        ])

    def _accumulate_batched(self, k, indices, end_times, rewards) -> 'np.ndarray | None':
        """Batched mean accumulation (``k == 1``, default reward) contracting ``_mean_occupation_grid`` with the
        stacked bin rewards. Returns ``None`` when not applicable, and the caller evaluates per bin."""
        if k != 1 or rewards is not None or self._flattening_applies(1):
            return None

        m_grid = self._mean_occupation_grid(end_times)  # (len(t), n_states)
        ss = self.state_space
        R = np.column_stack([
            np.asarray(CombinedReward([self.reward, self._get_sfs_reward(i)])._get(ss), dtype=float)
            for i in indices
        ])
        self._logger.debug("sfs accumulate (k=1): batched (shared occupation grid over %d bins)", len(indices))
        return (m_grid @ R).T  # (n_bins, len(t))

    def _plot_accumulation_data(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            rewards: Sequence[Reward] = None,
            center: bool = True,
            permute: bool = True
    ) -> '_CurveData':
        """
        The accumulation of the SFS moments over time that :meth:`plot_accumulation` draws, one curve per polymorphic
        bin.

        :param k: The order of the moment.
        :param end_times: Times at which to evaluate the moment. By default, :attr:`Settings.plot_n_grid` points up to
            the :attr:`Settings.plot_endpoint_quantile` quantile of the tree height.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :return: The curves, labelled by bin.
        """
        from ..visualization import _CurveData

        k = int(k)
        end_times = self._default_end_times() if end_times is None else np.asarray(list(end_times), dtype=float)
        rewards = (self.reward,) * k if rewards is None else rewards
        indices = self._get_indices()

        return _CurveData(
            x=end_times,
            y=self.accumulate(k, end_times, rewards, center, permute)[1:1 + len(indices)],
            labels=[str(i) for i in indices],
            xlabel='t',
            ylabel='moment',
            title=f"SFS Moment accumulation ({self._reward_names(rewards)})",
            legend_title='bin'
        )

    def _bin_items(self, bins: Sequence[int] | None) -> List[Tuple[int, 'RewardDistribution']]:
        """
        The requested bins with their distributions.

        :param bins: The bins, ``None`` for all polymorphic bins.
        :return: Each bin and its distribution.
        """
        indices = self._get_indices() if bins is None else np.atleast_1d(bins)

        return [(int(i), self._bin_distribution(i)) for i in indices]

    def bin(self, i: int) -> 'RewardDistribution':
        r"""The distribution of the branch length :math:`L_i` of bin ``i``, the total length of the branches subtending
        :math:`i` samples, as a :class:`~phasegen.distributions.RewardDistribution`, for example
        ``sfs.bin(2).quantile(0.9)``. The bin reward is combined with the reward of this spectrum, so the bin of a
        marginal view such as ``sfs.demes['pop_0']`` is restricted like its moments. The distribution is cached per
        bin and is the one behind the per-bin ``cdf``, ``pdf`` and ``quantile`` of this spectrum.

        :param i: The frequency class.
        :return: The distribution of :math:`L_i`.
        """
        d = self._bin_distribution(i)
        d.label = f"SFS bin {int(i)}"
        return d

    def joint_distribution(self, i: int, j: int) -> 'JointRewardDistribution':
        r"""
        Joint distribution of the branch lengths :math:`L_i` and :math:`L_j` of frequency classes :math:`i` and
        :math:`j` within one genealogy, as a :class:`~phasegen.distributions.JointRewardDistribution`.

        :param i: The first frequency class.
        :param j: The second frequency class.
        :return: The joint distribution of :math:`(L_i, L_j)`.
        """
        jd = super().joint_distribution(self._get_sfs_reward(i), self._get_sfs_reward(j))
        jd.label = f"SFS bins ({i}, {j})"
        return jd

    def _plot_data_cdf(self, x: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The CDF curve of each SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param x: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param bins: The bins (frequency classes) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by bin.
        """
        return self._reward_curves('cdf', self._bin_items(bins), x, n_points, 'SFS bin CDFs', 'bin')

    def _plot_data_pdf(self, x: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The density curve of each SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param x: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param bins: The bins (frequency classes) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by bin.
        """
        return self._reward_curves('pdf', self._bin_items(bins), x, n_points, 'SFS bin PDFs', 'bin')

    def _plot_data_quantile(
            self,
            q: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The quantile curve of each SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param q: Probabilities to evaluate at. By default, an evenly spaced grid in ``(0, 1)``.
        :param bins: The bins (frequency classes) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by bin.
        """
        return self._reward_curves('quantile', self._bin_items(bins), q, n_points, 'SFS bin quantile functions', 'bin')

    def get_accumulation(
            self,
            k: int,
            i: int,
            end_times: Iterable[float] | float,
            rewards: Sequence[SFSReward] = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray | float:
        """
        Get accumulation of moments for the ith site-frequency count.

        :param k: The order of the moment
        :param i: The ith site-frequency count.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Sequence of k rewards.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :return: The kth SFS (cross)-moment accumulations at the ith site-frequency count
        """
        if rewards is None:
            rewards = [self.reward] * k

        return super().accumulate(
            k=k,
            end_times=end_times,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards]),
            center=center,
            permute=permute
        )

    @cached_property
    def _cov_batched(self) -> Optional[TwoSFS]:
        """
        Batched 2-SFS covariance, contracting ``_two_point_occupation`` with the stacked bin rewards as in
        ``PhaseTypeDistribution.moment``.

        :return: The covariance, or ``None`` when not applicable, and the caller evaluates per pair.
        """
        if not Settings.closed_form_last_epoch:
            return None

        two_point = self._two_point_occupation()
        if two_point is None:
            return None

        K, idx_t = two_point
        ss = self.state_space
        base = np.asarray(self.reward._get(ss), dtype=float)
        indices = self._get_indices()
        R = np.column_stack([
            (base * np.asarray(self._get_sfs_reward(i)._get(ss), dtype=float))[idx_t] for i in indices
        ])

        sfs_matrix = R.T @ K @ R                       # R^T K R (one ordering)
        self._logger.debug("sfs.cov: centering with the outer product of bin means")
        mean = np.asarray(self.mean.data)[indices]
        cov = (sfs_matrix + sfs_matrix.T) - np.outer(mean, mean)

        out = np.zeros((self.lineage_config.n + 1, self.lineage_config.n + 1))
        for a, ia in enumerate(indices):
            out[ia, indices] = cov[a]
        return TwoSFS(out)

    @cached_property
    def cov(self) -> TwoSFS:
        """
        Covariance matrix across site-frequency counts, evaluated as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
        """
        batched = self._cov_batched
        if batched is not None:
            self._logger.debug("sfs.cov: batched (shared two-point occupation)")
            return batched

        # create list of arguments for each combination of i, j
        indices = [(i, j) for i in self._get_indices() for j in self._get_indices()]

        self._logger.debug("sfs.cov: per-pair matrix exponential over %d bin pairs", len(indices))

        # cross-moment of each bin pair (serial)
        sfs_results = [
            PhaseTypeDistribution.moment(self, k=2, permute=False, center=False, rewards=(
                CombinedReward([self.reward, self._get_sfs_reward(i)]),
                CombinedReward([self.reward, self._get_sfs_reward(j)])
            ))
            for i, j in indices
        ]

        # re-structure the results to a matrix form
        sfs = np.zeros((self.lineage_config.n + 1, self.lineage_config.n + 1))
        for ((i, j), result) in zip(indices, sfs_results):
            sfs[i, j] = result

        # get matrix of marginal moments
        m2 = np.outer(self.mean.data, self.mean.data)

        # calculate covariances
        cov = (sfs + sfs.T) / 2 - m2

        return TwoSFS(cov)

    @cached_property
    def var(self) -> SFS:
        """
        Variance across site-frequency counts, the diagonal of :attr:`cov` where the spectrum-wide evaluation of
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>` applies, and the
        second central moment of each bin otherwise.
        """
        batched = self._cov_batched
        if batched is not None:
            return SFS(np.diag(np.asarray(batched.data)))

        return self.moment(k=2, center=True)

    def get_cov(self, i: int, j: int) -> float:
        """
        Get the covariance between the ith and jth site-frequency.

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: covariance
        """
        if i in (0, self.lineage_config.n) or j in (0, self.lineage_config.n):
            return 0

        return super().moment(
            k=2,
            rewards=(
                CombinedReward([self.reward, self._get_sfs_reward(i)]),
                CombinedReward([self.reward, self._get_sfs_reward(j)])
            ),
            center=True
        )

    @cached_property
    def corr(self) -> TwoSFS:
        """
        Correlation matrix across site-frequency counts.
        """
        # get standard deviations
        std = np.sqrt(self.var.data)

        # monomorphic bins have zero variance; the resulting NaNs from dividing by a zero std are expected and
        # replaced with zeros below, so silence the benign divide warning at the source.
        with np.errstate(divide='ignore', invalid='ignore'):
            sfs = TwoSFS(self.cov.data / np.outer(std, std))

        # replace NaNs with zeros
        sfs.data[np.isnan(sfs.data)] = 0

        return sfs

    def get_corr(self, i: int, j: int) -> float:
        """
        Get the correlation coefficient between the ith and jth site-frequency.

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: Correlation coefficient
        """
        if i in (0, self.lineage_config.n) or j in (0, self.lineage_config.n):
            return 0

        return self.get_cov(i, j) / (np.sqrt(self.get_cov(i, i)) * np.sqrt(self.get_cov(j, j)))

    @cache
    def _get_P(self, n: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Single-epoch matrices :math:`\mathbf{G}_j` and :math:`\mathbf{g}` of
        :meth:`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`,
        by one dense inverse.

        :param n: The number of frequency classes :math:`J`.
        :param theta: The mutation rate :math:`\theta`.
        :return: The stacked matrices :math:`\mathbf{G}_j` and the vector :math:`\mathbf{g}`.
        """
        # get non-absorbing states
        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)

        e = self.state_space.e[non_absorbing]
        R = np.array([self._get_sfs_reward(i)._get(self.state_space) for i in range(1, n + 1)])[:, non_absorbing]
        r_total = R.T @ np.ones(n)

        S = self.state_space.S[non_absorbing, :][:, non_absorbing]
        I = np.eye(S.shape[0])

        P_total = np.linalg.inv(I - np.diag(1 / r_total) / theta @ S)
        p_total = (I - P_total) @ e
        P = np.array([P_total @ np.diag(R[i] / r_total) for i in range(n)])

        return P, p_total

    def _assert_no_window(self) -> None:
        """Guard the mutational-configuration path against a bounded accumulation window. The configuration
        probabilities are computed over the full to-absorption state-space rate matrix and take no ``start_time`` /
        ``end_time``, so on a windowed host they would silently return the to-absorption result regardless of the
        window. Fail loudly rather than return a value that ignores the configured window."""
        start = getattr(self.tree_height, 'start_time', 0) or 0
        end = getattr(self.tree_height, 'end_time', None)
        if start > 0 or end is not None:
            raise NotImplementedError(
                "get_mutation_config / get_mutation_configs are not implemented for a bounded accumulation window "
                f"(start_time={start}, end_time={end}): the mutational-configuration probabilities are computed over "
                "the full to-absorption state space and ignore start_time / end_time, so a windowed result would be "
                "the to-absorption one regardless. Use start_time=0 and no end_time."
            )

    def get_mutation_config(self, config: Sequence[int], theta: float) -> float:
        r"""
        Probability of a mutational configuration of a single locus under the infinite-sites model, with the notation
        of :class:`~phasegen.distributions.PhaseTypeDistribution`.

        A configuration :math:`\mathbf{m} = (m_1, \dots, m_J) \in \mathbb{N}_0^J` counts the segregating sites in each
        of the :math:`J` frequency classes. For the unfolded spectrum :math:`J = n - 1`, and class :math:`j` holds the
        mutations carried by :math:`j` of the :math:`n` sampled lineages. For the folded spectrum
        :math:`J = \lfloor n/2 \rfloor`, and class :math:`j` holds those carried by :math:`j` or :math:`n - j`
        lineages. On the block-counting state space, :math:`r_j(x) \in \mathbb{N}_0` is the number of blocks of state
        :math:`x` in class :math:`j`. The column vectors :math:`\mathbf{r}_j` and
        :math:`\bar{\mathbf{r}} = \sum_j \mathbf{r}_j` collect these rewards over the transient states, where
        :math:`\bar{\mathbf{r}}` is positive. The total branch length of class :math:`j` is
        :math:`\ell_j = \int_0^\tau r_j(X_u)\, \mathrm{d}u`. Let :math:`\theta \ge 0` be the mutation rate per unit of
        branch length, in the time units of the demography. Given the genealogy, the class counts :math:`Y_j` are
        independent Poisson variables with means :math:`\theta \ell_j`, so that the configuration
        :math:`\mathbf{Y} = (Y_1, \dots, Y_J)` has the distribution

        .. math::
            \mathbb{P}(\mathbf{Y} = \mathbf{m})
            = \mathbb{E}\left[ \prod_{j=1}^{J} e^{-\theta \ell_j} \frac{(\theta \ell_j)^{m_j}}{m_j!} \right].

        For a single epoch (:math:`M = 1`), mutations occur in state :math:`x` at rate :math:`\theta \bar{r}(x)` and
        leave the state unchanged. With :math:`\mathbf{I}` the :math:`n_T \times n_T` identity matrix, define

        .. math::
            \mathbf{G} = \left( \mathbf{I} - \theta^{-1} \operatorname{diag}(\bar{\mathbf{r}})^{-1}
            \mathbf{T}_1 \right)^{-1},
            \qquad
            \mathbf{G}_j = \mathbf{G} \operatorname{diag}(\mathbf{r}_j) \operatorname{diag}(\bar{\mathbf{r}})^{-1},
            \qquad
            \mathbf{g} = (\mathbf{I} - \mathbf{G})\, \mathbf{e}_T.

        For the process in state :math:`x`, the entry :math:`G_{xy}` is the probability that the next mutation occurs
        in state :math:`y`, the entry :math:`(\mathbf{G}_j)_{xy}` the probability that it also falls into class
        :math:`j`, and :math:`g_x` the probability that absorption precedes any further mutation. Let
        :math:`|\mathbf{m}| = \sum_j m_j` and let :math:`\mathcal{O}(\mathbf{m})` be the set of sequences
        :math:`(\sigma_1, \dots, \sigma_{|\mathbf{m}|}) \in \{1, \dots, J\}^{|\mathbf{m}|}` in which each class
        :math:`j` occurs :math:`m_j` times. Then

        .. math::
            \mathbb{P}(\mathbf{Y} = \mathbf{m})
            = \boldsymbol{\alpha}_T \sum_{\sigma \in \mathcal{O}(\mathbf{m})}
              \mathbf{G}_{\sigma_1} \cdots \mathbf{G}_{\sigma_{|\mathbf{m}|}}\, \mathbf{g},

        the coefficient of :math:`z_1^{m_1} \cdots z_J^{m_J}` in the probability generating function
        :math:`\boldsymbol{\alpha}_T (\mathbf{I} - \sum_j z_j \mathbf{G}_j)^{-1} \mathbf{g}` of Hobolth et al. (2025),
        with :math:`z_j \in [0, 1]`. The sum has :math:`|\mathbf{m}|! / \prod_j m_j!` terms. The matrices are formed
        by a dense inverse and cached per :math:`\theta`.

        For several epochs, the mutation counts are tracked on the lattice
        :math:`\mathcal{C} = \{\mathbf{c} \in \mathbb{N}_0^J : \mathbf{c} \le \mathbf{m}\}` of
        :math:`L = \prod_j (m_j + 1)` nodes. Let :math:`\boldsymbol{\delta}_{\mathbf{c}} \in \{0, 1\}^L` be the
        indicator column vector of node :math:`\mathbf{c}`, :math:`\mathbf{u}_j` the :math:`j`-th unit vector of
        :math:`\mathbb{N}_0^J`, :math:`\mathbf{I}_L` the :math:`L \times L` identity matrix, and
        :math:`\mathbf{N}_j \in \{0, 1\}^{L \times L}` the matrix whose entry
        :math:`(\mathbf{c}, \mathbf{c} + \mathbf{u}_j)` is one whenever :math:`c_j < m_j` and whose other entries are
        zero. In epoch :math:`i`, the process on lattice nodes and transient states has the sub-intensity matrix

        .. math::
            \mathbf{A}_i = \mathbf{I}_L \otimes \left( \mathbf{T}_i - \theta \operatorname{diag}(\bar{\mathbf{r}})
            \right) + \theta \sum_{j=1}^{J} \mathbf{N}_j \otimes \operatorname{diag}(\mathbf{r}_j),

        where :math:`\otimes` is the Kronecker product. A class-:math:`j` mutation moves the process from node
        :math:`\mathbf{c}` to :math:`\mathbf{c} + \mathbf{u}_j`, and a mutation at :math:`c_j = m_j` removes it, which
        realizes the factor :math:`e^{-\theta \ell_j}`. The row vector
        :math:`\mathbf{v}_i \in \mathbb{R}^{1 \times L n_T}` of sub-probabilities at time :math:`t_i` starts at
        :math:`\mathbf{v}_0 = \boldsymbol{\delta}_{\mathbf{0}}^\top \otimes \boldsymbol{\alpha}_T` and evolves as
        :math:`\mathbf{v}_i = \mathbf{v}_{i-1} \exp(\mathbf{A}_i \Delta_i)` for :math:`i < M`. The probability is the
        mass absorbed from node :math:`\mathbf{m}`, accumulated over all epochs:

        .. math::
            \mathbb{P}(\mathbf{Y} = \mathbf{m})
            = \sum_{i=1}^{M-1} (\mathbf{v}_i - \mathbf{v}_{i-1})\, \mathbf{A}_i^{-1}
              (\boldsymbol{\delta}_{\mathbf{m}} \otimes \mathbf{q}_i)
            + \mathbf{v}_{M-1} (-\mathbf{A}_M)^{-1} (\boldsymbol{\delta}_{\mathbf{m}} \otimes \mathbf{q}_M).

        Each :math:`\mathbf{A}_i` is block upper triangular with non-singular diagonal blocks for :math:`\theta > 0`,
        and for :math:`M = 1` the expression reduces to the single-epoch formula. The matrix :math:`\mathbf{A}_i` is
        assembled sparse and factorized by a block-triangular sparse LU decomposition once :math:`L n_T` reaches
        :attr:`Settings.closed_form_sparse_min_states <phasegen.settings.Settings.closed_form_sparse_min_states>`, and
        the matrix exponential is applied as an action on :math:`\mathbf{v}_{i-1}` once :math:`L n_T` reaches
        :attr:`Settings.expm_action_min_dim <phasegen.settings.Settings.expm_action_min_dim>`, using the active
        :class:`~phasegen.expm.Backend`. The cost grows with the number of sequences for one epoch and with
        :math:`L n_T` for several.

        :meth:`UnfoldedSFSDistribution.get_mutation_configs_by_count()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs_by_count>` enumerates the configurations
        in ascending order of :math:`|\mathbf{m}|`. :meth:`UnfoldedSFSDistribution.get_mutation_configs()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_configs>` enumerates them in descending order of
        probability. It starts at :math:`\operatorname{round}(\theta\, \mathbb{E}[\ell_j])`, with
        :math:`\mathbb{E}[\ell_j]` taken from :attr:`UnfoldedSFSDistribution.mean
        <phasegen.distributions.UnfoldedSFSDistribution.mean>`, moves to a more probable neighbour
        :math:`\mathbf{m} \pm \mathbf{u}_j` until none exists, and expands outward from that configuration with a
        priority queue. The order is exactly descending whenever every configuration other than the most probable one
        has a neighbour of at least equal probability. Both iterators set :attr:`UnfoldedSFSDistribution.generated_mass
        <phasegen.distributions.UnfoldedSFSDistribution.generated_mass>` to zero when the first configuration is
        requested and add every yielded probability to it. One minus this value is the total probability of the
        configurations not yet yielded, irrespective of the order.

        .. rubric:: References

        Hobolth, A., Boitard, S., Futschik, A. and Leblois, R. (2025). A matrix-analytical sampling formula for
        time-homogeneous coalescent processes under the infinite sites mutation model. Theoretical Population
        Biology, 163, 62-79. https://doi.org/10.1016/j.tpb.2025.03.002

        :param config: The configuration :math:`\mathbf{m}`, a sequence of :math:`J` non-negative integers ordered by
            class, starting at class 1. For :math:`n = 4`, the unfolded configuration ``[2, 1, 0]`` holds two
            singletons, one doubleton and no tripletons, and the folded configuration ``[2, 1]`` holds two singletons
            or tripletons and one doubleton.
        :param theta: The mutation rate :math:`\theta` per unit of branch length.
        :return: The probability :math:`\mathbb{P}(\mathbf{Y} = \mathbf{m})`.
        :raises ValueError: If ``theta`` is negative, or if ``config`` does not have :math:`J` entries or has an entry
            that is negative or not an integer.
        :raises NotImplementedError: If the coalescent has a positive start time or a finite end time.
        """
        # make sure theta is non-negative
        if theta < 0:
            raise ValueError("Theta must be greater than or equal to 0.")

        # the probabilities are to-absorption and ignore a configured window
        self._assert_no_window()

        # number of frequency bins
        n = len(self._get_configs(self.lineage_config.n, 0)[0])

        if len(config) != n:
            raise ValueError(
                "The length of the configuration must be equal to the number of frequency bins. "
                f"Expected {n}, got {len(config)}."
            )

        # entries are counts: integral values of any numeric type (R passes doubles) are accepted
        if any(not float(c).is_integer() or c < 0 for c in config):
            raise ValueError(f"The configuration entries must be non-negative integers, got {list(config)}.")

        config = tuple(int(c) for c in config)

        # handle special case when theta = 0
        if theta == 0:
            if sum(config) == 0:
                return 1

            return 0

        # the single-epoch resolvent integrates the inter-mutation waiting time in closed form, which requires a
        # constant rate matrix. Several epochs integrate the lattice process epoch by epoch.
        if self.demography.has_n_epochs(2):
            return self._get_mutation_config_inhomogeneous(config, n, theta)

        return self._get_mutation_config_homogeneous(config, n, theta)

    def _get_mutation_config_homogeneous(self, config: Tuple[int, ...], n: int, theta: float) -> float:
        """
        Single-epoch configuration probability of
        :meth:`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`,
        summing the products of ``_get_P`` matrices over the multiset permutations of the mutation classes.

        :param config: The configuration, one non-negative count per frequency class.
        :param n: The number of frequency classes.
        :param theta: The mutation rate.
        :return: The configuration probability.
        """
        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)
        k = non_absorbing.sum()

        alpha = self.state_space.alpha[non_absorbing]

        P, p_total = self._get_P(n, theta)

        q = list(itertools.chain(*[[i + 1] * j for i, j in enumerate(config)]))

        # iterate over permutations of q
        Q = np.zeros((k, k))
        for p in multiset_permutations(q):
            U = np.eye(k)

            for i in p:
                U @= P[i - 1]

            Q += U

        return alpha @ Q @ p_total

    @cached_property
    def _mutation_epoch_data(self) -> Tuple:
        """
        Configuration-independent inputs of ``_get_mutation_config_inhomogeneous``, cached per instance.

        :return: ``(non_absorbing, R, r_total, alpha, epochs)``: the transient-state mask, the per-class reward vectors
            and their sum, the transient initial distribution, and per epoch the dense sub-intensity matrix, the
            absorption-rate vector and the duration (``None`` for the unbounded last epoch).
        """
        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)
        n = len(self._get_configs(self.lineage_config.n, 0)[0])
        R = [self._get_sfs_reward(i + 1)._get(self.state_space)[non_absorbing] for i in range(n)]
        r_total = np.sum(R, axis=0)
        alpha = self.state_space.alpha[non_absorbing]

        epochs = []
        for epoch in self.demography.epochs:
            self.state_space.update_epoch(epoch)
            S = self.state_space.S[non_absorbing, :][:, non_absorbing]  # sparse-safe slice
            S = S.toarray() if sp.issparse(S) else np.asarray(S)
            e = -S @ np.ones(S.shape[0])  # coalescent absorption-rate vector (state_space.e is the all-ones vector)
            tau = None if np.isinf(epoch.end_time) else epoch.end_time - epoch.start_time
            epochs.append((S, e, tau))
            if tau is None:
                break

        # leave the state space in the first epoch for any subsequent caller that assumes it
        self.state_space.update_epoch(self.demography.get_epoch(0))

        return non_absorbing, R, r_total, alpha, epochs

    def _get_mutation_config_inhomogeneous(self, config: Tuple[int, ...], n: int, theta: float) -> float:
        """
        Multi-epoch configuration probability of
        :meth:`UnfoldedSFSDistribution.get_mutation_config() <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`,
        propagating the lattice sub-intensity matrix epoch by epoch and solving the last epoch to absorption.

        :param config: The configuration, one non-negative count per frequency class.
        :param n: The number of frequency classes.
        :param theta: The mutation rate.
        :return: The configuration probability.
        """
        non_absorbing, R, r_total, alpha, epochs = self._mutation_epoch_data
        m = len(alpha)

        # enumerate the mutation-count lattice nodes 0..k_i per bin and the super-diagonal (one-mutation) edges
        nodes = list(itertools.product(*[range(k + 1) for k in config]))
        index = {c: a for a, c in enumerate(nodes)}
        edges = [(index[c], index[c[:i] + (c[i] + 1,) + c[i + 1:]], i)
                 for c in nodes for i in range(n) if c[i] < config[i]]
        k_block = index[config] * m
        L = len(nodes)
        nt = L * m

        # mirror the moment machinery's two crossovers: keep the augmented generator sparse (and LU-solve it in
        # block-triangular form) above ``closed_form_sparse_min_states``, and propagate via the sparse
        # matrix-exponential action above ``expm_action_min_dim`` instead of forming the dense exponential. No
        # ``lamb`` reward-regularization applies here: the mutation rates ``theta R_i`` are genuine generator entries
        # (not a separately-accumulated reward), so there is nothing to rescale relative to ``S``.
        sparse = nt >= Settings.closed_form_sparse_min_states
        action = nt >= Settings.expm_action_min_dim

        def build_generator(S: np.ndarray) -> 'np.ndarray | sp.spmatrix':
            diag = S - theta * np.diag(r_total)
            if sparse:
                blocks = [[None] * L for _ in range(L)]
                for a in range(L):
                    blocks[a][a] = sp.csr_matrix(diag)
                for a, b, i in edges:
                    blocks[a][b] = sp.diags(theta * R[i])
                return sp.bmat(blocks, format='csr')

            A = np.zeros((nt, nt))
            for a in range(L):
                A[a * m:(a + 1) * m, a * m:(a + 1) * m] = diag
            for a, b, i in edges:
                A[a * m:(a + 1) * m, b * m:(b + 1) * m] = np.diag(theta * R[i])
            return A

        # entering row vector: alpha at the empty lattice node
        v = np.zeros(nt)
        v[:m] = alpha

        p = 0.0
        for S, e, tau in epochs:
            A = build_generator(S)

            if tau is None:
                # final unbounded epoch: integrated occupation to absorption is occ = v @ (-A)^{-1}
                occ = self._lu_solver((-A).T, sparse)(v)
                p += occ[k_block:k_block + m] @ e
                break

            # finite epoch: survivors u = v @ exp(A tau) (matrix-exponential action), then register absorption from
            # the integrated occupation occ = (u - v) @ A^{-1}, and carry the survivors into the next epoch
            if action:
                u = Backend.expm_multiply(A.T * tau, v)
            else:
                u = v @ expm((A.toarray() if sparse else A) * tau)
            occ = self._lu_solver(A.T, sparse)(u - v)
            p += occ[k_block:k_block + m] @ e
            v = u

        return float(p)

    def get_mutation_configs_by_count(self, theta: float) -> Iterator[Tuple[List[int], float]]:
        """
        Unending iterator over mutational configurations and their probabilities in ascending order of the total
        number of mutations, as described in :meth:`UnfoldedSFSDistribution.get_mutation_config()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.

        :param theta: The mutation rate per unit of branch length.
        :return: An iterator over pairs of configuration and probability.
        """
        # reset generated mass
        self.generated_mass = 0

        # iterate over number of mutations
        i = 0
        while True:
            # iterate over configurations
            for config in self._get_configs(self.lineage_config.n, i):
                p = self.get_mutation_config(config=config, theta=theta)
                self.generated_mass += p
                yield config, p

            # increase counter for number of mutations
            i += 1

    def get_mutation_configs(self, theta: float) -> Iterator[Tuple[Tuple[int, ...], float]]:
        """
        Unending iterator over mutational configurations and their probabilities in descending order of probability,
        as described in :meth:`UnfoldedSFSDistribution.get_mutation_config()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`. The following example consumes it until
        the yielded probability mass exceeds 0.8.

        ::

            coal = pg.Coalescent(n=5)

            it = coal.sfs.get_mutation_configs(theta=1)

            samples = list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))

        :param theta: The mutation rate per unit of branch length.
        :return: An iterator over pairs of configuration and probability.
        """
        # reset generated mass
        self.generated_mass = 0

        n = len(self._get_configs(self.lineage_config.n, 0)[0])

        # special case theta = 0: only the empty configuration carries mass
        if theta == 0:
            self.generated_mass = 1.0
            yield (0,) * n, 1.0
            return

        def neighbours(c: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
            for i in range(n):
                for step in (1, -1):
                    if c[i] + step >= 0:
                        yield c[:i] + (c[i] + step,) + c[i + 1:]

        # modal configuration from the expected per-bin branch lengths (E[# mutations in bin] = theta * E[ell])
        mean = np.asarray(self.mean.data)
        mode = tuple(max(0, int(round(theta * mean[idx]))) for idx in self._get_indices())

        # hill-climb to a local maximum of the configuration probability
        p_mode = self.get_mutation_config(mode, theta)
        improved = True
        while improved:
            improved = False
            for nb in neighbours(mode):
                p_nb = self.get_mutation_config(nb, theta)
                if p_nb > p_mode:
                    mode, p_mode, improved = nb, p_nb, True
                    break

        # best-first expansion outward, evaluating each configuration once
        seen = {mode}
        heap = [(-p_mode, mode)]
        while heap:
            neg_p, c = heapq.heappop(heap)
            self.generated_mass += -neg_p
            yield c, -neg_p

            for nb in neighbours(c):
                if nb not in seen:
                    seen.add(nb)
                    heapq.heappush(heap, (-self.get_mutation_config(nb, theta), nb))


class TajimaSFSMixin:
    """
    Mixin providing the branch-length diversity estimators and Tajima's :math:`D` from the site-frequency
    spectrum mean and covariance. Shared by the analytical :class:`UnfoldedSFSDistribution` and the
    simulation-based empirical SFS distribution, so the same statistics can be computed from either source.
    Subclasses supply the number of lineages and the mean and covariance of the polymorphic bins.
    """

    def _tajima_n(self) -> int:
        """Number of lineages."""
        raise NotImplementedError

    def _tajima_mean(self) -> np.ndarray:
        """Mean branch length per polymorphic SFS bin (``i = 1 .. n-1``)."""
        raise NotImplementedError

    def _tajima_cov(self) -> np.ndarray:
        """Covariance of the polymorphic SFS bins (``i, j = 1 .. n-1``)."""
        raise NotImplementedError

    @cached_property
    def _tajima_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-bin weights for the two diversity estimators: pairwise diversity ``pi`` and Watterson's ``theta_W``."""
        n = self._tajima_n()
        i = np.arange(1, n)
        w_pi = 2 * i * (n - i) / (n * (n - 1))
        w_w = np.full(n - 1, 1 / np.sum(1 / i))

        return w_pi, w_w

    @cached_property
    def theta_pi(self) -> float:
        r"""
        Expected pairwise diversity in branch-length units,

        .. math::

            \mathbb{E}[\pi] = \sum_{i=1}^{n-1} \frac{2 i (n - i)}{n (n - 1)}\, \mathbb{E}[L_i],

        where :math:`L_i` is the total length of the branches subtending :math:`i` of the :math:`n` samples.
        """
        w_pi, _ = self._tajima_weights

        return float(w_pi @ self._tajima_mean())

    @cached_property
    def theta_w(self) -> float:
        r"""
        Expectation of Watterson's estimator in branch-length units,

        .. math::

            \mathbb{E}[\theta_W] = \frac{1}{a_n} \sum_{i=1}^{n-1} \mathbb{E}[L_i],
            \qquad a_n = \sum_{\ell=1}^{n-1} \frac{1}{\ell},

        with :math:`L_i` as in :attr:`UnfoldedSFSDistribution.theta_pi
        <phasegen.distributions.UnfoldedSFSDistribution.theta_pi>`.
        """
        _, w_w = self._tajima_weights

        return float(w_w @ self._tajima_mean())

    @cached_property
    def tajimas_d(self) -> float:
        r"""
        Tajima's :math:`D` in branch form,

        .. math::

            D = \frac{\mathbb{E}[\pi] - \mathbb{E}[\theta_W]}{\sqrt{\mathbf{c}^\top \boldsymbol{\Sigma}\, \mathbf{c}}},
            \qquad c_i = \frac{2 i (n - i)}{n (n - 1)} - \frac{1}{a_n},

        with :math:`\mathbb{E}[\pi]`, :math:`\mathbb{E}[\theta_W]`, :math:`L_i` and :math:`a_n` as in
        :attr:`UnfoldedSFSDistribution.theta_pi <phasegen.distributions.UnfoldedSFSDistribution.theta_pi>` and
        :attr:`UnfoldedSFSDistribution.theta_w <phasegen.distributions.UnfoldedSFSDistribution.theta_w>`,
        :math:`\boldsymbol{\Sigma}` the covariance matrix of :math:`L_1, \dots, L_{n-1}` and
        :math:`\mathbf{c} = (c_1, \dots, c_{n-1})`. The value is 0 when the variance vanishes. :math:`D` is 0 under the
        standard neutral constant-size model, negative under population growth and positive under contraction. The
        normalization is the branch-length covariance, not the mutation-based variance of the classical sample
        estimator.
        """
        w_pi, w_w = self._tajima_weights
        c = w_pi - w_w

        num = c @ self._tajima_mean()
        var = c @ self._tajima_cov() @ c

        if var <= 0:
            return 0.0

        return float(num / np.sqrt(var))


class UnfoldedSFSDistribution(SFSDistribution, TajimaSFSMixin):
    """
    Unfolded site-frequency spectrum distribution.
    """

    def _get_sfs_reward(self, i: int) -> UnfoldedSFSReward:
        """
        Get the reward for the ith site-frequency count.

        :param i: The ith site-frequency count.
        :return: The reward.
        """
        return UnfoldedSFSReward(i)

    def _get_indices(self) -> np.ndarray:
        """
        Get the indices for the site-frequency spectrum.

        :return: The indices.
        """
        return np.arange(1, self.lineage_config.n)

    def _tajima_n(self) -> int:
        return self.lineage_config.n

    def _tajima_mean(self) -> np.ndarray:
        n = self.lineage_config.n
        return np.asarray(self.mean.data)[1:n]

    def _tajima_cov(self) -> np.ndarray:
        n = self.lineage_config.n
        return np.asarray(self.cov.data)[1:n, 1:n]

    @staticmethod
    def _get_configs(n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Get all possible mutational configurations for a given number of mutations.

        :param n: The number of lineages.
        :param k: The number of mutations.
        :return: An iterator over all possible mutational configurations.
        """
        return StateSpace._get_partitions(n=k, k=n - 1)


class FoldedSFSDistribution(SFSDistribution):
    """
    Folded site-frequency spectrum distribution.
    """

    def _get_sfs_reward(self, i: int) -> FoldedSFSReward:
        """
        Get the reward for the ith site-frequency count.

        :param i: The ith site-frequency count.
        :return: The reward.
        """
        return FoldedSFSReward(i)

    def _get_indices(self) -> np.ndarray:
        """
        Get the indices for the site-frequency spectrum.

        :return: The indices.
        """
        return np.arange(1, self.lineage_config.n // 2 + 1)

    @staticmethod
    def _get_configs(n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Get all possible mutational configurations for a given number of mutations.

        :param n: The number of lineages.
        :param k: The number of mutations.
        :return: An iterator over all possible mutational configurations.
        """
        return StateSpace._get_partitions(n=k, k=n // 2)

    def _unfold(self, config: Sequence[int]) -> Set[Tuple[int, ...]]:
        """
        Unfold a folded configuration into all possible unfolded configurations.

        :param config: The folded configuration. A sequence of integers of length n // 2 where n is the number of
            lineages.
        :return: The unfolded configurations.
        """
        n = self.lineage_config.n

        if n // 2 != len(config):
            raise ValueError("The length of the configuration must equal n // 2 where n is the number of lineages.")

        if n % 2 == 1:
            lower_counts = [range(i + 1) for i in config]
            i_center = len(config)
        else:
            lower_counts = [range(i + 1) for i in config[:-1]] + [[config[-1]]]
            i_center = len(config) - 1

        unfolded = []
        # iterate over unfolded configurations
        for lower in itertools.product(*lower_counts):
            # get higher counts
            higher = (np.array(config) - np.array(lower))[:i_center][::-1]

            unfolded += [list(lower) + list(higher)]

        return set(tuple(u) for u in unfolded)


class _JointSFSAggregateFunction:
    """Per-bin joint-SFS function, looping ``JointSFSDistribution._bin_distribution`` over the descendant
    configurations."""

    def __call__(self, t) -> 'JointSFS | np.ndarray':
        """
        Evaluate the function of every joint SFS bin under the spectrum's reward, as for
        :class:`~phasegen.distributions.SFSCDF`.

        :param t: A point or an array of points, or probability levels for a quantile function.
        :return: For a scalar ``t``, a :class:`~sfsutils.spectrum.JointSFS` with one value per descendant
            configuration. For an array, an array of shape ``(len(t),) + shape``, with ``shape`` the shape of the joint
            SFS.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        d = self._distribution
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros((t_arr.size,) + d.shape)
        for config in d._get_configs():
            out[(slice(None),) + tuple(config)] = getattr(d._bin_distribution(config), self.kind)(t_arr)
        return JointSFS(out[0], pop_names=d.lineage_config.pop_names) if np.ndim(t) == 0 else out

    def plot(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the function of every joint SFS bin at once, one curve per descendant configuration.

        :param ax: Axes to plot on.
        :param x: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.
        :param configs: The joint bins (descendant configurations) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(x=x, configs=configs, n_points=n_points),
                                         file=file, show=show, clear=clear, label=label, title=title, **kwargs)


class JointSFSDensity(_JointSFSAggregateFunction, MarginalDensity):
    """Per-bin densities of the joint SFS, each that of the bin's :class:`~phasegen.distributions.RewardDistribution`."""


class JointSFSCDF(_JointSFSAggregateFunction, MarginalCDF):
    """Per-bin CDFs of the joint SFS, each that of the bin's :class:`~phasegen.distributions.RewardDistribution`."""


class JointSFSQuantileFunction(_JointSFSAggregateFunction, MarginalQuantileFunction):
    """Per-bin quantile functions of the joint SFS, each that of the bin's
    :class:`~phasegen.distributions.RewardDistribution`."""

    def plot(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Plot the quantile function of every joint SFS bin at once (bin branch length versus probability ``q``).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate at. By default, :attr:`~phasegen.settings.Settings.plot_n_grid` points
            from ``1 - Settings.plot_endpoint_quantile`` to :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param configs: The joint bins (descendant configurations) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the current figure.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, configs=configs, n_points=n_points),
                                         file=file, show=show, clear=clear, label=label, title=title, **kwargs)


class JointSFSDistribution(PhaseTypeDistribution):
    r"""
    Joint (multi-population) site-frequency spectrum distribution.

    Moments are returned as a multi-dimensional array of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, where ``n_p`` is the
    sample size of population ``p``. The entry at index :math:`(c_0, \dots, c_{P-1})` is the moment of the branch
    length :math:`L_{(c_0, \dots, c_{P-1})}` subtending exactly :math:`c_p` samples from population :math:`p`, for
    :math:`P` populations. The mean is :math:`\mathbb{E}[L_{(c_0, \dots, c_{P-1})}]`. The monomorphic bins (the
    all-zero and the full :math:`(n_0, \dots, n_{P-1})` configuration) are zero by convention.

    The evaluation of spectrum-wide moments is described in
    :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
    """
    # per-bin (per descendant configuration) pdf/cdf/quantile -> joint-SFS aggregate flavours (the per-config loop
    # lives on these function objects)
    _pdf_function = JointSFSDensity
    _cdf_function = JointSFSCDF
    _quantile_function = JointSFSQuantileFunction

    @property
    def pdf(self) -> JointSFSDensity:
        """Per-bin (per descendant configuration) probability density functions: callable and plottable."""
        return super().pdf

    @property
    def cdf(self) -> JointSFSCDF:
        """Per-bin (per descendant configuration) cumulative distribution functions: callable and plottable."""
        return super().cdf

    @property
    def quantile(self) -> JointSFSQuantileFunction:
        """Per-bin (per descendant configuration) quantile functions: callable and plottable."""
        return super().quantile

    def __init__(
            self,
            state_space: JointBlockCountingStateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography,
            reward: Reward = None
    ) -> None:
        """
        Initialize the distribution.

        :param state_space: Joint block-counting state space.
        :param tree_height: The tree height distribution.
        :param demography: The demography.
        :param reward: The reward to multiply the joint SFS reward with. By default, the unit reward is used, which
            has no effect.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(
            state_space=state_space,
            tree_height=tree_height,
            demography=demography,
            reward=reward
        )

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the joint SFS array, ``(n_0 + 1, ..., n_{P-1} + 1)``.
        """
        return tuple(int(n_p) + 1 for n_p in self.lineage_config.lineages)

    def _get_configs(self) -> List[Tuple[int, ...]]:
        """
        Get the descendant vectors corresponding to (polymorphic) joint SFS bins, i.e. all block configurations
        except the full-sample configuration (which corresponds to the monomorphic, fixed sites).

        :return: List of descendant vectors.
        """
        full = tuple(int(n_p) for n_p in self.lineage_config.lineages)

        return [c for c in self.state_space.block_configs if c != full]

    def sample(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> np.ndarray:
        r"""
        Draw samples of the joint site-frequency spectrum as
        :meth:`UnfoldedSFSDistribution.sample() <phasegen.distributions.UnfoldedSFSDistribution.sample>` does, with one
        entry per descendant vector :math:`\mathbf{c} = (c_0, \dots, c_{P-1})`, where :math:`c_p` counts the
        descendants from population :math:`p` of :math:`P`. The entry is the branch length accumulated by the lineages
        with descendant vector :math:`\mathbf{c}`. The all-zero and the full configuration are zero.

        :param n_samples: Number of joint spectra.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: Array of shape ``(n_samples, *shape)``, whose means over the first axis estimate :attr:`mean`.
        """
        configs = self._get_configs()
        rewards = [CombinedReward([self.reward, JointSFSReward(c)]) for c in configs]
        sampled = self._sample(n_samples, rewards=rewards, rng=np.random.default_rng(seed))

        out = np.zeros((n_samples,) + self.shape)
        for j, config in enumerate(configs):
            out[(slice(None),) + config] = sampled[:, j]

        return out

    def to_empirical(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> 'EmpiricalJointSFSDistribution':
        """
        Build an empirical joint spectrum from ``n_samples`` samples of
        :meth:`JointSFSDistribution.sample() <phasegen.distributions.JointSFSDistribution.sample>`, holding the raw
        moments described at :class:`~phasegen.distributions.EmpiricalJointSFSDistribution` and a capped subset of the
        samples.

        :param n_samples: Number of trajectories.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: The empirical joint spectrum.
        """
        from .empirical import EmpiricalJointSFSDistribution, MsprimeCoalescent

        samples = self.sample(n_samples, seed=seed)  # (n_samples, *shape)

        # non-central moments of orders 1 .. max (matching the msprime joint-SFS ground truth)
        max_order = MsprimeCoalescent._jsfs_max_order
        moments = np.stack([(samples ** order).mean(axis=0) for order in range(1, max_order + 1)])

        cap = MsprimeCoalescent._jsfs_sample_cap

        return EmpiricalJointSFSDistribution(moments=moments, samples=samples[:cap], n_samples=samples.shape[0])

    def moment(
            self,
            k: int,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        r"""
        The :math:`k`-th moment of every joint site-frequency spectrum bin, central by default, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :param end_time: The end time :math:`t_\mathrm{end}`. By default, the end time of the distribution, or
            absorption.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :return: An array of shape :attr:`shape` holding the :math:`k`-th moment of each joint SFS bin.
        """
        effective_start = self.tree_height.start_time if start_time is None else start_time

        # batched mean: all joint bins share one occupation-time vector, so the whole joint SFS mean is a single
        # contraction over the stacked bin rewards (closed form's spectrum path). Only for the plain mean (k=1, no
        # custom end time); a non-zero start time subtracts the occupation up to it. Other cases fall through to the
        # per-bin accumulation.
        if (
                Settings.closed_form_last_epoch and
                int(k) == 1 and
                end_time is None and
                self.tree_height.end_time is None
        ):
            occupation = self._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
                if effective_start > 0:
                    m = m - self._occupation_times(cap=effective_start)[0]
                base = np.asarray(self.reward._get(self.state_space), dtype=float)
                configs = self._get_configs()
                R = np.column_stack([
                    (base * np.asarray(JointSFSReward(config)._get(self.state_space), dtype=float))[idx_t]
                    for config in configs
                ])
                values = m @ R
                out = np.zeros(self.shape)
                for config, value in zip(configs, values):
                    out[config] = value
                return JointSFS(out, pop_names=self.lineage_config.pop_names)

        # like the base distribution, a moment is the accumulation over the [start_time, end_time] window
        if start_time is None:
            start_time = self.tree_height.start_time

        if end_time is None:
            # evaluate the moment to absorption: signal the closed-form path with an infinite end time when it
            # applies (no explicit end time, accumulation from 0, and absorption certain in the last epoch), but not
            # when flattening applies (which takes precedence and delegates to the smaller lineage-counting space),
            # otherwise use the estimated absorption time
            if (
                    Settings.closed_form_last_epoch and
                    not self._flattening_applies(k) and
                    start_time == 0 and
                    self.tree_height.end_time is None and
                    self._absorption_certain_in_last_epoch()
            ):
                end_time = np.inf
            else:
                end_time = self.tree_height.t_max

        if start_time > 0 and int(k) == 1:
            # the mean is additive in time, so the windowed mean is the difference of the two cumulative means
            acc = self.accumulate(k, [start_time, end_time], center=center, permute=permute)
            out = acc[..., 1] - acc[..., 0]
        elif start_time > 0:
            # for k >= 2 the windowed moment ``E[(Y_b - Y_a)^k]`` is NOT the difference of the cumulative-from-0
            # moments (that omits the cross terms); accumulate each bin directly over the [start_time, end_time]
            # window (see MomentEvaluator._accumulate_windowed)
            out = np.zeros(self.shape)
            for config in self._get_configs():
                rewards = tuple(CombinedReward([self.reward, JointSFSReward(config)]) for _ in range(k))
                out[config] = float(PhaseTypeDistribution.accumulate(
                    self,
                    k=k,
                    end_times=[end_time],
                    rewards=rewards,
                    center=center,
                    permute=permute,
                    start_time=start_time
                )[0])
        else:
            out = self.accumulate(k, [end_time], center=center, permute=permute)[..., 0]

        if np.isnan(out).any():
            raise ValueError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return JointSFS(out, pop_names=self.lineage_config.pop_names)

    def _config_items(
            self,
            configs: Sequence[Tuple[int, ...]] | None
    ) -> List[Tuple[Tuple[int, ...], 'RewardDistribution']]:
        """
        The requested joint bins with their distributions, under this spectrum's reward.

        :param configs: The descendant configurations, ``None`` for all polymorphic bins.
        :return: Each configuration and its distribution.
        """
        configs = self._get_configs() if configs is None else [tuple(int(x) for x in c) for c in configs]

        return [(c, self._bin_distribution(c)) for c in configs]

    def _bin_distribution(self, config: Tuple[int, ...]) -> 'RewardDistribution':
        """
        The distribution of the joint SFS bin ``config`` under this spectrum's reward, cached per bin in
        ``_bin_distributions`` so the cosine fit behind its cdf, pdf and quantile is built once. Honors
        ``Settings.cache``.

        :param config: The descendant configuration, one count per population.
        :return: The distribution of the bin's branch length.
        """
        config = tuple(int(c) for c in config)

        if not Settings.cache:
            return self.distribution(reward=CombinedReward([self.reward, JointSFSReward(config)]))

        cache = self.__dict__.setdefault('_bin_distributions', {})
        if config not in cache:
            cache[config] = self.distribution(reward=CombinedReward([self.reward, JointSFSReward(config)]))
        return cache[config]

    def bin(self, *config: int) -> 'RewardDistribution':
        """The distribution of the branch length of the joint SFS bin with the given descendant counts per population,
        as a :class:`~phasegen.distributions.RewardDistribution`, for example ``jsfs.bin(1, 0).quantile(0.9)``. The
        bin reward is combined with the reward of this spectrum, as for :meth:`UnfoldedSFSDistribution.bin()
        <phasegen.distributions.UnfoldedSFSDistribution.bin>`, and the distribution is cached per bin.

        :param config: The descendant configuration, one count per population.
        :return: The distribution of the bin's branch length.
        """
        config = tuple(int(c) for c in config)
        d = self._bin_distribution(config)
        d.label = f"jSFS bin {config}"
        return d

    def joint_distribution(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> 'JointRewardDistribution':
        """
        Joint distribution of the branch lengths of two joint SFS bins within one genealogy, as a
        :class:`~phasegen.distributions.JointRewardDistribution`.

        :param config_a: The first descendant configuration, one count per population.
        :param config_b: The second descendant configuration.
        :return: The joint distribution of the two branch lengths.
        """
        jd = super().joint_distribution(JointSFSReward(tuple(config_a)), JointSFSReward(tuple(config_b)))
        jd.label = f"jSFS bins {tuple(config_a)} x {tuple(config_b)}"
        return jd

    def _plot_data_cdf(
            self,
            x: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The CDF curve of each joint SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param x: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param configs: The joint bins (descendant configurations) to include. By default, all of them.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by configuration.
        """
        return self._reward_curves('cdf', self._config_items(configs), x, n_points, 'Joint SFS bin CDFs', 'config')

    def _plot_data_pdf(
            self,
            x: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The density curve of each joint SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param x: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param configs: The joint bins (descendant configurations) to include. By default, all of them.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by configuration.
        """
        return self._reward_curves('pdf', self._config_items(configs), x, n_points, 'Joint SFS bin PDFs', 'config')

    def _plot_data_quantile(
            self,
            q: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The quantile curve of each joint SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param q: Probabilities to evaluate at. By default, an evenly spaced grid in ``(0, 1)``.
        :param configs: The joint bins (descendant configurations) to include. By default, all of them.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by configuration.
        """
        return self._reward_curves('quantile', self._config_items(configs), q, n_points,
                                   'Joint SFS bin quantile functions', 'config')

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        r"""
        The :math:`k`-th moment of every joint site-frequency spectrum bin accumulated up to each end time
        :math:`t_\mathrm{end}` in ``end_times``, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t_\mathrm{end}` at which to evaluate the moment.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :return: Array of shape :attr:`shape` ``+ (len(end_times),)`` with the moment of each bin over time.
        """
        k = int(k)
        configs = self._get_configs()
        end_times = np.array(list(end_times))

        # batched mean accumulation (k=1): all configs share the occupation-up-to-t grid m(t), so the whole joint
        # accumulation is one contraction m_grid @ R over the stacked config rewards
        if k == 1 and not self._flattening_applies(1):
            m_grid = self._mean_occupation_grid(end_times)
            ss = self.state_space
            R = np.column_stack([
                np.asarray(CombinedReward([self.reward, JointSFSReward(config)])._get(ss), dtype=float)
                for config in configs
            ])
            self._logger.debug("jsfs accumulate (k=1): batched (shared occupation grid over %d bins)", len(configs))
            accumulation = (m_grid @ R).T
        else:
            accumulation = np.array([
                PhaseTypeDistribution.accumulate(
                    self,
                    k=k,
                    end_times=end_times,
                    rewards=tuple(CombinedReward([self.reward, JointSFSReward(config)]) for _ in range(k)),
                    center=center,
                    permute=permute
                )
                for config in configs
            ])

        out = np.zeros(self.shape + (len(end_times),))
        for config, acc in zip(configs, accumulation):
            out[config] = acc

        return out

    def _plot_accumulation_data(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            center: bool = True,
            permute: bool = True
    ) -> '_CurveData':
        """
        The accumulation of the joint SFS moments over time that :meth:`plot_accumulation` draws, one curve per
        polymorphic bin.

        :param k: The order of the moment.
        :param end_times: Times at which to evaluate the moment. By default, :attr:`Settings.plot_n_grid` points up to
            the :attr:`Settings.plot_endpoint_quantile` quantile of the tree height.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :return: The curves, labelled by descendant configuration.
        """
        from ..visualization import _CurveData

        k = int(k)
        end_times = self._default_end_times() if end_times is None else np.asarray(list(end_times), dtype=float)
        configs = self._get_configs()
        accumulation = self.accumulate(k, end_times, center=center, permute=permute)

        return _CurveData(
            x=end_times,
            y=np.array([accumulation[config] for config in configs]).reshape(len(configs), len(end_times)),
            labels=[str(config) for config in configs],
            xlabel='t',
            ylabel='moment',
            title=f"Joint SFS moment accumulation (order {k})",
            legend_title='config'
        )

    def plot_accumulation(
            self,
            k: int = 1,
            end_times: Iterable[float] = None,
            center: bool = True,
            permute: bool = True,
            ax: 'plt.Axes' = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = None
    ) -> 'plt.Axes':
        """
        Plot accumulation of joint SFS moments over time, one curve per polymorphic bin.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. By default, :attr:`~phasegen.settings.Settings.plot_n_grid`
            points up to the :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile of the tree height.
        :param center: Whether to center the moment around the mean.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Plot title, ``None`` for the default title.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_accumulation_data(k, end_times, center, permute),
                                         file=file, show=show, clear=clear, title=title)

    @cached_property
    def mean(self) -> JointSFS:
        """
        Mean of the joint site-frequency spectrum, the expected branch length subtending each descendant
        configuration, evaluated as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> JointSFS:
        """
        Variance of the joint site-frequency spectrum, the diagonal of :attr:`cov` where the spectrum-wide evaluation
        of :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>` applies, and
        the second central moment of each bin otherwise.
        """
        batched = self._cov_batched
        if batched is not None:
            configs, cov = batched
            out = np.zeros(self.shape)
            for a, config in enumerate(configs):
                out[config] = cov[a, a]
            return JointSFS(out, pop_names=self.lineage_config.pop_names)

        return self.moment(k=2, center=True)

    def get_cov(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> float:
        """
        Get the covariance between the branch lengths subtending two descendant configurations.

        :param config_a: First descendant configuration.
        :param config_b: Second descendant configuration.
        :return: The covariance.
        """
        return PhaseTypeDistribution.moment(
            self,
            k=2,
            center=True,
            rewards=tuple(CombinedReward([self.reward, JointSFSReward(c)]) for c in (config_a, config_b))
        )

    @cached_property
    def _cov_batched(self) -> Optional[Tuple[List[Tuple[int, ...]], np.ndarray]]:
        """
        Batched joint-SFS covariance, contracting ``_two_point_occupation`` with the stacked bin rewards as in
        ``PhaseTypeDistribution.moment``. Cached so that ``cov`` and ``var`` share one solve.

        :return: ``(configs, cov)`` with ``cov`` the bins-by-bins covariance over the polymorphic ``configs``, or
            ``None`` when not applicable, and callers evaluate per pair.
        """
        if not Settings.closed_form_last_epoch:
            return None

        two_point = self._two_point_occupation()
        if two_point is None:
            return None

        K, idx_t = two_point
        ss = self.state_space
        base = np.asarray(self.reward._get(ss), dtype=float)
        configs = self._get_configs()
        R = np.column_stack([
            (base * np.asarray(JointSFSReward(config)._get(ss), dtype=float))[idx_t] for config in configs
        ])

        sfs_matrix = R.T @ K @ R                       # R^T K R (one ordering)
        self._logger.debug("jsfs.cov: centering with the outer product of bin means")
        mean = np.array([self.mean.data[config] for config in configs])
        cov = (sfs_matrix + sfs_matrix.T) - np.outer(mean, mean)

        return configs, cov

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance between the branch lengths of all pairs of (polymorphic) joint SFS bins. Returned as an array of
        shape :attr:`shape` ``+`` :attr:`shape`, where ``cov[a_0, ..., a_{P-1}, b_0, ..., b_{P-1}]`` is the covariance
        between bins ``(a_0, ..., a_{P-1})`` and ``(b_0, ..., b_{P-1})``. Evaluated as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
        """
        batched = self._cov_batched
        if batched is not None:
            self._logger.debug("jsfs.cov: batched (shared two-point occupation)")
            configs, cov = batched
            out = np.zeros(self.shape + self.shape)
            for a, config_a in enumerate(configs):
                for b, config_b in enumerate(configs):
                    out[tuple(config_a) + tuple(config_b)] = cov[a, b]
            return out

        configs = self._get_configs()
        pairs = [(a, b) for a in configs for b in configs]

        self._logger.debug("jsfs.cov: per-pair matrix exponential over %d config pairs", len(pairs))

        results = [self.get_cov(a, b) for a, b in pairs]

        out = np.zeros(self.shape + self.shape)
        for (a, b), result in zip(pairs, results):
            out[tuple(a) + tuple(b)] = result

        return out


class TwoLocusSFSDistribution(PhaseTypeDistribution):
    r"""
    Two-locus site-frequency spectrum under recombination. Entry :math:`(i, j)` of the (symmetrized) mean is the
    second cross-moment :math:`\mathbb{E}[L^0_i\, L^1_j]`, the expected product of the branch length subtending
    :math:`i` samples at locus 0 and :math:`j` samples at locus 1, computed from two per-locus SFS rewards on the
    two-locus block-counting state space. With recombination rate :math:`\rho \ge 0` between the loci, it reduces to
    :attr:`UnfoldedSFSDistribution.cov <phasegen.distributions.UnfoldedSFSDistribution.cov>` (plus the outer product of
    the marginal means) as :math:`\rho \to 0`, and for the standard coalescent to the outer product of the marginal
    SFS as :math:`\rho \to \infty`.

    The evaluation of :attr:`mean` is described in
    :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
    """

    def __init__(
            self,
            state_space: TwoLocusBlockCountingStateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography,
            reward: Reward = None
    ) -> None:
        """
        Initialize the distribution.

        :param state_space: Two-locus block-counting state space.
        :param tree_height: The (two-locus) tree height distribution, whose absorption time is when both loci have
            reached their MRCA.
        :param demography: The demography.
        :param reward: An optional reward to multiply the per-locus SFS rewards with. By default the unit reward.
        """
        if reward is None:
            reward = UnitReward()

        super().__init__(state_space=state_space, tree_height=tree_height, demography=demography, reward=reward)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the two-locus SFS array, ``(n + 1, n + 1)`` (one axis per locus).
        """
        n = int(self.lineage_config.n)
        return n + 1, n + 1

    def _get_indices(self) -> List[int]:
        """
        Polymorphic SFS bins ``1, ..., n - 1`` (the monomorphic ``0`` and ``n`` bins carry no information).
        """
        return list(range(1, self.lineage_config.n))

    def _no_univariate_distribution(self, *args, **kwargs) -> None:
        """A two-locus SFS entry ``(i, j)`` is the cross-moment ``E[L^0_i · L^1_j]`` — a product of two distinct
        branch lengths — so it has no single univariate distribution to invert. The marginal per-locus branch-length
        distributions are the ordinary single-locus SFS bin distributions (``pg.Coalescent(...).sfs``)."""
        raise NotImplementedError(
            "A two-locus SFS entry (i, j) is a cross-moment E[L^0_i . L^1_j] (a product of two rewards), so it has "
            "no single univariate CDF/PDF/quantile. For the marginal branch-length distribution of a frequency "
            "class, use the single-locus spectrum: pg.Coalescent(...).sfs.cdf / .pdf / .plot_cdf / .plot_pdf."
        )

    cdf = pdf = quantile = plot_cdf = plot_pdf = bin = _no_univariate_distribution

    def joint_distribution(self, i: int, j: int) -> 'JointRewardDistribution':
        r"""
        Joint distribution of the branch length :math:`L^0_i` of frequency class :math:`i` at locus 0 and the branch
        length :math:`L^1_j` of frequency class :math:`j` at locus 1, as a
        :class:`~phasegen.distributions.JointRewardDistribution`.

        :param i: The locus-0 frequency class.
        :param j: The locus-1 frequency class.
        :return: The joint distribution of :math:`(L^0_i, L^1_j)`.
        """
        jd = PhaseTypeDistribution.joint_distribution(self, TwoLocusSFSReward(0, i), TwoLocusSFSReward(1, j))
        jd.label = f"locus-0 bin {i} x locus-1 bin {j}"
        return jd

    @cached_property
    def mean(self) -> TwoLocusSFS:
        r"""
        Mean two-locus SFS, :math:`\mathbb{E}[L^0_i\, L^1_j]` for all polymorphic bins, symmetrized over the two loci
        and evaluated as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.
        """
        batched = self._mean_batched()
        if batched is not None:
            return batched

        n = self.lineage_config.n
        indices = [(i, j) for i in self._get_indices() for j in self._get_indices()]

        results = [
            PhaseTypeDistribution.moment(
                self, k=2, permute=False, center=False,
                rewards=(
                    CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),
                    CombinedReward([self.reward, TwoLocusSFSReward(1, j)])
                )
            )
            for i, j in indices
        ]

        out = np.zeros((n + 1, n + 1))
        for (i, j), result in zip(indices, results):
            out[i, j] = result

        # symmetrize over the two (exchangeable) loci, as for the single-locus SFS covariance
        return TwoLocusSFS((out + out.T) / 2)

    def _mean_batched(self) -> Optional[TwoLocusSFS]:
        """
        Batched mean two-locus SFS by the factored form of ``PhaseTypeDistribution.moment``, which never forms the
        dense two-point occupation matrix. Single epoch without an accumulation window only.

        :return: The mean two-locus SFS, or ``None`` when not applicable, and the caller evaluates per pair.
        """
        if not (Settings.closed_form_last_epoch and self.tree_height.end_time is None
                and self.tree_height.start_time == 0):
            return None

        epochs = self._get_epochs_until_unbounded()
        if len(epochs) > 1 or not self._absorption_certain_in_last_epoch():
            return None

        ss = self.state_space
        ss.update_epoch(epochs[-1])
        idx_t = np.where(~ss.absorbing)[0]
        use_action = len(idx_t) >= Settings.closed_form_sparse_min_states

        base = np.asarray(self.reward._get(ss), dtype=float)
        indices = self._get_indices()
        R0 = np.column_stack([
            (base * np.asarray(TwoLocusSFSReward(0, i)._get(ss), dtype=float))[idx_t] for i in indices
        ])
        R1 = np.column_stack([
            (base * np.asarray(TwoLocusSFSReward(1, j)._get(ss), dtype=float))[idx_t] for j in indices
        ])

        neg_t = -self._transient_block(idx_t, sparse=use_action)
        alpha = np.asarray(ss.alpha)[idx_t].astype(float)
        m = self._lu_solver(neg_t.T, use_action)(alpha)  # m = alpha (-T)^{-1}
        solve = self._lu_solver(neg_t, use_action)
        uncentered = (m[:, None] * R0).T @ solve(R1) + solve(R0).T @ (m[:, None] * R1)

        n = self.lineage_config.n
        out = np.zeros((n + 1, n + 1))
        for a, ia in enumerate(indices):
            out[ia, indices] = uncentered[a]

        # symmetrize over the two (exchangeable) loci, as the per-pair path does
        return TwoLocusSFS((out + out.T) / 2)

    def sample_per_locus(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Draw the branch lengths of both loci from shared trajectories, simulated as in
        :meth:`UnfoldedSFSDistribution.sample() <phasegen.distributions.UnfoldedSFSDistribution.sample>`. Entry
        :math:`i` for locus :math:`\ell \in \{0, 1\}` is the branch length :math:`L^\ell_i` accumulated by the lineages
        that subtend :math:`i` of the :math:`n` lineages at locus :math:`\ell`, whatever they subtend at the other
        locus.

        :param n_samples: Number of trajectories.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: The locus-0 and locus-1 branch lengths, each of shape ``(n_samples, n + 1)``.
        """
        indices = self._get_indices()
        rewards = (
            [CombinedReward([self.reward, TwoLocusSFSReward(0, i)]) for i in indices] +
            [CombinedReward([self.reward, TwoLocusSFSReward(1, j)]) for j in indices]
        )
        sampled = self._sample(n_samples, rewards=rewards, rng=np.random.default_rng(seed))
        n_bins = len(indices)

        left = np.zeros((n_samples, self.lineage_config.n + 1))
        right = np.zeros((n_samples, self.lineage_config.n + 1))
        left[:, 1:1 + n_bins] = sampled[:, :n_bins]
        right[:, 1:1 + n_bins] = sampled[:, n_bins:]

        return left, right

    def sample(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> np.ndarray:
        r"""
        Draw samples of the two-locus site-frequency spectrum. With the branch lengths :math:`L^0_i` and :math:`L^1_j`
        of :meth:`TwoLocusSFSDistribution.sample_per_locus()
        <phasegen.distributions.TwoLocusSFSDistribution.sample_per_locus>`, entry :math:`(i, j)` of a sample is
        :math:`\tfrac{1}{2}(L^0_i L^1_j + L^0_j L^1_i)`.

        :param n_samples: Number of two-locus spectra.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: Array of shape ``(n_samples, n + 1, n + 1)``, whose means over the first axis estimate :attr:`mean`.
        """
        left, right = self.sample_per_locus(n_samples, seed=seed)
        out = np.einsum('ni,nj->nij', left, right)

        return (out + out.transpose(0, 2, 1)) / 2

    def to_empirical(self, n_samples: int, seed: Union[int, np.random.Generator] = None) -> 'EmpiricalTwoLocusSFSDistribution':
        """
        Build an empirical two-locus spectrum from ``n_samples`` trajectories of
        :meth:`TwoLocusSFSDistribution.sample_per_locus()
        <phasegen.distributions.TwoLocusSFSDistribution.sample_per_locus>`, as described at
        :class:`~phasegen.distributions.EmpiricalTwoLocusSFSDistribution`.

        :param n_samples: Number of trajectories.
        :param seed: Integer seed of a :class:`numpy.random.Generator`, or the generator itself. ``None`` draws fresh
            entropy.
        :return: The empirical two-locus spectrum.
        """
        from .empirical import EmpiricalTwoLocusSFSDistribution

        left, right = self.sample_per_locus(n_samples, seed=seed)
        mean = np.einsum('ni,nj->ij', left, right) / n_samples  # non-symmetrized, as in the msprime path

        return EmpiricalTwoLocusSFSDistribution(mean, left=left, right=right)

    @cached_property
    def corr(self) -> TwoLocusSFS:
        r"""
        Pearson correlation between the locus-0 and locus-1 branch lengths,

        .. math::
            \operatorname{Corr}(L^0_i, L^1_j) = \frac{\mathbb{E}[L^0_i L^1_j] - \mathbb{E}[L^0_i]\, \mathbb{E}[L^1_j]}
            {\operatorname{sd}(L^0_i)\, \operatorname{sd}(L^1_j)},

        for all polymorphic bins :math:`(i, j)`, where :math:`\operatorname{sd}` is the standard deviation. This is the
        centered, scale-free companion to the uncentered cross-moment :attr:`mean`. With recombination rate
        :math:`\rho` it reduces to the single-locus SFS correlation as :math:`\rho \to 0`, and for the standard
        coalescent it tends to 0 as :math:`\rho \to \infty`. The per-locus means and variances are the marginals of
        the two-locus space and coincide for the two exchangeable loci.
        """
        indices = self._get_indices()
        n = self.lineage_config.n

        # marginal locus-0 mean and variance per bin (identical for locus 1 by exchangeability, and independent of r)
        mean = {
            i: PhaseTypeDistribution.moment(
                self, k=1, center=False,
                rewards=(CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),)
            )
            for i in indices
        }
        var = {
            i: PhaseTypeDistribution.moment(
                self, k=2, center=True,
                rewards=(CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),) * 2
            )
            for i in indices
        }

        cross = self.mean.data
        out = np.zeros((n + 1, n + 1))
        for i in indices:
            for j in indices:
                denom = np.sqrt(var[i] * var[j])
                if denom > 0:
                    out[i, j] = (cross[i, j] - mean[i] * mean[j]) / denom

        return TwoLocusSFS(out)

