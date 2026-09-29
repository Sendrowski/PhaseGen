"""Site-frequency-spectrum distributions (SFS, folded, joint, two-locus)."""

import logging
from abc import ABC, abstractmethod
from ..caching import cached_property, cache
from typing import List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING
import numpy as np
from ..errors import ModelError
from ..demography import Demography
from ..rewards import Reward, UnfoldedSFSReward, UnitReward, CombinedReward, FoldedSFSReward, SFSReward, JointSFSReward, TwoLocusSFSReward, RestrictedReward
from ..settings import Settings
from ..spectrum import SFS, TwoSFS, JointSFS, TwoLocusSFS
from ..state_space import BlockCountingStateSpace, StateSpace, JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace

from ._common import _make_hashable, _validate_order
from .base import MarginalDensity, MarginalCDF, MarginalQuantileFunction
from .phase_type import PhaseTypeDistribution, TreeHeightDistribution
from .mutation_configs import MutationConfig, MutationLayout, MutationConfigMixin

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from ..visualization import _CurveData
    from .reward import JointRewardDistribution, RewardDistribution
    from .empirical import (
        EmpiricalPhaseTypeSFSDistribution,
        EmpiricalJointSFSDistribution,
        EmpiricalTwoLocusSFSDistribution,
    )

logger = logging.getLogger('phasegen')


class _SFSAggregateFunction:
    """Per-bin SFS function, looping ``SFSDistribution._bin_distribution`` over the polymorphic bins."""

    def __call__(self, t) -> 'SFS | np.ndarray':
        """
        Evaluate the function of every polymorphic bin, each that of the bin's
        :class:`~phasegen.distributions.RewardDistribution` under the spectrum's reward.

        :param t: A point or an array of points, or probability levels for a quantile function.
        :return: For a scalar ``t``, a spectrum with one value per bin. The bins that are zero almost surely, the
            monomorphic and folded-away ones, hold the function of a point mass at zero. For an array, an array of
            shape ``t.shape + (n + 1,)``, which is ``(len(t), n + 1)`` for a 1-D ``t``.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        d = self._distribution
        t_arr = np.asarray(t, dtype=float).ravel()
        out = np.zeros((t_arr.size, d.lineage_config.n + 1))
        if self.kind == 'cdf':
            out[:] = np.where(np.isnan(t_arr), np.nan, t_arr >= 0)[:, None]
        for i in d._get_indices():
            out[:, i] = getattr(d._bin_distribution(i), self.kind)(t_arr)
        return SFS(out[0]) if np.ndim(t) == 0 else out.reshape(np.shape(t) + out.shape[1:])

    def plot(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
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
        :param t: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.
        :param bins: The bins (frequency classes) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(t=t, bins=bins, n_points=n_points), file=file,
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
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, bins=bins, n_points=n_points), file=file,
                                         show=show, clear=clear, label=label, title=title, **kwargs)


class SFSDistribution(MutationConfigMixin, PhaseTypeDistribution, ABC):
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

    def _mutation_class_reward(self, label: 'int | Tuple[str, int]') -> Reward:
        """
        The reward of an elementary frequency class of :meth:`UnfoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>`.

        :param label: The unfolded class :math:`i`, or ``(pop, i)`` for class :math:`i` restricted to the deme
            ``pop``.
        :return: The reward.
        """
        if isinstance(label, tuple):
            pop, i = label
            return RestrictedReward(UnfoldedSFSReward(i), pop=pop)

        return UnfoldedSFSReward(label)

    def mutation_layout(self, folded: bool = False, demes: bool = False) -> MutationLayout:
        r"""
        The layout of the mutational configurations of this spectrum. The elementary classes are the unfolded
        polymorphic classes :math:`i = 1, \dots, n - 1`, labelled ``i``, placed at index ``i`` of an array of length
        :math:`n + 1`. By default, each class is one bin.

        :param folded: Whether to merge the classes :math:`i` and :math:`n - i` into the bin of the smaller one.
        :param demes: Whether to resolve each bin by the deme in which the mutation occurs, with the class labels
            ``(pop, i)`` placed at index ``(p, i)`` of an array of shape :math:`(P, n + 1)` for the deme ``pop`` at
            position :math:`p` of the :math:`P` demes, and the bins ordered by deme and then by class.
        :return: The layout.
        """
        return self._layout_of(int(self.lineage_config.n), list(self.lineage_config.pop_names), folded, demes)

    @staticmethod
    def _layout_of(n: int, pops: List[str], folded: bool, demes: bool) -> MutationLayout:
        """
        The layout of :meth:`UnfoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>` for a sample size and demes.

        :param n: The number of lineages.
        :param pops: The deme names.
        :param folded: Whether to merge the classes :math:`i` and :math:`n - i`.
        :param demes: Whether to resolve each bin by deme.
        :return: The layout.
        """
        indices = list(range(1, n))

        if folded:
            groups = [(i,) if i == n - i else (i, n - i) for i in indices if i <= n - i]
        else:
            groups = [(i,) for i in indices]

        if not demes:
            return MutationLayout(groups, {i: (i,) for i in indices}, (n + 1,), ('class',))

        return MutationLayout(
            [tuple((pop, i) for i in g) for pop in pops for g in groups],
            {(pop, i): (p, i) for p, pop in enumerate(pops) for i in indices},
            (len(pops), n + 1),
            ('deme', 'class')
        )

    def _mutation_start(self, layout: MutationLayout, theta: float) -> MutationConfig:
        r"""
        The configuration from which ``get_mutation_configs()`` climbs to the most probable one,
        :math:`\operatorname{round}(\theta\, \mathbb{E}[\ell_j])`.

        :param layout: The layout.
        :param theta: The mutation rate.
        :return: The configuration.
        """
        if layout == self.mutation_layout():
            mean = np.asarray(self.mean.data)[self._get_indices()]
        else:
            mean = [PhaseTypeDistribution.moment(self, k=1, rewards=(self._bin_reward(b),), center=False)
                    for b in layout.bins]

        return MutationConfig([max(0, int(round(theta * mu))) for mu in mean], layout)

    def _bin_index(self, i: int) -> int:
        """
        Validate a frequency class of the spectrum array.

        :param i: The frequency class, an integer from 0 to :math:`n`.
        :return: The frequency class as an integer.
        :raises ValueError: If ``i`` is not an integer from 0 to :math:`n`.
        """
        n = self.lineage_config.n

        if isinstance(i, bool) or not float(i).is_integer() or not 0 <= i <= n:
            raise ValueError(f"The frequency class must be an integer from 0 to {n}, got {i}.")

        return int(i)

    def _polymorphic_bin(self, i: int) -> int:
        """
        Validate a polymorphic frequency class, one with a branch-length distribution.

        :param i: The frequency class.
        :return: The frequency class as an integer.
        :raises ValueError: If ``i`` is not one of the polymorphic classes of this spectrum.
        """
        indices = self._get_indices()

        if self._bin_index(i) not in indices:
            raise ValueError(
                f"The frequency class must be a polymorphic class from {indices[0]} to {indices[-1]}, got {i}."
            )

        return int(i)

    def _bin_distribution(self, i: int) -> 'RewardDistribution':
        """The reward distribution of SFS bin ``i`` under this spectrum's reward, cached so the expensive cosine / LST
        fit behind its cdf / pdf / quantile is built once and reused across repeated calls and across the three
        curves, rather than rebuilt on every ``sfs.cdf(t)``. Honors :attr:`Settings.cache`."""
        i = self._polymorphic_bin(i)

        cache = self.__dict__.setdefault('_bin_distributions', {})
        if i in cache:
            return cache[i]

        d = self.distribution(reward=CombinedReward([self.reward, self._get_sfs_reward(i)]))
        if Settings.cache:
            cache[i] = d
        return d

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
        :raises ValueError: if ``k`` is not integral or is negative, or if the start time is negative, exceeds
            the end time, or lies beyond the time of almost sure absorption.
        """
        k = _validate_order(k)

        if rewards is None:
            rewards = (self.reward,) * k

        effective_start, effective_end = self._resolve_window(start_time, end_time)

        # batched mean: every bin's mean is ``occupation . r_bin`` with the same occupation-time vector, so the whole
        # spectrum is one contraction instead of a per-bin solve. This is the closed form's spectrum path (it shares
        # the transient solve across bins); only for the plain mean (k=1, default reward, accumulation until
        # absorption). Where flattening applies, the occupation is that of the lineage-counting state space and the
        # bin rewards are flattened onto it, from a zero start time only. Otherwise a non-zero start time is handled by
        # subtracting the occupation up to it (occupation is additive in time). Other cases fall through to the
        # per-bin path.
        flatten = self._flattening_applies(k)
        if (
                Settings.closed_form_last_epoch and
                k == 1 and
                np.isinf(effective_end) and
                rewards == (self.reward,) and
                not (flatten and effective_start > 0)
        ):
            occupation = (self.tree_height if flatten else self)._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
                if effective_start > 0:
                    m = m - self._occupation_times(cap=effective_start)[0]
                bin_rewards = [CombinedReward([self.reward, self._get_sfs_reward(i)]) for i in self._get_indices()]
                R = np.column_stack([
                    (self._flattened_weights(r) if flatten else np.asarray(r._get(self.state_space), dtype=float))
                    for r in bin_rewards
                ])[idx_t]
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
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        r"""
        The :math:`k`-th moment of every bin of the site-frequency spectrum accumulated from the start time
        :math:`t_\mathrm{start}` to each end time :math:`t_\mathrm{end}` in ``end_times``, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t_\mathrm{end}` at which to evaluate the moment.
        :param rewards: Sequence of :math:`k` rewards. By default, the reward of the distribution for each factor.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards. Without averaging, the result
            equals the cross-moment only when all rewards are equal.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :return: Array of shape ``(len(end_times), n + 1)`` of the moments accumulated at the specified times, one
            column per site-frequency count.
        """
        k = _validate_order(k)
        indices = self._get_indices()
        end_times = np.array(list(end_times))

        accumulation = self._accumulate_batched(k, indices, end_times, rewards, start_time)
        if accumulation is None:
            accumulation = np.array([
                self.get_accumulation(k, i, end_times, rewards, center, permute, start_time) for i in indices
            ])

        # pad with zeros
        return np.concatenate([
            np.zeros((1, len(end_times))),
            accumulation,
            np.zeros((self.lineage_config.n - len(indices), len(end_times)))
        ]).T

    def _accumulate_batched(self, k, indices, end_times, rewards, start_time) -> 'np.ndarray | None':
        """Batched mean accumulation (``k == 1``, default reward) contracting ``_mean_occupation_grid`` with the
        stacked bin rewards. Returns ``None`` when not applicable, and the caller evaluates per bin."""
        if k != 1 or (rewards is not None and tuple(rewards) != (self.reward,)) or self._flattening_applies(1):
            return None

        m_grid = self._mean_occupation_grid(end_times, start_time=start_time)  # (len(t), n_states)
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

        k = _validate_order(k)
        end_times = self._default_end_times() if end_times is None else np.asarray(list(end_times), dtype=float)
        rewards = (self.reward,) * k if rewards is None else rewards
        indices = self._get_indices()

        return _CurveData(
            x=end_times,
            y=self.accumulate(k, end_times, rewards, center, permute).T[1:1 + len(indices)],
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
        :raises ValueError: If ``i`` is not a polymorphic frequency class of this spectrum.
        """
        d = self._bin_distribution(i)
        d.label = f"SFS bin {int(i)}"
        return d

    def joint_distribution(self, i: int, j: int) -> 'JointRewardDistribution':
        r"""
        Joint distribution of the branch lengths :math:`L_i` and :math:`L_j` of frequency classes :math:`i` and
        :math:`j` within one genealogy, as a :class:`~phasegen.distributions.JointRewardDistribution`. Both bin rewards
        are combined with the reward of this spectrum, as for :meth:`UnfoldedSFSDistribution.bin()
        <phasegen.distributions.UnfoldedSFSDistribution.bin>`.

        :param i: The first frequency class.
        :param j: The second frequency class.
        :return: The joint distribution of :math:`(L_i, L_j)`.
        :raises ValueError: If ``i`` or ``j`` is not a polymorphic frequency class of this spectrum.
        """
        i, j = self._polymorphic_bin(i), self._polymorphic_bin(j)

        jd = super().joint_distribution(
            CombinedReward([self.reward, self._get_sfs_reward(i)]),
            CombinedReward([self.reward, self._get_sfs_reward(j)])
        )
        jd.label = f"SFS bins ({i}, {j})"
        return jd

    def _plot_data_cdf(self, t: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The CDF curve of each SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param t: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param bins: The bins (frequency classes) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by bin.
        """
        return self._reward_curves('cdf', self._bin_items(bins), t, n_points, 'SFS bin CDFs', 'bin')

    def _plot_data_pdf(self, t: np.ndarray = None, bins: Sequence[int] = None, n_points: int = None) -> '_CurveData':
        """
        The density curve of each SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param t: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param bins: The bins (frequency classes) to include. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by bin.
        """
        return self._reward_curves('pdf', self._bin_items(bins), t, n_points, 'SFS bin PDFs', 'bin')

    def _plot_data_quantile(
            self,
            q: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The quantile curve of each SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param q: Probabilities to evaluate at. By default, an evenly spaced grid from
            ``1 - Settings.plot_endpoint_quantile`` to :attr:`Settings.plot_endpoint_quantile`.
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
            permute: bool = True,
            start_time: float = None
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
        :param start_time: Time from which to accumulate. By default, the start time of the distribution.
        :return: The kth SFS (cross)-moment accumulations at the ith site-frequency count, a float for a single time
            and an array for a sequence of times.
        """
        k = _validate_order(k)

        if rewards is None:
            rewards = [self.reward] * k

        scalar = np.ndim(end_times) == 0

        accumulation = super().accumulate(
            k=k,
            end_times=[end_times] if scalar else end_times,
            rewards=tuple([CombinedReward([r, self._get_sfs_reward(i)]) for r in rewards]),
            center=center,
            permute=permute,
            start_time=start_time
        )

        return float(accumulation[0]) if scalar else accumulation

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

        m, solve, idx_t = two_point
        ss = self.state_space
        indices = self._get_indices()
        R = np.column_stack([
            np.asarray(CombinedReward([self.reward, self._get_sfs_reward(i)])._get(ss), dtype=float)[idx_t]
            for i in indices
        ])

        sfs_matrix = (m[:, None] * R).T @ solve(R)  # R^T diag(m) (-T)^{-1} R (one ordering)
        self._logger.debug("sfs.cov: centering with the outer product of bin means")
        mean = np.asarray(self.mean.data)[indices]
        cov = (sfs_matrix + sfs_matrix.T) - np.outer(mean, mean)

        out = np.zeros((self.lineage_config.n + 1, self.lineage_config.n + 1))
        for a, ia in enumerate(indices):
            out[ia, indices] = cov[a]
        return TwoSFS(out)

    def _cov_stacked(self) -> bool:
        """
        Whether :attr:`cov` evaluates the cross-moments of the bin pairs in closed form, stacked over the first bin.

        :return: Whether the stacked closed form applies.
        """
        _, end_time = self._resolve_window()

        return bool(
            np.isinf(end_time) and Settings.closed_form_last_epoch and self._absorption_certain_in_last_epoch(2)
        )

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

        indices = self._get_indices()
        bin_rewards = [CombinedReward([self.reward, self._get_sfs_reward(i)]) for i in indices]
        start_time, _ = self._resolve_window()

        if self._cov_stacked():
            self._logger.debug("sfs.cov: closed form over %d bin pairs, stacked over the first bin", len(indices) ** 2)
            Reward._check_accumulable(self.state_space, bin_rewards)

            # the cross-moment of each bin pair, one evaluation per second bin
            cross = np.column_stack([
                self._accumulate_closed_form(2, (r, r), start_time, heads=bin_rewards) for r in bin_rewards
            ])

            if np.isnan(cross).any():
                raise ModelError(
                    "NaN value encountered when computing moment. "
                    "This is likely due to an ill-conditioned rate matrix."
                )
        else:
            self._logger.debug("sfs.cov: per-pair matrix exponential over %d bin pairs", len(indices) ** 2)

            # cross-moment of each bin pair (serial)
            cross = np.array([
                [
                    PhaseTypeDistribution.moment(self, k=2, permute=False, center=False, rewards=(r_i, r_j))
                    for r_j in bin_rewards
                ]
                for r_i in bin_rewards
            ])

        # re-structure the results to a matrix form
        sfs = np.zeros((self.lineage_config.n + 1, self.lineage_config.n + 1))
        sfs[np.ix_(indices, indices)] = cross

        # get matrix of marginal moments
        m2 = np.outer(self.mean.data, self.mean.data)

        # calculate covariances
        cov = (sfs + sfs.T) / 2 - m2

        return TwoSFS(cov)

    @cached_property
    def var(self) -> SFS:
        """
        Variance across site-frequency counts, the diagonal of :attr:`cov` where the spectrum-wide evaluation of
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>` applies or a
        cached :attr:`cov` holds the closed form, and the second central moment of each bin otherwise.
        """
        batched = self._cov_batched
        if batched is not None:
            return SFS(np.diag(np.asarray(batched.data)))

        if 'cov' in self.__dict__ and self._cov_stacked():
            return SFS(np.diag(np.asarray(self.cov.data)))

        return self.moment(k=2, center=True)

    def get_cov(self, i: int, j: int) -> float:
        """
        Get the covariance between the ith and jth site-frequency.

        :param i: The ith frequency count
        :param j: The jth frequency count
        :return: The covariance, zero if a class is not polymorphic in this spectrum.
        :raises ValueError: If ``i`` or ``j`` is not an integer from 0 to :math:`n`.
        """
        i, j = self._bin_index(i), self._bin_index(j)

        if i not in self._get_indices() or j not in self._get_indices():
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
        :return: The correlation coefficient, zero if a class is not polymorphic in this spectrum.
        :raises ValueError: If ``i`` or ``j`` is not an integer from 0 to :math:`n`.
        """
        i, j = self._bin_index(i), self._bin_index(j)

        if i not in self._get_indices() or j not in self._get_indices():
            return 0

        return self.get_cov(i, j) / (np.sqrt(self.get_cov(i, i)) * np.sqrt(self.get_cov(j, j)))


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

    def mutation_layout(self, demes: bool = False) -> MutationLayout:
        r"""
        The layout of the mutational configurations of this spectrum, the folded layout
        :meth:`UnfoldedSFSDistribution.mutation_layout(folded=True)
        <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>`, with bin :math:`i` merging the unfolded
        classes :math:`i` and :math:`n - i` for :math:`i = 1, \dots, \lfloor n/2 \rfloor`.

        :param demes: Whether to resolve each bin by the deme in which the mutation occurs.
        :return: The layout.
        """
        return super().mutation_layout(folded=True, demes=demes)


class _JointSFSAggregateFunction:
    """Per-bin joint-SFS function, looping ``JointSFSDistribution._bin_distribution`` over the descendant
    configurations."""

    def __call__(self, t) -> 'JointSFS | np.ndarray':
        """
        Evaluate the function of every joint SFS bin under the spectrum's reward, as for
        :class:`~phasegen.distributions.SFSCDF`.

        :param t: A point or an array of points, or probability levels for a quantile function.
        :return: For a scalar ``t``, a :class:`~sfsutils.spectrum.JointSFS` with one value per descendant
            configuration, where the monomorphic configurations hold the function of a point mass at zero. For an
            array, an array of shape ``t.shape + shape``, with ``shape`` the shape of the joint SFS.
        :raises NotImplementedError: If the coalescent has a bounded accumulation window.
        """
        d = self._distribution
        t_arr = np.asarray(t, dtype=float).ravel()
        out = np.zeros((t_arr.size,) + d.shape)
        if self.kind == 'cdf':
            out[:] = np.where(np.isnan(t_arr), np.nan, t_arr >= 0).reshape((-1,) + (1,) * len(d.shape))
        for config in d._get_configs():
            out[(slice(None),) + tuple(config)] = getattr(d._bin_distribution(config), self.kind)(t_arr)
        if np.ndim(t) == 0:
            return JointSFS(out[0], pop_names=d.lineage_config.pop_names)
        return out.reshape(np.shape(t) + d.shape)

    def plot(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
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
        :param t: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile` quantile.
        :param configs: The joint bins (descendant configurations) to plot. By default, all polymorphic bins.
        :param n_points: Number of points of the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(t=t, configs=configs, n_points=n_points),
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
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
        :param label: Legend label of the curves, ``None`` for the default labels.
        :param title: Plot title, ``None`` for the default title.
        :param kwargs: Line styling passed to the curves, such as ``alpha`` or ``lw``.
        :return: Axes.
        """
        from ..visualization import Visualization

        return Visualization.plot_curves(ax=ax, data=self._plot_data(q=q, configs=configs, n_points=n_points),
                                         file=file, show=show, clear=clear, label=label, title=title, **kwargs)


class JointSFSDistribution(MutationConfigMixin, PhaseTypeDistribution):
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

    def _mutation_class_reward(self, label: Tuple[int, ...]) -> Reward:
        """
        The reward of an elementary frequency class of :meth:`JointSFSDistribution.mutation_layout()
        <phasegen.distributions.JointSFSDistribution.mutation_layout>`.

        :param label: The descendant vector of the joint SFS bin.
        :return: The reward.
        """
        return JointSFSReward(label)

    def mutation_layout(self, folded: bool = False) -> MutationLayout:
        r"""
        The layout of the mutational configurations of this joint spectrum. By default, one bin per polymorphic
        descendant vector :math:`\mathbf{c} = (c_0, \dots, c_{P-1})`, labelled ``c``, in the order of the state
        space's block configurations.

        :param folded: Whether to merge :math:`\mathbf{c}` with its complement :math:`\mathbf{n} - \mathbf{c}`, where
            :math:`\mathbf{n}` holds the sample sizes.
        :return: The layout.
        """
        full = tuple(int(n_p) for n_p in self.lineage_config.lineages)
        configs = self._get_configs()

        if folded:
            groups = []
            for c in configs:
                d = tuple(f - x for f, x in zip(full, c))
                if c <= d:
                    groups.append((c,) if c == d else (c, d))
        else:
            groups = [(c,) for c in configs]

        return MutationLayout(groups, {c: c for c in configs}, self.shape, tuple(self.lineage_config.pop_names))

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
            rewards: Sequence[Reward] = None,
            start_time: float = None,
            end_time: float = None,
            center: bool = True,
            permute: bool = True
    ) -> JointSFS:
        r"""
        The :math:`k`-th moment of every joint site-frequency spectrum bin, central by default, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param rewards: Sequence of :math:`k` rewards, each multiplied with the bin reward. By default, the reward of
            the distribution for each factor.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :param end_time: The end time :math:`t_\mathrm{end}`. By default, the end time of the distribution, or
            absorption.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :return: A joint site-frequency spectrum of shape :attr:`shape` holding the :math:`k`-th moment of each bin.
        :raises ValueError: If the start time is negative, exceeds the end time, or lies beyond the time of almost sure
            absorption, or if the moment is not a number.
        """
        k = _validate_order(k)

        if rewards is None:
            rewards = (self.reward,) * k

        if k == 1 and tuple(rewards) == (self.reward,):
            start, end = self._resolve_window(start_time, end_time)

            # the mean is additive in time, so every bin is the difference of two batched accumulations
            acc = self.accumulate(1, [start, end], start_time=0.0)
            out = acc[1] - acc[0]
        else:
            out = np.zeros(self.shape)
            for config in self._get_configs():
                out[config] = PhaseTypeDistribution.moment(
                    self,
                    k=k,
                    rewards=tuple(CombinedReward([r, JointSFSReward(config)]) for r in rewards),
                    start_time=start_time,
                    end_time=end_time,
                    center=center,
                    permute=permute
                )

        if np.isnan(out).any():
            raise ModelError(
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
        configs = self._get_configs() if configs is None else [self._bin_config(c) for c in configs]

        return [(c, self._bin_distribution(c)) for c in configs]

    def _bin_distribution(self, config: Tuple[int, ...]) -> 'RewardDistribution':
        """
        The distribution of the joint SFS bin ``config`` under this spectrum's reward, cached per bin in
        ``_bin_distributions`` so the cosine fit behind its cdf, pdf and quantile is built once. Honors
        ``Settings.cache``.

        :param config: The descendant configuration, one count per population.
        :return: The distribution of the bin's branch length.
        :raises ValueError: If ``config`` is not the descendant configuration of a polymorphic joint SFS bin.
        """
        config = self._bin_config(config)

        cache = self.__dict__.setdefault('_bin_distributions', {})
        if config in cache:
            return cache[config]

        d = self.distribution(reward=CombinedReward([self.reward, JointSFSReward(config)]))
        if Settings.cache:
            cache[config] = d
        return d

    def _bin_config(self, config: Sequence[int]) -> Tuple[int, ...]:
        """
        Validate the descendant configuration of a polymorphic joint SFS bin.

        :param config: The descendant configuration, one count per population.
        :return: The configuration as a tuple of integers.
        :raises ValueError: If ``config`` is not the descendant configuration of a polymorphic joint SFS bin.
        """
        full = tuple(int(n_p) for n_p in self.lineage_config.lineages)
        config = tuple(config)

        if (
                len(config) != len(full)
                or any(isinstance(c, bool) or not float(c).is_integer() for c in config)
                or not all(0 <= c <= n for c, n in zip(config, full))
                or not any(config)
                or tuple(int(c) for c in config) == full
        ):
            raise ValueError(
                f"The descendant configuration must hold one integer count per population, each from 0 to its sample "
                f"size {full}, and be neither all zero nor {full}, got {config}."
            )

        return tuple(int(c) for c in config)

    def bin(self, *config: int) -> 'RewardDistribution':
        """The distribution of the branch length of the joint SFS bin with the given descendant counts per population,
        as a :class:`~phasegen.distributions.RewardDistribution`, for example ``jsfs.bin(1, 0).quantile(0.9)``. The
        bin reward is combined with the reward of this spectrum, as for :meth:`UnfoldedSFSDistribution.bin()
        <phasegen.distributions.UnfoldedSFSDistribution.bin>`, and the distribution is cached per bin.

        :param config: The descendant configuration, one count per population.
        :return: The distribution of the bin's branch length.
        :raises ValueError: If ``config`` is not the descendant configuration of a polymorphic joint SFS bin.
        """
        config = self._bin_config(config)
        d = self._bin_distribution(config)
        d.label = f"jSFS bin {config}"
        return d

    def joint_distribution(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> 'JointRewardDistribution':
        """
        Joint distribution of the branch lengths of two joint SFS bins within one genealogy, as a
        :class:`~phasegen.distributions.JointRewardDistribution`. Both bin rewards are combined with the reward of this
        spectrum, as for :meth:`JointSFSDistribution.bin() <phasegen.distributions.JointSFSDistribution.bin>`.

        :param config_a: The first descendant configuration, one count per population.
        :param config_b: The second descendant configuration.
        :return: The joint distribution of the two branch lengths.
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        config_a, config_b = self._bin_config(config_a), self._bin_config(config_b)

        jd = super().joint_distribution(
            CombinedReward([self.reward, JointSFSReward(config_a)]),
            CombinedReward([self.reward, JointSFSReward(config_b)])
        )
        jd.label = f"jSFS bins {config_a} x {config_b}"
        return jd

    def _plot_data_cdf(
            self,
            t: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The CDF curve of each joint SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param t: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param configs: The joint bins (descendant configurations) to include. By default, all of them.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by configuration.
        """
        return self._reward_curves('cdf', self._config_items(configs), t, n_points, 'Joint SFS bin CDFs', 'config')

    def _plot_data_pdf(
            self,
            t: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The density curve of each joint SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param t: Points to evaluate at. By default, an evenly spaced grid up to the largest bin's
            :attr:`Settings.plot_endpoint_quantile` quantile.
        :param configs: The joint bins (descendant configurations) to include. By default, all of them.
        :param n_points: Number of points of the default grid.
        :return: The curves, labelled by configuration.
        """
        return self._reward_curves('pdf', self._config_items(configs), t, n_points, 'Joint SFS bin PDFs', 'config')

    def _plot_data_quantile(
            self,
            q: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None
    ) -> '_CurveData':
        """
        The quantile curve of each joint SFS bin (see :meth:`PhaseTypeDistribution._reward_curves`).

        :param q: Probabilities to evaluate at. By default, an evenly spaced grid from
            ``1 - Settings.plot_endpoint_quantile`` to :attr:`Settings.plot_endpoint_quantile`.
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
            permute: bool = True,
            start_time: float = None
    ) -> np.ndarray:
        r"""
        The :math:`k`-th moment of every joint site-frequency spectrum bin accumulated from the start time
        :math:`t_\mathrm{start}` to each end time :math:`t_\mathrm{end}` in ``end_times``, as described in
        :meth:`PhaseTypeDistribution.moment() <phasegen.distributions.PhaseTypeDistribution.moment>`.

        :param k: The order :math:`k` of the moment.
        :param end_times: The end times :math:`t_\mathrm{end}` at which to evaluate the moment.
        :param center: Whether to return the central moment.
        :param permute: Whether to average over the :math:`k!` orderings of the rewards.
        :param start_time: The start time :math:`t_\mathrm{start}`. By default, the start time of the distribution.
        :return: Array of shape ``(len(end_times),) +`` :attr:`shape` with the moment of each bin over time.
        """
        k = _validate_order(k)
        configs = self._get_configs()
        end_times = np.array(list(end_times))

        # batched mean accumulation (k=1): all configs share the occupation-up-to-t grid m(t), so the whole joint
        # accumulation is one contraction m_grid @ R over the stacked config rewards
        if k == 1 and not self._flattening_applies(1):
            m_grid = self._mean_occupation_grid(end_times, start_time=start_time)
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
                    permute=permute,
                    start_time=start_time
                )
                for config in configs
            ])

        out = np.zeros((len(end_times),) + self.shape)
        for config, acc in zip(configs, accumulation):
            out[(slice(None),) + config] = acc

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

        k = _validate_order(k)
        end_times = self._default_end_times() if end_times is None else np.asarray(list(end_times), dtype=float)
        configs = self._get_configs()
        accumulation = self.accumulate(k, end_times, center=center, permute=permute)

        return _CurveData(
            x=end_times,
            y=np.array([accumulation[(slice(None),) + config] for config in configs]).reshape(
                len(configs), len(end_times)
            ),
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
        :param clear: Whether to draw on a new figure when ``ax`` is not given, otherwise onto the current axes.
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
        :raises ValueError: If a configuration is not the descendant configuration of a polymorphic joint SFS bin.
        """
        return PhaseTypeDistribution.moment(
            self,
            k=2,
            center=True,
            rewards=tuple(
                CombinedReward([self.reward, JointSFSReward(self._bin_config(c))]) for c in (config_a, config_b)
            )
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

        m, solve, idx_t = two_point
        ss = self.state_space
        configs = self._get_configs()
        R = np.column_stack([
            np.asarray(CombinedReward([self.reward, JointSFSReward(config)])._get(ss), dtype=float)[idx_t]
            for config in configs
        ])

        sfs_matrix = (m[:, None] * R).T @ solve(R)  # R^T diag(m) (-T)^{-1} R (one ordering)
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


class TwoLocusSFSDistribution(MutationConfigMixin, PhaseTypeDistribution):
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

    def _mutation_class_reward(self, label: Tuple[int, int]) -> Reward:
        """
        The reward of an elementary frequency class of :meth:`TwoLocusSFSDistribution.mutation_layout()
        <phasegen.distributions.TwoLocusSFSDistribution.mutation_layout>`.

        :param label: The pair ``(locus, i)`` of frequency class :math:`i` at a locus.
        :return: The reward.
        """
        return TwoLocusSFSReward(*label)

    def mutation_layout(self, loci: Sequence[int] = (0, 1), folded: bool = False) -> MutationLayout:
        """
        The layout of the mutational configurations of the two loci. By default, one bin per locus and polymorphic
        frequency class :math:`i`, labelled ``(locus, i)`` and ordered by locus and then by class. The mutations of
        the loci left out are not counted, so the probabilities are marginal over them.

        :param loci: The loci whose mutations are counted.
        :param folded: Whether to merge the classes :math:`i` and :math:`n - i` of a locus into one bin.
        :return: The layout.
        :raises ValueError: If ``loci`` is empty, repeats a locus or holds a locus other than 0 and 1.
        """
        loci = tuple(loci)

        if not loci or len(set(loci)) != len(loci) or not set(loci) <= {0, 1}:
            raise ValueError(f"The loci must be distinct entries of (0, 1), got {loci}.")

        n = int(self.lineage_config.n)
        indices = self._get_indices()

        if folded:
            groups = [(i,) if i == n - i else (i, n - i) for i in indices if i <= n - i]
        else:
            groups = [(i,) for i in indices]

        return MutationLayout(
            [tuple((locus, i) for i in g) for locus in loci for g in groups],
            {(locus, i): (locus, i) for locus in (0, 1) for i in indices},
            (2, n + 1),
            ('locus', 'class')
        )

    def _no_univariate_distribution(self, *args, **kwargs) -> None:
        """A two-locus SFS entry ``(i, j)`` is the cross-moment ``E[L^0_i · L^1_j]`` — a product of two distinct
        branch lengths — so it has no single univariate distribution to invert. The marginal per-locus branch-length
        distributions are the ordinary single-locus SFS bin distributions (``pg.Coalescent(...).sfs``)."""
        raise NotImplementedError(
            "A two-locus SFS entry (i, j) is a cross-moment E[L^0_i . L^1_j] (a product of two rewards), so it has "
            "no single univariate CDF/PDF/quantile. For the marginal branch-length distribution of a frequency "
            "class, use the single-locus spectrum: pg.Coalescent(...).sfs.cdf / .pdf and their .plot()."
        )

    plot_cdf = plot_pdf = bin = _no_univariate_distribution
    cdf = pdf = quantile = property(_no_univariate_distribution)

    def _unsupported(self, *args, **kwargs) -> None:
        """
        Reject a member of :class:`~phasegen.distributions.PhaseTypeDistribution` that the two-locus spectrum does
        not provide.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{type(self).__name__} provides mean, corr, joint_distribution, sample, sample_per_locus and "
            "to_empirical. Higher moments of a pair of frequency classes are available from joint_distribution(i, j), "
            "and the per-locus marginals from the single-locus spectrum pg.Coalescent(...).sfs."
        )

    moment = accumulate = plot_accumulation = distribution = _unsupported
    var = std = m2 = loci = demes = property(_unsupported)

    def _polymorphic_class(self, i: int) -> int:
        """
        Validate a polymorphic frequency class.

        :param i: The frequency class.
        :return: The frequency class as an integer.
        :raises ValueError: If ``i`` is not an integer from 1 to :math:`n - 1`.
        """
        n = self.lineage_config.n

        if isinstance(i, bool) or not float(i).is_integer() or not 1 <= i <= n - 1:
            raise ValueError(f"The frequency class must be a polymorphic class from 1 to {n - 1}, got {i}.")

        return int(i)

    def joint_distribution(self, i: int, j: int) -> 'JointRewardDistribution':
        r"""
        Joint distribution of the branch length :math:`L^0_i` of frequency class :math:`i` at locus 0 and the branch
        length :math:`L^1_j` of frequency class :math:`j` at locus 1, as a
        :class:`~phasegen.distributions.JointRewardDistribution`. Both locus rewards are combined with the reward of
        this spectrum, as for
        :attr:`TwoLocusSFSDistribution.mean <phasegen.distributions.TwoLocusSFSDistribution.mean>`.

        :param i: The locus-0 frequency class.
        :param j: The locus-1 frequency class.
        :return: The joint distribution of :math:`(L^0_i, L^1_j)`.
        :raises ValueError: If ``i`` or ``j`` is not a polymorphic class from 1 to :math:`n - 1`.
        """
        i, j = self._polymorphic_class(i), self._polymorphic_class(j)

        jd = PhaseTypeDistribution.joint_distribution(
            self,
            CombinedReward([self.reward, TwoLocusSFSReward(0, i)]),
            CombinedReward([self.reward, TwoLocusSFSReward(1, j)])
        )
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
        Batched mean two-locus SFS, contracting ``_two_point_occupation`` with the stacked per-locus bin rewards as in
        ``PhaseTypeDistribution.moment``.

        :return: The mean two-locus SFS, or ``None`` when not applicable, and the caller evaluates per pair.
        """
        two_point = self._two_point_occupation()
        if two_point is None:
            return None

        m, solve, idx_t = two_point
        ss = self.state_space
        indices = self._get_indices()
        R0 = np.column_stack([
            np.asarray(CombinedReward([self.reward, TwoLocusSFSReward(0, i)])._get(ss), dtype=float)[idx_t]
            for i in indices
        ])
        R1 = np.column_stack([
            np.asarray(CombinedReward([self.reward, TwoLocusSFSReward(1, j)])._get(ss), dtype=float)[idx_t]
            for j in indices
        ])

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

