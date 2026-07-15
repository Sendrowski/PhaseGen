"""Site-frequency-spectrum distributions (SFS, folded, joint, two-locus)."""

import heapq
import itertools
import logging
from abc import ABC, abstractmethod
from ..caching import cached_property, cache
from typing import List, Tuple, Iterable, Iterator, Optional, Sequence, Set, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp
from ..demography import Demography
from ..expm import Backend
from ..rewards import Reward, TreeHeightReward, UnfoldedSFSReward, UnitReward, CombinedReward, FoldedSFSReward, SFSReward, JointSFSReward, TwoLocusSFSReward
from ..settings import Settings
from ..spectrum import SFS, SFS2, JointSFS, TwoLocusSFS
from ..state_space import BlockCountingStateSpace, StateSpace, JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace
from ..utils import multiset_permutations

from ._common import _make_hashable
from .base import MarginalDensity, MarginalCDF, MarginalQuantileFunction
from .phase_type import PhaseTypeDistribution, TreeHeightDistribution

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .reward import JointRewardDistribution, RewardDistribution
    from .empirical import (
        EmpiricalPhaseTypeSFSDistribution,
        EmpiricalJointSFSDistribution,
        EmpiricalTwoLocusSFSDistribution,
    )

expm = Backend.expm
logger = logging.getLogger('phasegen')


class _SFSAggregateFunction:
    """Mixin: a per-bin SFS function object evaluates by looping the spectrum's frequency classes -- each a single-
    reward :class:`RewardDistribution` -- and stacking their cdf / pdf / quantile (selected by :attr:`kind`) into an
    :class:`SFS` (one value per class; the monomorphic edges stay 0). A scalar argument returns an :class:`SFS`; an
    array returns a ``(len(t), n + 1)`` stack. The spectrum it hangs off supplies the per-bin distributions."""

    def __call__(self, t) -> 'SFS | np.ndarray':
        d = self._distribution
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros((t_arr.size, d.lineage_config.n + 1))
        for i in d._get_indices():
            out[:, i] = getattr(d._bin_distribution(i), self.kind)(t_arr)
        return SFS(out[0]) if np.ndim(t) == 0 else out


class SFSDensity(_SFSAggregateFunction, MarginalDensity):
    """Per-bin SFS densities -- the density of each frequency class's branch length, one curve per bin.

    - **Callable** ``pdf(t)``: every bin's ``pdf(t)``, the derivative of that bin's cosine CDF grid.
    - **Plot** ``pdf.plot()``: overlays those same curves, one per polymorphic bin.
    """


class SFSCDF(_SFSAggregateFunction, MarginalCDF):
    """Per-bin SFS cumulative distribution functions -- the probability each frequency class's branch length is at
    most ``t``, one curve per bin.

    - **Callable** ``cdf(t)``: every bin's ``cdf(t)``, read off that bin's cosine CDF grid.
    - **Plot** ``cdf.plot()``: overlays those same curves, one per polymorphic bin.
    """


class SFSQuantileFunction(_SFSAggregateFunction, MarginalQuantileFunction):
    """Per-bin SFS quantile functions, one per frequency class (the inverse CDF of each bin's branch length).

    - **Callable** ``quantile(q)``: every bin's ``quantile(q)`` -- the inverse interpolation of that bin's cosine CDF
      grid, handing over to the de Hoog bisection above :attr:`~phasegen.settings.Settings.dehoog_tail_quantile`.
    - **Plot** ``quantile.plot()``: overlays those same curves, one per bin.
    """


class SFSDistribution(PhaseTypeDistribution, ABC):
    """
    Base class for site-frequency spectrum distributions.

    The spectrum-wide moment accessors (:attr:`mean`, :attr:`cov`) share a single occupation-time solve across all
    bins rather than solving each bin separately.
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

        #: Generated probability mass by iterator returned from :meth:`get_mutation_configs`.
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
        """
        Get the kth moments of the site-frequency spectrum.

        The plain mean (``k = 1``, default reward) is computed once for the whole spectrum as a single occupation-time
        contraction shared across bins, rather than a separate solve per bin; other moments fall through to the
        per-bin path.

        :param k: The order of the moment
        :param rewards: Sequence of k rewards
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
        :return: A site-frequency spectrum of kth order moments.
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
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
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

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Draw samples of the site-frequency spectrum by simulating trajectories. Each sampled trajectory yields the
        branch lengths subtending every (polymorphic) frequency class; the monomorphic edge bins are zero.

        :param n_samples: Number of spectra to sample.
        :return: Array of shape ``(n_samples, n + 1)`` whose per-sample mean equals :attr:`mean`.
        """
        indices = self._get_indices()
        rewards = [CombinedReward([self.reward, self._get_sfs_reward(i)]) for i in indices]
        sampled = self._sample(n_samples, rewards=rewards)

        out = np.zeros((n_samples, self.lineage_config.n + 1))
        out[:, 1:1 + len(indices)] = sampled

        return out

    def to_empirical(self, n_samples: int) -> 'EmpiricalPhaseTypeSFSDistribution':
        """
        Build an empirical (sample-based) SFS counterpart by simulating ``n_samples`` trajectories, broken down per
        deme. Single-locus only (``LocusReward`` is unsupported on the block-counting state space).

        :param n_samples: Number of trajectories to simulate.
        :return: An :class:`~phasegen.distributions.empirical.EmpiricalPhaseTypeSFSDistribution`.
        """
        from .empirical import EmpiricalPhaseTypeSFSDistribution

        if self.locus_config.n != 1:
            raise NotImplementedError("Sampled SFS is only available for single-locus scenarios.")

        pops = self.lineage_config.pop_names
        n = self.lineage_config.n
        indices = self._get_indices()

        # stacked rewards over (deme, polymorphic bin); one sampling pass yields the full per-deme spectrum
        rewards = [CombinedReward([self.demes[pop].reward, self._get_sfs_reward(i)]) for pop in pops for i in indices]
        sampled = self._sample(n_samples, rewards=rewards).reshape(n_samples, len(pops), len(indices))

        # (loci=1, demes, samples, n + 1); the polymorphic bins scatter into their index positions
        branch_lengths = np.zeros((1, len(pops), n_samples, n + 1))
        for bi, i in enumerate(indices):
            branch_lengths[0, :, :, i] = sampled[:, :, bi].T

        # no mutations sampled by default; the polymorphic-only shape mirrors the msprime path (``mutations.T[1:-1].T``)
        mutations = np.zeros((1, len(pops), n_samples, n - 1))

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=branch_lengths,
            mutations=mutations,
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
        """
        Evaluate the kth (non-central) moments for site-frequency spectrum at different end times.

        :param k: The order of the moment.
        :param end_times: Times or time when to evaluate the moment.
        :param rewards: Sequence of k rewards. By default, the reward of the underlying distribution.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
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
        """Batched *mean* accumulation (``k == 1``, default reward): every bin shares the occupation-up-to-t vector
        ``m(t)``, so the whole spectrum's accumulation is ``m_grid @ R`` over the stacked bin rewards instead of a
        per-bin solve. Returns ``None`` (caller falls back to the per-bin path) when not applicable."""
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
        Plot accumulation of (non-central) SFS moments at different times.

        .. note:: This is different from a CDF, as it shows the accumulation of moments rather than the probability
            of having reached absorption at a certain time.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. Defaults to a grid over
            :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
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
        import matplotlib.pyplot as plt
        from ..visualization import Visualization

        k = int(k)

        if ax is None:
            ax = plt.gca()

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(Settings.plot_endpoint_quantile),
                                    Settings.plot_n_grid)

        if rewards is None:
            rewards = (self.reward,) * k

        if title is None:
            title = (f"SFS Moment accumulation "
                     f"({', '.join(r.__class__.__name__.replace('Reward', '') for r in rewards)})")

        # get accumulation of moments
        accumulation = self.accumulate(k, end_times, rewards, center, permute)

        for i, acc in zip(self._get_indices(), accumulation[1: -1]):
            Visualization.plot(
                ax=ax,
                x=end_times,
                y=acc,
                xlabel='t',
                ylabel='moment',
                label=f'{i}',
                file=file,
                show=i == self._get_indices()[-1] and show,
                clear=clear,
                title=title
            )

        return ax

    def _bin_distribution_items(self, bins: Sequence[int]) -> List[Tuple[int, SFSReward]]:
        """``(bin, reward)`` pairs for the requested bins (all polymorphic bins by default)."""
        indices = list(self._get_indices()) if bins is None else [int(b) for b in np.atleast_1d(bins)]
        return [(i, self._get_sfs_reward(i)) for i in indices]

    def bin(self, i: int) -> 'RewardDistribution':
        """The 1D distribution of bin ``i``'s branch length ``L_i`` — a callable-and-plottable
        :class:`RewardDistribution` (e.g. ``sfs.bin(2).pdf.plot()``, ``sfs.bin(2).quantile(0.9)``).

        :param i: The frequency class.
        :return: The accumulated-reward distribution of ``L_i``.
        """
        d = self.distribution(reward=self._get_sfs_reward(i))
        d.label = f"SFS bin {i}"
        return d

    def joint_distribution(self, i: int, j: int) -> 'JointRewardDistribution':
        """The joint distribution of the branch lengths of bins ``i`` and ``j`` *within a tree* (the within-tree
        2-SFS / ``cov`` cross-moment as a bivariate distribution). See :class:`RewardDistribution`'s joint variant.

        :param i: The first frequency class.
        :param j: The second frequency class.
        :return: The joint accumulated-reward distribution of ``(L_i, L_j)``.
        """
        jd = super().joint_distribution(self._get_sfs_reward(i), self._get_sfs_reward(j))
        jd.label = f"SFS bins ({i}, {j})"
        return jd

    def _plot_cdf(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = 'SFS bin CDFs'
    ) -> 'plt.Axes':
        """
        Plot the cumulative distribution function of every SFS bin at once.

        :param ax: Axes to plot on.
        :param x: Values to evaluate the CDFs at. By default, an evenly spaced grid up to the largest bin's support.
        :param bins: The bins (frequency classes) to plot. By default, all of them.
        :param n_points: Number of evaluation points for the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        return self._plot_reward_curves('cdf', self._bin_distribution_items(bins), ax, x, n_points, show, file,
                                        clear, title)

    def _plot_pdf(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = 'SFS bin PDFs',
    ) -> 'plt.Axes':
        """
        Plot the probability density function of every SFS bin at once.

        :param ax: Axes to plot on.
        :param x: Values to evaluate the PDFs at. By default, an evenly spaced grid up to the largest bin's support.
        :param bins: The bins (frequency classes) to plot. By default, all of them.
        :param n_points: Number of evaluation points for the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        return self._plot_reward_curves('pdf', self._bin_distribution_items(bins), ax, x, n_points, show, file,
                                        clear, title)

    def _plot_quantile(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            bins: Sequence[int] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = 'SFS bin quantile functions',
    ) -> 'plt.Axes':
        """
        Plot the quantile function of every SFS bin at once (bin branch length versus probability ``q``).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate the quantiles at. By default, an evenly spaced grid in ``(0, 1)``.
        :param bins: The bins (frequency classes) to plot. By default, all of them.
        :param n_points: Number of evaluation points for the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        return self._plot_reward_curves('quantile', self._bin_distribution_items(bins), ax, q, n_points, show, file,
                                        clear, title)

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
        :param permute: For cross-moments, whether to average over all permutations of rewards. Default is ``True``,
            which will provide the correct cross-moment. If set to ``False``, the cross-moment will be conditioned on
            the order of rewards.
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

    def _cov_batched(self) -> Optional[SFS2]:
        """
        Batched 2-SFS: all ``O(n^2)`` bin pairs share one two-point occupation operator ``K`` (see
        :meth:`_two_point_occupation`), so the whole covariance is ``cov = R^T (K + K^T) R - outer(mean)`` via a
        single contraction over the stacked bin rewards instead of a cross-moment per pair.

        :return: The covariance, or ``None`` when not applicable (closed form disabled, explicit end time, or
            absorption not almost sure) so the caller falls back to the per-pair path.
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
        return SFS2(out)

    @cached_property
    def cov(self) -> SFS2:
        """
        Covariance matrix across site-frequency counts.
        """
        batched = self._cov_batched()
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

        return SFS2(cov)

    @cached_property
    def var(self) -> SFS:
        """
        Variance across site-frequency counts. When the closed form applies this is the diagonal of the batched
        :attr:`cov` (one shared two-point occupation solve for the whole spectrum); otherwise it falls back to the
        per-bin central moment, which is cheaper than the per-pair covariance the fallback would otherwise build.
        """
        batched = self._cov_batched()
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
    def corr(self) -> SFS2:
        """
        Correlation matrix across site-frequency counts.
        """
        # get standard deviations
        std = np.sqrt(self.var.data)

        # monomorphic bins have zero variance; the resulting NaNs from dividing by a zero std are expected and
        # replaced with zeros below, so silence the benign divide warning at the source.
        with np.errstate(divide='ignore', invalid='ignore'):
            sfs = SFS2(self.cov.data / np.outer(std, std))

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
        """
        Get transition matrix for mutational configuration probabilities.

        :param n: The number of frequency bins.
        :param theta: The mutation rate.
        :return: Transition matrix and exit vector.
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

    def get_mutation_config(self, config: Sequence[int], theta: float) -> float:
        """
        Get the probabilities of observing the given mutational configurations according to the infinite sites model.

        .. note::
            This supports piecewise time-homogeneous demography (any number of epochs). Recombination is not
            supported.

        :param config: The mutational configuration. A sequence of integers of length n - 1 for unfolded configurations
            and n // 2 for folded configurations, where n is the number of
            lineages. Each element in the sequence is an integer representing the number of mutations
            at each frequency count starting from 1. For example, the unfolded configuration [2, 1, 0] represents two
            singleton, one doubleton and zero tripleton mutations for a sample size of 4 lineages. Similarly, the
            folded configuration [2, 1] represents two singleton or tripleton and one doubleton mutation for the same
            number of lineages.
        :param theta: The mutation rate.
        :return: The probability of observing the given mutational configuration.
        """
        # make sure theta is non-negative
        if theta < 0:
            raise ValueError("Theta must be greater than or equal to 0.")

        # number of frequency bins
        n = len(self._get_configs(self.lineage_config.n, 0)[0])

        if len(config) != n:
            raise ValueError(
                "The length of the configuration must be equal to the number of frequency bins. "
                f"Expected {n}, got {len(config)}."
            )

        # explicitly convert to tuple of integers
        config = tuple(int(c) for c in config)

        # handle special case when theta = 0
        if theta == 0:
            if sum(config) == 0:
                return 1

            return 0

        # the single-epoch resolvent integrates the inter-mutation waiting time in closed form, which requires a
        # constant rate matrix; for several epochs the augmented process is integrated epoch by epoch instead
        if self.demography.has_n_epochs(2):
            return self._get_mutation_config_inhomogeneous(config, n, theta)

        return self._get_mutation_config_homogeneous(config, n, theta)

    def _get_mutation_config_homogeneous(self, config: Tuple[int, ...], n: int, theta: float) -> float:
        """
        Mutational-configuration probability for a single (time-homogeneous) epoch, summing the embedded
        jump-chain transition matrices :meth:`_get_P` over all orderings of the mutation events.

        :param config: The mutational configuration as a tuple of integers, one per frequency bin.
        :param n: The number of frequency bins.
        :param theta: The mutation rate.
        :return: The probability of observing the given mutational configuration.
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
        Configuration-independent inputs to the multi-epoch mutational-configuration probability, computed once and
        reused across all configurations (the descending iterator evaluates many against the same demography): the
        non-absorbing mask, the per-bin SFS reward vectors and their total, the initial distribution, and for each
        epoch the (dense) sub-intensity matrix, the coalescent absorption-rate vector and the duration (``None`` for
        the final, unbounded epoch).

        :return: ``(non_absorbing, R, r_total, alpha, epochs)`` where ``epochs`` is a list of ``(S, e, tau)``.
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
        Mutational-configuration probability for piecewise time-homogeneous demography.

        Conditional on the coalescent tree the class-``i`` mutation count is Poisson with mean ``theta * ell_i``,
        where ``ell_i`` is the ``i``-ton branch length, so the probability of the configuration ``k`` is the
        expectation over the tree of the product of these Poisson masses. This is evaluated with an augmented killed
        process on (phase, mutation-count lattice), the lattice node ``c`` ranging over ``0 <= c_i <= k_i``:

        - diagonal block ``(c, c)``: the epoch sub-generator ``S_j`` minus ``theta * diag(sum_i R_i)``;
        - super-diagonal block ``c -> c + e_i`` (only while ``c_i < k_i``): ``theta * diag(R_i)``.

        A class-``i`` mutation at the cap ``c_i = k_i`` leaks out (killed), which realises the ``exp(-theta ell_i)``
        factor and pins the count to exactly ``k_i``. The process is propagated through each epoch with its matrix
        exponential, accumulating the mass absorbed at the top lattice node ``k`` in every epoch (the most recent
        common ancestor can be reached in any epoch, not only the last); the final unbounded epoch is integrated to
        absorption with the resolvent. For a single epoch this reduces to :meth:`_get_mutation_config_homogeneous`.

        :param config: The mutational configuration as a tuple of integers, one per frequency bin.
        :param n: The number of frequency bins.
        :param theta: The mutation rate.
        :return: The probability of observing the given mutational configuration.
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

    def get_mutation_configs_by_count(self, theta: float) -> Iterator[Tuple[Tuple[float, ...], float]]:
        """
        An iterator over the probabilities of observing mutational configurations according to the infinite sites
        model, generated in ascending order of the number of mutations. Unlike the default :meth:`get_mutation_configs`
        (descending probability), this order is deterministic and independent of the configuration probabilities, but
        reaches a given probability mass only after evaluating every configuration up to the truncating mutation count.
        See :meth:`get_mutation_config` for more information on mutational configurations.

        .. note::
            This supports piecewise time-homogeneous demography (any number of epochs); recombination is not
            supported. Also note that the number of configurations is infinite, so this iterator will never stop.
            However, depending on the mutation rate, the probability of observing configurations of higher mutation
            counts will decrease over time. You can keep track of the generated probability mass by checking the
            :attr:`~.generated_mass` attribute, which is reset every time this method is called.

        :param theta: The mutation rate.
        :return: An iterator over the probabilities of observing mutational configurations.
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
        An iterator over the probabilities of observing mutational configurations according to the infinite sites
        model, generated in descending order of probability. A target probability mass is thus reached after
        evaluating far fewer configurations than the ascending-count :meth:`get_mutation_configs_by_count`, which is
        particularly valuable for piecewise time-homogeneous demography, where each configuration probability is more
        expensive to evaluate, and for heavy-tailed spectra (high mutation rate, growth).
        See :meth:`get_mutation_config` for more information on mutational configurations.

        The search is seeded at the modal configuration ``round(theta * E[ell_i])``, where the expected
        ``i``-ton branch length ``E[ell_i]`` is available in closed form (also across epochs); the seed is
        refined by a local hill-climb and the lattice is then expanded outward with a max-priority queue. The
        ordering is exact when the configuration probability is unimodal along each axis (the typical case);
        the accumulated :attr:`~.generated_mass` is exact irrespective of the ordering, since the probabilities
        over all configurations sum to one.

        .. note::
            This supports piecewise time-homogeneous demography (any number of epochs); recombination is not
            supported. As the number of configurations is infinite, the iterator does not stop on its own; consume
            it until :attr:`~.generated_mass` exceeds a threshold.

        Code example:

        ::

            coal = pg.Coalescent(n=5)

            it = coal.sfs.get_mutation_configs(theta=1)

            # continue until generated mass is above 0.8
            samples = list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))

        :param theta: The mutation rate.
        :return: An iterator over the probabilities of observing mutational configurations.
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
    Subclasses supply :meth:`_tajima_n`, :meth:`_tajima_mean` and :meth:`_tajima_cov`.
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
        Mean pairwise diversity :math:`\pi = \sum_i \frac{2 i (n - i)}{n (n - 1)} \mathbb{E}[L_i]`, the branch-length
        estimator of :math:`\theta` based on the expected number of pairwise differences.
        """
        w_pi, _ = self._tajima_weights

        return float(w_pi @ self._tajima_mean())

    @cached_property
    def theta_w(self) -> float:
        r"""
        Watterson's estimator :math:`\theta_W = L_\text{total} / a_n` with :math:`a_n = \sum_{k=1}^{n-1} 1/k`, the
        branch-length estimator of :math:`\theta` based on the total branch length.
        """
        _, w_w = self._tajima_weights

        return float(w_w @ self._tajima_mean())

    @cached_property
    def tajimas_d(self) -> float:
        r"""
        Tajima's :math:`D` in branch form: :math:`D = (\pi - \theta_W) / \sqrt{c^\top \, \mathrm{Cov}[L] \, c}`
        with weights :math:`c_i = \frac{2 i (n - i)}{n (n - 1)} - 1/a_n`. It is ``0`` under the standard neutral
        constant-size model, negative under population growth (excess of low-frequency variants) and positive under
        contraction. The normalization uses the branch-length covariance rather than the mutation-based variance of
        the classical sample estimator.
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
    """Mixin: a per-bin joint-SFS function object evaluates by looping the spectrum's descendant configurations --
    each a single-reward :class:`RewardDistribution` -- and stacking their cdf / pdf / quantile (selected by
    :attr:`kind`) into a :class:`JointSFS` (one value per configuration; monomorphic bins 0). A scalar argument
    returns a :class:`JointSFS`; an array returns a ``(len(t),) + shape`` stack."""

    def __call__(self, t) -> 'JointSFS | np.ndarray':
        d = self._distribution
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros((t_arr.size,) + d.shape)
        for config in d._get_configs():
            bin_dist = d.distribution(reward=JointSFSReward(config))
            out[(slice(None),) + config] = [getattr(bin_dist, self.kind)(float(v)) for v in t_arr]
        return JointSFS(out[0], pop_names=d.lineage_config.pop_names) if np.ndim(t) == 0 else out


class JointSFSDensity(_JointSFSAggregateFunction, MarginalDensity):
    """Per-bin joint-SFS densities (one per descendant configuration). See :class:`_JointSFSAggregateFunction`."""


class JointSFSCDF(_JointSFSAggregateFunction, MarginalCDF):
    """Per-bin joint-SFS cumulative distribution functions (one per descendant configuration)."""


class JointSFSQuantileFunction(_JointSFSAggregateFunction, MarginalQuantileFunction):
    """Per-bin joint-SFS quantile functions (one per descendant configuration)."""


class JointSFSDistribution(PhaseTypeDistribution):
    """
    Joint (multi-population) site-frequency spectrum distribution.

    Moments are returned as a multi-dimensional array of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, where ``n_p`` is the
    sample size of population ``p``. The entry at index ``(k_0, ..., k_{P-1})`` is the moment for branches subtending
    exactly ``k_p`` samples from population ``p``. The monomorphic bins (the all-zero and the full
    ``(n_0,...,n_{P-1})`` configuration) are zero by convention.

    The spectrum-wide moment accessors (:attr:`mean`, :attr:`var`, :attr:`cov`) share a single occupation-time solve
    across all bins rather than solving each bin separately.
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

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Draw samples of the joint site-frequency spectrum by simulating trajectories. Each sample is an array of
        shape :attr:`shape` holding the branch length subtending every (polymorphic) descendant configuration.

        :param n_samples: Number of joint spectra to sample.
        :return: Array of shape ``(n_samples, *shape)`` whose per-sample mean equals :meth:`moment` (k=1).
        """
        configs = self._get_configs()
        rewards = [CombinedReward([self.reward, JointSFSReward(c)]) for c in configs]
        sampled = self._sample(n_samples, rewards=rewards)

        out = np.zeros((n_samples,) + self.shape)
        for j, config in enumerate(configs):
            out[(slice(None),) + config] = sampled[:, j]

        return out

    def to_empirical(self, n_samples: int) -> 'EmpiricalJointSFSDistribution':
        """
        Build an empirical (sample-based) joint SFS counterpart by simulating ``n_samples`` trajectories.

        :param n_samples: Number of trajectories to simulate.
        :return: An :class:`~phasegen.distributions.empirical.EmpiricalJointSFSDistribution`.
        """
        from .empirical import EmpiricalJointSFSDistribution, MsprimeCoalescent

        samples = self.sample(n_samples)  # (n_samples, *shape)

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
        """
        Get the kth moments of the joint site-frequency spectrum.

        The plain mean (``k = 1``) is computed once for the whole spectrum as a single occupation-time contraction
        shared across all joint bins, rather than a separate solve per bin; other moments fall through to the per-bin
        path.

        :param k: The order of the moment.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :return: An array of shape :attr:`shape` holding the kth moment of each joint SFS bin.
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

        if start_time > 0:
            acc = self.accumulate(k, [start_time, end_time], center=center, permute=permute)
            out = acc[..., 1] - acc[..., 0]
        else:
            out = self.accumulate(k, [end_time], center=center, permute=permute)[..., 0]

        if np.isnan(out).any():
            raise ValueError(
                "NaN value encountered when computing moment. "
                "This is likely due to an ill-conditioned rate matrix."
            )

        return JointSFS(out, pop_names=self.lineage_config.pop_names)

    def _config_distribution_items(self, configs: Sequence[Tuple[int, ...]]) -> List[Tuple[Tuple[int, ...], Reward]]:
        """``(config, reward)`` pairs for the requested joint bins (all polymorphic bins by default)."""
        cfgs = self._get_configs() if configs is None else list(configs)
        return [(c, JointSFSReward(c)) for c in cfgs]

    def bin(self, *config: int) -> 'RewardDistribution':
        """The 1D branch-length distribution of the joint SFS bin with the given per-population descendant counts —
        a callable-and-plottable :class:`RewardDistribution` (e.g. ``jsfs.bin(1, 0).pdf.plot()``). The number of
        indices is the number of populations.

        :param config: The descendant configuration (one count per population).
        :return: The accumulated-reward distribution of ``L_{config}``.
        """
        d = self.distribution(reward=JointSFSReward(tuple(config)))
        d.label = f"jSFS bin {tuple(config)}"
        return d

    def joint_distribution(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> 'JointRewardDistribution':
        """The joint distribution of the branch lengths of two joint SFS bins *within a tree* — the bivariate object
        behind the within-tree cross-moment ``E[L_{config_a} · L_{config_b}]`` of the multi-population SFS. Its
        ``(1, 1)`` cross-moment is that entry and its ``corr`` is the within-tree correlation of the two bins.

        :param config_a: The first descendant configuration (one count per population).
        :param config_b: The second descendant configuration.
        :return: The joint accumulated-reward distribution of ``(L_{config_a}, L_{config_b})``.
        """
        jd = super().joint_distribution(JointSFSReward(tuple(config_a)), JointSFSReward(tuple(config_b)))
        jd.label = f"jSFS bins {tuple(config_a)} x {tuple(config_b)}"
        return jd

    def _plot_cdf(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = 'Joint SFS bin CDFs',
    ) -> 'plt.Axes':
        """
        Plot the cumulative distribution function of every joint SFS bin at once.

        :param ax: Axes to plot on.
        :param x: Values to evaluate the CDFs at. By default, an evenly spaced grid up to the largest bin's support.
        :param configs: The joint bins (descendant configurations) to plot. By default, all of them.
        :param n_points: Number of evaluation points for the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        return self._plot_reward_curves('cdf', self._config_distribution_items(configs), ax, x, n_points, show, file,
                                        clear, title)

    def _plot_pdf(
            self,
            ax: 'plt.Axes' = None,
            x: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = 'Joint SFS bin PDFs',
    ) -> 'plt.Axes':
        """
        Plot the probability density function of every joint SFS bin at once.

        :param ax: Axes to plot on.
        :param x: Values to evaluate the PDFs at. By default, an evenly spaced grid up to the largest bin's support.
        :param configs: The joint bins (descendant configurations) to plot. By default, all of them.
        :param n_points: Number of evaluation points for the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        return self._plot_reward_curves('pdf', self._config_distribution_items(configs), ax, x, n_points, show, file,
                                        clear, title)

    def _plot_quantile(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            configs: Sequence[Tuple[int, ...]] = None,
            n_points: int = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            title: str = 'Joint SFS bin quantile functions',
    ) -> 'plt.Axes':
        """
        Plot the quantile function of every joint SFS bin at once (bin branch length versus probability ``q``).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate the quantiles at. By default, an evenly spaced grid in ``(0, 1)``.
        :param configs: The joint bins (descendant configurations) to plot. By default, all of them.
        :param n_points: Number of evaluation points for the default grid.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        return self._plot_reward_curves('quantile', self._config_distribution_items(configs), ax, q, n_points, show,
                                        file, clear, title)

    def accumulate(
            self,
            k: int,
            end_times: Iterable[float],
            center: bool = True,
            permute: bool = True
    ) -> np.ndarray:
        """
        Evaluate the kth moments of the joint site-frequency spectrum at different end times.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moments.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :return: Array of shape :attr:`shape` ``+ (len(end_times),)`` with each bin's kth moment over time.
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
        Plot accumulation of joint SFS moments over time, one curve per (polymorphic) bin.

        :param k: The order of the moment.
        :param end_times: Times when to evaluate the moment. Defaults to :attr:`~phasegen.settings.Settings.plot_n_grid` points up to
            :attr:`~phasegen.settings.Settings.plot_endpoint_quantile`.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :param ax: The axes to plot on.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param title: Title of the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from ..visualization import Visualization

        k = int(k)

        if ax is None:
            ax = plt.gca()

        if end_times is None:
            end_times = np.linspace(0, self.tree_height.quantile(Settings.plot_endpoint_quantile),
                                    Settings.plot_n_grid)

        end_times = np.asarray(list(end_times))

        if title is None:
            title = f"Joint SFS moment accumulation (order {k})"

        configs = self._get_configs()
        accumulation = self.accumulate(k, end_times, center=center, permute=permute)

        for i, config in enumerate(configs):
            Visualization.plot(
                ax=ax,
                x=end_times,
                y=accumulation[config],
                xlabel='t',
                ylabel='moment',
                label=str(config),
                file=file,
                show=(i == len(configs) - 1) and show,
                clear=clear,
                title=title
            )

        return ax

    @cached_property
    def mean(self) -> JointSFS:
        """
        Mean of the joint site-frequency spectrum, i.e. the expected branch length subtending each descendant
        configuration. Computed as a single occupation-time contraction shared across all joint bins.
        """
        return self.moment(k=1)

    @cached_property
    def var(self) -> JointSFS:
        """
        Variance of the joint site-frequency spectrum.
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
        Batched joint-SFS covariance: all ``O(n^{2P})`` bin pairs share one two-point occupation operator ``K``
        (see :meth:`_two_point_occupation`), so the whole covariance is ``cov = R^T (K + K^T) R - outer(mean)`` via a
        single contraction over the stacked bin rewards instead of a cross-moment per pair. Cached so that
        :attr:`cov` and :attr:`var` share the single (potentially expensive) ``K`` solve.

        :return: ``(configs, cov)`` with ``cov`` the bins-by-bins covariance over the polymorphic ``configs``, or
            ``None`` when not applicable (closed form disabled, explicit end time, or absorption not almost sure) so
            callers fall back.
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
        between bins ``(a_0, ..., a_{P-1})`` and ``(b_0, ..., b_{P-1})``.
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
    """
    Two-locus site-frequency spectrum under recombination. Entry ``(i, j)`` of the (symmetrized) mean is
    ``E[L^0_i · L^1_j]`` — the expected product of the branch length subtending ``i`` samples at locus 0 and ``j``
    samples at locus 1 — computed as a second cross-moment of two per-locus SFS rewards on the two-locus
    block-counting state space. It reduces to ``Coalescent.sfs.cov`` (plus the outer product of the marginal means)
    as ``r → 0`` and to the outer product of the marginal SFS as ``r → ∞``.

    The :attr:`mean` is computed for the whole spectrum at once as a single two-point occupation contraction shared
    across all bin pairs rather than a cross-moment per pair.
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
        """The joint distribution of the locus-0 bin-``i`` and locus-1 bin-``j`` branch lengths — the bivariate
        object behind the two-locus SFS entry ``E[L^0_i L^1_j]``. Its ``(1, 1)`` cross-moment is that entry, and
        its ``corr`` is the cross-locus correlation.

        :param i: The locus-0 frequency class.
        :param j: The locus-1 frequency class.
        :return: The joint accumulated-reward distribution of ``(L^0_i, L^1_j)``.
        """
        jd = PhaseTypeDistribution.joint_distribution(self, TwoLocusSFSReward(0, i), TwoLocusSFSReward(1, j))
        jd.label = f"locus-0 bin {i} x locus-1 bin {j}"
        return jd

    @cached_property
    def mean(self) -> TwoLocusSFS:
        """
        Mean two-locus SFS, ``E[L^0_i · L^1_j]`` for all polymorphic bins, symmetrized over the two loci. Computed for
        the whole spectrum at once as a single two-point occupation contraction shared across all bin pairs, falling
        back to a per-pair cross-moment when that closed form does not apply (a multi-epoch demography, an explicit end
        time, or absorption not almost sure).
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
        Batched mean two-locus SFS. Each bin pair ``(i, j)`` is the uncentered cross-moment
        ``E[L^0_i · L^1_j] = r^0_i (K + K^T) r^1_j`` with two-point occupation ``K = diag(m) (-T)^{-1}`` and occupation
        times ``m = alpha (-T)^{-1}``. The dense ``K`` is never formed (the two-locus state space is large, so an
        ``O(n_states^2)`` operator would cost more than every per-pair solve combined). Instead it is factored: with
        ``A = (-T)^{-1} R1`` and ``B = (-T)^{-1} R0`` over the stacked per-locus bin rewards,
        ``E[L^0 (L^1)^T] = (m ⊙ R0)^T A + B^T (m ⊙ R1)`` needs only ``2 (n - 1) + 1`` back-substitutions against one
        factorization of the transient generator.

        Restricted, like :meth:`_two_point_occupation`, to a single (unbounded) epoch with almost-sure absorption and
        no accumulation window; other cases return ``None`` and the caller falls back to the per-pair cross-moment.

        :return: The mean two-locus SFS, or ``None`` when the closed form does not apply.
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

    def sample_per_locus(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw the per-locus branch-length vectors ``(L^0, L^1)`` from the *same* trajectories. The two-locus SFS
        entry ``(i, j)`` is the cross-moment ``E[L^0_i · L^1_j]``, so both loci must come from one trajectory.

        :param n_samples: Number of trajectories to sample.
        :return: A pair of arrays, each of shape ``(n_samples, n + 1)`` (locus-0 and locus-1 branch lengths).
        """
        indices = self._get_indices()
        rewards = (
            [CombinedReward([self.reward, TwoLocusSFSReward(0, i)]) for i in indices] +
            [CombinedReward([self.reward, TwoLocusSFSReward(1, j)]) for j in indices]
        )
        sampled = self._sample(n_samples, rewards=rewards)
        n_bins = len(indices)

        left = np.zeros((n_samples, self.lineage_config.n + 1))
        right = np.zeros((n_samples, self.lineage_config.n + 1))
        left[:, 1:1 + n_bins] = sampled[:, :n_bins]
        right[:, 1:1 + n_bins] = sampled[:, n_bins:]

        return left, right

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Draw samples of the two-locus site-frequency spectrum. Each sample is the (symmetrized) outer product of the
        two per-locus branch-length vectors of one trajectory, so its per-sample mean equals :attr:`mean`.

        :param n_samples: Number of two-locus spectra to sample.
        :return: Array of shape ``(n_samples, n + 1, n + 1)``.
        """
        left, right = self.sample_per_locus(n_samples)
        out = np.einsum('ni,nj->nij', left, right)

        return (out + out.transpose(0, 2, 1)) / 2

    def to_empirical(self, n_samples: int) -> 'EmpiricalTwoLocusSFSDistribution':
        """
        Build an empirical (sample-based) two-locus SFS counterpart by simulating ``n_samples`` trajectories. The
        per-locus branch-length vectors come from the same trajectory, so cross-moments and joint surfaces are exact.

        :param n_samples: Number of trajectories to simulate.
        :return: An :class:`~phasegen.distributions.empirical.EmpiricalTwoLocusSFSDistribution`.
        """
        from .empirical import EmpiricalTwoLocusSFSDistribution

        left, right = self.sample_per_locus(n_samples)
        mean = np.einsum('ni,nj->ij', left, right) / n_samples  # non-symmetrized, as in the msprime path

        return EmpiricalTwoLocusSFSDistribution(mean, left=left, right=right)

    @cached_property
    def corr(self) -> TwoLocusSFS:
        """
        Pearson correlation between the locus-0 and locus-1 branch lengths,
        ``Corr(L^0_i, L^1_j) = (E[L^0_i L^1_j] - E[L^0_i] E[L^1_j]) / (sd(L^0_i) sd(L^1_j))``, for all polymorphic
        bins ``(i, j)``. This is the centered, scale-free companion to :attr:`mean` (which is the *uncentered*
        cross-moment ``E[L^0_i L^1_j]`` and therefore tends to the outer product of the marginal SFS means as the
        loci decouple). It is ``0`` as ``r → ∞`` (independent loci) and reduces to the single-locus SFS correlation
        as ``r → 0`` (fully linked). The per-locus means and variances are the marginals of the two-locus space and
        coincide for the two exchangeable loci.
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

