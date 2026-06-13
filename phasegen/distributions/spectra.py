"""Site-frequency-spectrum distributions (SFS, folded, joint, two-locus)."""

import itertools
import logging
from abc import ABC, abstractmethod
from ..caching import cached_property, cache
from typing import List, Tuple, Iterable, Iterator, Optional, Sequence, Set, TYPE_CHECKING
import numpy as np
from ..demography import Demography
from ..expm import Backend
from ..rewards import Reward, TreeHeightReward, UnfoldedSFSReward, UnitReward, CombinedReward, FoldedSFSReward, SFSReward, JointSFSReward, TwoLocusSFSReward
from ..settings import Settings
from ..spectrum import SFS, SFS2, JointSFS, TwoLocusSFS
from ..state_space import BlockCountingStateSpace, StateSpace, JointBlockCountingStateSpace, TwoLocusBlockCountingStateSpace
from ..utils import multiset_permutations

from ._common import _make_hashable
from .phase_type import PhaseTypeDistribution, TreeHeightDistribution

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

expm = Backend.expm
logger = logging.getLogger('phasegen')


class SFSDistribution(PhaseTypeDistribution, ABC):
    """
    Base class for site-frequency spectrum distributions.
    """

    def __init__(
            self,
            state_space: BlockCountingStateSpace,
            tree_height: TreeHeightDistribution,
            demography: Demography,
            reward: Reward = None
    ):
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

        # batched mean: every bin's mean is ``occupation . r_bin`` with the same occupation-time vector, so the whole
        # spectrum is one contraction instead of a per-bin solve. This is the closed form's spectrum path (it shares
        # the transient solve across bins); only for the plain mean to absorption (k=1, default reward, no custom
        # accumulation window) and when flattening does not apply (flattening reduces the state space and wins).
        # Other cases fall through to the per-bin path.
        if (
                Settings.closed_form_last_epoch and
                not self._flattening_applies(k) and
                k == 1 and
                start_time is None and
                end_time is None and
                self.tree_height.end_time is None and
                rewards == (self.reward,)
        ):
            occupation = self._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
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

        accumulation = np.array([
            self.get_accumulation(k, i, end_times, rewards)
            for i in indices
        ])

        # pad with zeros
        return np.concatenate([
            np.zeros((1, len(end_times))),
            accumulation,
            np.zeros((self.lineage_config.n - len(indices), len(end_times)))
        ])

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
        :param end_times: Times when to evaluate the moment. By default, 200 evenly spaced values between 0 and
            the 99th percentile.
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
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

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
            This currently only works for a single epoch, i.e. a time-homogeneous demography, and recombination is not
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
        # raise not implemented error if more than one epoch
        if self.demography.has_n_epochs(2):
            raise NotImplementedError("Sampling not implemented for more than one epoch.")

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

        # get non-absorbing states
        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)

        # number of non-absorbing states
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

        p = alpha @ Q @ p_total

        return p

    def get_mutation_configs(self, theta: float) -> Iterator[Tuple[Tuple[float, ...], float]]:
        """
        An iterator over the probabilities of observing mutational configurations according to the infinite sites model.
        The order of the mutational configurations generated ascends in the number of mutations observed.
        See :meth:`get_mutation_config` for more information on mutational configurations.

        .. note::
            This currently only works for a single epoch, i.e. a time-homogeneous demography, and recombination is not
            supported. Also note that the number of configurations is infinite, so this iterator will never stop.
            However, depending on the mutation rate, the probability of observing configurations of higher mutation
            counts will decrease over time. You can keep track of the generated probability mass by checking the
            :attr:`~.generated_mass` attribute, which is reset every time this method is called.
            A good approach is thus to keep generating configurations until the generated mass is above a certain
            threshold. More complex demographic models, larger sample sizes, and higher mutation rates all increase
            the number of generated configurations necessary to reach a certain mass.

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


class JointSFSDistribution(PhaseTypeDistribution):
    """
    Joint (multi-population) site-frequency spectrum distribution.

    Moments are returned as a multi-dimensional array of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, where ``n_p`` is the
    sample size of population ``p``. The entry at index ``(k_0, ..., k_{P-1})`` is the moment for branches subtending
    exactly ``k_p`` samples from population ``p``. The monomorphic bins (the all-zero and the full
    ``(n_0,...,n_{P-1})`` configuration) are zero by convention.
    """

    def __init__(
            self,
            state_space: JointBlockCountingStateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography,
            reward: Reward = None
    ):
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

        :param k: The order of the moment.
        :param start_time: Time when to start accumulation of moments. By default, the start time specified when
            initializing the distribution.
        :param end_time: Time when to end accumulation of moments. By default, either the end time specified when
            initializing the distribution or the time until almost sure absorption.
        :param center: Whether to center the moment around the mean.
        :param permute: For cross-moments, whether to average over all permutations of rewards.
        :return: An array of shape :attr:`shape` holding the kth moment of each joint SFS bin.
        """
        # batched mean: all joint bins share one occupation-time vector, so the whole joint SFS mean is a single
        # contraction over the stacked bin rewards (closed form's spectrum path). Only for the plain mean to
        # absorption; other cases fall through to the per-bin accumulation.
        if (
                Settings.closed_form_last_epoch and
                int(k) == 1 and
                start_time is None and
                end_time is None and
                self.tree_height.end_time is None
        ):
            occupation = self._occupation_times()
            if occupation is not None:
                m, idx_t = occupation
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
        :param end_times: Times when to evaluate the moment. Defaults to 200 points up to the 99th percentile.
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
            end_times = np.linspace(0, self.tree_height.quantile(0.99), 200)

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
        configuration.
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
    """

    def __init__(
            self,
            state_space: TwoLocusBlockCountingStateSpace,
            tree_height: 'TreeHeightDistribution',
            demography: Demography,
            reward: Reward = None
    ):
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

    @cached_property
    def mean(self) -> TwoLocusSFS:
        """
        Mean two-locus SFS, ``E[L^0_i · L^1_j]`` for all polymorphic bins, symmetrized over the two loci.
        """
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

