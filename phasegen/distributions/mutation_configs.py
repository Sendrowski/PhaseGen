"""
Mutational configurations: the numbers of mutations in the frequency classes of a spectrum, and their probabilities
under the infinite-sites model on any state space whose rewards count the branches of a frequency class.
"""
import heapq
import itertools
from typing import Dict, Hashable, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from ..expm import Backend
from ..rewards import CombinedReward, Reward, SumReward, TreeHeightReward
from ..settings import Settings
from ..state_space import StateSpace

if TYPE_CHECKING:
    from .phase_type import PhaseTypeDistribution

expm = Backend.expm

#: Largest number of floats held by the single-epoch lattice memo of one distribution before it is cleared.
_LATTICE_MEMO_MAX_FLOATS = 2 ** 25


class MutationLayout:
    r"""
    The bins of a mutational configuration. Each bin merges one or more elementary frequency classes of a spectrum,
    and a configuration counts the mutations per bin. The elementary classes are labelled as the spectrum labels its
    bins: the unfolded class :math:`i` of a single-locus spectrum, the pair ``(pop, i)`` of class :math:`i` in which
    the mutation occurs in the deme ``pop``, the descendant vector :math:`(c_0, \dots, c_{P-1})` of a joint spectrum,
    and the pair ``(locus, i)`` of a two-locus spectrum. Layouts are obtained from
    :meth:`UnfoldedSFSDistribution.mutation_layout() <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>`,
    :meth:`FoldedSFSDistribution.mutation_layout() <phasegen.distributions.FoldedSFSDistribution.mutation_layout>`,
    :meth:`JointSFSDistribution.mutation_layout() <phasegen.distributions.JointSFSDistribution.mutation_layout>` and
    :meth:`TwoLocusSFSDistribution.mutation_layout() <phasegen.distributions.TwoLocusSFSDistribution.mutation_layout>`,
    or constructed with any merge of the elementary classes of a spectrum.
    """

    def __init__(
            self,
            bins: Sequence[Sequence[Hashable]],
            positions: Dict[Hashable, Tuple[int, ...]],
            shape: Tuple[int, ...],
            axes: Sequence[str]
    ) -> None:
        """
        Initialize the layout.

        :param bins: The bins, each a sequence of the elementary class labels it merges.
        :param positions: The index of each elementary class label in the spectrum array.
        :param shape: The shape of the spectrum array.
        :param axes: The names of the axes of the spectrum array.
        :raises ValueError: If there are no bins, a bin is empty, a class label appears in more than one bin, or a
            class label has no position within ``shape``.
        """
        bins = tuple(tuple(b) for b in bins)
        labels = [label for b in bins for label in b]

        if not bins or any(len(b) == 0 for b in bins) or len(set(labels)) != len(labels):
            raise ValueError(f"The bins must be non-empty and disjoint, got {bins}.")

        shape = tuple(int(s) for s in shape)

        for label in labels:
            pos = tuple(np.atleast_1d(positions[label])) if label in positions else None
            if pos is None or len(pos) != len(shape) or not all(0 <= i < s for i, s in zip(pos, shape)):
                raise ValueError(f"The class {label} needs a position within the shape {shape}, got {pos}.")

        #: The bins, each a tuple of the elementary class labels it merges.
        self.bins: Tuple[Tuple[Hashable, ...], ...] = bins

        #: The index of each elementary class label in the spectrum array.
        self.positions: Dict[Hashable, Tuple[int, ...]] = {
            label: tuple(int(i) for i in np.atleast_1d(positions[label])) for label in labels
        }

        #: The shape of the spectrum array.
        self.shape: Tuple[int, ...] = shape

        #: The names of the axes of the spectrum array.
        self.axes: Tuple[str, ...] = tuple(axes)

    def __len__(self) -> int:
        """
        The number of bins.

        :return: The number of bins.
        """
        return len(self.bins)

    def __eq__(self, other) -> bool:
        """
        Whether the layouts have the same bins.

        :param other: The other layout.
        :return: Whether they are equal.
        """
        return isinstance(other, MutationLayout) and self.bins == other.bins

    def __hash__(self) -> int:
        """
        Hash of the bins.

        :return: The hash.
        """
        return hash(self.bins)

    def __repr__(self) -> str:
        """
        Representation listing the bins.

        :return: The representation.
        """
        return f"MutationLayout(bins={list(self.bins)})"

    def config(self, counts: Sequence[int]) -> 'MutationConfig':
        """
        The configuration with the given counts.

        :param counts: One non-negative integer count per bin.
        :return: The configuration.
        :raises ValueError: If ``counts`` does not have one non-negative integer per bin.
        """
        return MutationConfig(counts, self)

    def from_array(self, counts: np.ndarray) -> 'MutationConfig':
        """
        The configuration of per-class counts held in a spectrum-shaped array, summing the classes of each bin.

        :param counts: Array of shape :attr:`MutationLayout.shape <phasegen.distributions.MutationLayout.shape>`.
        :return: The configuration.
        :raises ValueError: If ``counts`` does not have the shape of the layout or does not hold non-negative integers.
        """
        counts = np.asarray(counts)

        if counts.shape != self.shape:
            raise ValueError(f"The counts must have shape {self.shape}, got {counts.shape}.")

        if counts.dtype.kind not in 'iuf' or not np.all(np.isfinite(counts) & (counts >= 0) & (counts % 1 == 0)):
            raise ValueError(f"The counts must be non-negative integers, got {counts.tolist()}.")

        return MutationConfig([sum(counts[self.positions[label]] for label in b) for b in self.bins], self)

    def configs(self, k: int) -> Iterator['MutationConfig']:
        """
        The configurations with ``k`` mutations in total.

        :param k: The total number of mutations.
        :return: The configurations, in the order of ``StateSpace._get_partitions``.
        """
        for counts in StateSpace._get_partitions(n=k, k=len(self)):
            yield MutationConfig(counts, self)


class MutationConfig(tuple):
    """
    A mutational configuration: the number of mutations in each bin of a
    :class:`~phasegen.distributions.MutationLayout`. It is a tuple of the counts in bin order, so it compares and
    hashes equal to the plain tuple of its counts, and the counts are also available by bin label and as a
    spectrum-shaped array.
    """

    #: The layout.
    layout: MutationLayout

    def __new__(cls, counts: Sequence[int], layout: MutationLayout) -> 'MutationConfig':
        """
        Create the configuration.

        :param counts: One non-negative integer count per bin.
        :param layout: The layout.
        :return: The configuration.
        :raises ValueError: If ``counts`` does not have one non-negative integer per bin.
        """
        counts = tuple(counts)

        if len(counts) != len(layout):
            raise ValueError(
                "The length of the configuration must be equal to the number of frequency bins. "
                f"Expected {len(layout)}, got {len(counts)}."
            )

        # entries are counts: integral values of any numeric type (R passes doubles) are accepted
        if any(isinstance(c, bool) or not float(c).is_integer() or c < 0 for c in counts):
            raise ValueError(f"The configuration entries must be non-negative integers, got {list(counts)}.")

        obj = super().__new__(cls, (int(c) for c in counts))
        obj.layout = layout

        return obj

    def __getnewargs__(self) -> Tuple:
        """
        Arguments of ``__new__`` for pickling.

        :return: The counts and the layout.
        """
        return tuple(self), self.layout

    @property
    def total(self) -> int:
        """
        The total number of mutations.
        """
        return sum(self)

    def count_of(self, label: Hashable) -> int:
        """
        The count of the bin holding the elementary class ``label``.

        :param label: The class label, as in :attr:`MutationLayout.bins <phasegen.distributions.MutationLayout.bins>`.
        :return: The count.
        :raises KeyError: If no bin holds the label.
        """
        for b, c in zip(self.layout.bins, self):
            if label in b:
                return c

        raise KeyError(f"No bin holds the class {label}.")

    def to_array(self) -> np.ndarray:
        """
        The counts as a spectrum-shaped array, the count of a bin that merges several classes placed at its first
        class.

        :return: Integer array of shape :attr:`MutationLayout.shape <phasegen.distributions.MutationLayout.shape>`.
        """
        out = np.zeros(self.layout.shape, dtype=int)
        for b, c in zip(self.layout.bins, self):
            out[self.layout.positions[b[0]]] = c

        return out


class MutationConfigMixin:
    r"""
    Probabilities of mutational configurations for a phase-type distribution whose reward vectors
    :math:`\mathbf{r}_j` count the branches of the bins :math:`j` of a :class:`~phasegen.distributions.MutationLayout`.
    Given the genealogy, the count :math:`Y_j` of bin :math:`j` is Poisson with mean :math:`\theta \ell_j`, where
    :math:`\ell_j` is the reward accumulated to absorption, and

    .. math::

        \mathbb{P}(\mathbf{Y} = \mathbf{m})
        = \mathbb{E}\left[ \prod_{j=1}^{J} e^{-\theta \ell_j} \frac{(\theta \ell_j)^{m_j}}{m_j!} \right],

    which depends on the state space only through the sub-intensity matrices, the initial distribution and the
    reward vectors. A spectrum provides the elementary class rewards in ``_mutation_class_reward`` and its default
    layout in ``mutation_layout``.
    """

    #: Probability mass yielded by the most recently started configuration iterator.
    generated_mass: float = 0

    def _mutation_class_reward(self: 'PhaseTypeDistribution', label: Hashable) -> Reward:
        """
        The reward of an elementary frequency class.

        :param label: The class label.
        :return: The reward.
        """
        raise NotImplementedError

    def mutation_layout(self, *args, **kwargs) -> MutationLayout:
        """
        The layout of the configurations of this spectrum.

        :return: The layout.
        """
        raise NotImplementedError

    def _bin_reward(self: 'PhaseTypeDistribution', b: Tuple[Hashable, ...]) -> Reward:
        """
        The reward of a bin, the sum of its class rewards times the reward of this distribution.

        :param b: The bin.
        :return: The reward.
        """
        rewards = [self._mutation_class_reward(label) for label in b]

        return CombinedReward([self.reward, rewards[0] if len(rewards) == 1 else SumReward(rewards)])

    def _as_mutation_config(self, config: Sequence[int]) -> MutationConfig:
        """
        The configuration as a :class:`~phasegen.distributions.MutationConfig`, in the default layout unless it
        carries its own.

        :param config: The configuration.
        :return: The configuration.
        :raises ValueError: If ``config`` does not have one non-negative integer per bin.
        """
        if isinstance(config, MutationConfig):
            return config

        if np.isscalar(config):
            config = (config,)

        return MutationConfig(config, self.mutation_layout())

    def _assert_no_window(self: 'PhaseTypeDistribution') -> None:
        """Guard the mutational-configuration path against a bounded accumulation window. The configuration
        probabilities are computed to absorption and take no ``start_time`` / ``end_time``.

        :raises NotImplementedError: If the coalescent has a positive start time or a finite end time.
        """
        if self._windowed:
            start, end = self.tree_height.start_time, self.tree_height.end_time
            raise NotImplementedError(
                "get_mutation_config / get_mutation_configs are not implemented for a bounded accumulation window "
                f"(start_time={start}, end_time={end}): the mutational-configuration probabilities are computed over "
                "the full to-absorption state space and ignore start_time / end_time, so a windowed result would be "
                "the to-absorption one regardless. Use start_time=0 and no finite end_time."
            )

    def get_mutation_config(self: 'PhaseTypeDistribution', config: Sequence[int], theta: float) -> float:
        r"""
        Probability of a mutational configuration under the infinite-sites model, with the notation of
        :class:`~phasegen.distributions.PhaseTypeDistribution`.

        A configuration :math:`\mathbf{m} = (m_1, \dots, m_J)` counts the mutations in each of the :math:`J` bins of
        a :class:`~phasegen.distributions.MutationLayout`, by default one bin per polymorphic frequency class of the
        spectrum: the classes of :meth:`UnfoldedSFSDistribution.mutation_layout()
        <phasegen.distributions.UnfoldedSFSDistribution.mutation_layout>`, which can be folded or resolved by the deme
        in which the mutation occurs, the descendant vectors of :meth:`JointSFSDistribution.mutation_layout()
        <phasegen.distributions.JointSFSDistribution.mutation_layout>`, or the classes of both loci of
        :meth:`TwoLocusSFSDistribution.mutation_layout()
        <phasegen.distributions.TwoLocusSFSDistribution.mutation_layout>`. The reward vector :math:`\mathbf{r}_j` holds
        for each state the number of lineages that subtend bin :math:`j`, multiplied by the reward of this
        distribution, so its accumulated reward :math:`\ell_j` is the branch length of the bin. On a marginal view
        such as ``sfs.demes['pop_0']``, the branch lengths and hence the configuration probabilities are restricted like
        the moments of the view. Given the genealogy, the bin counts :math:`Y_j` are independent Poisson variables with
        means :math:`\theta \ell_j`, where :math:`\theta \ge 0` is the mutation rate per unit of branch length and per
        locus. Hence

        .. math::

            \mathbb{P}(\mathbf{Y} = \mathbf{m})
            = \mathbb{E}\left[ \prod_{j=1}^{J} e^{-\theta \ell_j} \frac{(\theta \ell_j)^{m_j}}{m_j!} \right].

        .. rubric:: Single epoch

        Mutations occur in state :math:`x` at rate :math:`\theta \bar{r}(x)`, where
        :math:`\bar{\mathbf{r}} = \sum_j \mathbf{r}_j`. With the resolvent
        :math:`\mathbf{U} = (\theta \operatorname{diag}(\bar{\mathbf{r}}) - \mathbf{T}_1)^{-1}`, the entry
        :math:`(\mathbf{G}_j)_{xy}` of :math:`\mathbf{G}_j = \theta\, \mathbf{U} \operatorname{diag}(\mathbf{r}_j)` is
        the probability, starting in state :math:`x`, that the next mutation falls into bin :math:`j` while the
        process is in state :math:`y`, and the entry :math:`g_x` of :math:`\mathbf{g} = \mathbf{U} \mathbf{q}_1` is
        the probability of absorption before the next mutation. The probability generating function of Hobolth et al.
        (2025) is

        .. math::

            \mathbb{E}\Big[ \prod_{j=1}^{J} z_j^{Y_j} \Big]
            = \boldsymbol{\alpha}_T \Big( \mathbf{I} - \sum_{j=1}^{J} z_j \mathbf{G}_j \Big)^{-1} \mathbf{g},

        with :math:`z_j \in [0, 1]` and :math:`\mathbf{I}` the identity matrix. Its coefficient of
        :math:`z_1^{m_1} \cdots z_J^{m_J}` is :math:`\mathbf{x}_\mathbf{m} \mathbf{g}`, where
        :math:`\mathbf{x}_\mathbf{0} = \boldsymbol{\alpha}_T` and
        :math:`\mathbf{x}_\mathbf{c} = \sum_{j : c_j > 0} \mathbf{x}_{\mathbf{c} - \mathbf{e}_j} \mathbf{G}_j` for the
        count vectors :math:`\mathbf{c} \le \mathbf{m}`, with :math:`\mathbf{e}_j` the unit vector of bin :math:`j`.

        .. rubric:: Several epochs

        The mutation counts are tracked jointly with the state, on the :math:`L = \prod_j (m_j + 1)` count vectors
        that do not exceed :math:`\mathbf{m}`. In epoch :math:`i`, this process has the sub-intensity matrix

        .. math::

            \mathbf{A}_i = \mathbf{I}_L \otimes \big( \mathbf{T}_i - \theta \operatorname{diag}(\bar{\mathbf{r}}) \big)
            + \theta \sum_{j=1}^{J} \mathbf{N}_j \otimes \operatorname{diag}(\mathbf{r}_j),

        where :math:`\otimes` is the Kronecker product, :math:`\mathbf{I}_L` the :math:`L \times L` identity matrix,
        and :math:`\mathbf{N}_j` raises the count of bin :math:`j` by one while it is below :math:`m_j`. A mutation
        that would exceed :math:`m_j` removes the process. The mass absorbed at count vector :math:`\mathbf{c}`,
        accumulated over all epochs, is :math:`\mathbb{P}(\mathbf{Y} = \mathbf{c})` for every
        :math:`\mathbf{c} \le \mathbf{m}`.

        .. rubric:: Implementation

        - In a single epoch, the resolvent is formed by one dense inverse and cached for the most recent layout and
          :math:`\theta`, together with the row vectors :math:`\mathbf{x}_\mathbf{c} \mathbf{U}` of every count vector
          evaluated so far, so a configuration costs one vector-matrix product per count vector not yet evaluated.
        - Over several epochs, the finite epochs are propagated by matrix exponentials and the last epoch is closed by
          a linear solve. The solves use a sparse LU factorization once :math:`L n_T` reaches
          :attr:`Settings.closed_form_sparse_min_states <phasegen.settings.Settings.closed_form_sparse_min_states>`,
          and the exponentials become sparse actions once it reaches
          :attr:`Settings.expm_action_min_dim <phasegen.settings.Settings.expm_action_min_dim>`. The probabilities of
          all :math:`L` count vectors are cached for the most recent layout and :math:`\theta`.
        - ``get_mutation_configs_by_count()`` yields configurations in ascending order of :math:`|\mathbf{m}|`.
        - ``get_mutation_configs()`` climbs from the empty configuration to a local maximum of the probability and
          expands outward with a priority queue. The order is exactly descending when every other configuration has a
          neighbour, differing by one mutation, of at least equal probability.
        - Both iterators reset ``generated_mass`` when the first configuration is requested and add each yielded
          probability to it, so one minus its value is the probability not yet yielded.

        .. rubric:: References

        Hobolth, A., Boitard, S., Futschik, A. and Leblois, R. (2025). A matrix-analytical sampling formula for
        time-homogeneous coalescent processes under the infinite sites mutation model. Theoretical Population
        Biology, 163, 62-79. https://doi.org/10.1016/j.tpb.2025.03.002

        :param config: A :class:`~phasegen.distributions.MutationConfig`, or one non-negative integer per bin of the
            default layout, a single integer for a one-bin layout. For :math:`n = 4`, the unfolded configuration
            ``[2, 1, 0]`` holds two singletons, one doubleton and no tripletons, and the folded configuration ``[2, 1]``
            holds two singletons or tripletons and one doubleton.
        :param theta: The mutation rate :math:`\theta` per unit of branch length.
        :return: The probability :math:`\mathbb{P}(\mathbf{Y} = \mathbf{m})`.
        :raises ValueError: If ``theta`` is negative or not finite, or if ``config`` does not have one non-negative
            integer per bin.
        :raises NotImplementedError: If the coalescent has a positive start time or a finite end time.
        :raises ModelError: If some state carrying mass can never reach a common ancestor.
        """
        if not 0 <= theta < np.inf:
            raise ValueError(f"Theta must be a finite number greater than or equal to 0, got {theta}.")

        self._assert_no_window()
        self._assert_absorbs()

        config = self._as_mutation_config(config)

        if theta == 0:
            return 1 if config.total == 0 else 0

        if self.demography.has_n_epochs(2):
            return self._get_mutation_config_inhomogeneous(config, theta)

        return self._get_mutation_config_homogeneous(config, theta)

    def _mutation_rewards(self: 'PhaseTypeDistribution', layout: MutationLayout) -> Tuple[np.ndarray, ...]:
        """
        The transient-state mask, the transient initial distribution and the transient bin rewards, cached per layout.

        :param layout: The layout.
        :return: ``(non_absorbing, alpha, R)`` with ``R`` of shape ``(J, n_T)``.
        """
        cache = self.__dict__.get('_mutation_rewards_cache', {})
        if layout in cache:
            return cache[layout]

        non_absorbing = TreeHeightReward()._get(self.state_space).astype(bool)
        alpha = self.state_space.alpha[non_absorbing]
        R = np.array([self._bin_reward(b)._get(self.state_space) for b in layout.bins], dtype=float)[:, non_absorbing]

        if Settings.cache:
            self.__dict__.setdefault('_mutation_rewards_cache', {})[layout] = (non_absorbing, alpha, R)

        return non_absorbing, alpha, R

    def _get_resolvent(self: 'PhaseTypeDistribution', layout: MutationLayout, theta: float) -> Tuple:
        r"""
        Single-epoch resolvent :math:`\mathbf{U} = (\theta \operatorname{diag}(\bar{\mathbf{r}}) - \mathbf{T}_1)^{-1}`,
        the scaled bin rewards :math:`\theta \mathbf{r}_j` and the absorption vector :math:`\mathbf{g}`, with the
        lattice memo of the row vectors :math:`\mathbf{x}_\mathbf{c} \mathbf{U}`. The most recent
        :math:`(\text{layout}, \theta)` is cached.

        :param layout: The layout.
        :param theta: The mutation rate :math:`\theta`.
        :return: ``(U, theta R, g, memo)``.
        """
        cached = self.__dict__.get('_resolvent')
        if cached is not None and cached[0] == (layout, theta):
            return cached[1]

        # the state space may be shared with other coalescents, so set it to this one's rates
        self.state_space.update_epoch(self.demography.get_epoch(0))

        non_absorbing, alpha, R = self._mutation_rewards(layout)

        S = self.state_space.S[non_absorbing, :][:, non_absorbing]
        S = S.toarray() if sp.issparse(S) else np.asarray(S)

        U = np.linalg.inv(theta * np.diag(R.sum(axis=0)) - S)
        g = U @ (-S @ np.ones(S.shape[0]))

        memo = {(0,) * len(layout): alpha @ U}
        resolvent = (U, theta * R, g, memo)

        if Settings.cache:
            self.__dict__['_resolvent'] = ((layout, theta), resolvent)

        return resolvent

    def _get_mutation_config_homogeneous(self: 'PhaseTypeDistribution', config: MutationConfig, theta: float) -> float:
        r"""
        Single-epoch configuration probability :math:`\mathbf{x}_\mathbf{m} \mathbf{g}`, with
        :math:`\mathbf{x}_\mathbf{0} = \boldsymbol{\alpha}_T` and
        :math:`\mathbf{x}_\mathbf{c} = \sum_{j : c_j > 0} \mathbf{x}_{\mathbf{c} - \mathbf{e}_j} \mathbf{G}_j`, the sum
        over the orderings of the mutations accumulated on the lattice of count vectors below :math:`\mathbf{m}`. The
        row vectors :math:`\mathbf{y}_\mathbf{c} = \mathbf{x}_\mathbf{c} \mathbf{U}` are memoized with the resolvent,
        so :math:`\mathbf{x}_\mathbf{c} = \sum_{j : c_j > 0} \mathbf{y}_{\mathbf{c} - \mathbf{e}_j} \circ
        \theta \mathbf{r}_j` costs one product with :math:`\mathbf{U}` per lattice node over all calls.

        :param config: The configuration.
        :param theta: The mutation rate.
        :return: The configuration probability.
        """
        U, R, g, memo = self._get_resolvent(config.layout, theta)
        origin = (0,) * len(config)

        if len(memo) * len(g) > _LATTICE_MEMO_MAX_FLOATS:
            y0 = memo[origin]
            memo.clear()
            memo[origin] = y0

        def x_of(c: Tuple[int, ...]) -> np.ndarray:
            if c == origin:
                return self._mutation_rewards(config.layout)[1]

            x = 0
            for j, cj in enumerate(c):
                if cj > 0:
                    x = x + memo[c[:j] + (cj - 1,) + c[j + 1:]] * R[j]
            return x

        # the nodes strictly below m in ascending order of their total, so that predecessors come first
        target = tuple(config)
        for c in sorted(itertools.product(*[range(m + 1) for m in target]), key=sum):
            if c != target and c not in memo:
                memo[c] = x_of(c) @ U

        return float(x_of(target) @ g)

    def _mutation_epoch_data(self: 'PhaseTypeDistribution', layout: MutationLayout) -> Tuple:
        """
        Configuration-independent inputs of ``_get_mutation_config_inhomogeneous``, cached per layout.

        :param layout: The layout.
        :return: ``(R, r_total, alpha, epochs)``: the transient bin rewards and their sum, the transient initial
            distribution, and per epoch the dense sub-intensity matrix, the absorption-rate vector, the duration
            (``None`` for the unbounded last epoch) and the mask of ``_leaking_states``.
        """
        cache = self.__dict__.get('_mutation_epoch_cache', {})
        if layout in cache:
            return cache[layout]

        non_absorbing, alpha, R = self._mutation_rewards(layout)
        r_total = R.sum(axis=0)

        epochs = []
        for epoch in self._get_epochs_until_unbounded():
            self.state_space.update_epoch(epoch)
            S = self.state_space.S[non_absorbing, :][:, non_absorbing]
            S = S.toarray() if sp.issparse(S) else np.asarray(S)
            e = -S @ np.ones(S.shape[0])
            tau = None if np.isinf(epoch.end_time) else epoch.end_time - epoch.start_time
            epochs.append((S, e, tau, self._leaking_states(S, e, r_total)))

        # leave the state space in the first epoch for any subsequent caller that assumes it
        self.state_space.update_epoch(self.demography.get_epoch(0))

        data = (list(R), r_total, alpha, epochs)

        if Settings.cache:
            self.__dict__.setdefault('_mutation_epoch_cache', {})[layout] = data

        return data

    @staticmethod
    def _leaking_states(S: np.ndarray, e: np.ndarray, r_total: np.ndarray) -> Optional[np.ndarray]:
        """
        The transient states of an epoch from which the process can reach absorption or a state with positive reward.
        The other states form closed classes without reward or absorption, which leave the epoch's lattice generator
        singular and contribute no configuration probability.

        :param S: The transient sub-intensity matrix of the epoch.
        :param e: The absorption-rate vector of the epoch.
        :param r_total: The total bin reward of each transient state.
        :return: The boolean mask of these states, or ``None`` if it holds every state.
        """
        reach = (e > 0) | (r_total > 0)

        if reach.all():
            return None

        adjacent = S != 0
        while True:
            expanded = reach | (adjacent @ reach)
            if (expanded == reach).all():
                return None if reach.all() else reach
            reach = expanded

    def _get_mutation_config_inhomogeneous(
            self: 'PhaseTypeDistribution',
            config: MutationConfig,
            theta: float
    ) -> float:
        """
        Multi-epoch configuration probability, propagating the lattice sub-intensity matrix over the count vectors
        below the configuration epoch by epoch and solving the last epoch to absorption. The mass absorbed at every
        node of the lattice is the probability of that node's configuration, so all of them are cached for the most
        recent ``(layout, theta)``.

        :param config: The configuration.
        :param theta: The mutation rate.
        :return: The configuration probability.
        """
        layout = config.layout
        key = (layout, theta)

        cached = self.__dict__.get('_mutation_probs')
        if cached is not None and cached[0] == key and tuple(config) in cached[1]:
            return cached[1][tuple(config)]

        R, r_total, alpha, epochs = self._mutation_epoch_data(layout)
        m = len(alpha)
        J = len(layout)

        nodes = list(itertools.product(*[range(k + 1) for k in config]))
        index = {c: a for a, c in enumerate(nodes)}
        edges = [(index[c], index[c[:i] + (c[i] + 1,) + c[i + 1:]], i)
                 for c in nodes for i in range(J) if c[i] < config[i]]
        L = len(nodes)
        nt = L * m

        # the one-mutation shifts of each bin, as (source node, target node) pairs
        shifts = [[(a, b) for a, b, i in edges if i == j] for j in range(J)]

        sparse = self._solve_sparse(nt)
        action = nt >= Settings.expm_action_min_dim

        def build_generator(S: np.ndarray) -> 'np.ndarray | sp.spmatrix':
            diag = S - theta * np.diag(r_total)
            if sparse:
                A = sp.kron(sp.identity(L, format='csr'), sp.csr_matrix(diag), format='csr')
                for i in range(J):
                    N = sp.csr_matrix(([1.0] * len(shifts[i]), tuple(zip(*shifts[i])) if shifts[i] else ([], [])),
                                      shape=(L, L))
                    A = A + sp.kron(N, sp.diags(theta * R[i]), format='csr')
                return A

            A = np.zeros((nt, nt))
            for a in range(L):
                A[a * m:(a + 1) * m, a * m:(a + 1) * m] = diag
            for a, b, i in edges:
                A[a * m:(a + 1) * m, b * m:(b + 1) * m] = np.diag(theta * R[i])
            return A

        v = np.zeros(nt)
        v[:m] = alpha

        p = np.zeros(L)
        for S, e, tau, leaking in epochs:
            A = build_generator(S)

            if tau is None:
                occ = self._lu_solver((-A).T, sparse)(v)
                p += occ.reshape(L, m) @ e
                break

            if action:
                u = Backend.expm_multiply(A.T * tau, v)
            else:
                u = v @ expm((A.toarray() if sparse else A) * tau)

            if leaking is None:
                occ = self._lu_solver(A.T, sparse)(u - v)
            else:
                # the occupation of the closed classes enters no equation of the others and exits nowhere
                k = np.tile(leaking, L)
                occ = np.zeros(nt)
                occ[k] = self._lu_solver(A[k][:, k].T, sparse)((u - v)[k])
            p += occ.reshape(L, m) @ e
            v = u

        probs = dict(zip(nodes, p.tolist()))

        if Settings.cache:
            if cached is not None and cached[0] == key:
                cached[1].update(probs)
            else:
                self.__dict__['_mutation_probs'] = (key, probs)

        return probs[tuple(config)]

    def get_mutation_configs_by_count(
            self: 'PhaseTypeDistribution',
            theta: float,
            layout: MutationLayout = None
    ) -> Iterator[Tuple[MutationConfig, float]]:
        """
        Unending iterator over mutational configurations and their probabilities in ascending order of the total
        number of mutations, as described in :meth:`UnfoldedSFSDistribution.get_mutation_config()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`.

        :param theta: The mutation rate per unit of branch length.
        :param layout: The layout of the configurations, by default the layout of one bin per polymorphic frequency
            class of the spectrum.
        :return: An iterator over pairs of configuration and probability.
        """
        layout = self.mutation_layout() if layout is None else layout

        self.generated_mass = 0

        k = 0
        while True:
            for config in layout.configs(k):
                p = self.get_mutation_config(config=config, theta=theta)
                self.generated_mass += p
                yield config, p

            k += 1

    def get_mutation_configs(
            self: 'PhaseTypeDistribution',
            theta: float,
            layout: MutationLayout = None
    ) -> Iterator[Tuple[MutationConfig, float]]:
        """
        Unending iterator over mutational configurations and their probabilities, starting at a local maximum of the
        probability and in descending order of probability under the neighbour condition described in
        :meth:`UnfoldedSFSDistribution.get_mutation_config()
        <phasegen.distributions.UnfoldedSFSDistribution.get_mutation_config>`. The following example consumes it until
        the yielded probability mass exceeds 0.8.

        ::

            coal = pg.Coalescent(n=5)

            it = coal.sfs.get_mutation_configs(theta=1)

            samples = list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))

        :param theta: The mutation rate per unit of branch length.
        :param layout: The layout of the configurations, by default the layout of one bin per polymorphic frequency
            class of the spectrum.
        :return: An iterator over pairs of configuration and probability.
        :raises ModelError: If some state carrying mass can never reach a common ancestor.
        """
        layout = self.mutation_layout() if layout is None else layout
        J = len(layout)

        self._assert_absorbs()

        self.generated_mass = 0

        if theta == 0:
            self.generated_mass = 1.0
            yield MutationConfig((0,) * J, layout), 1.0
            return

        def neighbours(c: Tuple[int, ...]) -> Iterator[MutationConfig]:
            for i in range(J):
                for step in (1, -1):
                    if c[i] + step >= 0:
                        yield MutationConfig(c[:i] + (c[i] + step,) + c[i + 1:], layout)

        # the empty configuration has positive probability
        mode = MutationConfig((0,) * J, layout)
        p_mode = self.get_mutation_config(mode, theta)

        improved = True
        while improved:
            improved = False
            for nb in neighbours(mode):
                p_nb = self.get_mutation_config(nb, theta)
                if p_nb > p_mode:
                    mode, p_mode, improved = nb, p_nb, True
                    break

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
