"""
State space.
"""

import logging
import time
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from typing import List, Tuple, Dict, Callable, cast

import numpy as np

from .coalescent_models import CoalescentModel, StandardCoalescent
from .demography import Epoch
from .lineage import LineageConfig
from .locus import LocusConfig
from .state_space_old import StateSpace as OldStateSpace, LineageCountingStateSpace as OldLineageCountingStateSpace, \
    BlockCountingStateSpace as OldBlockCountingStateSpace

logger = logging.getLogger('phasegen')


class StateSpace(ABC):
    """
    State space.
    """

    def __init__(
            self,
            lineage_config: LineageConfig,
            locus_config: LocusConfig = None,
            model: CoalescentModel = None,
            epoch: Epoch = None,
            cache: bool = True
    ):
        """
        Create a rate matrix.

        :param lineage_config: Population configuration.
        :param locus_config: Locus configuration. One locus is used by default.
        :param model: Coalescent model. By default, the standard coalescent is used.
        :param epoch: Epoch.
        :param cache: Whether to cache the rate matrix for different epochs.
        """
        if locus_config is None:
            locus_config = LocusConfig()

        if model is None:
            model = StandardCoalescent()

        if epoch is None:
            epoch = Epoch()

        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: Coalescent model
        self.model: CoalescentModel = model

        #: Population configuration
        self.lineage_config: LineageConfig = lineage_config

        #: Locus configuration
        self.locus_config: LocusConfig = locus_config

        #: Epoch
        self.epoch: Epoch = epoch

        #: Whether to cache the rate matrix for different epochs.
        self.cache: bool = cache

        #: Cached rate matrices
        self._cache: Dict[Epoch, Tuple[Dict[Tuple['State', 'State'], Tuple[float, str]], List['State']]] = {}

        # time in seconds to compute original rate matrix
        self.time: float | None = None

    @cached_property
    def states(self) -> List['State']:
        """
        The states.
        """
        start = time.time()

        # get all possible transitions
        transitions, states = self.get_transitions()

        # record time to compute rate matrix
        self.time = time.time() - start

        # cache rate matrix if specified
        if self.cache:
            self._cache[self.epoch] = (transitions, states)

        return states

    @cached_property
    def lineages(self) -> np.ndarray:
        """
        The lineage configurations. Each configuration describes the lineages per block, deme and locus, i.e.,
        ``[[[a_ijk]]]`` for block ``i``, deme ``j`` and locus ``k``.
        """
        return np.array([s.lineages for s in self.states])

    @cached_property
    def linked(self) -> np.ndarray:
        """
        The linked lineages per block, deme and locus.
        :return:
        """
        return np.array([s.linked for s in self.states])

    @property
    def unlinked(self) -> np.ndarray:
        """
        Unlinked lineages.
        """
        return self.lineages - self.linked

    @abstractmethod
    def _get_old(self) -> OldStateSpace:
        """
        Get the old state space.
        """
        pass

    def _get_old_ordering(self) -> List[int]:
        """
        Get the ordering of the states in the old state space relative to the new state space.

        :return: Ordering of the states in the old state space.
        """
        old = self._get_old()

        # reorder the states of s2 to match s1
        return cast(List[int], [
            np.where(((old.states == self.lineages[i]) & (old.linked == self.linked[i])).all(axis=(1, 2, 3)))[0][0]
            for i in range(self.k)
        ])

    @staticmethod
    def _get_partitions(n: int, k: int) -> List[List[int]]:
        """
        Find all vectors of length `k` with non-negative integers that sum to `n`.

        :param n: The sum.
        :param k: The length of the vectors.
        :return: All vectors of length `k` with non-negative integers that sum to `n`.
        """
        if k == 0:
            return [[]]

        if k == 1:
            return [[n]]

        vectors = []
        for i in range(n + 1):
            for vector in StateSpace._get_partitions(n - i, k - 1):
                vectors.append(vector + [i])

        return vectors

    def get_transitions(self) -> Tuple[Dict[Tuple['State', 'State'], Tuple[float, str]], List['State']]:
        """
        Get all possible transitions from the given state.

        :return: All possible transitions from the given state.
        """
        sources = [self._get_initial()]
        transitions = {}
        visited = []
        i = 0

        while True:

            targets_new = {}

            for source in sources:

                # skip if source has been visited already
                if source in visited:
                    continue

                # get all possible transitions from source
                targets = self.transition.transit(source)

                # add visited source state
                visited += [source]

                # add transitions to dictionary
                for target, transition in targets.items():
                    transitions[(source, target)] = transition

                # add targets to new targets
                targets_new |= targets

                # increment state counter
                i += 1

                if i in [1000, 10000, 100000]:
                    levels = {1000: 'slow', 10000: 'very slow', 100000: 'extremely slow'}

                    self._logger.warning(
                        f'State space size exceeds {i} states. Computation may be {levels[i]}.'
                    )

            # break if no more targets
            if len(targets_new) == 0:
                break

            # take new targets as source states
            sources = tuple(targets_new.keys())

        return transitions, visited

    @cached_property
    def e(self) -> np.ndarray:
        """
        Vector with ones of size ``k``.
        """
        return np.ones(self.k)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Intensity matrix.
        """
        return self._get_rate_matrix()

    @cached_property
    def alpha(self) -> np.ndarray:
        """
        Initial state vector.
        """
        pops = self.lineage_config._get_initial_states(self)
        loci = self.locus_config._get_initial_states(self)

        # combine initial states
        alpha = pops * loci

        # return normalized vector
        # normalization ensures that the initial state vector is a probability distribution
        # as we may have multiple initial states
        return alpha / alpha.sum()

    @cached_property
    def k(self) -> int:
        """
        Number of states.
        """
        k = len(self.states)

        # warn if state space is large
        if k > 400:
            self._logger.warning(f'State space is large ({k} states). Note that the computation time '
                                 f'increases exponentially with the number of states.')

        return k

    @cached_property
    def transition(self) -> 'Transition':
        """
        Transition.
        """
        return Transition(self)

    def update_epoch(self, epoch: Epoch):
        """
        Update the epoch.

        :param epoch: Epoch.
        :return: State space.
        """
        # only remove cached properties if epoch has changed
        if self.epoch != epoch:
            self.drop_S()

        self.epoch = epoch

    def __eq__(self, other):
        """
        Check if two state spaces are equal. We do not check for equivalence of the epochs as we can
        update the epoch of a state space dynamically.

        :param other: Other state space
        :return: Whether the two state spaces are equal
        """
        return (
                self.__class__ == other.__class__ and
                self.lineage_config == other.lineage_config and
                self.locus_config == other.locus_config and
                self.model == other.model
        )

    def drop_S(self):
        """
        Drop the current rate matrix.
        """
        try:
            # noinspection all
            del self.S
        except AttributeError:
            pass

    def drop_cache(self):
        """
        Drop the rate matrix cache and current rate matrix.
        """
        self.drop_S()

        self._cache = {}

    @abstractmethod
    def _get_initial(self):
        """
        Get the initial state.
        """
        pass

    def _get_rate_matrix(self) -> np.ndarray:
        """
        Get the rate matrix.

        TODO don’t compute transitions twice for disabled caching

        :return: The rate matrix.
        """
        # check if epoch is in cache
        if self.cache and self.epoch in self._cache:
            transitions, states = self._cache[self.epoch]

        else:
            # get all possible transitions
            transitions, states = self.get_transitions()

            # cache rate matrix if specified
            if self.cache:
                self._cache[self.epoch] = (transitions, states)

        return self._graph_to_matrix(transitions)

    def _graph_to_matrix(
            self,
            transitions: Dict[Tuple['State', 'State'], Tuple[float, str]]
    ) -> np.ndarray:
        """
        Convert transition graph to rate matrix.

        :param transitions: Transitions.
        :return: Rate matrix.
        """
        S = np.zeros((self.k, self.k))

        # order of original states
        ordering = {s: i for i, s in enumerate(self.states)}

        # fill rate matrix
        for (source, target), transition in transitions.items():
            S[ordering[source], ordering[target]] = transition[0]

        # fill diagonal with negative sum of row
        S[np.diag_indices_from(S)] = -np.sum(S, axis=1)

        return S

    def get_sparsity(self) -> float:
        """
        Get the sparsity of the rate matrix.

        :return: The sparsity.
        """
        return 1 - np.count_nonzero(self.S) / self.S.size

    def _get_color_state(self, i: int) -> str:
        """
        Get color of the state indexed by `i`.
        """
        if self.states[i].is_absorbing():
            return '#f1807e'

        if self.alpha[i] > 0:
            return 'lightgreen'

        return 'lightblue'

    def plot_rates(
            self,
            file: str,
            view: bool = True,
            cleanup: bool = False,
            dpi: int = 400,
            ratio: float = 0.6,
            background_color: str = 'white',
            extension: str = 'png',
            format_state: Callable[[np.array], str] = None,
            format_transition: Callable[['Transition'], str] = None
    ):
        """
        Plot the rate matrix using graphviz. Note that graphviz must be installed which is an external dependency.

        :param file: File to save plot to.
        :param view: Whether to view the plot.
        :param cleanup: Whether to remove the source file.
        :param dpi: Dots per inch.
        :param ratio: Aspect ratio.
        :param background_color: Background color.
        :param extension: File format.
        :param format_state: Function to format state with state array as argument.
        :param format_transition: Function to format transition with transition as argument.
        """
        import graphviz

        if format_state is None:
            def format_state(s: Tuple[np.ndarray, np.ndarray]) -> str:
                """
                Format state.

                :param s: State.
                :return: Formatted state.
                """
                return str(s[0]).replace('\n', '') + '\n' + str(s[1]).replace('\n', '')

        if format_transition is None:
            def format_transition(rate: float, kind: str) -> str:
                """
                Format transition.

                :param rate: Rate.
                :param kind: Kind.
                :return: Formatted transition.
                """
                return f' {kind}: ' + '{:.2f}'.format(rate).rstrip('0').rstrip('.')

        graph = graphviz.Digraph()

        # add nodes
        for i, state in enumerate(self.states):
            graph.node(
                name=format_state(state.data),
                fillcolor=self._get_color_state(i),
                style='filled'
            )

        transitions, _ = self.get_transitions()

        # add non-zero edges
        for (source, target), transition in transitions.items():
            if not source.is_absorbing():
                graph.edge(
                    tail_name=format_state(source.data),
                    head_name=format_state(target.data),
                    label=format_transition(*transition),
                    color=Transition._colors[transition[1]],
                    fontcolor=Transition._colors[transition[1]]
                )

        graph.graph_attr['dpi'] = str(dpi)
        graph.graph_attr['ratio'] = str(ratio)
        graph.graph_attr['bgcolor'] = background_color

        graph.render(
            filename=file,
            view=view,
            cleanup=cleanup,
            format=extension
        )


class LineageCountingStateSpace(StateSpace):
    """
    Default rate matrix where there is one state per number of lineages for each deme and locus.
    """

    def _get_initial(self) -> 'State':
        """
        Get the initial state.
        """
        data = tuple(np.zeros((self.locus_config.n, self.lineage_config.n_pops, 1), dtype=int) for _ in range(2))
        data[0][:, 0, 0] = self.lineage_config.n

        return State(data)

    def _get_old(self) -> OldLineageCountingStateSpace:
        """
        Get the old state space.
        """
        return OldLineageCountingStateSpace(
            lineage_config=self.lineage_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.epoch
        )


class BlockCountingStateSpace(StateSpace):
    r"""
    Rate matrix for block-counting state space where there is one state per sample configuration:

    A block-counting state is a vector of length ``n`` where each element represents the number of lineages
    subtending ``i`` lineages in the coalescent tree.

        .. math::
            (a_1,...,a_n) \in \mathbb{Z}_+^n : \sum_{i=1}^{n} i a_i = n.

    per deme and per locus. This state space can distinguish between different tree topologies
    and is thus used when computing statistics based on the SFS.
    """

    def __init__(
            self,
            lineage_config: LineageConfig,
            locus_config: LocusConfig = None,
            model: CoalescentModel = None,
            epoch: Epoch = None
    ):
        """
        Create a rate matrix.

        :param lineage_config: Population configuration.
        :param locus_config: Locus configuration. One locus is used by default.
        :param model: Coalescent model. By default, the standard coalescent is used.
        :param epoch: Epoch.
        """
        # currently only one locus is supported, due to a very complex state space for multiple loci
        if locus_config is not None and locus_config.n > 1:
            raise NotImplementedError('Block-counting state space only supports one locus.')

        super().__init__(lineage_config=lineage_config, locus_config=locus_config, model=model, epoch=epoch)

    def _get_initial(self) -> 'State':
        """
        Get the initial state.
        """
        data = tuple(
            np.zeros((self.locus_config.n, self.lineage_config.n_pops, self.lineage_config.n), dtype=int)
            for _ in range(2)
        )

        data[0][:, 0, 0] = self.lineage_config.n

        return State(data)

    def _get_old(self) -> OldBlockCountingStateSpace:
        """
        Get the old state space.
        """
        return OldBlockCountingStateSpace(
            lineage_config=self.lineage_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.epoch
        )


class Transition:
    """
    Class representing a transition between two states.
    """

    #: Colors for different types of transitions
    _colors: Dict[str, str] = {
        'recombination': 'orange',
        'coalescence': 'darkgreen',
        'locus_coalescence': 'darkgreen',
        'linked_coalescence': 'darkgreen',
        'unlinked_coalescence': 'darkgreen',
        'mixed_coalescence': 'darkgreen',
        'unlinked_coalescence+mixed_coalescence': 'darkgreen',
        'mixed_coalescence+unlinked_coalescence': 'darkgreen',
        'linked_migration': 'blue',
        'unlinked_migration': 'blue',
        'migration': 'blue',
        'invalid': 'red'
    }

    def __init__(
            self,
            state_space: StateSpace
    ):
        """
        Initialize a transition.

        :param state_space: State space.
        """
        #: State space.
        self.state_space: StateSpace = state_space

    def transit(self, source: 'State') -> Dict['State', Tuple[float, str]]:
        """
        Get all possible target states from the given source state.

        :param source: Source state.
        :return: All possible target states.
        """
        targets: Dict['State', Tuple[float, str]] = {}

        targets |= self.migrate(source)

        if source.is_absorbing():
            return targets

        targets |= self.coalesce(source)

        targets |= self.recombine(source)

        return targets

    @staticmethod
    def add_target(targets: Dict['State', Tuple[float, str]], target: 'State', rate: float, kind: str):
        """
        Add a target state to the list of targets.

        :param targets: Dictionary of target states.
        :param target: New target state.
        :param rate: Rate of the transition.
        :param kind: Kind of the transition.
        """
        if target in targets:
            targets[target] = (targets[target][0] + rate, targets[target][1] + '+' + kind)
        else:
            targets[target] = (rate, kind)

    def coalesce(self, source: 'State') -> Dict['State', Tuple[float, str]]:
        """
        Get all possible coalescent transitions from the given state.

        :param source: Source state.
        :return: All possible coalescent transitions from the given state.
        """
        targets: Dict['State', Tuple[float, str]] = {}
        pop_sizes = [self.state_space.epoch.pop_sizes[pop] for pop in self.state_space.lineage_config.pop_names]

        if source.n_loci == 1:
            locus = 0
            for deme in range(source.n_demes):

                blocks = self.state_space.model.coalesce(
                    self.state_space.lineage_config.n,
                    source.lineages[locus, deme]
                )

                for block, rate in blocks:
                    target = source.copy()
                    target.lineages[locus, deme] = block

                    time_scale = self.state_space.model._get_timescale(pop_sizes[deme])
                    self.add_target(targets, target, rate / time_scale, 'coalescence')

            return targets

        if source.n_loci == 2:

            if not isinstance(self.state_space, LineageCountingStateSpace):
                raise NotImplementedError(
                    'Coalescence with recombination is only implemented for LineageCountingStateSpace.'
                )

            if not isinstance(self.state_space.model, StandardCoalescent):
                raise NotImplementedError('Coalescence with recombination is only implemented for StandardCoalescent.')

            bins = dict(
                linked=source.linked[0],
                unlinked1=source.unlinked[0],
                unlinked2=source.unlinked[1]
            )

            for deme in range(source.n_demes):

                time_scale = self.state_space.model._get_timescale(pop_sizes[deme])

                for ((class1, counts1), (class2, counts2)) in product(bins.items(), repeat=2):

                    target = source.copy()

                    # linked or unlinked coalescence
                    if class1 == class2:
                        # we need at least 2 lineages to coalesce
                        if np.any(counts1[deme] < 2):
                            continue

                        # if the classes are the same, the counts are the same
                        rate = self.state_space.model._get_rate(b=counts1[deme, 0], k=2)

                        # unlinked coalescence in locus 1
                        if 'unlinked1' in class1:

                            target.lineages[0, deme] -= 1
                            self.add_target(targets, target, rate / time_scale, 'unlinked_coalescence')

                        # unlinked coalescence in locus 2
                        elif 'unlinked2' in class1:

                            target.lineages[1, deme] -= 1
                            self.add_target(targets, target, rate / time_scale, 'unlinked_coalescence')

                        # linked coalescence in both loci
                        elif np.all(source.linked[:, deme] > 0):

                            target.lineages[:, deme] -= 1
                            target.linked[:, deme] -= 1
                            self.add_target(targets, target, rate / time_scale, 'linked_coalescence')

                    # mixed or locus coalescence
                    # use lower than operator to ensure we only consider each case once
                    elif class1 < class2:
                        if counts1[deme] < 1 or counts2[deme] < 1:
                            continue

                        rate = counts1[deme, 0] * counts2[deme, 0]

                        # mixed coalescence of linked and unlinked lineages
                        if 'linked' in (class1, class2) and ('unlinked' in class1 or 'unlinked' in class2):
                            locus = 0 if '1' in class1 or '1' in class2 else 1

                            # condition already checked above
                            if target.lineages[locus, deme, 0] > 1:
                                target.lineages[locus, deme, 0] -= 1

                                self.add_target(targets, target, rate / time_scale, 'mixed_coalescence')

                        # locus coalescence of unlinked lineages
                        else:
                            # make sure we have unlinked lineages in both loci
                            if np.all(source.unlinked[:, deme, 0] > 0):
                                target.linked[:, deme, 0] += 1

                                self.add_target(targets, target, rate / time_scale, 'locus_coalescence')

            return targets

        raise NotImplementedError('Coalescence is not implemented for more than 2 loci.')

    def migrate(self, source: 'State') -> Dict['State', Tuple[float, str]]:
        """
        Get all possible migration transitions from the given state.

        :param source: Source state.
        :return: All possible migration transitions from the given state.
        """
        return self.migrate_linked(source) | self.migrate_unlinked(source)

    def migrate_unlinked(self, source: 'State') -> Dict['State', Tuple[float, str]]:
        """
        Get all possible unlinked migration transitions from the given state.
        Note that we also consider migration to unlinked when there is only one locus.

        :param source: Source state.
        :return: All possible migration transitions from the given state.
        """
        targets: Dict['State', Tuple[float, str]] = {}
        pop_names = self.state_space.lineage_config.pop_names
        kind = 'migration' if source.n_loci == 1 else 'unlinked_migration'

        for locus in range(source.n_loci):
            for d1, d2 in filter(lambda x: x[0] != x[1], product(range(source.n_demes), repeat=2)):

                for block in range(source.n_blocks):

                    # skip if no lineages to migrate
                    if source.lineages[locus, d1, block] > 0 and source.unlinked[locus, d1, block] > 0:
                        target = source.copy()

                        target.lineages[locus, d1, block] -= 1
                        target.lineages[locus, d2, block] += 1

                        base_rate = self.state_space.epoch.migration_rates[(pop_names[d1], pop_names[d2])]

                        # scale migration rate by number of lineages in source deme
                        rate = base_rate * cast(int, source.unlinked[locus, d1, block])

                        self.add_target(targets, target, rate, kind)

        return targets

    def migrate_linked(self, source: 'State') -> Dict['State', Tuple[float, str]]:
        """
        Get all possible linked migration transitions from the given state.

        :param source: Source state.
        :return: All possible migration transitions from the given state.
        """
        targets: Dict['State', Tuple[float, str]] = {}

        # no linked migration if there is only one locus
        if source.n_loci == 1:
            return targets

        pop_names = self.state_space.lineage_config.pop_names

        for d1, d2 in filter(lambda x: x[0] != x[1], product(range(source.n_demes), repeat=2)):

            for block in range(source.n_blocks):

                # skip if no lineages to migrate
                if np.all(source.lineages[:, d1, block]) > 0 and np.all(source.linked[:, d1, block] > 0):
                    target = source.copy()

                    target.lineages[:, d1, block] -= 1
                    target.lineages[:, d2, block] += 1

                    target.linked[:, d1, block] -= 1
                    target.linked[:, d2, block] += 1

                    base_rate = self.state_space.epoch.migration_rates[(pop_names[d1], pop_names[d2])]

                    # scale migration rate by number of lineages in source deme
                    # both loci are assumed to have the same number of linked lineages here
                    rate = base_rate * cast(int, source.linked[0, d1, block])

                    self.add_target(targets, target, rate, 'linked_migration')

        return targets

    def recombine(self, state: 'State') -> Dict['State', Tuple[float, str]]:
        """
        Get all possible recombination transitions from the given state.

        :param state: State.
        :return: All possible recombination transitions from the given state.
        """
        targets: Dict['State', Tuple[float, str]] = {}
        r = self.state_space.locus_config.recombination_rate

        # only recombine if there is more than one locus
        if self.state_space.locus_config.n == 1:
            return targets

        if isinstance(self.state_space, LineageCountingStateSpace):

            # iterate over demes
            for deme in range(state.n_demes):

                # make sure we have linked lineages that can recombine
                if np.all(state.linked[:, deme] > 0):
                    target = state.copy()
                    target.linked[:, deme] -= 1
                    rate = r * state.linked[0, deme, 0]

                    self.add_target(targets, target, cast(float, rate), 'recombination')

            return targets

        raise NotImplementedError(f'Recombination is not yet implemented for {self.state_space.__class__.__name__}.')


class State:
    """
    State utility class.
    """
    #: Axis for linkage.
    LINKAGE = 0

    #: Axis for loci.
    LOCUS = 1

    #: Axis for demes.
    DEME = 2

    #: Axis for lineage blocks.
    BLOCK = 3

    def __init__(self, data: (np.ndarray, np.ndarray)):
        """
        Initialize a state.

        :param data: State data.
        """
        #: State data
        self.data: Tuple[np.ndarray, np.ndarray] = data

    def __hash__(self) -> int:
        """
        Hash function.

        :return: Hash of the state.
        """
        return hash((self.data[0].tobytes(), self.data[1].tobytes()))

    def __eq__(self, other: 'State') -> bool:
        """
        Check if two states are equal.

        :param other: Other state.
        :return: Whether the two states are equal.
        """
        return hash(self) == hash(other)

    def copy(self) -> 'State':
        """
        Copy the state.

        :return: Copy of the state.
        """
        return State((self.data[0].copy(), self.data[1].copy()))

    def is_absorbing(self) -> bool:
        """
        Whether a state is absorbing.

        :return: Whether the state is absorbing.
        """
        return np.all(self.lineages.sum(axis=(1, 2)) == 1)

    @property
    def n_demes(self) -> int:
        """
        Get the number of demes.

        :return: The number of demes.
        """
        return self.lineages.shape[1]

    @property
    def n_loci(self) -> int:
        """
        Get the number of loci.

        :return: The number of loci.
        """
        return self.lineages.shape[0]

    @property
    def n_blocks(self) -> int:
        """
        Get the number of lineage blocks.

        :return: The number of lineage blocks.
        """
        return self.lineages.shape[2]

    @property
    def lineages(self) -> np.ndarray:
        """
        Get the number of lineages.

        :return: The number of lineages.
        """
        return self.data[0]

    @property
    def linked(self) -> np.ndarray:
        """
        Get the number of linked lineages.

        :return: The number of linked lineages.
        """
        return self.data[1]

    @property
    def unlinked(self) -> np.ndarray:
        """
        Get the number of unlinked lineages.

        :return: The number of unlinked lineages.
        """
        return self.lineages - self.linked
