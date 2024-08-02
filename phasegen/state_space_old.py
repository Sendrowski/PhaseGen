"""
State space.
"""

import itertools
import logging
import time
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from typing import List, Tuple, cast, Dict, Callable

import numpy as np
from scipy.special import comb

from .coalescent_models import CoalescentModel, StandardCoalescent
from .demography import Epoch
from .lineage import LineageConfig
from .locus import LocusConfig

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
        self._cache: Dict[Epoch, np.ndarray] = {}

        # number of lineages linked across loci
        self.linked: np.ndarray | None = None

        # time in seconds to compute original rate matrix
        self.time: float | None = None

    @cached_property
    def _non_zero_states(self) -> Tuple[np.ndarray, ...]:
        """
        Indices of non-zero rates.
        """
        # cache the current epoch
        epoch = self.epoch

        # we first determine the non-zero states by using default values for the demography
        self.epoch = Epoch(
            pop_sizes={p: 1 for p in epoch.pop_names},
            migration_rates={(p, q): 1 for p, q in product(epoch.pop_names, epoch.pop_names) if p != q}
        )

        start = time.time()

        # get the rate matrix for the default demography
        default_rate_matrix = np.fromfunction(
            np.vectorize(self._get_rate, otypes=[float]),
            (self.k, self.k),
            dtype=int
        )

        # record time to compute rate matrix
        self.time = time.time() - start

        # restore the epoch
        self.epoch = epoch

        # indices of non-zero rates
        # this improves performance when recomputing the rate matrix for different epochs
        return np.where(default_rate_matrix != 0)

    @cached_property
    @abstractmethod
    def states(self) -> np.ndarray:
        """
        The states. Each state describes the lineage configuration per deme and locus, i.e.,
        one state has the structure ``[[[a_ijk]]]`` where ``i`` is the lineage configuration, ``j`` is the deme
        and ``k`` is the locus.
        """
        pass

    @cached_property
    def e(self) -> np.ndarray:
        """
        Vector with ones of size `k`.
        """
        return np.ones(self.k)

    @cached_property
    def S(self) -> np.ndarray:
        """
        Intensity matrix.
        """
        # obtain intensity matrix
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
        Check if two state spaces are equal. We do not for equivalence of the epochs as we can
        update the epoch of a state space.

        :param other: Other state space
        :return: Whether the two state spaces are equal
        """
        return (
                self.__class__ == other.__class__ and
                self.lineage_config == other.lineage_config and
                self.locus_config == other.locus_config and
                self.model == other.model
        )

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
    def m(self) -> int:
        """
        Length of state vector for a single deme.
        """
        return self.states.shape[2]

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

    def _get_rate_matrix(self) -> np.ndarray:
        """
        Get the rate matrix.

        :return: The rate matrix.
        """
        # create empty matrix
        S = np.zeros((self.k, self.k))

        # check if we can use the cached rate matrix
        if self.cache and self.epoch in self._cache:
            S[self._non_zero_states] = self._cache[self.epoch]

        else:
            # vectorize function to compute rates
            get_rates = np.vectorize(self._get_rate, otypes=[float])

            # get non-zero rates
            rates = get_rates(*self._non_zero_states)

            # fill matrix with non-zero rates
            S[self._non_zero_states] = rates

            # cache rate matrix if specified
            if self.cache:
                self._cache[self.epoch] = rates

        # fill diagonal with negative sum of row
        S[np.diag_indices_from(S)] = -np.sum(S, axis=1)

        return S

    def _get_sparsity(self) -> float:
        """
        Get the sparsity of the rate matrix.

        :return: The sparsity.
        """
        return 1 - np.count_nonzero(self.S) / self.S.size

    @abstractmethod
    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state `s1` to state `s2`.

        :param n: Number of lineages.
        :param s1: Block configuration of state 1.
        :param s2: Block configuration of state 2.
        :return: The coalescent rate from state `s1` to state `s2`.
        """
        pass

    def _get_rate(self, i: int, j: int) -> float:
        """
        Get the rate from the state indexed by i to the state indexed by j.

        :param i: Index of outgoing state.
        :param j: Index of incoming state.
        :return: The rate from the state indexed by i to the state indexed by j.
        """
        return self._get_transition(i=i, j=j).get_rate()

    def _get_transition(self, i: int, j: int) -> 'Transition':
        """
        Get the transition from the state indexed by i to the state indexed by j.

        :param i: Index of outgoing state.
        :param j: Index of incoming state.
        :return: The transition from the state indexed by i to the state indexed by j.
        """
        return Transition(
            marginal1=self.states[i],
            marginal2=self.states[j],
            linked1=self.linked[i],
            linked2=self.linked[j],
            state_space=self
        )

    def _get_color_state(self, i: int) -> str:
        """
        Get color of the state indexed by `i`.
        """
        if State.is_absorbing(self.states[i]):
            return '#f1807e'

        if self.alpha[i] > 0:
            return 'lightgreen'

        return 'lightblue'

    def _plot_rates(
            self,
            file: str,
            view: bool = True,
            cleanup: bool = False,
            dpi: int = 400,
            ratio: float = 0.6,
            extension: str = 'png',
            format_state: Callable[[np.array], str] = None,
            format_transition: Callable[['Transition'], str] = None
    ):
        """
        Plot the rate matrix using graphviz.

        :param file: File to save plot to.
        :param view: Whether to view the plot.
        :param cleanup: Whether to remove the source file.
        :param dpi: Dots per inch.
        :param ratio: Aspect ratio.
        :param extension: File format.
        :param format_state: Function to format state with state array as argument.
        :param format_transition: Function to format transition with transition as argument.
        """
        if format_state is None:
            def format_state(s: np.ndarray) -> str:
                """
                Format state.

                :param s: State.
                :return: Formatted state.
                """
                return str(s[0]).replace('\n', '') + '\n' + str(s[1]).replace('\n', '')

        if format_transition is None:
            def format_transition(t: 'Transition') -> str:
                """
                Format transition.

                :param t: Transition.
                :return: Formatted transition.
                """
                return f' {t.type}: {t.get_rate():.2g}'

        import graphviz

        graph = graphviz.Digraph()

        # add nodes
        for i in range(len(self.states)):
            graph.node(
                name=format_state(np.array([self.states[i], self.linked[i]])),
                fillcolor=self._get_color_state(i),
                style='filled'
            )

        # add non-zero edges
        for i, j in zip(*self._non_zero_states):

            t = self._get_transition(i=i, j=j)

            if not State.is_absorbing(t.marginal1):
                graph.edge(
                    tail_name=format_state(np.array([self.states[i], self.linked[i]])),
                    head_name=format_state(np.array([self.states[j], self.linked[j]])),
                    label=format_transition(t),
                    color=t._get_color(),
                    fontcolor=t._get_color()
                )

        graph.graph_attr['dpi'] = str(dpi)
        graph.graph_attr['ratio'] = str(ratio)

        graph.render(
            filename=file,
            view=view,
            cleanup=cleanup,
            format=extension
        )

    @staticmethod
    def _find_vectors(n: int, k: int) -> List[List[int]]:
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
            for vector in StateSpace._find_vectors(n - i, k - 1):
                vectors.append(vector + [i])

        return vectors

    @staticmethod
    def p(n: int, k: int) -> int:
        """
        Partition function. Get number of ways to partition `n` into `k` positive integers.

        :param n: Number to partition.
        :param k: Number of parts.
        :return: Number of ways to partition `n` into `k` positive integers.
        """
        return comb(n - 1, k - 1, exact=True)

    @classmethod
    def p0(cls, n: int, k: int) -> int:
        """
        Partition function. Get number of ways to partition `n` into `k` non-negative integers.

        :param n: Number to partition.
        :param k: Number of parts.
        :return: Number of ways to partition `n` into `k` non-negative integers.
        """
        return cls.p(n + k, k)

    @staticmethod
    def P(n: int) -> int:
        """
        Calculate the number of partitions of a non-negative integer.

        :param n: The non-negative integer to partition.
        :type n: int
        :return: The number of partitions of n.
        :rtype: int
        """
        partitions: List[int] = [0] * (n + 1)
        partitions[0] = 1

        for i in range(1, n + 1):
            for j in range(i, n + 1):
                partitions[j] += partitions[j - i]

        return partitions[n]

    def _get_outgoing_rates(self, i: int, remove_zero: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the outgoing rates from state indexed by `i`.

        :param i: Index of outgoing state.
        :param remove_zero: Whether to remove zero rates.
        :return: The outgoing rates and the indices of the states to which the rates correspond.
        """
        rates = self.S[i, :]
        indices = np.arange(self.k)

        # mask diagonal
        mask = np.arange(self.k) != i

        if remove_zero:
            mask &= rates != 0

        rates = rates[mask]
        indices = indices[mask]

        return rates, indices


class LineageCountingStateSpace(StateSpace):
    """
    Default rate matrix where there is one state per number of lineages for each deme and locus.
    """

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state `s1` to state `s2`.

        :param n: Number of lineages.
        :param s1: Block configuration of state 1.
        :param s2: Block configuration of state 2.
        :return: The coalescent rate from state `s1` to state `s2`.
        """
        return self.model.get_rate(s1=s1[0], s2=s2[0])

    def _expand_loci(self, states: np.ndarray) -> np.ndarray:
        """
        Expand the given states to include all possible combinations of locus configurations.

        :param states: States.
        """
        if self.locus_config.n == 1:
            # add extra dimension for locus configuration
            states = states[:, np.newaxis]

            # no lineages are linked
            self.linked = np.zeros_like(states, dtype=int)

            return states

        if self.locus_config.n == 2:
            n_pops = self.lineage_config.n_pops

            # determine number of linked lineage configurations irrespective of states
            linked_locus = np.array(list(itertools.product(range(self.lineage_config.n + 1), repeat=n_pops)))
            linked_locus = linked_locus[linked_locus.sum(axis=1) <= self.lineage_config.n]

            # expand loci, each deme needs to have the same number of linked lineages
            linked = np.repeat(linked_locus[:, np.newaxis], 2, axis=1)

            # add extra dimension for lineage blocks
            linked = linked[..., np.newaxis]

            # take product of number of linked lineages and states
            states_new = np.array(list(itertools.product(linked, itertools.product(states, states))))

            # remove states where `linked` is larger than marginal states
            states_new = states_new[(states_new[:, 0] <= states_new[:, 1]).all(axis=(1, 2, 3))]

            self.linked = states_new[:, 0, :, :]

            return states_new[:, 1, :, :]

        raise NotImplementedError("Only 1 or 2 loci are currently supported.")

    @cached_property
    def states(self) -> np.ndarray:
        """
        The states. Each state describes the lineage configuration per deme and locus, i.e.,
        one state has the structure `[[[a_ijk]]]` where `i` is the lineage configuration, `j` is the deme and `k` is
        the locus.
        """
        # the number of lineages
        lineages = np.arange(1, self.lineage_config.n + 1)[::-1]

        # iterate lineage configurations and find all possible deme configurations
        states = []
        for i in lineages:
            states += self._find_vectors(n=i, k=self.epoch.n_pops)

        # convert to numpy array
        states = np.array(states)

        # add extra dimension for lineage configuration
        states = states.reshape(states.shape + (1,))

        # expand the states to include all possible combinations of locus configurations
        states = self._expand_loci(states)

        return states

    def get_k(self) -> int:
        """
        Get number of states.
        Currently no support for multiple loci.

        :return: The number of states.
        """
        n = self.lineage_config.n
        d = self.lineage_config.n_pops

        i = np.arange(1, n + 1)[::-1]

        k = np.sum([self.p0(j, d) for j in i])

        return k


class BlockCountingStateSpace(StateSpace):
    r"""
    Rate matrix for block-counting state space where there is one state per sample configuration:
    :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`,

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

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state `s1` to state `s2`.

        :param n: Number of lineages.
        :param s1: Block configuration of state 1.
        :param s2: Block configuration of state 2.
        :return: The coalescent rate from state `s1` to state `s2`.
        """
        return self.model.get_rate_block_counting(n=n, s1=s1, s2=s2)

    def _expand_loci(self, states: np.ndarray) -> np.ndarray:
        """
        Expand the given states to include all possible combinations of locus configurations.
        Two-locus state space not sufficient for computing SFS as lineages are not longer
          exchangeable in this case.

        :param states: States.
        """
        if self.locus_config.n == 1:
            # add extra dimension for locus configuration
            states = states[:, np.newaxis]

            # all lineages are linked
            self.linked = np.zeros_like(states, dtype=int)

            # add extra dimension for locus configuration
            return states

        raise NotImplementedError("Only 1 locus is currently supported.")

        if self.locus_config.n == 2:

            locus = []

            for state in states.reshape(states.shape[0], -1):

                linked = []
                for block in state:
                    linked += [range(block + 1)]

                locus += itertools.product([state], itertools.product(*linked))

            # combine loci
            expanded = np.array(list(itertools.product(locus, repeat=2)))

            # reshape two last dimensions to get the correct shape
            expanded = expanded.reshape(*expanded.shape[:3], *states.shape[1:])

            # states for which the number of linked lineages is the same across loci
            same_linked = expanded[:, 0, 1].sum(axis=(1, 2)) == expanded[:, 1, 1].sum(axis=(1, 2))

            # remove states where the number of linked lineages is not the same across loci
            expanded = expanded[same_linked]

            self.linked = expanded[:, :, 1]

            return expanded[:, :, 0]

    @cached_property
    def states(self) -> np.ndarray:
        """
        The states. Each state describes the lineage configuration per deme and locus, i.e.,
        one state has the structure `[[[a_ijk]]]` where `i` is the lineage configuration, `j` is the deme and `k` is
        the locus.
        """
        # the possible allele configurations
        lineage_configs = np.array(self._find_sample_configs(m=self.lineage_config.n, n=self.lineage_config.n))

        # iterate over possible allele configurations and find all possible deme configurations
        states = []
        for config in lineage_configs:

            # iterate over possible number of lineages with multiplicity k and
            # find all possible deme configurations
            vectors = []
            for i in config:
                vectors += [self._find_vectors(n=i, k=self.epoch.n_pops)]

            # find all possible combinations of deme configurations for each multiplicity
            states += list(product(*vectors))

        # transpose the array to have the deme configurations as columns
        states = np.transpose(np.array(states), (0, 2, 1))

        # expand the states to include all possible combinations of locus configurations
        states = self._expand_loci(states)

        return states

    @classmethod
    def _find_sample_configs(cls, m: int, n: int) -> List[List[int]]:
        """
        Function to find all vectors of length m such that sum_{i=0}^{m} i*x_{m-i} equals n.

        :param m: Length of the vectors.
        :param n: Target sum.
        :returns: list of vectors satisfying the condition
        """
        # base case, when the length of vector is 0
        # if n is also 0, return an empty vector, otherwise no solutions
        if m == 0:
            return [[]] if n == 0 else []

        vectors = []
        # iterate over possible values for the first component
        for x in range(n // m + 1):  # Adjusted for 1-based index
            # recursively find vectors with one less component and a smaller target sum
            for vector in cls._find_sample_configs(m - 1, n - x * m):  # Adjusted for 1-based index
                # prepend the current component to the recursively found vectors
                vectors.append(vector + [x])  # Reversed vectors

        return vectors

    def get_k(self) -> int:
        """
        Get number of states.

        :return: The number of states.
        """
        n = self.lineage_config.n
        d = self.lineage_config.n_pops

        # len(i) == self.P(n)
        i = np.array(self._find_sample_configs(m=n, n=n))

        k = np.sum([np.prod([self.p0(l, d) for l in j]) for j in i])

        return k


class Transition:
    """
    Class representing a transition between two states.
    """
    #: Colors for different types of transitions
    _colors: Dict[str, str] = {
        'recombination': 'orange',
        'locus_coalescence': 'darkgreen',
        'linked_coalescence': 'darkgreen',
        'unlinked_coalescence': 'darkgreen',
        'mixed_coalescence': 'darkgreen',
        'unlinked_coalescence+mixed_coalescence': 'darkgreen',
        'linked_migration': 'blue',
        'unlinked_migration': 'blue',
        'invalid': 'red'
    }

    #: Event types
    _event_types: List[str] = [
        'recombination',
        'locus_coalescence',
        'linked_coalescence',
        'unlinked_coalescence',
        'mixed_coalescence',
        'linked_migration',
        'unlinked_migration'
    ]

    def __init__(
            self,
            state_space: StateSpace,
            marginal1: np.ndarray,
            marginal2: np.ndarray,
            linked1: np.ndarray,
            linked2: np.ndarray
    ):
        """
        Initialize a transition.

        :param state_space: State space.
        :param marginal1: Marginal lineages in outgoing state.
        :param marginal2: Marginal lineages in incoming state.
        :param linked1: Numbers of linked lineages in outgoing state.
        :param linked2: Numbers of linked lineages in incoming state.
        """
        #: State space.
        self.state_space: StateSpace = state_space

        #: Marginal lineages in outgoing state.
        self.marginal1: np.ndarray = marginal1

        #: Marginal lineages in incoming state.
        self.marginal2: np.ndarray = marginal2

        #: linked lineages in outgoing state.
        self.linked1: np.ndarray = linked1

        #: linked lineages in incoming state.
        self.linked2: np.ndarray = linked2

    @cached_property
    def unlinked1(self) -> np.ndarray:
        """
        Unlinked lineages in outgoing state.
        """
        return self.marginal1 - self.linked1

    @cached_property
    def unlinked2(self) -> np.ndarray:
        """
        Unlinked lineages in incoming state.
        """
        return self.marginal2 - self.linked2

    @cached_property
    def diff_marginal(self) -> np.ndarray:
        """
        Difference between marginal lineages.
        """
        return self.marginal1 - self.marginal2

    @cached_property
    def diff_linked(self) -> np.ndarray:
        """
        Difference in linked lineages.
        """
        return self.linked1 - self.linked2

    @cached_property
    def diff_unlinked(self) -> np.ndarray:
        """
        Difference in unlinked lineages.
        """
        return self.unlinked1 - self.unlinked2

    @cached_property
    def n_loci(self) -> int:
        """
        Number of loci.
        """
        return self.state_space.locus_config.n

    @cached_property
    def n_blocks(self) -> int:
        """
        Number of lineage blocks.
        """
        return self.marginal1.shape[2]

    @cached_property
    def n_demes_marginal(self) -> int:
        """
        Number of affected demes with respect to marginal lineages.
        """
        return self.is_diff_demes_marginal.sum()

    @cached_property
    def n_diff_loci_deme_coal(self) -> int:
        """
        Number of loci where coalescence event occurs in deme where coalescence event occurs.
        """
        if not self.has_diff_marginal:
            return 0

        return self.is_diff_loci_deme_coal.sum()

    @cached_property
    def is_diff_demes_marginal(self) -> np.ndarray:
        """
        Mask for demes with affected lineages.
        """
        return np.any(self.diff_marginal != 0, axis=(0, 2))

    @cached_property
    def is_diff_demes_linked(self) -> np.ndarray:
        """
        Mask for demes with affected linked lineages.
        """
        return np.any(self.diff_linked != 0, axis=(0, 2))

    @cached_property
    def is_diff_loci(self) -> np.ndarray:
        """
        Mask for affected loci with respect to marginal lineages.
        """
        return np.any(self.diff_marginal != 0, axis=(1, 2))

    @cached_property
    def is_diff_loci_deme_coal(self) -> np.ndarray:
        """
        Mask for affected loci with respect to deme where coalescence event occurs.
        """
        return np.any(self.diff_marginal[:, self.deme_coal] != 0, axis=1)

    @cached_property
    def has_diff_demes_linked(self) -> bool:
        """
        Whether there are any affected linked lineages.
        """
        return cast(bool, self.is_diff_demes_linked.any())

    @cached_property
    def has_diff_marginal(self) -> bool:
        """
        Whether there are any affected marginal lineages.
        """
        return cast(bool, self.is_diff_demes_marginal.any())

    @cached_property
    def deme_coal(self) -> int:
        """
        Index of deme where coalescence event occurs.
        """
        return cast(int, np.where(self.is_diff_demes_marginal)[0][0])

    @cached_property
    def locus_coal_unlinked(self) -> int:
        """
        Index of locus where unlinked coalescence event occurs.
        """
        return cast(int, np.where(self.is_diff_loci)[0][0])

    @cached_property
    def locus_migration(self) -> int:
        """
        Index of first locus where a migration event occurs.
        """
        return cast(int, np.where(self.diff_marginal.any(axis=(1, 2)))[0][0])

    @cached_property
    def deme_migration_source(self) -> int:
        """
        Get the source deme of the migration event.
        """
        return int(np.where((self.diff_marginal[self.locus_migration] == 1).sum(axis=1) == 1)[0][0])

    @cached_property
    def deme_migration_dest(self) -> int:
        """
        Get the destination deme of the migration event.
        """
        return int(np.where((self.diff_marginal[self.locus_migration] == -1).sum(axis=1) == 1)[0][0])

    @cached_property
    def block_migration(self) -> int:
        """
        Get the index of the lineage block where the migration event occurs.
        """
        return int(np.where(self.diff_marginal[self.locus_migration] == 1)[1][0])

    @cached_property
    def is_absorbing(self) -> bool:
        """
        Whether either the outgoing or incoming state is absorbing.
        """
        return State.is_absorbing(self.marginal1) or State.is_absorbing(self.marginal2)

    @cached_property
    def is_eligible_recombination_or_locus_coalescence(self) -> bool:
        """
        Whether the transition is eligible for a recombination or locus coalescence event.
        """
        # there have to be affected lineages
        if self.has_diff_marginal:
            return False

        # there have to be exactly `n_loci` affected lineages
        if not np.all((self.diff_linked == 0).sum() == self.linked1.size - self.n_loci):
            return False

        # make sure change in linked lineages is in the same deme for each locus
        demes = np.where(self.diff_linked != 0)[1]
        if not np.all(demes == demes[0]):
            return False

        # not possible from or to absorbing state
        if self.is_absorbing:
            return False

        return True

    @cached_property
    def is_recombination(self) -> bool:
        """
        Whether transition is a recombination event.
        """
        # if not eligible for recombination, it can't be a recombination event
        if not self.is_eligible_recombination_or_locus_coalescence:
            return False

        # if there is not exactly one more linked lineage in state 1 than in state 2 for each locus,
        # it can't be a recombination event
        if not np.all((self.diff_linked == 1).sum(axis=(1, 2)) == 1):
            return False

        return True

    @cached_property
    def is_locus_coalescence(self) -> bool:
        """
        Whether the transition is a locus coalescence event.
        """
        # if not eligible for recombination, it can't be a locus coalescence
        if not self.is_eligible_recombination_or_locus_coalescence:
            return False

        # if there is not exactly one more lineage in state 2 than in state 1 for each locus,
        # it can't be a recombination event
        if not np.all((self.diff_linked == -1).sum(axis=(1, 2)) == 1):
            return False

        return True

    @cached_property
    def is_eligible_coalescence(self) -> bool:
        """
        Whether the transition is eligible for a coalescence event.
        """
        # if not exactly one deme is affected, it can't be a coalescence event
        return self.n_demes_marginal == 1

    @cached_property
    def is_eligible_linked_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for linked coalescence.
        """
        return self.n_diff_loci_deme_coal > 1

    @cached_property
    def is_eligible_unlinked_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for unlinked coalescence.
        """
        if self.n_diff_loci_deme_coal != 1:
            return False

        if self.has_diff_demes_linked:
            return False

        if self.unlinked1[self.locus_coal_unlinked, self.deme_coal].sum() < 2:
            return False

        return True

    @cached_property
    def is_eligible_mixed_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for mixed coalescence.
        """
        if self.n_diff_loci_deme_coal != 1:
            return False

        if self.n_blocks == 1:
            if self.has_diff_demes_linked:
                return False

            # we need at least two marginal lineages
            if self.marginal1[self.locus_coal_unlinked, self.deme_coal].sum() < 2:
                return False

            # we need at least one linked lineage
            if self.linked1[self.locus_coal_unlinked, self.deme_coal].sum() < 1:
                return False

            return True

        if np.sum(np.any(self.diff_linked != 0, axis=(1, 2))) != 1:
            return False

        return True

    @cached_property
    def is_eligible_recombination(self) -> bool:
        """
        Alias for `is_eligible_recombination_or_locus_coalescence`.
        """
        return self.is_eligible_recombination_or_locus_coalescence

    @cached_property
    def is_eligible_locus_coalescence(self) -> bool:
        """
        Alias for `is_eligible_recombination_or_locus_coalescence`.
        """
        return self.is_eligible_recombination_or_locus_coalescence

    @cached_property
    def is_eligible_migration(self) -> bool:
        """
        Whether the transition is eligible for a migration event.
        """
        # two demes must be affected
        return self.n_demes_marginal == 2

    @cached_property
    def is_eligible_linked_migration(self) -> bool:
        """
        Alias for `is_eligible_migration`.
        """
        return self.is_eligible_migration

    @cached_property
    def is_eligible_unlinked_migration(self) -> bool:
        """
        Alias for `is_eligible_migration`.
        """
        return self.is_eligible_migration

    @cached_property
    def is_eligible(self) -> bool:
        """
        Whether the transition is eligible for any event. This is supposed to rule out impossible
        transitions as quickly as possible.
        """
        if self.is_eligible_coalescence:
            return self.is_coalescence

        if self.is_eligible_recombination_or_locus_coalescence:
            return self.is_recombination or self.is_locus_coalescence

        if self.is_eligible_migration:
            return self.is_migration

        return False

    @cached_property
    def is_valid_lineage_reduction_linked_coalescence(self) -> bool:
        """
        In case of a linked coalescence event, whether the reduction in the number of linked lineages is equal
        to the reduction of marginal lineages.
        """
        return np.all(self.diff_linked == self.diff_marginal)

    @cached_property
    def has_sufficient_linked_lineages_linked_coalescence(self) -> bool:
        """
        In case of a linked coalescence event, whether the number of linked lineages is greater than
        equal to the number of linked coalesced lineages.
        """
        linked = self.linked1[:, self.deme_coal].sum(axis=1)
        coalesced = self.diff_marginal[:, self.deme_coal].sum(axis=1) + 1

        return np.all(linked >= coalesced)

    @cached_property
    def is_lineage_reduction(self) -> bool:
        """
        Whether we have a lineage reduction.
        """
        return self.diff_marginal.sum() > 0

    @cached_property
    def is_binary_lineage_reduction_mixed_coalescence(self) -> bool:
        """
        Whether the mixed coalescence event is a binary merger.
        """
        reduction = self.diff_unlinked[self.is_diff_loci_deme_coal][0][self.deme_coal]

        return reduction.sum() == 1

    @cached_property
    def is_valid_lineage_reduction_unlinked_coalescence(self) -> bool:
        """
        In an unlinked coalescence event, whether the reduction in the number of unlinked lineages is equal
        to the reduction in the number of coalesced lineages.
        """
        unlinked = self.diff_unlinked[self.is_diff_loci_deme_coal][0][self.deme_coal]
        diff = self.diff_marginal[self.locus_coal_unlinked, self.deme_coal]

        return unlinked.sum() == diff.sum()

    @cached_property
    def is_valid_lineage_reduction_mixed_coalescence(self) -> bool:
        """
        In a mixed coalescence event there has to be reduction in the number of linked lineages.
        """
        # in the lineage-counting state space, where we only keep track of the number of linked lineages per deme,
        # we may have a mixed coalescence event where the number of linked lineages does not change
        if self.n_blocks == 1 and not self.has_diff_demes_linked:
            return True

        diff_unlinked = self.diff_unlinked[self.locus_coal_unlinked, self.deme_coal]

        # make sure number of unlinked lineages is reduced by one in one lineage block
        if np.abs(diff_unlinked).sum() != 1:
            return False

        diff_linked = self.diff_linked[self.locus_coal_unlinked, self.deme_coal]

        # exactly one linked lineage must be lost in one block and one
        # linked lineage must be gained in another block
        if not (1 in diff_linked and -1 in diff_linked and np.abs(diff_linked).sum() == 2):
            return False

        return True

    @cached_property
    def is_linked_coalescence(self) -> bool:
        """
        Whether the coalescence event is a linked coalescence event, i.e. only linked lineages coalesce.
        """
        return (
                self.is_lineage_reduction and
                self.is_eligible_linked_coalescence and
                self.is_valid_lineage_reduction_linked_coalescence and
                self.has_sufficient_linked_lineages_linked_coalescence and
                self.is_eligible_coalescence
        )

    @cached_property
    def is_unlinked_coalescence(self) -> bool:
        """
        Whether the coalescence event is an unlinked coalescence event, i.e. only unlinked lineages coalesce.
        """
        return (
                self.is_lineage_reduction and
                self.is_eligible_unlinked_coalescence and
                self.is_valid_lineage_reduction_unlinked_coalescence and
                self.is_eligible_coalescence
        )

    @cached_property
    def is_mixed_coalescence(self) -> bool:
        """
        Whether the coalescence event is a mixed coalescence event, i.e. both linked and unlinked lineages coalesce.
        """
        return (
                self.n_loci > 1 and
                self.is_lineage_reduction and
                self.is_eligible_mixed_coalescence and
                self.is_binary_lineage_reduction_mixed_coalescence and
                self.is_valid_lineage_reduction_mixed_coalescence and
                self.is_eligible_coalescence
        )

    @cached_property
    def is_coalescence(self) -> bool:
        """
        Whether the transition is a coalescence event (except for a locus coalescence).
        """
        return (
                self.is_linked_coalescence or
                self.is_unlinked_coalescence or
                self.is_mixed_coalescence
        )

    @cached_property
    def is_valid_migration_one_locus_only(self) -> bool:
        """
        Whether the migration event is only affecting one locus.
        """
        return self.is_diff_loci.sum() == 1

    @cached_property
    def is_valid_linked_migration(self) -> bool:
        """
        Whether the migration event is a valid linked migration event.
        """
        # number of affected demes must be 2 for linked lineages
        if np.any(self.diff_linked != 0, axis=(0, 2)).sum() != self.n_loci:
            return False

        # difference in linked lineages and marginal lineages must be the same
        if not np.all(self.diff_marginal == self.diff_linked):
            return False

        # difference across marginal lineages must be some for all loci
        if not np.all(self.diff_marginal == self.diff_marginal[0]):
            return False

        # difference across linked lineages must be some for all loci
        if not np.all(self.diff_linked == self.diff_linked[0]):
            return False

        return True

    @cached_property
    def is_one_migration_event(self) -> bool:
        """
        Whether there is exactly one migration event.
        """
        diff = self.diff_marginal[self.locus_migration]

        # make sure exactly one lineage is moved from one deme to another
        return (
                (diff == 1).sum() == 1 and
                (diff == -1).sum() == 1 and
                (diff == 0).sum() == diff.size - 2
        )

    @cached_property
    def has_sufficient_linked_lineages_migration(self) -> bool:
        """
        Whether there are sufficient linked lineages to allow for a migration event.
        """
        lineages = self.linked1[
            self.locus_migration,
            self.deme_migration_source,
            self.block_migration
        ]

        return cast(bool, lineages > 0)

    @cached_property
    def has_sufficient_unlinked_lineages_migration(self) -> bool:
        """
        Whether there are sufficient unlinked lineages to allow for a migration event.
        """
        lineages = self.unlinked1[
            self.locus_migration,
            self.deme_migration_source,
            self.block_migration
        ]

        return cast(bool, lineages > 0)

    @cached_property
    def is_unlinked_migration(self) -> bool:
        """
        Whether the transition is a migration event.
        """
        return (
                not self.has_diff_demes_linked and
                self.is_valid_migration_one_locus_only and
                self.is_one_migration_event and
                self.has_sufficient_unlinked_lineages_migration and
                self.is_eligible_migration
        )

    @cached_property
    def is_linked_migration(self) -> bool:
        """
        Whether the migration event is a linked migration event, i.e. a linked lineage migrates.
        """
        return (
                self.is_one_migration_event and
                self.has_sufficient_linked_lineages_migration and
                self.has_diff_demes_linked and
                self.is_valid_linked_migration and
                self.is_eligible_migration
        )

    @cached_property
    def is_migration(self) -> bool:
        """
        Whether the transition is a migration event.
        """
        return self.is_unlinked_migration or self.is_linked_migration

    @cached_property
    def type(self) -> str:
        """
        Get the type of transition.
        """
        types = []

        for t in self._event_types:
            if getattr(self, f'is_eligible_{t}') and getattr(self, f'is_{t}'):
                types.append(t)

        return '+'.join(types) or 'invalid'

    def get_rate_recombination(self) -> float:
        """
        Get the rate of a recombination event.
        Here we assume the number of linked lineages is the same across loci which should be the case.
        """
        # number of linked lineages need not be the same across loci for different lineage blocks
        linked1 = self.linked1[self.diff_linked == 1]

        rate = linked1[0] * self.state_space.locus_config.recombination_rate

        return cast(float, rate)

    def get_rate_locus_coalescence(self) -> float:
        """
        Get the rate of a locus coalescence event.
        """
        # return 0 if locus coalescence is not allowed
        if not self.state_space.locus_config._allow_coalescence:
            return 0

        # get unlinked lineage counts
        unlinked1 = self.unlinked1[self.diff_linked == -1]

        # index of deme where linked coalescence event occurs.
        deme_coal = np.where(self.is_diff_demes_linked)[0][0]

        # get population size of deme where coalescence event occurs
        pop_size = self.state_space.epoch.pop_sizes[self.state_space.epoch.pop_names[deme_coal]]

        # scale population size
        pop_size_scaled = self.state_space.model._get_timescale(pop_size)

        return unlinked1.prod() / pop_size_scaled

    def get_pop_size_coalescence(self) -> float:
        """
        Get the population size of the deme where the coalescence event occurs.
        """
        return self.state_space.epoch.pop_sizes[self.state_space.epoch.pop_names[self.deme_coal]]

    def get_scaled_pop_size_coalescence(self) -> float:
        """
        Get the scaled population size of the deme where the coalescence event occurs.
        """
        return self.state_space.model._get_timescale(self.get_pop_size_coalescence())

    def get_rate_linked_coalescence(self) -> float:
        """
        Get the rate of a linked coalescence event.

        It seems as though the current parametrization does not allow us to compute the site-frequency spectrum for
        more than one locus. The problem is that initially if all lineages are linked, transitions with different
        coalescent patterns between loci are not allowed. However, once we have experienced a recombination event,
        a mixed or unlinked coalescence event, and a subsequent locus coalescence event, we can have different
        coalescent patterns between loci. We would thus be required to keep track of the associations between
        lineages which would further expand the state space.

        For example, let there be n > 2 lineages, and two loci. Assume we start with completely linked loci.
        Now assume there is a linked coalescence event, so that we have 1 linked doubleton and n - 2 linked singletons.
        Now conditional on the fact that no recombination even has occurred, we can cannot have linked coalescence
        where one of the lineages is a doubleton in one locus and a singleton in the other locus. However, assume we
        first experience n recombination events so that our loci are now completely unlinked. Now assume we have an
        unlinked coalescence event in each locus, and subsequently n - 1 locus coalescence events. Now the state looks
        identical to the first scenario but it should be allowed to have a linked coalescence event where one of the
        lineages is a doubleton in one locus and a singleton in the other locus, given that an unlinked doubleton
        coalesced with an unlinked singleton. We thus need to keep track of the associations between lineages, which
        means they are no longer exchangeable.
        """
        return self.state_space._get_coalescent_rate(
            n=self.state_space.lineage_config.n,
            s1=self.linked1[0, self.deme_coal],
            s2=self.linked2[0, self.deme_coal]
        ) / self.get_scaled_pop_size_coalescence()

        # if (
        #        np.all(self.linked1[:, self.deme_coal] == self.linked1[0, self.deme_coal]) and
        #        np.any(self.linked2[:, self.deme_coal] != self.linked2[0, self.deme_coal])
        # ):
        #    return 0

        # rates = np.zeros(self.n_loci)
        # for i in range(self.n_loci):
        #    rates[i] = self.state_space._get_coalescent_rate(
        #        n=self.state_space.lineage_config.n,
        #        s1=self.linked1[i, self.deme_coal],
        #        s2=self.linked2[i, self.deme_coal]
        #    )

        # return rates.min() / self.get_scaled_pop_size_coalescence()

    def get_rate_unlinked_coalescence(self) -> float:
        """
        Get the rate of an unlinked coalescence event.
        """
        unlinked1 = self.unlinked1[self.locus_coal_unlinked, self.deme_coal]
        unlinked2 = self.unlinked2[self.locus_coal_unlinked, self.deme_coal]

        rate = self.state_space._get_coalescent_rate(
            n=self.state_space.lineage_config.n,
            s1=unlinked1,
            s2=unlinked2
        )

        return rate / self.get_scaled_pop_size_coalescence()

    def get_rate_mixed_coalescence(self) -> float:
        """
        Get the rate of a mixed coalescence event.
        """
        unlinked1 = self.unlinked1[self.locus_coal_unlinked, self.deme_coal]
        linked1 = self.linked1[self.locus_coal_unlinked, self.deme_coal]

        # lineage blocks where coalescence event occurs
        blocks = self.diff_marginal[self.locus_coal_unlinked, self.deme_coal] > 0

        n_blocks = blocks.sum()

        if n_blocks == 1:
            rates_cross = unlinked1[blocks] * linked1[blocks]
        elif n_blocks == 2:
            rates_cross = [unlinked1[blocks][0] * linked1[blocks][1], unlinked1[blocks][1] * linked1[blocks][0]]
        else:
            raise ValueError('Invalid number of blocks.')

        return np.sum(rates_cross) / self.get_scaled_pop_size_coalescence()

    def get_rate_unlinked_migration(self) -> float:
        """
        Get the rate of an unlinked migration event which happens marginally on one locus.
        """
        # get the deme names
        source = self.state_space.epoch.pop_names[self.deme_migration_source]
        dest = self.state_space.epoch.pop_names[self.deme_migration_dest]

        # get the number of lineages in deme i before migration
        n_lineages_source = self.unlinked1[
            self.locus_migration,
            self.deme_migration_source,
            self.block_migration
        ]

        # get migration rate from source to destination
        migration_rate = self.state_space.epoch.migration_rates[(source, dest)]

        # scale migration rate by number of lineages in source deme
        rate = migration_rate * n_lineages_source

        return cast(float, rate)

    def get_rate_linked_migration(self) -> float:
        """
        Get the rate of a linked migration event where a linked lineage migrates.
        """
        # get the deme names
        source = self.state_space.epoch.pop_names[self.deme_migration_source]
        dest = self.state_space.epoch.pop_names[self.deme_migration_dest]

        # get the number of lineages in deme i before migration
        n_lineages_source = self.linked1[:, self.deme_migration_source, self.block_migration].min()

        # get migration rate from source to destination
        migration_rate = self.state_space.epoch.migration_rates[(source, dest)]

        # scale migration rate by number of lineages in source deme
        rate = migration_rate * n_lineages_source

        return rate

    def get_rate(self) -> float:
        """
        Get the rate of the transition.
        """
        if not self.is_eligible:
            return 0

        if self.is_recombination:
            return self.get_rate_recombination()

        if self.is_locus_coalescence:
            return self.get_rate_locus_coalescence()

        if self.is_linked_coalescence:
            return self.get_rate_linked_coalescence()

        if self.is_linked_migration:
            return self.get_rate_linked_migration()

        if self.is_unlinked_migration:
            return self.get_rate_unlinked_migration()

        # From here on we may have both unlinked and mixed coalescence simultaneously,
        # if using the lineage-counting state space.
        rate = 0

        if self.is_unlinked_coalescence:
            rate += self.get_rate_unlinked_coalescence()

        if self.is_mixed_coalescence:
            rate += self.get_rate_mixed_coalescence()

        return rate

    def _get_color(self) -> str:
        """
        Get the color of the transition indicating the type of event.

        :return: The color of the transition.
        """
        return self._colors[self.type]


class State:
    """
    State utility class.
    """
    #: Axis for loci.
    LOCUS = 0

    #: Axis for demes.
    DEME = 1

    #: Axis for lineage blocks.
    BLOCK = 2

    @staticmethod
    def is_absorbing(state: np.ndarray) -> bool:
        """
        Whether a state is absorbing.

        :param state: State array.
        :return: Whether the state is absorbing.
        """
        return np.all(np.sum(state * np.arange(1, state.shape[2] + 1)[::-1], axis=(1, 2)) == 1)
