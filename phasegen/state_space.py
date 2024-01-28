import itertools
import logging
from abc import ABC, abstractmethod
import time
from functools import cached_property
from itertools import product
from typing import List, Tuple

import numpy as np

from .coalescent_models import CoalescentModel, StandardCoalescent
from .demography import Epoch
from .lineage import LineageConfig
from .locus import LocusConfig
from .transition import Transition
from .state import State
from .utils import expm

logger = logging.getLogger('phasegen')


class StateSpace(ABC):
    """
    State space.
    """

    def __init__(
            self,
            pop_config: LineageConfig,
            locus_config: LocusConfig = LocusConfig(),
            model: CoalescentModel = StandardCoalescent(),
            epoch: Epoch = Epoch()
    ):
        """
        Create a rate matrix.

        :param pop_config: Population configuration.
        :param model: Coalescent model.
        :param epoch: Time homogeneous demography (we can only construct a state space
            for a fixed demography).
        """
        if not isinstance(model, StandardCoalescent) and locus_config.n > 1:
            raise NotImplementedError('Only one locus is currently supported for non-standard coalescent models.')

        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: Coalescent model
        self.model: CoalescentModel = model

        #: Population configuration
        self.pop_config: LineageConfig = pop_config

        #: Locus configuration
        self.locus_config: LocusConfig = locus_config

        #: Epoch
        self.epoch: Epoch = epoch

        # number of lineages linked across loci
        self.linked: np.ndarray | None = None

        # time in seconds to compute original rate matrix
        self.time: float | None = None

        # warn if state space is large
        if self.k > 2000:
            self._logger.warning(f'State space is large ({self.k} states). Note that the computation time '
                                 f'increases exponentially with the number of states.')

    @cached_property
    def _non_zero_states(self) -> Tuple[np.ndarray, ...]:
        """
        Get the indices of non-zero rates. This improves performance when computing the rate matrix.

        :return: Indices of non-zero rates.
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
            np.vectorize(self._matrix_indices_to_rates, otypes=[float]),
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
        Get the states. Each state describes the lineage configuration per deme and locus, i.e.
        one state has the structure [[[a_ijk]]] where i is the lineage configuration, j is the deme and k is the locus.
        """
        pass

    @cached_property
    def e(self) -> np.ndarray:
        """
        Vector with ones of size ``n``.
        """
        return np.ones(self.S.shape[0])

    @cached_property
    def S(self) -> np.ndarray:
        """
        Get full intensity matrix.
        """
        # obtain intensity matrix
        return self._get_rate_matrix()

    @cached_property
    def alpha(self) -> np.ndarray:
        """
        Initial state vector.
        """
        pops = self.pop_config.get_initial_states(self)
        loci = self.locus_config.get_initial_states(self)

        # combine initial states
        alpha = pops * loci

        # return normalized vector
        # normalization ensures that the initial state vector is a probability distribution
        # as we may have multiple initial states
        return alpha / alpha.sum()

    def update_epoch(self, epoch: Epoch):
        """
        Update the epoch.

        TODO check for equality of epoch

        :param epoch: Epoch.
        :return: State space.
        """
        # update the demography
        self.epoch = epoch

        # try to delete rate matrix cache
        try:
            # noinspection all
            del self.S
        except AttributeError:
            pass

        # try to delete transition matrix cache
        try:
            # noinspection all
            del self.T
        except AttributeError:
            pass

    @cached_property
    def k(self) -> int:
        """
        Get number of states.

        :return: The number of states.
        """
        return len(self.states)

    @cached_property
    def m(self) -> int:
        """
        Length of the state vector for a single deme.

        :return: The length
        """
        return self.states.shape[2]

    def _get_rate_matrix(self) -> np.ndarray:
        """
        Get the rate matrix.

        :return: The rate matrix.
        """
        matrix_indices_to_rates = np.vectorize(self._matrix_indices_to_rates, otypes=[float])

        # create empty matrix
        S = np.zeros((self.k, self.k))

        # fill matrix with non-zero rates
        S[self._non_zero_states] = matrix_indices_to_rates(*self._non_zero_states)

        # fill diagonal with negative sum of row
        S[np.diag_indices_from(S)] = -np.sum(S, axis=1)

        return S

    def _get_sparseness(self) -> float:
        """
        Get the sparseness of the rate matrix.

        :return: The sparseness.
        """
        return 1 - np.count_nonzero(self.S) / self.S.size

    @cached_property
    def T(self) -> np.ndarray:
        """
        The transition matrix.

        :return: The transition matrix.
        """
        return expm(self.S)

    @abstractmethod
    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
        """
        pass

    def _matrix_indices_to_rates(self, i: int, j: int) -> float:
        """
        Get the rate from the state indexed by i to the state indexed by j.

        :param i: Index of outgoing state.
        :param j: Index of incoming state.
        :return: The rate from the state indexed by i to the state indexed by j.
        """
        return self.get_transition(i=i, j=j).get_rate()

    def get_transition(self, i: int, j: int) -> Transition:
        """
        Get the transition from the state indexed by i to the state indexed by j.
        TODO remove debug code

        :param i: Index of outgoing state.
        :param j: Index of incoming state.
        :return: The transition from the state indexed by i to the state indexed by j.
        """
        transition = Transition(
            marginal1=self.states[i],
            marginal2=self.states[j],
            linked1=self.linked[i],
            linked2=self.linked[j],
            state_space=self
        )

        data = dict(
            marginal1=transition.marginal1,
            marginal2=transition.marginal2,
            linked1=transition.linked1,
            linked2=transition.linked2,
            # unlinked1=transition.unlinked1,
            # unlinked2=transition.unlinked2
        )

        kind = transition.type

        rate = transition.get_rate()

        if rate != 0:
            pass

        return transition

    def _display_state(self, i: int) -> str:
        """
        Display the state indexed by `i`.

        :param i: The state index.
        :return: Textual representation of the state.
        """
        return str(self.states[i]).replace('\n', '') + '\n' + str(self.linked[i]).replace('\n', '')

    def _get_color_state(self, i: int) -> str:
        """
        Get color of the state indexed by `i`.
        """
        if State.is_absorbing(self.states[i]):
            return '#f1807e'

        if self.alpha[i] > 0:
            return 'lightgreen'

        return 'lightblue'

    def _plot_rates(self):
        """
        Plot the rate matrix using graphviz.
        """
        import graphviz

        graph = graphviz.Digraph()

        # add nodes
        for i in range(len(self.states)):
            graph.node(
                name=self._display_state(i),
                fillcolor=self._get_color_state(i),
                style='filled'
            )

        # add non-zero edges
        for i, j in zip(*self._non_zero_states):

            t = self.get_transition(i=i, j=j)

            if not State.is_absorbing(t.marginal1):
                graph.edge(
                    self._display_state(i),
                    self._display_state(j),
                    label=f'{t.type}: {t.get_rate():.2g}',
                    color=t.get_color(),
                    fontcolor=t.get_color()
                )

        graph.render('rate matrix', view=True)

    @staticmethod
    def _find_vectors(n: int, k: int) -> List[List[int]]:
        """
        Find all vectors of length ``k`` with non-negative integers that sum to ``n``.

        :param n: The sum.
        :param k: The length of the vectors.
        :return: All vectors of length ``k`` with non-negative integers that sum to ``n``.
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

    def _get_default(self):
        """
        Get the default state space.
        """
        return DefaultStateSpace(
            pop_config=self.pop_config,
            locus_config=self.locus_config,
            model=self.model,
            epoch=self.epoch
        )


class DefaultStateSpace(StateSpace):
    """
    Default rate matrix where there is one state per number of lineages for each deme and locus.
    """

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
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
            n_pops = self.pop_config.n_pops
            linked_locus = np.array(list(itertools.product(range(self.pop_config.n + 1), repeat=n_pops)))
            linked_locus = linked_locus[linked_locus.sum(axis=1) <= self.pop_config.n]

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
        Get the states. Each state describes the lineage configuration per deme and locus, i.e.
        one state has the structure [[[a_ijk]]] where i is the lineage configuration, j is the deme and k is the locus.
        """
        # the number of lineages
        lineages = np.arange(1, self.pop_config.n + 1)[::-1]

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


class BlockCountingStateSpace(StateSpace):
    r"""
    Rate matrix for block counting state space where there is one state per sample configuration:
    :math:`{ (a_1,...,a_n) \in \mathbb{Z}^+ : \sum_{i=1}^{n} a_i = n \}`,

    per deme and per locus. This state space can distinguish between different tree topologies
    and is thus used when computing statistics based on the SFS.
    """

    def __init__(
            self,
            pop_config: LineageConfig,
            locus_config: LocusConfig = LocusConfig(),
            model: CoalescentModel = StandardCoalescent(),
            epoch: Epoch = Epoch()
    ):
        # currently only one locus is supported, due to a very complex state space for multiple loci
        if locus_config.n > 1:
            raise NotImplementedError('BlockCountingStateSpace only supports one locus.')

        super().__init__(pop_config=pop_config, locus_config=locus_config, model=model, epoch=epoch)

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
        """
        return self.model.get_rate_block_counting(n=n, s1=s1, s2=s2)

    def _expand_loci(self, states: np.ndarray) -> np.ndarray:
        """
        Expand the given states to include all possible combinations of locus configurations.
        TODO two-locus state space not sufficient for computing SFS as lineages are not longer
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

        raise NotImplementedError("Only 1 or 2 loci are currently supported.")

    @cached_property
    def states(self) -> np.ndarray:
        """
        Get the states. Each state describes the lineage configuration per deme and locus, i.e.
        one state has the structure [[[a_ijk]]] where i is the lineage configuration, j is the deme and k is the locus.

        :return: The states.
        """
        # the possible allele configurations
        lineage_configs = np.array(self._find_sample_configs(m=self.pop_config.n, n=self.pop_config.n))

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
