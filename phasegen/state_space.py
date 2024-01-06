import itertools
import logging
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from typing import List, Tuple

import numpy as np

from .coalescent_models import CoalescentModel, StandardCoalescent
from .demography import Epoch
from .locus import LocusConfig
from .population import PopConfig

logger = logging.getLogger('phasegen')


class StateSpace(ABC):
    """
    State space.
    """

    def __init__(
            self,
            pop_config: PopConfig,
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
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: Coalescent model
        self.model: CoalescentModel = model

        #: Population configuration
        self.pop_config: PopConfig = pop_config

        #: Locus configuration
        self.locus_config: LocusConfig = locus_config

        #: Epoch
        self.epoch: Epoch = epoch

        # number of lineages shared across loci
        self.n_shared: np.ndarray | None = None

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

        # get the rate matrix for the default demography
        default_rate_matrix = np.fromfunction(
            np.vectorize(self._matrix_indices_to_rates, otypes=[float]),
            (self.k, self.k),
            dtype=int
        )

        # restore the epoch
        self.epoch = epoch

        # indices of non-zero rates
        # this improves performance when recomputing the rate matrix for different epochs
        return np.where(default_rate_matrix != 0)

    def _expand_loci(self, states: np.ndarray) -> np.ndarray:
        """
        Expand the given states to include all possible combinations of locus configurations.
        TODO clean up

        :param states: States.
        """
        if self.locus_config.n == 1:
            # all lineages are shared
            self.n_shared = np.zeros(states.shape[0], dtype=int)

            # add extra dimension for locus configuration
            return states[:, np.newaxis]

        if self.locus_config.n == 2:
            # create array with same shape and fill first element with number of shared lineages
            n_shared = np.zeros((self.pop_config.n + 1,) + states.shape[-2:], dtype=int)
            n_shared[:, 0, 0] = np.arange(self.pop_config.n + 1)[::-1]

            # take product of number of shared lineages and states
            states = np.array(list(itertools.product(n_shared, states, states)))

            n_lineages = states.sum(axis=(2, 3)).T

            # remove states where n_shared is larger than the total number of lineages
            states = states[(n_lineages[0] <= n_lineages[1]) & (n_lineages[0] <= n_lineages[2])]

            self.n_shared = states[:, 0, 0, 0]

            return states[:, 1:, :, :]

        raise NotImplementedError("Only 1 or 2 loci are currently supported.")

    @cached_property
    @abstractmethod
    def states(self) -> np.ndarray:
        """
        Get the states. Each state describes the lineage configuration per deme and locus, i.e.
        one state has the structure [[[a_ijk]]] where i is the lineage configuration, j is the deme and k is the locus.
        """
        pass

    @cached_property
    def s(self) -> np.ndarray:
        """
        Get exit rate vector.
        """
        return -self.S[:-1, :-1] @ self.e[:-1]

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

        return pops * loci

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

    def _matrix_indices_to_rates(self, i: int, j: int) -> float:
        """
        Get the rate from the state indexed by i to the state indexed by j.

        :param i: Index of outgoing state.
        :param j: Index of incoming state.
        :return: The rate from the state indexed by i to the state indexed by j.
        """
        if i == j:
            return 0

        # get the states
        s1 = self.states[i]
        s2 = self.states[j]

        # get the difference between the states
        diff = s1 - s2

        # mask for affected demes
        has_diff_demes = np.any(diff != 0, axis=(0, 2))

        # number of affected demes
        n_demes = has_diff_demes.sum()

        n_shared1 = self.n_shared[i]
        n_shared2 = self.n_shared[j]

        # possible recombination event
        if n_demes == 0:

            # no recombination or back recombination from or to absorbing state
            if np.all(s1.sum(axis=(1, 2)) == 1) or np.all(s2.sum(axis=(1, 2)) == 1):
                return 0

            # recombination onto different loci
            if n_shared1 - n_shared2 == 1:
                rate = self.n_shared[i] * self.locus_config.recombination_rate
                return rate

            # back recombination onto same locus
            if n_shared1 - n_shared2 == -1:
                rate = (s1[0].sum() - n_shared1) * (s1[1].sum() - n_shared1)
                return rate

            return 0

        # possible coalescent event
        if n_demes == 1:

            # get the index of the affected deme
            deme = np.where(has_diff_demes)[0][0]

            deme_s1 = s1[:, deme]
            deme_s2 = s2[:, deme]

            diff_deme = deme_s1 - deme_s2

            # number of loci that are affected within deme
            has_diff_loci = np.any(diff_deme != 0, axis=1)

            n_diff_loci = has_diff_loci.sum()

            is_shared = n_diff_loci > 1

            if is_shared:
                # if coalescent event is shared across loci but the reduction in
                # the number of shared lineages is not equal to the reduction in
                # the number of coalesced lineages for each locus, then the rate
                # is zero.
                if np.any(n_shared1 - n_shared2 != diff_deme.sum(axis=1)):
                    return 0

                # Alternatively, if the number of shared lineages is
                # less than the number of shared coalesced lineages, then the
                # rate is also zero.
                if diff_deme[0].sum() + 1 > n_shared1:
                    return 0

                base_rate = self._get_coalescent_rate(
                    n=self.pop_config.n,
                    s1=np.array([n_shared1]),
                    s2=np.array([n_shared2])
                )

            # not shared
            else:

                n_unshared1 = deme_s1[has_diff_loci].sum() - n_shared1
                n_unshared2 = deme_s2[has_diff_loci].sum() - n_shared2

                # number of reduced unshared lineages has to equal numbers of coalesced
                # lineages if coalescent event is not shared across loci
                if n_unshared1 - n_unshared2 != diff_deme[has_diff_loci].sum():
                    return 0

                if n_unshared1 - n_unshared2 == 1:
                    base_rate = self._get_coalescent_rate(
                        n=self.pop_config.n,
                        s1=np.array([n_unshared1]),
                        s2=np.array([n_unshared2])
                    ) + n_shared1 * n_unshared1
                else:
                    return 0

                x = 1
                pass

            pop_size = self.epoch.pop_sizes[self.epoch.pop_names[deme]]

            return base_rate / self.model._get_timescale(pop_size)

        # eligible for migration event
        # TODO if loci are linked, we allow lineage movement in several locus contexts simultaneously
        if n_demes == 2:

            # number of locus contexts that are affected
            has_diff_loci = np.any(diff != 0, axis=(1, 2))

            # migration only possible if one locus context is affected
            if has_diff_loci.sum() == 1:
                s1_locus = s1[has_diff_loci][0]
                s2_locus = s2[has_diff_loci][0]

                diff_locus = s1_locus - s2_locus

                # check if one migration event
                if (diff_locus == 1).sum(axis=1).sum() == 1 and (diff_locus == -1).sum(axis=1).sum() == 1:

                    # make sure that the other demes are not changing
                    if (diff_locus != 0).sum(axis=1).sum() == 2:

                        # make sure that the number of lineages is greater than 1
                        if s1_locus.sum() > 1:
                            # get the indices of the source and destination demes
                            i_source = np.where((diff_locus == 1).sum(axis=1) == 1)[0][0]
                            i_dest = np.where((diff_locus == -1).sum(axis=1) == 1)[0][0]

                            # get the deme names
                            source = self.epoch.pop_names[i_source]
                            dest = self.epoch.pop_names[i_dest]

                            # get the number of lineages in deme i before migration
                            n_lineages_source = s1_locus[i_source][np.where(diff_locus == 1)[1][0]]

                            # get migration rate from source to destination
                            migration_rate = self.epoch.migration_rates[(source, dest)]

                            # scale migration rate by number of lineages in source deme
                            rate = migration_rate * n_lineages_source

                            return rate

        return 0

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

    def _get_coalescent_rate(self, n: int, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Get the coalescent rate from state ``s1`` to state ``s2``.

        :param n: Number of lineages.
        :param s1: State 1.
        :param s2: State 2.
        :return: The coalescent rate from state ``s1`` to state ``s2``.
        """
        return self.model.get_rate_block_counting(n=n, s1=s1, s2=s2)

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
