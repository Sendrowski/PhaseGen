import logging
from functools import cached_property

import numpy as np

logger = logging.getLogger('phasegen')


class State:
    """
    Class representing a state.
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
        return np.all(state.sum(axis=(1, 2)) == 1)


class Transition:
    """
    Class representing a transition between two states.
    """

    def __init__(
            self,
            state_space: 'StateSpace',
            marginal1: np.ndarray,
            marginal2: np.ndarray,
            shared1: np.ndarray,
            shared2: np.ndarray
    ):
        """
        Initialize a transition.

        :param state_space: State space.
        :param marginal1: Marginal lineages in outgoing state.
        :param marginal2: Marginal lineages in incoming state.
        :param shared1: Numbers of shared lineages in outgoing state.
        :param shared2: Numbers of shared lineages in incoming state.
        """
        #: State space.
        self.state_space: 'StateSpace' = state_space

        #: Marginal lineages in outgoing state.
        self.marginal1: np.ndarray = marginal1

        #: Marginal lineages in incoming state.
        self.marginal2: np.ndarray = marginal2

        #: Shared lineages in outgoing state.
        self.shared1: np.ndarray = shared1

        #: Shared lineages in incoming state.
        self.shared2: np.ndarray = shared2

    @cached_property
    def n_loci(self) -> int:
        """
        Number of loci.
        """
        return self.state_space.locus_config.n

    @cached_property
    def diff_marginal(self) -> np.ndarray:
        """
        Difference between marginal lineages.
        """
        return self.marginal1 - self.marginal2

    @cached_property
    def diff_shared(self) -> np.ndarray:
        """
        Difference in shared lineages.
        """
        return self.shared1 - self.shared2

    @cached_property
    def has_diff_demes_marginal(self) -> np.ndarray:
        """
        Mask for demes with affected lineages.
        """
        return np.any(self.diff_marginal != 0, axis=(0, 2))

    @cached_property
    def has_diff_marginal(self) -> bool:
        """
        Whether there are any marginal differences.
        """
        return bool(self.has_diff_demes_marginal.any())

    @cached_property
    def has_diff_demes_shared(self) -> np.ndarray:
        """
        Mask for demes with affected shared lineages.
        """
        return np.any(self.diff_shared != 0, axis=(0, 2))

    @cached_property
    def has_diff_shared(self) -> bool:
        """
        Whether there are any affected shared lineages.
        """
        return bool(self.has_diff_demes_shared.any())

    @cached_property
    def has_diff_loci(self) -> np.ndarray:
        """
        Mask for affected loci with respect to marginal lineages.
        """
        return np.any(self.diff_marginal != 0, axis=(1, 2))

    @cached_property
    def n_diff_loci(self) -> int:
        """
        Number of affected loci with respect to marginal lineages.
        """
        return self.has_diff_loci.sum()

    @cached_property
    def n_demes_marginal(self) -> int:
        """
        Number of affected demes with respect to marginal lineages.
        """
        return self.has_diff_demes_marginal.sum()

    @cached_property
    def is_absorbing1(self) -> bool:
        """
        Whether state 1 is absorbing.
        """
        return State.is_absorbing(self.marginal1)

    @cached_property
    def is_absorbing2(self) -> bool:
        """
        Whether state 2 is absorbing.
        """
        return State.is_absorbing(self.marginal2)

    @cached_property
    def is_absorbing(self) -> bool:
        """
        Whether one of the states is absorbing.
        TODO necessary/possible is context of recombination despite need to get max(mrca) for tree height distribution?
        """
        return self.is_absorbing1 or self.is_absorbing2

    @cached_property
    def is_eligible_recombination(self) -> bool:
        """
        Whether the transition is eligible for a recombination event.
        """
        # if there are any affected lineages, it can't be a recombination event
        if self.has_diff_marginal:
            return False

        # if there are not exactly `n_loci` different lineages, it can't be a recombination event
        if not np.all((self.diff_shared == 0).sum() == self.shared1.size - self.n_loci):
            return False

        # make sure change in shared lineages is in the same deme for each locus
        demes = np.where(self.diff_shared != 0)[1]
        if not np.all(demes == demes[0]):
            return False

        # no recombination from or to absorbing state
        if self.is_absorbing:
            return False

        return True

    @cached_property
    def is_forward_recombination(self) -> bool:
        """
        Whether transition is a forward recombination event.
        """
        # if not eligible for recombination, it can't be a recombination event
        if not self.is_eligible_recombination:
            return False

        # if there is not exactly one more lineage in state 1 than in state 2 for each locus,
        # it can't be a recombination event
        if not np.all((self.diff_shared == 1).sum(axis=(1, 2)) == 1):
            return False

        return True

    @cached_property
    def is_backward_recombination(self) -> bool:
        """
        Whether the transition is a backward recombination event.
        """
        # if not eligible for recombination, it can't be a recombination event
        if not self.is_eligible_recombination:
            return False

        # if there is not exactly one more lineage in state 2 than in state 1 for each locus,
        # it can't be a recombination event
        if not np.all((self.diff_shared == -1).sum(axis=(1, 2)) == 1):
            return False

        return True

    @cached_property
    def is_recombination(self) -> bool:
        """
        Whether the transition is a recombination event.
        """
        return self.is_forward_recombination or self.is_backward_recombination

    @cached_property
    def is_eligible_coalescence(self) -> bool:
        """
        Whether the transition is eligible for a coalescence event.
        """
        # if not exactly one deme is affected, it can't be a coalescence event
        return self.n_demes_marginal == 1

    @cached_property
    def deme_coal(self) -> int:
        """
        Index of deme where coalescence event occurs.
        """
        return int(np.where(self.has_diff_demes_marginal)[0][0])

    @cached_property
    def locus_coal_unshared(self) -> int:
        """
        Index of locus where coalescence event occurs.
        """
        return int(np.where(self.has_diff_loci)[0][0])

    @cached_property
    def deme_unshared_coal_marginal1(self) -> np.ndarray:
        """
        Marginal block config in deme and locus where coalescence event occurs in state 1.
        """
        return self.marginal1[self.locus_coal_unshared, self.deme_coal]

    @cached_property
    def deme_unshared_coal_marginal2(self) -> np.ndarray:
        """
        Marginal block config in deme and locus where coalescence event occurs in state 2.
        """
        return self.marginal2[self.locus_coal_unshared, self.deme_coal]

    @cached_property
    def deme_coal_marginal1(self) -> np.ndarray:
        """
        Marginal block config in deme where coalescence event occurs in state 1.
        """
        return self.marginal1[:, self.deme_coal]

    @cached_property
    def deme_coal_marginal2(self) -> np.ndarray:
        """
        Marginal block config in deme where coalescence event occurs in state 2.
        """
        return self.marginal2[:, self.deme_coal]

    @cached_property
    def diff_deme_coal_marginal(self) -> np.ndarray:
        """
        Difference in marginal number of lineages in deme where coalescence event occurs.
        """
        return self.deme_coal_marginal1 - self.deme_coal_marginal2

    @cached_property
    def has_diff_loci_deme_coal(self) -> np.ndarray:
        """
        Mask for affected loci with respect to deme where coalescence event occurs.
        """
        return np.any(self.diff_deme_coal_marginal != 0, axis=1)

    @cached_property
    def n_diff_loci_deme_coal(self) -> int:
        """
        Number of loci where coalescence event occurs in deme where coalescence event occurs.
        """
        return self.has_diff_loci_deme_coal.sum()

    @cached_property
    def is_eligible_shared_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for shared coalescence.
        """
        return self.n_diff_loci_deme_coal > 1

    @cached_property
    def is_eligible_unshared_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for marginal coalescence.
        """
        return self.n_diff_loci_deme_coal == 1

    @cached_property
    def is_eligible(self) -> bool:
        """
        Whether the transition is eligible for any event. This is supposed to rule out impossible
        transitions as quickly as possible.
        """
        # rates for staying in the same state are determined later
        if not self.has_diff_marginal and not self.has_diff_shared:
            return False

        if self.is_eligible_recombination and not self.is_recombination:
            return False

        if self.is_eligible_coalescence and not self.is_coalescence:
            return False

        if self.is_eligible_migration and not self.is_migration:
            return False

        return True

    @cached_property
    def is_valid_lineage_reduction_shared_coalescence(self) -> bool:
        """
        In case of a shared coalescence event, whether the reduction in the number of shared lineages is equal
        to the reduction in the number of coalesced lineages for each locus.
        """
        return np.all(self.diff_shared[:, self.deme_coal] == self.diff_marginal[:, self.deme_coal])

    @cached_property
    def has_sufficient_shared_lineages_shared_coalescence(self) -> bool:
        """
        In case of a shared coalescence event, whether the number of shared lineages is greater than
        equal to the number of shared coalesced lineages.
        """
        shared = self.shared1[:, self.deme_coal].sum(axis=1)
        coalesced = self.diff_deme_coal_marginal.sum(axis=1) + 1

        return np.all(shared >= coalesced)

    @cached_property
    def is_shared_coalescence(self) -> bool:
        """
        Whether the coalescence event is a shared coalescence event.
        """
        return (
                self.is_eligible_coalescence and
                self.is_eligible_shared_coalescence and
                self.is_valid_lineage_reduction_shared_coalescence and
                self.has_sufficient_shared_lineages_shared_coalescence
        )

    @cached_property
    def deme_coal_shared1(self) -> np.ndarray:
        """
        Shared lineages in deme and locus where coalescence event occurs in state 1.
        """
        return self.shared1[self.has_diff_loci_deme_coal][0][self.deme_coal]

    @cached_property
    def deme_coal_shared2(self) -> np.ndarray:
        """
        Shared lineages in deme and locus where coalescence event occurs in state 2.
        """
        return self.shared2[self.has_diff_loci_deme_coal][0][self.deme_coal]

    @cached_property
    def deme_coal_unshared1(self) -> np.ndarray:
        """
        Unshared lineages in deme and locus where coalescence event occurs in state 1.
        """
        marginal = self.deme_coal_marginal1[self.has_diff_loci_deme_coal][0]
        shared = self.shared1[self.has_diff_loci_deme_coal][0][self.deme_coal]

        return marginal - shared

    @cached_property
    def deme_coal_unshared2(self) -> np.ndarray:
        """
        Unshared lineages in deme where coalescence event occurs in state 2.
        """
        marginal = self.deme_coal_marginal2[self.has_diff_loci_deme_coal][0]
        shared = self.shared2[self.has_diff_loci_deme_coal][0][self.deme_coal]

        return marginal - shared

    @cached_property
    def is_binary_lineage_reduction_marginal_coalescence(self) -> bool:
        """
        Whether the marginal coalescence event is a binary merger.
        """
        reduction = self.deme_coal_unshared1 - self.deme_coal_unshared2

        return reduction.sum() == 1

    @cached_property
    def is_valid_lineage_reduction_marginal_coalescence(self) -> bool:
        """
        In case of a marginal coalescence event, whether the reduction in the number of unshared lineages is equal
        to the reduction in the number of coalesced lineages.
        """
        reduction = self.deme_coal_unshared1 - self.deme_coal_unshared2
        diff = self.diff_deme_coal_marginal[self.locus_coal_unshared]

        return (reduction == diff).all()

    @cached_property
    def is_unshared_coalescence(self) -> bool:
        """
        Whether the coalescence event is a marginal coalescence event.
        """
        return (
                self.is_eligible_coalescence and
                self.is_eligible_unshared_coalescence and
                self.is_binary_lineage_reduction_marginal_coalescence and
                self.is_valid_lineage_reduction_marginal_coalescence
        )

    @cached_property
    def is_coalescence(self) -> bool:
        """
        Whether the transition is a coalescence event.
        """
        return self.is_shared_coalescence or self.is_unshared_coalescence

    @cached_property
    def is_eligible_migration(self) -> bool:
        """
        Whether the transition is eligible for a migration event.
        TODO do we also need to move shared lineages?
        """
        # two demes must be affected
        return self.n_demes_marginal == 2

    @cached_property
    def is_valid_migration_one_locus_only(self) -> bool:
        """
        Whether the migration event is only affecting one locus.
        """
        return self.n_diff_loci == 1

    @cached_property
    def lineages_locus1(self) -> np.ndarray:
        """
        Lineages in locus where migration event occurs in state 1.
        """
        return self.marginal1[self.has_diff_loci][0]

    @cached_property
    def lineages_locus2(self) -> np.ndarray:
        """
        Lineages in locus where migration event occurs in state 2.
        """
        return self.marginal2[self.has_diff_loci][0]

    @cached_property
    def diff_lineages_locus(self) -> np.ndarray:
        """
        Difference in lineages in locus where migration event occurs.
        """
        return self.lineages_locus1 - self.lineages_locus2

    @cached_property
    def is_one_migration_event(self) -> bool:
        """
        Whether there is exactly one migration event.
        """
        # make sure exactly one lineage is moved from one deme to another
        return (
                (self.diff_lineages_locus == 1).sum() == 1 and
                (self.diff_lineages_locus == -1).sum() == 1 and
                (self.diff_lineages_locus == 0).sum() == self.diff_lineages_locus.size - 2
        )

    @cached_property
    def has_enough_lineages_migration(self) -> bool:
        """
        Whether there are enough lineages to perform a migration event.
        """
        return self.lineages_locus1.sum() > 1

    @cached_property
    def is_migration(self) -> bool:
        """
        Whether the transition is a migration event.
        """
        return (
                self.is_eligible_migration and
                self.is_valid_migration_one_locus_only and
                self.is_one_migration_event and
                self.has_enough_lineages_migration
        )

    def get_rate_forward_recombination(self) -> float:
        """
        Get the rate of a forward recombination event.
        TODO here we assume there number of shared lineages is the same across loci which need not be the case
        """
        return self.shared1[self.diff_shared == 1][0] * self.state_space.locus_config.recombination_rate

    def get_rate_backward_recombination(self) -> float:
        """
        Get the rate of a backward recombination event.
        """
        marginal = self.marginal1[self.diff_shared == -1]
        shared = self.shared1[self.diff_shared == -1]

        return (marginal - shared).prod()

    def get_pop_size_deme_coalescence(self) -> float:
        """
        Get the population size of the deme where the coalescence event occurs.
        """
        return self.state_space.epoch.pop_sizes[self.state_space.epoch.pop_names[self.deme_coal]]

    def get_scaled_pop_size_deme_coalescence(self) -> float:
        """
        Get the scaled population size of the deme where the coalescence event occurs.
        """
        return self.state_space.model._get_timescale(self.get_pop_size_deme_coalescence())

    def get_rate_shared_coalescence(self) -> float:
        """
        Get the rate of a shared coalescence event.
        """
        # TODO we assume shared lineages are the same across loci which need not be the case
        rate = self.state_space._get_coalescent_rate(
            n=self.state_space.pop_config.n,
            s1=self.shared1[0, self.deme_coal],
            s2=self.shared2[0, self.deme_coal]
        )

        return rate / self.get_scaled_pop_size_deme_coalescence()

    def get_rate_unshared_coalescence(self) -> float:
        """
        Get the rate of a marginal coalescence event.
        """
        # get the total coalescent rate when all lineages are considered
        rate_all = self.state_space._get_coalescent_rate(
            n=self.state_space.pop_config.n,
            s1=self.deme_unshared_coal_marginal1,
            s2=self.deme_unshared_coal_marginal2
        )

        # difference in marginal number of lineages between state 1 and state 2
        diff = self.deme_unshared_coal_marginal2 - self.deme_unshared_coal_marginal1

        # get coalescence rate of shared lineages
        rate_shared = self.state_space._get_coalescent_rate(
            n=self.state_space.pop_config.n,
            s1=self.shared1[0, self.deme_coal],
            s2=self.shared1[0, self.deme_coal] + diff
        )

        # get unshared coalescence rate by subtracting shared coalescence rate from total coalescence rate
        return (rate_all - rate_shared) / self.get_scaled_pop_size_deme_coalescence()

    def get_rate_migration(self) -> float:
        """
        Get the rate of a migration event.
        """
        # get the indices of the source and destination demes
        i_source = np.where((self.diff_lineages_locus == 1).sum(axis=1) == 1)[0][0]
        i_dest = np.where((self.diff_lineages_locus == -1).sum(axis=1) == 1)[0][0]

        # get the deme names
        source = self.state_space.epoch.pop_names[i_source]
        dest = self.state_space.epoch.pop_names[i_dest]

        # get the number of lineages in deme i before migration
        n_lineages_source = self.lineages_locus1[i_source][np.where(self.diff_lineages_locus == 1)[1][0]]

        # get migration rate from source to destination
        migration_rate = self.state_space.epoch.migration_rates[(source, dest)]

        # scale migration rate by number of lineages in source deme
        rate = migration_rate * n_lineages_source

        return rate

    def get_rate(self) -> float:
        """
        Get the rate of the transition.

        :return: The rate of the transition.
        """
        if not self.is_eligible:
            return 0

        if self.is_forward_recombination:
            return self.get_rate_forward_recombination()

        if self.is_backward_recombination:
            return self.get_rate_backward_recombination()

        if self.is_shared_coalescence:
            return self.get_rate_shared_coalescence()

        if self.is_unshared_coalescence:
            return self.get_rate_unshared_coalescence()

        if self.is_migration:
            return self.get_rate_migration()

        return 0

    @cached_property
    def type(self) -> str:
        """
        Get the type of the transition.
        """
        if self.is_forward_recombination:
            return 'forward_recombination'

        if self.is_backward_recombination:
            return 'backward_recombination'

        if self.is_shared_coalescence:
            return 'shared_coalescence'

        if self.is_unshared_coalescence:
            return 'unshared_coalescence'

        if self.is_migration:
            return 'migration'

        return 'invalid'
