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
        # if there are an affected lineages, it can't be a recombination event
        if self.has_diff_marginal:
            return False

        # if there are no affected shared lineages, it can't be a recombination event
        if not self.has_diff_shared:
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
        return self.is_eligible_recombination and self.shared1 - self.shared2 == 1

    @cached_property
    def is_backward_recombination(self) -> bool:
        """
        Whether the transition is a backward recombination event.
        """
        return self.is_eligible_recombination and self.shared1 - self.shared2 == -1

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
    def deme_coal(self) -> int | None:
        """
        Index of deme where coalescence event occurs.
        """
        if not self.is_eligible_coalescence:
            return None

        return int(np.where(self.has_diff_demes_marginal)[0][0])

    @cached_property
    def lineages_deme_coal1(self) -> np.ndarray | None:
        """
        Lineages in deme where coalescence event occurs in state 1.
        """
        if not self.is_eligible_coalescence:
            return None

        return self.marginal1[:, self.deme_coal]

    @cached_property
    def lineages_deme_coal2(self) -> np.ndarray | None:
        """
        Lineages in deme where coalescence event occurs in state 2.
        """
        if not self.is_eligible_coalescence:
            return None

        return self.marginal2[:, self.deme_coal]

    @cached_property
    def diff_lineages_deme_coal(self) -> np.ndarray | None:
        """
        Difference in lineages in deme where coalescence event occurs.
        """
        if not self.is_eligible_coalescence:
            return None

        return self.lineages_deme_coal1 - self.lineages_deme_coal2

    @cached_property
    def n_diff_loci_deme_coal(self) -> int | None:
        """
        Number of loci where coalescence event occurs in deme where coalescence event occurs.
        """
        if not self.is_eligible_coalescence:
            return None

        return self.np.any(self.diff_lineages_deme_coal != 0, axis=1).sum()

    @cached_property
    def is_eligible_shared_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for shared coalescence.
        """
        if not self.is_eligible_coalescence:
            return False

        return self.n_diff_loci_deme_coal > 1

    @cached_property
    def is_eligible_marginal_coalescence(self) -> bool:
        """
        Whether the coalescence event is eligible for marginal coalescence.
        """
        if not self.is_eligible_coalescence:
            return False

        return self.n_diff_loci_deme_coal == 1

    @cached_property
    def is_eligible(self) -> bool:
        """
        Whether the transition is eligible for any event. This is supposed to rule out impossible
        transitions as quickly as possible.
        """
        # rate for staying in the same state are determined later
        if self.i == self.j:
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
        return not np.any(self.shared1 - self.shared2 != self.diff_lineages_deme_coal.sum(axis=1))

    @cached_property
    def has_sufficient_shared_lineages_shared_coalescence(self) -> bool:
        """
        In case of a shared coalescence event, whether the number of shared lineages is greater than
        equal to the number of shared coalesced lineages.
        """
        return self.diff_lineages_deme_coal[0].sum() + 1 > self.shared1

    @cached_property
    def is_shared_coalescence(self) -> bool:
        """
        Whether the coalescence event is a shared coalescence event.
        """
        return (
                self.is_eligible_shared_coalescence and
                self.is_valid_lineage_reduction_shared_coalescence and
                self.has_sufficient_shared_lineages_shared_coalescence
        )

    @cached_property
    def unshared_deme_marginal_coalescence1(self) -> int | None:
        """
        Unshared lineages in deme where coalescence event occurs in state 1.
        """
        if not self.is_eligible_coalescence:
            return None

        return (self.diff_lineages_deme_coal[self.has_diff_loci_deme_coal] -
                self.shared1[self.has_diff_loci_deme_coal, self.deme_coal])

    @cached_property
    def unshared_deme_marginal_coalescence2(self) -> int | None:
        """
        Unshared lineages in deme where coalescence event occurs in state 2.
        """
        if not self.is_eligible_coalescence:
            return None

        return (self.diff_lineages_deme_coal[self.has_diff_loci_deme_coal] -
                self.shared2[self.has_diff_loci_deme_coal, self.deme_coal])

    @cached_property
    def is_binary_lineage_reduction_marginal_coalescence(self) -> bool:
        """
        Whether the marginal coalescence event is a binary merger.
        """
        return self.unshared_deme_marginal_coalescence1 - self.unshared_deme_marginal_coalescence2 == 1

    @cached_property
    def is_valid_lineage_reduction_marginal_coalescence(self) -> bool:
        """
        In case of a marginal coalescence event, whether the reduction in the number of unshared lineages is equal
        to the reduction in the number of coalesced lineages.
        """
        return (self.unshared_deme_marginal_coalescence1 - self.unshared_deme_marginal_coalescence2 ==
                self.n_diff_loci_deme_coal)

    @cached_property
    def is_marginal_coalescence(self) -> bool:
        """
        Whether the coalescence event is a marginal coalescence event.
        """
        return (
                self.is_eligible_marginal_coalescence and
                self.is_binary_lineage_reduction_marginal_coalescence and
                self.is_valid_lineage_reduction_marginal_coalescence
        )

    @cached_property
    def is_coalescence(self) -> bool:
        """
        Whether the transition is a coalescence event.
        """
        return self.is_shared_coalescence or self.is_marginal_coalescence

    @cached_property
    def is_eligible_migration(self) -> bool:
        """
        Whether the transition is eligible for a migration event.
        TODO if loci are linked, we allow lineage movement in several locus contexts simultaneously?
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
    def lineages_locus1(self) -> np.ndarray | None:
        """
        Lineages in locus where migration event occurs in state 1.
        """
        if not self.is_eligible_migration:
            return None

        return self.marginal1[self.has_diff_loci][0]

    @cached_property
    def lineages_locus2(self) -> np.ndarray | None:
        """
        Lineages in locus where migration event occurs in state 2.
        """
        if not self.is_eligible_migration:
            return None

        return self.marginal2[self.has_diff_loci][0]

    @cached_property
    def diff_lineages_locus(self) -> np.ndarray | None:
        """
        Difference in lineages in locus where migration event occurs.
        """
        if not self.is_eligible_migration:
            return None

        return self.lineages_locus1 - self.lineages_locus2

    @cached_property
    def is_one_migration_event(self) -> bool:
        """
        Whether there is only one migration event.
        """
        return (
                (self.diff_lineages_locus == 1).sum(axis=1).sum() == 1 and
                (self.diff_lineages_locus == -1).sum(axis=1).sum() == 1
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
        """
        return self.shared1 * self.state_space.locus_config.recombination_rate

    def get_rate_backward_recombination(self) -> float:
        """
        Get the rate of a backward recombination event.
        """
        return (self.marginal1[0].sum() - self.shared1) * (self.marginal1[1].sum() - self.shared1)

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
        rate = self.state_space._get_coalescent_rate(
            n=self.state_space.pop_config.n,
            s1=np.array([self.shared1]),
            s2=np.array([self.shared2])
        )

        return rate / self.get_scaled_pop_size_deme_coalescence()

    def get_rate_marginal_coalescence(self) -> float:
        """
        Get the rate of a marginal coalescence event.
        """
        rate = self.state_space._get_coalescent_rate(
            n=self.state_space.pop_config.n,
            s1=np.array([self.unshared_deme_marginal_coalescence1]),
            s2=np.array([self.unshared_deme_marginal_coalescence2])
        ) + self.shared1 * self.unshared_deme_marginal_coalescence1

        return rate / self.get_scaled_pop_size_deme_coalescence()

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

        if self.is_marginal_coalescence:
            return self.get_rate_marginal_coalescence()

        if self.is_migration:
            return self.get_rate_migration()

        return 0
