import logging
from functools import cached_property
from typing import cast

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
        return np.all(np.sum(state * np.arange(1, state.shape[2] + 1)[::-1], axis=(1, 2)) == 1)


class Transition:
    """
    Class representing a transition between two states.
    """

    def __init__(
            self,
            state_space: 'StateSpace',
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
        self.state_space: 'StateSpace' = state_space

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
    def diff_deme_coal_marginal(self) -> np.ndarray:
        """
        Difference in marginal number of lineages in deme where coalescence event occurs.
        """
        return self.deme_coal_marginal1 - self.deme_coal_marginal2

    @cached_property
    def diff_lineages_locus(self) -> np.ndarray:
        """
        Difference in lineages in locus where migration event occurs.
        """
        return self.lineages_locus1 - self.lineages_locus2

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
    def n_diff_loci(self) -> int:
        """
        Number of affected loci with respect to marginal lineages.
        """
        return self.is_diff_loci.sum()

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
        return self.is_diff_loci_deme_coal.sum()

    @cached_property
    def is_diff_demes_marginal(self) -> np.ndarray:
        """
        Mask for demes with affected lineages.
        """
        return np.any(self.diff_marginal != 0, axis=(0, 2))

    @cached_property
    def has_diff_marginal(self) -> bool:
        """
        Whether there are any marginal differences.
        """
        return bool(self.is_diff_demes_marginal.any())

    @cached_property
    def is_diff_demes_linked(self) -> np.ndarray:
        """
        Mask for demes with affected linked lineages.
        """
        return np.any(self.diff_linked != 0, axis=(0, 2))

    @cached_property
    def is_diff_demes_unlinked(self) -> np.ndarray:
        """
        Mask for demes with affected unlinked lineages.
        """
        return np.any(self.diff_unlinked != 0, axis=(0, 2))

    @cached_property
    def has_diff_linked(self) -> bool:
        """
        Whether there are any affected linked lineages.
        """
        return bool(self.is_diff_demes_linked.any())

    @cached_property
    def has_diff_unlinked(self) -> bool:
        """
        Whether there are any affected unlinked lineages.
        """
        return bool(self.is_diff_demes_unlinked.any())

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
        return np.any(self.diff_deme_coal_marginal != 0, axis=1)

    @cached_property
    def deme_coal(self) -> int:
        """
        Index of deme where coalescence event occurs.
        """
        return int(np.where(self.is_diff_demes_marginal)[0][0])

    @cached_property
    def deme_locus_coal(self) -> int:
        """
        Index of deme where coalescence event occurs.
        """
        return int(np.where(self.is_diff_demes_linked)[0][0])

    @cached_property
    def locus_coal_unlinked(self) -> int:
        """
        Index of locus where coalescence event occurs.
        """
        return int(np.where(self.is_diff_loci)[0][0])

    @cached_property
    def deme_unlinked_coal_marginal1(self) -> np.ndarray:
        """
        Marginal block config in deme and locus where coalescence event occurs in state 1.
        """
        return self.marginal1[self.locus_coal_unlinked, self.deme_coal]

    @cached_property
    def deme_unlinked_coal_marginal2(self) -> np.ndarray:
        """
        Marginal block config in deme and locus where coalescence event occurs in state 2.
        """
        return self.marginal2[self.locus_coal_unlinked, self.deme_coal]

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
    def deme_coal_linked1(self) -> np.ndarray:
        """
        linked lineages in deme and locus where coalescence event occurs in state 1.
        """
        return self.linked1[self.is_diff_loci_deme_coal][0][self.deme_coal]

    @cached_property
    def deme_coal_linked2(self) -> np.ndarray:
        """
        linked lineages in deme and locus where coalescence event occurs in state 2.
        """
        return self.linked2[self.is_diff_loci_deme_coal][0][self.deme_coal]

    @cached_property
    def deme_coal_unlinked1(self) -> np.ndarray:
        """
        Unlinked lineages in deme and locus where coalescence event occurs in state 1.
        """
        marginal = self.deme_coal_marginal1[self.is_diff_loci_deme_coal][0]
        linked = self.linked1[self.is_diff_loci_deme_coal][0][self.deme_coal]

        return marginal - linked

    @cached_property
    def deme_coal_unlinked2(self) -> np.ndarray:
        """
        Unlinked lineages in deme where coalescence event occurs in state 2.
        """
        marginal = self.deme_coal_marginal2[self.is_diff_loci_deme_coal][0]
        linked = self.linked2[self.is_diff_loci_deme_coal][0][self.deme_coal]

        return marginal - linked

    @cached_property
    def deme_migration_source(self) -> int:
        """
        Get the source deme of the migration event.
        """
        return int(np.where((self.diff_lineages_locus == 1).sum(axis=1) == 1)[0][0])

    @cached_property
    def deme_migration_dest(self) -> int:
        """
        Get the destination deme of the migration event.
        """
        return int(np.where((self.diff_lineages_locus == -1).sum(axis=1) == 1)[0][0])

    @cached_property
    def block_migration(self) -> int:
        """
        Get the block where the migration event occurs.
        """
        return int(np.where(self.diff_lineages_locus == 1)[1][0])

    @cached_property
    def lineages_locus1(self) -> np.ndarray:
        """
        Lineages in locus where migration event occurs in state 1.
        """
        return self.marginal1[self.is_diff_loci][0]

    @cached_property
    def lineages_locus2(self) -> np.ndarray:
        """
        Lineages in locus where migration event occurs in state 2.
        """
        return self.marginal2[self.is_diff_loci][0]

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

        if self.has_diff_linked:
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

        # if not self.has_diff_linked:
        #    return False

        if self.n_blocks == 1:
            if self.has_diff_linked:
                return False

            if self.marginal1[self.locus_coal_unlinked, self.deme_coal].sum() < 2:
                return False

            return True

        if np.sum(np.any(self.diff_linked != 0, axis=(1, 2))) != 1:
            return False

        return True

    @cached_property
    def is_eligible(self) -> bool:
        """
        Whether the transition is eligible for any event. This is supposed to rule out impossible
        transitions as quickly as possible.
        """
        return True

    @cached_property
    def is_valid_lineage_reduction_linked_coalescence(self) -> bool:
        """
        In case of a linked coalescence event, whether the reduction in the number of linked lineages is equal
        to the reduction of marginal lineages.
        """
        return np.all(self.diff_linked[:, self.deme_coal] == self.diff_marginal[:, self.deme_coal])

    @cached_property
    def has_sufficient_linked_lineages_linked_coalescence(self) -> bool:
        """
        In case of a linked coalescence event, whether the number of linked lineages is greater than
        equal to the number of linked coalesced lineages.
        """
        linked = self.linked1[:, self.deme_coal].sum(axis=1)
        coalesced = self.diff_deme_coal_marginal.sum(axis=1) + 1

        return np.all(linked >= coalesced)

    @cached_property
    def is_lineage_reduction(self) -> bool:
        """
        Whether we have a lineage reduction.
        """
        return self.diff_marginal.sum() > 0

    @cached_property
    def is_linked_coalescence(self) -> bool:
        """
        Whether the coalescence event is a linked coalescence event, i.e. only linked lineages coalesce.
        """
        return (
                self.is_eligible_coalescence and
                self.is_eligible_linked_coalescence and
                self.is_lineage_reduction and
                self.is_valid_lineage_reduction_linked_coalescence and
                self.has_sufficient_linked_lineages_linked_coalescence
        )

    @cached_property
    def is_binary_lineage_reduction_unlinked_coalescence(self) -> bool:
        """
        Whether the unlinked coalescence event is a binary merger.
        """
        reduction = self.deme_coal_unlinked1 - self.deme_coal_unlinked2

        return reduction.sum() == 1

    @cached_property
    def is_binary_lineage_reduction_mixed_coalescence(self) -> bool:
        """
        Whether the mixed coalescence event is a binary merger.
        """
        reduction = self.deme_coal_unlinked1 - self.deme_coal_unlinked2

        return reduction.sum() == 1

    @cached_property
    def is_valid_lineage_reduction_unlinked_coalescence(self) -> bool:
        """
        In an unlinked coalescence event, whether the reduction in the number of unlinked lineages is equal
        to the reduction in the number of coalesced lineages.
        """
        reduction = self.deme_coal_unlinked1 - self.deme_coal_unlinked2
        diff = self.diff_deme_coal_marginal[self.locus_coal_unlinked]

        return reduction.sum() == diff.sum()

    @cached_property
    def is_valid_lineage_reduction_mixed_coalescence(self) -> bool:
        """
        In a mixed coalescence event there has to be reduction in the number of linked lineages.
        """
        # in the default state space, where we only keep track of the number of linked lineages per deme,
        # we may have a mixed coalescence event where the number of linked lineages does not change
        if self.n_blocks == 1 and not self.has_diff_linked:
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
    def is_unlinked_coalescence(self) -> bool:
        """
        Whether the coalescence event is an unlinked coalescence event, i.e. only unlinked lineages coalesce.
        """
        return (
                self.is_eligible_coalescence and
                self.is_eligible_unlinked_coalescence and
                self.is_lineage_reduction and
                # self.is_binary_lineage_reduction_unlinked_coalescence and
                self.is_valid_lineage_reduction_unlinked_coalescence
        )

    @cached_property
    def is_mixed_coalescence(self) -> bool:
        """
        Whether the coalescence event is a mixed coalescence event, i.e. both linked and unlinked lineages coalesce.
        """
        return (
                self.n_loci > 1 and
                self.is_eligible_coalescence and
                self.is_eligible_mixed_coalescence and
                self.is_lineage_reduction and
                self.is_binary_lineage_reduction_mixed_coalescence and
                self.is_valid_lineage_reduction_mixed_coalescence
        )

    @cached_property
    def is_coalescence(self) -> bool:
        """
        Whether the transition is a coalescence event.
        """
        return (
                self.is_linked_coalescence or
                self.is_unlinked_coalescence or
                self.is_mixed_coalescence or
                self.is_locus_coalescence
        )

    @cached_property
    def is_eligible_migration(self) -> bool:
        """
        Whether the transition is eligible for a migration event.
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
        # make sure exactly one lineage is moved from one deme to another
        return (
                (self.diff_lineages_locus == 1).sum() == 1 and
                (self.diff_lineages_locus == -1).sum() == 1 and
                (self.diff_lineages_locus == 0).sum() == self.diff_lineages_locus.size - 2
        )

    @cached_property
    def has_sufficient_linked_lineages_migration(self) -> bool:
        """
        Whether there are sufficient linked lineages to allow for a migration event.
        """
        return cast(bool, self.linked1[0, self.deme_migration_source, self.block_migration] > 0)

    @cached_property
    def has_sufficient_unlinked_lineages_migration(self) -> bool:
        """
        Whether there are sufficient unlinked lineages to allow for a migration event.
        """
        return cast(bool, self.unlinked1[0, self.deme_migration_source, self.block_migration] > 0)

    @cached_property
    def is_unlinked_migration(self) -> bool:
        """
        Whether the transition is a migration event.
        """
        return (
                not self.has_diff_linked and
                self.is_eligible_migration and
                self.is_valid_migration_one_locus_only and
                self.is_one_migration_event and
                self.has_sufficient_unlinked_lineages_migration
        )

    @cached_property
    def is_linked_migration(self) -> bool:
        """
        Whether the migration event is a linked migration event, i.e. a linked lineage migrates.
        """
        return (
                self.is_eligible_migration and
                self.is_one_migration_event and
                self.has_sufficient_linked_lineages_migration and
                self.has_diff_linked and
                self.is_valid_linked_migration
        )

    @cached_property
    def type(self) -> str:
        """
        Get the type of the transition.
        """
        types = []

        opts = [
            'recombination',
            'locus_coalescence',
            'linked_coalescence',
            'unlinked_coalescence',
            'mixed_coalescence',
            'linked_migration',
            'unlinked_migration'
        ]

        for t in opts:
            if getattr(self, f'is_{t}'):
                types.append(t)

        return '+'.join(types) or 'invalid'

    def get_rate_recombination(self) -> float:
        """
        Get the rate of a recombination event.
        Here we assume the number of linked lineages is the same across loci which should be the case.
        TODO number of linked lineages need not be the same across loci for lineage blocks
        """
        # linked1 = self.linked1[self.diff_linked == 1]
        linked1 = self.linked1.sum(axis=(1, 2))

        rate = linked1[0] * self.state_space.locus_config.recombination_rate

        return rate

    def get_rate_locus_coalescence(self) -> float:
        """
        Get the rate of a locus coalescence event.
        """
        # return 0 if locus coalescence is not allowed
        if not self.state_space.locus_config.allow_coalescence:
            return 0

        # get unlined lineage counts
        unlinked1 = self.unlinked1[self.diff_linked == -1]

        # get population size of deme where coalescence event occurs
        pop_size = self.state_space.epoch.pop_sizes[self.state_space.epoch.pop_names[self.deme_locus_coal]]

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
        coalescent pattern between loci are not allowed. However, once we have experienced a recombination event,
        a mixed or unlined coalescence event, and a subsequent locus coalescence event, we can have different
        coalescent patterns between loci. We would thus be required to keep track of the associations between
        lineages which would further expand the state space.
        """
        return self.state_space._get_coalescent_rate(
            n=self.state_space.pop_config.n,
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
        #        n=self.state_space.pop_config.n,
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
            n=self.state_space.pop_config.n,
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

        blocks = self.diff_marginal[self.locus_coal_unlinked, self.deme_coal] > 0

        if blocks.sum() == 1:
            rates_cross = unlinked1[blocks] * linked1[blocks]
        elif blocks.sum() == 2:
            rates_cross = [unlinked1[blocks][0] * linked1[blocks][1], unlinked1[blocks][1] * linked1[blocks][0]]
        else:
            raise ValueError('Invalid number of blocks.')

        return np.sum(rates_cross) / self.get_scaled_pop_size_coalescence()

    def get_rate_unlinked_migration(self) -> float:
        """
        Get the rate of a migration event.
        """
        # get the deme names
        source = self.state_space.epoch.pop_names[self.deme_migration_source]
        dest = self.state_space.epoch.pop_names[self.deme_migration_dest]

        # get the number of lineages in deme i before migration
        n_lineages_source = self.unlinked1[0, self.deme_migration_source, self.block_migration]

        # get migration rate from source to destination
        migration_rate = self.state_space.epoch.migration_rates[(source, dest)]

        # scale migration rate by number of lineages in source deme
        rate = migration_rate * n_lineages_source

        return rate

    def get_rate_linked_migration(self) -> float:
        """
        Get the rate of a migration event.
        """
        # get the deme names
        source = self.state_space.epoch.pop_names[self.deme_migration_source]
        dest = self.state_space.epoch.pop_names[self.deme_migration_dest]

        # get the number of lineages in deme i before migration
        n_lineages_source = self.linked1[0, self.deme_migration_source, self.block_migration]

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
        # if using the default state space.
        rate = 0

        if self.is_unlinked_coalescence:
            rate += self.get_rate_unlinked_coalescence()

        if self.is_mixed_coalescence:
            rate += self.get_rate_mixed_coalescence()

        return rate
