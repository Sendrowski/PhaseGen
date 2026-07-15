"""
Test-side parity helpers. Build the legacy pure-Python state space corresponding to a current one, and the
permutation reordering the legacy states to match the current enumeration. Used only by the suite, to validate the
current state-space construction against the reference implementation in :mod:`testing.state_space_old`.
"""
from typing import List

import numpy as np

from phasegen.state_space import StateSpace, LineageCountingStateSpace, BlockCountingStateSpace
from testing.state_space_old import (
    StateSpace as OldStateSpace,
    LineageCountingStateSpace as OldLineageCountingStateSpace,
    BlockCountingStateSpace as OldBlockCountingStateSpace,
)


def build_old(ss: StateSpace) -> OldStateSpace:
    """
    Build the legacy state space matching ``ss`` (same lineage/locus config, model and epoch).

    :param ss: A current state space.
    :return: The corresponding legacy state space.
    :raises NotImplementedError: For state spaces with no legacy equivalent (e.g. the joint / two-locus spaces).
    """
    kwargs = dict(lineage_config=ss.lineage_config, locus_config=ss.locus_config, model=ss.model, epoch=ss.epoch)

    if isinstance(ss, LineageCountingStateSpace):
        return OldLineageCountingStateSpace(**kwargs)

    if isinstance(ss, BlockCountingStateSpace):
        return OldBlockCountingStateSpace(**kwargs)

    raise NotImplementedError(f"No legacy equivalent for {type(ss).__name__}.")


def old_ordering(ss: StateSpace) -> List[int]:
    """
    The permutation reordering the legacy states to match ``ss``'s enumeration.

    :param ss: A current state space.
    :return: For each current state index, the index of the matching legacy state.
    """
    old = build_old(ss)

    return [
        int(np.where(((old.states == ss.lineages[i]) & (old.linked == ss.linked[i])).all(axis=(1, 2, 3)))[0][0])
        for i in range(ss.k)
    ]
