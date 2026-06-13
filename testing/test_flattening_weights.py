"""
Tests for the closed-form (combinatorial) flattening weights.

The flattened SFS used to obtain its per-lineage-level weights by traversing the ``p(n)``-state block-counting
space and weighting each state by its conditional probability. The fast path replaces that with the closed-form
Kingman block-size expectation ``E[# size-b blocks | k] = k C(n-b-1, k-2) / C(n-1, k-1)``. These tests reimplement
the original traversal as the reference and assert the new path reproduces it exactly, for both the unfolded and
the folded SFS, and that the fast path no longer builds the block-counting space.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.settings import Settings
from phasegen.rewards import (
    UnfoldedSFSReward, FoldedSFSReward, UnitReward, CombinedReward, TotalBranchLengthReward,
)


@pytest.fixture(autouse=True)
def _restore_flatten():
    prev = Settings.flatten_block_counting
    yield
    Settings.flatten_block_counting = prev


def _traversal_weights(state_space, reward, n):
    """
    Reference: the original flattening weights, obtained by traversing the block-counting state space and weighting
    each state's reward by its probability conditioned on the number of lineages. ``weights[n - k] = E[reward | k]``.
    """
    probs = state_space._state_probs
    r = reward._get(state_space)
    weights = np.zeros(n)
    for idx, s in enumerate(state_space.states):
        weights[n - s.lineages.sum()] += probs[idx] * r[idx]
    return weights


@pytest.mark.parametrize("n", [4, 6, 9])
def test_combinatorial_weights_match_traversal_unfolded(n):
    """The closed-form weights equal the block-counting traversal for every unfolded SFS bin."""
    dist = pg.Coalescent(n=n).sfs
    block_ss = dist.state_space
    for i in range(1, n):
        reward = CombinedReward([UnitReward(), UnfoldedSFSReward(i)])  # as the SFS moment path builds it
        new = dist._flattened_sfs_weights(reward, n)
        ref = _traversal_weights(block_ss, reward, n)
        assert new is not None, f"n={n} unfolded bin={i}: expected the closed-form path"
        np.testing.assert_allclose(new, ref, atol=1e-12, err_msg=f"n={n} unfolded bin={i}")


@pytest.mark.parametrize("n", [4, 6, 9])
def test_combinatorial_weights_match_traversal_folded(n):
    """The closed-form weights equal the block-counting traversal for every folded SFS bin (i and n-i)."""
    dist = pg.Coalescent(n=n).fsfs
    block_ss = dist.state_space
    for i in range(1, n // 2 + 1):
        reward = CombinedReward([UnitReward(), FoldedSFSReward(i)])
        new = dist._flattened_sfs_weights(reward, n)
        ref = _traversal_weights(block_ss, reward, n)
        assert new is not None, f"n={n} folded bin={i}: expected the closed-form path"
        np.testing.assert_allclose(new, ref, atol=1e-12, err_msg=f"n={n} folded bin={i}")


def test_unsupported_reward_falls_back():
    """A non-SFS reward (or a non-unit product of SFS rewards) yields None, so the caller uses the traversal."""
    dist = pg.Coalescent(n=5).sfs
    assert dist._flattened_sfs_weights(TotalBranchLengthReward(), 5) is None
    assert dist._flattened_sfs_weights(CombinedReward([UnfoldedSFSReward(1), UnfoldedSFSReward(2)]), 5) is None


@pytest.mark.parametrize("folded", [False, True])
def test_flattening_matches_full_block_space(folded):
    """End-to-end: the flattened (closed-form) SFS mean equals the full block-counting computation."""
    n = 12

    Settings.flatten_block_counting = True
    fast = np.asarray((pg.Coalescent(n=n).fsfs if folded else pg.Coalescent(n=n).sfs).mean.data)

    Settings.flatten_block_counting = False
    full = np.asarray((pg.Coalescent(n=n).fsfs if folded else pg.Coalescent(n=n).sfs).mean.data)

    np.testing.assert_allclose(fast, full, atol=1e-10)


@pytest.mark.parametrize("dist_name", ["sfs", "fsfs"])
def test_fast_flattening_does_not_build_block_space(dist_name):
    """The closed-form path must not construct the (large) block-counting state space."""
    Settings.flatten_block_counting = True
    dist = getattr(pg.Coalescent(n=40), dist_name)
    _ = dist.mean

    # the block-counting state-space object exists, but its expensive states/rate matrix were never built
    assert "states" not in dist.state_space.__dict__, f"{dist_name}: block states were built"
    assert "S" not in dist.state_space.__dict__, f"{dist_name}: block rate matrix was built"
