"""
Tests that the per-distribution memos of the moment evaluation spare repeated work without changing what it computes.
"""
import logging
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp

import phasegen as pg
from phasegen.distributions import PhaseTypeDistribution
from phasegen.distributions.phase_type import TreeHeightDistribution


def _growth(n: int = 4) -> pg.Coalescent:
    """Open-ended discretized exponential growth, whose epochs continue past the time of almost sure absorption."""
    return pg.Coalescent(n=n, demography=pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1.0}, growth_rate=1.0, start_time=0.0
    )]))


def test_higher_orders_resume_the_epoch_search_of_the_first_order(monkeypatch):
    """The epochs before the time of almost sure absorption, and the survival and scale there, do not depend on the
    order. Regression: every order of a demography held past that time repeated the propagation to it, the survival
    sweep and the scale solve, which made tree-height moments about 1.8 times slower."""
    calls = Counter()
    survival = TreeHeightDistribution._survival

    def counted(self, t):
        calls['survival'] += 1
        return survival(self, t)

    monkeypatch.setattr(TreeHeightDistribution, '_survival', counted)

    coal = _growth()
    moments = [coal.tree_height.moment(k, center=False) for k in range(1, 5)]

    assert coal.tree_height._epochs_cache[1][1]
    assert calls['survival'] == 1

    # the full search of each order on a fresh distribution gives the same epochs and moments
    for k in range(2, 5):
        fresh = _growth()
        fresh.tree_height._get_epochs_until_unbounded(1)
        del fresh.tree_height.__dict__['_epochs_search']

        expected = fresh.tree_height._get_epochs_until_unbounded(k)
        epochs = coal.tree_height._get_epochs_until_unbounded(k)

        assert [(e.start_time, e.end_time, e.index) for e in epochs] == \
               [(e.start_time, e.end_time, e.index) for e in expected]
        assert fresh.tree_height.moment(k, center=False) == pytest.approx(moments[k - 1], rel=1e-12)


def test_absorption_gate_reuses_the_last_epoch_reachability(monkeypatch):
    """The certainty of absorption and the support of the closed form read one memoized last-epoch reachability.
    Regression: the certainty check computed it a second time, which made two-deme joint SFS means about 1.3 times
    slower."""
    calls = Counter()
    reaches = PhaseTypeDistribution._reaches_absorption

    def counted(self):
        calls[id(self.state_space)] += 1
        return reaches(self)

    monkeypatch.setattr(PhaseTypeDistribution, '_reaches_absorption', counted)

    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.5}},
        migration_rates={('pop_0', 'pop_1'): {0: 0.75}, ('pop_1', 'pop_0'): {0: 0.75}}
    ))
    coal.jsfs.mean

    assert calls and set(calls.values()) == {1}


def test_stability_check_agrees_on_dense_and_sparse_generators(caplog):
    """The communicating-class analysis of the stability check reads the off-diagonal rates of a dense generator and
    of its sparse form alike, here two demes whose migration hides the slow coalescence from the exit rates."""
    def build() -> pg.Coalescent:
        return pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.0}},
            migration_rates={('pop_0', 'pop_1'): {0: 1e12}, ('pop_1', 'pop_0'): {0: 1e12}}
        ))

    messages = []
    for to_sparse in (False, True):
        dist = build().tree_height
        S = np.asarray(dist.state_space.S)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='phasegen'):
            dist._check_numerical_stability(sp.csr_matrix(S) if to_sparse else S, 0)

        messages.append([r.getMessage() for r in caplog.records])

    assert messages[0] and messages[0] == messages[1]


def test_higher_orders_after_a_serialization_round_trip_of_the_epoch_search():
    """A payload carrying the stored search of the first order gives the higher moments of a fresh coalescent."""
    coal = _growth()
    coal.tree_height.moment(1, center=False)

    restored = pg.Coalescent.from_json(coal.to_json())

    for k in range(2, 4):
        assert restored.tree_height.moment(k, center=False) == pytest.approx(
            _growth().tree_height.moment(k, center=False), rel=1e-12)
