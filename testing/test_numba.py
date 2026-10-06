"""
Tests for the numba-accelerated state-space construction.

These verify that the numba path reproduces the pure-Python construction (up to the allowed state reordering) across
state-space types and coalescent models, and that forcing the Python fallback gives identical results.
"""
import numpy as np
import pytest

import phasegen as pg
from phasegen.settings import Settings
from phasegen.state_space import (
    LineageCountingStateSpace,
    BlockCountingStateSpace,
    JointBlockCountingStateSpace,
)


def _demography(pop_sizes, migration_rate=1.0):
    """Two-or-more-deme demography with symmetric migration (or a single deme with no migration)."""
    from itertools import product

    pops = list(pop_sizes)

    if len(pops) == 1:
        return pg.Demography(pop_sizes=pop_sizes)

    migration_rates = {(a, b): migration_rate for a, b in product(pops, repeat=2) if a != b}

    return pg.Demography(pop_sizes=pop_sizes, migration_rates=migration_rates)


def _build_S(make_state_space, use_numba):
    """Build a state space with the given construction path and return its states and rate matrix."""
    Settings.use_numba = use_numba
    ss = make_state_space()
    rows = ss.lineages[:, 0, :, :].reshape(ss.k, -1)
    return rows, np.asarray(ss.S)


def _assert_S_parity(make_state_space, label, atol=1e-11):
    """Assert that the numba and Python rate matrices agree up to a state (row/column) permutation."""
    rows_p, S_p = _build_S(make_state_space, use_numba=False)
    rows_n, S_n = _build_S(make_state_space, use_numba=True)

    assert len(rows_n) == len(rows_p), f"{label}: state count {len(rows_n)} != {len(rows_p)}"

    index = {tuple(r): i for i, r in enumerate(rows_p)}
    perm = np.array([index[tuple(r)] for r in rows_n])  # numba index -> python index

    reordered = np.zeros_like(S_p)
    reordered[np.ix_(perm, perm)] = S_n

    assert np.abs(reordered - S_p).max() < atol, f"{label}: max|S| diff {np.abs(reordered - S_p).max():.2e}"


# state spaces spanning all three types and all three models
STATE_SPACES = [
    ("lineage 1-deme n=8 std",
     lambda: LineageCountingStateSpace(lineage_config=pg.LineageConfig(8),
                                       epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("lineage 2-deme n=3+3 std",
     lambda: LineageCountingStateSpace(lineage_config=pg.LineageConfig({'pop_0': 3, 'pop_1': 3}),
                                       epoch=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75).get_epoch(0))),
    ("block n=7 std",
     lambda: BlockCountingStateSpace(lineage_config=pg.LineageConfig(7),
                                     epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("block n=7 beta",
     lambda: BlockCountingStateSpace(lineage_config=pg.LineageConfig(7), model=pg.BetaCoalescent(alpha=1.5),
                                     epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("block n=7 dirac",
     lambda: BlockCountingStateSpace(lineage_config=pg.LineageConfig(7), model=pg.DiracCoalescent(psi=0.5, c=1.0),
                                     epoch=_demography({'pop_0': 1.0}).get_epoch(0))),
    ("joint 2-deme n=3+3 std",
     lambda: JointBlockCountingStateSpace(lineage_config=pg.LineageConfig({'pop_0': 3, 'pop_1': 3}),
                                          epoch=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75).get_epoch(0))),
    ("joint 3-deme n=2+1+1 std",
     lambda: JointBlockCountingStateSpace(lineage_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 1, 'pop_2': 1}),
                                          epoch=_demography({'pop_0': 1.0, 'pop_1': 1.0, 'pop_2': 1.0}).get_epoch(0))),
]


@pytest.mark.parametrize("label, make", STATE_SPACES, ids=[s[0] for s in STATE_SPACES])
def test_rate_matrix_parity_up_to_permutation(label, make):
    """The numba rate matrix matches the pure-Python one up to a state permutation."""
    _assert_S_parity(make, label)


# end-to-end moment parity through the public API
MOMENT_CASES = [
    ("sfs n=6", lambda: pg.Coalescent(n=6), lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs beta n=6", lambda: pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.5)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("sfs dirac n=6", lambda: pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.5, c=1.0)),
     lambda c: np.asarray(c.sfs.mean.data)),
    ("tree height variance n=10", lambda: pg.Coalescent(n=10), lambda c: c.tree_height.var),
    ("jsfs 2-deme n=2+2", lambda: pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=_demography({'pop_0': 1.0, 'pop_1': 1.5}, 0.75)), lambda c: np.asarray(c.jsfs.mean.data)),
]


@pytest.mark.parametrize("label, make, get", MOMENT_CASES, ids=[m[0] for m in MOMENT_CASES])
def test_moment_parity_numba_vs_python(label, make, get):
    """Moments computed via the numba and Python construction paths agree to floating-point tolerance."""
    Settings.use_numba = True
    numba = np.asarray(get(make()))

    Settings.use_numba = False
    python = np.asarray(get(make()))

    np.testing.assert_allclose(numba, python, atol=1e-10, err_msg=label)


def test_two_loci_uses_numba_path():
    """The 2-locus (recombination) lineage-counting space is numba-accelerated (kernel kind 3), and the numba and
    pure-Python constructions agree -- validated end-to-end on the tree height (its generator is identical up to a
    state permutation, so the moments match to floating-point tolerance)."""
    assert pg.Coalescent(n=4, loci=2, recombination_rate=1.0).lineage_counting_state_space._use_numba()

    Settings.use_numba = True
    mean_numba = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).tree_height.mean

    Settings.use_numba = False
    mean_python = pg.Coalescent(n=4, loci=2, recombination_rate=1.0).tree_height.mean

    assert mean_numba > 0
    np.testing.assert_allclose(mean_numba, mean_python, atol=1e-10)


def test_fallback_setting_disables_numba():
    """Setting ``Settings.use_numba = False`` routes construction through the pure-Python path."""
    Settings.use_numba = False
    ss = BlockCountingStateSpace(lineage_config=pg.LineageConfig(5),
                                 epoch=_demography({'pop_0': 1.0}).get_epoch(0))

    assert not ss._use_numba()
    assert ss.S.shape[0] == ss.k


class _DoubledRateCoalescent(pg.StandardCoalescent):
    """Standard coalescent whose merger rates are doubled."""

    def _get_rate(self, b: int, k: int) -> float:
        """Twice the standard rate."""
        return 2 * super()._get_rate(b, k)


def test_model_subclass_uses_the_python_construction():
    """The numba kernels implement the rates of the built-in models only. A subclass takes the pure-Python
    construction, which evaluates its rates, whatever ``Settings.use_numba`` is. Regression: a subclass was dispatched
    to the kernel of its built-in ancestor, so overridden rates were silently ignored (E[T_MRCA] = 1 at n = 2 for
    doubled rates), and later the construction raised NotImplementedError on the same coalescent even after
    ``Settings.use_numba = False``."""
    from phasegen.state_space import _numba_model_params

    with pytest.raises(NotImplementedError, match='_DoubledRateCoalescent'):
        _numba_model_params(_DoubledRateCoalescent())

    coal = pg.Coalescent(n=2, model=_DoubledRateCoalescent())
    np.testing.assert_allclose(coal.tree_height.mean, 0.5)
    assert not coal.lineage_counting_state_space._use_numba()


@pytest.mark.parametrize('first', [True, False])
def test_toggling_use_numba_after_the_states_keeps_their_order(first):
    """The construction path is fixed when the states are built, so toggling ``Settings.use_numba`` afterwards leaves
    every epoch's rate matrix in the order of the states. Regression: the rate matrix of a later epoch was built by
    the other path, whose state order differs, giving a two-locus tree height of 0.7236 against 0.8035 and a two-deme
    joint SFS off by up to 4.66."""
    def two_loci():
        return pg.Coalescent(n=3, loci=2, recombination_rate=1.0,
                             demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}}))

    def two_demes():
        return pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(
            pop_sizes={'a': {0: 1, 0.5: 3}, 'b': {0: 2}}, migration_rates={('a', 'b'): 1, ('b', 'a'): 0.5}))

    for make, space, stat in [
        (two_loci, 'lineage_counting_state_space', lambda c: c.tree_height.mean),
        (two_demes, 'joint_block_counting_state_space', lambda c: c.jsfs.mean.data),
    ]:
        Settings.use_numba = first
        expected = np.asarray(stat(make()))

        coal = make()
        _ = getattr(coal, space).states
        Settings.use_numba = not first

        np.testing.assert_allclose(np.asarray(stat(coal)), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('model', [pg.BetaCoalescent(alpha=1.5), pg.DiracCoalescent(psi=0.3, c=2)])
def test_multiple_merger_rates_are_finite_and_exact_for_large_samples(model):
    """The kernel rates agree with the model formulae on both sides of the switch to log space, and stay finite beyond
    the range of a float binomial coefficient. Regression: they overflowed to inf (nan for Beta) from about 1026
    lineages."""
    from scipy.special import betaln, gammaln
    from phasegen.state_space import _numba_model_params
    from phasegen.state_space_numba import _rate_block, _rate_pairwise

    model_id, alpha, psi, c = _numba_model_params(model)

    for b, k in [(2, 2), (7, 3), (40, 17), (100, 50)]:
        assert _rate_pairwise(model_id, alpha, psi, c, b, k) == pytest.approx(model._get_rate(b=b, k=k), rel=1e-12)
        assert _rate_block(model_id, alpha, psi, c, b + 3, np.array([b, 3]), np.array([k, 1])) == pytest.approx(
            model._get_rate_block_counting(n=b + 3, b=[b, 3], k=[k, 1]), rel=1e-12)

    log_comb = lambda n, j: gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)

    for b in [511, 512, 1100, 3000]:
        for k in [2, b // 2, b]:
            if model_id == 1:
                expected = np.exp(log_comb(b, k) + betaln(k - alpha, b - k + alpha) - betaln(alpha, 2 - alpha))
            else:
                expected = (b * (b - 1) / 2 if k == 2 else 0) + c * np.exp(
                    log_comb(b, k) + k * np.log(psi) + (b - k) * np.log1p(-psi))

            assert _rate_pairwise(model_id, alpha, psi, c, b, k) == pytest.approx(expected, rel=1e-10)

    assert 0 < _rate_block(model_id, alpha, psi, c, 1103, np.array([1100, 3]), np.array([550, 1])) < np.inf


def test_python_beta_rates_match_the_kernel_in_log_space():
    """The Python Beta rates switch to log space at the kernel's threshold and agree with the kernel bitwise from
    there on, staying finite beyond the range of a float binomial coefficient. Regression: _get_rate raised
    OverflowError from about 1030 lineages."""
    from phasegen.state_space_numba import _LOG_SPACE_MIN_LINEAGES, _rate_block, _rate_pairwise

    model = pg.BetaCoalescent(alpha=1.5)

    for b in [_LOG_SPACE_MIN_LINEAGES, 1100, 3000]:
        for k in [2, b // 2, b]:
            rate = model._get_rate(b=b, k=k)
            assert 0 <= rate < np.inf
            assert rate == _rate_pairwise(1, 1.5, 0.0, 0.0, b, k)

        assert model._get_rate_block_counting(n=b + 3, b=[b, 3], k=[b // 2, 1]) == _rate_block(
            1, 1.5, 0.0, 0.0, b + 3, np.array([b, 3]), np.array([b // 2, 1]))

    b = _LOG_SPACE_MIN_LINEAGES - 1
    assert model._get_rate(b=b, k=b // 2) == pytest.approx(_rate_pairwise(1, 1.5, 0.0, 0.0, b, b // 2), rel=1e-12)


def test_the_pure_python_construction_logs_its_deprecation_once(caplog):
    """Choosing the pure-Python construction is logged as a deprecation warning, once per state space. Regression:
    a DeprecationWarning attributed to the package was hidden by the default filters."""
    Settings.use_numba = False
    ss = LineageCountingStateSpace(pg.LineageConfig(n=3))
    _ = ss.S
    ss.update_epoch(pg.Epoch(pop_sizes={'pop_0': 2}))
    _ = ss.S

    assert sum('pure-Python construction, which is deprecated' in r.getMessage() for r in caplog.records) == 1
