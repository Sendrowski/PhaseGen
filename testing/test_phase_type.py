"""
Test the tree-height quantile grid and the sweep of ``TreeHeightDistribution``.
"""

import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import brentq

import phasegen as pg
from phasegen.distributions import TreeHeightDistribution


def _exact_quantile(th: TreeHeightDistribution, q: float) -> float:
    """
    The level ``q`` inverted by root finding on the exact negative log-survival of the propagated state distribution,
    read off the absorbed mass below a CDF of one half and off the surviving mass above.
    """
    e = np.asarray(th._e, dtype=float)

    def log_survival(x: float) -> float:
        w = th._sweep_to(np.asarray(th.state_space.alpha, dtype=float), 0.0, x, th.demography.get_epoch(0))
        cdf, survival = w @ (1 - e) / w.sum(), w @ e / w.sum()
        return -np.log1p(-cdf) if cdf <= 0.5 else -np.log(survival)

    log_t_max = np.log(th.t_max)

    return float(np.exp(brentq(lambda lx: log_survival(np.exp(lx)) + np.log1p(-q), log_t_max - 50, log_t_max,
                               xtol=1e-15, rtol=1e-15)))


@pytest.mark.parametrize('pop_sizes', [
    {0: 1, 0.1: 1e-3, 0.11: 1},
    {0: 1, 0.5: 0.2, 1.5: 2},
])
def test_tree_height_quantile_matches_exact_inversion(pop_sizes):
    """
    The quantile inverts the exact CDF to a relative error of 1e-8 from the lower to the upper tail, across a
    bottleneck and across epoch boundaries. The fixed grid of 8192 nodes interpolated linearly in the negative
    log-survival was off by up to 2e-5.
    """
    th = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={'pop_0': pop_sizes})).tree_height
    levels = np.array([1e-12, 1e-6, 1e-3, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999, 1 - 1e-9, 1 - 1e-12])

    exact = np.array([_exact_quantile(th, q) for q in levels])

    np.testing.assert_allclose(th.quantile(levels), exact, rtol=1e-8, atol=0)


def test_tree_height_quantile_grid_is_error_controlled():
    """
    The quantile grid holds only the nodes its tolerance needs. The fixed grid built 8192 nodes for any coalescent, so
    a few quantiles of a fresh coalescent cost several times more than a per-level bisection.
    """
    th = pg.Coalescent(n=10).tree_height

    assert len(th.quantile._cdf_grid()[0]) < 2000

    # an exponential tree height has a linear negative log-survival, which a single segment interpolates exactly
    assert len(pg.Coalescent(n=2).tree_height.quantile._cdf_grid()[0]) == 3


def test_tree_height_quantile_grid_under_recent_growth_takes_few_exponentials():
    """
    Under a recent 1e4-fold expansion ``t_max`` lies about 1e4 times beyond the rise of the CDF. Bisecting each epoch
    from its full width formed a propagator for every one of the about 15 halvings down to the rise, 143 matrix
    exponentials in all, and made the quantile about twice as slow as the fixed grid it replaced at n of 200 to
    300. The grid keeps its accuracy.
    """
    import phasegen.distributions.phase_type as phase_type

    th = pg.Coalescent(n=50, demography=pg.Demography(pop_sizes={'pop_0': {0: 1e4, 0.1: 1}})).tree_height
    levels = np.array([1e-6, 0.01, 0.5, 0.9, 0.99])
    _ = th.t_max

    with patch.object(phase_type, 'expm', wraps=phase_type.expm) as expm:
        q = th.quantile(levels)

    assert expm.call_count <= 40

    np.testing.assert_allclose(-np.log1p(-th.cdf(q)), -np.log1p(-levels), rtol=1e-8, atol=0)


class _LogBudget:
    """
    Stands in for the ``math`` module of ``phase_type``, raising once the grid has taken more than ``budget``
    logarithms, about one per evaluated node, so that a grid that does not terminate fails fast.
    """

    def __init__(self, budget: int):
        self.calls, self.budget = 0, budget

    def __getattr__(self, name):
        return getattr(math, name)

    def _spend(self):
        self.calls += 1

        if self.calls > self.budget:
            raise RuntimeError(f'the grid took more than {self.budget} logarithms')

    def log(self, x):
        self._spend()
        return math.log(x)

    def log1p(self, x):
        self._spend()
        return math.log1p(x)


def _island(n: dict, within: float, between: float, pairs: bool) -> pg.Coalescent:
    """
    Four demes of size 1. With ``pairs`` the demes form the pairs ``(pop_0, pop_1)`` and ``(pop_2, pop_3)``, with rate
    ``within`` inside a pair and ``between`` across. Otherwise every rate is ``between``.
    """
    pops = [f'pop_{i}' for i in range(4)]
    paired = lambda a, b: pairs and {a, b} in ({'pop_0', 'pop_1'}, {'pop_2', 'pop_3'})

    return pg.Coalescent(n=n, demography=pg.Demography(
        pop_sizes={p: 1 for p in pops},
        migration_rates={(a, b): within if paired(a, b) else between for a in pops for b in pops if a != b}))


@pytest.mark.parametrize('coal', [
    pytest.param(lambda: _island({f'pop_{i}': 1 for i in range(4)}, 0.1, 0.1, False), id='island_one_per_deme'),
    pytest.param(lambda: _island({f'pop_{i}': 2 for i in range(4)}, 2, 0.1, True), id='1_epoch_4_pops_tree_n_2'),
])
def test_tree_height_quantile_grid_terminates_when_the_ancestor_migrates(coal):
    """
    The single lineage left at absorption keeps migrating between demes, so the absorbing states carry rates whose
    rows sum to zero only up to rounding. The grid evaluates a bounded number of nodes and inverts the CDF to a
    relative error of 1e-8 in the negative log-survival, and the tail density is non-negative. Regression: the slope
    of the negative log-survival in the far tail read the rounding of those row sums, turned negative, and every
    segment there was bisected to the depth limit, so the quantile did not return.
    """
    import phasegen.distributions.phase_type as phase_type

    th = coal().tree_height
    levels = np.array([1e-6, 0.01, 0.5, 0.9, 0.99, 1 - 1e-6])
    _ = th.t_max

    budget = _LogBudget(4000)
    with patch.object(phase_type, 'math', budget):
        q = th.quantile(levels)

    assert 0 < budget.calls <= 4000
    np.testing.assert_allclose(-np.log1p(-th.cdf(q)), -np.log1p(-levels), rtol=1e-8, atol=0)
    assert np.all(th.pdf(np.linspace(0.5, 1, 6) * th.t_max) >= 0)


def test_tree_height_quantile_boundary_levels():
    """
    Levels 0 and 1 return the ends of the support, NaN levels return NaN, and the shape of the levels is kept.
    """
    th = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2}})).tree_height
    q = np.array([[0.0, 0.5], [np.nan, 1.0]])

    out = th.quantile(q)

    assert out.shape == (2, 2)
    assert out[0, 0] == 0
    assert out[1, 1] == th.t_max
    assert np.isnan(out[1, 0])
    assert th.quantile(0.5) == out[0, 1]


@pytest.mark.parametrize('cache', [True, False])
def test_tree_height_quantile_grid_honours_the_cache_flag(cache):
    """
    The grid is stored on the distribution and built once when caching is enabled, and rebuilt for every call and
    never stored when it is disabled. Either way the quantiles agree.
    """
    from phasegen.distributions.phase_type import _ExpmFunction

    expected = pg.Coalescent(n=6).tree_height.quantile([0.1, 0.5, 0.9])
    th = pg.Coalescent(n=6).tree_height

    prev = pg.Settings.cache
    try:
        pg.Settings.cache = cache

        with patch.object(_ExpmFunction, '_build_cdf_grid', autospec=True,
                          side_effect=_ExpmFunction._build_cdf_grid) as build:
            first = th.quantile([0.1, 0.5, 0.9])
            second = th.quantile([0.1, 0.5, 0.9])
    finally:
        pg.Settings.cache = prev

    assert build.call_count == (1 if cache else 2)
    assert ('expm_cdf_grid' in th.__dict__.get('_lst_curve_cache', {})) == cache
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


def test_sweep_reads_each_epoch_once():
    """
    A sweep over many points checks the stability of each epoch and reads its absorption rates once, not once per
    point.
    """
    th = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2, 1.5: 2}})).tree_height
    _ = th.t_max

    with patch.object(TreeHeightDistribution, '_check_numerical_stability', autospec=True,
                      side_effect=TreeHeightDistribution._check_numerical_stability) as stability, \
            patch.object(TreeHeightDistribution, '_exit_rates', autospec=True,
                         side_effect=TreeHeightDistribution._exit_rates) as rates:
        th._sweep(np.linspace(0, 4, 200))

    assert stability.call_count == 3
    assert rates.call_count == 3


def test_tree_height_pdf_lower_tail_relative_accuracy():
    """
    The density keeps its relative precision far into the lower tail of a two-deme coalescent with migration, where
    it falls to 1e-11. Absorption rates taken as the negated row sums over the transient columns cancel to a rounding
    residual on states that do not absorb directly, which put relative errors of up to 3e-5 on the density at a CDF of
    1e-15. The reference values are the densities of the same rate matrices in 60-digit arithmetic (mpmath), at the
    levels 1e-15, 1e-12 and 1e-6 of the CDF and at the median.
    """
    coal = pg.Coalescent(
        n={'pop_0': 2, 'pop_1': 2},
        demography=pg.Demography(
            pop_sizes={'pop_0': {0: 0.5, 1: 0.5}, 'pop_1': {0: 2}},
            migration_rates={('pop_0', 'pop_1'): {0: 1}, ('pop_1', 'pop_0'): {0: 0.2}}
        )
    )
    t = [0.00027026238749769724, 0.0015203809069530894, 0.04885328141256676, 3.616369621142335]
    expected = [1.4799206499923377e-11, 2.6296870454997604e-09, 8.042806633234013e-05, 0.16113377457891845]

    np.testing.assert_allclose(coal.tree_height.pdf(t), expected, rtol=1e-12, atol=0)


def _pointwise_sweep(th: TreeHeightDistribution, t: np.ndarray) -> tuple:
    """The CDF and density at each time from a sweep over that time alone."""
    return tuple(np.array(x) for x in zip(*[[v[0] for v in th._sweep(np.array([u]))] for u in t]))


def test_sweep_reuses_one_exponential_per_step():
    """
    A sweep over an equally spaced grid forms one dense exponential per epoch and step length, not one per point, and
    gives the values of sweeps over each point alone. ``np.linspace`` steps differ in their last bits, which the runs
    of equal steps absorb.
    """
    from phasegen.distributions import phase_type

    th = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2, 1.5: 2}})).tree_height
    t = np.linspace(0, 4, 400)
    _ = th.t_max

    with patch.object(phase_type, 'expm', side_effect=phase_type.expm) as spy:
        cdf, pdf = th._sweep(t)

    # three epochs, each with a first step from its start, the run of grid steps and, but for the last, its end
    assert spy.call_count <= 8

    cdf_ref, pdf_ref = _pointwise_sweep(th, t)
    np.testing.assert_allclose(cdf, cdf_ref, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(pdf, pdf_ref, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize('action', [False, True])
def test_sweep_over_unequal_grid_crossing_epochs(action):
    """
    A sweep over unequally spaced times, some of them on epoch boundaries, repeated and beyond the last boundary,
    agrees with sweeps over each time alone, on the dense path and on the sparse path, where equal steps take one
    interval action.
    """
    from phasegen.settings import Settings

    Settings.expm_action_min_dim = 0 if action else 10 ** 9
    th = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.2, 1.5: 2}})).tree_height

    rng = np.random.default_rng(0)
    t = np.sort(np.concatenate([rng.uniform(0, 5, 30), [0, 0.5, 0.5, 1.5], np.linspace(0.6, 1.4, 20)]))
    cdf, pdf = th._sweep(t)
    cdf_ref, pdf_ref = _pointwise_sweep(th, t)

    np.testing.assert_allclose(cdf, cdf_ref, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(pdf, pdf_ref, rtol=1e-11, atol=1e-15)


def test_step_runs_split_equal_steps():
    """
    Runs of equal steps continue only within a few unit roundoffs of the time reached, and times at or before it
    take a zero step.
    """
    from phasegen.distributions._grid import _StepRuns

    runs, end = _StepRuns.split(0.5, np.linspace(0.5, 2.5, 5)[1:])
    assert runs == [(0.5, 4)] and abs(end - 2.5) < 1e-15

    runs, end = _StepRuns.split(0.0, np.array([0.0, 1.0, 2.0, 2.0, 3.5, 5.0, np.inf, np.inf]))
    assert runs == [(0.0, 1), (1.0, 2), (0.0, 1), (1.5, 2), (np.inf, 1), (0.0, 1)] and end == np.inf

    runs, _ = _StepRuns.split(0.0, np.array([1.0, 2.0 + 1e-14]))
    assert len(runs) == 2


@pytest.mark.parametrize('action', [False, True])
def test_sweep_over_a_grid_longer_than_one_run(action):
    """
    A run of equal steps longer than ``_MAX_RUN_STEPS`` is cut into runs of at most that many steps, which bounds the
    vectors a sweep holds at once, and the sweep over the cut runs agrees with sweeps over single times, including
    those at the cuts. Uncut, an equispaced grid of 10^5 points on 1329 states held about 1 GB.
    """
    from phasegen.distributions._grid import _MAX_RUN_STEPS, _StepRuns
    from phasegen.settings import Settings

    t = np.linspace(0, 6, 3 * _MAX_RUN_STEPS + 6)
    runs, end = _StepRuns.split(0.0, t)

    assert [c for _, c in runs] == [1, _MAX_RUN_STEPS, _MAX_RUN_STEPS, _MAX_RUN_STEPS, 5] and abs(end - 6) < 1e-14

    Settings.expm_method = 'action' if action else 'dense'
    th = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 2: 0.3}})).tree_height
    cdf, pdf = th._sweep(t)

    idx = np.array([1, 2, _MAX_RUN_STEPS, _MAX_RUN_STEPS + 1, 2 * _MAX_RUN_STEPS + 1, 3 * _MAX_RUN_STEPS + 2,
                    len(t) - 1])
    cdf_ref, pdf_ref = _pointwise_sweep(th, t[idx])

    np.testing.assert_allclose(cdf[idx], cdf_ref, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(pdf[idx], pdf_ref, rtol=1e-11, atol=1e-15)
