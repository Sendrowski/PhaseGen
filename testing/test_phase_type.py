"""
Test the tree-height quantile grid and the sweep of ``TreeHeightDistribution``.
"""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import brentq

import phasegen as pg
from phasegen.distributions import TreeHeightDistribution


def _exact_quantile(th: TreeHeightDistribution, q: float) -> float:
    """
    The level ``q`` inverted by root finding on the exact cumulative hazard of the propagated state distribution,
    read off the absorbed mass below a CDF of one half and off the surviving mass above.
    """
    e = np.asarray(th._e, dtype=float)

    def hazard(x: float) -> float:
        w = th._sweep_to(np.asarray(th.state_space.alpha, dtype=float), 0.0, x, th.demography.get_epoch(0))
        cdf, survival = w @ (1 - e) / w.sum(), w @ e / w.sum()
        return -np.log1p(-cdf) if cdf <= 0.5 else -np.log(survival)

    log_t_max = np.log(th.t_max)

    return float(np.exp(brentq(lambda lx: hazard(np.exp(lx)) + np.log1p(-q), log_t_max - 50, log_t_max,
                               xtol=1e-15, rtol=1e-15)))


@pytest.mark.parametrize('pop_sizes', [
    {0: 1, 0.1: 1e-3, 0.11: 1},
    {0: 1, 0.5: 0.2, 1.5: 2},
])
def test_tree_height_quantile_matches_exact_inversion(pop_sizes):
    """
    The quantile inverts the exact CDF to a relative error of 1e-8 from the lower to the upper tail, across a
    bottleneck and across epoch boundaries. The fixed grid of 8192 nodes interpolated linearly in the cumulative hazard
    was off by up to 2e-5.
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

    # an exponential tree height has a linear cumulative hazard, which a single segment interpolates exactly
    assert len(pg.Coalescent(n=2).tree_height.quantile._cdf_grid()[0]) == 3


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


def test_tree_height_quantile_without_cache():
    """
    With caching disabled the grid is rebuilt for every call and gives the same quantiles.
    """
    th = pg.Coalescent(n=6).tree_height
    expected = th.quantile([0.1, 0.5, 0.9])

    prev = pg.Settings.cache
    try:
        pg.Settings.cache = False
        np.testing.assert_array_equal(pg.Coalescent(n=6).tree_height.quantile([0.1, 0.5, 0.9]), expected)
    finally:
        pg.Settings.cache = prev


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
