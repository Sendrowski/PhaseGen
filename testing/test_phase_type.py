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
