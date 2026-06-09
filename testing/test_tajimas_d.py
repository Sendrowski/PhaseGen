"""
Test Tajima's D and the two theta estimators (branch form) on ``UnfoldedSFSDistribution``.

These are simulation-free SFS functionals: ``pi`` and ``theta_W`` are linear functionals of the mean SFS, and
Tajima's D standardizes their difference by the SFS covariance. The diagnostic signs are checked against
well-known demographic expectations.
"""
import numpy as np
import pytest

import phasegen as pg


def _growth(end_size: float) -> pg.Demography:
    """Single population that was smaller in the past (recent expansion when ``end_size < 1``)."""
    return pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: end_size}})


def test_neutral_constant_size_is_zero():
    """Under the standard neutral constant-size model E[pi] = E[theta_W], so Tajima's D = 0."""
    coal = pg.Coalescent(n=10)
    assert coal.sfs.theta_pi == pytest.approx(coal.sfs.theta_w, abs=1e-9)
    assert coal.sfs.tajimas_d == pytest.approx(0.0, abs=1e-9)


def test_growth_gives_negative_d():
    """Population expansion produces an excess of low-frequency variants, so pi < theta_W and D < 0."""
    coal = pg.Coalescent(n=10, demography=_growth(0.05))
    assert coal.sfs.theta_pi < coal.sfs.theta_w
    assert coal.sfs.tajimas_d < 0


def test_contraction_gives_positive_d():
    """Population contraction produces an excess of intermediate-frequency variants, so pi > theta_W and D > 0."""
    coal = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.1: 5.0}}))
    assert coal.sfs.theta_pi > coal.sfs.theta_w
    assert coal.sfs.tajimas_d > 0


def test_theta_estimators_positive():
    """Both theta estimators are positive."""
    coal = pg.Coalescent(n=6)
    assert coal.sfs.theta_pi > 0
    assert coal.sfs.theta_w > 0


@pytest.mark.parametrize("model", [pg.BetaCoalescent(alpha=1.5), pg.DiracCoalescent(psi=0.5, c=1.0)])
def test_multiple_merger_models_supported(model):
    """Tajima's D is well-defined under multiple-merger models (which themselves skew the SFS)."""
    d = pg.Coalescent(n=8, model=model).sfs.tajimas_d
    assert np.isfinite(d)
