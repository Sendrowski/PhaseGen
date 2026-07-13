"""
Validate the analytic nested-inversion conditional against an msprime ground truth.

A conditional ``L_j | L_i = x`` (one SFS bin's branch length given another's) has no closed-form msprime
analogue, but it does have a *binned* one: simulate the joint SFS branch lengths, keep the replicates whose
``L_i`` lands in a narrow window around ``x``, and read off the empirical distribution of their ``L_j``. This
is the only external check on the conditional's *shape* (the self-consistency tests in
``test_reward_distribution`` pin only the first moment via the law of total expectation), and it exercises the
double de Hoog inversion of :class:`~phasegen.distributions.reward.JointRewardDistribution.conditional`.

The comparison uses the atom-correct ``cdf.curve`` (the monotone de Hoog spline backing the plots), not the
per-point ``cdf`` scalar, which returns the left limit ``0`` at exactly ``t = 0`` and so omits ``P(R = 0)``.
"""
import numpy as np
import pytest

import phasegen as pg
from testing import TestCase


class ConditionalMsprimeTestCase(TestCase):
    """
    The nested-inversion conditional ``L_j | L_i = x`` against a binned msprime ground truth, for an
    off-diagonal SFS bin pair with strong dependence and one that is nearly independent.
    """

    N = 6
    NUM_REPLICATES = 100000
    SEED = 42
    #: fraction of ``std(L_i)`` used as the half-width of the conditioning window around ``x``
    WINDOW = 0.06
    #: quantiles of the (positive) L_i sample used as conditioning values
    QUANTILES = (0.4, 0.65)
    #: bin pairs: (2, 4) is strongly dependent, (1, 3) is nearly independent
    PAIRS = ((2, 4), (1, 3))

    @classmethod
    def setUpClass(cls) -> None:
        """One serial, seeded msprime simulation, reused across every conditioning case."""
        ms = pg.Coalescent(n=cls.N).to_msprime(
            num_replicates=cls.NUM_REPLICATES, parallelize=False, n_threads=1, seed=cls.SEED
        )
        cls.samples = np.asarray(ms.sfs.samples)  # (num_replicates, n + 1)
        cls.coalescent = pg.Coalescent(n=cls.N)

    def _empirical_conditional(self, i: int, j: int, x: float, h: float) -> np.ndarray:
        """Sorted ``L_j`` over the replicates whose ``L_i`` lies within ``[x - h, x + h]``."""
        li, lj = self.samples[:, i], self.samples[:, j]
        return np.sort(lj[(li >= x - h) & (li <= x + h)])

    def test_conditional_matches_binned_msprime(self):
        """The analytic conditional CDF and mean track the binned msprime ground truth as the conditioning
        value moves, for both a strongly dependent and a nearly independent SFS bin pair."""
        for i, j in self.PAIRS:
            li = self.samples[:, i]
            h = self.WINDOW * float(li.std())
            jd = self.coalescent.sfs.joint_distribution(i, j)

            for q in self.QUANTILES:
                x = float(np.quantile(li[li > 0], q))
                lj = self._empirical_conditional(i, j, x, h)
                assert lj.size > 3000, f"pair {(i, j)} q={q}: only {lj.size} conditioned replicates"

                cond = jd.conditional('a', x)

                # conditional mean shifts strongly with x -> a discriminating check on the nested inversion.
                # The window averages the conditional mean over [x - h, x + h], a bias (~a few %) that does not
                # shrink with more replicates, so the tolerance sits comfortably above it rather than pinning it.
                mean_emp = float(lj.mean())
                mean_ana = float(cond.mean)
                assert abs(mean_emp - mean_ana) < 0.08 * mean_emp + 0.01, \
                    f"pair {(i, j)} L_{i}={x:.3f}: E_emp={mean_emp:.4f} vs E_ana={mean_ana:.4f}"

                # CDF shape: the atom-correct de Hoog curve vs the empirical CDF over the bulk
                grid = np.linspace(0.0, float(np.quantile(lj, 0.99)) + 1e-9, 40)
                emp = np.searchsorted(lj, grid, side='right') / lj.size
                ana = np.asarray(cond.cdf(grid), dtype=float)
                max_abs = float(np.abs(emp - ana).max())
                assert max_abs < 0.05, \
                    f"pair {(i, j)} L_{i}={x:.3f}: max|F_emp - F_ana|={max_abs:.4f}"
