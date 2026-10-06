"""
Targeted fast tests filling small coverage gaps in utility helpers, demographic events and the folded SFS.
"""
import numpy as np

import phasegen as pg
from phasegen.utils import take_n, takewhile_inclusive
from testing import TestCase


class CoverageGapsTestCase(TestCase):
    """
    Small, fast tests that exercise otherwise-uncovered helper paths.
    """

    def test_combined_reward_does_not_mutate_input_list(self):
        """CombinedReward must not mutate (or alias) the caller's reward list."""
        rewards = [pg.TotalBranchLengthReward(), pg.LocusReward(0)]
        pg.CombinedReward(rewards)
        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(rewards[0], pg.TotalBranchLengthReward)
        self.assertIsInstance(rewards[1], pg.LocusReward)

    def test_point_mass_reward_distribution_functions(self):
        """A reward that is zero almost surely (a full atom at 0) yields the trivial point-mass CDF / quantile
        without a division-by-zero crash in the cosine fit."""
        d = pg.Coalescent(n=4).distribution(pg.CustomReward(func=lambda ss: np.zeros(ss.k)))
        self.assertAlmostEqual(d.cdf(1.0), 1.0)
        self.assertEqual(d.quantile(0.5), 0.0)

    def test_multinomial_likelihood_finite_for_zero_probability_category(self):
        """The multinomial likelihood stays finite when the model gives an observed category zero probability."""
        val = pg.MultinomialLikelihood().compute(observed=[5, 3, 2], modelled=[1.0, 0.0, 1.0])
        self.assertTrue(np.isfinite(val))
        self.assertTrue(np.isfinite(pg.MultinomialLikelihood().compute(observed=[1, 2], modelled=[0.0, 0.0])))

    def test_touch_persists_moments_under_disabled_cache(self):
        """_touch() must persist the cached moments even with Settings.cache = False, so _drop() cannot corrupt the
        object (the _touch/_drop serialization contract is independent of the debug cache switch)."""
        from phasegen.settings import Settings

        emp = pg.Coalescent(n=3).tree_height.to_empirical(500)
        t = np.linspace(0, float(np.max(emp.samples)), 50)
        old = Settings.cache
        try:
            Settings.cache = False
            emp._touch(t)
            emp._drop()
            self.assertTrue(np.isfinite(emp.mean))
        finally:
            Settings.cache = old

    def test_sampler_honors_accumulation_window(self):
        """to_empirical accumulates the reward only over the coalescent's [start_time, end_time] window, not the
        full time to absorption."""
        windowed = pg.Coalescent(n=4, end_time=1.0)
        exact = float(windowed.total_branch_length.mean)
        sampled = float(windowed.total_branch_length.to_empirical(40000, seed=0).mean)
        self.assertAlmostEqual(exact, sampled, delta=0.1)  # windowed sampler matches windowed exact
        # and differs clearly from the full to-absorption value (the pre-fix behaviour)
        self.assertGreater(abs(float(pg.Coalescent(n=4).total_branch_length.mean) - exact), 0.5)

    def test_sampled_coalescent_restores_global_rng_state(self):
        """A seeded sampler must not perturb the caller's global numpy RNG state."""
        np.random.seed(1)
        expected = np.random.rand(3)

        np.random.seed(1)
        sc = pg.Coalescent(n=3).to_empirical(200, seed=42)
        _ = sc.tree_height
        got = np.random.rand(3)

        np.testing.assert_array_equal(expected, got)

    def test_take_n_and_takewhile_inclusive(self):
        """The ``take_n`` and ``takewhile_inclusive`` iterator helpers."""
        self.assertEqual(list(take_n(range(10), 3)), [0, 1, 2])
        # a shorter-than-n iterable is truncated rather than raising (PEP 479 turns a bare StopIteration
        # inside the generator into a RuntimeError)
        self.assertEqual(list(take_n(iter([1, 2]), 5)), [1, 2])
        # takewhile_inclusive keeps the first element that fails the predicate
        self.assertEqual(list(takewhile_inclusive(lambda x: x < 3, [1, 2, 3, 4])), [1, 2, 3])

    def test_demography_empty_dicts_are_treated_as_unspecified(self):
        """An explicit empty ``migration_rates`` or ``pop_sizes`` dict must be accepted, not raise IndexError."""
        d = pg.Demography(pop_sizes={'pop_0': 1.0}, migration_rates={})
        self.assertEqual(d.n_pops, 1)

        # an empty pop_sizes dict falls through to the unspecified case as well
        pg.Demography(pop_sizes={})

    def test_msprime_coalescent_num_replicates_below_n_threads(self):
        """``num_replicates`` below the default ``n_threads`` clamps threads instead of flooring per-thread reps to 0."""
        from phasegen.distributions.empirical import MsprimeCoalescent

        m = MsprimeCoalescent(n=2, num_replicates=50, parallelize=False)
        m._touch()

        self.assertEqual(m.n_threads, 50)
        self.assertGreater(m.n_total, 0)

    def test_population_split_demography(self):
        """A population split builds valid epochs and plots, exercising the split event and migration plotting."""
        d = pg.Demography(
            pop_sizes={'pop_0': 1.0, 'pop_1': 1.0},
            events=[pg.PopulationSplit(time=1.0, derived='pop_0', ancestral='pop_1')]
        )

        # building the epochs applies the split event
        epochs = list(d.get_epochs(np.array([0.0, 1.5])))
        self.assertEqual(len(epochs), 2)

        d.plot_migration(show=False)
        d.plot_pop_sizes(show=False)

    def test_folded_sfs_mean(self):
        """The folded SFS distribution produces a non-trivial mean."""
        folded = pg.Coalescent(n=4).fsfs.mean
        self.assertGreater(np.asarray(folded.data).sum(), 0)

    def test_single_locus_and_population_guards(self):
        """The single-locus SFS and multi-population statistics raise clear errors for invalid configurations."""
        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=2, loci=2, recombination_rate=1.0).sfs

        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=2, loci=2, recombination_rate=1.0).fsfs

        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=3).sfs2

        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=3).fst

    def test_moment_rejects_single_reward(self):
        """Passing a single Reward instead of a one-element sequence raises a clear ValueError naming the type,
        rather than a cryptic 'not subscriptable' from an internal rewards[0]."""
        coal = pg.Coalescent(n=4)

        with self.assertRaises(ValueError) as ctx:
            coal.moment(1, rewards=pg.TotalBranchLengthReward())

        self.assertIn('Reward', str(ctx.exception))

        # the normal one-element list case is unaffected
        self.assertTrue(np.isfinite(coal.moment(1, rewards=[pg.TotalBranchLengthReward()])))

    def test_tree_height_density_cdf_quantile(self):
        """Evaluate the tree-height CDF, density and quantile, exercising the numerical paths."""
        coal = pg.Coalescent(n=4)
        t = np.linspace(0.1, 3, 5)

        self.assertTrue(np.all(np.isfinite(coal.tree_height.cdf(t))))
        self.assertTrue(np.all(np.isfinite(coal.tree_height.pdf(t))))
        self.assertTrue(np.isfinite(coal.tree_height.quantile(0.5)))

    def test_empirical_sfs_mean_is_sfs_type(self):
        """The sampled and msprime SFS statistics must return the same SFS / TwoSFS containers as the exact
        Coalescent, so the empirical distributions are drop-in interchangeable with the exact one."""
        from phasegen.spectrum import SFS, TwoSFS

        exact = pg.Coalescent(n=5).sfs
        self.assertIsInstance(exact.mean, SFS)

        for emp in (pg.Coalescent(n=5).to_empirical(2000).sfs, pg.Coalescent(n=5).to_msprime(2000).sfs):
            self.assertIsInstance(emp.mean, SFS)
            self.assertIsInstance(emp.var, SFS)
            self.assertIsInstance(emp.m2, SFS)
            self.assertIsInstance(emp.cov, TwoSFS)
            self.assertIsInstance(emp.corr, TwoSFS)
            # the wrapped statistics still expose their array through np.asarray (used by the Tajima mixin)
            self.assertEqual(np.asarray(emp.mean).shape, (6,))
            self.assertEqual(np.asarray(emp.cov).shape, (6, 6))

    def test_sfs_cov_batched_is_cached(self):
        """SFSDistribution._cov_batched is memoized, so accessing sfs.corr (which reads both var and cov, each of
        which needs the shared two-point occupation) runs the expensive solve only once, without changing results."""
        import phasegen.distributions.spectra as sp

        # reference correlation before instrumenting the solve
        reference = np.asarray(pg.Coalescent(n=6).sfs.corr.data).copy()

        calls = {'n': 0}
        original = sp.SFSDistribution._two_point_occupation

        def counting(self, *args, **kwargs):
            calls['n'] += 1
            return original(self, *args, **kwargs)

        sp.SFSDistribution._two_point_occupation = counting
        try:
            corr = np.asarray(pg.Coalescent(n=6).sfs.corr.data)
        finally:
            # the method is inherited from the moment-evaluator mixin, so it is removed rather than assigned back:
            # assigning would leave a shadowing entry in ``SFSDistribution.__dict__`` that later patches of the base
            # class never reach
            del sp.SFSDistribution._two_point_occupation

        self.assertEqual(calls['n'], 1)  # a single shared solve, not one per var and cov
        np.testing.assert_allclose(corr, reference)
