"""
Test coalescents.
"""
from itertools import islice, permutations
from typing import cast
from testing import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

import phasegen as pg
from phasegen.distributions import MsprimeCoalescent
from phasegen.errors import ModelError


class CoalescentTestCase(TestCase):
    """
    Test coalescents.
    """

    def get_simple_coalescent(self):
        """
        Get simple coalescent.
        """
        return pg.Coalescent(
            n=pg.LineageConfig(n=2),
            model=pg.StandardCoalescent(),
            demography=pg.Demography([pg.PopSizeChange(pop='pop_0', time=0, size=1)])
        )

    def get_complex_demography(self):
        """
        Get complex demography.
        """
        return pg.Demography([
            pg.PopSizeChanges({'pop_0': {0: 1, 0.2: 1.2, 0.4: 1.4}, 'pop_1': {0: 1, 0.2: 1.2, 0.4: 1.4}}),
            pg.MigrationRateChanges({
                ('pop_0', 'pop_1'): {0: 0, 0.2: 0.2, 0.4: 0.4},
                ('pop_1', 'pop_0'): {0: 0, 0.3: 0.3, 0.6: 0.6},
                ('pop_1', 'pop_2'): {0: 0, 0.4: 0.4, 0.8: 0.8},
                ('pop_2', 'pop_1'): {0: 0, 0.5: 0.5, 1: 1}
            })
        ])

    def get_complex_coalescent(self):
        """
        Get complex coalescent.
        """
        return pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        )

    def test_simple_coalescent(self):
        """
        Test simple coalescent.
        """
        coal = self.get_simple_coalescent()

        m = coal.tree_height.mean
        coal.tree_height.pdf.plot()

        self.assertAlmostEqual(m, 1)

    def test_t_max_standard_coalescent(self):
        """
        Test time until almost sure absorption for standard coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.StandardCoalescent()
        )

        t = coal.tree_height.t_max

        self.assertEqual(t, 64)

    def test_t_max_complex_coalescent(self):
        """
        Test time until almost sure absorption for complex coalescent.
        """
        coal = self.get_complex_coalescent()

        t = coal.tree_height.t_max

        self.assertEqual(t, 128)

    def test_t_max_exponential_growth(self):
        """
        Test time until almost sure absorption for exponential growth.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.StandardCoalescent(),
            demography=pg.Demography([pg.ExponentialPopSizeChanges(
                initial_size={'pop_0': 1},
                growth_rate={'pop_0': 1},
                start_time={'pop_0': 0}
            )])
        )

        t = coal.tree_height.t_max

        self.assertTrue(3 < t < 5)

    def test_non_absorbing_demography_raises(self):
        """
        A demography with an isolated deme (a lineage that can never coalesce) has no almost-sure absorption
        time and must raise rather than silently returning the doubling-search ceiling.
        """
        coal = pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 1, 'pop_2': 1},
            demography=pg.Demography(
                pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1},
                # pop_1 is isolated: migration only ever connects pop_0 and pop_2
                migration_rates={('pop_0', 'pop_2'): 1}
            )
        )

        with self.assertRaisesRegex(ValueError, "does not absorb"):
            _ = coal.tree_height.mean

    def test_non_absorbing_demography_quantile_raises(self):
        """
        The quantile, cdf and pdf raise on a demography that never absorbs, like every other statistic.
        """
        coal = pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 1, 'pop_2': 1},
            demography=pg.Demography(
                pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1},
                migration_rates={('pop_0', 'pop_2'): 1}
            )
        )

        with self.assertRaisesRegex(ValueError, "does not absorb"):
            _ = coal.tree_height.quantile(0.99)

        with self.assertRaisesRegex(pg.ModelError, "does not absorb"):
            _ = coal.tree_height.cdf(1.0)

        with self.assertRaisesRegex(pg.ModelError, "does not absorb"):
            _ = coal.tree_height.pdf(1.0)

    def test_temporary_isolation_resolves_and_absorbs(self):
        """
        A demography whose demes are isolated in the first epoch but connected by migration in a later (unbounded)
        epoch does absorb: the non-absorption guard must not fire for either the mean or the quantile.
        """
        demog = pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1},
            migration_rates={
                ('pop_0', 'pop_1'): {0: 0.0, 1.0: 1.0},
                ('pop_1', 'pop_0'): {0: 0.0, 1.0: 1.0},
            }
        )
        coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=demog)

        self.assertTrue(np.isfinite(coal.tree_height.mean))
        self.assertTrue(np.isfinite(coal.tree_height.quantile(0.99)))

    def test_connected_migration_demography_absorbs(self):
        """
        The non-absorption guard must not fire for a fully connected migration demography: every lineage can
        reach a common ancestor, so the mean tree height is finite.
        """
        coal = pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 2},
            demography=pg.Demography(
                pop_sizes={'pop_0': 1, 'pop_1': 1},
                migration_rates={('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}
            )
        )

        self.assertTrue(np.isfinite(coal.tree_height.mean))

    def test_fully_isolated_demes_raise(self):
        """
        Multiple demes each holding lineages but with no migration at all (the default migration-free demography)
        can never reach a common ancestor across demes, so there is no almost-sure absorption time.
        """
        with self.assertRaisesRegex(ValueError, "does not absorb"):
            _ = pg.Coalescent(n={'pop_0': 3, 'pop_1': 3, 'pop_2': 3}).tree_height.mean

    def test_empty_extra_demes_absorb(self):
        """
        The guard must not fire when the extra demes are empty (zero lineages): all lineages live in one deme and
        coalesce there, so absorption is certain even without migration.
        """
        coal = pg.Coalescent(n={'pop_0': 3, 'pop_1': 0, 'pop_2': 0})
        self.assertTrue(np.isfinite(coal.tree_height.mean))

    def test_constructor_does_not_mutate_caller_demography(self):
        """
        The constructor must fill in missing populations on a *copy* of the caller-supplied Demography, never on the
        object itself. Regression: ``__init__`` called ``demography.add_event(PopSizeChanges(...))`` in place, so a
        Demography reused across Coalescents silently accumulated the extra demes; a later single-population
        coalescent then saw those demes as unspecified lineages and enlarged its lineage config (spurious deme with
        0 lineages, e.g. ``n_pops`` = 2 for what the user built as one population).
        """
        demo = pg.Demography(pop_sizes={'A': {0: 1}})

        # constructing with an extra population 'B' must not touch the original demography
        _ = pg.Coalescent(n={'A': 2, 'B': 3}, demography=demo)
        self.assertEqual(demo.pop_names, ['A'])  # pre-fix: ['A', 'B'] (add_event mutated demo in place)

        # reusing the same, still-pristine demo for a single-population coalescent must not inherit a spurious 'B'
        coal = pg.Coalescent(n={'A': 2}, demography=demo)
        self.assertEqual(coal.lineage_config.pop_names, ['A'])  # pre-fix: ['A', 'B'] (B accumulated on demo)
        self.assertEqual(coal.lineage_config.n_pops, 1)  # pre-fix: 2

    def test_constructor_does_not_mutate_caller_locus_config(self):
        """
        The same deepcopy contract holds for the ``LocusConfig``: overriding ``recombination_rate`` must not mutate
        the caller-supplied object. Regression: the constructor set ``self.locus_config.recombination_rate`` on the
        passed-in config in place.
        """
        loci = pg.LocusConfig(n=2, recombination_rate=0.0)

        coal = pg.Coalescent(n=2, loci=loci, recombination_rate=5.0)

        self.assertEqual(loci.recombination_rate, 0.0)  # pre-fix: 5.0 (mutated in place)
        self.assertEqual(coal.locus_config.recombination_rate, 5.0)

    def test_sfs2_mean_is_cached(self):
        """
        The two-locus SFS distribution and its mean are memoized on the Coalescent when caching is enabled
        (regression: ``sfs2.mean`` must not recompute on every access), and recomputed when caching is disabled.
        """
        coal = pg.Coalescent(n=4, loci=2)

        # enabled (default): re-access returns the very same cached objects, i.e. no recomputation
        self.assertIs(coal.sfs2, coal.sfs2)
        self.assertIs(coal.sfs2.mean, coal.sfs2.mean)

        # disabled: a fresh, un-warmed Coalescent recomputes on each access, but yields an identical result
        fresh = pg.Coalescent(n=4, loci=2)
        try:
            pg.Settings.cache = False
            a = fresh.sfs2.mean
            b = fresh.sfs2.mean
        finally:
            pg.Settings.cache = True

        self.assertIsNot(a, b)
        np.testing.assert_allclose(a.data, b.data)

    def test_complex_coalescent(self):
        """
        Test complex coalescent.
        """
        coal = self.get_complex_coalescent()

        m = coal.tree_height.mean
        coal.tree_height.pdf.plot()

        self.assertAlmostEqual(m, 5.919799579948861, delta=1e-6 * 5.919799579948861)

    @pytest.mark.slow
    def test_demes_complex_coalescent(self):
        """
        Validate first moments for deme-wise complex coalescent.
        """
        coals = [
            pg.Coalescent(
                n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
                model=pg.BetaCoalescent(alpha=1.7),
                demography=self.get_complex_demography()
            ),
            MsprimeCoalescent(
                n_threads=1,
                num_replicates=1000,
                n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
                model=pg.BetaCoalescent(alpha=1.7),
                demography=self.get_complex_demography(),
                record_migration=True
            )
        ]

        for coal in coals:
            # make sure the deme-wise tree heights add upp
            np.testing.assert_array_almost_equal(
                coal.tree_height.mean,
                np.sum([coal.tree_height.demes[p].mean for p in coal.demography.pop_names], axis=0),
                decimal=8
            )

            # make sure the deme-wise total branch lengths add upp
            np.testing.assert_array_almost_equal(
                coal.total_branch_length.mean,
                np.sum([coal.total_branch_length.demes[p].mean for p in coal.demography.pop_names], axis=0),
                decimal=8
            )

            # make sure the deme-wise SFS add upp
            np.testing.assert_array_almost_equal(
                coal.sfs.mean.data,
                np.sum([coal.sfs.demes[p].mean.data for p in coal.demography.pop_names], axis=0),
                decimal=8
            )

    @pytest.mark.skip(reason="Too slow")
    def test_msprime_complex_coalescent(self):
        """
        Test msprime complex coalescent.
        """
        coal = MsprimeCoalescent(
            n_threads=100,
            parallelize=True,
            num_replicates=1000000,
            n=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            # model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography(),
            record_migration=True
        )

        coal._touch()

        pass

    def test_two_loci_one_deme_n_2_tree_height(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        )

        self.assertAlmostEqual(1, coal.tree_height.loci[0].mean)
        self.assertAlmostEqual(1, coal.tree_height.loci[0].var)

        self.assertAlmostEqual(2, coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),)))

        pass

    def test_two_loci_one_deme_n_2(self):
        """
        Two loci, one deme, two lineages: the per-locus marginals match the single-locus coalescent and the tree
        height summed over both loci is twice the single-locus mean.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            loci=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        marginal = pg.Coalescent(n=pg.LineageConfig(2))

        self.assertAlmostEqual(marginal.tree_height.mean * 2,
                               coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),)))

        self.assertAlmostEqual(marginal.tree_height.mean, coal.tree_height.loci[0].mean)
        self.assertAlmostEqual(marginal.tree_height.var, coal.tree_height.loci[0].var)

    def test_n_unlinked_leaves_the_marginal_locus_unchanged(self):
        """
        ``LocusConfig.n_unlinked`` sets how many lineages start unlinked between the loci. It redistributes the
        initial state across the locus dimension and so must leave each locus's own distribution equal to the
        single-locus coalescent, whatever its value.
        """
        dem = pg.Demography(
            pop_sizes=dict(pop_0={0: 1}, pop_1={0: 1}),
            migration_rates={('pop_0', 'pop_1'): {0: 1}, ('pop_1', 'pop_0'): {0: 1}}
        )

        cases = [
            ('one deme, n=2', dict(n=pg.LineageConfig(2)), 2),
            ('one deme, n=3', dict(n=pg.LineageConfig(3)), 3),
            ('two demes, n=2', dict(n=pg.LineageConfig(dict(pop_0=1, pop_1=1)), demography=dem), 2),
        ]

        for label, kwargs, n in cases:
            marginal = pg.Coalescent(**kwargs)

            for n_unlinked in range(n + 1):
                with self.subTest(case=label, n_unlinked=n_unlinked):
                    coal = pg.Coalescent(
                        loci=pg.LocusConfig(n=2, recombination_rate=0, n_unlinked=n_unlinked),
                        **kwargs
                    )

                    self.assertAlmostEqual(marginal.tree_height.mean, coal.tree_height.loci[0].mean)
                    self.assertAlmostEqual(marginal.tree_height.var, coal.tree_height.loci[0].var)

                    self.assertAlmostEqual(marginal.tree_height.mean * 2,
                                           coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),)))

    def test_n_unlinked_decreases_the_two_locus_cross_moment(self):
        """
        Starting more lineages unlinked weakens the dependence between the loci, so the second cross-moment of the
        tree height over both loci decreases strictly in ``n_unlinked``. The first moment cannot see this, being
        additive over loci, so only the second moment pins the initial linkage down.
        """
        rewards = (pg.TotalTreeHeightReward(),) * 2

        for n in [2, 3]:
            with self.subTest(n=n):
                moments = [
                    pg.Coalescent(
                        n=pg.LineageConfig(n),
                        loci=pg.LocusConfig(n=2, recombination_rate=0, n_unlinked=n_unlinked)
                    ).tree_height.moment(2, rewards)
                    for n_unlinked in range(n + 1)
                ]

                self.assertTrue(all(a > b for a, b in zip(moments, moments[1:])), moments)


    def test_two_loci_one_deme_n_4(self):
        """
        Test two loci.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(4),
            loci=pg.LocusConfig(n=2, recombination_rate=1),
        )

        marginal = pg.Coalescent(n=pg.LineageConfig(4))

        # assert total branch length to be twice as long as marginal
        self.assertAlmostEqual(marginal.total_branch_length.mean * 2, coal.total_branch_length.mean)

        self.assertAlmostEqual(marginal.total_branch_length.mean, coal.total_branch_length.loci[0].mean)
        self.assertAlmostEqual(marginal.total_branch_length.var, coal.total_branch_length.loci[0].var)

        # assert total tree height to be twice as long as marginal
        self.assertAlmostEqual(marginal.tree_height.mean * 2, coal.tree_height.moment(1, (pg.TotalTreeHeightReward(),)))

        # assert marginal locus moments
        self.assertAlmostEqual(marginal.tree_height.mean, coal.tree_height.loci[0].mean)
        self.assertAlmostEqual(marginal.tree_height.var, coal.tree_height.loci[0].var)

        pass

    def test_two_loci_two_demes(self):
        """
        Two loci in two demes without migration never absorb, which every statistic reports as the one-locus case
        does. The states stuck in separate demes form a recurrent class under recombination, so the absorption-time
        search used to overflow to NaN and skip the absorption check, returning a finite branch-length mean.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig([2, 2]),
            loci=pg.LocusConfig(n=2, recombination_rate=1.11),
        )

        for dist in [coal.tree_height, coal.total_branch_length]:
            with pytest.raises(ModelError, match="does not absorb"):
                _ = dist.mean

    def test_beta_4_n(self):
        """
        Test beta coalescent.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(4),
            model=pg.BetaCoalescent(alpha=1.7)
        )

        m = coal.tree_height.mean

        pass

    def test_2_loci_sfs_raises(self):
        """
        Test that the single-locus SFS raises a clear error when two loci are configured (use ``sfs2`` instead).
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(4),
            loci=pg.LocusConfig(2)
        )

        with self.assertRaises(ValueError):
            _ = coal.sfs.mean

    def test_beta_coalescent_n_2_alpha_close_to_2_lineage_counting_state_space(self):
        """
        Test beta coalescent with lineage-counting state space for n = 2.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig(2),
            model=pg.BetaCoalescent(alpha=1.999)
        )

        # coalescent time coincides with timescale in this case
        self.assertAlmostEqual(coal.tree_height.mean, coal.model._get_timescale(1), places=15)

        pass

    def test_serialize_coalescent(self):
        """
        Test serialization of coalescent.
        """
        coal = self.get_complex_coalescent()

        coal.to_file('scratch/test_serialize_simple_coalescent.json')

        coal2 = pg.Coalescent.from_file('scratch/test_serialize_simple_coalescent.json')

        self.assertEqual(coal.tree_height.mean, coal2.tree_height.mean)

    def test_serialize_assert_getstate_method_called(self):
        """
        Test serialization of coalescent.
        """
        coal = self.get_complex_coalescent()

        with patch.object(coal, '__getstate__', return_value=None) as mock_getstate:
            try:
                coal.to_file('scratch/test_serialize_simple_coalescent.json')
            except Exception:
                pass

            mock_getstate.assert_called_once()

    def test_coalescent_negative_end_time_raises_value_error(self):
        """
        Test negative end time raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                end_time=-1
            ).tree_height

        self.assertTrue('End time' in str(context.exception))

    def test_coalescent_negative_start_time_raises_value_error(self):
        """
        Test negative start time raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                start_time=-1
            ).tree_height

        self.assertTrue('Start time' in str(context.exception))

    def test_end_time_before_start_time_raises_value_error(self):
        """
        Test end time before start time raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                start_time=1,
                end_time=0
            ).tree_height

        self.assertTrue('End time' in str(context.exception))

    def test_start_greater_than_t_abs_raises_value_error(self):
        """
        Test start time greater than t_abs raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = pg.Coalescent(
                n=2,
                start_time=100000
            ).tree_height.mean

        self.assertTrue('start time' in str(context.exception))

    def test_start_time_equal_end_time_zero_moments(self):
        """
        Test start time equal to end time gives zero moments.
        """
        coal = pg.Coalescent(
            n=2,
            start_time=1,
            end_time=1
        )

        self.assertEqual(coal.tree_height.mean, 0)
        self.assertEqual(coal.tree_height.var, 0)

    def test_simple_coalescent_start_time(self):
        """
        Test simple coalescent start time moment.
        """
        coal = pg.Coalescent(
            n=2,
            start_time=1
        )

        _ = coal.tree_height.mean
        _ = coal.tree_height.var
        _ = coal.total_branch_length.mean
        _ = coal.total_branch_length.var
        _ = coal.sfs.mean
        _ = coal.sfs.corr

        # the distribution functions describe the unwindowed law, so they refuse a windowed coalescent
        with self.assertRaises(NotImplementedError):
            coal.tree_height.pdf(1)
        with self.assertRaises(NotImplementedError):
            coal.tree_height.cdf(1)

    def test_batched_spectrum_mean_honours_start_time(self):
        """
        The batched SFS / jSFS mean must honour a configured ``start_time``. Regression test: the batched
        occupation-time contraction integrates from 0, so a non-zero start time was silently ignored and the
        ``start_time = 0`` spectrum returned. The n=2 single-population smoke test above never caught it because
        flattening bypasses the batched path there; this uses n >= 4 with a multi-epoch demography so the batched
        path is actually taken.
        """
        dem = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.3}})

        def sfs_serial(sfs):
            # per-bin serial mean: same closed form, but bin-by-bin instead of the batched contraction
            n = sfs.lineage_config.n
            vals = [sfs._moment(1, i, (sfs.reward,), None, None, True, True) for i in sfs._get_indices()]
            return np.array([0] + list(vals) + [0] * (n - len(vals)))

        for start_time in (0.2, 0.8):  # mid-epoch and past the 0.5 boundary
            sfs = pg.Coalescent(n=5, demography=dem, start_time=start_time).sfs
            batched = np.asarray(sfs.mean.data, dtype=float)

            # the batched result must match the per-bin serial path exactly (same numerics, different aggregation)
            np.testing.assert_allclose(batched, sfs_serial(sfs), atol=1e-9)

            # and it must actually differ from the start_time = 0 spectrum, else the assertion above is vacuous
            full = np.asarray(pg.Coalescent(n=5, demography=dem, start_time=0).sfs.mean.data, dtype=float)
            self.assertGreater(np.abs(batched - full).max(), 1e-3)

        # additivity identity for both spectra: occupation(start, absorption) = occupation(0, absorption) -
        # occupation(0, start), the head accumulated with a finite end time through the independent serial path
        start_time = 0.3
        for n in (5,):
            full = np.asarray(pg.Coalescent(n=n, demography=dem, start_time=0).sfs.mean.data, dtype=float)
            head = np.asarray(pg.Coalescent(n=n, demography=dem, start_time=0, end_time=start_time).sfs.mean.data,
                              dtype=float)
            win = np.asarray(pg.Coalescent(n=n, demography=dem, start_time=start_time).sfs.mean.data, dtype=float)
            np.testing.assert_allclose(win, full - head, atol=1e-9)

        dem2 = pg.Demography(
            pop_sizes={'pop_0': {0: 1.0, 0.5: 0.4}, 'pop_1': {0: 1.0}},
            migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
        )
        mk = lambda **kw: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem2, **kw)
        full = np.asarray(mk(start_time=0).jsfs.mean.data, dtype=float)
        head = np.asarray(mk(start_time=0, end_time=start_time).jsfs.mean.data, dtype=float)
        win = np.asarray(mk(start_time=start_time).jsfs.mean.data, dtype=float)
        np.testing.assert_allclose(win, full - head, atol=1e-9)
        self.assertGreater(np.abs(win - full).max(), 1e-3)

        # the batched covariance has no windowed form; it must fall back rather than silently return the start=0 value
        _ = mk(start_time=start_time).jsfs.cov  # must not raise

    def test_marginal_view_curves_honour_own_reward(self):
        """
        A marginal SFS view (``sfs.demes[pop]``) must build its per-bin cdf/pdf/quantile from its own reward, not the
        bin reward alone. Regression test: the per-bin distribution dropped ``self.reward``, so a deme view's curves
        described the whole tree while its moments were deme-restricted. Here the mean recovered from the cdf as
        ``E[X] = int (1 - F) dx`` must match the deme-restricted moment, and must differ from the full-tree spectrum.
        """
        dem = pg.Demography(
            pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.0}},
            migration_rates={('pop_0', 'pop_1'): 1.0, ('pop_1', 'pop_0'): 1.0}
        )
        coal = pg.Coalescent(n={'pop_0': 3, 'pop_1': 3}, demography=dem)
        sfs = coal.sfs
        deme = sfs.demes['pop_0']

        t = np.linspace(0, float(coal.tree_height.quantile(0.9999)), 2000)
        mean_from_cdf = np.trapezoid(1 - np.asarray(deme.cdf(t)), t, axis=0)
        deme_mean = np.asarray(deme.mean.data, dtype=float)
        full_mean = np.asarray(sfs.mean.data, dtype=float)

        interior = slice(1, sfs.lineage_config.n)
        # the deme cdf and the deme moments must describe the same law
        np.testing.assert_allclose(mean_from_cdf[interior], deme_mean[interior], atol=5e-3)
        # and the deme restriction must be real: the deme mean is not the full-tree mean
        self.assertGreater(np.abs(deme_mean[interior] - full_mean[interior]).max(), 1e-2)

    def test_absorption_time_warns_when_iterations_exhausted(self):
        """
        When the doubling search for the almost-sure absorption time runs out of iterations without reaching
        ``p_absorption``, it returns the (possibly truncated) doubling ceiling and must warn. Regression test: the
        guard was ``i - 1 == max_iter``, unreachable because the loop never lets ``i`` exceed ``max_iter``, so the
        truncated time was returned silently.
        """
        tree_height = pg.Coalescent(n=4).tree_height
        tree_height.max_iter = 1

        with self.assertLogs('phasegen', level='WARNING') as cm:
            tree_height._get_absorption_time()

        self.assertTrue(any('maximum number of iterations' in m for m in cm.output))

    def test_sfs_accumulate_forwards_center(self):
        """
        ``SFSDistribution.accumulate`` must forward ``center`` to the per-bin fallback. Regression test: it dropped
        ``center`` / ``permute``, so ``center=False`` silently returned centered moments. With ``center=True`` the
        order-2 accumulation to absorption is the variance; with ``center=False`` it is ``E[X^2] = var + mean^2``.
        """
        sfs = pg.Coalescent(n=4).sfs
        t_max = pg.Coalescent(n=4).tree_height.t_max

        centered = sfs.accumulate(2, [t_max], center=True)[0]
        uncentered = sfs.accumulate(2, [t_max], center=False)[0]
        var = np.asarray(sfs.var.data)
        mean = np.asarray(sfs.mean.data)

        interior = slice(1, 4)
        self.assertGreater(np.abs(centered[interior] - uncentered[interior]).max(), 1e-6)
        np.testing.assert_allclose(centered[interior], var[interior], atol=1e-9)
        np.testing.assert_allclose(uncentered[interior], (var + mean ** 2)[interior], atol=1e-9)

    def test_jsfs_moment_explicit_zero_start_time(self):
        """``jsfs.moment(k=1, start_time=0)`` must equal the mean, not raise. Regression: an explicit ``start_time=0``
        missed the batched path and fell through to ``accumulate([inf])``, which exponentiated an infinite time."""
        dem = pg.Demography(
            pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.0}},
            migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5}
        )
        jsfs = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem).jsfs
        np.testing.assert_allclose(
            np.asarray(jsfs.moment(k=1, start_time=0).data), np.asarray(jsfs.mean.data), atol=1e-12
        )

    def test_mutation_configs_sparse_rate_matrix(self):
        """
        The multi-epoch mutation-configuration path must handle a sparsely stored rate matrix. Regression:
        ``np.asarray(state_space.S)`` returned a 0-d object array for a sparse ``S``, crashing above
        ``dense_rate_matrix_max_states``. The result must match the dense path exactly.
        """
        from phasegen.settings import Settings

        dem = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.3}})
        config, theta = (2, 1, 0), 1.5
        original = Settings.dense_rate_matrix_max_states
        try:
            Settings.dense_rate_matrix_max_states = 10 ** 6
            sfs = pg.Coalescent(n=4, demography=dem).sfs
            dense = sfs._get_mutation_config_inhomogeneous(sfs.mutation_layout().config(config), theta)
            Settings.dense_rate_matrix_max_states = 0
            sfs = pg.Coalescent(n=4, demography=dem).sfs
            sparse = sfs._get_mutation_config_inhomogeneous(sfs.mutation_layout().config(config), theta)
        finally:
            Settings.dense_rate_matrix_max_states = original

        self.assertAlmostEqual(dense, sparse, places=12)

    def test_lst_cdf_pdf_below_support(self):
        """CDF and density of an accumulated-reward (LST) distribution are 0 below the support. Regression:
        ``np.interp`` clamped a negative argument to the first grid node, returning the atom for the CDF and ``f(0+)``
        for the pdf. The 5-ton branch length at n=6 carries a large atom P(R=0), so the clamped CDF would be that
        atom rather than 0."""
        from phasegen.rewards import CombinedReward

        sfs = pg.Coalescent(n=6).sfs
        d = sfs.distribution(reward=CombinedReward([sfs.reward, sfs._get_sfs_reward(5)]))

        self.assertGreater(float(d.cdf(0.0)), 0.1)  # a real atom at 0, so the clamp bug would surface
        self.assertEqual(float(d.cdf(-1.0)), 0.0)
        self.assertEqual(float(d.pdf(-1.0)), 0.0)
        np.testing.assert_array_equal(np.asarray(d.cdf(np.array([-2.0, -1.0]))), [0.0, 0.0])

    def test_lst_density_degenerate_single_node_grid(self):
        """The density must not raise on a degenerate single-node grid (a near-total atom whose mass sits above the
        tail cut leaves one node); ``np.gradient`` needs two. Regression: it raised ``IndexError``."""
        pdf = pg.Coalescent(n=4).total_branch_length.distribution(reward=pg.TotalBranchLengthReward()).pdf

        out = pdf._interp_pdf(np.array([0.0, 1.0, 2.0]), np.array([0.0]), np.array([0.5]))
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_sfs_var_equals_cov_diagonal(self):
        """``sfs.var`` reuses the batched covariance's diagonal (one shared solve) and must equal both the per-bin
        central moment and the diagonal of ``sfs.cov``, batched and in the multi-epoch fallback."""
        for coal in (pg.Coalescent(n=6),
                     pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.3}}))):
            sfs = coal.sfs
            var = np.asarray(sfs.var.data, dtype=float)
            np.testing.assert_allclose(var, np.asarray(sfs.moment(k=2, center=True).data, dtype=float), atol=1e-9)
            np.testing.assert_allclose(var, np.diag(np.asarray(sfs.cov.data, dtype=float)), atol=1e-12)

    def test_joint_density_plot_guards_diagonal(self):
        """Plotting a joint density that is singular on the diagonal (``R_a = R_b`` almost surely) must raise, like
        calling it does. Regression: the guard sat only in ``__call__``; ``plot`` / ``plot_surface`` reached
        ``_grid_values`` directly and drew a surface for a law with no 2D density. A non-diagonal pair still plots."""
        c = pg.Coalescent(n=5)

        diagonal = c.total_branch_length.joint(
            pg.TotalBranchLengthReward(), pg.TotalBranchLengthReward()
        )
        self.assertEqual(diagonal._ratio, 1.0)
        with self.assertRaises(NotImplementedError):
            diagonal.pdf.plot(show=False)
        with self.assertRaises(NotImplementedError):
            diagonal.pdf(1.0, 2.0)

        off_diagonal = c.sfs.joint(1, 2)
        self.assertIsNone(off_diagonal._ratio)
        off_diagonal.pdf.plot(show=False)  # must not raise

    def test_plot_accumulation_center_permute(self):
        """
        Test accumulation plot for center and permute.
        """
        coal = pg.Coalescent(
            n=3
        )

        values = np.linspace(0, coal.tree_height.quantile(0.99), 10)
        rewards = (pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2))

        fig, ax = plt.subplots(1)

        for i, kwargs in enumerate([
            dict(center=True, permute=True),
            dict(center=False, permute=True),
            dict(center=True, permute=False),
            dict(center=False, permute=False)
        ]):
            coal.plot_accumulation(
                k=2,
                end_times=values,
                rewards=rewards,
                ax=ax,
                show=False,
                label=str(kwargs),
                **kwargs
            )

        plt.show()

    @pytest.mark.slow
    def test_plot_accumulation(self):
        """
        Test accumulation plot.
        """
        coal = self.get_complex_coalescent()

        values = np.linspace(0, coal.tree_height.quantile(0.99), 10)

        coal.tree_height.plot_accumulation(1, values)
        coal.tree_height.plot_accumulation(2, values)
        coal.sfs.plot_accumulation(1, values)
        coal.sfs.plot_accumulation(2, values)

    def test_large_accumulation_equal_moments(self):
        """
        Test large accumulation equals moments.
        """
        # this validates the matrix-exponential invariant ``accumulate(t_max) == moment``; the closed-form path
        # evaluates the moment to infinity directly, so it would not be bit-identical to accumulating to t_max
        pg.Settings.closed_form_last_epoch = False

        coal = self.get_complex_coalescent()

        self.assertEqual(
            coal.tree_height.accumulate(1, [coal.tree_height.t_max])[0],
            coal.tree_height.moment(1)
        )

        self.assertEqual(
            coal.tree_height.accumulate(2, [coal.tree_height.t_max])[0],
            coal.tree_height.moment(2)
        )

        # the SFS mean accumulation is batched (shared occupation grid across bins), an independent matrix-
        # exponential route from the per-bin moment, so they agree to floating point rather than bit-for-bit
        np.testing.assert_allclose(
            coal.sfs.accumulate(1, [coal.tree_height.t_max])[0],
            coal.sfs.moment(1).data,
            rtol=1e-12, atol=1e-12
        )

    def test_precision_regularization_large_N(self):
        """
        Make sure regularization works for large rates.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1e40}})
        )

        lamb = coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S)

        self.assertTrue(1e39 <= lamb <= 1e41)

        self.assertAlmostEqual(coal.tree_height.mean, 1.5e40, delta=1e27)

    def test_precision_regularization_small_N(self):
        """
        Make sure regularization works for small rates.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1e-40}})
        )

        lamb = coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S)

        self.assertTrue(1e-41 <= lamb <= 1e-39)

        self.assertAlmostEqual(coal.tree_height.mean, 1.5e-40, delta=1e-26)

    def test_warning_disconnected_demes(self):
        """
        Make sure disconnected demes raise warning.
        """
        with self.assertLogs(level='WARNING', logger=pg.logger) as cm:
            pg.Demography(
                pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 1}}
            )

        self.assertTrue('zero migration rates' in cm.output[0])

    def test_value_error_extreme_imprecision(self):
        """
        Make sure extreme imprecision raises ValueError.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1e-40}, 'pop_1': {0: 1e40}},
                migration_rates={('pop_0', 'pop_1'): {0: 1}}
            )
        )

        with self.assertRaises(ValueError):
            _ = coal.tree_height.mean

    def test_value_error_large_imprecision(self):
        """
        Make sure no warning is raised for large imprecision.
        """
        coal = pg.Coalescent(
            n=4,
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1e10}, 'pop_1': {0: 1e4}},
                migration_rates={('pop_0', 'pop_1'): {0: 1e4}}
            ),
        )

        coal.tree_height.cdf.plot()
        coal.tree_height.plot_accumulation(1)

        self.assertNoLogs(level='WARNING', logger=coal._logger)

    def test_low_recombination_rate(self):
        """
        Test low recombination rate.
        """
        coal = pg.Coalescent(
            n=4,
            loci=pg.LocusConfig(n=2, recombination_rate=1e11)
        )

        with self.assertLogs(level='WARNING', logger=coal.tree_height._logger) as cm:
            _ = coal.tree_height.mean

        self.assertIn("numerical instability", cm.output[0])

    def test_symmetric_deme_covariance(self):
        """
        Make sure deme covariance is symmetric.
        """
        coal = self.get_complex_coalescent()

        np.testing.assert_array_equal(coal.tree_height.demes.cov, coal.tree_height.demes.cov.T)
        np.testing.assert_array_equal(coal.tree_height.demes.corr, coal.tree_height.demes.corr.T)

        # check that diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(coal.tree_height.demes.corr), 1)

    def test_variance(self):
        """
        Test kurtosis.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        var = coal.moment(2, rewards, center=True)

        self.assertEqual(var, coal.moment(2, rewards, center=False) - coal.moment(1, rewards[:1], center=False) ** 2)

    def test_kurtosis(self):
        """
        Test kurtosis.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        kurtosis = coal.moment(3, rewards, center=True)

        m1 = coal.moment(1, rewards[:1], center=False)
        m2 = coal.moment(2, rewards[:2], center=False)
        m3 = coal.moment(3, rewards, center=False)

        self.assertAlmostEqual(kurtosis, m3 - 3 * m2 * m1 + 2 * m1 ** 3)

    def test_skewness(self):
        """
        Test skewness.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        skewness = coal.moment(4, rewards, center=True)

        m4 = coal.moment(4, rewards, center=False)
        m1 = coal.moment(1, rewards[:1], center=False)
        m3m1 = coal.moment(3, rewards[:3], center=False) * m1
        m2m2 = coal.moment(2, rewards[:2], center=False) * m1 ** 2

        self.assertAlmostEqual(skewness, m4 - 4 * m3m1 + 6 * m2m2 - 3 * m1 ** 4)

    def test_central_m5(self):
        """
        Test central 5th moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward(),
            pg.TreeHeightReward()
        )

        moment = coal.moment(5, rewards, center=True)

        m5 = coal.moment(5, rewards, center=False)
        m1 = coal.moment(1, rewards[:1], center=False)
        m4m1 = coal.moment(4, rewards[:4], center=False) * m1
        m3m2 = coal.moment(3, rewards[:3], center=False) * m1 ** 2
        m2m3 = coal.moment(2, rewards[:2], center=False) * m1 ** 3

        self.assertAlmostEqual(moment, m5 - 5 * m4m1 + 10 * m3m2 - 10 * m2m3 + 4 * m1 ** 5)

    def test_central_2nd_order_cross_moment(self):
        """
        Test 2nd order central cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(3)
        )

        moment = coal.moment(2, rewards, center=True)
        xy = coal.moment(2, (pg.UnfoldedSFSReward(2), pg.UnfoldedSFSReward(3)), center=False, permute=False)
        yx = coal.moment(2, (pg.UnfoldedSFSReward(3), pg.UnfoldedSFSReward(2)), center=False, permute=False)
        x = coal.moment(1, (pg.UnfoldedSFSReward(2),), center=False, permute=False)
        y = coal.moment(1, (pg.UnfoldedSFSReward(3),), center=False, permute=False)

        self.assertAlmostEqual(moment, (xy + yx) / 2 - x * y)

    @pytest.mark.slow
    def test_central_3rd_order_cross_moment(self):
        """
        Test 3rd order cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(3),
            pg.UnfoldedSFSReward(4)
        )

        moment = coal.moment(3, rewards, center=True)
        xyz = coal.moment(3, rewards, center=False)
        xy = coal.moment(2, tuple(np.array(rewards)[[0, 1]]), center=False)
        xz = coal.moment(2, tuple(np.array(rewards)[[0, 2]]), center=False)
        yz = coal.moment(2, tuple(np.array(rewards)[[1, 2]]), center=False)
        xy_centered = coal.moment(2, tuple(np.array(rewards)[[0, 1]]))
        xz_centered = coal.moment(2, tuple(np.array(rewards)[[0, 2]]))
        yz_centered = coal.moment(2, tuple(np.array(rewards)[[1, 2]]))
        x = coal.moment(1, (pg.UnfoldedSFSReward(2),))
        y = coal.moment(1, (pg.UnfoldedSFSReward(3),))
        z = coal.moment(1, (pg.UnfoldedSFSReward(4),))

        self.assertAlmostEqual(moment, xyz - xy * z - xz * y - yz * x + 2 * x * y * z)
        self.assertAlmostEqual(moment, xyz - xy_centered * z - xz_centered * y - yz_centered * x - x * y * z)

    @pytest.mark.slow
    def test_3rd_order_uncentered_cross_moment(self):
        """
        Test 3rd order uncentered cross moment.
        """
        coal = self.get_complex_coalescent()

        rewards = (
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(2),
            pg.UnfoldedSFSReward(4)
        )

        moment = coal.moment(3, rewards, center=False)
        ordered = [coal.moment(3, order, center=False, permute=False) for order in permutations(rewards)]

        self.assertAlmostEqual(moment, np.mean(ordered))

    @pytest.mark.slow
    def test_uncentered_cross_moments_msprime(self):
        """
        Test higher-order uncentered cross-moments against Msprime coalescent. Each tolerance is at least four
        standard errors of the simulated moment at 1e5 replicates, whose relative standard error is about 0.024 for
        the index set (2, 3, 4) and at most about 0.01 for the others.
        """
        coal = self.get_complex_coalescent()
        ms = coal.to_msprime(num_replicates=100000, seed=42)

        # test uncentered moments
        for indices, tol in [([2, 3, 4], 0.1), ([1, 1, 4], 0.07), ([1, 1, 1], 0.07), ([4, 2, 1], 0.07)]:
            m_ms = np.mean(ms.sfs.samples[:, indices].prod(axis=1))
            m_ph = coal.moment(3, tuple(pg.UnfoldedSFSReward(l) for l in indices), center=False)

            self.assertLess(2 * np.abs((m_ms - m_ph) / (m_ms + m_ph)), tol)

    @pytest.mark.slow
    def test_mutation_configuration_probability_mass_close_to_one(self):
        """
        Test mutation configuration probability mass is close to one.
        """
        coal = pg.Coalescent(n=5)

        ms = coal.to_msprime(
            num_replicates=1000,
            seed=42,
            n_threads=1,
            parallelize=False,
            mutation_rate=0.01,
            simulate_mutations=True,
        )

        self.assertAlmostEqual(1, sum(map(lambda x: x[1], islice(ms.sfs.get_mutation_configs(), 100))))
        self.assertAlmostEqual(1, sum(map(lambda x: x[1],
                                          islice(coal.sfs.get_mutation_configs(theta=ms.mutation_rate), 100))))

    def test_pdf_large_N(self):
        """
        Test plotting pdf for different population sizes.
        """
        Ns = np.array([1e-30, 1e-20, 1e-10, 1, 1e10, 1e20, 1e30])
        _, axs = plt.subplots(len(Ns), 1, figsize=(5, 10))
        data = []

        for i, N in enumerate(Ns):
            coal = pg.Coalescent(
                n=4,
                demography=pg.Demography(
                    pop_sizes=cast(float, N)
                )
            )

            t = np.linspace(0, coal.tree_height.quantile(0.99), 200)
            data += [coal.tree_height.pdf(t=t)]
            coal.tree_height.pdf.plot(ax=axs[i], label=f'N={N}', show=False, t=t)

        plt.show()

        data = np.array(data)
        self.assertTrue((np.var(data.T * Ns[None, :], axis=1) < 1e-10).all())

    def test_accumulate_same_as_moment(self):
        """
        Test accumulation is the same as moment with adjusted end time.
        """
        coal = pg.Coalescent(n=4)

        rewards = [
            (pg.TreeHeightReward(),),
            (pg.TotalTreeHeightReward(),),
            (pg.TreeHeightReward(), pg.TreeHeightReward()),
            (pg.UnfoldedSFSReward(1), pg.UnfoldedSFSReward(2)),
        ]

        times = np.linspace(0, coal.tree_height.quantile(0.99), 10)

        for reward in rewards:
            moments = [coal.moment(k=len(reward), rewards=reward, end_time=t) for t in times]
            accumulation = coal.accumulate(k=len(reward), rewards=reward, end_times=times)

            np.testing.assert_array_almost_equal(moments, accumulation)

    def test_accumulate_same_as_moment_sfs(self):
        """
        Test accumulation is the same as moment with adjusted end time for SFS-based moments.
        """
        coal = pg.Coalescent(n=4)

        times = np.linspace(0, coal.tree_height.quantile(0.99), 10)

        for k in [1, 2]:
            moments = np.array([coal.sfs.moment(k=k, end_time=t).data for t in times])
            accumulation = coal.sfs.accumulate(k=k, end_times=times)

            np.testing.assert_array_almost_equal(moments, accumulation)

    def test_get_cov_sfs(self):
        """
        Test get_cov method for SFS.
        """
        n = 4
        coal = pg.Coalescent(n=n)

        cov = coal.sfs.cov.data
        cov2 = np.array([[coal.sfs.get_cov(i, j) for i in range(n + 1)] for j in range(n + 1)])

        np.testing.assert_array_almost_equal(cov, cov2)

    def test_get_corr_sfs(self):
        """
        Test get_corr method for SFS.
        """
        n = 4
        coal = pg.Coalescent(n=n)

        corr = coal.sfs.corr.data
        corr2 = np.array([[coal.sfs.get_corr(i, j) for i in range(n + 1)] for j in range(n + 1)])

        np.testing.assert_array_almost_equal(corr, corr2)

    def test_disable_regularization(self):
        """
        Test disabling regularization.
        """
        pg.Settings.regularize = False

        coal = pg.Coalescent(n=4)

        self.assertEqual(1, coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S))
        self.assertEqual(1, coal.total_branch_length._get_regularization_factor(coal.lineage_counting_state_space.S))
        self.assertEqual(1, coal.sfs._get_regularization_factor(coal.block_counting_state_space.S))

        pg.Settings.regularize = True

    def test_enable_regularization(self):
        """
        Test enabling regularization.
        """
        pg.Settings.regularize = True

        coal = pg.Coalescent(n=4)

        self.assertNotEqual(1, coal.tree_height._get_regularization_factor(coal.lineage_counting_state_space.S))
        self.assertNotEqual(
            1, coal.total_branch_length._get_regularization_factor(coal.lineage_counting_state_space.S)
        )
        self.assertNotEqual(1, coal.sfs._get_regularization_factor(coal.block_counting_state_space.S))

    def test_fewer_than_2_lineages_raises_error(self):
        """
        Test fewer than 2 lineages raises error.
        """
        with self.assertRaises(ValueError):
            _ = pg.Coalescent(n=1)

    def test_recombination_tree_height_covariance(self):
        """
        Test tree height covariance against theoretical expectations
        """
        covs = []
        covs_exp = []
        ps = [10 ** -i for i in range(10)[::-1]]

        for p in ps:
            coal = pg.Coalescent(
                n=2,
                loci=pg.LocusConfig(n=2, recombination_rate=p / 2)
            )

            cov = coal.tree_height.loci.cov[0, 1]
            cov_exp = (p + 18) / (p ** 2 + 13 * p + 18)

            covs += [cov]
            covs_exp += [cov_exp]

        plt.plot(covs, label='Observed')
        plt.plot(covs_exp, label='Expected')
        plt.xticks(range(len(covs)), ps)
        plt.legend()
        plt.show()

        np.testing.assert_allclose(covs, covs_exp, atol=1e-14, rtol=0)

    def test_rescale_S_single_kingman(self):
        """
        Test rate matrix rescaling for Kingman coalescent with a single population.
        """
        coal = pg.Coalescent(n=6)
        self.assertFalse('S' in coal.block_counting_state_space.__dict__)

        _ = coal.block_counting_state_space.S
        coal.block_counting_state_space.update_epoch(pg.Epoch(pop_sizes={'pop_0': 3}))

        np.testing.assert_array_almost_equal(
            coal.block_counting_state_space.S * 3,
            pg.Coalescent(n=6).block_counting_state_space.S
        )
        self.assertTrue('S' in coal.block_counting_state_space.__dict__)

    def test_rescale_S_beta(self):
        """
        Test rate matrix rescaling for Beta coalescent with a single population.
        """
        coal1 = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.7))
        coal2 = pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=1.7),
                              demography=pg.Demography(pop_sizes={'pop_0': {0: 3}}))

        r = coal2.block_counting_state_space._get_scaling_factor(
            epoch_prev=next(coal1.demography.epochs),
            epoch_next=next(coal2.demography.epochs)
        )

        np.testing.assert_array_almost_equal(
            coal1.block_counting_state_space.S * r,
            coal2.block_counting_state_space.S
        )

    def test_rescale_S_dirac(self):
        """
        Test rate matrix rescaling for Dirac coalescent with a single population.
        """
        coal1 = pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.4, c=5))
        coal2 = pg.Coalescent(n=6, model=pg.DiracCoalescent(psi=0.4, c=5),
                              demography=pg.Demography(pop_sizes={'pop_0': {0: 3}}))

        r = coal2.block_counting_state_space._get_scaling_factor(
            epoch_prev=next(coal1.demography.epochs),
            epoch_next=next(coal2.demography.epochs)
        )

        np.testing.assert_array_almost_equal(
            coal1.block_counting_state_space.S * r,
            coal2.block_counting_state_space.S
        )

    def test_rescale_S_multi_pop(self):
        """
        Test that rescaling does not cache S for multiple populations.
        """
        coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2})
        _ = coal.block_counting_state_space.S
        coal.block_counting_state_space.update_epoch(pg.Epoch(pop_sizes={'pop_0': 3}))

        self.assertFalse('S' in coal.block_counting_state_space.__dict__)

    def test_flattened_block_counting_standard_coalescent_2_epochs(self):
        """
        Make sure flattening block counting states works correctly.
        """
        pg.Settings.flatten_block_counting = True
        times = np.linspace(0, 30, 10)
        n = 10
        demography = pg.Demography(
            pop_sizes={'pop_0': {0: 1, 1: 10}}
        )

        coal_flattened = pg.Coalescent(n=n, demography=demography)
        flattened = np.array([coal_flattened.sfs.get_accumulation(1, i, times) for i in range(10)])

        # the monomorphic corner (bin 0, here over range(10)) falls back to the block-counting traversal, which
        # builds the state probabilities; the polymorphic bins use the closed-form Kingman weights
        self.assertTrue('_state_probs' in coal_flattened.block_counting_state_space.__dict__)

        pg.Settings.flatten_block_counting = False
        coal_original = pg.Coalescent(n=n, demography=demography)
        original = np.array([coal_original.sfs.get_accumulation(1, i, times) for i in range(10)])

        # make sure state probabilities are not cached
        self.assertFalse('_state_probs' in coal_original.block_counting_state_space.__dict__)

        np.testing.assert_array_almost_equal(original, flattened, decimal=14)
        pg.Settings.flatten_block_counting = True

    def test_sample_accrues_reward_across_zero_rate_epochs(self):
        """
        The sampler must accrue elapsed time and reward across zero-rate epochs (temporarily isolated demes). With
        migration switched off until t=1, coalescence is impossible before t=1, so no sample may fall below that
        floor and the empirical distribution must match the exact phase-type result (regression for the sampler
        dropping the isolation interval).
        """
        dem = pg.Demography(
            pop_sizes={'p0': {0: 1.0}, 'p1': {0: 1.0}},
            migration_rates={('p0', 'p1'): {0: 0.0, 1: 1.0}, ('p1', 'p0'): {0: 0.0, 1: 1.0}}
        )
        th = pg.Coalescent(n={'p0': 1, 'p1': 1}, demography=dem).tree_height

        s = th.sample(20000, seed=0)

        # coalescence is impossible while the demes are isolated (t < 1)
        self.assertTrue((s >= 1.0).all())

        # empirical mean and CDF agree with the exact result
        self.assertAlmostEqual(s.mean(), th.mean, delta=0.1)
        for q in (0.25, 0.5, 0.75):
            x = float(th.quantile(q))
            self.assertAlmostEqual(np.mean(s <= x), q, delta=0.03)

    def test_sample_batching_matches_single_pass(self):
        """
        Batched sampling (small :attr:`Settings.sample_batch_size`) must realise the same distribution as a single
        ensemble pass and agree with the exact result, across a multi-epoch multi-population demography with a
        zero-rate (isolated) initial epoch.
        """
        from scipy import stats

        dem = pg.Demography(
            pop_sizes={'p0': {0: 1.0, 1.5: 2.0}, 'p1': {0: 1.0, 1.5: 0.5}},
            migration_rates={('p0', 'p1'): {0: 0.0, 1: 1.0}, ('p1', 'p0'): {0: 0.0, 1: 1.0}}
        )
        th = pg.Coalescent(n={'p0': 3, 'p1': 3}, demography=dem).tree_height

        saved = pg.Settings.sample_batch_size
        try:
            pg.Settings.sample_batch_size = None  # single pass
            single = th.sample(20000, seed=0)

            pg.Settings.sample_batch_size = 3000  # several batches, including a short final one
            batched = th.sample(20000, seed=0)
        finally:
            pg.Settings.sample_batch_size = saved

        # batching does not change the law (no significant KS difference)
        self.assertEqual(batched.shape, single.shape)
        self.assertGreater(stats.ks_2samp(single, batched).pvalue, 0.01)
        # and both match the exact mean; coalescence is impossible while the demes are isolated (t < 1)
        self.assertAlmostEqual(batched.mean(), th.mean, delta=0.1)
        self.assertTrue((batched >= 1.0).all())

    def test_sample_empirical_pdf(self):
        """
        The sampled mean of an SFS bin agrees with its exact moment. The sampler is seeded and the tolerance clears
        four standard errors of the sampled mean: at 10,000 unseeded draws the relative standard error is 2.5%, so
        the previous 5% threshold sat at two standard errors and failed about one run in twenty.
        """
        coal = pg.Coalescent(
            n=10,
            model=pg.BetaCoalescent(alpha=1.7),
            demography=pg.Demography(
                pop_sizes={'pop_0': {0: 1, 1: 10}}
            )
        )

        exact = coal.moment(1, (pg.UnfoldedSFSReward(2),))

        empirical = coal.sfs.sample(100000, seed=42)[:, 2].mean()

        # 100,000 draws give a relative standard error of about 0.8%
        rel_diff = np.abs(empirical - exact) / exact

        self.assertLessEqual(rel_diff, 0.032)

    def test_sample_empirical_cdf(self):
        """
        Test the empirical CDF against the exact CDF.
        """
        coal = pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 1, 'pop_1': 1, 'pop_2': 1}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        )

        t = np.linspace(0, coal.tree_height.quantile(0.99), 100)
        empirical = coal.tree_height.to_empirical(10000, seed=0).cdf(t)
        exact = coal.tree_height.cdf(t=t)

        # the early bins of the exact CDF are ~0 (division by zero); only the tail [20:] is asserted on, so
        # silence the benign divide warning rather than emit it
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(empirical - exact) / exact

        self.assertLess(rel_diff[20:].mean(), 0.02)

    def test_plot_empirical_cdf(self):
        """
        Test plotting the empirical CDF.
        """
        pg.Coalescent(
            n=pg.LineageConfig({'pop_0': 1, 'pop_1': 1, 'pop_2': 1}),
            model=pg.BetaCoalescent(alpha=1.7),
            demography=self.get_complex_demography()
        ).tree_height.to_empirical(1000, seed=0).cdf.plot(show=False)

    def test_empirical_cdf_of_non_default_reward(self):
        """The empirical CDF of the total branch length is a valid CDF."""
        coal = pg.Coalescent(n=4)
        t = np.linspace(0, coal.total_branch_length.quantile(0.99), 50)
        y = coal.total_branch_length.to_empirical(500, seed=0).cdf(t)

        self.assertEqual(np.shape(y), t.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all((y >= 0) & (y <= 1)))
        self.assertTrue(np.all(np.diff(y) >= 0))  # a CDF is non-decreasing

    def test_plot_accumulation_returns_axes(self):
        """``plot_accumulation`` is annotated ``-> plt.Axes`` and must return the axes rather than ``None``."""
        coal = pg.Coalescent(n=4)
        ax = coal.tree_height.plot_accumulation(1, np.linspace(0, 2, 10), show=False)
        self.assertIsInstance(ax, plt.Axes)

    def test_sfs_cdf_caches_bin_distributions_and_matches_scalar(self):
        """``sfs.cdf`` caches the per-bin distribution (so the cosine fit is built once, not rebuilt on every call)
        and the vectorized array evaluation matches the scalar path. Regression for the per-call fit rebuild."""
        coal = pg.Coalescent(n=5)
        t = np.linspace(0.1, 2.5, 10)

        arr = np.asarray(coal.sfs.cdf(t))
        self.assertTrue(coal.sfs.__dict__.get('_bin_distributions'))  # per-bin fits cached on the spectrum

        # the array path equals the scalar path bin for bin
        for k, tv in enumerate(t):
            np.testing.assert_allclose(np.asarray(coal.sfs.cdf(float(tv)).data), arr[k], rtol=1e-9, atol=1e-9)

    def test_cdf_grid_cache_rebuilds_when_the_inversion_settings_change(self):
        """The shared CDF grid is built for one inversion configuration: the tail cut places the exact nodes, the de
        Hoog degree sets their values, and the term count the fit below them. Changing any of the three on a live
        distribution must discard the stale grid and rebuild, not silently reuse it."""
        coal = pg.Coalescent(n=5)
        bd = coal.sfs._bin_distribution(2)
        originals = (pg.Settings.dehoog_tail_quantile, pg.Settings.dehoog_degree, pg.Settings.cos_terms)
        try:
            _ = bd.quantile(0.9)
            self.assertEqual(bd.__dict__['_lst_curve_cache']['_config'], originals)

            for attr, changed in (
                    ('dehoog_tail_quantile', 0.9 if originals[0] != 0.9 else 0.95),
                    ('dehoog_degree', originals[1] + 1),
                    ('cos_terms', originals[2] * 2)
            ):
                setattr(pg.Settings, attr, changed)
                _ = bd.quantile(0.9)
                self.assertEqual(
                    bd.__dict__['_lst_curve_cache']['_config'],
                    (pg.Settings.dehoog_tail_quantile, pg.Settings.dehoog_degree, pg.Settings.cos_terms),
                    attr
                )
        finally:
            (pg.Settings.dehoog_tail_quantile, pg.Settings.dehoog_degree, pg.Settings.cos_terms) = originals

    def test_compare_state_reward_flattened(self):
        """
        Test that flattened state rewards match the original state rewards. The absorbing state is excluded: the
        accumulation runs until absorption and therefore never occupies an absorbing state, so a reward with mass
        there is rejected rather than accumulated. The engine did not return the mathematical value zero for it but
        the time the finite epochs spend absorbed, measured for ``n = 4`` as 0 with a single epoch, 0.0092 with one
        redundant identical boundary and 1.5896 with three.
        """
        # this exercises the block-counting flatten path, which the closed-form moment evaluation would bypass
        pg.Settings.closed_form_last_epoch = False

        coal = pg.Coalescent(n=10)
        transient = np.where(~coal.block_counting_state_space.absorbing)[0]

        pg.Settings.flatten_block_counting = True
        flattened = [coal.moment(1, rewards=(pg.StateReward(i),)) for i in transient]
        self.assertTrue('_state_probs' in coal.block_counting_state_space.__dict__)

        coal = pg.Coalescent(n=10)
        pg.Settings.flatten_block_counting = False
        original = [coal.moment(1, rewards=(pg.StateReward(i),)) for i in transient]
        self.assertFalse('_state_probs' in coal.block_counting_state_space.__dict__)

        np.testing.assert_array_almost_equal(flattened, original)
        pg.Settings.flatten_block_counting = True

    def test_not_flattened_block_counting_beta_coalescent(self):
        """
        Make sure that not flattening block counting states works correctly.
        """
        coal_original = pg.Coalescent(
            n=3,
            model=pg.BetaCoalescent(alpha=1.7),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1}})
        )
        _ = coal_original.sfs.mean.data

        # make sure state probabilities were not accessed
        self.assertFalse('_state_probs' in coal_original.block_counting_state_space.__dict__)

    def test_beta_coalescent_state_props(self):
        """
        Compare state probabilities of beta coalescent with empirical sampling.
        """
        coal = pg.Coalescent(
            n=10,
            model=pg.BetaCoalescent(1.7),
            demography=pg.Demography(pop_sizes={'pop_0': 1})
        )

        samples, probs_empirical = coal.sfs._sample(10000, record_visits=True)

        probs = coal.block_counting_state_space._state_probs

        self.assertLess((np.abs(probs_empirical - probs) / probs).mean(), 0.08)

    def test_recorded_visits_include_initial_state(self):
        """
        ``record_visits`` must count each walker's initial state (drawn from alpha), not only the states entered by a
        jump. Regression: the initial state's visit frequency was reported as 0. The single-population coalescent
        starts deterministically in the all-singletons state, so its recorded frequency must be ~1.
        """
        coal = pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': 1}))
        ss = coal.block_counting_state_space

        _, probs = coal.sfs._sample(5000, record_visits=True)

        initial = np.asarray(ss.alpha) > 0
        self.assertEqual(initial.sum(), 1)  # a single deterministic starting state
        self.assertAlmostEqual(float(probs[initial][0]), 1.0, delta=0.02)

    def test_stuck_walker_finite_for_zero_rate_rewards(self):
        """
        A walker stuck in a zero-exit-rate state (isolated demes never reach the grand MRCA) accumulates an infinite
        reward only for components with a positive rate there. Regression: every reward component was set to inf. With
        3+3 samples in two isolated demes the stuck state is one lineage per deme, each subtending 3 samples, so only
        the 3-ton bin has a positive rate; the others stay finite.
        """
        from phasegen.rewards import CombinedReward

        dem = pg.Demography(pop_sizes={'pop_0': {0: 1.0}, 'pop_1': {0: 1.0}})  # no migration -> isolated
        sfs = pg.Coalescent(n={'pop_0': 3, 'pop_1': 3}, demography=dem).sfs
        rewards = [CombinedReward([sfs.reward, sfs._get_sfs_reward(i)]) for i in sfs._get_indices()]

        samp = np.asarray(sfs._sample(2000, rewards=rewards))
        finite_frac = np.mean(np.isfinite(samp), axis=0)

        # both stuck lineages subtend 3 samples, so exactly the 3-ton bin accumulates an infinite reward while
        # every other bin stays finite in every sample
        indices = list(sfs._get_indices())
        three_ton = indices.index(3)
        self.assertEqual(finite_frac[three_ton], 0.0)
        self.assertTrue(np.all(np.delete(finite_frac, three_ton) == 1.0))


def test_discretized_demography_with_infinite_end_time_terminates():
    """
    A discretized event caps every epoch at its next discretization step, so with the default infinite end time the
    demography has infinitely many epochs and none of them is unbounded. Every moment materialized epochs until an
    unbounded one appeared and therefore never returned. Epoch consumption must stop once absorption is almost sure,
    with the remainder treated as the last epoch.
    """
    def build() -> pg.Coalescent:
        return pg.Coalescent(n=2, demography=pg.Demography(events=[pg.ExponentialPopSizeChanges(
            initial_size={'pop_0': 1.0}, growth_rate=1.0, start_time=0.0, step_size=0.5
        )]))

    coal = build()
    epochs = coal.tree_height._get_epochs_until_unbounded()

    # the epochs stop at the first one beginning at or after the time of almost sure absorption
    assert np.isinf(epochs[-1].end_time)
    assert epochs[-2].start_time < coal.tree_height.t_max <= epochs[-1].start_time
    assert len(epochs) == int(np.ceil(coal.tree_height.t_max / 0.5)) + 1

    # the closed-form last epoch (the default) and the matrix exponential up to the absorption time must agree
    pg.Settings.closed_form_last_epoch = False
    reference = build().tree_height.mean
    pg.Settings.closed_form_last_epoch = True

    assert coal.tree_height.mean == pytest.approx(reference, rel=1e-10)

    # the sampler and the mutation configurations consume the same epochs
    assert np.isfinite(pg.Coalescent(n=4, demography=coal.demography).tree_height.sample(10, seed=0)).all()
    assert 0 < pg.Coalescent(n=4, demography=coal.demography).sfs.get_mutation_config((1, 0, 0), theta=1.0) < 1


def test_accumulate_starts_at_the_configured_start_time():
    """
    ``accumulate`` hard-defaulted its start time to zero, so on a coalescent with a positive start time it
    accumulated over a window no other method used: ``accumulate(1, [2.0])`` returned the from-zero 1.3305 where
    ``moment(1, end_time=2.0)`` returned 0.8414, and the accumulation curve converged to the from-zero 2(1 - 1/n)
    rather than to ``tree_height.mean``.
    """
    coal = pg.Coalescent(n=5, start_time=0.5)
    t_max = coal.tree_height.t_max

    assert coal.accumulate(1, [2.0])[0] == pytest.approx(coal.moment(1, end_time=2.0), rel=1e-10)
    assert coal.tree_height.accumulate(2, [2.0])[0] == pytest.approx(coal.tree_height.moment(2, end_time=2.0),
                                                                     rel=1e-10)
    assert coal.tree_height.accumulate(1, [t_max])[0] == pytest.approx(coal.tree_height.mean, rel=1e-10)

    # an end time at or before the start time accumulates nothing
    assert coal.tree_height.accumulate(1, [0.2])[0] == 0

    # the spectrum takes the same window, through its batched occupation grid
    np.testing.assert_allclose(np.asarray(coal.sfs.accumulate(1, [2.0]))[0],
                               np.asarray(coal.sfs.moment(1, end_time=2.0).data), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(coal.sfs.accumulate(1, [t_max]))[0],
                               np.asarray(coal.sfs.mean.data), rtol=1e-10)

    # a coalescent without a window is unaffected
    ref = pg.Coalescent(n=5)
    assert ref.accumulate(1, [2.0])[0] == pytest.approx(ref.moment(1, end_time=2.0), rel=1e-10)


def test_epoch_truncation_keeps_every_epoch_carrying_probability_mass():
    """
    ``_get_epochs_until_unbounded`` holds an epoch beginning at or after the time of almost sure absorption and
    extends that epoch's rates over the remaining time. Given an absorption time at which absorption
    is genuine, the truncated epoch list must describe the same model as the full demography: with a time at which
    most of the mass was still unabsorbed, the two epochs after it were discarded and the mean tree height came out
    as 1499.999 against 6.0000005 from an explicit end time past absorption.
    """
    from phasegen.distributions.phase_type import TreeHeightDistribution

    demography = pg.Demography(pop_sizes={'pop_0': {0: 1e-6, 1e-12: 1e3, 3: 1e3, 6: 1e-6}})

    # the population contracts to 1e-6 at t = 6, so no mass survives to t = 8
    with patch.object(TreeHeightDistribution, '_get_absorption_time', lambda self: 8.0):
        coal = pg.Coalescent(n=4, demography=demography)
        epochs = coal.tree_height._get_epochs_until_unbounded()
        mean = coal.tree_height.mean

    assert [e.start_time for e in epochs] == [0, 1e-12, 3.0, 6.0]
    assert np.isinf(epochs[-1].end_time)

    reference = pg.Coalescent(n=4, demography=demography, end_time=20.0).tree_height.mean
    assert mean == pytest.approx(reference, rel=1e-9)


def test_moment_accepts_an_integral_float_order_and_rejects_a_non_integral_one():
    """
    ``Coalescent.moment`` and ``SFSDistribution.moment`` built their default rewards by replicating a sequence with
    the raw order, so a float order raised ``TypeError: can't multiply sequence by non-int of type 'float'`` where
    ``PhaseTypeDistribution.moment`` accepted it. R numerics are doubles unless suffixed with ``L``, so ``coal$moment(2)``
    failed. The lenient paths in turn truncated a non-integral order silently, returning the order-2 moment for
    ``k = 2.5``.
    """
    coal = pg.Coalescent(n=4)

    assert coal.moment(2.0) == pytest.approx(coal.moment(2), rel=1e-12)
    assert coal.sfs.moment(2.0).data[1] == pytest.approx(coal.sfs.moment(2).data[1], rel=1e-12)
    assert coal.accumulate(2.0, [1.0])[0] == pytest.approx(coal.accumulate(2, [1.0])[0], rel=1e-12)
    assert coal.sfs.accumulate(2.0, [1.0])[0][1] == pytest.approx(coal.sfs.accumulate(2, [1.0])[0][1], rel=1e-12)

    calls = (
        coal.moment,
        coal.sfs.moment,
        coal.tree_height.moment,
        coal.total_branch_length.moment,
        lambda k: coal.tree_height.accumulate(k, [1.0]),
        lambda k: coal.sfs.get_accumulation(k, 1, 1.0),
    )

    for call in calls:
        with pytest.raises(ValueError, match='must be an integer'):
            call(2.5)

        with pytest.raises(ValueError, match='must be non-negative'):
            call(-1)

        # the order zero is one, in bin 1 for the spectra
        zero = np.atleast_1d(np.asarray(getattr(call(0), 'data', call(0)), dtype=float)).ravel()
        assert zero[min(1, zero.size - 1)] == 1

        with pytest.raises(TypeError, match='must be an integer'):
            call('2')


def test_epoch_extension_keeps_consuming_while_a_later_epoch_would_misplace_the_accumulation():
    """
    Almost sure absorption is a statement about probability alone, and the accumulated reward has a time scale the
    probability does not carry. The epoch reached at the absorption estimate stands in for every epoch after it, so
    with 1e-15 of the mass still transient and an epoch whose own absorption time is 1e18, the substitution
    misplaced 4.25 of expected tree height, returning 5.248354 where the exact value is 1. The surviving mass is
    read off the propagated state vector because ``1 - cdf`` underflows to exactly zero there.
    """
    demography = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 40: 1e18, 70: 1e18, 100: 1e-9}})

    coal = pg.Coalescent(n=2, demography=demography)
    bounded = pg.Coalescent(n=2, demography=demography, end_time=1e3)

    assert float(coal.tree_height.mean) == pytest.approx(float(bounded.tree_height.mean), rel=1e-9)

    # the epoch whose rates are held until absorption is the one whose own time scale makes the remainder negligible
    assert coal.tree_height._get_epochs_until_unbounded()[-1].start_time == 100.0

    # 1 - cdf cannot see the surviving mass that drives the error, so the criterion may not be built on it
    assert 1 - float(coal.tree_height.cdf(coal.tree_height.t_max)) == 0.0
    assert 0 < coal.tree_height._survival(coal.tree_height.t_max) < 1e-15


def test_state_spaces_lists_the_spaces_the_configuration_supports():
    """Spaces a configuration cannot build are left out rather than raising."""
    assert list(pg.Coalescent(n=3).state_spaces) == [
        'lineage_counting_state_space', 'block_counting_state_space', 'joint_block_counting_state_space'
    ]
    assert list(pg.Coalescent(n=3, loci=2).state_spaces) == [
        'lineage_counting_state_space', 'two_locus_block_counting_state_space'
    ]


@pytest.mark.parametrize('pop_sizes, n_epochs', [({0: 1}, 1), ({0: 1, 0.3: 0.2, 1.0: 3}, 3)])
def test_stability_warning_names_real_epochs_once_each(caplog, pop_sizes, n_epochs):
    """The exit-rate spread warning of two demes joined by rare migration names the epoch and fires once per epoch.
    Regression: the tree-height grid passed its segment index as the epoch, logging 168 warnings for 'epochs' 0 to
    129 on a single epoch."""
    import re

    coal = pg.Coalescent(n={'a': 2, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': pop_sizes, 'b': {0: 1}}, migration_rates={('a', 'b'): {0: 1e-11}, ('b', 'a'): {0: 1e-11}}))

    with caplog.at_level('WARNING'):
        coal.tree_height.quantile(0.5)
        coal.tree_height.cdf(1.0)

    pattern = r'epoch (\d+) has total exit'
    epochs = [int(m.group(1)) for r in caplog.records if (m := re.search(pattern, r.getMessage()))]

    assert epochs
    assert len(epochs) == len(set(epochs))
    assert set(epochs) == set(range(n_epochs))


@pytest.mark.parametrize('n, psi', [(10, 0.1), (10, 0.05), (20, 0.3)])
def test_no_stability_warning_for_rare_dirac_mergers(caplog, n, psi):
    """A Dirac merger rate far below the other rates out of the same state leaves the exit rates, and the exact
    moments, unaffected, so no stability warning is logged. Regression: the warning compared every positive rate and
    fired on moments exact to 1e-16 once psi ** n fell below 1e-10 of the largest rate."""
    coal = pg.Coalescent(n=n, model=pg.DiracCoalescent(psi=psi, c=1))

    with caplog.at_level('WARNING'):
        coal.tree_height.mean
        coal.tree_height.var

    assert not [r for r in caplog.records if 'orders of magnitude' in r.getMessage()]


def test_high_moments_after_a_short_epoch_match_an_extended_precision_reference():
    """The Van Loan step of an epoch whose rates times its duration are small keeps its higher-order blocks resolved.
    Regression: the balancing factor drawn from the rates alone left the scaled step at about 1e-3, so the fifth raw
    moment lost 1e-8 relative after a short large epoch (the fifth central moment 73%), and 4.5e-2 in a window ending
    shortly into the last epoch. References from a 50-digit mpmath Van Loan computation."""
    coal = pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1e3, 0.5: 1e-3}}))

    assert coal.tree_height.moment(5, center=False) == pytest.approx(0.031670133796944716, rel=1e-12)
    assert coal.tree_height.moment(5) == pytest.approx(-5.6834198753562267e-10, rel=1e-5, abs=0)

    pg.Settings.expm_action_min_dim = 1
    coal = pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1e-3, 1e-4: 1e3}}))

    raw = coal.moment(5, [pg.rewards.TreeHeightReward()] * 5, center=False, end_time=1.5e-4)
    assert raw == pytest.approx(7.4974726817100355e-20, rel=1e-12, abs=0)


@pytest.mark.parametrize('sparse', [False, True])
def test_high_moments_across_a_very_short_epoch_match_an_extended_precision_reference(sparse):
    """An epoch of length 1e-12 leaves the higher raw moments of non-triangular generators exact. Regression: the
    balancing factor capped at the epoch's length rebased the extended vector by about 1e12 per order, which amplified
    the rounding the dense exponential leaves in its structurally zero lower blocks (the third raw moment of the total
    branch length of two demes came out at -1.4e11). A first epoch of length 1e-70 overflowed the rebase to infinity.
    References from a 100-digit mpmath Van Loan computation."""
    if sparse:
        pg.Settings.closed_form_sparse_min_states = 1
        pg.Settings.expm_action_min_dim = 0

    two_demes = pg.Coalescent(n=pg.LineageConfig({'a': 2, 'b': 2}), demography=pg.Demography(
        pop_sizes={'a': {0: 1, 1: 2, 1 + 1e-12: 0.5}, 'b': {0: 1, 1: 3}},
        migration_rates={('a', 'b'): {0: 1, 1: 0.2}, ('b', 'a'): {0: 1}}))
    two_loci = pg.Coalescent(n=3, loci=pg.LocusConfig(n=2, recombination_rate=1),
                             demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 2, 1 + 1e-12: 0.5}}))
    beta = pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=1.5),
                         demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 2, 1 + 1e-12: 0.5}}))
    tiny_first = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 1e-70: 2}}))

    assert two_demes.total_branch_length.moment(3, center=False) == pytest.approx(625.12101468818567, rel=1e-12)
    assert two_demes.total_branch_length.moment(5, center=False) == pytest.approx(121557.67449041378, rel=1e-12)
    assert two_loci.tree_height.moment(5, center=False) == pytest.approx(25.839672442948937, rel=1e-12)
    assert beta.total_branch_length.moment(5, center=False) == pytest.approx(36034.653923276051, rel=1e-12)
    assert tiny_first.tree_height.moment(5, center=False) == pytest.approx(6896.2962962962963, rel=1e-12)


def test_moments_survive_absorption_memos_stored_in_an_older_form():
    """Payloads from v1.2 store the absorption-certainty memo as a bool, which is replaced rather than indexed.
    Regression: every uncached moment of such a payload raised TypeError: argument of type 'bool' is not iterable."""
    coal = pg.Coalescent(n=4)

    for dist in (coal.tree_height, coal.total_branch_length, coal.sfs):
        dist.__dict__['_absorption_certain_cache'] = True
        dist.__dict__['_alpha_support_cache'] = None

    restored = pg.Coalescent.from_json(coal.to_json())

    assert restored.tree_height.moment(3, center=False) == pytest.approx(
        pg.Coalescent(n=4).tree_height.moment(3, center=False), rel=1e-12)
    np.testing.assert_allclose(restored.sfs.moment(3, center=False).data,
                               pg.Coalescent(n=4).sfs.moment(3, center=False).data, rtol=1e-12)


def test_action_path_moments_do_not_depend_on_the_other_end_times():
    """On the sparse-action path each step is balanced on its own length, so a moment at one end time is the same
    whether or not later end times are requested with it. Regression: the factor was taken from the last end time,
    and the fifth raw moment at t = 1e-3 came out 44% off when evaluated together with t = 10."""
    pg.Settings.expm_action_min_dim = 1
    reward = (pg.rewards.TreeHeightReward(),) * 5
    times = [1.5e-4, 1e-3, 1e-2, 10]

    def dist():
        return pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1e-3, 1e-4: 1e3}})).tree_height

    together = dist().accumulate(5, times, reward, center=False)
    alone = [dist().accumulate(5, [t], reward, center=False)[0] for t in times]

    np.testing.assert_allclose(together, alone, rtol=1e-12)
    assert together[0] == pytest.approx(7.4974726817100355e-20, rel=1e-12, abs=0)


def test_sfs_covariance_at_large_n_solves_instead_of_inverting(monkeypatch):
    """The batched SFS covariance solves with the LU of -T and never forms its inverse, so n = 35 (14,883 states)
    takes about a second. Regression: the dense inverse took 150 s and 6.9 GB there and ran out of memory at n = 40.
    The covariances sum to the variance of the total branch length, 4 sum_{k<n} 1/k^2 for the standard coalescent."""
    import scipy.linalg

    def refuse(*args, **kwargs):
        raise AssertionError('the covariance formed a dense inverse')

    monkeypatch.setattr(scipy.linalg, 'inv', refuse)

    n = 35
    cov = pg.Coalescent(n=n).sfs.cov.data

    assert cov.sum() == pytest.approx(4 * sum(1 / k ** 2 for k in range(1, n)), rel=1e-12)


def test_population_split_example_emits_no_singular_matrix_warning():
    """An epoch without an absorption path is recognised without scipy's singular-matrix warning, which was raised
    through the warnings module for the documented split example."""
    import warnings
    from scipy.linalg import LinAlgWarning

    dem = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 3}, events=[pg.PopulationSplit(2, 'pop_1', 'pop_0')])
    coal = pg.Coalescent(n={'pop_0': 4, 'pop_1': 4}, demography=dem)

    with warnings.catch_warnings():
        warnings.simplefilter('error', LinAlgWarning)
        assert np.isfinite(coal.tree_height.mean)
        assert np.all(np.isfinite(coal.sfs.mean.data))


def test_tree_height_cdf_and_pdf_pass_nan_through():
    """A NaN point gives NaN, the other points their values. Regression: the NaN sorted to the end of the sweep and
    took the value of the last finite point, or 0.0 for a scalar."""
    th = pg.Coalescent(n=3).tree_height

    np.testing.assert_array_equal(np.isnan(th.cdf(np.array([np.nan, 1.0]))), [True, False])
    assert th.cdf(np.array([np.nan, 1.0]))[1] == pytest.approx(th.cdf(1.0), rel=1e-12)
    assert np.isnan(th.cdf(np.nan)) and np.isnan(th.pdf(np.nan))


@pytest.mark.parametrize("demography", [
    pg.Demography(pop_sizes={'pop_0': 1}),
    pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 3}}),
])
def test_infinite_end_time_equals_no_end_time(demography):
    """``end_time=np.inf`` behaves as no end time. Regression: t_max returned inf, and tree_height.quantile and the
    default accumulation grid raised IndexError."""
    ref = pg.Coalescent(n=4, demography=demography)
    coal = pg.Coalescent(n=4, demography=demography, end_time=np.inf)

    assert coal.end_time is None and coal.tree_height.end_time is None
    assert coal.tree_height.t_max == ref.tree_height.t_max
    assert coal.tree_height.quantile(0.5) == pytest.approx(ref.tree_height.quantile(0.5), rel=1e-10)
    assert coal.tree_height.cdf(1.0) == pytest.approx(ref.tree_height.cdf(1.0), rel=1e-12)
    assert coal.tree_height.mean == pytest.approx(ref.tree_height.mean, rel=1e-12)
    np.testing.assert_array_equal(coal.tree_height._default_end_times(), ref.tree_height._default_end_times())


def _migration_switch(n: dict, before: float, after: float, T: float) -> pg.Coalescent:
    """Two demes of size one whose symmetric migration rate switches from ``before`` to ``after`` at ``T``."""
    return pg.Coalescent(n=n, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1},
        migration_rates={('a', 'b'): {0: before, T: after}, ('b', 'a'): {0: before, T: after}}
    ))


def _doubling_point(n: dict, before: float, j: int) -> float:
    """The time ``scale * 2 ** j`` of the absorption search, which depends on the first epoch alone."""
    return _migration_switch(n, before, before, 1e6).tree_height._get_absorption_scale() * 2 ** j


@pytest.mark.parametrize('n', [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}, {'a': 2, 'b': 2}])
@pytest.mark.parametrize('j', [1, 2, 3, 4])
@pytest.mark.parametrize('shift', [1.0, 1.001])
def test_isolation_then_migration_absorbs_on_and_off_the_doubling_grid(n, j, shift, monkeypatch):
    """Isolated demes joined by migration in the final epoch absorb, also when the final epoch starts exactly on a
    doubling point of the absorption search. Regression: there the reachability was read off the isolated epoch
    before it, and the search raised ModelError. For one lineage per deme the mean is ``T + 2.5``."""
    T = _doubling_point(n, 0.0, j) * shift
    coal = _migration_switch(n, 0.0, 1.0, T)

    assert np.isfinite(coal.tree_height.t_max)
    assert np.isfinite(coal.tree_height.quantile(0.5))

    closed = coal.tree_height.mean
    monkeypatch.setattr(pg.Settings, 'closed_form_last_epoch', False)
    windowed = _migration_switch(n, 0.0, 1.0, T).tree_height.mean

    assert windowed == pytest.approx(closed, rel=1e-8)
    if n == {'a': 1, 'b': 1}:
        assert closed == pytest.approx(T + 2.5, rel=1e-10)


@pytest.mark.parametrize('n', [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}])
@pytest.mark.parametrize('T', [('grid', 1), ('grid', 3), ('grid', 5), ('off', 1.001 * 2 ** 3), ('off', 20.0)])
def test_migration_then_isolation_never_absorbs(n, T):
    """Demes isolated in the final epoch do not absorb, however little mass is left to be stranded there, and every
    statistic that needs the whole distribution says so alike. Regression: on a doubling point the reachability was
    read off the migrating epoch before, and a stranded mass below 1e-8 passed the search, so the means came back as
    the stranded mass times the doubling ceiling."""
    T = _doubling_point(n, 1.0, T[1]) if T[0] == 'grid' else _doubling_point(n, 1.0, 0) * T[1]
    coal = _migration_switch(n, 1.0, 0.0, T)

    for get in [lambda: coal.tree_height.t_max, lambda: coal.tree_height.mean,
                lambda: coal.total_branch_length.mean, lambda: coal.distribution(pg.TreeHeightReward()).cdf(1.0),
                lambda: coal.tree_height.cdf(1.0), lambda: coal.tree_height.pdf(1.0)]:
        with pytest.raises(ModelError, match="does not absorb"):
            get()


@pytest.mark.parametrize('n', [2, 4])
@pytest.mark.parametrize('j', [1, 2, 3])
def test_one_deme_size_change_on_the_doubling_grid(n, j, monkeypatch):
    """A single deme whose size changes on a doubling point of the absorption search gives the same mean on the
    closed-form and the windowed path."""
    T = pg.Coalescent(n=n).tree_height._get_absorption_scale() * 2 ** j
    coal = lambda: pg.Coalescent(n=n, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, T: 3}}))

    closed = coal().tree_height.mean
    monkeypatch.setattr(pg.Settings, 'closed_form_last_epoch', False)

    assert coal().tree_height.mean == pytest.approx(closed, rel=1e-8)


def test_conditioning_check_uses_the_coalescence_rates_of_the_model():
    """The conditioning check measures the pairwise coalescence rates of the model, which for the Beta and Dirac
    coalescents do not scale as the inverse population size. Regression: it used ``1 / N``, rejecting a
    well-conditioned Beta coalescent and passing a Dirac coalescent whose rates span 1e20."""
    def coal(model, sizes):
        return pg.Coalescent(n={'a': 1, 'b': 1}, model=model, demography=pg.Demography(
            pop_sizes={'a': sizes[0], 'b': sizes[1]}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))

    assert coal(pg.BetaCoalescent(alpha=1.1), (1e-9, 1e9)).tree_height.mean == pytest.approx(7.212276119517529,
                                                                                              rel=1e-8)

    with pytest.raises(ModelError, match="ill-conditioned"):
        _ = coal(pg.DiracCoalescent(psi=0.5, c=1), (1e-5, 1e5)).tree_height.mean

    with pytest.raises(ModelError, match="ill-conditioned"):
        _ = coal(pg.StandardCoalescent(), (1e-9, 1e9)).tree_height.mean


def test_tree_height_cdf_terminates_when_the_row_sum_norm_overflows():
    """A population size near 1e-308 gives finite rates whose row sums overflow. Regression: the step length was
    ``_max_step_norm / inf = 0`` and the propagation never ended."""
    coal = pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 1.1e-308}}))

    assert coal.tree_height.cdf(2) == 1.0


def test_tree_height_cdf_and_pdf_at_infinity_on_the_sparse_action_path(monkeypatch):
    """At infinity the CDF is one and the density vanishes, on the sparse action path as on the dense one. Regression: the action path propagated until the transient mass underflowed, which left a
    density of 1e-322 and took about 30 times as long as the CDF at ``t_max``."""
    def coals():
        return [pg.Coalescent(n={'a': 6, 'b': 6}, demography=pg.Demography(
            pop_sizes={'a': 1, 'b': 2}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 1}))]

    dense = [(c.tree_height.cdf(np.inf), c.tree_height.cdf(1e6)) for c in coals()]
    monkeypatch.setattr(pg.Settings, 'expm_action_min_dim', 2)

    for c, (cdf_inf, cdf_far) in zip(coals(), dense):
        assert c.tree_height.cdf(np.inf) == pytest.approx(cdf_inf, rel=1e-12)
        assert c.tree_height.cdf(np.inf) == pytest.approx(cdf_far, rel=1e-12)
        assert c.tree_height.pdf(np.inf) == 0.0

    assert dense[0][0] == 1.0


@pytest.mark.parametrize("kwargs, match", [
    (dict(model='beta'), "model must be"),
    (dict(demography={'pop_0': 1}), "demography must be")
])
def test_coalescent_rejects_a_model_or_demography_of_the_wrong_type(kwargs, match):
    """A wrong model or demography failed later with an AttributeError."""
    with pytest.raises(TypeError, match=match):
        pg.Coalescent(n=3, **kwargs)



def test_accumulate_takes_one_shot_iterables():
    """A centered accumulation of order two or more reads the end times once per component. Regression: a one-shot
    iterator was exhausted by the first component and the centering raised an IndexError."""
    c = pg.Coalescent(n=4)
    times = [0.5, 1.0, np.inf]

    np.testing.assert_allclose(c.accumulate(2, iter(times)), c.accumulate(2, times), rtol=1e-14)
    np.testing.assert_allclose(c.tree_height.accumulate(2, (t for t in times)), c.tree_height.accumulate(2, times),
                               rtol=1e-14)


def test_accumulate_rejects_a_negative_start_time():
    """Every accumulation entry point rejects a negative start time, as the moments do. Regression: the windowed
    path integrated over a window extended below zero while the other paths clamped it to zero."""
    c = pg.Coalescent(n=3)

    for accumulate in (c.accumulate, c.tree_height.accumulate, c.sfs.accumulate):
        with pytest.raises(ValueError, match='Start time must be greater than or equal to 0'):
            accumulate(1, [1.0], start_time=-1.0)


def test_batched_spectrum_accumulation_passes_nan_end_times_on_the_action_path(monkeypatch):
    """A NaN end time gives a NaN bin on the sparse-action path of the batched spectrum accumulation, and the other
    end times the values of the dense path. Regression: the action path raised a ValueError."""
    def acc():
        return pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=1.5)).sfs.accumulate(1, [0.5, np.nan, 2.0])

    dense = acc()
    monkeypatch.setattr(pg.Settings, 'expm_action_min_dim', 0)
    action = acc()

    assert np.isnan(action[1, 1:-1]).all()
    np.testing.assert_allclose(action[[0, 2]], dense[[0, 2]], rtol=1e-10)


def test_conditioning_check_looks_at_the_epoch_held_until_absorption(monkeypatch):
    """The conditioning check measures the epoch held until absorption. A short first epoch whose rates span more
    than double precision passes and gives the values of a nearby well-conditioned rate, while such a last epoch
    raises on both moment paths. Regression: the check read only the first epoch, rejecting the former and passing
    the latter, whose variance came out negative."""
    def short_first(m0):
        return pg.Coalescent(n={'a': 1, 'b': 1}, demography=pg.Demography(
            pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): {0: m0, 0.01: 1}, ('b', 'a'): {0: m0, 0.01: 1}}))

    reference, extreme = short_first(0), short_first(1e-20)
    assert extreme.tree_height.mean == pytest.approx(reference.tree_height.mean, rel=1e-12)
    assert extreme.tree_height.var == pytest.approx(reference.tree_height.var, rel=1e-12)
    assert extreme.tree_height.quantile(0.5) == pytest.approx(reference.tree_height.quantile(0.5), rel=1e-6)

    def extreme_last():
        return pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(
            pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): {0: 1, 1: 1e-20}, ('b', 'a'): {0: 1, 1: 1e-20}}))

    with pytest.raises(ModelError, match='ill-conditioned'):
        _ = extreme_last().tree_height.mean

    monkeypatch.setattr(pg.Settings, 'closed_form_last_epoch', False)

    with pytest.raises(ModelError, match='ill-conditioned'):
        _ = extreme_last().tree_height.var


@pytest.mark.parametrize('n, expected', [
    ({'a': 1, 'b': 1}, (1.634575250341490265, 5.2903057214446315199)),
    ({'a': 2, 'b': 2}, (2.2672003781939281323, 6.5911361633084092669))
])
@pytest.mark.parametrize('closed_form', [True, False])
def test_conditioning_check_looks_at_every_finite_epoch(n, expected, closed_form, monkeypatch):
    """A finite epoch whose rates span more than double precision over its duration raises on both moment paths, the
    first as a later one, while a spread within it gives the reference moments, evaluated from the same rate
    matrices with 80-digit mpmath matrix exponentials. Their rates span 1e15, which leaves a relative error of about
    2e-8 in double precision. Regression: only the epoch held until absorption was checked, and a first epoch with a
    pairwise coalescence rate of 1e18 gave a mean of 66.6 and a negative variance for two lineages per deme."""
    def coal(sizes):
        return pg.Coalescent(n=n, demography=pg.Demography(
            pop_sizes={'a': {0: 1}, 'b': sizes}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))

    monkeypatch.setattr(pg.Settings, 'closed_form_last_epoch', closed_form)

    for sizes in [{0: 1e-18, 1: 1.5}, {0: 1, 0.5: 1e-18, 1: 1.5}]:
        with pytest.raises(ModelError, match='ill-conditioned'):
            _ = coal(sizes).tree_height.mean

    th = coal({0: 1e-15, 1: 1.5}).tree_height
    assert th.mean == pytest.approx(expected[0], rel=1e-7)
    assert th.var == pytest.approx(expected[1], rel=1e-7)


def test_stability_warning_ignores_the_exit_rates_of_absorbing_states(caplog):
    """The rate-spread warning measures the transient states. Here the transient rates span a factor of three while
    the single lineage in the slow deme leaves at 1e-12. Regression: the exit rates of the absorbing states were
    counted, and the warning fired on an exact result."""
    c = pg.Coalescent(n={'a': 2, 'b': 0}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1e-12}))

    with caplog.at_level('WARNING'):
        _ = c.tree_height.mean

    assert not [r for r in caplog.records if 'orders of magnitude' in r.getMessage()]


def test_tree_height_quantile_behind_a_migration_barrier():
    """Behind a migration barrier the CDF is exactly zero up to the epoch start ``T = 100``, and the lower-tail
    quantiles resolve its rise after ``T``. Regression: the locating octaves were geometric from zero, the excess
    over ``T`` was underestimated by up to 99.9%, and ``cdf(quantile(1e-6))`` was 8.5e-9."""
    c = pg.Coalescent(n={'pop_0': 1, 'pop_1': 1}, demography=pg.Demography(
        pop_sizes={'pop_0': 1, 'pop_1': 1},
        migration_rates={('pop_0', 'pop_1'): {0: 0, 100: 1}, ('pop_1', 'pop_0'): {0: 0, 100: 1}}))

    for q in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.5]:
        assert c.tree_height.cdf(c.tree_height.quantile(q)) == pytest.approx(q, rel=1e-4)


def test_tree_height_quantile_behind_a_migration_barrier_followed_by_a_change():
    """Behind a migration barrier that ends at ``T = 100`` and is followed by a change of the migration rate at
    ``T + 1``, the lower-tail quantiles resolve the rise after ``T``. Regression: only the last epoch start before the
    first positive probe was tested for a zero CDF, and ``cdf(quantile(1e-6))`` was 1.4e-8."""
    c = pg.Coalescent(n={'pop_0': 1, 'pop_1': 1}, demography=pg.Demography(
        pop_sizes={'pop_0': 1, 'pop_1': 1},
        migration_rates={('pop_0', 'pop_1'): {0: 0, 100: 1, 101: 0.1}, ('pop_1', 'pop_0'): {0: 0, 100: 1, 101: 0.1}}))

    for q in [1e-6, 1e-4, 1e-2, 0.5]:
        assert c.tree_height.cdf(c.tree_height.quantile(q)) == pytest.approx(q, rel=1e-4)


def test_sfs_mean_under_overflowing_growth_matches_stopped_growth():
    """Growth by a factor of e^600 over two time units drives the coalescence rates to 1e248, so every lineage
    coalesces within the first epochs, and the SFS mean equals that of growth stopped at 0.4. Regression: the solve of
    the occupation times raised a plain ValueError of scipy, and then scipy.linalg.expm squared 2^31 times once the
    1-norm of an epoch's generator passed about 1e38."""
    def sfs(end_time):
        return pg.Coalescent(n=5, demography=pg.Demography([pg.ExponentialPopSizeChanges(
            initial_size={'pop_0': 1}, growth_rate={'pop_0': 300}, start_time={'pop_0': 0},
            end_time={'pop_0': end_time})])).sfs

    np.testing.assert_allclose(sfs(2).mean.data, sfs(0.4).mean.data, rtol=1e-12)


def test_accumulate_from_beyond_absorption_raises():
    """Accumulating to absorption from a start time beyond almost sure absorption raises as ``moment`` does.
    Regression: a start time of 1e100 or infinity returned NaN."""
    coal = pg.Coalescent(n=4)

    for start in (1e100, np.inf):
        with pytest.raises(ValueError, match="beyond the time of almost sure absorption"):
            coal.tree_height.accumulate(1, [np.inf], start_time=start)

        with pytest.raises(ValueError, match="beyond the time of almost sure absorption"):
            coal.sfs.accumulate(1, [np.inf], start_time=start)


def test_tree_height_quantile_after_a_size_drop_to_a_negligible_size():
    """A size drop to 1e-100 absorbs every lineage at the drop, so the median lies at the drop time. Regression: the
    propagator of a grid segment overflowed to NaN and the quantile was NaN."""
    c = pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 1e-100}}))

    assert c.tree_height.cdf(1.0) < 0.5
    assert c.tree_height.quantile(0.5) == pytest.approx(1.0, rel=1e-6)


def test_isolated_epoch_after_almost_sure_absorption_is_not_held(monkeypatch):
    """A finite epoch of isolated demes that starts after almost sure absorption and is followed by migration is not
    held until absorption, from which it cannot absorb. The transform then evaluates, and the closed-form moments
    agree with the windowed ones. Regression: the isolated epoch was held and the transform raised that the
    demography does not absorb."""
    def iso():
        return pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(
            pop_sizes={'a': 1, 'b': 1},
            migration_rates={('a', 'b'): {0: 1, 1000: 0, 1010: 1}, ('b', 'a'): {0: 1, 1000: 0, 1010: 1}}))

    connected = pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(
        pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))

    assert iso().total_branch_length.cdf(3.0) == pytest.approx(connected.total_branch_length.cdf(3.0), rel=1e-6)

    mean = iso().tree_height.mean
    monkeypatch.setattr(pg.Settings, 'closed_form_last_epoch', False)

    assert iso().tree_height.mean == pytest.approx(mean, rel=1e-10)


def test_tree_height_cdf_and_pdf_vanish_below_zero():
    """The tree-height CDF and density are zero at negative times, as those of every other distribution.
    Regression: they raised a ValueError."""
    th = pg.Coalescent(n=3).tree_height

    assert th.cdf(-1.0) == 0.0
    assert th.cdf(-np.inf) == 0.0
    np.testing.assert_array_equal(th.pdf(np.array([-1.0, -0.5])), [0.0, 0.0])
    assert th.pdf(np.array([-1.0, 1.0]))[1] == pytest.approx(th.pdf(1.0), rel=1e-14)


def test_plot_accumulation_rejects_a_non_integral_order():
    """The accumulation plot of a phase-type distribution validates the order as ``accumulate`` does. Regression: it
    truncated ``k=1.5`` and ``k=True`` to the first moment."""
    th = pg.Coalescent(n=3).tree_height

    with pytest.raises(ValueError, match='must be an integer'):
        th.plot_accumulation(k=1.5, show=False)

    with pytest.raises(TypeError, match='must be an integer'):
        th.plot_accumulation(k=True, show=False)


def test_accumulated_reward_density_rejects_unknown_keywords():
    """The density of an accumulated reward takes no keyword besides the point. Regression: any keyword was
    swallowed."""
    c = pg.Coalescent(n=3)

    with pytest.raises(TypeError):
        c.total_branch_length.pdf(1.0, foo=1)

    with pytest.raises(TypeError):
        c.distribution(pg.TotalBranchLengthReward()).pdf(1.0, foo=1)


@pytest.mark.parametrize("start", [0.5, 1.5])
def test_windowed_moments_opening_past_an_epoch_boundary(monkeypatch, start):
    """A moment window that opens after one or two epoch boundaries agrees across the finite-end Van Loan walk, the
    closed-form infinite end, and the sparse action of both, and its mean is the difference of the accumulations
    from zero."""
    def coal():
        return pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.3: 2, 1.0: 0.5}}))

    def moments():
        th = coal().tree_height
        return [th.moment(k, start_time=start, end_time=end, center=False) for k in (1, 2) for end in (1e3, None)]

    th = coal().tree_height
    acc = th.accumulate(1, [start, np.inf], start_time=0.0)
    dense = moments()

    assert dense[0] == pytest.approx(acc[1] - acc[0], rel=1e-10)
    np.testing.assert_allclose(dense[::2], dense[1::2], rtol=1e-10)

    monkeypatch.setattr(pg.Settings, 'expm_action_min_dim', 0)
    monkeypatch.setattr(pg.Settings, 'closed_form_sparse_min_states', 1)

    np.testing.assert_allclose(moments(), dense, rtol=1e-8)
