"""
Test Demography class.
"""

from itertools import islice
from testing import TestCase

import numpy as np
import pytest
from numpy import testing

import phasegen as pg


class DemographyTestCase(TestCase):
    """
    Test Demography class.
    """

    def test_pop_size_change(self):
        """
        Test creating pop size change.
        """
        e = pg.PopSizeChanges({'pop_0': {0: 0.1, 1.2: 1}, 'pop_1': {0: 0.3, 1.2: 2}, 'pop_2': {0: 0.5, 1.3: 3}})

        np.testing.assert_array_equal(e.times, [0.0, 1.2, 1.3])
        self.assertDictEqual(e.pop_sizes[0], {'pop_0': 0.1, 'pop_1': 0.3, 'pop_2': 0.5})
        self.assertDictEqual(e.pop_sizes[1.2], {'pop_0': 1, 'pop_1': 2})
        self.assertDictEqual(e.pop_sizes[1.3], {'pop_2': 3})

        self.assertEqual(e.start_time, 0)

        epoch = pg.Epoch(start_time=0, end_time=0, pop_sizes={'pop_0': 0.3, 'pop_3': 2})
        e._apply(epoch)
        # doesn't work because 0 times are removed
        # self.assertDictEqual(epoch.pop_sizes, {'pop_0': 0.1, 'pop_1': 0.3, 'pop_2': 0.5, 'pop_3': 2})

        epoch = pg.Epoch(start_time=1.2, end_time=2, pop_sizes={'pop_0': 0.3, 'pop_3': 2})
        e._apply(epoch)
        self.assertDictEqual(epoch.pop_sizes, {'pop_0': 1, 'pop_1': 2, 'pop_2': 3, 'pop_3': 2})

        epoch = pg.Epoch(start_time=1.2, end_time=1.2, pop_sizes={'pop_0': 0.3, 'pop_3': 2})
        e._apply(epoch)
        self.assertDictEqual(epoch.pop_sizes, {'pop_0': 0.3, 'pop_3': 2})

        epoch = pg.Epoch(start_time=0, end_time=np.inf, pop_sizes={'pop_0': 0.3, 'pop_3': 2})
        e._apply(epoch)
        self.assertDictEqual(epoch.pop_sizes, {'pop_0': 1, 'pop_1': 2, 'pop_2': 3, 'pop_3': 2})

    def test_create_demography_from_rate_changes(self):
        """
        Test creating demography.
        """
        d = pg.Demography(events=[
            pg.DiscreteRateChanges(pop_sizes={
                'pop_0': {0: 0.1, 1.2: 1},
                'pop_1': {0: 0.3, 1.2: 2},
                'pop_2': {0: 0.5, 1.3: 3}}
            ),
            pg.DiscreteRateChanges(pop_sizes={'pop_0': {4: 5}}),
        ])

        epochs = list(islice(d.epochs, 10))

        self.assertEqual((epochs[0].start_time, epochs[0].end_time), (0, 1.2))
        self.assertEqual((epochs[1].start_time, epochs[1].end_time), (1.2, 1.3))
        self.assertEqual((epochs[2].start_time, epochs[2].end_time), (1.3, 4))
        self.assertEqual((epochs[3].start_time, epochs[3].end_time), (4, np.inf))

    def test_plot_discrete_demography(self):
        """
        Test plotting discrete demography.
        """
        d = pg.Demography(events=[
            pg.DiscreteRateChanges(pop_sizes={
                'pop_0': {0: 0.1, 1.2: 1},
                'pop_1': {0: 0.3, 1.2: 2},
                'pop_2': {0: 0.5, 1.3: 3}}
            ),
            pg.DiscreteRateChanges(pop_sizes={'pop_0': {1.8: 0.2}})
        ])

        d.plot_pop_sizes(t=np.linspace(0, 2, 200))

        pass

    def test_plot_demography_exponential_growth(self):
        """
        Test creating demography.
        """
        d = pg.Demography(
            events=[pg.ExponentialPopSizeChanges(
                initial_size={'pop_0': 1.5},
                growth_rate=0.1,
                start_time=0.1,
                end_time=9)
            ]
        )

        d.plot_pop_sizes()

        pass

    def test_plot_complex_demography(self):
        """
        Test plotting complex demography.
        """
        d = pg.Demography(
            events=[
                pg.PopSizeChanges({'pop_0': {0: 0.1, 1.2: 1}, 'pop_1': {0: 0.3, 1.2: 2}, 'pop_2': {0: 0.5, 1.3: 3}}),
                pg.DiscreteRateChanges(pop_sizes={'pop_0': {4: 5}}),
                pg.ExponentialPopSizeChanges(initial_size={'pop_0': 1.5}, growth_rate=0.1, start_time=0.1, end_time=9),
                pg.ExponentialPopSizeChanges(initial_size={'pop_0': 0.9}, growth_rate=3, start_time=0.2, end_time=3),
                pg.ExponentialPopSizeChanges(initial_size={'pop_0': 1.2}, growth_rate=-3, start_time=0.05, end_time=2),
            ]
        )

        d.plot_pop_sizes()

        epoch = list(islice(d.epochs, 100))

        pass

    @staticmethod
    def test_create_migration_rates_from_dicts_piecewise_constant_demography():
        """
        Test creating migration rates from dicts for piecewise constant demography.
        """
        d = pg.Demography(events=[
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1}, b={0: 1}),
                migration_rates={
                    ('a', 'b'): {0: 0.1, 1: 0.2},
                    ('b', 'a'): {0: 0.3, 1.5: 0.4},
                }
            )
        ])

        np.testing.assert_array_equal([e.start_time for e in islice(d.epochs, 3)], [0, 1, 1.5])
        np.testing.assert_array_equal([e.end_time for e in islice(d.epochs, 3)], [1, 1.5, np.inf])

        np.testing.assert_array_equal([e.pop_sizes['a'] for e in islice(d.epochs, 3)], [1, 1, 1])
        np.testing.assert_array_equal([e.pop_sizes['b'] for e in islice(d.epochs, 3)], [1, 1, 1])

        np.testing.assert_array_equal([e.migration_rates[('a', 'b')] for e in islice(d.epochs, 3)], [0.1, 0.2, 0.2])
        np.testing.assert_array_equal([e.migration_rates[('a', 'a')] for e in islice(d.epochs, 3)], [0, 0, 0])
        np.testing.assert_array_equal([e.migration_rates[('b', 'a')] for e in islice(d.epochs, 3)], [0.3, 0.3, 0.4])
        np.testing.assert_array_equal([e.migration_rates[('b', 'b')] for e in islice(d.epochs, 3)], [0, 0, 0])

    @staticmethod
    def test_piecewise_constant_demography_plot_pop_sizes():
        """
        Test plotting pop sizes of piecewise constant demography.
        """
        d = pg.Demography(events=[
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 6, 2: 3.5})
            )
        ])

        d.plot_pop_sizes()

    @staticmethod
    def test_piecewise_constant_demography_plot_migration_rates():
        """
        Test plotting migration rates of piecewise constant demography.
        """
        d = pg.Demography(events=[
            pg.DiscreteRateChanges(
                migration_rates={('a', 'b'): {0: 0.1, 1: 0.2}, ('a', 'c'): {0: 0.3, 1.5: 0.4}, ('b', 'a'): {0: 0.5}}
            )
        ])

        d.plot_migration()

    def test_piecewise_constant_demography_raises_value_error_pop_sizes_lower_than_zero(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 6, 2: -3.5})
            )

        self.assertEqual(str(error.exception), "Population sizes must be finite and positive at all times.")

    def test_piecewise_constant_demography_raises_value_error_pop_sizes_zero(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 0, 2: 3.5})
            )

        self.assertEqual(str(error.exception), "Population sizes must be finite and positive at all times.")

    def test_piecewise_constant_demography_raises_value_error_negative_migration_rate(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 6, 2: 3.5}),
                migration_rates={
                    ('a', 'b'): {0: 0.1, 1: 0.2},
                    ('b', 'a'): {0: 0.3, 1.5: -0.4},
                    ('c', 'a'): {0: 0.5}
                }
            )

        self.assertEqual(str(error.exception), "Migration rates must be finite and non-negative at all times.")

    def test_piecewise_constant_demography_raises_value_error_negative_times(self):
        """
        Test piecewise constant demography raises ValueError if times is negative.
        """
        with self.assertRaises(ValueError) as error:
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 6, 2: 3.5}),
                migration_rates={
                    ('a', 'b'): {0: 0.1, -1: 0.2},
                    ('b', 'a'): {0: 0.3, 1.5: 0.4},
                    ('c', 'a'): {0: 0.5}
                }
            )

        self.assertEqual(str(error.exception), "All times must not be negative.")

    def test_piecewise_constant_demography_to_msprime(self):
        """
        Test converting piecewise constant demography to msprime.
        """
        d = pg.Demography([
            pg.ExponentialPopSizeChanges(initial_size={'a': 1, 'b': 2}, growth_rate=0.1, start_time=0.1, end_time=9),
            pg.ExponentialRateChanges(
                initial_rate={('a', 'b'): 1., ('b', 'a'): 2.},
                growth_rate=0.1,
                start_time=0,
                end_time=9
            )
        ])

        d_msprime = d.to_msprime()

        self.assertEqual(2, d_msprime.num_populations)
        self.assertEqual(d.pop_names, ['a', 'b'])
        testing.assert_array_almost_equal(
            d_msprime.migration_matrix, np.array([[0, 0.995025], [1.99005, 0]]),
            decimal=6
        )
        self.assertEqual(d.pop_names, [pop.name for pop in d_msprime.populations])

    @pytest.mark.skip(reason="deprecated")
    def test_passing_different_pop_names_to_demography_and_n_lineages_raises_value_error(self):
        """
        Test passing different population names to demography and n_lineages raises ValueError.
        """
        with self.assertRaises(ValueError) as error:
            pg.Coalescent(
                demography=pg.Demography([
                    pg.DiscreteRateChanges(
                        pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}),
                    )
                ]),
                n=dict(c=1, d=2)
            )

            print(error)

    def test_demography_pop_size_at_zero_time_defaults_to_one(self):
        """
        Test that population size at time 0 defaults to 1.
        """
        d = pg.Demography([
            pg.DiscreteRateChanges(
                pop_sizes=dict(
                    a={1: 2},
                    b={1: 4, 2: 5, 3: 6},
                    c={0: 7, 2: 8, 3: 9},
                    d={0.000000001: 10},
                    e={0: 0.1}
                )
            )
        ])

        epoch = next(d.epochs)

        self.assertEqual(epoch.pop_sizes['a'], 1)
        self.assertEqual(epoch.pop_sizes['b'], 1)
        self.assertEqual(epoch.pop_sizes['c'], 7)
        self.assertEqual(epoch.pop_sizes['d'], 1)
        self.assertEqual(epoch.pop_sizes['e'], 0.1)

    def test_bug_demography(self):
        """
        Test that population size at time 0 defaults to 1.
        """
        d = pg.Demography(
            pop_sizes={
                'pop_1': {0: 1.2, 5: 0.1, 5.1: 0.8},
                'pop_0': {0: 1.0}
            },
            migration_rates={
                ('pop_0', 'pop_1'): {0: 0.2, 5: 0.3},
                ('pop_1', 'pop_0'): {0: 0.5}
            },
            warn_n_epochs=4
        )

        self.assertEqual(0.1, d.get_epoch(5).pop_sizes['pop_1'])

        pass

    def test_demography_equivalent(self):
        """
        Test that two demographies are equivalent.
        """
        d1 = pg.Demography(
            pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 2.5, 1: 0.8}},
            migration_rates={
                ('pop_0', 'pop_1'): {0: 1.7, 0.7: 2},
                ('pop_1', 'pop_0'): {0: 3}
            }
        )

        d2 = pg.Demography()

        d2.add_event(pg.PopSizeChange(pop='pop_0', time=0, size=1))
        d2.add_event(pg.PopSizeChange(pop='pop_1', time=0, size=2.5))
        d2.add_event(pg.PopSizeChange(pop='pop_1', time=1, size=0.8))

        d2.add_event(pg.MigrationRateChange(source='pop_0', dest='pop_1', time=0, rate=1.7))
        d2.add_event(pg.MigrationRateChange(source='pop_0', dest='pop_1', time=0.7, rate=2))
        d2.add_event(pg.MigrationRateChange(source='pop_1', dest='pop_0', time=0, rate=3))

        for epoch1, epoch2 in zip(d1.epochs, d2.epochs):
            self.assertEqual(epoch1, epoch2)

    def test_to_demes(self):
        """
        Test converting a demography without migration to a demes graph.
        """
        d = pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 0.5}, 'pop_1': {0: 2.5}})

        graph = d._to_demes()

        self.assertEqual(sorted(d.pop_names), sorted(deme.name for deme in graph.demes))

    def test_to_demes_with_migration(self):
        """
        Migration rates of at most 1 convert, one migration record per direction.
        """
        d = pg.Demography(
            pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 2.5}},
            migration_rates={('pop_0', 'pop_1'): 0.3, ('pop_1', 'pop_0'): 0.6}
        )

        graph = d._to_demes()

        self.assertEqual(sorted([0.3, 0.6]), sorted(m.rate for m in graph.migrations))

    def test_to_demes_with_migration_raises(self):
        """
        demes accepts migration rates of at most 1, so msprime's conversion rejects larger rates as an invalid
        migration, and the wrapper surfaces that rather than returning a graph.
        """
        d = pg.Demography(
            pop_sizes={'pop_0': {0: 1}, 'pop_1': {0: 2.5}},
            migration_rates={('pop_0', 'pop_1'): 1.7, ('pop_1', 'pop_0'): 3}
        )

        with self.assertRaises(ValueError):
            d._to_demes()

    def test_population_split(self):
        """
        Test population split.
        """
        coal = pg.Coalescent(
            n={'pop_0': 4, 'pop_1': 4},
            demography=pg.Demography(
                pop_sizes={'pop_0': 1, 'pop_1': 3},
                events=[
                    pg.PopulationSplit(
                        derived='pop_0',
                        ancestral='pop_1',
                        time=2
                    )
                ]
            )
        )

        self.assertEqual(coal.demography.get_epoch(1).pop_sizes['pop_0'], 1)
        self.assertEqual(coal.demography.get_epoch(1).pop_sizes['pop_1'], 3)

        self.assertEqual(coal.demography.get_epoch(2).pop_sizes['pop_0'], 1)
        self.assertEqual(coal.demography.get_epoch(2).pop_sizes['pop_1'], 3)

        self.assertEqual(coal.demography.get_epoch(1).migration_rates[('pop_1', 'pop_0')], 0)
        self.assertEqual(coal.demography.get_epoch(1).migration_rates[('pop_0', 'pop_1')], 0)

        self.assertEqual(coal.demography.get_epoch(2).migration_rates[('pop_1', 'pop_0')], 0)
        self.assertEqual(coal.demography.get_epoch(2).migration_rates[('pop_0', 'pop_1')], 100)

    def test_population_split_pools_into_ancestral(self):
        """
        Backward in time, lineages must pool into the *ancestral* deme (regression for the reversed split
        direction). With derived size 1 and ancestral size 3, E[T_mrca] for two lineages reflects coalescence in
        the size-3 ancestral deme (2 + 3 = 5), not the size-1 derived deme (2 + 1 = 3).
        """
        coal = pg.Coalescent(
            n={'pop_0': 1, 'pop_1': 1},
            demography=pg.Demography(
                pop_sizes={'pop_0': 1, 'pop_1': 3},
                events=[pg.PopulationSplit(derived='pop_0', ancestral='pop_1', time=2)]
            )
        )

        self.assertAlmostEqual(coal.tree_height.mean, 5, delta=0.1)

    def test_get_epochs_unsorted_times_returns_enclosing_epochs(self):
        """
        ``get_epochs`` must return, for each query time, the epoch enclosing it, regardless of input order
        (regression for the inverse-permutation bug when restoring the original order of unsorted times).
        """
        d = pg.Demography(pop_sizes={'pop_0': {0: 1, 1: 2, 3: 3, 5: 4}})
        times = [5.5, 0.5, 3.5, 1.5]
        epochs = d.get_epochs(times)

        for t, epoch in zip(times, epochs):
            self.assertTrue(epoch.start_time <= t < epoch.end_time)

    def test_epoch_to_string_two_pops_migration(self):
        """
        Test epoch to string.
        """
        epoch = pg.Epoch(
            start_time=1 / 6,
            end_time=2,
            pop_sizes={'pop_0': 1.1134, 'pop_1': 2.22},
            migration_rates={('pop_0', 'pop_1'): 0.1, ('pop_1', 'pop_0'): 0.2}
        )

        # make sure migration rates are not included in the string when only one population is present
        self.assertEqual(
            (
                "Epoch(start_time=0.1667, end_time=2, pop_sizes=(pop_0=1.113, pop_1=2.22), "
                "migration_rates=(pop_0->pop_1=0.1, pop_1->pop_0=0.2)"
            ),
            str(epoch)
        )

    def test_epoch_to_string_one_pop(self):
        """
        Test epoch to string.
        """
        epoch = pg.Epoch(
            start_time=1 / 6,
            end_time=2,
            pop_sizes={'pop_0': 1.11}
        )

        # make sure migration rates are not included in the string when only one population is present
        self.assertEqual("Epoch(start_time=0.1667, end_time=2, pop_sizes=(pop_0=1.11)", str(epoch))

    def test_overlapping_discretized_events_keep_finer_grid_boundaries(self):
        """
        Two overlapping ExponentialPopSizeChanges with different step sizes (0.1 on pop_0, 0.25 on pop_1)
        over [0, 1] must retain the fine (0.1) grid boundaries, not just the coarse (0.25) grid
        (regression for DiscretizedRateChange._broadcast overwriting end_time unconditionally instead of
        taking the minimum).
        """
        d = pg.Demography(events=[
            pg.ExponentialPopSizeChanges(
                initial_size={'pop_0': 1}, growth_rate={'pop_0': 1}, start_time=0, end_time=1, step_size=0.1
            ),
            pg.ExponentialPopSizeChanges(
                initial_size={'pop_1': 1}, growth_rate={'pop_1': 1}, start_time=0, end_time=1, step_size=0.25
            ),
        ])

        boundaries = sorted({e.start_time for e in islice(d.epochs, 20)})

        # the fine 0.1 grid boundaries must be present, not erased by the coarse 0.25 grid
        for t in (0.1, 0.2, 0.3):
            self.assertTrue(
                any(abs(b - t) < 1e-9 for b in boundaries),
                msg=f"missing fine-grid boundary at {t}: {boundaries}"
            )

        # pop_0 at t=0.15 must match the value when the step-0.1 event runs alone (midpoint over [0.1, 0.2]),
        # not the coarse midpoint over [0, 0.25] (pre-fix value 0.8894)
        d_alone = pg.Demography(events=[
            pg.ExponentialPopSizeChanges(
                initial_size={'pop_0': 1}, growth_rate={'pop_0': 1}, start_time=0, end_time=1, step_size=0.1
            ),
        ])

        self.assertAlmostEqual(d.get_epoch(0.15).pop_sizes['pop_0'], 0.8617840855569707, places=12)
        self.assertAlmostEqual(
            d.get_epoch(0.15).pop_sizes['pop_0'],
            d_alone.get_epoch(0.15).pop_sizes['pop_0'],
            places=12
        )

    def test_discretized_event_end_time_not_multiple_of_step_size(self):
        """
        A single discretized event whose end_time (0.35) is not a multiple of its step_size (0.1) must place a
        boundary at 0.35 and carry the correct discretized trajectory value over the partial final epoch
        [0.3, 0.35), rather than reusing the previous epoch's rate (regression for _broadcast overshooting the
        clamped end_time).
        """
        d = pg.Demography(events=[
            pg.ExponentialPopSizeChanges(
                initial_size={'pop_0': 1}, growth_rate={'pop_0': 1}, start_time=0, end_time=0.35, step_size=0.1
            ),
        ])

        epochs = list(islice(d.epochs, 10))

        # a boundary must be placed at the clamped end_time 0.35
        boundaries = sorted({e.start_time for e in epochs} | {e.end_time for e in epochs})
        self.assertTrue(
            any(abs(b - 0.35) < 1e-9 for b in boundaries),
            msg=f"missing boundary at 0.35: {boundaries}"
        )

        # the partial final epoch [0.3, 0.35) must exist with the correct discretized trajectory value
        partial = next(e for e in epochs if abs(e.start_time - 0.3) < 1e-9)
        self.assertAlmostEqual(partial.end_time, 0.35, places=12)

        # discretized midpoint over [0.3, 0.35), not the previous epoch's reused rate (pre-fix value 0.7798)
        self.assertAlmostEqual(d.get_epoch(0.32).pop_sizes['pop_0'], 0.7227531552002157, places=12)

    def test_demography_pop_size_input_format(self):
        """
        Test Demography pop_sizes input formats.
        """
        d = pg.Demography(pop_sizes={0: 1, 2: 2})

        epochs = list(d.epochs)
        self.assertEqual(epochs[0].pop_sizes['pop_0'], 1)
        self.assertEqual(epochs[1].pop_sizes['pop_0'], 2)

        d = pg.Demography(pop_sizes=2)
        self.assertEqual(next(d.epochs).pop_sizes['pop_0'], 2)

        d = pg.Demography(pop_sizes={'pop_A': 3, 'pop_B': 1})
        epochs = list(d.epochs)
        self.assertEqual(epochs[0].pop_sizes['pop_A'], 3)
        self.assertEqual(epochs[0].pop_sizes['pop_B'], 1)

        d = pg.Demography(pop_sizes={'pop_A': {0: 1, 1: 2}, 'pop_B': {1: 3}})
        epochs = list(d.epochs)
        self.assertEqual(epochs[0].pop_sizes['pop_A'], 1)
        self.assertEqual(epochs[0].pop_sizes['pop_B'], 1)
        self.assertEqual(epochs[1].pop_sizes['pop_A'], 2)
        self.assertEqual(epochs[1].pop_sizes['pop_B'], 3)


@pytest.mark.parametrize("order", [0, 1])
def test_chained_population_splits_at_same_time_independent_of_order(order):
    """
    Two splits at the same time chaining ``a -> b -> c`` were applied in list order, and a later split zeroed the
    drain an earlier split had set into its derived population. With ``[a -> b, b -> c]`` the rate ``a -> b`` was 0,
    lineages in ``a`` could never leave, and the tree height raised a non-absorption error.
    """
    events = [pg.PopulationSplit(1, 'a', 'b'), pg.PopulationSplit(1, 'b', 'c')]
    d = pg.Demography(pop_sizes={'a': 1., 'b': 1., 'c': 1.}, events=events[::1 - 2 * order])

    rates = list(islice(d.epochs, 2))[1].migration_rates
    assert {k: v for k, v in rates.items() if v} == {('a', 'b'): 100.0, ('b', 'c'): 100.0}

    th = pg.Coalescent(n={'a': 1, 'b': 0, 'c': 1}, demography=d).tree_height.mean
    assert th == pytest.approx(2.02, abs=1e-6)


@pytest.mark.parametrize("order", [0, 1])
def test_population_split_uses_same_time_pop_size_change(order):
    """
    The drain rate of a split was computed from the derived population size at the moment the split was applied, so
    a PopSizeChange of that population at the same time counted only when listed before the split, leaving the rate
    at the value implied by the pre-change size.
    """
    events = [pg.PopulationSplit(1, 'b', 'c'), pg.PopSizeChange('b', time=1, size=10)]
    d = pg.Demography(pop_sizes={'b': 1., 'c': 100.}, events=events[::1 - 2 * order])

    assert list(islice(d.epochs, 2))[1].migration_rates[('b', 'c')] == 10


@pytest.mark.parametrize("kwargs, reference", [
    (dict(pop_sizes=dict(zip(['a', 'b'], np.array([1, 2])))), dict(pop_sizes={'a': 1, 'b': 2})),
    (dict(pop_sizes=np.int64(2)), dict(pop_sizes=2)),
    (dict(pop_sizes=np.float32(2)), dict(pop_sizes=2.0)),
    (dict(pop_sizes={0: np.float32(1), np.int64(1): np.int64(2)}), dict(pop_sizes={0: 1, 1: 2})),
    (
            dict(pop_sizes={'a': 1., 'b': 1.}, migration_rates={('a', 'b'): np.int64(1), ('b', 'a'): np.int64(1)}),
            dict(pop_sizes={'a': 1., 'b': 1.}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1})
    ),
])
def test_demography_accepts_numpy_scalars(kwargs, reference):
    """
    The shorthand forms of ``pop_sizes`` and ``migration_rates`` were detected with ``isinstance(x, (float, int))``,
    which numpy scalars other than ``np.float64`` fail, so they were misread as nested time dictionaries and crashed.
    """
    epochs = list(islice(pg.Demography(**kwargs).epochs, 3))
    expected = list(islice(pg.Demography(**reference).epochs, 3))

    assert len(epochs) == len(expected)

    for e, r in zip(epochs, expected):
        assert e.pop_sizes == r.pop_sizes
        assert e.migration_rates == r.migration_rates


def test_population_split_drain_rate_dominates_coalescence_at_any_population_size():
    """
    The drain rate of a split was ``N * multiplier`` while a pair coalesces at ``1 / N``, so the drain stopped
    dominating coalescence as ``N`` fell below one and lineages coalesced inside the drained derived population:
    with both demes of size ``N`` and both lineages sampled in the derived one, the model is a single population of
    size ``N``, yet ``E[T_MRCA] / N`` grew from 1.006 at ``N = 1`` to 2.19 at ``N = 0.01``.
    """
    scaled = []

    for N in (1.0, 0.1, 0.01):
        d = pg.Demography(
            pop_sizes={'a': {0: N}, 'b': {0: N}},
            events=[pg.PopulationSplit(time=0.5 * N, derived='a', ancestral='b')]
        )

        assert list(islice(d.epochs, 2))[1].migration_rates[('a', 'b')] == 100 / N

        scaled += [pg.Coalescent(n={'a': 2, 'b': 0}, demography=d).tree_height.mean / N]

    assert scaled[0] == pytest.approx(1, rel=0.01)
    assert scaled[1] == pytest.approx(scaled[0], rel=1e-8)
    assert scaled[2] == pytest.approx(scaled[0], rel=1e-8)


def test_population_split_drain_rate_dominates_the_fastest_coalescence_rate_of_the_epoch():
    """
    The drain rate of a split was ``multiplier / N_derived``, a fixed multiple of the coalescence rate of the derived
    population alone, so a split into a smaller ancestral population left the lineages in the drained derived
    population for a stretch that is long on the ancestral coalescent clock: with both lineages sampled in a derived
    population of size 100 and a split at time 0 into an ancestral population of size 0.01, the model is a single
    population of size 0.01, yet ``E[T_MRCA]`` came out as 1.5025, a factor of 150 too large. The preceding formula
    ``N_derived * multiplier`` failed in the opposite regime, and a rate taken from the two populations of the split
    alone failed for splits chained through an intermediate population of a larger size, where ``E[T_MRCA]`` of the
    chain below came out as 0.0250 rather than 0.01.
    """
    for N_derived, N_ancestral in ((100.0, 0.01), (0.01, 100.0), (1.0, 1.0)):
        d = pg.Demography(
            pop_sizes={'a': N_derived, 'b': N_ancestral},
            events=[pg.PopulationSplit(time=0, derived='a', ancestral='b')]
        )

        assert next(d.epochs).migration_rates[('a', 'b')] == 100 / min(N_derived, N_ancestral)

        mean = pg.Coalescent(n={'a': 2, 'b': 0}, demography=d).tree_height.mean

        assert mean == pytest.approx(N_ancestral, rel=0.02)

    chained = pg.Demography(
        pop_sizes={'a': 1.0, 'b': 1.0, 'c': 0.01},
        events=[pg.PopulationSplit(time=0, derived='a', ancestral='b'),
                pg.PopulationSplit(time=0, derived='b', ancestral='c')]
    )

    rates = next(chained.epochs).migration_rates
    assert {k: v for k, v in rates.items() if v} == {('a', 'b'): 10000.0, ('b', 'c'): 10000.0}

    mean = pg.Coalescent(n={'a': 2, 'b': 0, 'c': 0}, demography=chained).tree_height.mean

    assert mean == pytest.approx(0.01, rel=0.05)


@pytest.mark.parametrize('model, N', [
    (pg.StandardCoalescent(), 0.1),
    (pg.DiracCoalescent(psi=0.5, c=50), 1.0),
    (pg.DiracCoalescent(psi=0.5, c=1), 0.1),
    (pg.BetaCoalescent(alpha=1.5), 100.0),
])
def test_population_split_drain_time_is_a_fixed_fraction_of_the_pairwise_coalescence_time(model, N):
    """
    The drain rate of a split was ``multiplier / min(N)`` under every coalescent model, ignoring the model's time
    scale and the Dirac point-mass rate, so the drain was only 7.4 times faster than pairwise coalescence under
    Dirac(psi=0.5, c=50) at N = 1 and inflated split-demography moments by 4 to 11% under the Beta and Dirac
    coalescents. With one lineage in each of two demes of equal size and a split at time 0, the lineage in the
    derived deme leaves after a mean time ``1 / m`` and the pair then coalesces after a mean time ``tau(N) / lambda``,
    with ``lambda`` the pairwise merger rate and ``tau`` the model's time scale, so the tree height has the closed-form
    mean ``tau(N) / lambda * (1 + 1 / multiplier)``. The tolerance only absorbs the precision of the phase-type
    computation.
    """
    pair = model._get_timescale(N) / model._get_rate(b=2, k=2)

    d = pg.Demography(
        pop_sizes={'a': N, 'b': N},
        events=[pg.PopulationSplit(time=0, derived='a', ancestral='b')]
    )

    coal = pg.Coalescent(n={'a': 1, 'b': 1}, model=model, demography=d)

    assert coal.demography.get_epoch(0).migration_rates[('a', 'b')] == pytest.approx(100 / pair, rel=1e-12)
    assert coal.tree_height.mean == pytest.approx(pair * (1 + 1 / 100), rel=1e-8)


def test_exponential_growth_survives_a_serialization_round_trip():
    """An exponential rate change keeps its trajectory through to_json / from_json. Regression: the trajectory was a
    closure, which jsonpickle drops, so a coalescent saved before computing anything raised AttributeError on every
    quantity after loading."""
    dem = pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1}, growth_rate=0.5, start_time=0.2, end_time=1.0
    )])

    restored = pg.Coalescent.from_json(pg.Coalescent(n=3, demography=dem).to_json())

    assert restored.tree_height.mean == pytest.approx(pg.Coalescent(n=3, demography=dem).tree_height.mean, rel=1e-12)


@pytest.mark.parametrize('make', [
    lambda: pg.ExponentialPopSizeChanges(initial_size={'pop_0': 1}, growth_rate=0.5, start_time=0, step_size=-0.1),
    lambda: pg.ExponentialPopSizeChanges(initial_size={'pop_0': 1}, growth_rate=0.5, start_time=0, step_size=0),
    lambda: pg.PopulationSplit(1, 'a', 'b', multiplier=-5),
    lambda: pg.PopulationSplit(-1, 'a', 'b'),
    lambda: pg.Demography(pop_sizes={'pop_0': {0: 1, np.nan: 2}}),
    lambda: pg.Demography(pop_sizes={'pop_0': {0: np.nan}}),
])
def test_invalid_demographic_input_is_rejected_at_construction(make):
    """Non-positive step sizes, negative split parameters and NaN times or sizes raise. Regression: a negative step
    size hung epoch generation, a zero one raised ZeroDivisionError, a negative multiplier gave a tree height of 6e14,
    and a NaN change time was silently dropped."""
    with pytest.raises(ValueError):
        make()


def test_a_trajectory_reaching_a_negative_size_is_rejected():
    """A discretized trajectory that reaches a negative population size raises. Regression: it gave a negative
    expected tree height."""
    dem = pg.Demography(events=[pg.DiscretizedRateChange(
        trajectory=lambda t: 1 - t, start_time=0, end_time=3, pop='pop_0', step_size=0.1
    )])

    with pytest.raises(ValueError, match='negative'):
        _ = pg.Coalescent(n=3, demography=dem).tree_height.mean


def test_nan_start_time_is_rejected():
    """A NaN start time raises. Regression: it was treated as 0 and the unwindowed moment returned."""
    with pytest.raises(ValueError):
        _ = pg.Coalescent(n=3, start_time=np.nan).tree_height.mean


def test_lineage_configurations_differing_in_population_order_differ():
    """The order of the populations fixes the deme axis and the demes the unlinked lineages come from, so it is part
    of the configuration. Regression: the configurations compared equal and Inference reused a two-locus state space
    whose initial distribution belonged to the other order."""
    assert pg.LineageConfig({'a': 2, 'b': 1}) != pg.LineageConfig({'b': 1, 'a': 2})
    assert pg.LineageConfig({'a': 2, 'b': 1}) == pg.LineageConfig({'a': 2, 'b': 1})


def test_events_setting_one_rate_in_the_same_epoch_warn(caplog):
    """Two events that set the same population size in one epoch log a warning naming the population, since the one
    starting last silently took precedence. Regression: the same changes grouped differently gave different moments
    without notice."""
    dem = pg.Demography(events=[
        pg.PopSizeChanges(pop_sizes={'pop_0': {0: 1, 0.5: 2}}),
        pg.DiscretizedRateChange(trajectory=lambda t: 1 + t, start_time=0.2, end_time=1, pop='pop_0')
    ])

    with caplog.at_level('WARNING'):
        list(dem.epochs)

    assert any('pop_0' in r.getMessage() and 'takes precedence' in r.getMessage() for r in caplog.records)


def test_population_names_that_are_not_identifiers_simulate():
    """Population names msprime rejects are mapped to identifiers for the simulation. Regression: to_msprime raised
    for names such as 'pop-1'."""
    coal = pg.Coalescent(
        n={'pop-1': 2, 'pop 2': 1},
        demography=pg.Demography(pop_sizes={'pop-1': 1, 'pop 2': 1}, migration_rates={('pop-1', 'pop 2'): 1,
                                                                                    ('pop 2', 'pop-1'): 1})
    )
    ms = coal.to_msprime(num_replicates=20000, parallelize=False, seed=1)

    assert ms.tree_height.mean == pytest.approx(coal.tree_height.mean, rel=0.05)


@pytest.mark.parametrize('gap, simulated', [(0.02, 2.0306), (0.05, 2.0594), (0.5, 2.5111)])
def test_chained_splits_at_different_times_keep_draining(gap, simulated):
    """A split keeps draining its derived population after a later split isolates the ancestral one. Regression: the
    later split zeroed the earlier drain, stranding a fraction exp(-100 gap) of the lineages, so the tree height
    raised as non-absorbing or came out near 1e10. References from 100,000 msprime replicates (standard error about
    0.003)."""
    dem = pg.Demography(
        pop_sizes={'a': 1, 'b': 1, 'c': 1},
        events=[pg.PopulationSplit(1, 'a', 'b'), pg.PopulationSplit(1 + gap, 'b', 'c')]
    )

    mean = pg.Coalescent(n={'a': 1, 'b': 0, 'c': 1}, demography=dem).tree_height.mean

    assert mean == pytest.approx(simulated, abs=0.013)


def test_a_decaying_trajectory_underflowing_to_zero_converts_to_msprime():
    """An exponential decline whose size underflows to zero far out is valid input. Regression: the size check
    rejected the zero and to_msprime raised."""
    dem = pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1}, growth_rate=10, start_time=0, step_size=0.1
    )])

    assert dem.to_msprime() is not None


def test_a_zero_size_reached_by_the_exact_computation_raises_clearly():
    """A trajectory reaching a population size of zero is accepted for simulation, but the exact computation raises a
    clear ValueError where it reaches that epoch. Regression: it raised ZeroDivisionError from the rate-matrix builder
    or returned a NaN CDF."""
    dem = pg.Demography(events=[pg.DiscretizedRateChange(
        trajectory=lambda t: max(1 - t, 0), start_time=0, end_time=3, pop='pop_0', step_size=0.1
    )])
    coal = pg.Coalescent(n=3, demography=dem)

    for call in (lambda: coal.tree_height.mean, lambda: coal.tree_height.cdf(1.5), lambda: coal.sfs.mean):
        with pytest.raises(ValueError, match='needs a positive size'):
            call()


def test_discretized_epochs_advance_at_large_start_times():
    """Every epoch of a discretized change starting at 1e7 has positive length. Regression: once the start time
    exceeded about 2.1e6, rounding cancelled the offset in the step count and the epochs had zero length forever."""
    dem = pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1}, growth_rate=0.1, start_time=1e7, step_size=0.1
    )])

    epochs = list(islice(dem.epochs, 50))

    assert all(e.end_time > e.start_time for e in epochs)
    assert epochs[-1].start_time > 1e7


def test_to_msprime_warns_when_truncating_epochs(caplog):
    """to_msprime keeps max_epochs epoch changes and warns when the demography has more."""
    dem = pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1}, growth_rate=0.1, start_time=0, end_time=3, step_size=0.1
    )])

    with caplog.at_level('WARNING'):
        dem.to_msprime(max_epochs=100)
    assert not any('more than' in r.getMessage() for r in caplog.records)

    with caplog.at_level('WARNING'):
        dem.to_msprime(max_epochs=10)
    assert any('more than 11 epochs' in r.getMessage() for r in caplog.records)


def test_cdf_and_pdf_at_infinity_on_infinitely_many_epochs():
    """The tree-height CDF is 1 and the density 0 at infinity on a demography with infinitely many epochs, and an
    epoch lookup at infinity or NaN raises. Regression: all of them never returned."""
    dem = pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1}, growth_rate=0.5, start_time=0, step_size=0.1
    )])
    coal = pg.Coalescent(n=3, demography=dem)

    np.testing.assert_array_equal(coal.tree_height.cdf([1.0, np.inf])[1:], [1.0])
    assert coal.tree_height.pdf(np.inf) == 0.0

    for t in (np.inf, np.nan, -1.0):
        with pytest.raises(ValueError):
            dem.get_epoch(t)


def test_to_json_restores_a_lambda_trajectory():
    """A coalescent with a lambda trajectory round-trips through JSON. Regression: the lambda was dropped silently and
    the restored coalescent raised AttributeError on its first uncached statistic."""
    def make():
        return pg.Coalescent(n=3, demography=pg.Demography(events=[pg.DiscretizedRateChange(
            trajectory=lambda t: 1 + t, start_time=0, end_time=1, pop='pop_0'
        )]))

    restored = pg.Coalescent.from_json(make().to_json())

    assert restored.tree_height.mean == pytest.approx(make().tree_height.mean, rel=1e-12)


def test_migration_rates_keyed_by_other_than_population_pairs_raise():
    """A migration rate keyed by other than a (source, destination) pair raises. Regression: the key 'ab' set no rate
    and the model ran without migration."""
    with pytest.raises(ValueError, match='pairs of population names'):
        pg.MigrationRateChanges({'ab': {0: 5}})


@pytest.mark.parametrize("size", [np.inf, np.nan])
def test_non_finite_population_size_raises(size):
    """A non-finite population size raises at construction. Regression: an infinite size passed and ended in a
    ZeroDivisionError in epoch 0, and in a zero-size reduction error in a later epoch."""
    with pytest.raises(ValueError, match='Population sizes must be finite and positive'):
        pg.Demography(pop_sizes={'pop_0': {0: 1, 1: size}})


@pytest.mark.parametrize("rate", [np.inf, np.nan])
def test_non_finite_migration_rate_raises(rate):
    """A non-finite migration rate raises at construction. Regression: an infinite rate passed and ended in an
    ill-conditioning error whose remedy could not work."""
    with pytest.raises(ValueError, match='Migration rates must be finite and non-negative'):
        pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): rate, ('b', 'a'): 1})


def test_non_finite_split_multiplier_raises():
    """An infinite migration rate multiplier of a population split raises at construction."""
    with pytest.raises(ValueError, match='positive and finite'):
        pg.PopulationSplit(time=1, derived='a', ancestral='b', multiplier=np.inf)


def test_non_finite_discretized_trajectory_raises():
    """A trajectory evaluating to an infinite population size raises where the epoch is set."""
    event = pg.DiscretizedRateChange(trajectory=lambda t: np.inf, start_time=0, end_time=1, pop='pop_0')
    demography = pg.Demography(events=[event])

    with pytest.raises(ValueError, match='negative or not finite'):
        list(islice(demography.epochs, 3))


def test_self_migration_raises():
    """A migration rate from a population to itself raises at construction. Regression: the exact path dropped the
    diagonal rate silently, while msprime rejected the same demography."""
    with pytest.raises(ValueError, match='distinct populations'):
        pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'a'): 1, ('a', 'b'): 1})

    with pytest.raises(ValueError, match='distinct populations'):
        pg.MigrationRateChange(source='a', dest='a', time=1, rate=1)

    with pytest.raises(ValueError, match='distinct populations'):
        pg.DiscretizedRateChange(trajectory=lambda t: 1, start_time=0, end_time=1, source='a', dest='a')


def test_split_onto_derived_population_raises():
    """A population split whose ancestral population is among the derived ones raises at construction. Regression:
    it passed and surfaced later as a non-absorption error."""
    with pytest.raises(ValueError, match='must not be among the derived'):
        pg.PopulationSplit(time=1, derived=['a', 'b'], ancestral='a')

    with pytest.raises(ValueError, match='must not be among the derived'):
        pg.PopulationSplit(time=1, derived='a', ancestral='a')
