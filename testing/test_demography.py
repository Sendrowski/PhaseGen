from itertools import islice
from unittest import TestCase

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
        self.assertEqual(e.end_time, 1.3)

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

        self.assertEqual(str(error.exception), "Population sizes must be positive at all times.")

    def test_piecewise_constant_demography_raises_value_error_pop_sizes_zero(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.DiscreteRateChanges(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 0, 2: 3.5})
            )

        self.assertEqual(str(error.exception), "Population sizes must be positive at all times.")

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

        self.assertEqual(str(error.exception), "Migration rates must not be negative at all times.")

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

        self.assertEqual(0.1, d.get_epochs(5).pop_sizes['pop_1'])

        pass
