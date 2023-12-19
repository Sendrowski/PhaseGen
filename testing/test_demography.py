import itertools
from itertools import islice
from unittest import TestCase

import numpy as np
from numpy import testing

import phasegen as pg


class DemographyTestCase(TestCase):
    """
    Test Demography class.
    """

    @staticmethod
    def test_create_migration_rates_from_arrays_piecewise_constant_demography():
        """
        Test creating migration rates from arrays for piecewise constant demography.
        """
        d = pg.PiecewiseConstantDemography(
            pop_sizes=[{0: 1}] * 2,
            migration_rates={
                0: np.array([[0, 0.1], [0.3, 0]]),
                1.5: np.array([[0, 0.3], [0.3, 0]]),
            }
        )

        np.testing.assert_array_equal(d._times, [0.0, 1.5])

        np.testing.assert_array_equal(d._pop_sizes['pop_0'], [1., 1.])
        np.testing.assert_array_equal(d._pop_sizes['pop_1'], [1., 1.])

        np.testing.assert_array_equal(d._migration_rates[('pop_0', 'pop_1')], [0.1, 0.3])
        np.testing.assert_array_equal(d._migration_rates[('pop_0', 'pop_0')], [0, 0])
        np.testing.assert_array_equal(d._migration_rates[('pop_1', 'pop_0')], [0.3, 0.3])
        np.testing.assert_array_equal(d._migration_rates[('pop_1', 'pop_1')], [0, 0])

    @staticmethod
    def test_create_migration_rates_from_dicts_piecewise_constant_demography():
        """
        Test creating migration rates from dicts for piecewise constant demography.
        """
        d = pg.PiecewiseConstantDemography(
            pop_sizes=dict(a={0: 1}, b={0: 1}),
            migration_rates={
                ('a', 'b'): {0: 0.1, 1: 0.2},
                ('b', 'a'): {0: 0.3, 1.5: 0.4},
            }
        )

        np.testing.assert_array_equal(d._times, [0.0, 1.0, 1.5])

        np.testing.assert_array_equal(d._pop_sizes['a'], [1., 1., 1.])
        np.testing.assert_array_equal(d._pop_sizes['b'], [1., 1., 1.])

        np.testing.assert_array_equal(d._migration_rates[('a', 'b')], [0.1, 0.2, 0.2])
        np.testing.assert_array_equal(d._migration_rates[('a', 'a')], [0, 0, 0])
        np.testing.assert_array_equal(d._migration_rates[('b', 'a')], [0.3, 0.3, 0.4])
        np.testing.assert_array_equal(d._migration_rates[('b', 'b')], [0, 0, 0])

    @staticmethod
    def test_piecewise_constant_demography_plot_pop_sizes():
        """
        Test plotting pop sizes of piecewise constant demography.
        """
        d = pg.PiecewiseConstantDemography(
            pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 6, 2: 3.5})
        )

        d.plot_pop_sizes()

    @staticmethod
    def test_piecewise_constant_demography_plot_migration_rates():
        """
        Test plotting migration rates of piecewise constant demography.
        """
        d = pg.PiecewiseConstantDemography(
            migration_rates={('a', 'b'): {0: 0.1, 1: 0.2}, ('a', 'c'): {0: 0.3, 1.5: 0.4}, ('b', 'a'): {0: 0.5}}
        )

        d.plot_migration()

    @staticmethod
    def test_constant_demography_from_dict():
        """
        Test creating constant demography from dict.
        """
        d = pg.ConstantDemography(
            pop_sizes=dict(a=1, b=2, c=3),
            migration_rates={('a', 'b'): 0.1, ('a', 'c'): 0.2, ('b', 'a'): 0.3}
        )

        np.testing.assert_array_equal(d._times, [0.0])

        np.testing.assert_array_equal(d._pop_sizes['a'], [1.])
        np.testing.assert_array_equal(d._pop_sizes['b'], [2.])
        np.testing.assert_array_equal(d._pop_sizes['c'], [3.])

        np.testing.assert_array_equal(d._migration_rates[('a', 'b')], [0.1])
        np.testing.assert_array_equal(d._migration_rates[('a', 'c')], [0.2])
        np.testing.assert_array_equal(d._migration_rates[('b', 'a')], [0.3])
        np.testing.assert_array_equal(d._migration_rates[('b', 'c')], [0])
        np.testing.assert_array_equal(d._migration_rates[('c', 'a')], [0])
        np.testing.assert_array_equal(d._migration_rates[('c', 'b')], [0])
        np.testing.assert_array_equal(d._migration_rates[('a', 'a')], [0])
        np.testing.assert_array_equal(d._migration_rates[('b', 'b')], [0])
        np.testing.assert_array_equal(d._migration_rates[('c', 'c')], [0])

    @staticmethod
    def test_constant_demography_from_array():
        """
        Test creating constant demography from array.
        """
        d = pg.ConstantDemography(
            pop_sizes=[1, 2],
            migration_rates=np.array([[0, 0.1], [0.3, 0]])
        )

        np.testing.assert_array_equal(d._times, [0.0])

        np.testing.assert_array_equal(d._pop_sizes['pop_0'], [1.])
        np.testing.assert_array_equal(d._pop_sizes['pop_1'], [2.])

        np.testing.assert_array_equal(d._migration_rates[('pop_0', 'pop_1')], [0.1])
        np.testing.assert_array_equal(d._migration_rates[('pop_0', 'pop_0')], [0])
        np.testing.assert_array_equal(d._migration_rates[('pop_1', 'pop_0')], [0.3])
        np.testing.assert_array_equal(d._migration_rates[('pop_1', 'pop_1')], [0])

    @staticmethod
    def test_constant_demography_from_scalar():
        """
        Test creating constant demography from scalar.
        """
        d = pg.ConstantDemography(
            pop_sizes=1
        )

        np.testing.assert_array_equal(d._times, [0.0])

        np.testing.assert_array_equal(d._pop_sizes['pop_0'], [1.])

        np.testing.assert_array_equal(d._migration_rates[('pop_0', 'pop_0')], [0.])

    @staticmethod
    def test_plot_exponential_demography_adaptive_step_size():
        """
        Test exponential demography.
        """
        d = pg.DiscretizedDemography(
            pop_sizes=pg.Demography.exponential_growth(x0=dict(a=1, b=2), growth_rate=dict(a=0.1, b=-0.2)),
            migration_rates=pg.Demography.exponential_growth(x0={('a', 'b'): 0.1}, growth_rate={('a', 'b'): 0.02})
        )

        d.plot()

        pass

    @staticmethod
    def test_exponential_demography_adaptive_step_size():
        """
        Test exponential demography.
        """
        d = pg.DiscretizedDemography(
            pop_sizes=pg.Demography.exponential_growth(x0=dict(a=10, b=20), growth_rate=dict(a=0.1, b=-2)),
            migration_rates=pg.Demography.exponential_growth(x0={('a', 'b'): 0.1}, growth_rate={('a', 'b'): 0.02})
        )

        times = list(islice(d.times, 5))
        pop_sizes = dict((p, list(islice(d.pop_sizes[p], 5))) for p in d.pop_names)
        m = dict((p, list(islice(d.migration_rates[p], 5))) for p in itertools.product(d.pop_names, repeat=2))

        np.testing.assert_array_equal(times, [0, 0.015625, 0.03125, 0.046875, 0.0625])

        np.testing.assert_array_almost_equal(pop_sizes['a'], [10, 9.97, 9.96, 9.95, 9.93], decimal=2)
        np.testing.assert_array_almost_equal(pop_sizes['b'], [20, 20.96, 21.63, 22.31, 23.02], decimal=2)

        np.testing.assert_array_almost_equal(m[('a', 'b')], [0.1, 0.09995, 0.09992, 0.09989, 0.09986], decimal=5)
        np.testing.assert_array_almost_equal(m[('b', 'a')], [0] * 5)
        np.testing.assert_array_almost_equal(m[('a', 'a')], [0] * 5)
        np.testing.assert_array_almost_equal(m[('b', 'b')], [0] * 5)

    def test_constant_demography_raises_value_error_pop_sizes_lower_than_zero(self):
        """
        Test constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.ConstantDemography(
                pop_sizes=dict(a=1, b=4, c=-2)
            )

        print(error.exception)

    def test_piecewise_constant_demography_raises_value_error_pop_sizes_lower_than_zero(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.PiecewiseConstantDemography(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 6, 2: -3.5})
            )

        self.assertEqual(str(error.exception), "Population sizes must be positive at all times.")

    def test_piecewise_constant_demography_raises_value_error_pop_sizes_zero(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.PiecewiseConstantDemography(
                pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}, c={0: 7, 1: 0, 2: 3.5})
            )

        self.assertEqual(str(error.exception), "Population sizes must be positive at all times.")

    def test_piecewise_constant_demography_raises_value_error_negative_migration_rate(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError) as error:
            pg.PiecewiseConstantDemography(
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
            pg.PiecewiseConstantDemography(
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
        d = pg.DiscretizedDemography(
            pop_sizes=pg.Demography.exponential_growth(x0=dict(a=10, b=20), growth_rate=dict(a=0.1, b=-2)),
            migration_rates=pg.Demography.exponential_growth(
                x0={('a', 'b'): 0.1, ('b', 'a'): 0.2},
                growth_rate={('a', 'b'): 0.02, ('b', 'a'): 0.3}
            )
        )

        d_msprime = d.to_msprime()

        self.assertEqual(2, d_msprime.num_populations)
        testing.assert_array_equal(d_msprime.migration_matrix, np.array([[0, 0.1], [0.2, 0]]))
        self.assertEqual(d.pop_names, [pop.name for pop in d_msprime.populations])

    def test_passing_different_pop_names_to_demography_and_n_lineages_raises_value_error(self):
        """
        Test passing different population names to demography and n_lineages raises ValueError.
        """
        with self.assertRaises(ValueError) as error:
            pg.Coalescent(
                demography=pg.PiecewiseConstantDemography(
                    pop_sizes=dict(a={0: 1, 1: 2, 2: 3}, b={0: 4, 1: 5, 2: 6}),
                ),
                n=dict(c=1, d=2)
            )

            print(error)
