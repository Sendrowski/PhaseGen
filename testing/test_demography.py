from itertools import islice
from unittest import TestCase

import numpy as np
from numpy import testing

import phasegen as pg


class DemographyTestCase(TestCase):
    """
    Test Demography class.
    """

    def test_exponential_demography(self):
        """
        Test exponential demography.
        """
        d = pg.ExponentialDemography(
            growth_rate=dict(a=0.1, b=-0.2, c=0),
            N0=dict(a=1, b=2, c=3)
        )

        times = list(islice(d.times, 3))
        pop_sizes = dict((p, list(islice(d.pop_sizes[p], 3))) for p in d.pop_names)

        testing.assert_array_equal(times, [0, 1, 1.5])
        testing.assert_array_equal(pop_sizes['a'], [1.0, 0.8617840855569707, 0.8197543797482314])
        testing.assert_array_equal(pop_sizes['b'], [2.0, 2.71322745580144, 2.9985800782761314])
        testing.assert_array_equal(pop_sizes['c'], [3.0, 3.0, 3.0])

    def test_plot_exponential_demography(self):
        """
        Test plotting exponential demography.
        """
        pg.ExponentialDemography(
            growth_rate=dict(a=0.1, b=-0.2, c=0),
            N0=dict(a=1, b=2, c=3)
        ).plot()

    def test_plot_piecewise_constant_demography(self):
        """
        Test plotting piecewise constant demography.
        """
        pg.PiecewiseTimeHomogeneousDemography(
            pop_sizes=dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 6, 3.5]),
            times=dict(a=[0, 1, 2], b=[0, 0.5, 1], c=[0, 0.25, 0.5])
        ).plot()

    def test_plot_constant_demography(self):
        """
        Test plotting constant demography.
        """
        pg.TimeHomogeneousDemography(
            pop_size=dict(a=1, b=4, c=2)
        ).plot()

    def test_piecewise_constant_demography_flatten_pop_sizes(self):
        """
        Test flattening piecewise constant demography.
        """
        times, pop_sizes = pg.PiecewiseTimeHomogeneousDemography.flatten(
            times=dict(a=[0, 1, 2], b=[0, 0.5, 1], c=[0, 0.25, 0.5]),
            pop_sizes=dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 6, 3.5])
        )

        testing.assert_array_equal(times, [0, 0.25, 0.5, 1, 2])
        testing.assert_array_equal(pop_sizes['a'], [1, 1, 1, 2, 3])
        testing.assert_array_equal(pop_sizes['b'], [4, 4, 5, 6, 6])
        testing.assert_array_equal(pop_sizes['c'], [7, 6, 3.5, 3.5, 3.5])

    def test_piecewise_constant_demography_generators(self):
        """
        Test generators of piecewise constant demography.
        """
        d = pg.PiecewiseTimeHomogeneousDemography(
            pop_sizes=dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 6, 3.5]),
            times=dict(a=[0, 1, 2], b=[0, 0.5, 1], c=[0, 0.25, 0.5])
        )

        times = list(islice(d.times, 5))
        pop_sizes = dict((p, list(islice(d.pop_sizes[p], 5))) for p in d.pop_names)

        testing.assert_array_equal(times, [0, 0.25, 0.5, 1, 2])
        testing.assert_array_equal(pop_sizes['a'], [1, 1, 1, 2, 3])
        testing.assert_array_equal(pop_sizes['b'], [4, 4, 5, 6, 6])
        testing.assert_array_equal(pop_sizes['c'], [7, 6, 3.5, 3.5, 3.5])

    def test_constant_demography_raises_value_error_pop_sizes_lower_than_zero(self):
        """
        Test constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError):
            pg.TimeHomogeneousDemography(
                pop_size=dict(a=1, b=4, c=-2)
            )

    def test_piecewise_constant_demography_raises_value_error_pop_sizes_lower_than_zero(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes is lower than zero.
        """
        with self.assertRaises(ValueError):
            pg.PiecewiseTimeHomogeneousDemography(
                pop_sizes=dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 6, -3.5]),
                times=dict(a=[0, 1, 2], b=[0, 0.5, 1], c=[0, 0.25, 0.5])
            )

    def test_piecewise_constant_demography_raises_value_different_dict_keys(self):
        """
        Test piecewise constant demography raises ValueError if pop_sizes and times have different dict keys.
        """
        with self.assertRaises(ValueError):
            pg.PiecewiseTimeHomogeneousDemography(
                pop_sizes=dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 6, 3.5]),
                times=dict(a=[0, 1, 2], b=[0, 0.5, 1])
            )

    def test_piecewise_constant_demography_from_lists_for_one_population(self):
        """
        Test piecewise constant demography from lists for one population.
        """
        d = pg.PiecewiseTimeHomogeneousDemography(
            times=[0, 1, 2],
            pop_sizes=[1, 2, 3]
        )

        times = list(islice(d.times, 3))
        pop_sizes = dict((p, list(islice(d.pop_sizes[p], 3))) for p in d.pop_names)

        testing.assert_array_equal(times, [0, 1, 2])
        testing.assert_array_equal(pop_sizes['pop_0'], [1, 2, 3])

    def test_constant_demography_from_lists_for_one_population(self):
        """
        Test constant demography from lists for one population.
        """
        d = pg.TimeHomogeneousDemography(
            pop_size=1
        )

        times = list(islice(d.times, 3))
        pop_sizes = dict((p, list(islice(d.pop_sizes[p], 3))) for p in d.pop_names)

        testing.assert_array_equal(times, [0])
        testing.assert_array_equal(pop_sizes['pop_0'], [1])

    def test_pass_migration_matrix(self):
        """
        Test passing migration matrix.
        """
        d = pg.TimeHomogeneousDemography(
            pop_size=dict(a=1, b=4, c=2),
            migration_matrix=np.arange(9).reshape(3, 3)
        )

        self.assertDictEqual(d.migration_rates, {('a', 'b'): 1, ('a', 'c'): 2,
                                                 ('b', 'a'): 3, ('b', 'c'): 5,
                                                 ('c', 'a'): 6, ('c', 'b'): 7})

    def test_time_homogenous_demography_to_msprime(self):
        """
        Test converting time homogeneous demography to msprime.
        """
        d = pg.TimeHomogeneousDemography(
            pop_size=dict(a=1, b=4),
            migration_matrix=np.arange(4).reshape(2, 2)
        )

        d_msprime = d.to_msprime()

        self.assertEqual(2, d_msprime.num_populations)
        testing.assert_array_equal(d.migration_matrix, d_msprime.migration_matrix)
        self.assertEqual(d.pop_names, [pop.name for pop in d_msprime.populations])

    def test_piecewise_time_homogenous_demography_to_msprime(self):
        """
        Test converting piecewise time homogeneous demography to msprime.
        """
        d = pg.PiecewiseTimeHomogeneousDemography(
            pop_sizes=dict(a=[1, 2, 3], b=[4, 5, 6]),
            times=dict(a=[0, 1, 2], b=[0, 0.5, 1]),
            migration_matrix=np.arange(4).reshape(2, 2)
        )

        d_msprime = d.to_msprime()

        self.assertEqual(2, d_msprime.num_populations)
        testing.assert_array_equal(d.migration_matrix, d_msprime.migration_matrix)
        self.assertEqual(d.pop_names, [pop.name for pop in d_msprime.populations])

        self.assertEqual(0.5, d_msprime.events[0].time)
        self.assertEqual(1, d_msprime.events[0].initial_size)
        self.assertEqual('a', d_msprime.events[0].population)

        self.assertEqual(0.5, d_msprime.events[1].time)
        self.assertEqual(5, d_msprime.events[1].initial_size)
        self.assertEqual('b', d_msprime.events[1].population)
