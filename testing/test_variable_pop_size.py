import numpy as np

from PH import Comparison
from custom_test_case import CustomTestCase


class VariablePopSizeTestCase(CustomTestCase):

    def assert_mean(self, s: Comparison, diff1: float, diff2: float):
        assert self.diff_rel_max_abs(s.msprime.tree_height.mean, s.ph.tree_height.mean) < diff1
        assert self.diff_rel_max_abs(s.msprime.tree_height.var, s.ph.tree_height.var) < diff2

    def test_moments_height_scenario_1(self):
        s = Comparison(
            n=2,
            pop_sizes=[1, 0.00000001],
            times=[0, 1],
            num_replicates=100000
        )

        self.assert_mean(s, 0.01, 0.01)

    def test_moments_height_scenario_2(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        self.assert_mean(s, 0.02, 0.05)

    def test_moments_height_scenario_larger_n(self):
        s = Comparison(
            n=10,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=1000000
        )

        self.assert_mean(s, 0.01, 0.05)

    def test_plot_f_tree_height(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 1000)
        s.ph.tree_height.plot_f(x=x, show=False, label='PH')
        s.msprime.tree_height.plot_f(x=x, clear=False, label='msprime')

    def test_plot_F_tree_height(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 100)
        s.ph.tree_height.plot_F(x=x, show=False, label='PH')
        s.msprime.tree_height.plot_F(x=x, clear=False, label='msprime')
