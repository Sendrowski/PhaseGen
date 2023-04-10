import PH
from PH import Comparison, ConstantPopSizeCoalescent, BetaCoalescent, StandardCoalescent
from custom_test_case import CustomTestCase


class ConstantPopSizeTestCase(CustomTestCase):

    def assert_moments(self, s):
        assert self.diff_rel_max_abs(s.msprime.tree_height.mean, s.ph.tree_height.mean) < 0.01
        assert self.diff_rel_max_abs(s.msprime.tree_height.var, s.ph.tree_height.var) < 0.05
        assert self.diff_rel_max_abs(s.msprime.total_branch_length.mean, s.ph.total_branch_length.mean) < 0.01
        assert self.diff_rel_max_abs(s.msprime.total_branch_length.var, s.ph.total_branch_length.var) < 0.05

    def test_moments_height_standard_coalscent_n_2(self):
        s = ConstantPopSizeCoalescent(
            n=2,
            Ne=1
        )

        assert s.tree_height.mean == 1
        assert s.tree_height.var == 1

        assert s.total_branch_length.mean == 2
        assert s.total_branch_length.var == 4

    def test_moments_height_standard_coalescent(self):
        s = Comparison(
            n=4,
            num_replicates=100000,
            pop_sizes=[1],
            times=[0]
        )

        self.assert_moments(s)

    def test_moments_height_standard_coalescent_low_pop_size(self):
        s = Comparison(
            n=4,
            num_replicates=100000,
            pop_sizes=[0.001],
            times=[0]
        )

        self.assert_moments(s)

    def test_moments_height_standard_coalescent_high_pop_size(self):
        s = Comparison(
            n=4,
            num_replicates=100000,
            pop_sizes=[1000],
            times=[0]
        )

        self.assert_moments(s)

    def test_beta_coalescent_model_approaches_kingman(self):
        S1 = ConstantPopSizeCoalescent.get_rate_matrix(10, StandardCoalescent())
        S2 = ConstantPopSizeCoalescent.get_rate_matrix(10, BetaCoalescent(alpha=1.99999999))

        assert self.diff_max_abs(S1, S2) < 1e-6

    def test_sfs(self):
        PH.set_precision(50)

        cd = ConstantPopSizeCoalescent(
            n=20,
            model=BetaCoalescent(alpha=1.5)
        )

        cd.sfs(theta=2)

    def test_plot_f_tree_height(self):
        s = Comparison(
            n=4,
            num_replicates=10000,
            pop_sizes=[1],
            times=[0]
        )

        s.ph.tree_height.plot_f(show=False, label='PH var')
        s.ph_const.tree_height.plot_f(show=False, clear=False, label='PH const')
        s.msprime.tree_height.plot_f(clear=False, label='msprime')

    def test_plot_F_tree_height(self):
        s = Comparison(
            n=4,
            num_replicates=10000,
            pop_sizes=[1],
            times=[0]
        )

        s.ph.tree_height.plot_F(show=False, label='PH var')
        s.ph_const.tree_height.plot_F(show=False, clear=False, label='PH const')
        s.msprime.tree_height.plot_F(clear=False, label='msprime')

    def test_plot_F_tree_height_large_Ne(self):
        s = Comparison(
            n=4,
            num_replicates=10000,
            pop_sizes=[100],
            times=[0]
        )

        s.ph.tree_height.plot_F(show=False, label='PH var')
        s.ph_const.tree_height.plot_F(show=False, clear=False, label='PH const')
        s.msprime.tree_height.plot_F(clear=False, label='msprime')

    def test_plot_f_total_branch_length(self):
        s = Comparison(
            n=4,
            num_replicates=10000,
            pop_sizes=[1],
            times=[0]
        )

        s.ph.total_branch_length.plot_f(show=False, label='PH var')
        s.ph_const.total_branch_length.plot_f(show=False, clear=False, label='PH const')
        s.msprime.total_branch_length.plot_f(clear=False, label='msprime')

    def test_plot_F_total_branch_length(self):
        s = Comparison(
            n=4,
            num_replicates=10000,
            pop_sizes=[1],
            times=[0]
        )

        s.ph.total_branch_length.plot_F(show=False, label='PH var')
        s.ph_const.total_branch_length.plot_F(show=False, clear=False, label='PH const')
        s.msprime.total_branch_length.plot_F(clear=False, label='msprime')
