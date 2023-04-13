import numpy as np

from PH import Comparison
from custom_test_case import CustomTestCase, add_method_name_as_title


class VariablePopSizeTestCase(CustomTestCase):

    @add_method_name_as_title
    def test_plot_pdf_tree_height(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 1000)
        s.ph.tree_height.plot_pdf(x=x, clear=False, show=False, label='PH')
        s.msprime.tree_height.plot_pdf(x=x, clear=False, label='msprime')

    @add_method_name_as_title
    def test_plot_cdf_tree_height(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 100)
        s.ph.tree_height.plot_cdf(x=x, clear=False, show=False, label='PH')
        s.msprime.tree_height.plot_cdf(x=x, clear=False, label='msprime')

    @add_method_name_as_title
    def test_plot_pdf_total_branch_length(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 1000)
        s.ph.total_branch_length.plot_pdf(x=x, clear=False, show=False, label='PH')
        s.msprime.total_branch_length.plot_pdf(x=x, clear=False, label='msprime')

    @add_method_name_as_title
    def test_plot_cdf_total_branch_length(self):
        s = Comparison(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 100)
        s.ph.total_branch_length.plot_cdf(x=x, clear=False, show=False, label='PH')
        s.msprime.total_branch_length.plot_cdf(x=x, clear=False, label='msprime')

    @add_method_name_as_title
    def test_plot_pdf_total_branch_length_larger_n(self):
        s = Comparison(
            n=5,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=1000000
        )

        x = np.linspace(0, 2, 1000)
        s.ph.total_branch_length.plot_pdf(x=x, clear=False, show=False, label='PH')
        s.msprime.total_branch_length.plot_pdf(x=x, clear=False, label='msprime')

    @add_method_name_as_title
    def test_plot_cdf_total_branch_length_larger_n(self):
        s = Comparison(
            n=5,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        x = np.linspace(0, 2, 100)
        s.ph.total_branch_length.plot_cdf(x=x, clear=False, show=False, label='PH')
        s.msprime.total_branch_length.plot_cdf(x=x, clear=False, label='msprime')
