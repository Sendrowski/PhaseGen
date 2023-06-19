from unittest import TestCase

from phasegen import Comparison


class ScenariosTestCase(TestCase):
    pass


configs = [
    '2_epoch_n_2',
    'standard_coalescent_n_4',
    'rapid_decline_n_2',
    'rapid_decline_n_5',
    '4_epoch_up_down_n_10',
    '4_epoch_up_down_n_2'
]


def test_generator(config):
    def test(self):
        Comparison.from_file(f"../results/comparisons/serialized/{config}.json").compare()

    return test


for config in configs:
    test_name = f'test_{config}'
    test = test_generator(config)
    setattr(ScenariosTestCase, test_name, test)


def test_plot_pdf_total_branch_length(self):
    """
    Make sure an error is raised when comparing the total branch length density.
    """
    with self.assertRaises(NotImplementedError):
        self._compare(0.001, 'pdf', 'total_branch_length')


def test_plot_cdf_total_branch_length(self):
    """
    Make sure an error is raised when comparing the cumulative total branch length distribution.
    """
    with self.assertRaises(NotImplementedError):
        self._compare(0.001, 'cdf', 'total_branch_length')
