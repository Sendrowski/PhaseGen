from unittest import TestCase

from phasegen.comparison import Comparison


class ScenariosTestCase(TestCase):
    """
    Test scenarios.
    """
    pass


configs = [
    '4_epoch_up_down_n_10',
    'standard_coalescent_ph_const_n_4',
    '2_epoch_n_2',
    'standard_coalescent_ph_n_4',
    'rapid_decline_n_2',
    'rapid_decline_n_5',
    '4_epoch_up_down_n_2',
    '3_epoch_extreme_bottleneck_n_5',
]
"""
"""


def generate_tests(config):
    def run_test(self):
        Comparison.from_file(f"../results/comparisons/serialized/{config}.json").compare()

    return run_test


for config in configs:
    setattr(ScenariosTestCase, f'test_{config}', generate_tests(config))
