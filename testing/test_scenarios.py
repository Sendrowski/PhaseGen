import os
from pathlib import Path
from typing import List
from unittest import TestCase

from phasegen.comparison import Comparison

# configs = get_filenames("resources/configs")

configs = [
    '5_epoch_varying_migration_2_pops',
    '3_epoch_migration_disparate_migration_sizes_2_each_n_6',
    '4_epoch_up_down_n_2',
    '1_epoch_migration_disparate_migration_sizes_2_each_n_6',
    '1_epoch_migration_one_each_n_6',
    '4_epoch_up_down_n_10',
    '3_epoch_extreme_bottleneck_n_5',
    '2_epoch_rapid_decline_n_2',
    '1_epoch_migration_one_each_n_2',
    '2_epoch_n_2',
    '1_epoch_migration_disparate_pop_size_one_each_n_2',
    '1_epoch_n_4',
    '1_epoch_migration_disparate_pop_size_one_all_n_2',
    '2_epoch_rapid_decline_n_5',
    '2_epoch_varying_migration_barrier',
    '2_epoch_varying_migration_low_coalescence',
]

"""
"""


class ScenariosTestCase(TestCase):
    """
    Test scenarios.
    """
    pass


def get_filenames(path) -> List[str]:
    """
    Get all filenames in a directory.

    :param path: Path to directory
    :return: Filenames without extension
    """
    return [os.path.splitext(file.name)[0] for file in Path(path).glob('*') if file.is_file()]


def generate_tests(config):
    def run_test(self):
        Comparison.from_file(f"../results/comparisons/serialized/{config}.json").compare(title=config)

    return run_test


for config in configs:
    setattr(ScenariosTestCase, f'test_{config}', generate_tests(config))
