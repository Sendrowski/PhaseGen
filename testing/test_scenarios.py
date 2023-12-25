import os
from pathlib import Path
from typing import List
from unittest import TestCase

from phasegen.comparison import Comparison

# configs = get_filenames("resources/configs")

configs = [
    '1_epoch_migration_one_each_n_6',
    '1_epoch_n_2_test_size',
    '1_epoch_migration_one_each_n_2',
    '1_epoch_migration_disparate_migration_sizes_2_each_n_6',
    '1_epoch_n_4',
    '5_epoch_varying_migration_2_pops',
    '5_epoch_beta_varying_migration_2_pops',
    '4_epoch_up_down_n_2',
    '4_epoch_up_down_n_10',
    '3_epoch_migration_disparate_migration_sizes_2_each_n_6',
    '3_epoch_extreme_bottleneck_n_5',
    '3_epoch_beta_migration_disparate_migration_sizes_2_each_n_6',
    '3_epoch_2_pops_n_5',
    '2_epoch_varying_migration_low_coalescence',
    '2_epoch_varying_migration_barrier',
    '2_epoch_rapid_decline_n_5',
    '2_epoch_rapid_decline_n_2',
    '2_epoch_n_5',
    '2_epoch_n_2',
    '2_epoch_2_pops_n_5',
    '1_epoch_n_2',
    '1_epoch_migration_disparate_pop_size_one_each_n_2',
    '1_epoch_migration_disparate_pop_size_one_all_n_2',
    '1_epoch_dirac_n_6_psi_1_c_1',
    '1_epoch_dirac_n_6_psi_0_5_c_0',
    '1_epoch_dirac_n_5_psi_1_c_50',
    '1_epoch_dirac_n_2_psi_0_5_c_0',
    '1_epoch_beta_n_6_alpha_1_7',
    '1_epoch_beta_n_6_alpha_1_1',
]

configs_suspended = [
    '1_epoch_dirac_n_2_psi_0_5_c_1',  # TODO shorter SFS bins than msprime
    '1_epoch_dirac_n_6_psi_0_5_c_50',  # TODO shorter SFS bins than msprime
    '1_epoch_dirac_n_6_psi_0_5_c_1',  # TODO shorter SFS bins than msprime
    '1_epoch_beta_n_2_alpha_1_9',  # TODO disagrees with Msprime for large values of alpha
    '1_epoch_beta_n_2_alpha_1_999',  # TODO disagrees with Msprime for large values of alpha
    '1_epoch_beta_n_6_alpha_1_9',  # TODO disagrees with Msprime for large values of alpha
    '1_epoch_beta_n_6_alpha_1_999',  # TODO disagrees with Msprime for large values of alpha
]


class ScenariosTestCase(TestCase):
    """
    Test scenarios.
    """
    pass


def get_filenames(path) -> List[str]:
    """
    Get all filenames in the given directory.

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
