import os
from pathlib import Path
from typing import List
from unittest import TestCase

from phasegen.comparison import Comparison

# configs = get_filenames("resources/configs")

configs = [
    '1_epoch_2_loci_n_4_r_1',
]

configs_suspended = [
    '1_epoch_2_loci_n_2_r_1',
    '1_epoch_migration_disparate_migration_sizes_2_each_n_6',
    '1_epoch_migration_one_each_n_6',
    '2_epoch_n_5',
    '2_epoch_n_2',
    '2_epoch_varying_migration_low_coalescence',
    '2_epoch_rapid_decline_n_5',
    '2_epoch_rapid_decline_n_2',
    '1_epoch_2_loci_n_2_r_10',
    '1_epoch_2_loci_n_10_r_1',
    '1_epoch_2_loci_n_2_r_0_1',
    '1_epoch_2_loci_n_2_r_0',
    '1_epoch_2_loci_n_3_r_1',
    '2_epoch_2_pops_n_5',
    '2_epoch_varying_migration_barrier',
    '1_epoch_migration_zero_rates_n_6',
    '1_epoch_migration_one_each_n_2',
    '1_epoch_n_2',
    '1_epoch_n_2_test_size',
    '1_epoch_n_4',
    '5_epoch_varying_migration_2_pops',
    '5_epoch_beta_varying_migration_2_pops',
    '4_epoch_up_down_n_2',
    '4_epoch_up_down_n_10',
    '3_epoch_migration_disparate_migration_sizes_2_each_n_6',
    '3_epoch_extreme_bottleneck_n_5',
    '3_epoch_beta_migration_disparate_migration_sizes_2_each_n_6',
    '3_epoch_2_pops_n_5',
    '1_epoch_migration_disparate_pop_size_one_each_n_2',
    '1_epoch_migration_disparate_pop_size_one_all_n_2',
    '1_epoch_dirac_n_6_psi_1_c_1',
    '1_epoch_dirac_n_6_psi_0_5_c_0',
    '1_epoch_dirac_n_5_psi_1_c_50',
    '1_epoch_dirac_n_2_psi_0_5_c_0',
    '1_epoch_beta_n_6_alpha_1_7',
    '1_epoch_beta_n_6_alpha_1_1',
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
    #: Whether assert that compare statistics are within specified tolerance
    do_assertion: bool = True


def get_filenames(path) -> List[str]:
    """
    Get all filenames in the given directory.

    :param path: Path to directory
    :return: Filenames without extension
    """
    return [os.path.splitext(file.name)[0] for file in Path(path).glob('*') if file.is_file()]


def generate_tests(config: str):
    """
    Generate tests for the given config.

    :param config: Config name
    :return: Test function
    """
    def run_test(self):
        """
        Run test for the given config.
        """
        Comparison.from_file(f"../results/comparisons/serialized/{config}.json").compare(
            title=config,
            do_assertion=ScenariosTestCase.do_assertion
        )

    return run_test


for config in configs:
    setattr(ScenariosTestCase, f'test_{config}', generate_tests(config))
