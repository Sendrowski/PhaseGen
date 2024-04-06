"""
Test comparison.
"""

from unittest import TestCase

import pytest

from phasegen.comparison import Comparison


class ComparisonTestCase(TestCase):
    """
    Test comparison.
    """

    @staticmethod
    def test_1_epoch_n_2_test_size():
        """
        Test simple comparison.
        """
        c = Comparison.from_yaml("resources/configs/1_epoch_n_2_test_size.yaml")

        # touch msprime stats to cache them
        c.ms.touch()

        # drop simulated data
        c.ms.drop()

        c.compare(title="1_epoch_n_2_test_size")

    @staticmethod
    @pytest.mark.skip("not needed")
    def test_3_epoch_beta_migration_disparate_migration_sizes_2_each_n_6_early_end_time():
        """
        Test simple comparison.
        """
        c = Comparison.from_yaml(
            "../resources/configs/3_epoch_beta_migration_disparate_migration_sizes_2_each_n_6_early_end_time.yaml")

        c.parallelize = False
        c.num_replicates = 10000

        # touch msprime stats to cache them
        c.ms.touch()

        # drop simulated data
        c.ms.drop()

        c.compare(title="3_epoch_beta_migration_disparate_migration_sizes_2_each_n_6_early_end_time")

    @staticmethod
    @pytest.mark.skip(reason="takes too long, using cached scenarios instead")
    def test_1_epoch_n_10():
        """
        Test simple comparison.
        """
        c = Comparison.from_yaml("../resources/configs/1_epoch_n_10.yaml")

        # touch msprime stats to cache them
        c.ms.touch()

        # drop simulated data
        c.ms.drop()

        c.compare(title="1_epoch_n_10")
