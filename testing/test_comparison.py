from unittest import TestCase

from phasegen.comparison import Comparison


class ComparisonTestCase(TestCase):
    """
    Test comparison.
    """

    @staticmethod
    def test_simple_comparison():
        """
        Test simple comparison.
        """
        c = Comparison.from_yaml("../resources/configs/1_epoch_n_2_test_size.yaml")

        # touch msprime stats to cache them
        c.ms.touch()

        # drop simulated data
        c.ms.drop()

        c.compare()
