"""
Test comparison.
"""

from unittest import TestCase

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

    def test_mutation_configs_test_size(self):
        """
        Test simple comparison for mutation configs.
        """
        for n in [2, 3]:
            c = Comparison(
                n=n,
                num_replicates=1000,
                mutation_rate=1,
                simulate_mutations=True,
                pop_sizes={'pop_0': {0: 1}},
                parallelize=False,
                seed=42,
                comparisons={'tolerance': {
                    'sfs': {'mutation_configs': 0.5},
                    'fsfs': {'mutation_configs': 0.5}
                }}
            )

            c.compare(title="mutation_config_test_size")
