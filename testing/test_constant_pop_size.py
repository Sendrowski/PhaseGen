import logging

import PH
from PH import ConstantPopSizeCoalescent, BetaCoalescent, StandardCoalescent
from custom_test_case import CustomTestCase
from scripts.comp import diff_max_abs

logger = logging.getLogger()


class ConstantPopSizeTestCase(CustomTestCase):
    def test_moments_height_standard_coalscent_n_2(self):
        s = ConstantPopSizeCoalescent(
            n=2,
            Ne=1
        )

        assert s.tree_height.mean == 1
        assert s.tree_height.var == 1

        assert s.total_branch_length.mean == 2
        assert s.total_branch_length.var == 4

    def test_beta_coalescent_model_approaches_kingman(self):
        S1 = ConstantPopSizeCoalescent.get_rate_matrix(10, StandardCoalescent())
        S2 = ConstantPopSizeCoalescent.get_rate_matrix(10, BetaCoalescent(alpha=1.99999999))

        assert diff_max_abs(S1, S2) < 1e-6

    def test_sfs(self):
        PH.set_precision(50)

        cd = ConstantPopSizeCoalescent(
            n=20,
            model=BetaCoalescent(alpha=1.5)
        )

        cd.get_n_segregating(theta=2)
