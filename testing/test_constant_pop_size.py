import PH
from PH import Simulator, ConstantPopSizeCoalescent, BetaCoalescent, StandardCoalescent
from custom_test_case import CustomTestCase


class ConstantPopSizeTestCase(CustomTestCase):

    def test_moments_height_standard_coalscent_n_2(self):
        s = Simulator(
            n=2,
            pop_sizes=[1],
            times=[0]
        )

        s.simulate_ph()

        assert s.ph['height']['mu'] == 1
        assert s.ph['height']['var'] == 1

    def test_moments_height_standard_coalescent(self):
        s = Simulator(
            n=4,
            num_replicates=100000,
            pop_sizes=[1],
            times=[0]
        )

        s.simulate()

        assert self.diff_rel_max_abs(s.msprime['height']['mu'], s.ph['height']['mu']) < 0.01
        assert self.diff_rel_max_abs(s.msprime['height']['var'], s.ph['height']['var']) < 0.05

    def test_moments_height_standard_coalescent_low_pop_size(self):
        s = Simulator(
            n=4,
            num_replicates=100000,
            pop_sizes=[0.001],
            times=[0]
        )

        s.simulate()

        assert self.diff_rel_max_abs(s.msprime['height']['mu'], s.ph['height']['mu']) < 0.01
        assert self.diff_rel_max_abs(s.msprime['height']['var'], s.ph['height']['var']) < 0.05

    def test_moments_height_standard_coalescent_high_pop_size(self):
        s = Simulator(
            n=4,
            num_replicates=100000,
            pop_sizes=[1000],
            times=[0]
        )

        s.simulate()

        assert self.diff_rel_max_abs(s.msprime['height']['mu'], s.ph['height']['mu']) < 0.01
        assert self.diff_rel_max_abs(s.msprime['height']['var'], s.ph['height']['var']) < 0.05

    def test_beta_coalescent_model(self):
        S1 = ConstantPopSizeCoalescent.get_rate_matrix(10, StandardCoalescent())
        S2 = ConstantPopSizeCoalescent.get_rate_matrix(10, BetaCoalescent(alpha=1.99999999))

        assert self.diff_max_abs(S1, S2) < 1e-6

    def test_sfs(self):
        PH.set_precision(50)

        cd = ConstantPopSizeCoalescent(
            n=20,
            model=BetaCoalescent(alpha=1.5)
        )

        cd.sfs(theta=2)
