from PH import Simulator
from custom_test_case import CustomTestCase


class VariablePopSizeTestCase(CustomTestCase):

    def test_moments_height_scenario_1(self):
        s = Simulator(
            n=2,
            pop_sizes=[1, 0.00000001],
            times=[0, 1],
            num_replicates=100000
        )

        s.simulate()

        assert self.diff_rel_max_abs(s.msprime['height']['mu'], s.ph['height']['mu']) < 0.01
        assert self.diff_rel_max_abs(s.msprime['height']['var'], s.ph['height']['var']) < 0.01

    def test_moments_height_scenario_2(self):
        s = Simulator(
            n=2,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=100000
        )

        s.simulate()

        assert self.diff_rel_max_abs(s.msprime['height']['mu'], s.ph['height']['mu']) < 0.02
        assert self.diff_rel_max_abs(s.msprime['height']['var'], s.ph['height']['var']) < 0.05

    def test_moments_height_scenario_larger_n(self):
        s = Simulator(
            n=10,
            pop_sizes=[1.2, 10, 0.8, 10],
            times=[0, 0.3, 1, 1.4],
            num_replicates=1000000
        )

        s.simulate()

        assert self.diff_rel_max_abs(s.msprime['height']['mu'], s.ph['height']['mu']) < 0.01
        assert self.diff_rel_max_abs(s.msprime['height']['var'], s.ph['height']['var']) < 0.05
