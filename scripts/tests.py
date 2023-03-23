import unittest
from PH import *
from numpy import testing

class PhaseTypeTest(unittest.TestCase):
    def test_diagonalize(self):
        cd = CoalescentDistribution(StandardCoalescent(), n=10, alpha=CoalescentDistribution.e_i(10, 0))

        U, lam, U_inv = cd.diagonalize(cd.S)

        testing.assert_array_almost_equal(cd.S, U @ np.diag(lam) @ U_inv)

if __name__ == '__main__':
    unittest.main()
