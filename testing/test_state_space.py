"""
Test StateSpace class.
"""
import itertools
from collections import defaultdict
from unittest import TestCase

import numpy as np
import pytest
from numpy import testing

import phasegen as pg


class StateSpaceTestCase(TestCase):
    """
    Test StateSpace class.
    """

    @staticmethod
    def test_default_intensity_matrix_n_4():
        """
        Test default intensity matrix for n = 4.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=4),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        testing.assert_array_equal(s.S, np.array([[-6., 6., 0., 0.],
                                                  [0., -3., 3., 0.],
                                                  [0., 0., -1., 1.],
                                                  [0., 0., 0., -0.]]))

    @staticmethod
    def test_n_2_2_demes():
        """
        Test n = 2, 2 demes.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 2}
            )
        )

        np.testing.assert_array_equal(s.S, np.array([[-1., 0., 0., 1., 0.],
                                                     [0., -0., 0., 0., 0.],
                                                     [0., 0., -0.5, 0., 0.5],
                                                     [0., 0., 0., -0., 0.],
                                                     [0., 0., 0., 0., -0.]]))

    @staticmethod
    def test_2_loci_default_state_space_n_2():
        """
        Test two loci, n = 2.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        expected = np.array(
            [[-6., 1., 1., 0., 4., 0., 0., 0., 0.],
             [0., -3., 0., 1., 0., 2., 0., 0., 0.],
             [0., 0., -3., 1., 0., 0., 2., 0., 0.],
             [0., 0., 0., -0., 0., 0., 0., 0., 0.],
             [1.11, 0., 0., 0., -4.11, 1., 1., 0., 1.],
             [0., 1.11, 0., 0., 0., -2.11, 0., 1., 0.],
             [0., 0., 1.11, 0., 0., 0., -2.11, 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., -0., 0.],
             [0., 0., 0., 0., 2.22, 0., 0., 1., -3.22]]
        )

        np.testing.assert_array_almost_equal(s.S, expected)

    @staticmethod
    def test_2_loci_default_state_space_n_3():
        """
        Test two loci, n = 3.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        expected = np.array(
            [[-15., 3., 0., 3., 0., 0., 0., 0., 0., 9., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., -10., 1., 0., 3., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., -6., 0., 0., 3., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., -10., 3., 0., 1., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., -6., 1., 0., 1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., -3., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., -6., 3., 0., 0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., -3., 1., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1.11, 0., 0., 0., 0., 0., 0., 0., 0., -11.11, 3., 0., 3., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.],
             [0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -7.11, 1., 0., 3., 0., 0., 0., 0., 0., 2., 0., 0., 0.],
             [0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -4.11, 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -7.11, 3., 0., 1., 0., 0., 0., 0., 2., 0., 0.],
             [0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -4.11, 1., 0., 1., 0., 0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -2.11, 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -4.11, 3., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 1.11, 0., 0., 0., 0., 0., 0., 0., 0., -2.11, 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., 0., -8.22, 2., 2., 0., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., 0., -5.22, 0., 2., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., -5.22, 2., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.22, 0., 0., 0., 1., 0., 0., 0., -3.22, 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.33, 0., 0., 3., -6.33]]
        )

        np.testing.assert_array_almost_equal(s.S, expected)

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_2(self):
        """
        Test two loci, one deme, two lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_3(self):
        """
        Test two loci, one deme, two lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        _ = s.S

        pass

    @staticmethod
    def test_default_state_space_two_loci_one_deme_n_4():
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

        pass

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_4(self):
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

        pass

    @staticmethod
    def test_default_state_space_two_loci_two_demes_n_4():
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        _ = s.S

    @pytest.mark.skip(reason="recombination not implemented for block counting state space")
    def test_block_counting_state_space_two_loci_two_demes_n_4(self):
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        s._get_rate(223, 400)

        _ = s.S

    def test_default_state_space_size(self):
        """
        Test default state space size.
        """
        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(10)
        ).k, 10)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(20)
        ).k, 20)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 65)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 494)

        self.assertEqual(pg.DefaultStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 815)

    def test_block_counting_state_space_size(self):
        """
        Test block counting state space size.
        """
        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(10)
        ).k, 42)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(20)
        ).k, 627)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 481)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 2580)

        self.assertEqual(pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 35581)

    @pytest.mark.skip('Getting permission denied error since I fiddled with the permissions')
    def test_plot_rates(self):
        """
        Test plot rates.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=3),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/plot_rates', view=False)

    @pytest.mark.skip('Not needed anymore')
    def test_block_counting_state_space_n_4_dirac(self):
        """
        Test block counting state space for n = 4, dirac.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=4),
            model=pg.DiracCoalescent(c=50, psi=0.5),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/block_counting_state_space_n_4_dirac')

        pass

    @pytest.mark.skip('Not needed anymore')
    def test_block_counting_state_space_n_5_dirac(self):
        """
        Test block counting state space for n = 4, dirac.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=5),
            model=pg.DiracCoalescent(c=50, psi=0.5),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/block_counting_state_space_n_5_dirac')

        pass

    @pytest.mark.skip('Not needed anymore')
    def test_block_counting_state_space_n_4_dirac_psi_0_7_c_50(self):
        """
        Test block counting state space for n = 4, dirac.
        """
        s = pg.BlockCountingStateSpace(
            pop_config=pg.LineageConfig(n=4),
            model=pg.DiracCoalescent(c=50, psi=0.7),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/block_counting_state_space_n_4_dirac_psi_0_7_c_50')

        pass

    @pytest.mark.skip('Not a test')
    def test_default_state_space_beta_2_loci_n_3_alpha_1_5(self):
        """
        Test default state space for beta, n = 3, alpha = 1.5.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            model=pg.BetaCoalescent(alpha=1.5),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/default_state_space_beta_2_loci_n_3_alpha_1_5')

        pass

    @pytest.mark.skip('Not a test')
    def test_default_state_space_beta_2_loci_n_2_alpha_1_5(self):
        """
        Test default state space for beta, n = 2, alpha = 1.5.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            model=pg.BetaCoalescent(alpha=1.5, scale_time=False),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/default_state_space_beta_2_loci_n_2_alpha_1_5')

        pass

    @pytest.mark.skip('Not a test')
    def test_default_state_space_kingman_2_loci_n_2(self):
        """
        Test default state space for kingman, n = 2, alpha = 1.5.
        """
        s = pg.DefaultStateSpace(
            pop_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            epoch=pg.Epoch()
        )

        s._plot_rates('tmp/default_state_space_kingman_2_loci_n_2')

        pass

    def test_determine_state_space_size(self):
        """
        Test determine state space size.
        """
        size = defaultdict(dict)

        for n in range(2, 10):
            for d in range(1, 5):
                coal = pg.Coalescent(
                    n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)}),
                )

                size['default.observed'][(n, d)] = coal.default_state_space.k
                size['block_counting.observed'][(n, d)] = coal.block_counting_state_space.k

                size['default.theoretical'][(n, d)] = coal.default_state_space.get_k()
                size['block_counting.theoretical'][(n, d)] = coal.block_counting_state_space.get_k()

                self.assertEqual(size['default.observed'][(n, d)], size['default.theoretical'][(n, d)])
                self.assertEqual(size['block_counting.observed'][(n, d)], size['block_counting.theoretical'][(n, d)])

        pass

    def test_state_space_size_sequence_2_loci(self):
        """
        Test state space size sequence for 2 loci.
        """
        size = defaultdict(dict)

        # for one deme: a = lambda n: n*(2*n**2 + 9*n + 1)/6
        # https://oeis.org/search?q=9%2C+23%2C+46%2C+80%2C+127%2C+189%2C+268%2C+366&go=Search

        for n in range(2, 10):
            for d in range(1, 4):
                coal = pg.Coalescent(
                    n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)}),
                    loci=2
                )

                size['default.observed'][(n, d)] = coal.default_state_space.k

        pass

    def test_get_sequence(self):
        """
        Test get sequence.
        """
        for n in np.arange(3, 10):
            for d in np.arange(1, 4):
                x = np.array(list(itertools.product(np.arange(n + 1), repeat=d)))
                y = x[x.sum(axis=1) <= n]
                p = [1] + [pg.state_space.StateSpace.p0(i, d) for i in np.arange(1, n + 1)]

                n1 = len(y)
                n2 = sum(p)

                self.assertEqual(n1, n2)

                pass

    def test_equivalence_block_counting_state_space(self):
        """
        Make sure size of block counting state space is equivalent the number of partitions of n.
        """
        n = np.arange(1, 10)
        k = np.zeros((len(n), 2))

        for i in n:
            n1 = np.array(pg.state_space.BlockCountingStateSpace._find_sample_configs(m=i, n=i))
            n2 = pg.state_space.StateSpace.P(i)

            k[i - 1, 0] = len(n1)
            k[i - 1, 1] = n2

            self.assertEqual(len(n1), n2)

        pass
