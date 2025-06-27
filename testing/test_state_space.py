"""
Test StateSpace class.
"""
import itertools
import sys
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
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
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
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
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
    def test_2_loci_lineage_counting_state_space_n_2():
        """
        Test two loci, n = 2.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
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
    def test_2_loci_lineage_counting_state_space_n_3():
        """
        Test two loci, n = 3.
        """
        s = pg.state_space_old.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
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

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_2(self):
        """
        Test two loci, one deme, two lineages.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_3(self):
        """
        Test two loci, one deme, two lineages.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11)
        )

        _ = s.S

        pass

    @staticmethod
    def test_lineage_counting_state_space_two_loci_one_deme_n_4():
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

        pass

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_block_counting_state_space_two_loci_one_deme_n_4(self):
        """
        Test two loci, one deme, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            locus_config=pg.LocusConfig(n=2)
        )

        _ = s.S

        pass

    @staticmethod
    def test_lineage_counting_state_space_two_loci_two_demes_n_4():
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        _ = s.S

    @pytest.mark.skip(reason="recombination not implemented for block-counting state space")
    def test_block_counting_state_space_two_loci_two_demes_n_4(self):
        """
        Test two loci, two demes, four lineages.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig([2, 2]),
            locus_config=pg.LocusConfig(n=2),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        )

        s._get_rate(223, 400)

        _ = s.S

    def test_lineage_counting_state_space_size(self):
        """
        Test lineage-counting state space size.
        """
        self.assertEqual(pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(10)
        ).k, 10)

        self.assertEqual(pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(20)
        ).k, 20)

        self.assertEqual(pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 65)

        self.assertEqual(pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 494)

        self.assertEqual(pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5, 'pop_2': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1})
        ).k, 815)

    def test_block_counting_state_space_size(self):
        """
        Test block-counting state space size.
        """
        self.assertEqual(pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(10)
        ).k, 42)

        self.assertEqual(pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(20)
        ).k, 627)

        self.assertEqual(pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1})
        ).k, 481)

        self.assertEqual(pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2, 'pop_3': 2}),
            epoch=pg.Epoch(pop_sizes={'pop_0': 1, 'pop_1': 1, 'pop_2': 1, 'pop_3': 1})
        ).k, 2580)

    def test_plot_rates(self):
        """
        Test plot rates.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/plot_rates', view=False)

    @pytest.mark.skip('Not needed anymore')
    def test_block_counting_state_space_n_4_dirac(self):
        """
        Test block-counting state space for n = 4, dirac.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            model=pg.DiracCoalescent(c=50, psi=0.5),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/block_counting_state_space_n_4_dirac')

        pass

    @pytest.mark.skip('Not needed anymore')
    def test_block_counting_state_space_n_5_dirac(self):
        """
        Test block-counting state space for n = 4, dirac.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=5),
            model=pg.DiracCoalescent(c=50, psi=0.5),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/block_counting_state_space_n_5_dirac')

        pass

    @pytest.mark.skip('Not needed anymore')
    def test_block_counting_state_space_n_4_dirac_psi_0_7_c_50(self):
        """
        Test block-counting state space for n = 4, dirac.
        """
        s = pg.BlockCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            model=pg.DiracCoalescent(c=50, psi=0.7),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/block_counting_state_space_n_4_dirac_psi_0_7_c_50')

        pass

    @pytest.mark.skip('Not a test')
    def test_lineage_counting_state_space_beta_2_loci_n_3_alpha_1_5(self):
        """
        Test lineage-counting state space for beta, n = 3, alpha = 1.5.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=3),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            model=pg.BetaCoalescent(alpha=1.5),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/lineage_counting_state_space_beta_2_loci_n_3_alpha_1_5')

        pass

    @pytest.mark.skip('Not a test')
    def test_lineage_counting_state_space_beta_2_loci_n_2_alpha_1_5(self):
        """
        Test lineage-counting state space for beta, n = 2, alpha = 1.5.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            model=pg.BetaCoalescent(alpha=1.5, scale_time=False),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/lineage_counting_state_space_beta_2_loci_n_2_alpha_1_5')

        pass

    @pytest.mark.skip('Not a test')
    def test_lineage_counting_state_space_kingman_2_loci_n_2(self):
        """
        Test lineage-counting state space for kingman, n = 2, alpha = 1.5.
        """
        s = pg.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=2),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            epoch=pg.Epoch()
        )

        s.plot_rates('scratch/lineage_counting_state_space_kingman_2_loci_n_2')

        pass

    def test_determine_state_space_size(self):
        """
        Test determine state space size.
        """
        size = defaultdict(dict)

        for n in range(2, 8):
            for d in range(1, 4):
                coal = pg.Coalescent(
                    n=pg.LineageConfig({'pop_0': n} | {f'pop_{i}': 0 for i in range(1, d)}),
                )

                size['lineage_counting.observed'][(n, d)] = coal.lineage_counting_state_space.k
                size['block_counting.observed'][(n, d)] = coal.block_counting_state_space.k

                size['lineage_counting.theoretical'][(n, d)] = coal.lineage_counting_state_space._get_old().get_k()
                size['block_counting.theoretical'][(n, d)] = coal.block_counting_state_space._get_old().get_k()

                self.assertEqual(size['lineage_counting.observed'][(n, d)], size['lineage_counting.theoretical'][(n, d)])
                self.assertEqual(size['block_counting.observed'][(n, d)], size['block_counting.theoretical'][(n, d)])

        pass

    def test_get_sequence(self):
        """
        Test get sequence.
        """
        for n in np.arange(3, 10):
            for d in np.arange(1, 4):
                x = np.array(list(itertools.product(np.arange(n + 1), repeat=d)))
                y = x[x.sum(axis=1) <= n]
                p = [1] + [pg.state_space_old.StateSpace.p0(i, d) for i in np.arange(1, n + 1)]

                n1 = len(y)
                n2 = sum(p)

                self.assertEqual(n1, n2)

                pass

    def test_equivalence_block_counting_state_space(self):
        """
        Make sure size of block-counting state space is equivalent the number of partitions of n.
        """
        n = np.arange(1, 10)
        k = np.zeros((len(n), 2))

        for i in n:
            n1 = np.array(pg.state_space_old.BlockCountingStateSpace._find_sample_configs(m=i, n=i))
            n2 = pg.state_space_old.StateSpace.P(i)

            k[i - 1, 0] = len(n1)
            k[i - 1, 1] = n2

            self.assertEqual(len(n1), n2)

        pass

    def compare_state_spaces(
            self,
            state_space_old: pg.state_space_old.StateSpace,
            state_space: pg.state_space.StateSpace,
            plot: bool = False
    ):
        """
        Compare state spaces.
        """
        if plot:
            state_space_old._plot_rates('scratch/state_space_old')
            state_space.plot_rates('scratch/state_space')

        self.assertEqual(state_space.k, state_space_old.k)

        # reorder the states of s2 to match s1
        ordering = [
            np.where(
                ((state_space_old.states == state_space.lineages[i]) & (
                        state_space_old.linked == state_space.linked[i])).all(
                    axis=(1, 2, 3)))[0][0] for i in range(state_space.k)
        ]

        testing.assert_array_equal(state_space.lineages.astype(int), state_space_old.states[ordering])
        testing.assert_array_equal(state_space.linked.astype(int), state_space_old.linked[ordering])

        testing.assert_array_almost_equal(state_space.S, state_space_old.S[ordering][:, ordering], decimal=14)
        print(f"graph: {state_space.time}, matrix: {state_space_old.time}")

    def test_equivalence_lineage_counting_state_space_standard_coalescent(self):
        """
        Test equivalence of lineage-counting state space and graph space for standard coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=10),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.LineageCountingStateSpace(**kwargs),
            pg.LineageCountingStateSpace(**kwargs)
        )

    def test_equivalence_block_counting_state_space_standard_coalescent(self):
        """
        Test equivalence of block-counting state space and graph space for standard coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=6),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.BlockCountingStateSpace(**kwargs),
            pg.BlockCountingStateSpace(**kwargs)
        )

    def test_equivalence_lineage_counting_state_space_beta_coalescent(self):
        """
        Test equivalence of lineage-counting state space and graph space for beta coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=10),
            model=pg.BetaCoalescent(alpha=1.5),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.LineageCountingStateSpace(**kwargs),
            pg.LineageCountingStateSpace(**kwargs)
        )

    def test_equivalence_block_counting_state_space_beta_coalescent(self):
        """
        Test equivalence of block-counting state space and graph space for beta coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=10),
            model=pg.BetaCoalescent(alpha=1.5, scale_time=False),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.BlockCountingStateSpace(**kwargs),
            pg.BlockCountingStateSpace(**kwargs)
        )

    def test_equivalence_lineage_counting_state_space_dirac_coalescent(self):
        """
        Test equivalence of lineage-counting state space and graph space for dirac coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=10),
            model=pg.DiracCoalescent(c=50, psi=0.5),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.LineageCountingStateSpace(**kwargs),
            pg.LineageCountingStateSpace(**kwargs)
        )

    def test_equivalence_block_counting_state_space_dirac_coalescent(self):
        """
        Test equivalence of block-counting state space and graph space for dirac coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=4),
            model=pg.DiracCoalescent(c=50, psi=0.5),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.BlockCountingStateSpace(**kwargs),
            pg.BlockCountingStateSpace(**kwargs)
        )

    def test_equivalence_lineage_counting_state_space_migration(self):
        """
        Test equivalence of lineage-counting state space and graph space for migration coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig({'pop_0': 5, 'pop_1': 5}),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 1},
                migration_rates={('pop_0', 'pop_1'): 0.1, ('pop_1', 'pop_0'): 0.2}
            )
        )

        self.compare_state_spaces(
            pg.state_space_old.LineageCountingStateSpace(**kwargs),
            pg.LineageCountingStateSpace(**kwargs)
        )

    def test_equivalence_block_counting_state_space_migration(self):
        """
        Test equivalence of block-counting state space and graph space for migration coalescent.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig({'pop_0': 2, 'pop_1': 2, 'pop_2': 2}),
            model=pg.BetaCoalescent(alpha=1.5, scale_time=False),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 1, 'pop_1': 3, 'pop_2': 2},
                migration_rates={
                    ('pop_0', 'pop_1'): 0.1,
                    ('pop_1', 'pop_0'): 0.2,
                    ('pop_1', 'pop_2'): 0.3,
                    ('pop_2', 'pop_1'): 0.4,
                    ('pop_2', 'pop_0'): 0.5,
                    ('pop_0', 'pop_2'): 0.6
                }
            )
        )

        self.compare_state_spaces(
            pg.state_space_old.BlockCountingStateSpace(**kwargs),
            pg.BlockCountingStateSpace(**kwargs)
        )

    def test_equivalence_lineage_counting_state_space_recombination(self):
        """
        Test equivalence of lineage-counting state space and graph space for recombination.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig(n=6),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch()
        )

        self.compare_state_spaces(
            pg.state_space_old.LineageCountingStateSpace(**kwargs),
            pg.LineageCountingStateSpace(**kwargs),
        )

    def test_equivalence_lineage_counting_state_space_recombination_demes(self):
        """
        Test equivalence of lineage-counting state space and graph space for recombination and multiple demes.
        """
        kwargs = dict(
            lineage_config=pg.LineageConfig({'pop_0': 1, 'pop_1': 1}),
            locus_config=pg.LocusConfig(n=2, recombination_rate=1.11),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(
                pop_sizes={'pop_0': 3, 'pop_1': 1},
                migration_rates={('pop_0', 'pop_1'): 0.1, ('pop_1', 'pop_0'): 0.2}
            )
        )

        self.compare_state_spaces(
            pg.state_space_old.LineageCountingStateSpace(**kwargs),
            pg.LineageCountingStateSpace(**kwargs)
        )

    def test_state_equality(self):
        """
        Test that the equality operator works as expected.
        """
        self.assertEqual(
            pg.state_space.State((np.array([1, 2]), np.array([3, 4]))),
            pg.state_space.State((np.array([1, 2]), np.array([3, 4])))
        )

        self.assertNotEqual(
            pg.state_space.State((np.array([1, 2]), np.array([3, 4]))),
            pg.state_space.State((np.array([1, 2]), np.array([3, 5])))
        )

        self.assertNotEqual(
            pg.state_space.State((np.array([1, 2]), np.array([3, 4]))),
            pg.state_space.State((np.array([1, 2]), np.array([4, 4])))
        )

    def test_state_space_equality(self):
        """
        Test that the equality operator works as expected.
        """
        self.assertEqual(
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch()
            ),
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch()
            )
        )

        self.assertNotEqual(
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch()
            ),
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=3),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch()
            )
        )

        self.assertNotEqual(
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch()
            ),
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.BetaCoalescent(alpha=1.5),
                epoch=pg.Epoch()
            )
        )

        # epoch are ignored
        self.assertEqual(
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch(pop_sizes={'pop_0': 2})
            ),
            pg.state_space.LineageCountingStateSpace(
                lineage_config=pg.LineageConfig(n=2),
                model=pg.StandardCoalescent(),
                epoch=pg.Epoch(pop_sizes={'pop_0': 1})
            )
        )

    def test_state_space_caching(self):
        """
        Test that the state space is cached.
        """
        s = pg.state_space.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 2})
        )

        # check that the cache is empty
        self.assertEqual(s._cache, {})

        # compute rate matrix
        _ = s.S

        # check that the rate matrix is in the cache
        self.assertTrue(s.epoch in s._cache)

        pg.Settings.cache_epochs = False

        s = pg.state_space.LineageCountingStateSpace(
            lineage_config=pg.LineageConfig(n=4),
            model=pg.StandardCoalescent(),
            epoch=pg.Epoch(pop_sizes={'pop_0': 2})
        )

        _ = s.S

        # check that the rate matrix is not in the cache
        self.assertEqual(s._cache, {})

        pg.Settings.cache_epochs = True

    def test_epoch_equality(self):
        """
        Test that the equality operator works as expected.
        """
        self.assertEqual(
            pg.state_space.Epoch(start_time=0, end_time=1, pop_sizes={'pop_0': 2}, migration_rates={}),
            pg.state_space.Epoch(start_time=0, end_time=1, pop_sizes={'pop_0': 2}, migration_rates={})
        )

        self.assertEqual(
            pg.state_space.Epoch(start_time=0, end_time=1, pop_sizes={'pop_0': 2}, migration_rates={}),
            pg.state_space.Epoch(start_time=0, end_time=2, pop_sizes={'pop_0': 2}, migration_rates={})
        )

        self.assertEqual(
            pg.state_space.Epoch(start_time=0, end_time=1, pop_sizes={'pop_0': 2}, migration_rates={}),
            pg.state_space.Epoch(start_time=0.5, end_time=1, pop_sizes={'pop_0': 2}, migration_rates={})
        )

        self.assertNotEqual(
            pg.state_space.Epoch(start_time=0, end_time=1, pop_sizes={'pop_0': 2}, migration_rates={}),
            pg.state_space.Epoch(start_time=0, end_time=1, pop_sizes={'pop_0': 3}, migration_rates={})
        )

        self.assertNotEqual(
            pg.state_space.Epoch(pop_sizes={'pop_0': 2, 'pop_1': 2}, migration_rates={}),
            pg.state_space.Epoch(pop_sizes={'pop_0': 2, 'pop_1': 2}, migration_rates={('pop_0', 'pop_1'): 0.1})
        )


