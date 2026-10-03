"""
Test distributions.
"""

import itertools
from testing import TestCase

import numpy as np
import pytest
from matplotlib import pyplot as plt

import phasegen as pg


class DistributionTestCase(TestCase):
    """
    Test distributions.
    """

    def test_sfs_accumulation_fast(self):
        """
        Exercise the SFS moment accumulation and its plot on a small coalescent (fast path).
        """
        coal = pg.Coalescent(n=3)
        end_times = np.linspace(0, 2, 5)

        acc = np.asarray(coal.sfs.accumulate(1, end_times=end_times))
        self.assertEqual(acc.shape[0], len(end_times))

        coal.sfs.plot_accumulation(end_times=end_times, show=False)

    @staticmethod
    def get_test_coalescent() -> pg.Coalescent:
        """
        Get a test coalescent.
        """
        return pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 1, 0.2: 5},
                    pop_1={0: 0.4, 0.1: 3, 0.25: 0.3},
                    pop_2={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 0.1},
                    ('pop_1', 'pop_2'): {0: 0.2, 0.1: 0.3},
                    ('pop_2', 'pop_0'): {0: 0.4, 0.1: 0.5, 0.2: 0.6},
                    ('pop_0', 'pop_2'): {0: 0.7, 0.1: 0.8, 0.2: 0.9, 0.3: 1},
                    ('pop_2', 'pop_1'): {0: 0.1}
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=1,
                pop_1=2,
                pop_2=3
            ))
        )

    def test_quantile_raises_error_below_0(self):
        """
        Test quantile function raises error when quantile is below 0.
        """
        with self.assertRaises(ValueError) as context:
            self.get_test_coalescent().tree_height.quantile(-0.1)

        self.assertEqual(str(context.exception), 'Specified quantile must be between 0 and 1.')

    def test_quantile_raises_error_above_1(self):
        """
        Test quantile function raises error when quantile is above 1.
        """
        with self.assertRaises(ValueError) as context:
            self.get_test_coalescent().tree_height.quantile(1.1)

        self.assertEqual(str(context.exception), 'Specified quantile must be between 0 and 1.')

    def test_propagated_vector_matches_the_dense_transition_matrix(self):
        """
        The tree height propagates the row vector ``w = alpha @ T`` rather than the ``k x k`` matrix ``T``, so that
        the (sparse) matrix-exponential action can be applied to it at large ``n``. Pin it against the dense
        transition matrix it replaces, and against the CDF read off it.
        """
        from phasegen.expm import Backend
        expm = Backend.expm

        dist = self.get_test_coalescent().tree_height
        e = dist.reward._get(dist.state_space)
        alpha = dist.state_space.alpha

        for t in [0.001, 0.01, 0.1, 1, 10, 100]:
            # the dense reference: T = prod_e exp(Q_e tau_e), accumulated epoch by epoch
            T = np.eye(dist.state_space.k)
            epoch, u_prev = dist.demography.get_epoch(0), 0.0
            while t > epoch.end_time:
                dist.state_space.update_epoch(epoch)
                T = T @ expm(dist._dense_rate_matrix() * (epoch.end_time - u_prev))
                u_prev, epoch = epoch.end_time, dist.demography.get_epoch(epoch.end_time)
            dist.state_space.update_epoch(epoch)
            T = T @ expm(dist._dense_rate_matrix() * (t - u_prev))

            w = dist._sweep_to(alpha.astype(float), 0.0, t, dist.demography.get_epoch(0))

            np.testing.assert_allclose(w, alpha @ T, rtol=1e-8, atol=1e-12)
            self.assertAlmostEqual(1 - alpha @ T @ e, dist.cdf(t))

    def test_matrix_exponential_action_agrees_with_the_dense_path(self):
        """
        The tree-height cdf / pdf / quantile must not depend on which propagation path
        :attr:`Settings.expm_action_min_dim` selects: the sparse action and the dense exponential compute the same
        thing. This is what a state space too large to densify relies on, and the CDF used to ignore the setting
        outright and form the dense ``k x k`` propagator regardless.
        """
        import phasegen as pg
        from phasegen.settings import Settings

        t = np.linspace(0.1, 6, 25)
        q = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        demo = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 0.1, 1.5: 2.0}})

        dense = pg.Coalescent(n=15, demography=demo).tree_height
        ref = dense.cdf(t), dense.pdf(t), dense.quantile(q)

        try:
            Settings.expm_action_min_dim = 1  # force every propagation through the sparse action
            action = pg.Coalescent(n=15, demography=demo).tree_height
            got = action.cdf(t), action.pdf(t), action.quantile(q)
        finally:
            Settings.expm_action_min_dim = 1500

        for a, b in zip(ref, got):
            np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10)

    def test_cdf_unsorted_times_preserve_input_order(self):
        """
        The array CDF must return values aligned to the input order, not the internally sorted order (regression
        for the inverse-permutation bug on >= 3 unsorted query times).
        """
        dist = self.get_test_coalescent().tree_height
        times = np.array([3.0, 0.5, 2.0, 1.0])

        got = dist.cdf(times)
        ref = np.array([float(dist.cdf(t)) for t in times])

        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)

    def test_tree_height_cdf_and_pdf_keep_the_shape_of_2d_input(self):
        """
        The tree-height cdf and pdf must accept an array of any shape and return values of that shape. They sorted the
        unflattened input, so a 2-D array was sorted row by row, indexed into a 3-D array and raised a TypeError.
        """
        th = pg.Coalescent(n=3).tree_height
        x = np.array([[0.5, 1.0], [2.0, 0.2]])

        for f in (th.cdf, th.pdf):
            got = f(x)
            self.assertEqual(got.shape, x.shape)
            np.testing.assert_allclose(got, [[f(v) for v in row] for row in x], rtol=1e-12)

    def test_tree_height_pdf_at_an_epoch_boundary_is_independent_of_duplicates(self):
        """
        Every copy of a point on an epoch boundary must get the same density, that of the epoch ending there. The
        sweep looked up the epoch after the boundary for the next point, so a repeated boundary point switched to the
        density of the following epoch.
        """
        th = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 3}})).tree_height

        single = th.pdf(0.5)
        np.testing.assert_allclose(th.pdf([0.5, 0.5, 0.5]), single, rtol=1e-12)
        np.testing.assert_allclose(th.pdf([0.4, 0.5, 0.5, 0.6])[1:3], single, rtol=1e-12)
        self.assertAlmostEqual(single, th.pdf(0.5 - 1e-9), delta=1e-6)

    def test_tree_height_cdf_and_pdf_at_infinity_and_huge_times(self):
        """
        The tree-height cdf must be 1 and the pdf 0 at infinity and at finite times far beyond absorption, on the dense
        and the sparse propagation path. Infinity raised StopIteration from the epoch lookup, and times around 1e100
        overflowed the matrix exponential to NaN.
        """
        from phasegen.settings import Settings

        dim = Settings.expm_action_min_dim

        try:
            for min_dim in (10 ** 9, 1):
                Settings.expm_action_min_dim = min_dim
                th = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 2: 0.5}})).tree_height

                x = np.array([1.0, 1e300, np.inf])
                cdf, pdf = th.cdf(x), th.pdf(x)

                self.assertAlmostEqual(cdf[0], th.cdf(1.0), delta=1e-14)
                np.testing.assert_array_equal(cdf[1:], 1.0)
                np.testing.assert_allclose(pdf[1:], 0.0, atol=1e-300)
                self.assertEqual(th.cdf(np.inf), 1.0)
        finally:
            Settings.expm_action_min_dim = dim

    def test_tree_height_quantile_resolves_the_lower_tail(self):
        """
        The tree-height quantile must invert the exact CDF in the lower tail and return 0 at level 0. The grid gave a
        full segment to the region where the CDF cancelled to zero and spread only a few hundred uniform nodes over the
        whole body below a negative log-survival of 1, so the quantile at 1e-6 was 7% low and the quantile at 0
        was 1e-5.
        """
        from scipy.optimize import brentq

        th = pg.Coalescent(n=4).tree_height

        self.assertEqual(th.quantile(0), 0)

        for q in [1e-10, 1e-6, 1e-4, 1e-2, 0.5]:
            root = np.exp(brentq(lambda lx: th.cdf(np.exp(lx)) - q, -30, np.log(th.t_max), xtol=1e-14))
            self.assertAlmostEqual(th.quantile(q) / root, 1, delta=1e-4)

    def test_quantile(self):
        """
        The quantile is the inverse of the CDF. It reads the log-survival grid, whose nodes are exact matrix-exponential
        CDF values, so the round trip is limited by the interpolation between them rather than by a bisection
        tolerance (which is what it used to be, and had to be passed in).
        """
        dist = self.get_test_coalescent().tree_height

        for q in [0.01, 0.5, 0.99]:
            self.assertAlmostEqual(dist.cdf(dist.quantile(q)), q, delta=1e-4)

        # the boundary levels are the ends of the support
        self.assertEqual(dist.quantile(0), 0)
        self.assertAlmostEqual(dist.cdf(dist.quantile(1)), 1, delta=1e-10)

    def test_tree_height_per_population(self):
        """
        Test population means.
        """
        dist = self.get_test_coalescent().tree_height

        m_demes = {pop: dist.moment(1, rewards=(pg.TreeHeightReward().prod(pg.DemeReward(pop)),)) for pop in
                   dist.demography.pop_names}
        m = dist.moment(1)

        self.assertAlmostEqual(m, sum(m_demes.values()), delta=1e-8)

        pass

    def test_total_branch_length_per_population(self):
        """
        Test population means.
        """
        dist = self.get_test_coalescent().total_branch_length

        m_demes = {pop: dist.moment(1, rewards=(pg.TotalBranchLengthReward().prod(pg.DemeReward(pop)),)) for pop
                   in dist.demography.pop_names}
        m = dist.moment(1)

        self.assertAlmostEqual(m, sum(m_demes.values()), delta=1e-10)

        pass

    def test_n_4_2_loci_wrong_lineage_config_raises_error(self):
        """
        Check that populations can be introduced later.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 2},
                    pop_1={0: 1.5}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_2', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=1,
                pop_1=1
            )),
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        )

        _ = coal.tree_height.mean

        self.assertEqual(
            set(coal.demography.get_epoch(0).pop_names),
            {'pop_0', 'pop_1', 'pop_2'}
        )

        # population size is 1 be default
        self.assertEqual(coal.demography.get_epoch(0).pop_sizes['pop_2'], 1)

    def test_n_4_2_loci_lineage_counting_state_space(self):
        """
        Test n=4, 2 loci, lineage-counting state space.
        """
        coal = pg.Coalescent(
            demography=pg.Demography(
                pop_sizes=dict(
                    pop_0={0: 2},
                    pop_1={0: 1}
                ),
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                }
            ),
            n=pg.LineageConfig(dict(
                pop_0=2,
                pop_1=2
            )),
            loci=pg.LocusConfig(n=2, recombination_rate=1)
        )

        _ = coal.lineage_counting_state_space.S

        pass

    def test_folded_mean_sfs_test_coalescent(self):
        """
        Test folded SFS.
        """
        coal = self.get_test_coalescent()

        observed = coal.fsfs.mean
        expected = coal.sfs.mean.fold()

        np.testing.assert_array_almost_equal(observed.data, expected.data)

    def test_folded_mean_sfs_n_10(self):
        """
        Test folded SFS.
        """
        coal = pg.Coalescent(
            n=10
        )

        observed = coal.fsfs.mean
        expected = coal.sfs.mean.fold()

        np.testing.assert_array_almost_equal(observed.data, expected.data)

    def test_folded_mean_sfs_n_11(self):
        """
        Test folded SFS.
        """
        coal = pg.Coalescent(
            n=11
        )

        observed = coal.fsfs.mean
        expected = coal.sfs.mean.fold()

        np.testing.assert_array_almost_equal(observed.data, expected.data)

    def test_lineage_reward_basic_coalescent_lineage_counting_state_space(self):
        """
        Test lineage reward for basic coalescent and lineage-counting state space.
        """
        coal = pg.Coalescent(
            n=10
        )

        times = [coal.moment(1, rewards=(pg.LineageReward(i),)) for i in range(2, 11)[::-1]]

        np.testing.assert_array_almost_equal(times, [1 / (i * (i - 1) / 2) for i in range(2, 11)[::-1]])

    def test_lineage_reward_basic_coalescent_block_counting_state_space(self):
        """
        Test lineage reward for basic coalescent and block-counting state space.
        """
        coal = pg.Coalescent(
            n=10
        )

        # make sure lineage-counting state space is not supported
        r = pg.ProductReward([pg.rewards.BlockCountingUnitReward(), pg.LineageReward(2)])
        self.assertFalse(pg.Reward.support(pg.LineageCountingStateSpace, [r]))

        times = [coal.moment(1, rewards=(pg.ProductReward([
            pg.rewards.BlockCountingUnitReward(), pg.LineageReward(i)]),)) for i in range(2, 11)[::-1]]

        np.testing.assert_array_almost_equal(times, [1 / (i * (i - 1) / 2) for i in range(2, 11)[::-1]])

    def test_lineage_reward_2_demes(self):
        """
        Test lineage reward for a 2-deme coalescent.
        """
        coal = pg.Coalescent(
            n={'pop_0': 6, 'pop_1': 4},
            demography=pg.Demography(
                migration_rates={
                    ('pop_0', 'pop_1'): {0: 1},
                    ('pop_1', 'pop_0'): {0: 1},
                },
            )
        )

        times = [coal.moment(1, rewards=(pg.LineageReward(i),)) for i in range(2, 11)[::-1]]

        # check that times add up to tree height
        self.assertAlmostEqual(sum(times), coal.tree_height.mean)

    def test_sampling_formula(self):
        """
        Test sampling formula.
        """
        coal = pg.Coalescent(
            n=3
        )

        self.assertAlmostEqual(coal.sfs.get_mutation_config(config=[0, 0], theta=1), 1 / 6)
        self.assertAlmostEqual(coal.sfs.get_mutation_config(config=[0, 0], theta=0), 1)
        self.assertAlmostEqual(coal.sfs.get_mutation_config(config=[0, 1], theta=0), 0)

        pass

    def test_single_epoch_mutation_config_holds_one_resolvent(self):
        """
        The single-epoch configuration probability cached a dense stack of the ``J`` matrices ``G_j`` for every theta
        it was called with. One resolvent must be held, for the most recent theta, and the probabilities must equal
        ``alpha G_{sigma_1} ... G_{sigma_m} g`` summed over the orderings, with the ``G_j`` formed explicitly.
        """
        import scipy.sparse as sp
        from phasegen.rewards import CombinedReward, TreeHeightReward

        sfs = pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=1.5)).sfs
        ss = sfs.state_space
        keep = TreeHeightReward()._get(ss).astype(bool)
        S = (ss.S.toarray() if sp.issparse(ss.S) else np.asarray(ss.S))[np.ix_(keep, keep)]
        R = np.array([CombinedReward([sfs.reward, sfs._get_sfs_reward(i)])._get(ss) for i in (1, 2, 3)])[:, keep]
        alpha = np.asarray(ss.alpha)[keep]

        for theta in (0.3, 1.0, 2.5):
            U = np.linalg.inv(theta * np.diag(R.sum(axis=0)) - S)
            G = [theta * U * r for r in R]
            g = U @ (-S @ np.ones(len(S)))

            for config in ((0, 0, 0), (2, 1, 0), (1, 1, 1), (0, 3, 1)):
                classes = [j for j, m in enumerate(config) for _ in range(m)]
                ref = sum(alpha @ np.linalg.multi_dot([np.eye(len(S))] + [G[j] for j in p] + [g])
                          for p in set(itertools.permutations(classes)))
                np.testing.assert_allclose(sfs.get_mutation_config(config, theta), ref, rtol=1e-12)

            self.assertEqual(sfs.__dict__['_resolvent'][0], (sfs.mutation_layout(), theta))

    def test_plot_prob_10_singletons_2_doubletons(self):
        """
        Test plot of probability of 10 singletons and 2 doubletons.
        """
        coal = pg.Coalescent(
            n=3
        )

        x = np.linspace(0, 30, 100)
        y = np.array([coal.sfs.get_mutation_config(config=[10, 2], theta=x) for x in x])

        plt.plot(x, y)
        plt.show()

        pass

    def test_consume_mutation_configs_threshold(self):
        """
        Test consume_sample_generator with threshold.
        """
        coal = pg.Coalescent(n=5)

        it = coal.sfs.get_mutation_configs(theta=1)
        samples = list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.8, it))

        configs, probs = zip(*samples)

        plt.plot(probs)
        # use configs as x-ticks labels
        plt.xticks(range(len(configs)), [str(config) for config in configs], rotation=90)
        plt.tight_layout()
        plt.show()

        pass

    def test_get_mutation_config_negative_theta_raises_error(self):
        """
        Test that sampling with negative theta raises an error.
        """
        with self.assertRaises(ValueError) as context:
            pg.Coalescent(n=5).sfs.get_mutation_config(config=[1, 1], theta=-1)

    def test_get_mutation_config_windowed_host_raises(self):
        """
        The mutational-configuration probabilities are computed over the full to-absorption state space and ignore
        start_time / end_time, so on a windowed host they must raise rather than silently return the to-absorption
        result. The default (no window) works.
        """
        # default: no window, works
        pg.Coalescent(n=4).sfs.get_mutation_config(config=[1, 0, 0], theta=1)

        for kwargs in (dict(start_time=0.5), dict(end_time=1.0)):
            sfs = pg.Coalescent(n=4, **kwargs).sfs
            with self.assertRaises(NotImplementedError):
                sfs.get_mutation_config(config=[1, 0, 0], theta=1)
            with self.assertRaises(NotImplementedError):
                next(sfs.get_mutation_configs(theta=1))
            with self.assertRaises(NotImplementedError):
                next(sfs.get_mutation_configs(theta=1, order='count'))

    def test_get_mutation_config_zero_theta(self):
        """
        Test that sampling with zero theta returns probability of 1 for no mutations and 0 otherwise.
        """
        coal = pg.Coalescent(n=5)

        self.assertEqual(coal.sfs.get_mutation_config(config=[0, 0, 0, 0], theta=0), 1)
        self.assertEqual(coal.sfs.get_mutation_config(config=[0, 1, 0, 0], theta=0), 0)
        self.assertEqual(coal.sfs.get_mutation_config(config=[1, 0, 0, 0], theta=0), 0)
        self.assertEqual(coal.sfs.get_mutation_config(config=[1, 1, 0, 0], theta=0), 0)

    def test_get_mutation_config_multiple_epochs_reduces_to_homogeneous(self):
        """
        Test that the mutational-configuration probability for piecewise time-homogeneous demography reduces to the
        single-epoch result when all epochs share the same population size.
        """
        config, theta = [1, 2, 0, 1], 1.5

        homogeneous = pg.Coalescent(n=5).sfs.get_mutation_config(config=config, theta=theta)

        piecewise = pg.Coalescent(
            n=5,
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 1, 1.3: 1}})
        ).sfs.get_mutation_config(config=config, theta=theta)

        self.assertAlmostEqual(homogeneous, piecewise, places=12)

    def test_get_mutation_config_multiple_epochs_returns_valid_probability(self):
        """
        Test that the mutational-configuration probability for a genuine multi-epoch demography is a valid
        probability and that the probabilities over all configurations sum to one.
        """
        coal = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3}}))

        p = coal.sfs.get_mutation_config(config=[1, 1, 0, 1], theta=1.5)
        self.assertTrue(0 <= p <= 1)

        # generated mass approaches one as configurations are exhausted in descending probability order
        list(pg.takewhile_inclusive(lambda _: coal.sfs.generated_mass < 0.95, coal.sfs.get_mutation_configs(theta=1.5)))
        self.assertGreaterEqual(coal.sfs.generated_mass, 0.95)

    def test_get_mutation_config_incorrect_length_value_error(self):
        """
        Test that an error is raised when the length of the configuration is not equal to the number of lineages
        minus one.
        """
        with self.assertRaises(ValueError) as context:
            pg.Coalescent(n=5).sfs.get_mutation_config(config=[1, 1, 1], theta=1)

    def test_get_mutation_config_invalid_entries_raise_value_error(self):
        """
        Negative or non-integer configuration entries raise ``ValueError`` on both the single-epoch and the multi-epoch
        path. Regression: a negative entry was treated as zero on the single-epoch path, so ``[-1, 0, 0]`` returned the
        probability of ``[0, 0, 0]``, and raised ``KeyError`` on the multi-epoch path, while ``[1.7, 0, 0]`` was
        silently truncated to ``[1, 0, 0]``.
        """
        demographies = [None, pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3}})]

        for dem in demographies:
            sfs = pg.Coalescent(n=4, demography=dem).sfs

            for config in ([-1, 0, 0], [1.7, 0, 0], [0, 0.5, 0]):
                with self.subTest(epochs=dem is not None, config=config):
                    with self.assertRaises(ValueError):
                        sfs.get_mutation_config(config=config, theta=1)

    def test_get_mutation_config_integral_floats_accepted(self):
        """
        Integral floats, as passed by R, give the same probability as the integer configuration on both paths.
        """
        demographies = [None, pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.3}})]

        for dem in demographies:
            sfs = pg.Coalescent(n=4, demography=dem).sfs

            with self.subTest(epochs=dem is not None):
                self.assertEqual(
                    sfs.get_mutation_config(config=[1.0, 1.0, 0.0], theta=1),
                    sfs.get_mutation_config(config=[1, 1, 0], theta=1)
                )

    def test_get_folded_mutation_config(self):
        """
        Test that the folded SFS probability is equal to the sum of the unfolded SFS probabilities.
        """
        coal = pg.Coalescent(n=5)

        for config in [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 0],
            [0, 2],
            [1, 2],
            [2, 1],
            [2, 2]
        ]:
            p_folded = coal.fsfs.get_mutation_config(config=config, theta=1)

            # unfolded configurations (u_1, ..., u_4) with u_1 + u_4 and u_2 + u_3 equal to the folded counts
            unfolded = [
                (a, b, config[1] - b, config[0] - a) for a in range(config[0] + 1) for b in range(config[1] + 1)
            ]
            p_unfolded = [coal.sfs.get_mutation_config(config=u, theta=1) for u in unfolded]

            self.assertAlmostEqual(p_folded, sum(p_unfolded))

    def test_get_mutation_config_restricted_by_spectrum_reward(self):
        """
        The configuration probabilities of a reward-restricted spectrum must be computed from the class branch lengths
        under the spectrum's reward, on the single-epoch and the several-epoch path. Both paths used the unrestricted
        class rewards, so a deme view returned the probabilities of the full spectrum. Checked against the exact
        identity for a constant reward of 2, which equals the full spectrum at twice the mutation rate, and against
        E[prod_j Pois(m_j; theta L_j)] over sampled deme branch lengths within four standard errors.
        """
        from scipy.stats import poisson

        migration = {('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}
        theta = 0.7

        for pop_sizes in [
            {'pop_0': 1, 'pop_1': 1},
            {'pop_0': {0: 1, 0.5: 0.3, 1.5: 2}, 'pop_1': {0: 1, 0.5: 1.5, 1.5: 0.7}}
        ]:
            sfs = pg.Coalescent(
                n={'pop_0': 2, 'pop_1': 2},
                demography=pg.Demography(pop_sizes=pop_sizes, migration_rates=migration)
            ).sfs

            scaled = pg.distributions.UnfoldedSFSDistribution(
                state_space=sfs.state_space,
                tree_height=sfs._tree_height,
                demography=sfs.demography,
                reward=pg.CustomReward(lambda s: np.full(s.k, 2.0))
            )

            for config in [(0, 0, 0), (1, 0, 1), (2, 1, 0)]:
                self.assertAlmostEqual(
                    scaled.get_mutation_config(config, theta), sfs.get_mutation_config(config, 2 * theta), places=12
                )

            deme = sfs.demes['pop_0']
            lengths = deme.sample(200000, seed=3)[:, 1:4]

            for config in [(0, 0, 0), (1, 0, 0), (0, 1, 1)]:
                weights = np.prod([poisson.pmf(config[j], theta * lengths[:, j]) for j in range(3)], axis=0)
                se = weights.std() / np.sqrt(len(weights))

                self.assertLess(abs(deme.get_mutation_config(config, theta) - weights.mean()), 4 * se)

    def test_sfs_joint_restricted_by_spectrum_reward(self):
        """
        The joint distribution of two bins of a deme view must carry the view's reward, so its marginal means and
        covariance equal the view's bin means and covariance. It used the bare bin rewards and so described the full
        spectrum.
        """
        migration = {('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}
        coal = pg.Coalescent(
            n={'pop_0': 2, 'pop_1': 1},
            demography=pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates=migration)
        )
        deme = coal.sfs.demes['pop_0']

        jd = deme.joint(1, 2)

        np.testing.assert_allclose(jd.mean, np.asarray(deme.mean.data)[[1, 2]], rtol=1e-10)
        self.assertAlmostEqual(jd.cov, deme.cov.data[1, 2], places=10)

    def test_sfs_accumulate_infinite_end_time(self):
        """
        An infinite end time accumulates until absorption on the batched spectrum paths. The batched mean
        accumulation exponentiated over an infinite step and returned NaN, while the per-bin path returned the mean.
        """
        migration = {('pop_0', 'pop_1'): 1, ('pop_1', 'pop_0'): 1}
        demography = pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1}, migration_rates=migration)
        coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=demography)
        mean = np.asarray(coal.sfs.mean.data)

        np.testing.assert_allclose(coal.sfs.accumulate(1, [np.inf])[0], mean, rtol=1e-10)
        np.testing.assert_allclose(coal.sfs.accumulate(1, [1.0, np.inf])[1], mean, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(coal.sfs.moment(1, end_time=np.inf).data), mean, rtol=1e-10)
        np.testing.assert_allclose(
            np.asarray(pg.Coalescent(n={'pop_0': 2, 'pop_1': 1}, demography=demography, end_time=np.inf).sfs.mean.data),
            mean,
            rtol=1e-10
        )

    def test_sfs_bin_index_validation(self):
        """
        Bin indices must be integers of the spectrum. The folded spectrum aliased classes above n // 2 onto their
        mirror bins, so fsfs.bin(3) for n = 4 was bin 1 and fsfs.get_corr(1, 3) was 1, and non-integer indices were
        truncated.
        """
        coal = pg.Coalescent(n=4)

        for dist, i in [(coal.fsfs, 3), (coal.fsfs, 1.7), (coal.sfs, 2.5), (coal.sfs, 4), (coal.sfs, 0)]:
            with self.assertRaises(ValueError):
                dist.bin(i)

        with self.assertRaises(ValueError):
            coal.fsfs.joint(1, 3)

        for i, j in [(1.5, 1), (1, 5), (-1, 1)]:
            with self.assertRaises(ValueError):
                coal.fsfs.get_cov(i, j)
            with self.assertRaises(ValueError):
                coal.fsfs.get_corr(i, j)

        self.assertEqual(coal.fsfs.get_cov(1, 3), coal.fsfs.cov.data[1, 3])
        self.assertEqual(coal.fsfs.get_corr(1, 3), 0)
        self.assertAlmostEqual(coal.fsfs.bin(2.0).mean, coal.fsfs.mean.data[2])

    def test_get_mutation_config_infinite_end_time_and_invalid_theta(self):
        """
        An infinite end time is accumulation until absorption, so the configuration probabilities must equal those
        without an end time. The guard rejected it as a bounded window. A theta of NaN or infinity must raise, where
        NaN passed the negativity check and returned NaN.
        """
        expected = pg.Coalescent(n=3).sfs.get_mutation_config((1, 0), 1.0)

        self.assertAlmostEqual(pg.Coalescent(n=3, end_time=np.inf).sfs.get_mutation_config((1, 0), 1.0), expected)

        for theta in [np.nan, np.inf]:
            with self.assertRaises(ValueError):
                pg.Coalescent(n=4).sfs.get_mutation_config((0, 0, 0), theta)

    def test_get_accumulation_scalar_end_time(self):
        """
        A scalar end time must return a float equal to the one-element accumulation. The scalar was passed on to an
        iteration and raised TypeError.
        """
        sfs = pg.Coalescent(n=4).sfs

        value = sfs.get_accumulation(1, 1, 0.5)

        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, sfs.get_accumulation(1, 1, [0.5])[0], places=12)


def test_stability_warning_names_the_offending_epoch(caplog):
    """The ill-conditioning warning exists to say which epoch is ill-conditioned, so the dense multi-epoch
    accumulation must report the epoch it is currently in. Regression: the check inside the epoch-advancing loop
    was passed a literal ``0``, so every epoch after the first was reported as epoch 0."""
    import logging

    dem = pg.Demography(
        pop_sizes={'pop_0': {0: 1.0, 1.0: 1e-6}, 'pop_1': {0: 1.0, 1.0: 1e-6}},
        migration_rates={('pop_0', 'pop_1'): {0: 1.0, 1.0: 1e-8},
                         ('pop_1', 'pop_0'): {0: 1.0, 1.0: 1e-8}}
    )
    coal = pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=dem)

    caplog.clear()
    # end times straddling the epoch boundary at t = 1, so the dense loop advances into epoch 1
    coal.tree_height.accumulate(1, [0.5, 2.0, 5.0])

    named = {r.getMessage().split('epoch ')[1].split(' ')[0]
             for r in caplog.records if r.levelno >= logging.WARNING and 'Intensity matrix in epoch' in r.getMessage()}

    assert named, "the ill-conditioned demography produced no stability warning"
    assert '1' in named, f"epoch 1 is the ill-conditioned one but the warning named {sorted(named)}"


def test_moments_balance_each_epoch_on_its_own_rates():
    """A demography whose epochs differ by orders of magnitude in rate scale must still give accurate high-order
    moments. Regression: one balancing factor was drawn from a single epoch and reused for all of them, and since the
    Van Loan reward blocks are the only part that carries the factor, the blocks of every other epoch sat far from
    one and the scaling-and-squaring of the exponential lost their cancellation. The tree height is non-negative, so
    a negative raw moment is impossible; E[T^4] at n = 10 came back as -8.65e56 against 1.97e13."""
    # rate scales of 1e3, 1e-4 and 1e2 across three epochs
    dem = pg.Demography(pop_sizes={'pop_0': {0: 1e-3, 0.01: 1e4, 5e4: 1e-2}})

    # reference values from an independent high-precision coalescent-clock computation
    expected = {(5, 3): 4.76889365e8, (5, 4): 1.601655e13, (7, 4): 1.8018618e13, (10, 4): 1.9656674e13}

    for (n, k), exact in expected.items():
        moment = pg.Coalescent(n=n, demography=dem).tree_height.moment(k, center=False)

        assert moment > 0, f"raw moment {k} of a non-negative variable came back negative at n = {n}: {moment}"
        np.testing.assert_allclose(moment, exact, rtol=1e-5, err_msg=f"n={n}, k={k}")


def test_finite_end_time_moments_do_not_depend_on_call_order():
    """The accumulated moment at a finite end time is a function of its arguments alone. Regression: the extended
    propagator was carried across epochs against a factor drawn from the first epoch, so round-off amplified by the
    epoch rate contrast left the same call returning -9.1e11, -1.1e13 or +1.7e12 depending on what had been asked of
    the object beforehand."""
    dem = pg.Demography(pop_sizes={'pop_0': {0: 1e-3, 0.01: 1e4, 5e4: 1e-2}})
    exact = 4.76889365e8

    fresh = pg.Coalescent(n=5, demography=dem).tree_height.moment(3, end_time=1e5, center=False)

    after_lower_orders = pg.Coalescent(n=5, demography=dem)
    after_lower_orders.tree_height.moment(1, end_time=1e5, center=False)
    after_lower_orders.tree_height.moment(2, end_time=1e5, center=False)
    second = after_lower_orders.tree_height.moment(3, end_time=1e5, center=False)

    after_other_end_time = pg.Coalescent(n=5, demography=dem)
    after_other_end_time.tree_height.moment(3, end_time=2e4, center=False)
    third = after_other_end_time.tree_height.moment(3, end_time=1e5, center=False)

    for label, value in (('fresh', fresh), ('after lower orders', second), ('after another end time', third)):
        assert value > 0, f"{label}: negative raw moment {value}"
        np.testing.assert_allclose(value, exact, rtol=1e-5, err_msg=label)


def test_zero_mass_deme_does_not_decline_the_closed_form():
    """A deme declared with no samples and no migration into it is unreachable, so it cannot bear on whether
    absorption is certain. Regression: the gate tested every transient state, so such a deme declined the closed
    form and routed the moment through the general path, and the answer must agree with the demography that simply
    omits the empty deme."""
    sizes = {0: 1.0, 0.5: 1e6}

    with_empty_deme = pg.Coalescent(
        n={'pop_0': 4, 'pop_1': 0},
        demography=pg.Demography(pop_sizes={'pop_0': dict(sizes), 'pop_1': dict(sizes)})
    )
    twin = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': dict(sizes)}))

    assert with_empty_deme.tree_height._absorption_certain_in_last_epoch()

    for k in (1, 2, 3):
        np.testing.assert_allclose(
            with_empty_deme.tree_height.moment(k, center=False),
            twin.tree_height.moment(k, center=False),
            rtol=1e-12, err_msg=f"k={k}"
        )


def test_windowed_moments_balance_each_epoch_on_its_own_rates():
    """The windowed accumulation must rebalance at each epoch like the cumulative one. Regression: it kept the factor
    of the epoch holding the window start, so a window opening in a fast epoch returned E[T^3] = -1.09e13 against
    4.77e8, and a window starting at 1e-12 disagreed with the cumulative moment it must equal."""
    dem = pg.Demography(pop_sizes={'pop_0': {0: 1e-3, 0.01: 1e4, 5e4: 1e-2}})

    # reference values from an independent high-precision Van Loan computation of the windowed moments
    expected = {2: 17425.8153517, 3: 476889103.356}

    for k, exact in expected.items():
        moment = pg.Coalescent(n=5, demography=dem).tree_height.moment(k, start_time=0.005, end_time=1e5, center=False)
        np.testing.assert_allclose(moment, exact, rtol=1e-5, err_msg=f"k={k}")


def test_tree_height_propagation_does_not_stop_early_on_rates_far_apart(caplog):
    """A later epoch whose rates span 1e18 warns about the rate spread and propagates the transient mass on, past
    steps that leave it unchanged in the last bit. Regression: the propagation ended at the first such step and the
    CDF froze at 0.46 up to infinity, against the exact 0.99999999889 at t = 1e19."""
    coal = pg.Coalescent(n={'pop_0': 1, 'pop_1': 1}, demography=pg.Demography(
        pop_sizes={'pop_0': 1, 'pop_1': 1},
        migration_rates={('pop_0', 'pop_1'): {0: 1, 0.5: 1e-18}, ('pop_1', 'pop_0'): {0: 1, 0.5: 1e-18}}))

    with caplog.at_level('WARNING'):
        cdf = coal.tree_height.cdf(np.array([1e17, 1e19]))

    assert cdf[1] > 0.8
    assert any('10 orders of magnitude' in r.getMessage() for r in caplog.records)


def test_sfs_var_reads_the_diagonal_of_a_closed_form_cov(monkeypatch):
    """The variance after a closed-form covariance is its diagonal and evaluates no bin again. Regression: it
    recomputed the second moment of every bin, which took 21 s after the covariance at n = 24."""
    def coal():
        return pg.Coalescent(n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.4: 0.25, 1.5: 2.0}}))

    var = coal().sfs.var.data
    c = coal()
    cov = c.sfs.cov.data

    def fail(*args, **kwargs):
        raise AssertionError("a bin moment was evaluated")

    monkeypatch.setattr(pg.distributions.spectra.SFSDistribution, '_moment', fail)

    np.testing.assert_allclose(c.sfs.var.data, np.diag(cov), rtol=0, atol=0)
    np.testing.assert_allclose(c.sfs.var.data, var, rtol=1e-12)


def test_high_moments_of_a_large_reward_balance_on_the_reward_scale():
    """The balancing factor must account for the size of the reward. Regression: it was sized for rewards of order
    one, so the Van Loan reward blocks of the total branch length, whose reward reaches n, forced the exponential into
    more squarings than its generator blocks needed, and E[L^4] at n = 200 came back as 10213.70 against 12386.40.
    The reference is the balancing-invariant value, which a factor 10 to 10^4 times larger reproduces to 1e-14."""
    dem = pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.3: 0.2, 1.0: 3.0}})

    moment = pg.Coalescent(n=200, demography=dem).total_branch_length.moment(4, center=False)

    np.testing.assert_allclose(moment, 12386.402537073, rtol=1e-10)


@pytest.mark.parametrize('sizes, expected', [
    ({0: 1e-3, 0.05: 1e3, 100: 1e2}, {3: 8.888893084268353e-09, 4: 3.755404427977085e-11}),
    ({0: 1e-3, 0.05: 1e3, 100: 1e4}, {3: 1.0475438391579955e-08, 4: 6.346201571908804e-05}),
    ({0: 1e-3, 0.05: 2e-3, 0.06: 1e4}, {3: 8.900585191891565e-09, 4: 4.6788797195885267e-07}),
])
def test_high_moments_keep_a_slow_epoch_after_almost_sure_absorption(sizes, expected):
    """An epoch beginning after the time of almost sure absorption may stand in for the epochs after it only where
    the moment it misplaces is negligible, and that share grows with the order. Regression: the criterion measured the
    first order and the held epoch alone, so the epoch beginning at 0.05 was held in place of the final one and
    E[T^4] came back 186 times too large (final size 1e2) or 1e-4 times too small (the other two). References from an
    mpmath Van Loan computation at 60 digits."""
    dem = pg.Demography(pop_sizes={'pop_0': sizes})
    coal = pg.Coalescent(n=3, demography=dem)

    for k, exact in expected.items():
        np.testing.assert_allclose(coal.tree_height.moment(k, center=False), exact, rtol=1e-6, err_msg=f"k={k}")


def test_sparse_closed_form_moments_survive_drop_cache():
    """The closed form of a sparse rate matrix must read the rates of the epoch it is in after the state-space cache
    is dropped. Regression: the per-epoch CSR memo aliased the live rate matrix, which a change of epoch rescales in
    place, so E[T^3] came back as 6.62 against 15.86 after ``drop_cache``."""
    pg.Settings.dense_rate_matrix_max_states = 1
    pg.Settings.closed_form_sparse_min_states = 1

    def build() -> pg.Coalescent:
        return pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.0, 0.5: 5.0, 2.0: 0.5}}))

    reference = build().tree_height.moment(3, center=False)

    coal = build()
    coal.tree_height.moment(2, center=False)
    coal.drop_cache()

    np.testing.assert_allclose(coal.tree_height.moment(3, center=False), reference, rtol=1e-12)


@pytest.mark.parametrize('round_trip', [True, False])
def test_dropped_cache_rebuilds_rate_matrix_as_fresh(round_trip):
    """A coalescent whose state space was left at a later epoch must rebuild the rate matrix of the first epoch bit for
    bit after its cache is dropped, directly or by serialization. Regression: the state space kept the later epoch, so
    the matrix was built there and rescaled back, which moved the marginal moments of the SFS in the last digits and,
    through the conditioning values, swung the derivative identity of the conditional moments on 4_epoch_up_down_n_4
    from a scaled error of 0.010 to 0.039."""

    def build() -> pg.Coalescent:
        return pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1.2, 0.3: 10, 1: 0.8, 1.4: 10}}))

    fresh = build()
    first = fresh.demography.get_epoch(0)
    fresh.block_counting_state_space.update_epoch(first)
    S_fresh = np.array(fresh.block_counting_state_space.S, copy=True)
    std_fresh = fresh.sfs.joint(1, 2).marginal('a').std

    coal = build()
    _ = coal.sfs.mean
    coal.sfs.state_space.update_epoch(coal.sfs._get_epochs_until_unbounded()[-1])

    if round_trip:
        coal = pg.Coalescent.from_json(coal.to_json())
    else:
        coal.drop_cache()

    assert coal.block_counting_state_space.epoch == first

    coal.block_counting_state_space.update_epoch(first)
    np.testing.assert_array_equal(coal.block_counting_state_space.S, S_fresh)
    assert coal.sfs.joint(1, 2).marginal('a').std == std_fresh


def test_stability_warning_for_fast_migration_hiding_slow_coalescence(caplog):
    """Fast migration within a class of states hides the slow coalescence out of it from the exit rates, which are all
    about 2, while the tree height is resolved on the time scale 1e12. Regression: the warning compared the exit rates
    alone and stayed silent while the CDF and the mean drifted."""
    coal = pg.Coalescent(n={'a': 1, 'b': 1}, demography=pg.Demography(
        pop_sizes={'a': 1e12, 'b': 1e12}, migration_rates={('a', 'b'): 1, ('b', 'a'): 1}))

    with caplog.at_level('WARNING'):
        coal.tree_height.mean

    assert any('10 orders of magnitude' in r.getMessage() for r in caplog.records)
