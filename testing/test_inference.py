"""
Test Inference class.
"""
import os
from testing import TestCase

import numpy as np
import pytest

import phasegen as pg


class InferenceTestCase(TestCase):
    """
    Test Inference class.
    """

    def get_basic_inference(self, kwargs: dict = {}):
        """
        Get basic inference.

        :param kwargs: Additional keyword arguments.
        """
        kwargs = dict(
            x0=dict(t=1, Ne=1),
            bounds=dict(t=(0, 4), Ne=(0.1, 1)),
            observation=pg.SFS([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            parallelize=False,
            n_runs=1,
            seed=42,
            do_bootstrap=False,
            coal=lambda t, Ne: (
                pg.Coalescent(
                    n=10,
                    demography=pg.Demography(
                        pop_sizes={'pop_0': {0: 1, t: Ne}}
                    )
                )
            ),
            loss=lambda coal, observation: (
                pg.PoissonLikelihood().compute(
                    observed=observation.normalize().polymorphic,
                    modelled=coal.sfs.mean.normalize().polymorphic
                )
            ),
            resample=lambda sfs, rng: sfs.resample(seed=rng.integers(1e10))
        ) | kwargs

        return pg.Inference(**kwargs)

    def get_fast_inference(self, kwargs: dict = {}):
        """
        Get a small (n=3, single parameter) inference that exercises the inference code paths quickly
        enough to stay in the non-slow suite. Correctness is covered by the slow ``get_basic_inference`` tests.

        :param kwargs: Additional keyword arguments.
        """
        kwargs = dict(
            x0=dict(t=0.5, Ne=0.5),
            bounds=dict(t=(0, 2), Ne=(0.1, 1)),
            observation=pg.SFS([100, 10, 5, 100]),
            parallelize=False,
            n_runs=1,
            seed=42,
            do_bootstrap=False,
            coal=lambda t, Ne: pg.Coalescent(
                n=3,
                demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})
            ),
            loss=lambda coal, observation: pg.PoissonLikelihood().compute(
                observed=observation.normalize().polymorphic,
                modelled=coal.sfs.mean.normalize().polymorphic
            ),
            resample=lambda sfs, rng: sfs.resample(seed=rng.integers(1e10))
        ) | kwargs

        return pg.Inference(**kwargs)

    def test_fast_inference_run_bootstrap_and_plots(self):
        """
        Run a small inference with bootstrapping and exercise the plotting and serialization paths.
        """
        inf = self.get_fast_inference(dict(do_bootstrap=True, n_bootstraps=3))

        inf.run()

        # the optimization produced inferred parameters and a finite loss
        assert 'Ne' in inf.params_inferred
        assert np.isfinite(inf.loss_inferred)
        assert len(inf.bootstraps) == 3

        # plotting paths
        inf.plot_pop_sizes()
        inf.plot_migration()
        inf.plot_demography()
        inf.plot_bootstraps(kind='hist')
        inf.plot_bootstraps(kind='kde')

        # serialization round-trip
        restored = pg.Inference.from_json(inf.to_json())
        assert restored.params_inferred['Ne'] == pytest.approx(inf.params_inferred['Ne'])

    def test_fast_inference_manual_runs_and_bootstraps(self):
        """
        Exercise the distributed run/bootstrap helpers (create/add run and bootstrap) on a small inference.
        """
        inf = self.get_fast_inference()

        # additional independent runs, merged back in
        run2 = inf.create_run()
        inf.run()
        run2.run()
        inf.add_run(run2)
        assert np.isfinite(inf.loss_inferred)

        # manual bootstrap created, run independently and added back
        bootstrap = inf.create_bootstrap()
        bootstrap.run()
        inf.add_bootstrap(bootstrap)
        assert len(inf.bootstraps) == 1

    @pytest.mark.slow
    def test_basic_inference(self):
        """
        Test basic inference.
        """
        # create inference object
        inf = self.get_basic_inference()

        inf._logger.setLevel('DEBUG')

        inf.run()

        self.assertAlmostEqual(0.15, inf.params_inferred['t'], places=2)
        self.assertAlmostEqual(0.48, inf.params_inferred['Ne'], places=2)
        self.assertAlmostEqual(2.37838, inf.loss_inferred, places=5)

    @pytest.mark.slow
    def test_basic_inference_3_runs_sequential(self):
        """
        Test basic inference with 3 runs in sequence.
        """
        # create inference object
        inf = self.get_basic_inference(dict(n_runs=3, parallelize=False))

        self.assertEqual(0, len(inf.runs))
        self.assertEqual(0, len(inf.bootstraps))

        inf.run()

        self.assertEqual(3, len(inf.runs))
        self.assertEqual(0, len(inf.bootstraps))

    @pytest.mark.skipif(not bool(os.getenv("PARALLEL", False)), reason="Not running parallel tests.")
    @pytest.mark.slow
    def test_basic_inference_3_runs_parallel(self):
        """
        Test basic inference with 3 runs in parallel.
        """
        # create inference object
        inf = self.get_basic_inference(dict(n_runs=3, parallelize=True))

        inf.run()

    @pytest.mark.slow
    def test_serialize_basic_inference_before_running(self):
        """
        Test serialization of basic inference.
        """
        # create inference object
        inf = self.get_basic_inference()

        inf.to_file('scratch/test_serialize_basic_inference.json')

        inf2 = pg.Inference.from_file('scratch/test_serialize_basic_inference.json')

        inf.run()
        inf2.run()

        self.assertAlmostEqual(inf.params_inferred['t'], inf2.params_inferred['t'])
        self.assertAlmostEqual(inf.params_inferred['Ne'], inf2.params_inferred['Ne'])
        self.assertAlmostEqual(inf.loss_inferred, inf2.loss_inferred)

    @pytest.mark.slow
    def test_serialize_basic_inference_after_running(self):
        """
        Test serialization of basic inference.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_bootstraps=3))

        inf.run()

        inf.to_file('scratch/test_serialize_run_basic_inference.json')

        inf2 = pg.Inference.from_file('scratch/test_serialize_run_basic_inference.json')

        params = list(inf.params_inferred.keys())
        self.assertAlmostEqual(inf.params_inferred['t'], inf2.params_inferred['t'])
        self.assertAlmostEqual(inf.params_inferred['Ne'], inf2.params_inferred['Ne'])
        self.assertAlmostEqual(inf.loss_inferred, inf2.loss_inferred)
        self.assertDictEqual(inf.bootstraps[params].var().to_dict(), inf2.bootstraps[params].var().to_dict())

    @pytest.mark.skipif(not bool(os.getenv("PARALLEL", False)), reason="Not running parallel tests.")
    @pytest.mark.slow
    def test_seeded_inference_parallel(self):
        """
        Test seeded inference with parallelization.
        """
        # create inference object
        inf = self.get_basic_inference(dict(seed=42, parallelize=True, x0=None, do_bootstrap=True, n_bootstraps=3))
        inf.run()

        # create inference object
        inf2 = self.get_basic_inference(dict(seed=42, parallelize=True, x0=None, do_bootstrap=True, n_bootstraps=3))
        inf2.run()

        self.assertAlmostEqual(inf.params_inferred['t'], inf2.params_inferred['t'])
        self.assertAlmostEqual(inf.params_inferred['Ne'], inf2.params_inferred['Ne'])
        self.assertAlmostEqual(inf.loss_inferred, inf2.loss_inferred)
        self.assertDictEqual(inf.bootstraps.var().to_dict(), inf2.bootstraps.var().to_dict())

    @pytest.mark.slow
    def test_unseeded_inference_yields_different_results(self):
        """
        Test unseeded inference yields different results.
        """
        # create inference object
        inf = self.get_basic_inference(dict(seed=None, n_runs=1, x0=None))
        inf.run()

        # create inference object
        inf2 = self.get_basic_inference(dict(seed=None, n_runs=1, x0=None))
        inf2.run()

        # check that we get different results
        self.assertNotEqual(inf.result.x[0], inf2.result.x[0])

    def test_automatic_boostrap_no_resample_raises_error(self):
        """
        Test automatic bootstrap without resampling raises error.
        """
        with self.assertRaises(ValueError) as context:
            self.get_basic_inference(dict(do_bootstrap=True, resample=None))

        print(context.exception)

    def test_automatic_boostrap_no_observation_raises_error(self):
        """
        Test automatic bootstrap without observation raises error.
        """
        with self.assertRaises(ValueError) as context:
            self.get_basic_inference(dict(do_bootstrap=True, observation=None))

        print(context.exception)

    @pytest.mark.slow
    def test_bootstrap_sequential(self):
        """
        Test sequential bootstrap.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_runs=3, parallelize=False, n_bootstraps=10))
        inf.run()

        self.assertEqual(10, len(inf.bootstraps))

        # make sure the bootstraps are different
        self.assertGreater(inf.bootstraps.t.var(), 0)

    @pytest.mark.skipif(not bool(os.getenv("PARALLEL", False)), reason="Not running parallel tests.")
    @pytest.mark.slow
    def test_bootstrap_parallel(self):
        """
        Test parallel bootstrap.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_runs=3, parallelize=True, n_bootstraps=10))
        inf.run()

        self.assertEqual(10, len(inf.bootstraps))

        # make sure the bootstraps are different
        self.assertGreater(inf.bootstraps.t.var(), 0)

    @pytest.mark.slow
    def test_manual_bootstrap(self):
        """
        Test manual bootstrap.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=False))
        inf.run()

        bootstraps = [inf.create_bootstrap() for _ in range(5)]
        [bootstrap.run() for bootstrap in bootstraps]
        [inf.add_bootstrap(bootstrap) for bootstrap in bootstraps]

        self.assertEqual(5, len(inf.bootstraps))
        self.assertGreater(inf.bootstraps.t.var(), 0)

        inf.plot_bootstraps()

    @pytest.mark.skip("Not working yet.")
    @pytest.mark.slow
    def test_manual_bootstrap_serialize_twice(self):
        """
        Test manual bootstrap serialization.
        """
        inf = self.get_basic_inference(dict(do_bootstrap=False))
        inf.run()

        inf.to_file('scratch/test_manual_bootstrap_serialization.json')
        inf = pg.Inference.from_file('scratch/test_manual_bootstrap_serialization.json')

        inf.to_file('scratch/test_manual_bootstrap_serialization2.json')
        pg.Inference.from_file('scratch/test_manual_bootstrap_serialization2.json')

    @pytest.mark.slow
    def test_plot_inference(self):
        """
        Test plotting inference.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_runs=3, parallelize=False, n_bootstraps=3))
        inf.run()

        inf.plot_migration()
        inf.plot_pop_sizes()
        inf.plot_demography()
        inf.plot_bootstraps(kind='hist')
        inf.plot_bootstraps(kind='kde')

    @pytest.mark.slow
    def test_manual_runs_unseeded(self):
        """
        Test unseeded manual runs.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=False))

        run2 = inf.create_run()
        run3 = inf.create_run()

        inf.run()
        run2.run()
        run3.run()

        # make sure the runs are different
        self.assertNotEqual(inf.result.x[0], run2.result.x[0])
        self.assertNotEqual(inf.result.x[0], run3.result.x[0])

        inf.add_runs([run2, run3])

        self.assertEqual(inf.loss_inferred, min([inf.loss_inferred, run2.loss_inferred, run3.loss_inferred]))

    @pytest.mark.slow
    def test_manual_runs_seeded(self):
        """
        Test seeded manual runs.
        """
        # create inference object
        inf = self.get_basic_inference(dict(seed=42, do_bootstrap=False))

        run2 = inf.create_run()
        run3 = inf.create_run()

        inf.run()
        run2.run()
        run3.run()

        # make sure the runs are not the same
        self.assertNotEqual(inf.result.x[0], run2.result.x[0])
        self.assertNotEqual(inf.result.x[0], run3.result.x[0])

        inf.add_runs([run2, run3])

        self.assertEqual(inf.loss_inferred, min([inf.loss_inferred, run2.loss_inferred, run3.loss_inferred]))

        # make sure seeds are different
        self.assertNotEqual(inf.seed, run2.seed)
        self.assertNotEqual(inf.seed, run3.seed)

    @pytest.mark.slow
    def test_add_run_without_run_itself(self):
        """
        Test adding a run without running it.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=False))

        run = inf.create_run()
        run.run()

        inf.add_run(run)

        self.assertEqual(inf.loss_inferred, run.loss_inferred)

    def test_add_not_run_run_raises_error(self):
        """
        Test adding a run which has not been run raises error.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=False))

        run = inf.create_run()

        with self.assertRaises(RuntimeError) as context:
            inf.add_run(run)

    @pytest.mark.slow
    def test_state_state_caching_vs_no_caching(self):
        """
        Test state caching vs no caching.
        """
        cached = self.get_basic_inference(dict(cache=True))
        uncached = self.get_basic_inference(dict(cache=False))

        # no apparent performance difference for an inference this simple
        cached.run()
        uncached.run()

        np.testing.assert_almost_equal(
            list(cached.params_inferred.values()),
            list(uncached.params_inferred.values()),
            decimal=6
        )

    def get_joint_sfs_inference(self, kwargs: dict = {}):
        """
        Get an inference whose loss is based on the joint (multi-population) site-frequency spectrum, exercising the
        joint block-counting state space.

        :param kwargs: Additional keyword arguments.
        """
        def coal(m):
            return pg.Coalescent(
                n={'pop_0': 2, 'pop_1': 2},
                demography=pg.Demography(
                    pop_sizes={'pop_0': 1, 'pop_1': 1},
                    migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): m}
                )
            )

        # observation generated from the model at a known migration rate
        observation = coal(0.7).jsfs.mean

        kwargs = dict(
            x0=dict(m=0.4),
            bounds=dict(m=(0.1, 2)),
            observation=observation,
            parallelize=False,
            n_runs=1,
            seed=42,
            do_bootstrap=False,
            cache=True,
            coal=coal,
            loss=lambda coal, observation: float(np.sum((coal.jsfs.mean - observation) ** 2)),
            resample=lambda obs, rng: obs
        ) | kwargs

        return pg.Inference(**kwargs)

    @pytest.mark.slow
    def test_joint_sfs_inference_runs_with_state_space_caching(self):
        """
        Inference using the joint SFS must run with state-space caching enabled (the joint block-counting state space
        is built once and reused across loss evaluations).
        """
        inf = self.get_joint_sfs_inference()

        inf.run()

        # the migration rate should be recovered reasonably well
        self.assertAlmostEqual(0.7, inf.params_inferred['m'], places=1)

    @pytest.mark.slow
    def test_joint_sfs_inference_serialization(self):
        """
        Inference using the joint SFS must serialize and deserialize (the cached joint state space is pickled via the
        ``__getstate__``/``__setstate__`` plumbing), yielding identical results.
        """
        inf = self.get_joint_sfs_inference()

        inf.to_file('scratch/test_joint_sfs_inference.json')
        inf2 = pg.Inference.from_file('scratch/test_joint_sfs_inference.json')

        inf.run()
        inf2.run()

        self.assertAlmostEqual(inf.params_inferred['m'], inf2.params_inferred['m'])
        self.assertAlmostEqual(inf.loss_inferred, inf2.loss_inferred)

    def test_weighted_loss(self):
        """
        Test weighted loss.
        """
        weighted_loss = pg.inference.WeightedLoss(dict(l1=0.5, l2=0.5))

        self.assertAlmostEqual(1, weighted_loss.compute(dict(l1=1, l2=1)))

        weighted_loss = pg.inference.WeightedLoss(dict(l1=0.5, l2=0.5))

        weighted_loss.compute(dict(l1=1, l2=2))
        weighted_loss.compute(dict(l1=2, l2=1))

        self.assertAlmostEqual(1, weighted_loss.compute(dict(l1=1, l2=1)))

        weighted_loss = pg.inference.WeightedLoss(dict(l1=0.5, l2=0.5))

        self.assertAlmostEqual(1.5, weighted_loss.compute(dict(l1=3, l2=1)))
        self.assertAlmostEqual(1.5, weighted_loss.compute(dict(l1=3, l2=1)))

        weighted_loss = pg.inference.WeightedLoss(dict(l1=0.5, l2=0.5))

        for _ in range(100):
            weighted_loss.compute(dict(l1=np.random.normal(3, 1), l2=np.random.normal(1, 1)))

        self.assertTrue(1.4 < weighted_loss.compute(dict(l1=3, l2=1)) < 1.6)
