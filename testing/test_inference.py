"""
Test Inference class.
"""
import os
from unittest import mock
from testing import TestCase

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

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

    def test_create_run_samples_independent_start_points(self):
        """
        Unseeded ``create_run`` must draw an independent start point on each call. Identical starts would make
        the cluster multi-start pointless, since ``add_run`` keeps only the lowest-loss of otherwise equal runs.
        """
        inf = self.get_fast_inference(dict(x0=None, seed=None))

        first = inf.create_run().x0
        second = inf.create_run().x0

        self.assertNotEqual(first, second)

    def get_fast_inference_with_estimate(self, kwargs: dict = {}) -> pg.Inference:
        """
        Get the fast inference with a stored estimate and optimization result, as after ``run()``.

        :param kwargs: Additional keyword arguments.
        """
        inf = self.get_fast_inference(kwargs)
        inf.params_inferred = {'t': 1.3, 'Ne': 0.2}
        inf.loss_inferred = 1.0
        inf.result = OptimizeResult(x=np.array([1.3, 0.2]), fun=1.0, success=True, nit=3, message='converged')

        return inf

    def test_create_bootstrap_reloaded_jobs_resample_independently(self):
        """
        Cluster jobs that each reload the same saved Inference and call ``create_bootstrap()`` or ``create_run()``
        must not replay the restored generator state, which gave every job the identical resample and child seed and
        collapsed the bootstrap variance to zero. With a job index the replicate is reproducible across reloads.
        """
        json = self.get_fast_inference_with_estimate(dict(seed=None)).to_json()
        jobs = [pg.Inference.from_json(json) for _ in range(3)]

        observations = [tuple(job.create_bootstrap().observation.data) for job in jobs]
        self.assertEqual(3, len(set(observations)))
        self.assertEqual(3, len({job.create_run().seed for job in jobs}))

        indexed = [tuple(job.create_bootstrap(index=i).observation.data) for i, job in enumerate(jobs)]
        self.assertEqual(3, len(set(indexed)))

        reloaded = [tuple(pg.Inference.from_json(json).create_bootstrap(index=i).observation.data) for i in range(3)]
        self.assertEqual(indexed, reloaded)

        self.assertEqual(
            pg.Inference.from_json(json).create_run(index=5).x0,
            pg.Inference.from_json(json).create_run(index=5).x0
        )

    def test_create_bootstrap_starts_from_estimate(self):
        """
        ``create_bootstrap`` must start the replicate from the estimate, as ``bootstrap()`` does, not from the
        original start point, from which a manual replicate on a flat loss surface returned ``x0`` unchanged.
        """
        inf = self.get_fast_inference_with_estimate()

        bootstrap = inf.create_bootstrap()

        self.assertEqual({'t': 1.3, 'Ne': 0.2}, bootstrap.x0)
        self.assertNotEqual(inf.x0, bootstrap.x0)

        with self.assertRaises(RuntimeError):
            self.get_fast_inference().create_bootstrap()

    def test_add_run_and_add_bootstrap_merge_reloaded_inference(self):
        """
        ``add_run`` must merge an Inference reloaded from JSON. The jsonpickle round trip of the scipy
        ``OptimizeResult`` added an empty ``__dict__`` item, on which ``str(result)`` raised ValueError.
        """
        inf = self.get_fast_inference()
        run = pg.Inference.from_json(self.get_fast_inference_with_estimate().to_json())

        self.assertIsInstance(run.result, OptimizeResult)
        self.assertNotIn('__dict__', run.result)

        inf.add_run(run)
        inf.add_bootstrap(run)

        self.assertEqual(1.0, inf.loss_inferred)
        self.assertEqual(1, len(inf.runs))
        self.assertEqual(1, len(inf.bootstraps))

    def test_opts_do_not_alias_default_opts(self):
        """
        Changing the options of one instance in place must not change ``Inference.default_opts``, which every later
        instance constructed without ``opts`` inherited.
        """
        inf = self.get_fast_inference()
        inf.opts['maxiter'] = 1

        self.assertEqual({}, pg.Inference.default_opts)
        self.assertEqual({}, self.get_fast_inference().opts)

    def test_add_bootstraps_accepts_parameter_dicts(self):
        """
        ``add_bootstraps`` must accept dictionaries of parameters as documented, which raised AttributeError.
        """
        inf = self.get_fast_inference()

        inf.add_bootstraps([{'t': 1.0, 'Ne': 0.3}, {'Ne': 0.4, 't': 1.1}])

        np.testing.assert_array_equal([[1.0, 0.3], [1.1, 0.4]], inf._bootstrap_values)
        self.assertTrue(inf.bootstraps.loss.isna().all())

        with self.assertRaises(ValueError):
            inf.add_bootstrap({'t': 1.0})

    def test_sampled_x0_independent_of_cache_setting(self):
        """
        A sampled start point must be drawn once per instance. With ``Settings.cache = False`` every access to
        ``x0`` drew a new point, so the reported start differed from the one used and seeded results depended on
        the cache setting.
        """
        cached = self.get_fast_inference(dict(x0=None, seed=1)).x0

        pg.Settings.cache = False
        try:
            inf = self.get_fast_inference(dict(x0=None, seed=1))
            first, second = inf.x0, inf.x0
        finally:
            pg.Settings.cache = True

        self.assertEqual(first, second)
        self.assertEqual(cached, first)

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

    def test_inferred_params_labeled_by_bounds_order(self):
        """
        Regression for bug #3: with ``bounds`` and ``x0`` in different key order and ``n_runs > 1``, the inferred
        parameters and the ``runs`` DataFrame columns must follow the canonical (bounds) key order, so each optimum
        is labeled with the correct parameter name.
        """
        t_true, Ne_true = 1.5, 0.3
        obs = pg.Coalescent(
            n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t_true: Ne_true}})
        ).sfs.mean

        inf = self.get_fast_inference(dict(
            x0=dict(Ne=0.5, t=0.5),  # deliberately different key order from bounds
            bounds=dict(t=(0, 2), Ne=(0.1, 1)),
            observation=obs,
            n_runs=2,
            loss=lambda coal, observation: float(np.sum((coal.sfs.mean.data - observation.data) ** 2))
        ))

        inf.run()

        # pre-fix: params_inferred zipped result.x (bounds order) against x0's own order -> keys ['Ne', 't'] and, for
        # a sampled best run, a's optimum silently stored under 'b'
        self.assertEqual(['t', 'Ne'], list(inf.params_inferred.keys()))
        self.assertEqual(['t', 'Ne', 'loss', 'result'], list(inf.runs.columns))

    def test_run_substitutes_finite_penalty_for_nonfinite_loss(self):
        """A loss evaluating to NaN/inf must not poison the optimizer's finite-difference gradient: ``_run``
        substitutes a large finite penalty so the run still completes with a finite best loss."""
        inf = self.get_fast_inference(dict(loss=lambda coal, observation: np.nan))

        inf.run()

        self.assertTrue(np.isfinite(inf.loss_inferred))

    def test_run_skips_single_failing_start_without_aborting(self):
        """One start point whose model evaluation raises must be isolated and skipped, not abort the whole
        multi-start; the remaining runs still yield a finite inferred loss."""
        calls = {'n': 0}

        def flaky_loss(coal, observation):
            calls['n'] += 1
            if calls['n'] == 1:  # fail the first run's first evaluation, let the rest through
                raise RuntimeError('boom')
            return pg.PoissonLikelihood().compute(
                observed=observation.normalize().polymorphic,
                modelled=coal.sfs.mean.normalize().polymorphic
            )

        inf = self.get_fast_inference(dict(n_runs=2, loss=flaky_loss))

        inf.run()

        self.assertTrue(np.isfinite(inf.loss_inferred))
        self.assertEqual(len(inf.runs), 2)  # the skipped run is recorded, not dropped

    def test_nan_loss_run_not_selected_as_best(self):
        """
        Regression for bug #9: a run whose loss is NaN must not be selected as the best result.
        """
        inf = self.get_fast_inference(dict(n_runs=3))

        # one run reports a NaN loss; the two others are finite
        results = [
            OptimizeResult(x=np.array([0.5, 0.5]), fun=np.nan, success=True),
            OptimizeResult(x=np.array([0.4, 0.6]), fun=2.0, success=True),
            OptimizeResult(x=np.array([0.3, 0.7]), fun=1.0, success=True)
        ]

        # pre-fix: min(results, key=lambda r: r.fun) returned the NaN run (x < NaN and NaN < x are both False), so
        # loss_inferred was NaN
        with mock.patch.object(pg.Inference, '_optimize', side_effect=results):
            inf._run()

        self.assertTrue(np.isfinite(inf.loss_inferred))
        self.assertAlmostEqual(1.0, inf.loss_inferred)
        np.testing.assert_array_equal([0.3, 0.7], inf.result.x)

    def test_create_run_x0_overrides_cached(self):
        """
        Regression for bug #10: after the ``x0`` cached_property has been materialized, ``create_run(x0=NEW)`` must
        honor NEW rather than the stale cached value.
        """
        inf = self.get_fast_inference()

        _ = inf.x0  # materialize the cached_property

        # pre-fix: the deepcopied __dict__['x0'] shadowed the new _x0, so run.x0 stayed {'t': 0.5, 'Ne': 0.5}
        run = inf.create_run(x0=dict(t=1.7, Ne=0.9))

        self.assertEqual({'t': 1.7, 'Ne': 0.9}, run.x0)

    def test_bootstrap_before_run_raises_runtime_error(self):
        """
        Regression for bug #17: calling ``bootstrap()`` before ``run()`` must raise a clear RuntimeError.
        """
        inf = self.get_fast_inference()

        # pre-fix: the `params_inferred is None` guard never fired (it is {}), so scipy raised a confusing
        # "not enough values to unpack (expected 2, got 0)" ValueError instead
        with self.assertRaises(RuntimeError):
            inf.bootstrap()

    def test_x0_of_a_payload_without_a_start_point_is_sampled_on_restore(self):
        """
        The start point moved from the ``x0`` accessor into ``__init__``, and ``__init__`` never runs on
        deserialization. Every payload written before that whose ``x0`` had not been given carries ``_x0: None``, so
        ``x0`` (and with it ``run()``, ``add_run()`` and ``bootstrap()``) raised
        ``TypeError: argument of type 'NoneType' is not iterable``.
        """
        inf = self.get_fast_inference(dict(x0=None, seed=1))

        state = inf.__getstate__()
        state['_x0'] = None  # what a payload written before the start point moved into __init__ restores

        restored = pg.Inference.__new__(pg.Inference)
        restored.__setstate__(state)

        x0 = restored.x0

        self.assertEqual(set(restored.bounds), set(x0))
        for key, (lower, upper) in restored.bounds.items():
            self.assertTrue(lower <= x0[key] <= upper)

        # the sampled point is drawn once and then kept, so run() starts where x0 reports
        self.assertEqual(x0, restored.x0)
        restored.run()
        self.assertTrue(np.isfinite(restored.loss_inferred))

    def test_partial_x0_raises_value_error(self):
        """
        Regression for the scan-2 finding: an x0 that does not cover every bounds parameter must raise, rather than
        silently optimize a lower-dimensional subspace on the first run (a ragged run set that crashes or mislabels).
        """
        # pre-fix: x0 dropped the missing key silently, so run 0 was 1-D while sampled runs were 2-D. The bounds
        # check now runs at construction, which is where the missing key surfaces.
        with self.assertRaises(ValueError):
            pg.Inference(
                bounds=dict(a=(0, 1), b=(0, 1)),
                x0=dict(a=0.5),  # missing 'b'
                coal=lambda a, b: pg.Coalescent(n=2),
                loss=lambda coal: 0.0,
            )

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
            loss=lambda coal, observation: float(np.sum((coal.jsfs.mean.data - observation.data) ** 2)),
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
        rng = np.random.default_rng(42)

        for _ in range(100):
            weighted_loss.compute(dict(l1=rng.normal(3, 1), l2=rng.normal(1, 1)))

        self.assertTrue(1.4 < weighted_loss.compute(dict(l1=3, l2=1)) < 1.6)

    def test_serialized_callbacks_carry_their_module_globals(self):
        """
        The ``coal``, ``loss`` and ``resample`` callbacks of a script were dumped without the module-level names they
        reference, so the restored callbacks resolved those names against whatever ``__main__`` the loading process
        happened to have. A loader that does not define the writer's package alias raised ``NameError`` on the first
        call (breaking the documented distributed-bootstrapping workflow and every payload written from R), a loader
        that bound the same name to something else silently minimised a different objective, and under the ``spawn``
        start method every ``parallelize=True`` run died in the worker for the same reason.
        """
        import json
        import subprocess
        import sys
        import tempfile

        script = '''
import json
import sys

import phasegen as pg

observation = pg.SFS([100, 10, 5, 100])
seed_max = 1e10


def coal(Ne):
    return pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': Ne}))


def loss(dist, obs):
    return pg.PoissonLikelihood().compute(
        observed=obs.normalize().polymorphic,
        modelled=dist.sfs.mean.normalize().polymorphic
    )


def resample(sfs, rng):
    return sfs.resample(seed=rng.integers(seed_max))


if __name__ == '__main__':
    kwargs = dict(bounds=dict(Ne=(0.1, 2)), x0=dict(Ne=0.5), observation=observation, coal=coal, loss=loss,
                  resample=resample, n_runs=2, seed=42, pbar=False)

    sequential = pg.Inference(parallelize=False, **kwargs)
    sequential.run()

    parallel = pg.Inference(parallelize=True, **kwargs)
    parallel.run()

    sequential.to_file(sys.argv[1])

    print(json.dumps(dict(
        loss=float(loss(coal(Ne=0.5), observation)),
        sequential=[float(v) for v in sequential.params_inferred.values()],
        parallel=[float(v) for v in parallel.params_inferred.values()]
    )))
'''

        with tempfile.TemporaryDirectory() as tmp:
            path_script = os.path.join(tmp, 'writer.py')
            path_payload = os.path.join(tmp, 'inference.json')

            with open(path_script, 'w') as fh:
                fh.write(script)

            process = subprocess.run(
                [sys.executable, path_script, path_payload],
                capture_output=True,
                text=True,
                env=os.environ | {'MPLBACKEND': 'Agg', 'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES'}
            )

            self.assertEqual(0, process.returncode, process.stderr)

            written = json.loads(process.stdout.strip().splitlines()[-1])

            restored = pg.Inference.from_file(path_payload)

        # the worker processes minimise the same objective as the calling process
        np.testing.assert_allclose(written['parallel'], written['sequential'], rtol=1e-10)

        # this process defines neither the alias `pg` nor the observation that the writer's callbacks reference
        self.assertNotIn('pg', vars(sys.modules['__main__']))

        self.assertAlmostEqual(
            written['loss'],
            float(restored.loss(restored.coal(Ne=0.5), restored.observation)),
            places=12
        )

        resampled = restored.resample(restored.observation, np.random.default_rng(0))

        self.assertTrue(np.isfinite(restored.loss(restored.coal(Ne=0.5), resampled)))

    def test_two_locus_inference_runs_with_the_default_cache(self):
        """The state-space cache is built eagerly from x0, and the block-counting spaces do not exist for two loci,
        so every loss evaluation raised and the run failed as 'no finite loss'. Spaces the configuration does not
        support are simply not cached."""
        inf = pg.Inference(
            bounds={'Ne': (0.5, 2.0)},
            x0={'Ne': 1.0},
            coal=lambda Ne: pg.Coalescent(
                n=3, loci=2, recombination_rate=1.0,
                demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})
            ),
            loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2),
            observation=1.5,
            n_runs=1,
            pbar=False
        )

        inf.run()

        self.assertIsNotNone(inf.params_inferred)
        self.assertLess(inf.loss_inferred, 1e-8)

        # the two-locus coalescent supports only the lineage-counting and two-locus spaces
        self.assertEqual(['lineage_counting_state_space', 'two_locus_block_counting_state_space'],
                         list(inf._state_spaces))

    def test_x0_outside_bounds_raises_at_construction(self):
        """An explicit start point outside the box contributes nothing to the multi-start, so it is rejected where
        the caller passed it rather than later in create_run."""
        with self.assertRaises(ValueError):
            pg.Inference(
                bounds={'Ne': (0.5, 2.0)},
                x0={'Ne': 50.0},
                coal=lambda Ne: pg.Coalescent(n=2),
                loss=lambda coal, obs: 0.0,
                observation=1.0
            )

    def test_spawned_run_does_not_bootstrap(self):
        """add_run merges only the main result, so a spawned run that inherited do_bootstrap would perform
        n_bootstraps fits per job and discard every one."""
        inf = pg.Inference(
            bounds={'Ne': (0.5, 2.0)},
            x0={'Ne': 1.0},
            coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
            loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2),
            observation=1.0,
            resample=lambda obs, rng: obs,
            do_bootstrap=True,
            n_bootstraps=3,
            n_runs=1,
            pbar=False
        )

        self.assertTrue(inf.do_bootstrap)
        self.assertFalse(inf.create_run(index=0).do_bootstrap)

    def test_everywhere_invalid_loss_warns_that_the_start_point_is_reported(self):
        """A loss that is non-finite at every point is replaced by the finite penalty, so the optimizer 'converges'
        and the start point is presented as an estimate. That must be said out loud."""
        import logging

        inf = pg.Inference(
            bounds={'Ne': (0.5, 2.0)},
            x0={'Ne': 1.0},
            coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
            loss=lambda coal, obs: float('nan'),
            observation=1.0,
            n_runs=1,
            pbar=False
        )

        log = logging.getLogger('phasegen')
        records = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Collect()
        log.addHandler(handler)
        try:
            inf.run()
        finally:
            log.removeHandler(handler)

        self.assertTrue(any('start point rather than an estimate' in m for m in records), records[-3:])

    def test_unrun_spawned_objects_are_rejected_when_merged(self):
        """A spawned run or bootstrap starts unfitted. Regression: the copy carried the parent's fitted state, so an
        un-run bootstrap was merged as the parent's point estimate and the documented RuntimeError never fired,
        silently shrinking the bootstrap spread."""
        inf = pg.Inference(
            bounds={'Ne': (0.5, 2.0)},
            x0={'Ne': 1.0},
            coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
            loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2),
            observation=1.2,
            resample=lambda obs, rng: obs * rng.uniform(0.9, 1.1),
            n_runs=1,
            pbar=False
        )
        inf.run()

        with self.assertRaises(RuntimeError):
            inf.add_bootstrap(inf.create_bootstrap(index=0))

        with self.assertRaises(RuntimeError):
            inf.add_run(inf.create_run(index=0))


def test_cached_state_space_takes_the_rates_of_the_new_parameters():
    """A state space reused from the cache starts in the first epoch of the coalescent it is handed to. Regression:
    it kept the rates of the last epoch it was set to, so single-epoch paths that read the rate matrix directly, such
    as the mutation-configuration probabilities, used the x0 rates and the likelihood did not depend on Ne."""
    inf = pg.Inference(
        bounds={'Ne': (0.1, 10)},
        x0={'Ne': 1.0},
        coal=lambda Ne: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
        loss=lambda coal, obs: 0.0,
        observation=None,
        pbar=False
    )

    inf.get_coal(Ne=1.0).sfs.get_mutation_config([1, 0, 0], theta=0.7)
    cached = inf.get_coal(Ne=2.5).sfs.get_mutation_config([1, 0, 0], theta=0.7)

    fresh = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 2.5}}))
    assert cached == pytest.approx(fresh.sfs.get_mutation_config([1, 0, 0], theta=0.7), rel=1e-12)


def test_a_zero_dimensional_array_loss_is_a_valid_loss():
    """A finite loss returned as a 0-d array is used as is. Regression: it failed an isscalar check, was replaced by
    the penalty, and the fit returned its start point."""
    inf = pg.Inference(
        bounds={'Ne': (0.5, 2.0)},
        x0={'Ne': 1.0},
        coal=lambda Ne: pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
        loss=lambda coal, obs: np.asarray((coal.tree_height.mean - obs) ** 2),
        observation=1.5,
        n_runs=1,
        pbar=False
    )

    inf.run()

    assert inf.loss_inferred < 1e-8
    assert inf.params_inferred['Ne'] == pytest.approx(1.125, rel=1e-3)


def test_a_raising_evaluation_on_the_bound_does_not_discard_the_run():
    """An evaluation at which the model raises, here zero migration between two demes that then cannot absorb, is
    penalised like a non-finite loss. Regression: the exception discarded the whole run, so a fit starting on the
    bound raised RuntimeError."""
    def coal(m):
        return pg.Coalescent(
            n={'a': 2, 'b': 2},
            demography=pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): m, ('b', 'a'): m})
        )

    inf = pg.Inference(
        bounds={'m': (0, 3)},
        x0={'m': 0.0},
        coal=coal,
        loss=lambda c, obs: float((c.tree_height.mean - obs) ** 2),
        observation=coal(0.4).tree_height.mean,
        n_runs=1,
        parallelize=False,
        pbar=False
    )

    inf.run()

    assert inf.params_inferred['m'] == pytest.approx(0.4, rel=1e-3)


def test_an_error_in_the_loss_propagates_rather_than_being_penalised():
    """Only a model the parameters make invalid is penalised. Regression: any ValueError, including a broken loss, was
    turned into the penalty, and the start point was reported as a converged estimate."""
    def loss(coal, obs):
        raise ValueError('broken loss')

    inf = pg.Inference(
        bounds={'Ne': (0.5, 2.0)},
        x0={'Ne': 1.0},
        coal=lambda Ne: pg.Coalescent(n=3, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
        loss=loss,
        observation=1.5,
        n_runs=1,
        parallelize=False,
        pbar=False
    )

    with pytest.raises((ValueError, RuntimeError)):
        inf.run()
