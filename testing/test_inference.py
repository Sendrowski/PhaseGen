"""
Test Inference class.
"""
import copy
import logging
import os
import traceback
from typing import Tuple
from unittest import mock
from testing import TestCase

import numpy as np
import pandas as pd
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

    def test_non_positive_runs_or_bootstraps_raise_value_error(self):
        """
        Regression: ``n_runs <= 0`` silently performed one optimization, and ``n_bootstraps <= 0`` returned an empty
        bootstrap table with NaN means.
        """
        for kwargs in (dict(n_runs=0), dict(n_runs=-1), dict(n_bootstraps=0), dict(n_bootstraps=-2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.get_fast_inference(kwargs)

        inf = self.get_fast_inference()
        inf.run()

        with self.assertRaises(ValueError):
            inf.create_bootstrap(n_runs=0)

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
        params = inf.param_names
        self.assertDictEqual(inf.bootstraps[params].var().to_dict(), inf2.bootstraps[params].var().to_dict())

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
        self.assertEqual(['lineage_counting', 'two_locus_block_counting'],
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

    def test_spawned_run_performs_one_optimization(self):
        """add_run records one row per spawned run, so a spawned run that inherited the parent's n_runs performed
        n_runs optimizations per cluster job and the runs table under-counted them."""
        inf = pg.Inference(
            bounds={'Ne': (0.5, 2.0)},
            coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
            loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2),
            observation=1.0,
            pbar=False
        )

        run = inf.create_run(index=0)
        run.run()

        self.assertEqual(10, inf.n_runs)
        self.assertEqual(1, len(run.runs))

    def test_invalid_bounds_and_method_raise_at_construction(self):
        """Reversed bounds raised numpy's 'high - low < 0' or blamed x0, and an unknown optimizer skipped every run
        and blamed the loss function. Both are rejected at construction with an error naming the cause."""
        kwargs = dict(coal=lambda Ne: pg.Coalescent(n=2), loss=lambda coal, obs: 0.0, observation=1.0)

        for x0 in (None, {'Ne': 1.0}):
            with self.assertRaisesRegex(ValueError, r"lower bound exceeds the upper bound for parameters \['Ne'\]"):
                pg.Inference(bounds={'Ne': (2.0, 0.5)}, x0=x0, **kwargs)

        with self.assertRaisesRegex(ValueError, "Unknown optimization method 'L-BFGS'"):
            pg.Inference(bounds={'Ne': (0.5, 2.0)}, method_mle='L-BFGS', **kwargs)

        pg.Inference(bounds={'Ne': (0.5, 2.0)}, method_mle='nelder-mead', **kwargs)

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

    def test_bootstrap_replicates_with_everywhere_invalid_loss_warn_and_are_not_converged(self):
        """A bootstrap replicate whose loss is invalid at every point returned the start point, the estimate of the
        main run, as a converged replicate without a warning, which shrank the bootstrap spread. It must be marked as
        not converged and reported."""
        inf = pg.Inference(
            bounds={'Ne': (0.5, 2.0)},
            x0={'Ne': 1.2},
            coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
            loss=lambda coal, obs: (coal.tree_height.mean - obs) ** 2 if obs < 1.5 else float('nan'),
            observation=1.0,
            resample=lambda obs, rng: 2.0,
            n_runs=1,
            n_bootstraps=2,
            pbar=False
        )
        inf._run()

        with self.assertLogs('phasegen', level='WARNING') as logs:
            inf.bootstrap()

        self.assertTrue(any('2 out of 2 bootstrap replicates' in m and 'are NaN' in m for m in logs.output))
        self.assertTrue(all('success: False' in r for r in inf.bootstraps['result']))
        self.assertTrue(inf.bootstraps[inf.param_names].isna().all().all())

    def test_plot_bootstraps_without_bootstraps_raises_runtime_error(self):
        """The empty bootstrap table passed the guard and pandas raised TypeError on the empty data."""
        inf = self.get_fast_inference()

        with self.assertRaisesRegex(RuntimeError, 'No bootstraps available'):
            inf.plot_bootstraps(show=False)

    def test_plot_bootstraps_kde_needs_two_finite_replicates_per_parameter(self):
        """A kernel density estimate of a parameter with fewer than two finite replicates raised scipy's opaque
        ValueError. It names the parameter, and two finite replicates suffice."""
        import matplotlib.pyplot as plt

        inf = self.get_fast_inference()
        inf.bootstraps = pd.DataFrame(dict(t=[0.4, 0.6, np.nan], Ne=[0.5, np.nan, np.nan]))

        with self.assertRaisesRegex(RuntimeError, 'at least two finite replicates.*Ne'):
            inf.plot_bootstraps(show=False, kind='kde')

        inf.bootstraps['Ne'] = [0.5, 0.7, np.nan]
        inf.plot_bootstraps(show=False, kind='kde')
        plt.close('all')

    def test_plot_bootstraps_subplots_are_three_quarters_of_the_default_height(self):
        """The figure of the subplots has the default width and three quarters of the default height."""
        import matplotlib.pyplot as plt

        inf = self.get_fast_inference()
        inf.bootstraps = pd.DataFrame(dict(t=[0.4, 0.6, 0.5], Ne=[0.5, 0.7, 0.6]))

        ax = inf.plot_bootstraps(show=False)
        width, height = plt.rcParams['figure.figsize']
        np.testing.assert_allclose(ax[0].figure.get_size_inches(), [width, 0.75 * height])
        plt.close('all')

    def test_plot_bootstraps_draws_the_truth(self):
        """The true value of each named parameter is drawn as a dashed vertical line, on its own axis or on the shared
        one, and an unknown name raises."""
        import matplotlib.pyplot as plt

        inf = self.get_fast_inference()
        inf.bootstraps = pd.DataFrame(dict(t=[0.4, 0.6, 0.5], Ne=[0.5, 0.7, 0.6]))

        def dashed(ax):
            return [line.get_xdata()[0] for line in ax.get_lines() if line.get_linestyle() == '--']

        axes = inf.plot_bootstraps(show=False, truth=dict(t=0.55))
        self.assertEqual([[0.55], []], [dashed(ax) for ax in axes])

        ax = inf.plot_bootstraps(show=False, subplots=False, truth=dict(t=0.55, Ne=0.65))
        self.assertEqual([0.55, 0.65], dashed(ax))

        with self.assertRaisesRegex(ValueError, 'unknown parameters: N1'):
            inf.plot_bootstraps(show=False, truth=dict(N1=1))

        plt.close('all')

    def test_plot_pop_sizes_rejects_a_truth_of_other_demes(self):
        """A true demography with trajectories the inferred one lacks raises, as unknown names in plot_bootstraps."""
        import matplotlib.pyplot as plt

        inf = self.get_fast_inference()
        inf.run()

        with self.assertRaisesRegex(ValueError, 'lacks: A'):
            inf.plot_pop_sizes(interval=None, show=False, truth=pg.Demography(pop_sizes={'A': 1}))

        plt.close('all')

    def test_plot_migration_and_demography_draw_the_true_trajectories(self):
        """The true migration rates are drawn dashed in plot_migration and plot_demography, and a true migration
        between demes the inferred demography lacks raises."""
        import matplotlib.pyplot as plt

        coal = lambda m: pg.Coalescent(n={'a': 2, 'b': 1}, demography=pg.Demography(
            pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): m, ('b', 'a'): 0.5}))
        inf = pg.Inference(coal=coal, loss=lambda c, obs: float((c.tree_height.mean - obs) ** 2), observation=2.5,
                           x0=dict(m=0.5), bounds=dict(m=(0.01, 5)), n_runs=1, pbar=False)
        inf.run()

        truth = pg.Demography(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 0.3, ('b', 'a'): 0.5})
        for plot in (inf.plot_migration, inf.plot_demography):
            ax = plot(interval=None, show=False, truth=truth)
            self.assertTrue(any(line.get_label().endswith('(true)') and line.get_linestyle() == '--'
                                for line in ax.get_lines()))

        other = pg.Demography(pop_sizes={'a': 1, 'b': 1, 'c': 1}, migration_rates={('a', 'c'): 0.3})
        with self.assertRaisesRegex(ValueError, 'lacks'):
            inf.plot_migration(interval=None, show=False, truth=other)

        plt.close('all')

    def test_exact_gradient_of_a_covariance_loss_matches_finite_differences(self):
        """An inference whose loss reads the mean, the variance and the covariance of the SFS reaches the same estimate
        and Fisher information with the exact gradient as with finite differences."""
        coal = lambda N, t: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: N}}))
        target = coal(2.0, 0.4)
        observation = (target.sfs.mean.data, target.sfs.var.data, target.sfs.cov.data)

        def loss(c, obs):
            return float(np.sum((c.sfs.mean.data - obs[0]) ** 2) + np.sum((c.sfs.var.data - obs[1]) ** 2) +
                         np.sum((c.sfs.cov.data - obs[2]) ** 2))

        results = []
        for gradient in ('exact', 'finite_differences'):
            inf = pg.Inference(coal=coal, loss=loss, observation=observation, x0=dict(N=1.2, t=0.6),
                               bounds=dict(N=(0.1, 5), t=(0.05, 2)), n_runs=1, pbar=False, gradient=gradient)
            inf.run()
            results.append((np.array(list(inf.params_inferred.values())), inf.fisher_information().to_numpy()))

        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-4)

    def test_plot_pop_sizes_draws_the_true_trajectory(self):
        """The true trajectory is drawn dashed in the colour of the inferred one with the same label."""
        import matplotlib.pyplot as plt

        inf = self.get_fast_inference()
        inf.run()

        truth = pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 0.5}})
        t = np.linspace(0, 2, 101)
        ax = inf.plot_pop_sizes(t=t, interval=None, show=False, truth=truth)

        inferred, true = ax.get_lines()
        self.assertEqual('pop_0 (true)', true.get_label())
        self.assertEqual('--', true.get_linestyle())
        self.assertEqual(inferred.get_color(), true.get_color())
        np.testing.assert_array_equal(np.where(t < 0.5, 1, 0.5), true.get_ydata())

        plt.close('all')

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


def _single_ne_inference(**kwargs) -> pg.Inference:
    """
    Single-parameter inference of the population size of one deme from its mean tree height.

    :param kwargs: Keyword arguments overriding the defaults.
    :return: Inference object.
    """
    return pg.Inference(**dict(
        bounds={'Ne': (0.5, 2.0)},
        x0={'Ne': 1.2},
        coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}})),
        loss=lambda coal, obs: (coal.tree_height.mean - obs) ** 2 if obs < 1.5 else float('nan'),
        observation=1.0,
        resample=lambda obs, rng: obs * rng.uniform(0.9, 1.1),
        n_runs=1,
        n_bootstraps=2,
        parallelize=False,
        pbar=False
    ) | kwargs)


def test_a_zero_population_size_reached_by_the_line_search_does_not_discard_the_run():
    """A population size of zero reached by the line search is penalised like any invalid model. Regression: it
    raised a plain ValueError, which the loss wrapper let through, and the run failed with RuntimeError."""
    inf = _single_ne_inference(
        bounds={'Ne': (0, 2)},
        x0={'Ne': 1.0},
        loss=lambda coal, obs: float(coal.tree_height.mean)
    )

    inf.run()

    assert 0 < inf.params_inferred['Ne'] < 1
    assert inf.loss_inferred < 1


def test_an_overflowing_population_size_trajectory_is_a_model_error():
    """A trajectory whose population size overflows to infinity raises ModelError, which inference penalises."""
    demography = pg.Demography(events=[pg.ExponentialPopSizeChanges(
        initial_size={'pop_0': 1}, growth_rate=-1e3, start_time=0, end_time=10
    )])

    with pytest.raises(pg.ModelError, match='negative or not finite'):
        list(demography.epochs)


def test_demography_plots_skip_bootstrap_replicates_without_estimate():
    """Bootstrap replicates with NaN parameters are left out of the demography plots. Regression: they were passed to
    the coalescent, whose validation raised ValueError."""
    inf = _single_ne_inference(resample=lambda obs, rng: 2.0)
    inf._run()
    inf.bootstrap()
    inf.add_bootstrap({'Ne': 1.1})

    inferred, bootstraps, _ = inf._plot_demography_data(kind='pop_sizes')

    assert len(bootstraps) == 1

    for plot in (inf.plot_demography, inf.plot_pop_sizes, inf.plot_migration):
        plot(show=False)


def test_add_bootstrap_stores_no_estimate_for_an_everywhere_invalid_loss():
    """A replicate from create_bootstrap whose loss is invalid at every point has NaN parameters, as in bootstrap().
    Regression: its start point, the main estimate, was stored as a finite replicate."""
    inf = _single_ne_inference(resample=lambda obs, rng: 2.0)
    inf._run()

    replicate = inf.create_bootstrap(index=0)
    replicate.run()
    inf.add_bootstrap(replicate)

    assert inf.bootstraps[inf.param_names].isna().all().all()
    assert 'success: False' in inf.bootstraps['result'][0]
    assert replicate.params_inferred['Ne'] == pytest.approx(inf.params_inferred['Ne'])


def test_a_raising_bootstrap_replicate_does_not_discard_the_others():
    """A replicate whose optimization raises gets NaN parameters, and the other replicates are kept. Regression: the
    exception aborted bootstrap() and no replicate was stored."""
    observations = iter([1.1, 3.0])

    def loss(coal, obs):
        if obs > 2:
            raise RuntimeError('boom')

        return float((coal.tree_height.mean - obs) ** 2)

    inf = _single_ne_inference(loss=loss, resample=lambda obs, rng: next(observations))
    inf._run()
    inf.bootstrap()

    assert inf.bootstraps['Ne'][0] == pytest.approx(1.1, rel=1e-3)
    assert np.isnan(inf.bootstraps['Ne'][1])
    assert 'boom' in inf.bootstraps['result'][1]


def _raising_loss(coal, obs):
    """Loss that raises for every parameter value at an observation above 1.5."""
    if obs > 1.5:
        raise RuntimeError('boom')

    return float((coal.tree_height.mean - obs) ** 2)


@pytest.mark.parametrize('loss', [_raising_loss, None], ids=['raising', 'invalid'])
def test_failed_bootstrap_replicate_gives_the_same_row_on_both_paths(loss):
    """A replicate whose loss raises, or is invalid, at every point gives the same row from bootstrap() and from
    create_bootstrap(), run() and add_bootstrap(). Regression: the distributed replicate of a raising loss raised
    RuntimeError in run() and could not be merged."""
    kwargs = dict(resample=lambda obs, rng: 2.0, n_bootstraps=1) | ({} if loss is None else dict(loss=loss))

    local = _single_ne_inference(**kwargs)
    local._run()
    local.bootstrap()

    distributed = _single_ne_inference(**kwargs)
    distributed._run()
    replicate = pg.Inference.from_json(distributed.create_bootstrap(index=0).to_json())
    replicate.run()
    distributed.add_bootstrap(replicate)

    assert np.isnan(distributed.bootstraps['Ne'][0])
    assert distributed.bootstraps['loss'][0] == local.bootstraps['loss'][0]
    assert distributed.bootstraps['result'][0] == local.bootstraps['result'][0]
    pd.testing.assert_frame_equal(local.bootstraps, distributed.bootstraps)


def test_payload_with_encoded_optimize_result_restores():
    """A payload whose result was encoded as an OptimizeResult, as v1.2.0 wrote it, restores to a usable result.
    Regression: the instance dictionary came back as an item '__dict__', and str(result) and add_bootstrap raised
    ValueError."""
    inf = _single_ne_inference(loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2))
    inf._run()

    getstate = pg.Inference.__getstate__

    with mock.patch.object(pg.Inference, '__getstate__', lambda self: getstate(self) | {'result': self.result}):
        restored = pg.Inference.from_json(inf.to_json())

    assert '__dict__' not in restored.result
    assert str(restored.result)

    inf.add_bootstrap(restored)

    assert inf.bootstraps['Ne'][0] == pytest.approx(inf.params_inferred['Ne'])


@pytest.mark.parametrize("method", ['BFGS', 'CG', 'Newton-CG', 'dogleg', 'trust-exact', 'trust-ncg', 'trust-krylov'])
def test_methods_that_ignore_bounds_or_need_a_jacobian_are_rejected(method):
    """These methods failed every run with 'Jacobian is required' or stepped out of the bounds, and run() blamed the
    loss function."""
    with pytest.raises(ValueError, match="does not honour bounds"):
        _single_ne_inference(method_mle=method)


@pytest.mark.parametrize("method", ['Nelder-Mead', 'Powell', 'TNC', 'SLSQP', 'COBYLA'])
def test_bounded_methods_are_accepted(method):
    """The methods that honour bounds without an analytic Jacobian run."""
    inf = _single_ne_inference(method_mle=method, loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2))
    inf.run()

    assert 0.5 <= inf.params_inferred['Ne'] <= 2.0


def test_trust_constr_is_rejected():
    """trust-constr walked onto the penalty of an invalid region, which its barrier term made attractive, and stopped
    there, so that run() reported a penalized point at t + d = 1.14 as the estimate of the bottleneck, or raised when
    building its coalescent."""
    with pytest.raises(ValueError, match="does not honour bounds"):
        _bottleneck_inference(method_mle='trust-constr')


def test_infinite_bound_raises_when_a_start_point_is_sampled():
    """An infinite bound raised numpy's OverflowError wherever a start point was sampled."""
    with pytest.raises(ValueError, match="non-finite bound"):
        _single_ne_inference(bounds={'Ne': (0.5, np.inf)}, x0=None)

    with pytest.raises(ValueError, match="non-finite bound"):
        _single_ne_inference(bounds={'Ne': (0.5, np.inf)}, n_runs=2).run()

    inf = _single_ne_inference(bounds={'Ne': (0.5, np.inf)},
                               loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2))
    inf.run()

    assert inf.params_inferred['Ne'] == pytest.approx(1.0, rel=1e-3)


def test_model_error_in_a_loss_evaluation_logs_one_warning(caplog):
    """A ModelError logged a second warning saying that the loss function returned NaN."""
    import logging

    def get_dist(Ne):
        raise pg.ModelError('invalid')

    loss = pg.Inference._get_loss_function(observation=None, x0={'Ne': 1.0}, pbar=None, get_dist=get_dist,
                                           get_loss=lambda coal, obs: 0.0, logger=logging.getLogger('test_penalty'))

    with caplog.at_level(logging.WARNING, logger='test_penalty'):
        assert loss([1.0]) == 1e100

    assert [r.getMessage() for r in caplog.records] == ['The model raised "invalid" for {\'Ne\': 1.0}; substituting a '
                                                        'finite penalty']


def test_cached_start_point_on_a_zero_population_size_bound():
    """With caching, the state spaces were built at x0 on every evaluation. A population size of zero there made the
    coalescent raise each time, so every evaluation was penalised and run() raised ModelError."""
    inf = _single_ne_inference(
        bounds={'Ne': (0.0, 2.0)},
        x0={'Ne': 0.0},
        loss=lambda coal, obs: float((coal.tree_height.mean - obs) ** 2),
        cache=True
    )
    inf.run()

    assert inf.params_inferred['Ne'] == pytest.approx(1.0, abs=1e-3)


def test_single_epoch_mutation_configs_use_the_rates_of_their_own_coalescent():
    """Coalescents from ``Inference(cache=True).get_coal`` share their state spaces, which hold the rates of the
    coalescent built last. Regression: the single-epoch mutation-configuration probabilities of an earlier coalescent,
    first evaluated after a later ``get_coal`` call, used the rates of the later one."""
    def coal(Ne):
        return pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne}}))

    inf = pg.Inference(coal=coal, loss=lambda c, o: 0.0, bounds=dict(Ne=(0.1, 10)), x0=dict(Ne=1.0), cache=True)

    first = inf.get_coal(Ne=1.0)
    _ = first.sfs.mean
    second = inf.get_coal(Ne=3.0)
    _ = second.sfs.mean

    for c, Ne in [(first, 1.0), (second, 3.0)]:
        expected = coal(Ne).sfs.get_mutation_config([1, 1, 0], theta=0.7)
        assert c.sfs.get_mutation_config([1, 1, 0], theta=0.7) == pytest.approx(expected, rel=1e-10)


def test_x0_with_a_parameter_outside_bounds_is_rejected():
    """Regression: a key of ``x0`` absent from ``bounds`` was silently dropped, so the callback received its default
    value for that parameter."""
    kwargs = dict(coal=lambda a, b=1.0: pg.Coalescent(n=2), loss=lambda c, o: 0.0, bounds=dict(a=(0.1, 10)))

    with pytest.raises(ValueError, match='unknown'):
        pg.Inference(x0=dict(a=1.0, b=3.0), **kwargs)

    with pytest.raises(ValueError, match='unknown'):
        pg.Inference(x0=dict(a=1.0), **kwargs).create_run(x0=dict(a=1.0, b=3.0))


def test_indexed_runs_are_reproducible_without_stored_entropy():
    """A payload serialized before the entropy was stored restores ``_entropy`` as ``None``. Regression: indexed runs
    and bootstraps of a seeded object were then drawn from fresh entropy on every call."""
    def make():
        return pg.Inference(coal=lambda a: pg.Coalescent(n=2), loss=lambda c, o: 0.0, bounds=dict(a=(0.1, 10)),
                            seed=42)

    old = make()
    old.__dict__.pop('_entropy')

    assert old.create_run(index=3).x0 == old.create_run(index=3).x0 == make().create_run(index=3).x0


def _linear_poisson_inference(gradient: str, seed: int = None) -> pg.Inference:
    """
    Inference of a constant population size ``Ne`` from an SFS whose expected counts are proportional to ``Ne``, so
    that the Hessian of the Poisson negative log-likelihood is :math:`\\sum_i k_i / Ne^2` for observed counts
    :math:`k_i`.
    """
    return pg.Inference(
        bounds=dict(Ne=(0.1, 10)),
        x0=dict(Ne=1),
        coal=lambda Ne: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': Ne})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
        ),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        n_runs=1,
        gradient=gradient,
        seed=seed,
        pbar=False
    )


@pytest.mark.parametrize('gradient', ['exact', 'finite_differences'])
def test_fisher_information_is_the_hessian_of_the_negative_log_likelihood(gradient):
    """The observed information of a Poisson model with means proportional to Ne equals sum(k) / Ne^2, from the exact
    gradient and from finite differences alike, and the standard error is its inverse square root."""
    inf = _linear_poisson_inference(gradient)
    inf.run()

    ne = inf.params_inferred['Ne']
    expected = (180 + 95 + 70 + 52) / ne ** 2

    np.testing.assert_allclose(inf.fisher_information().loc['Ne', 'Ne'], expected, rtol=1e-5)
    np.testing.assert_allclose(inf.standard_errors()['Ne'], expected ** -0.5, rtol=1e-5)


def test_fisher_information_of_a_two_parameter_model_matches_central_differences_of_the_loss():
    """The information of a two-epoch model matches a central second difference of the loss in each entry."""
    inf = InferenceTestCase.get_basic_inference(InferenceTestCase(), dict(gradient='exact'))
    inf.run()

    x = inf.params_inferred
    h = 1e-3

    def f(**shift):
        return inf.loss(inf.get_coal(**{k: v + shift.get(k, 0) for k, v in x.items()}), inf.observation)

    t_t = (f(t=h) - 2 * f() + f(t=-h)) / h ** 2
    t_ne = (f(t=h, Ne=h) - f(t=h, Ne=-h) - f(t=-h, Ne=h) + f(t=-h, Ne=-h)) / (4 * h ** 2)

    info = inf.fisher_information()

    np.testing.assert_allclose(info.loc['t', 't'], t_t, rtol=1e-3)
    np.testing.assert_allclose(info.loc['t', 'Ne'], t_ne, rtol=1e-3)
    np.testing.assert_allclose(info.loc['Ne', 't'], t_ne, rtol=1e-3)


def test_covariance_warns_about_a_parameter_on_its_bound(caplog):
    """A parameter estimated on its bound is named in a warning, since the inverse information does not describe its
    uncertainty there."""
    inf = _linear_poisson_inference('exact')
    inf.bounds = dict(Ne=(0.1, 0.5))
    inf.run()

    assert inf.params_inferred['Ne'] == 0.5

    inf.covariance()

    assert any('lie on a bound' in r.getMessage() and 'Ne' in r.getMessage() for r in caplog.records)


def test_fisher_information_requires_a_run():
    """The information is taken at the estimate, so it needs a run."""
    with pytest.raises(RuntimeError, match='run'):
        _linear_poisson_inference('exact').fisher_information()


def test_fisher_information_takes_the_exact_gradient():
    """With the exact gradient, the information differences the gradient of Gradient.compute without differencing the
    loss, and with finite differences it never asks for the exact gradient."""
    from phasegen.gradient import Gradient

    for gradient, expected in [('exact', True), ('finite_differences', False)]:
        inf = _linear_poisson_inference(gradient)
        inf.run()
        calls = []
        exact = Gradient._exact

        with mock.patch.object(Gradient, '_exact', lambda self, *a: calls.append(1) or exact(self, *a)):
            inf.fisher_information()

        assert bool(calls) == expected


def test_standard_error_of_a_parameter_the_loss_ignores_is_nan(caplog):
    """A parameter the loss does not depend on has an undefined standard error, with a warning, while the others keep
    theirs. Regression: the pseudo-inverse reported about zero."""
    import logging

    inf = pg.Inference(
        bounds=dict(Ne=(0.1, 10), z=(0, 1)),
        x0=dict(Ne=1, z=0.5),
        coal=lambda Ne, z: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': Ne})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
        ),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        n_runs=1,
        pbar=False
    )
    inf.run()

    with caplog.at_level(logging.WARNING, logger='phasegen'):
        se = inf.standard_errors()

    assert np.isnan(se['z'])
    assert np.isfinite(se['Ne']) and se['Ne'] > 0
    assert any('not positive definite' in r.getMessage() for r in caplog.records)


def test_fisher_information_follows_a_changed_observation():
    """The cached information is recomputed when the observation changes. Regression: it was keyed on the estimate
    only, so doubling the counts left it unchanged."""
    inf = _linear_poisson_inference('exact')
    inf.run()
    before = inf.fisher_information().loc['Ne', 'Ne']

    inf.observation = pg.SFS([0, 360, 190, 140, 104, 0])

    assert inf.fisher_information().loc['Ne', 'Ne'] == pytest.approx(2 * before, rel=1e-5)


def test_plot_bootstraps_bins_each_parameter_over_its_own_range():
    """Each histogram is binned over the range of its own parameter, and a colour can be passed. Regression: one set
    of bins spanned all parameters, and a colour raised a TypeError."""
    import matplotlib.pyplot as plt

    inf = InferenceTestCase.get_basic_inference(InferenceTestCase())
    inf.run()
    rng = np.random.default_rng(1)
    inf.bootstraps = pd.DataFrame({'t': rng.uniform(0, 0.001, 30), 'Ne': rng.uniform(0, 1, 30)})

    axes = inf.plot_bootstraps(show=False, kwargs=dict(bins=10, color='k'))
    widths = [max(p.get_width() for p in ax.patches) for ax in axes]

    assert widths[0] == pytest.approx(np.ptp(inf.bootstraps['t']) / 10, rel=1e-6)
    assert widths[1] == pytest.approx(np.ptp(inf.bootstraps['Ne']) / 10, rel=1e-6)
    plt.close('all')


def test_plot_bootstraps_rejects_fewer_axes_than_parameters():
    """Too few axes for the parameters raise a ValueError naming the required count. Regression: a single Axes with
    two parameters raised an IndexError."""
    import matplotlib.pyplot as plt

    inf = InferenceTestCase.get_basic_inference(InferenceTestCase())
    inf.bootstraps = pd.DataFrame({'t': np.linspace(0, 1, 10), 'Ne': np.linspace(0.1, 1, 10)})
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match='needs 2 axes, got 1'):
        inf.plot_bootstraps(ax=ax, show=False)

    plt.close('all')


def test_plot_bootstraps_returns_one_axis_per_parameter():
    """The returned axes are those of the parameters, without the hidden padding of the grid. Regression: five
    parameters on a grid of three columns returned six axes."""
    import matplotlib.pyplot as plt

    names = ['a', 'b', 'c', 'd', 'e']
    inf = InferenceTestCase.get_basic_inference(InferenceTestCase(), dict(
        x0={k: 0.5 for k in names},
        bounds={k: (0.1, 1) for k in names},
        coal=lambda **kw: pg.Coalescent(n=10)
    ))
    inf.bootstraps = pd.DataFrame({k: np.linspace(0.1, 1, 10) for k in names})

    axes = inf.plot_bootstraps(show=False)

    assert len(axes) == 5
    assert all(ax.get_visible() for ax in axes)
    plt.close('all')


def test_inference_payload_without_a_gradient_mode_uses_the_exact_gradient():
    """An inference serialized before the gradient option existed restores with the exact gradient, the class default,
    and without a cached Fisher information."""
    import jsonpickle

    inf = _linear_poisson_inference('finite_differences')
    payload = jsonpickle.decode(jsonpickle.encode(inf, keys=True), keys=True)
    payload.__dict__.pop('gradient', None)
    payload.__dict__.pop('_fisher_information', None)
    restored = jsonpickle.decode(jsonpickle.encode(payload, keys=True), keys=True)

    assert restored.gradient == 'exact'
    assert restored._fisher_information is None


def test_log_scale_reaches_the_same_optimum_on_the_original_scale():
    """Optimizing parameters as their logarithms, with exact and with finite-difference gradients, reaches the optimum
    of the original scale and reports it there."""
    results = {}
    for log_scale in (False, True, ['Ne']):
        for gradient in ('exact', 'finite_differences'):
            inf = InferenceTestCase.get_basic_inference(
                InferenceTestCase(), dict(bounds=dict(t=(0.01, 4), Ne=(0.1, 1)), log_scale=log_scale, gradient=gradient)
            )
            inf.run()
            results[(str(log_scale), gradient)] = (inf.loss_inferred, inf.params_inferred)
            assert all(inf.bounds[k][0] <= v <= inf.bounds[k][1] for k, v in inf.params_inferred.items())

    best = min(loss for loss, _ in results.values())
    for loss, params in results.values():
        assert loss == pytest.approx(best, rel=1e-6)


def test_log_scale_needs_a_positive_lower_bound():
    """A parameter whose lower bound is not positive cannot be optimized on a logarithmic scale."""
    with pytest.raises(ValueError, match='logarithmic scale'):
        InferenceTestCase.get_basic_inference(InferenceTestCase(), dict(log_scale=True))


def test_log_scale_gradient_is_that_of_the_logarithms():
    """The gradient the optimizer receives on a logarithmic scale is the derivative by the logarithms of the
    parameters, against central differences of the function it minimizes."""
    import scipy.optimize as opt

    captured = []
    minimize = opt.minimize

    def capture(fun, x0, **kwargs):
        captured.append((fun, np.array(x0)))
        return minimize(fun, x0, **kwargs)

    inf = InferenceTestCase.get_basic_inference(InferenceTestCase(), dict(bounds=dict(t=(0.01, 4), Ne=(0.1, 1)),
                                                                          x0=dict(t=0.5, Ne=0.5), log_scale=['Ne']))
    with mock.patch('phasegen.inference.opt.minimize', capture):
        inf.run()

    fun, y = captured[0]
    _, g = fun(y)
    h = 1e-5
    numeric = [(fun(y + h * e)[0] - fun(y - h * e)[0]) / (2 * h) for e in np.eye(len(y))]

    np.testing.assert_allclose(g, numeric, rtol=1e-5, atol=1e-8 * np.abs(numeric).max())


def _two_epoch_log_scale_inference(bounds: dict, log_scale, **kwargs) -> pg.Inference:
    """
    A two-epoch inference of the sizes ``Ne`` and ``r`` of the epochs before and after time 1.

    :param bounds: The bounds of ``Ne`` and ``r``.
    :param log_scale: The parameters on a logarithmic scale.
    :param kwargs: Further arguments of the inference.
    """
    return pg.Inference(
        bounds=bounds,
        x0=dict(Ne=0.2, r=1.0),
        coal=lambda Ne, r: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne, 1: r}})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
        ),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        n_runs=1,
        pbar=False,
        log_scale=log_scale,
        **kwargs
    )


def test_log_scale_estimate_on_a_bound_is_the_bound(caplog):
    """An estimate on the logarithmic scale that the optimizer leaves on its upper bound is that bound exactly, so the
    Fisher information and the trajectory intervals are defined and the on-bound warning is logged. Here
    exp(log(0.34)) exceeded 0.34 by one ulp, so the estimate lay outside the bounds and the Fisher information raised
    ``ValueError('x0 violates bound constraints')``."""
    inf = _two_epoch_log_scale_inference(dict(Ne=(0.1, 0.34), r=(0.1, 5)), True)
    inf.run()

    assert inf.params_inferred['Ne'] == 0.34

    inf.fisher_information()
    inf.trajectory_intervals()

    assert 'lie on a bound' in caplog.text


def test_log_scale_accepts_a_single_name():
    """A single parameter name as ``log_scale`` puts that parameter on the logarithmic scale. It was split into its
    characters, raising ``ValueError`` for the unknown parameters ``['N', 'e']``."""
    inf = _two_epoch_log_scale_inference(dict(Ne=(0.1, 5), r=(0.1, 5)), 'Ne')

    assert inf.log_scale == ('Ne',)


def test_log_scale_result_gradient_is_on_the_original_scale():
    """The gradient of the optimization result is that of the loss by the parameters on the original scale, and no
    inverse Hessian is reported. It was the gradient by the logarithms, x dL/dx, with the inverse Hessian in the
    logarithms."""
    for log_scale in (True, ['Ne']):
        inf = _two_epoch_log_scale_inference(dict(Ne=(0.1, 5), r=(0.1, 5)), log_scale)
        inf.run()

        params = inf.params_inferred
        assert all(inf.bounds[k][0] < v < inf.bounds[k][1] for k, v in params.items())

        _, expected = pg.Gradient(inf.get_coal, lambda coal: inf.loss(coal, inf.observation)).compute(params)

        # the gradient at the optimum is of order 1e-6, where the rescaled rate matrices cached across evaluations
        # leave it uncertain by about 0.3%, while the gradient by the logarithms differs by a factor of 0.83 or 1.24
        np.testing.assert_allclose(inf.result.jac, expected, rtol=2e-2)
        assert 'hess_inv' not in inf.result


def test_trajectory_intervals_by_the_delta_method():
    """The standard error of a population size trajectory is that of the delta method: for an epoch size estimated
    directly, its own standard error; for a time-scaled size, the error propagated through the scaling. The limits
    are those of the normal interval of the logarithm."""
    from scipy.stats import norm

    inf = pg.Inference(
        bounds=dict(Ne=(0.1, 10), r=(0.1, 10)),
        x0=dict(Ne=1, r=2),
        coal=lambda Ne, r: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne, 0.5: Ne * r}})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
        ),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        n_runs=1,
        pbar=False
    )
    inf.run()

    t = np.array([0.1, 1.0])
    intervals = inf.trajectory_intervals(t=t)
    cov = inf.covariance().to_numpy()
    Ne, r = inf.params_inferred['Ne'], inf.params_inferred['r']

    # N = Ne before 0.5 and N = Ne r after it, so that log N has the gradients (1 / Ne, 0) and (1 / Ne, 1 / r)
    se_before = np.sqrt(cov[0, 0])
    grad_after = np.array([1 / Ne, 1 / r])
    se_after = Ne * r * np.sqrt(grad_after @ cov @ grad_after)

    np.testing.assert_allclose(intervals['standard error'], [se_before, se_after], rtol=1e-5)

    # the limits of a positive trajectory are those of the normal interval of its logarithm, exponentiated
    half = norm.ppf(0.975) * intervals['standard error'] / intervals['estimate']
    np.testing.assert_allclose(intervals['lower'], intervals['estimate'] * np.exp(-half), rtol=1e-6)
    np.testing.assert_allclose(intervals['upper'], intervals['estimate'] * np.exp(half), rtol=1e-6)


def test_plot_pop_sizes_with_fisher_intervals():
    """The trajectory plot draws the Fisher intervals as one shaded band per trajectory."""
    import matplotlib.pyplot as plt

    inf = _linear_poisson_inference('exact')
    inf.run()
    ax = inf.plot_pop_sizes(show=False, interval='fisher')

    assert len(ax.collections) == 1
    plt.close('all')

    with pytest.raises(ValueError, match='interval'):
        inf.plot_pop_sizes(show=False, interval='bands')


def test_deprecated_include_bootstraps_maps_to_interval(caplog):
    """The deprecated include_bootstraps of the trajectory plots, by keyword or as a boolean in the position of
    interval, maps to interval='bootstrap' or None and logs a warning."""
    import matplotlib.pyplot as plt

    inf = _single_ne_inference(resample=lambda obs, rng: 2.0)
    inf._run()
    inf.bootstrap()
    caplog.clear()

    assert inf._deprecated_interval('fisher', None) == 'fisher'
    assert not any('include_bootstraps' in r.getMessage() for r in caplog.records)

    with caplog.at_level(logging.WARNING, logger='phasegen'):
        assert inf._deprecated_interval('bootstrap', False) is None
        assert inf._deprecated_interval(True, None) == 'bootstrap'
        assert inf._deprecated_interval(False, None) is None

        inf.plot_pop_sizes(show=False, include_bootstraps=False)
        inf.plot_demography(None, False, show=False)

    assert sum('include_bootstraps is deprecated' in r.getMessage() for r in caplog.records) == 5
    plt.close('all')


def test_log_scale_start_points_are_log_uniform():
    """Start points of a parameter on a logarithmic scale are uniform in its logarithm, those of the others uniform and
    drawn from the same generator position. Regression: they were uniform on the linear scale, so with bounds
    (0.01, 100) about 99% of them lay above 1."""
    kwargs = dict(coal=lambda a, b: pg.Coalescent(n=2), loss=lambda c, o: 0.0,
                  bounds=dict(a=(0.01, 100), b=(0.01, 100)), seed=7)
    inf = pg.Inference(log_scale=['a'], **kwargs)
    linear = pg.Inference(**kwargs)

    samples = pd.DataFrame([inf._sample() for _ in range(400)])
    reference = pd.DataFrame([linear._sample() for _ in range(400)])

    # log10 of a log-uniform draw on (0.01, 100) is uniform on (-2, 2)
    assert 0.4 < (samples['a'] < 1).mean() < 0.6
    assert abs(np.log10(samples['a']).mean()) < 0.2
    assert samples['a'].between(0.01, 100).all()
    assert (reference['a'] > 1).mean() > 0.95

    # the parameter on the linear scale draws the same values, and the start point drawn at construction is log-uniform
    np.testing.assert_array_equal(samples['b'], reference['b'])
    assert 0.4 < np.mean([pg.Inference(log_scale=True, **(kwargs | dict(seed=s))).x0['a'] < 1
                          for s in range(200)]) < 0.6


def _bottleneck_before_split(t: float, d: float) -> pg.Coalescent:
    """
    A bottleneck of size 0.1 from ``t`` to ``t + d``, invalid where it would end after a dated split at time 1.
    """
    if t + d > 1:
        raise pg.ModelError('The bottleneck ends after the split.')

    return pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: 0.1, t + d: 1}}))


def _bottleneck_inference(**kwargs) -> pg.Inference:
    """
    Inference of the bottleneck of :func:`_bottleneck_before_split` from its expected SFS at ``t = 0.3`` and
    ``d = 0.6``, next to the invalid region ``t + d > 1`` inside the bounds.
    """
    observation = pg.SFS(np.round(1000 * _bottleneck_before_split(0.3, 0.6).sfs.mean.data))

    return pg.Inference(**(dict(
        bounds=dict(t=(0.0, 1.0), d=(0.01, 1.0)),
        x0=dict(t=0.2, d=0.3),
        coal=_bottleneck_before_split,
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=1000 * coal.sfs.mean.polymorphic
        ),
        observation=observation,
        n_runs=1,
        pbar=False
    ) | kwargs))


@pytest.mark.parametrize('method', ['L-BFGS-B', 'TNC', 'SLSQP'])
@pytest.mark.parametrize('gradient', ['exact', 'finite_differences'])
def test_gradient_optimization_backtracks_from_an_invalid_region(gradient, method):
    """The gradient methods step into a region where the model raises and backtrack to the optimum next to it.
    Regressions: the penalty of 1e100 there collapsed the interpolated step of the line search, so L-BFGS-B stopped at
    its start point after one iteration and reported convergence. TNC with the forward differences of scipy, whose
    steps shared the penalty of a nearby invalid point, stopped at a loss of 28.9469."""
    inf = _bottleneck_inference(gradient=gradient, method_mle=method)
    inf.run()

    # the loss is 28.95 at the parameters the observation was rounded from, and Nelder-Mead reaches 28.94593
    assert inf.result.nit > 3
    assert inf.loss_inferred < 28.946
    assert inf.params_inferred['t'] == pytest.approx(0.3, abs=0.02)


@pytest.mark.parametrize('smooth', [False, True])
def test_exact_gradient_at_an_invalid_point_is_zero(smooth):
    """The exact-gradient loss gives an invalid point a zero gradient once the current iterate is valid, and a smooth
    penalty lies just above that iterate. Regression: it differenced the penalty of 1e100, which gave a gradient of
    order 1e108."""
    inf = _bottleneck_inference()
    x0 = dict(t=0.2, d=0.3)
    loss = pg.Inference._get_loss_function(inf.observation, x0, None, inf.get_coal, inf.loss,
                                           **(dict(smooth=True) if smooth else {}))
    f = pg.Inference._get_loss_and_gradient(loss, x0, inf.bounds, None, inf.get_coal, inf.loss, inf.observation)

    value, _ = f(np.array([0.2, 0.3]))
    # just inside the invalid region, so that a difference of the penalty reaches the valid region
    penalty, grad = f(np.array([0.4, 0.600001]))

    np.testing.assert_array_equal(grad, [0, 0])
    assert value < penalty < value * (1 + 1e-5) if smooth else penalty == 1e100


def test_penalty_is_flat_without_a_gradient():
    """A gradient-free optimization keeps the penalty of 1e100, which ranks an invalid point last."""
    inf = _bottleneck_inference()
    loss = pg.Inference._get_loss_function(inf.observation, dict(t=0.2, d=0.3), None, inf.get_coal, inf.loss)

    loss([0.2, 0.3])

    assert loss([0.5, 0.6]) == 1e100


def test_disagreeing_restarts_of_a_non_identifiable_model_warn(caplog):
    """Runs reaching the best loss at different estimates are reported, naming the parameters, when the loss depends on
    a product of two parameters only."""
    import logging

    inf = pg.Inference(
        bounds=dict(a=(0.2, 5), b=(0.2, 5)),
        coal=lambda a, b: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': a * b})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 100 * coal.sfs.mean.polymorphic),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        n_runs=4,
        seed=3,
        pbar=False
    )

    with caplog.at_level(logging.WARNING):
        inf.run()

    messages = [r.getMessage() for r in caplog.records if 'identifiable' in r.getMessage()]
    assert len(messages) == 1
    assert 'in a, b,' in messages[0]


def test_agreeing_restarts_and_worse_optima_do_not_warn(caplog):
    """Runs of a well-identified model reaching one estimate, and runs ending at a clearly worse local optimum, are not
    reported as disagreeing."""
    import logging

    identified = pg.Inference(
        bounds=dict(Ne=(0.2, 5)),
        coal=lambda Ne: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': Ne})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 100 * coal.sfs.mean.polymorphic),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        n_runs=4,
        seed=3,
        pbar=False
    )

    # the mean tree height of two lineages is Ne, so that the loss has local minima near Ne = 1 and Ne = 3, whose
    # losses differ by about 0.2
    bimodal = pg.Inference(
        bounds=dict(Ne=(0.2, 4)),
        coal=lambda Ne: pg.Coalescent(n=2, demography=pg.Demography(pop_sizes={'pop_0': Ne})),
        loss=lambda coal, obs: ((coal.tree_height.mean - 1) ** 2 * (coal.tree_height.mean - 3) ** 2
                                + 0.1 * coal.tree_height.mean),
        x0=dict(Ne=3.2),
        n_runs=8,
        seed=1,
        pbar=False
    )

    with caplog.at_level(logging.WARNING):
        identified.run()
        bimodal.run()

    assert len(set(np.round(bimodal.runs['Ne'], 1))) > 1
    assert not [r for r in caplog.records if 'identifiable' in r.getMessage()]


def test_fisher_information_follows_a_new_loss_at_a_reused_id():
    """The cached information is recomputed for a new loss, even one allocated at the address of the previous one.
    Regression: it was keyed on the id of the loss, which a new object may take once the old one is collected."""
    inf = _linear_poisson_inference('exact')
    inf.run()
    info = inf.fisher_information()
    names = inf.param_names

    new = lambda coal, obs: 2 * pg.PoissonLikelihood().compute(
        observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
    )

    # the entry an id-keyed cache holds when the new loss takes the address of the collected one
    x = tuple(float(inf.params_inferred[n]) for n in names)
    inf._fisher_information = ((x, inf._observation_key(), id(new), id(inf.coal),
                                tuple(inf.bounds[n] for n in names), inf.gradient), info)
    inf.loss = new

    assert inf.fisher_information().loc['Ne', 'Ne'] == pytest.approx(2 * info.loc['Ne', 'Ne'], rel=1e-5)


def test_fisher_information_cache_is_not_serialized():
    """The cached information, whose key holds the callables, is left out of the serialized state, and a restored
    object computes it afresh."""
    import jsonpickle

    inf = _linear_poisson_inference('exact')
    inf.run()
    info = inf.fisher_information()

    assert '_fisher_information' not in inf.__getstate__()

    restored = jsonpickle.decode(jsonpickle.encode(inf, keys=True), keys=True)

    assert restored._fisher_information is None
    pd.testing.assert_frame_equal(restored.fisher_information(), info)


def test_finite_difference_next_to_an_invalid_point_is_one_sided():
    """Within a step of the invalid region, the finite-difference gradient is a one-sided difference of the loss toward
    the valid side, and an invalid point has a zero gradient. Regression: a step from a penalized point shared its
    penalty over a radius set by the largest parameter, which hid every valid point of a smaller parameter."""
    inf = _bottleneck_inference()
    x0 = dict(t=0.2, d=0.3)
    loss = pg.Inference._get_loss_function(inf.observation, x0, None, inf.get_coal, inf.loss, smooth=True)
    f = pg.Inference._get_loss_and_gradient(loss, x0, inf.bounds, None, inf.get_coal, inf.loss, inf.observation,
                                            exact=False)
    raw = lambda t, d: inf.loss(inf.get_coal(t=t, d=d), inf.observation)

    # 5e-7 below the cliff t + d = 1, so that the forward steps of 1e-6 in t and d are invalid
    t, d = 0.4, 0.6 - 5e-7
    value, grad = f(np.array([t, d]))
    h = 1e-4
    backward = [(raw(t - 2 * h, d) - 4 * raw(t - h, d) + 3 * value) / (2 * h),
                (raw(t, d - 2 * h) - 4 * raw(t, d - h) + 3 * value) / (2 * h)]

    np.testing.assert_allclose(grad, backward, rtol=1e-4)
    assert not loss.invalid
    np.testing.assert_array_equal(f(np.array([0.4, 0.61]))[1], [0, 0])


def test_finite_differences_on_parameters_of_different_scales_reach_the_optimum():
    """An island model with an absolute population size next to a migration rate reaches the optimum of the exact
    gradient with finite differences. Regression: after the line search hit the migration bound of zero, every point
    within 1e-6 times the population size of it, about 0.04, shared its penalty, which covered the whole range of the
    migration rate, so the run stopped at its start point at a loss of 1190.1."""
    def coal(m, Ne):
        return pg.Coalescent(n={'pop_0': 4, 'pop_1': 4}, demography=pg.Demography(
            pop_sizes={'pop_0': 1e4, 'pop_1': Ne}, migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): m}))

    inf = pg.Inference(
        bounds=dict(m=(0.0, 1e-2), Ne=(1e3, 1e5)),
        x0=dict(m=5e-3, Ne=4e4),
        observation=pg.SFS(np.round(1e-1 * coal(2e-5, 2e4).sfs.mean.data)),
        coal=coal,
        loss=lambda c, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 1e-1 * c.sfs.mean.polymorphic),
        gradient='finite_differences',
        n_runs=1,
        pbar=False
    )
    inf.run()

    # the exact gradient reaches 946.1616
    assert inf.loss_inferred < 946.2


@pytest.mark.parametrize('method', ['L-BFGS-B', 'TNC', 'SLSQP'])
@pytest.mark.parametrize('gradient', ['exact', 'finite_differences'])
def test_fixed_parameter_is_held_out_of_the_optimizer(gradient, method, capsys):
    """A parameter whose bounds coincide is held at its value, with a zero gradient, and the methods agree on the free
    one. Regression: scipy wrapped the callback of a problem with fixed variables, which TNC called with an array, so
    every run raised and run() reported that no run returned a finite loss, while L-BFGS-B and SLSQP printed the
    callback to stdout."""
    inf = _bottleneck_inference(bounds=dict(t=(0.0, 1.0), d=(0.6, 0.6)), x0=dict(t=0.2, d=0.6), gradient=gradient,
                                method_mle=method)
    inf.run()

    assert inf.params_inferred['d'] == 0.6
    assert inf.params_inferred['t'] == pytest.approx(0.30044, abs=1e-4)
    assert inf.result.jac[1] == 0
    assert '<bound method' not in capsys.readouterr().out


def test_all_parameters_fixed_evaluates_the_start_point():
    """With every parameter fixed by its bounds, the result is the loss at the start point."""
    inf = _bottleneck_inference(bounds=dict(t=(0.3, 0.3), d=(0.6, 0.6)), x0=dict(t=0.3, d=0.6), method_mle='TNC')
    inf.run()

    assert inf.params_inferred == dict(t=0.3, d=0.6)
    assert inf.loss_inferred == pytest.approx(inf.loss(_bottleneck_before_split(0.3, 0.6), inf.observation), rel=1e-12)


def test_finite_differences_against_a_zero_migration_bound_reach_the_optimum():
    """An island model whose migration rate has a lower bound of zero, where the demography does not absorb, reaches
    the optimum with the finite-difference gradient of the optimizer. A penalty just above the iterate without a shared
    penalty for difference steps stopped this run after two iterations, at a loss of about 1e5, since the difference of
    the penalty reached the loss diverging toward the bound."""
    def coal(m, Ne):
        return pg.Coalescent(n={'pop_0': 4, 'pop_1': 4}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': Ne}, migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): m}))

    inf = pg.Inference(
        bounds=dict(m=(0.0, 2.0), Ne=(0.1, 10.0)),
        x0=dict(m=1.3649788833403462, Ne=2.815657862887755),
        observation=pg.SFS(np.round(1000 * coal(0.02, 2.0).sfs.mean.data)),
        coal=coal,
        loss=lambda c, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 1000 * c.sfs.mean.polymorphic),
        gradient='finite_differences',
        n_runs=1,
        pbar=False
    )
    inf.run()

    assert inf.loss_inferred < 35.7
    assert inf.params_inferred['m'] == pytest.approx(0.02, rel=1e-2)


def _poisson_resample(obs: pg.SFS, rng: np.random.Generator) -> pg.SFS:
    """A Poisson draw of each bin of an SFS."""
    return pg.SFS(rng.poisson(obs.data).astype(float))


def _negative_binomial_resample(obs: pg.SFS, rng: np.random.Generator) -> pg.SFS:
    """A negative binomial draw of each bin of an SFS with the bin as mean and four times the bin as variance."""
    data = np.asarray(obs.data, dtype=float)
    out = np.zeros_like(data)
    out[data > 0] = rng.negative_binomial(data[data > 0] / 3, 1 / 4)
    return pg.SFS(out)


def test_godambe_standard_error_matches_fisher_on_independent_poisson_bins():
    """With independent Poisson bins and means proportional to Ne, the variance of the score equals the observed
    information, so the Godambe standard error agrees with the Fisher one. The ratio of the two has a Monte-Carlo
    standard deviation of about 1 / sqrt(2 (n - 1)) = 0.022 for n = 1000, and the tolerance is four of them."""
    inf = _linear_poisson_inference('exact')
    inf.resample = _poisson_resample
    inf.run()

    ratio = inf.standard_errors(method='godambe', n=1000)['Ne'] / inf.standard_errors()['Ne']

    assert ratio == pytest.approx(1, abs=0.09)


def test_godambe_standard_errors_match_fisher_for_correlated_parameters():
    """With independent Poisson bins, the Godambe covariance of two strongly correlated parameters, the size of the
    first epoch and the ratio of the second to it, agrees with the Fisher covariance, which needs the off-diagonal
    entries of the score covariance. Each ratio of standard errors has a Monte-Carlo standard deviation of about
    1 / sqrt(2 (n - 1)) = 0.022 for n = 1000, and the tolerance is four of them."""
    inf = pg.Inference(
        bounds=dict(Ne=(0.1, 10), r=(0.1, 10)),
        x0=dict(Ne=1, r=2),
        coal=lambda Ne, r: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: Ne, 0.5: Ne * r}})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
        ),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        resample=_poisson_resample,
        n_runs=1,
        seed=1,
        pbar=False
    )
    inf.run()

    ratio = inf.standard_errors(method='godambe', n=1000) / inf.standard_errors()

    np.testing.assert_allclose(ratio[['Ne', 'r']], 1, atol=0.09)


def test_godambe_standard_error_exceeds_fisher_on_overdispersed_bins():
    """With bins resampled with four times the Poisson variance, the variance of the score is four times the observed
    information, so the Godambe standard error is twice the Fisher one, within four Monte-Carlo standard deviations of
    the ratio for n = 1000, widened for the excess kurtosis of the negative binomial."""
    inf = _linear_poisson_inference('finite_differences')
    inf.resample = _negative_binomial_resample
    inf.run()

    ratio = inf.standard_errors(method='godambe', n=1000)['Ne'] / inf.standard_errors()['Ne']

    assert ratio == pytest.approx(2, abs=0.2)


@pytest.mark.parametrize('resample, method, tolerance', [
    (_poisson_resample, 'fisher', 0.14),
    (_negative_binomial_resample, 'godambe', 0.2),
], ids=['poisson fisher', 'overdispersed godambe'])
def test_standard_error_matches_the_spread_of_the_bootstrap(resample, method, tolerance):
    """The standard error from the Fisher covariance with Poisson resampling, and from the Godambe covariance with
    overdispersed resampling, equals the standard deviation of 400 bootstrap estimates. The ratio of the two has a
    Monte-Carlo standard deviation of about 1 / sqrt(2 * 399) = 0.035, and the tolerance is four of them, widened for
    the Monte-Carlo error of the Godambe covariance and the excess kurtosis of the negative binomial."""
    inf = _linear_poisson_inference('exact', seed=1)
    inf.resample = resample
    inf.n_bootstraps = 400
    inf.run()
    inf.bootstrap()

    se = inf.standard_errors(method=method, n=1000)['Ne'] if method == 'godambe' else inf.standard_errors()['Ne']

    assert inf.bootstraps['Ne'].std(ddof=1) / se == pytest.approx(1, abs=tolerance)


def test_godambe_standard_error_on_a_bound_centres_the_scores():
    """On a bound, the score of the observation is not zero, while its variance over Poisson draws still equals the
    observed information. Centring the scores keeps the Godambe standard error at the Fisher one, where the uncentred
    mean of the outer products would add the square of the score of the observation and inflate it about tenfold."""
    inf = _linear_poisson_inference('exact')
    inf.bounds = dict(Ne=(0.1, 0.5))
    inf.resample = _poisson_resample
    inf.run()

    assert inf.params_inferred['Ne'] == 0.5

    ratio = inf.standard_errors(method='godambe', n=1000)['Ne'] / inf.standard_errors()['Ne']

    assert ratio == pytest.approx(1, abs=0.09)


def _godambe_spied(monkeypatch, gradient: str, loss=None) -> Tuple[pd.DataFrame, list, list]:
    """The Godambe covariance of the linear Poisson inference, with the results of ``Gradient._scores`` and, for each
    call of ``Gradient.compute``, whether the covariance of the scores made it."""
    from phasegen.gradient import Gradient

    inf = _linear_poisson_inference(gradient, seed=1)
    if loss is not None:
        inf.loss = loss
    inf.resample = _poisson_resample
    inf.run()

    scores, computed = [], []
    _scores, compute = Gradient._scores, Gradient.compute
    monkeypatch.setattr(Gradient, '_scores', lambda self, *a: scores.append(_scores(self, *a)) or scores[-1])
    def spied(self, *args, **kwargs):
        # the caller, seen through the frame of its comprehension before Python 3.12
        callers = [f.name for f in traceback.extract_stack()[-3:-1]]
        computed.extend([callers[1] == '_score_covariance' or callers == ['_score_covariance', '<listcomp>']])
        return compute(self, *args, **kwargs)

    monkeypatch.setattr(Gradient, 'compute', spied)

    return inf.covariance(method='godambe', n=20), scores, computed


def test_godambe_covariance_shares_the_moment_jacobian_across_resamples(monkeypatch):
    """The exact scores of the resampled observations come from one Jacobian of the moments, without a gradient per
    resample, and agree with the scores by finite differences."""
    exact, scores, computed = _godambe_spied(monkeypatch, 'exact')
    differenced, _, _ = _godambe_spied(monkeypatch, 'finite_differences')

    assert len(scores) == 1 and scores[0] is not None and not any(computed)
    np.testing.assert_allclose(exact.values, differenced.values, rtol=1e-4)


def test_godambe_covariance_of_losses_reading_different_moments(monkeypatch):
    """Resampled losses that read different moments take one exact gradient each, which agree with the scores by
    finite differences."""
    def loss(coal, obs):
        extra = 1e-2 * coal.tree_height.var if obs.data[1] > 180 else 0.0
        return float(pg.PoissonLikelihood().compute(observed=obs.polymorphic,
                                                    modelled=100 * coal.sfs.mean.polymorphic)) + extra

    exact, scores, computed = _godambe_spied(monkeypatch, 'exact', loss)
    differenced, _, _ = _godambe_spied(monkeypatch, 'finite_differences', loss)

    assert scores == [None] and sum(computed) == 20
    np.testing.assert_allclose(exact.values, differenced.values, rtol=1e-4)


def test_godambe_covariance_needs_resample_and_a_known_method():
    """The Godambe covariance draws observations with the resample callback, so it raises without one, and an unknown
    method raises as well."""
    inf = _linear_poisson_inference('exact')
    inf.run()

    with pytest.raises(ValueError, match='resample'):
        inf.standard_errors(method='godambe')

    with pytest.raises(ValueError, match='method'):
        inf.covariance(method='sandwich')

    inf.resample = _poisson_resample

    with pytest.raises(ValueError, match='at least 2'):
        inf.covariance(method='godambe', n=1)


def test_godambe_covariance_is_cached_and_reproducible():
    """The covariance of the score is cached for the same estimate, observation, callback and n, recomputed for
    another n or observation, and reproducible from the seed across objects."""
    calls = []

    def resample(obs, rng):
        calls.append(1)
        return _poisson_resample(obs, rng)

    def make():
        inf = _linear_poisson_inference('exact')
        inf.seed = 7
        inf._entropy = 7
        inf.resample = resample
        inf.run()
        return inf

    inf = make()
    first = inf.standard_errors(method='godambe', n=20)
    assert len(calls) == 20

    assert inf.standard_errors(method='godambe', n=20).equals(first)
    assert len(calls) == 20

    inf.standard_errors(method='godambe', n=30)
    assert len(calls) == 50

    inf.observation = pg.SFS([0, 360, 190, 140, 104, 0])
    inf.standard_errors(method='godambe', n=30)
    assert len(calls) == 80

    assert make().standard_errors(method='godambe', n=20).equals(first)


def test_godambe_standard_error_of_a_parameter_the_loss_ignores_is_nan():
    """A parameter the loss does not depend on has an undefined Godambe standard error, as with the Fisher one, while
    the others keep theirs."""
    inf = pg.Inference(
        bounds=dict(Ne=(0.1, 10), z=(0, 1)),
        x0=dict(Ne=1, z=0.5),
        coal=lambda Ne, z: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': Ne})),
        loss=lambda coal, obs: pg.PoissonLikelihood().compute(
            observed=obs.polymorphic, modelled=100 * coal.sfs.mean.polymorphic
        ),
        observation=pg.SFS([0, 180, 95, 70, 52, 0]),
        resample=_poisson_resample,
        n_runs=1,
        pbar=False
    )
    inf.run()

    se = inf.standard_errors(method='godambe', n=20)

    assert np.isnan(se['z'])
    assert np.isfinite(se['Ne']) and se['Ne'] > 0


def test_trajectory_intervals_and_plot_with_the_godambe_covariance():
    """The trajectory intervals take the Godambe covariance when asked, giving the standard error of the population
    size, which is the parameter itself, and the trajectory plot draws them as a band."""
    import matplotlib.pyplot as plt

    inf = _linear_poisson_inference('exact')
    inf.resample = _negative_binomial_resample
    inf.run()

    intervals = inf.trajectory_intervals(t=np.array([0.5]), method='godambe', n=50)

    np.testing.assert_allclose(intervals['standard error'], inf.standard_errors(method='godambe', n=50)['Ne'],
                               rtol=1e-5)

    ax = inf.plot_pop_sizes(show=False, interval='godambe')

    assert len(ax.collections) == 1
    plt.close('all')


def test_inference_payload_without_a_score_covariance_restores_none():
    """An inference serialized before the Godambe covariance existed restores without a cached score covariance."""
    import jsonpickle

    inf = _linear_poisson_inference('exact')
    payload = jsonpickle.decode(jsonpickle.encode(inf, keys=True), keys=True)
    payload.__dict__.pop('_score_covariance_cache', None)
    restored = jsonpickle.decode(jsonpickle.encode(payload, keys=True), keys=True)

    assert restored._score_covariance_cache is None


def test_score_covariance_follows_a_new_resample_at_a_reused_id():
    """The cached score covariance is recomputed for a new resample callback, even one allocated at the address of the
    previous one. Regression: it was keyed on the ids of the callbacks, which a new object may take once the old one is
    collected, so a noisier callback kept the stale standard error."""
    inf = _linear_poisson_inference('exact')
    inf.resample = _poisson_resample
    inf.run()
    se = inf.standard_errors(method='godambe', n=200)['Ne']
    names = inf.param_names

    def new(obs, rng):
        return _negative_binomial_resample(obs, rng)

    # the entry an id-keyed cache holds when the new callback takes the address of the collected one
    x = tuple(float(inf.params_inferred[n]) for n in names)
    entropy = inf.seed if inf._entropy is None else inf._entropy
    inf._score_covariance_cache = ((x, inf._observation_key(), id(new), id(inf.loss), id(inf.coal),
                                    tuple(inf.bounds[n] for n in names), inf.gradient, 200, entropy),
                                   inf._score_covariance_cache[1])
    inf.resample = new

    # four times the Poisson variance doubles the standard error
    assert inf.standard_errors(method='godambe', n=200)['Ne'] > 1.5 * se


def test_score_covariance_cache_is_not_serialized():
    """The cached score covariance, whose key holds the callables, is left out of the serialized state, and a restored
    object computes the same covariance afresh from the seed."""
    inf = _linear_poisson_inference('exact')
    inf.resample = _poisson_resample
    inf.run()
    cov = inf.covariance(method='godambe', n=20)

    assert '_score_covariance_cache' not in inf.__getstate__()

    restored = pg.Inference.from_json(inf.to_json())

    assert restored._score_covariance_cache is None
    pd.testing.assert_frame_equal(restored.covariance(method='godambe', n=20), cov)


def test_trajectory_interval_of_a_zero_rate_with_a_spread_is_linear_and_clipped(monkeypatch):
    """A migration rate estimated at zero, whose standard error is not zero, takes the limits of the linear scale with
    the lower one clipped at zero: with the variance 0.04 of the rate, the limits are 0 and z(0.975) * 0.2."""
    from scipy.stats import norm

    def coal(m, Ne):
        return pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': Ne}, migration_rates={('pop_0', 'pop_1'): m}))

    inf = pg.Inference(
        bounds=dict(m=(0, 5), Ne=(0.1, 10)),
        x0=dict(m=1, Ne=1),
        coal=coal,
        loss=lambda c, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 1000 * c.sfs.mean.polymorphic),
        observation=pg.SFS(np.round(1000 * coal(0.5, 2).sfs.mean.data)),
        n_runs=1,
        pbar=False
    )
    inf.run()

    inf.params_inferred = inf.params_inferred | dict(m=0.0)
    inf.dist_inferred = inf.get_coal(**inf.params_inferred)
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=['m', 'Ne'], columns=['m', 'Ne'])
    monkeypatch.setattr(inf, 'covariance', lambda method='fisher', n=100: cov)

    row = inf.trajectory_intervals(t=np.array([0.5]), kind='migration').set_index('trajectory').loc['pop_0->pop_1']

    assert row['estimate'] == 0
    assert row['standard error'] == pytest.approx(0.2, rel=1e-6)
    assert row['lower'] == 0
    assert row['upper'] == pytest.approx(norm.ppf(0.975) * 0.2, rel=1e-6)


def test_trajectory_interval_of_a_tiny_rate_next_to_a_zero_bound(monkeypatch):
    """A migration rate of 1e-7 with the lower bound 0 and the standard error 0.1 of its logarithm has that standard
    error on the log scale. An absolute difference step clipped at the bound gave NaN, and one at a bound of 1e-8 an
    error of a factor of two."""
    def coal(m, Ne):
        return pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': Ne}, migration_rates={('pop_0', 'pop_1'): m}))

    inf = pg.Inference(
        bounds=dict(m=(0, 5), Ne=(0.1, 10)),
        x0=dict(m=1, Ne=1),
        coal=coal,
        loss=lambda c, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 1000 * c.sfs.mean.polymorphic),
        observation=pg.SFS(np.round(1000 * coal(0.5, 2).sfs.mean.data)),
        n_runs=1,
        pbar=False
    )
    inf.run()

    m = 1e-7
    inf.params_inferred = inf.params_inferred | dict(m=m)
    inf.dist_inferred = inf.get_coal(**inf.params_inferred)
    cov = pd.DataFrame(np.diag([(0.1 * m) ** 2, 0.01]), index=['m', 'Ne'], columns=['m', 'Ne'])
    monkeypatch.setattr(inf, 'covariance', lambda method='fisher', n=100: cov)

    row = inf.trajectory_intervals(t=np.array([0.5]), kind='migration').set_index('trajectory').loc['pop_0->pop_1']

    assert row['standard error'] == pytest.approx(0.1 * m, rel=1e-4)


def test_trajectory_intervals_of_all_trajectories_at_another_level():
    """With kind='all', the intervals cover the population sizes and both migration directions. The absent direction
    is zero with a zero standard error and limits [0, 0], and the log-scale half-widths at level 0.9 are those at 0.95
    scaled by the ratio z(0.95) / z(0.975) of the normal quantiles."""
    from scipy.stats import norm

    def coal(m, Ne):
        return pg.Coalescent(n={'pop_0': 3, 'pop_1': 3}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': Ne}, migration_rates={('pop_0', 'pop_1'): m}))

    inf = pg.Inference(
        bounds=dict(m=(0.05, 5), Ne=(0.1, 10)),
        x0=dict(m=1, Ne=1),
        coal=coal,
        loss=lambda c, obs: pg.PoissonLikelihood().compute(obs.polymorphic, 1000 * c.sfs.mean.polymorphic),
        observation=pg.SFS(np.round(1000 * coal(0.5, 2).sfs.mean.data)),
        n_runs=1,
        pbar=False
    )
    inf.run()

    t = np.array([0.1, 1.0])
    narrow = inf.trajectory_intervals(t=t, kind='all', level=0.9).set_index(['trajectory', 'time'])
    wide = inf.trajectory_intervals(t=t, kind='all', level=0.95).set_index(['trajectory', 'time'])
    migration = inf.trajectory_intervals(t=t, kind='migration', level=0.9).set_index(['trajectory', 'time'])

    assert set(narrow.index.get_level_values('trajectory')) == {'pop_0', 'pop_1', 'pop_0->pop_1', 'pop_1->pop_0'}
    pd.testing.assert_frame_equal(migration, narrow.loc[['pop_0->pop_1', 'pop_1->pop_0']])

    absent = narrow.loc['pop_1->pop_0']
    np.testing.assert_array_equal(absent[['estimate', 'standard error', 'lower', 'upper']].to_numpy(), 0)

    # the fitted trajectories are positive and take log-scale limits, symmetric about the estimate in the logarithm
    fitted = narrow.index.get_level_values('trajectory').isin(['pop_1', 'pop_0->pop_1'])
    assert (narrow.loc[fitted, 'standard error'] > 0).all()

    ratio = (np.log(narrow['upper'] / narrow['estimate']) / np.log(wide['upper'] / wide['estimate']))[fitted]
    np.testing.assert_allclose(ratio, norm.ppf(0.95) / norm.ppf(0.975), rtol=1e-12)
    np.testing.assert_allclose(np.log(narrow['estimate'] / narrow['lower'])[fitted],
                               np.log(narrow['upper'] / narrow['estimate'])[fitted], rtol=1e-12)


def _quadratic_penalty(Ne: float) -> float:
    """A penalty of 5 (Ne - 1)^2, whose second derivative is 10."""
    return 5 * (Ne - 1) ** 2


@pytest.mark.parametrize('gradient', ['exact', 'finite_differences'])
def test_penalty_is_minimised_with_the_loss(gradient):
    """With a penalty, the estimate minimises the loss plus the penalty: the stationarity condition of the linear
    model, sum(k) / Ne - 100 sum(s) + 10 (Ne - 1) = 0, holds at the estimate, and the reported loss includes the
    penalty."""
    inf = _linear_poisson_inference(gradient)
    inf.penalty = _quadratic_penalty
    inf.run()

    ne = inf.params_inferred['Ne']
    s = pg.Coalescent(n=5).sfs.mean.polymorphic.sum()
    k = 180 + 95 + 70 + 52

    assert -k / ne + 100 * s + 10 * (ne - 1) == pytest.approx(0, abs=1e-3 * k)
    np.testing.assert_allclose(
        inf.loss_inferred, inf.loss(inf.dist_inferred, inf.observation) + _quadratic_penalty(ne), rtol=1e-12
    )


def test_penalty_keeps_the_exact_gradient(caplog):
    """A penalty is differentiated without the coalescent, so the loss keeps its exact gradient."""
    import logging
    caplog.set_level(logging.INFO, logger='phasegen')

    inf = _linear_poisson_inference('exact')
    inf.penalty = _quadratic_penalty
    inf.run()

    assert 'Using the exact gradient of the loss.' in caplog.text
    assert 'finite differences' not in caplog.text


def test_information_includes_the_hessian_of_the_penalty():
    """The information of a penalized fit is the Hessian sum(k) / Ne^2 of the loss at its estimate plus the second
    derivative 10 of the penalty."""
    inf = _linear_poisson_inference('exact')
    inf.penalty = _quadratic_penalty
    inf.run()

    ne = inf.params_inferred['Ne']

    np.testing.assert_allclose(inf.fisher_information().loc['Ne', 'Ne'], (180 + 95 + 70 + 52) / ne ** 2 + 10,
                               rtol=1e-5)


def test_penalty_survives_serialization():
    """The penalty is serialized with the other callables, and a payload without it, as from 2.1.0, restores no
    penalty and serializes and copies again."""
    inf = _linear_poisson_inference('exact')
    inf.penalty = _quadratic_penalty

    assert pg.Inference.from_json(inf.to_json()).penalty(Ne=2) == 5

    state = inf.__getstate__()
    state.pop('penalty_pickled')
    restored = object.__new__(pg.Inference)
    restored.__setstate__(state)

    assert restored.penalty is None
    assert pg.Inference.from_json(restored.to_json()).penalty is None
    assert copy.deepcopy(restored).penalty is None


def test_aic_counts_the_free_parameters():
    """The AIC is twice the loss plus twice the number of parameters whose bounds differ."""
    inf = _two_epoch_log_scale_inference(dict(Ne=(0.1, 5), r=(1, 1)), ())
    inf.run()

    assert inf.aic() == pytest.approx(2 * inf.loss_inferred + 2)


def test_aic_and_likelihood_ratio_test_reject_a_penalty_and_a_missing_run():
    """A penalized loss is not a negative log-likelihood, and neither criterion exists before a run."""
    inf = _linear_poisson_inference('exact')

    with pytest.raises(RuntimeError, match='run'):
        inf.aic()

    inf.penalty = _quadratic_penalty
    inf.run()

    with pytest.raises(ValueError, match='penalty'):
        inf.aic()


def test_likelihood_ratio_test_of_a_fixed_epoch_size():
    """The likelihood-ratio test of the two-epoch model against a null model fixing the size of the second epoch has
    one degree of freedom, the statistic twice the difference of the losses and the chi-squared tail as p-value."""
    from scipy.stats import chi2

    alt = _two_epoch_log_scale_inference(dict(Ne=(0.1, 5), r=(0.1, 5)), ())
    null = _two_epoch_log_scale_inference(dict(Ne=(0.1, 5), r=(1, 1)), ())
    alt.run()
    null.run()

    test = alt.likelihood_ratio_test(null)
    statistic = 2 * (null.loss_inferred - alt.loss_inferred)

    assert test['df'] == 1
    assert test['statistic'] == pytest.approx(statistic, rel=1e-12)
    assert statistic >= 0
    assert test['p-value'] == pytest.approx(chi2.sf(statistic, 1), rel=1e-12)

    with pytest.raises(ValueError, match='fewer free parameters'):
        null.likelihood_ratio_test(alt)

    other = _two_epoch_log_scale_inference(dict(Ne=(0.1, 5), r=(1, 1)), ())
    other.observation = pg.SFS([0, 170, 95, 70, 52, 0])
    other.run()

    with pytest.raises(ValueError, match='different observation'):
        alt.likelihood_ratio_test(other)
