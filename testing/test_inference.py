"""
Test Inference class.
"""
import os
from unittest import TestCase

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

    def test_basic_inference_3_runs_sequential(self):
        """
        Test basic inference with 3 runs in sequence.
        """
        # create inference object
        inf = self.get_basic_inference(dict(n_runs=3, parallelize=False))

        inf.run()

    @pytest.mark.skipif(os.getenv("parallel", True), reason="Not running parallel tests.")
    def test_basic_inference_3_runs_parallel(self):
        """
        Test basic inference with 3 runs in parallel.
        """
        # create inference object
        inf = self.get_basic_inference(dict(n_runs=3, parallelize=True))

        inf.run()

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

    def test_serialize_basic_inference_after_running(self):
        """
        Test serialization of basic inference.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_bootstraps=3))

        inf.run()

        inf.to_file('scratch/test_serialize_run_basic_inference.json')

        inf2 = pg.Inference.from_file('scratch/test_serialize_run_basic_inference.json')

        self.assertAlmostEqual(inf.params_inferred['t'], inf2.params_inferred['t'])
        self.assertAlmostEqual(inf.params_inferred['Ne'], inf2.params_inferred['Ne'])
        self.assertAlmostEqual(inf.loss_inferred, inf2.loss_inferred)
        self.assertDictEqual(inf.bootstraps.var().to_dict(), inf2.bootstraps.var().to_dict())

    @pytest.mark.skipif(os.getenv("parallel", True), reason="Not running parallel tests.")
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
            inf = self.get_basic_inference(dict(do_bootstrap=True, resample=None))

        self.assertEqual(
            'Observation and resample must be provided for automatic bootstrapping.',
            str(context.exception)
        )

    def test_automatic_boostrap_no_observation_raises_error(self):
        """
        Test automatic bootstrap without observation raises error.
        """
        with self.assertRaises(ValueError) as context:
            inf = self.get_basic_inference(dict(do_bootstrap=True, observation=None))

        self.assertEqual(
            'Observation and resample must be provided for automatic bootstrapping.',
            str(context.exception)
        )

    def test_bootstrap_sequential(self):
        """
        Test sequential bootstrap.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_runs=3, parallelize=False, n_bootstraps=10))
        inf.run()

        self.assertEqual(10, len(inf.bootstraps))

        # make sure the bootstraps are different
        self.assertGreater(inf.bootstraps.var().t, 0)

    @pytest.mark.skipif(os.getenv("parallel", True), reason="Not running parallel tests.")
    def test_bootstrap_parallel(self):
        """
        Test parallel bootstrap.
        """
        # create inference object
        inf = self.get_basic_inference(dict(do_bootstrap=True, n_runs=3, parallelize=True, n_bootstraps=10))
        inf.run()

        self.assertEqual(10, len(inf.bootstraps))

        # make sure the bootstraps are different
        self.assertGreater(inf.bootstraps.var().t, 0)

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
        self.assertGreater(inf.bootstraps.var().t, 0)

        inf.plot_bootstraps()

    @pytest.mark.skip("Not working yet.")
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
