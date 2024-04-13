"""
Test matrix exponentiation.
"""
import time
from unittest import TestCase

import numpy as np

import phasegen as pg


class ExpmTestCase(TestCase):
    """
    Test matrix exponentiation.
    """

    def test_expm_different_backends(self):
        """
        Test matrix exponential for medium-sized matrix.
        """

        coal = pg.Coalescent(
            n=10
        )

        A = pg.TensorFlowExpmBackend().compute(coal.block_counting_state_space.S)
        B = pg.SciPyExpmBackend().compute(coal.block_counting_state_space.S)
        C = pg.JaxExpmBackend().compute(coal.block_counting_state_space.S)

        np.testing.assert_array_almost_equal(A, B)
        np.testing.assert_array_almost_equal(A, C)

        pass

    def test_runtime_different_backends(self):
        """
        Test runtime for matrix exponential for different backends. Note that the backends scale differently under
        heavy load.
        """

        def run(backend: pg.ExpmBackend, warm_start: bool = True) -> float:
            """
            Run for backend.

            :param backend: Expm backend
            :param warm_start: Warm start
            :return: Execution time
            """
            # warm start
            if warm_start:
                run(backend, warm_start=False)

            pg.Backend.register(backend)

            start = time.time()

            coal = pg.Coalescent(
                n={'pop_0': 3, 'pop_1': 2},
                model=pg.BetaCoalescent(alpha=1.7),
                demography=pg.Demography(
                    pop_sizes={'pop_0': {0: 1, 1.2: 2}, 'pop_1': {0: 2}},
                    migration_rates={
                        ('pop_0', 'pop_1'): 2,
                        ('pop_1', 'pop_0'): 2
                    }
                )
            )

            _ = coal.sfs.var

            return time.time() - start

        backends = dict(
            scipy64=pg.SciPyExpmBackend(np.float64),
            scipy32=pg.SciPyExpmBackend(np.float32),
            scipy16=pg.SciPyExpmBackend(np.float16),
            tensorflow=pg.TensorFlowExpmBackend(),
            jax=pg.JaxExpmBackend()
        )

        times = {k: run(v) for k, v in backends.items()}

        pass
