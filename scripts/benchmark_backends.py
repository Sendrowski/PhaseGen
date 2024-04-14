"""
Benchmark matrix exponentiation methods of different backends.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-04-14"

import time

import numpy as np

import phasegen as pg


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
