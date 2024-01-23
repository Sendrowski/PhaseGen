"""
Benchmark the state space creation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-01-22"

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    name = "1_epoch_n_2_test_size"
    file = f"resources/configs/{name}.yaml"
    out = f"scratch/{name}.csv"

import time
from typing import Callable
import pandas as pd
from phasegen.comparison import Comparison


def benchmark(callback: Callable) -> float:
    """
    Benchmark a function.
    
    :param callback: function to benchmark
    :return: time in seconds
    """
    start = time.time()

    callback()

    end = time.time()

    return end - start


c = Comparison.from_yaml(file)

default = benchmark(lambda: c.ph.default_state_space.S)

try:
    block_counting = benchmark(lambda: c.ph.block_counting_state_space.S)
except NotImplementedError:
    block_counting = None

df = pd.DataFrame({
    'scenario': [file.split('/')[-1].split('.')[0]],
    'default': [default],
    'block_counting': [block_counting]
})

df.to_csv(out, index=False)

pass