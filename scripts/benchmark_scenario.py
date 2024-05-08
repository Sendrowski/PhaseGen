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
    name = "1_epoch_n_10"
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

time_lineage_counting = benchmark(lambda: c.ph.lineage_counting_state_space.S)
k_lineage_counting = c.ph.lineage_counting_state_space.k
mean_tree_height = benchmark(lambda: c.ph.tree_height.mean)

try:
    time_block_counting = benchmark(lambda: c.ph.block_counting_state_space.S)
    k_block_counting = c.ph.block_counting_state_space.k
    mean_sfs = benchmark(lambda: c.ph.sfs.mean)
except NotImplementedError:
    time_block_counting = None
    k_block_counting = None
    mean_sfs = None

df = pd.DataFrame({
    'scenario': [file.split('/')[-1].split('.')[0]],
    'lineage_counting.time': [time_lineage_counting],
    'lineage_counting.k': [k_lineage_counting],
    'lineage_counting.tree_height.mean': [mean_tree_height],
    'block_counting.k': [k_block_counting],
    'block_counting.time': [time_block_counting],
    'block_counting.sfs.mean': [mean_sfs]
})

df.to_csv(out, index=False)

pass