"""
Benchmark matrix exponentiation methods of different packages.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-02-01"

import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
import tensorflow as tf
import cupy as cp
from tqdm import tqdm

try:
    import sys

    # necessary to import local module
    sys.path.append('.')

    testing = False
except NameError:
    # testing
    testing = True


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


def generate_matrix(size: int, sparsity: float) -> np.ndarray:
    """
    Generate a random matrix with the given parameters.

    :param size: Size of the matrix
    :param sparsity: Sparsity of the matrix
    :return: Random matrix
    """
    return sp.random(
        size,
        size,
        density=1 - sparsity,
        data_rvs=lambda size: np.random.normal(mean, std_dev, size),
        random_state=np.random.default_rng()
    ).toarray()


# set the parameters for the normal distribution and the sparsity of the matrix
mean = 10
std_dev = 10
sparsity = 0.995

# Initialize the sizes of the matrices to be tested
sizes = np.logspace(1, 3.5, 20, dtype=int)

# initialize lists to store the computation times
df = pd.DataFrame({
    'size': [],
    'tf': [],
    'scipy_dense': [],
    'scipy_sparse': []
})

for i, size in enumerate(tqdm(sizes)):
    # Generate a random matrix with the given parameters
    m = generate_matrix(size, sparsity)

    m_sparse = sp.csc_matrix(m)

    # benchmark TensorFlow
    ts = benchmark(lambda: tf.linalg.expm(tf.convert_to_tensor(m, dtype=tf.float64)).numpy())

    # benchmark SciPy (dense)
    scipy_dense = benchmark(lambda: scipy.linalg.expm(m))

    cupy = benchmark(lambda: cp.linalg.expm(m))

    # benchmark SciPy (sparse)
    #scipy_sparse = benchmark(lambda: sp.linalg.expm(m_sparse))
    scipy_sparse = 0

    # store the results
    df.loc[i] = [size, ts, scipy_dense, scipy_sparse]

# plot the results
plt.plot(df['size'], df['tf'], label='TensorFlow')
plt.plot(df['size'], df['scipy_dense'], label='SciPy (dense)')
plt.plot(df['size'], df['scipy_sparse'], label='SciPy (sparse)')
plt.xlabel('Matrix size')
plt.ylabel('Time (s)')
plt.legend()
plt.show()

pass
