from typing import Callable, List

import numpy as np
import scipy
from multiprocess.pool import Pool
from tqdm import tqdm


def expm(m: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential.
    """
    if m.shape[0] < 400:
        return expm_scipy(m)

    return expm_ts(m)


def expm_ts(m: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential using TensorFlow. This is because scipy.linalg.expm sometimes produces
    erroneous results for large matrices (see https://github.com/scipy/scipy/issues/18086).

    TODO remove this function once the issue is resolved in scipy.

    :param m: Matrix
    :return: Matrix exponential
    """
    import tensorflow as tf

    return tf.linalg.expm(tf.convert_to_tensor(m, dtype=tf.float64)).numpy()


def expm_scipy(m: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential using SciPy.

    :param m: Matrix
    :return: Matrix exponential
    """
    return scipy.linalg.expm(m)


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        batch_size: int = None,
        desc: str = None
) -> np.ndarray:
    """
    Parallelize given function or execute sequentially.

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize
    :param pbar: Whether to show a progress bar
    :param batch_size: Number of units to show in the pbar per function
    :param desc: Description for tqdm progress bar
    :return: Array of results
    """

    if parallelize and len(data) > 1:
        # parallelize
        iterator = Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    if pbar:
        iterator = tqdm(iterator, total=len(data), unit_scale=batch_size, desc=desc)

    return np.array(list(iterator), dtype=object)
