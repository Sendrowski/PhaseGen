"""
Utility functions.
"""
from typing import Callable, List

import numpy as np
from multiprocess.pool import Pool
from tqdm import tqdm


def parallelize(
        func: Callable,
        data: List | np.ndarray,
        parallelize: bool = True,
        pbar: bool = True,
        batch_size: int = None,
        desc: str = None,
        dtype: type = float,
        delay: int = 0
) -> np.ndarray:
    """
    Parallelize given function or execute sequentially.

    :param func: Function to parallelize
    :param data: Data to parallelize over
    :param parallelize: Whether to parallelize
    :param pbar: Whether to show a progress bar
    :param batch_size: Number of units to show in the pbar per function
    :param desc: Description for tqdm progress bar
    :param dtype: Data type of the results
    :param delay: Delay for tqdm progress bar
    :return: Array of results
    """
    if parallelize and len(data) > 1:
        # parallelize
        iterator = Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    if pbar:
        iterator = tqdm(iterator, total=len(data), unit_scale=batch_size, desc=desc, delay=delay)

    return np.array(list(iterator), dtype=dtype)
