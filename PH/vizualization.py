import functools
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def clear_show_save(func: Callable) -> Callable:
    """
    Decorator for clearing current figure in the beginning
    and showing or saving produced plot subsequently.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # clear current figure
        plt.clf()

        # execute function
        func(*args, **kwargs)

        # show or save
        return show_and_save(
            file=kwargs['file'] if 'file' in kwargs else None,
            show=kwargs['show'] if 'show' in kwargs else None
        )

    return wrapper


def show_and_save(file: str = None, show=True) -> plt.axis:
    """
    Show and save plot.
    :param file:
    :type file:
    :param show:
    :return:
    """
    # save figure if file path given
    if file is not None:
        plt.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0.1)

    # show figure if specified
    if show:
        plt.show()

    # return axis
    return plt.gca()


class Visualization:
    @staticmethod
    def plot_func(x: np.ndarray, f: Callable, xlabel: str = 'x', ylabel: str = 'f(x)'):
        sns.lineplot(x=x, y=f(x))

        # set axis labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
