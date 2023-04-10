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
        if 'clear' not in kwargs or ('clear' in kwargs and kwargs['clear']):
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
    @clear_show_save
    def plot_func(
            x: np.ndarray,
            y: Callable,
            xlabel: str = 'x',
            ylabel: str = 'f(x)',
            file: str = None,
            show: bool = None,
            clear: bool = True,
            label: str = None
    ):
        sns.lineplot(x=x, y=y, ax=plt.gca(), label=label)

        # set axis labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
