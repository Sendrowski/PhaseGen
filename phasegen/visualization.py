import functools
from typing import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class Visualization:

    @staticmethod
    def clear_show_save(func: Callable) -> Callable:
        """
        Decorator for clearing current figure in the beginning
        and showing or saving produced plot subsequently.

        :param func: Function to decorate
        :return: Wrapper function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> plt.Axes:
            """
            Wrapper function.

            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Axes
            """

            # add axes if not given
            if 'ax' not in kwargs or ('ax' in kwargs and kwargs['ax'] is None):
                # clear current figure
                plt.clf()

                kwargs['ax'] = plt.gca()

            # execute function
            func(*args, **kwargs)

            # make layout tight
            plt.tight_layout()

            # show or save
            # show by default here
            return Visualization.show_and_save(
                file=kwargs['file'] if 'file' in kwargs else None,
                show=kwargs['show'] if 'show' in kwargs else True
            )

        return wrapper

    @staticmethod
    def show_and_save(file: str = None, show: bool = True) -> plt.Axes:
        """
        Show and save plot.

        :param file: File path to save plot to
        :param show: Whether to show plot
        :return: Axes

        """
        # save figure if file path given
        if file is not None:
            plt.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0.1)

        # show figure if specified and if not in interactive mode
        if show and not plt.isinteractive():
            plt.show()

        # return current axes
        return plt.gca()

    @staticmethod
    @clear_show_save
    def plot(
            ax: plt.Axes,
            x: np.ndarray,
            y: np.ndarray,
            xlabel: str = 'x',
            ylabel: str = 'f(x)',
            file: str = None,
            show: bool = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> plt.Axes:
        """
        Plot function.

        :param ax: Axes to plot on
        :param x: x values
        :param y: y values
        :param xlabel: x label
        :param ylabel: y label
        :param file: File to save plot to
        :param show: Whether to show plot
        :param clear: Whether to clear current figure
        :param label: Label for plot
        :param title: Title for plot
        :return: Axes
        """
        sns.lineplot(x=x, y=y, ax=ax, label=label)

        # set axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add title
        ax.set_title(title)

        return plt.gca()

    @staticmethod
    @clear_show_save
    def plot_pop_sizes(
            ax: plt.Axes,
            times: np.ndarray,
            pop_sizes: np.ndarray,
            t_max: float = None,
            xlabel: str = 't',
            ylabel: str = '$N_e(t)$',
            file: str = None,
            show: bool = None,
            clear: bool = True,
            title: str = 'population size trajectory'
    ) -> plt.Axes:
        """
        Plot function.

        :param ax: Axes to plot on
        :param times: x values
        :param pop_sizes: y values
        :param t_max: Maximum time to plot
        :param xlabel: x label
        :param ylabel: y label
        :param file: File to save plot to
        :param show: Whether to show plot
        :param clear: Whether to clear current figure
        :param title: Title for plot
        :return: Axes
        """
        # add last time point if t_max is given
        if t_max is not None and t_max > times[-1]:
            times = np.concatenate((times, [t_max]))
            pop_sizes = np.concatenate((pop_sizes, [pop_sizes[-1]]))

        plt.plot(times, pop_sizes, drawstyle='steps-post')

        # set axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add title
        ax.set_title(title)

        # set x limit
        if t_max is not None:
            ax.set_xlim(0, t_max)

        return plt.gca()