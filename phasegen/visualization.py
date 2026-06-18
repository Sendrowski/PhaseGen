"""
Visualization module.
"""

import functools
from typing import Callable, Dict, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class Visualization:
    """
    Visualization class.
    """

    @staticmethod
    def clear_show_save(func: Callable) -> Callable:
        """
        Decorator for clearing current figure in the beginning
        and showing or saving produced plot subsequently.

        :param func: Function to decorate
        :return: Wrapper function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> 'plt.Axes':
            """
            Wrapper function.

            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Axes
            """

            # add axes if not given
            if 'ax' not in kwargs or ('ax' in kwargs and kwargs['ax'] is None):
                # clear current figure
                plt.close()

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
    def show_and_save(file: str = None, show: bool = True) -> 'plt.Axes':
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
            ax: 'plt.Axes',
            x: np.ndarray,
            y: np.ndarray,
            xlabel: str = 'x',
            ylabel: str = 'f(x)',
            file: str = None,
            show: bool = None,
            clear: bool = True,
            label: str = None,
            title: str = None
    ) -> 'plt.Axes':
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

        # remove margins
        plt.margins(x=0)
        plt.tight_layout()

        return ax

    @staticmethod
    def plot_surface(
            xs: np.ndarray,
            ys: np.ndarray,
            Z: np.ndarray,
            surface: bool = False,
            ax: 'plt.Axes' = None,
            xlabel: str = '$R_a$',
            ylabel: str = '$R_b$',
            zlabel: str = 'f',
            title: str = None,
            vmin: float = None,
            vmax: float = None,
            file: str = None,
            show: bool = True
    ) -> 'plt.Axes':
        """
        Draw a bivariate function ``Z`` over the grid ``xs x ys`` (shape ``(len(xs), len(ys))``) as either a 3D
        surface (``surface=True``) or a 2D heatmap with colorbar.

        :param xs: Grid coordinates along the first axis.
        :param ys: Grid coordinates along the second axis.
        :param Z: Values on the ``xs x ys`` grid.
        :param surface: Draw a 3D surface instead of a heatmap.
        :param ax: Axes to draw on (a 3D axes is created if needed for ``surface``).
        :param xlabel: First-axis label.
        :param ylabel: Second-axis label.
        :param zlabel: Value-axis label (the z axis / colorbar quantity).
        :param title: Plot title.
        :param vmin: Lower colour/scale limit (e.g. ``0`` for a CDF).
        :param vmax: Upper colour/scale limit (e.g. ``1`` for a CDF).
        :param file: File to save the plot to.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        zlim = {}
        if vmin is not None:
            zlim['vmin'] = vmin
        if vmax is not None:
            zlim['vmax'] = vmax

        if surface:
            if ax is None:
                ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(*np.meshgrid(xs, ys), np.asarray(Z).T, cmap='viridis', **zlim)
            ax.set_zlabel(zlabel)
            if vmax is not None:
                ax.set_zlim(vmin if vmin is not None else 0.0, vmax)
        else:
            if ax is None:
                ax = plt.gca()
            mesh = ax.pcolormesh(xs, ys, np.asarray(Z).T, shading='auto', cmap='viridis', **zlim)
            ax.figure.colorbar(mesh, ax=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        Visualization.show_and_save(file=file, show=show)
        return ax

    @staticmethod
    @clear_show_save
    def plot_rates(
            ax: 'plt.Axes',
            times: List[float],
            rates: Dict[str, np.ndarray],
            xlabel: str = 't',
            ylabel: str = '$N_e(t)$',
            file: str = None,
            show: bool = None,
            clear: bool = True,
            title: str = 'rate trajectory',
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Plot function.

        :param ax: Axes to plot on
        :param times: Dictionary of times
        :param rates: Dictionary of rates
        :param xlabel: x label
        :param ylabel: y label
        :param file: File to save plot to
        :param show: Whether to show plot
        :param clear: Whether to clear current figure
        :param title: Title for plot
        :param kwargs: Keyword arguments passed to plot function
        :return: Axes
        """
        if kwargs is None:
            kwargs = {}

        # plot
        for key in rates:
            ax.plot(times, rates[key], drawstyle='steps-post', label=key, **kwargs)

        # set axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add title
        ax.set_title(title)

        # add legend if more than one rate
        if len(rates) > 1:
            ax.legend()

        plt.margins(x=0)

        return ax
