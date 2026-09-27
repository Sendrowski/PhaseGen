"""
Visualization module.
"""

import functools
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


@dataclass
class _CurveData:
    """
    The curves of a line plot, sharing one grid. Returned by the ``_plot_data`` methods and drawn by the corresponding
    ``plot`` methods.
    """
    #: Grid shared by all curves, of shape ``(n_points,)``.
    x: np.ndarray

    #: Curve values, of shape ``(n_series, n_points)``.
    y: np.ndarray

    #: Label of each curve, an empty string for an unlabelled one.
    labels: List[str]

    #: Label of the x-axis.
    xlabel: str

    #: Label of the y-axis.
    ylabel: str

    #: Plot title.
    title: str

    #: Title of the legend, ``None`` for none.
    legend_title: Optional[str] = None


@dataclass
class _SurfaceData:
    """
    A bivariate function on a grid, drawn as a heatmap or a 3D surface. Returned by the ``_plot_data`` method of a
    joint distribution function.
    """
    #: Grid of the first axis, of shape ``(n_x,)``.
    x: np.ndarray

    #: Grid of the second axis, of shape ``(n_y,)``.
    y: np.ndarray

    #: Values on the grid, of shape ``(n_x, n_y)``.
    z: np.ndarray

    #: Label of the first axis.
    xlabel: str

    #: Label of the second axis.
    ylabel: str

    #: Label of the values.
    zlabel: str

    #: Plot title.
    title: str

    #: Lower limit of the value scale, ``None`` for the smallest value.
    vmin: Optional[float] = None

    #: Upper limit of the value scale, ``None`` for the largest value.
    vmax: Optional[float] = None


class Visualization:
    """
    Visualization class.
    """

    @staticmethod
    def clear_show_save(func: Callable) -> Callable:
        """
        Decorator that prepares the axes and shows or saves the produced plot. Without ``ax`` the plot goes to a new
        figure, or onto the current axes when ``clear`` is false. With ``ax`` it goes onto those axes as they are.

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

            clear = kwargs.get('clear', True)

            if kwargs.get('ax') is None:
                # a fresh figure, or the current axes to draw onto
                if clear:
                    plt.close()

                kwargs['ax'] = plt.gca()

            # execute function
            func(*args, **kwargs)

            # make layout tight
            kwargs['ax'].figure.tight_layout()

            # show or save
            # show by default here
            return Visualization.show_and_save(
                kwargs['ax'],
                file=kwargs['file'] if 'file' in kwargs else None,
                show=kwargs['show'] if 'show' in kwargs else True
            )

        return wrapper

    @staticmethod
    def show_and_save(ax: 'plt.Axes | np.ndarray', file: str = None, show: bool = True) -> 'plt.Axes | np.ndarray':
        """
        Show and save the figure of the given axes.

        :param ax: Axes, or an array of axes of one figure.
        :param file: File path to save the figure to
        :param show: Whether to show the figure
        :return: The axes passed
        """
        # save figure if file path given
        if file is not None:
            np.ravel(ax)[0].figure.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0.1)

        # show figure if specified and if not in interactive mode
        if show and not plt.isinteractive():
            plt.show()

        return ax

    @staticmethod
    @clear_show_save
    def plot_curves(
            ax: 'plt.Axes',
            data: _CurveData,
            file: str = None,
            show: bool = None,
            clear: bool = True,
            label: str = None,
            title: str = None,
            **kwargs
    ) -> 'plt.Axes':
        """
        Draw a set of labelled curves sharing one axis.

        :param ax: Axes to plot on.
        :param data: The curves.
        :param file: File to save the plot to.
        :param show: Whether to show the plot.
        :param clear: Whether to clear the current figure.
        :param label: Legend label replacing the labels of the curves, ``None`` to keep them.
        :param title: Title replacing the title of the curves, ``None`` to keep it.
        :param kwargs: Additional line styling forwarded to the underlying plot (e.g. ``alpha``, ``lw``, ``ls``).
        :return: Axes.
        """
        labels = data.labels if label is None else [label] * len(data.labels)

        for y, lab in zip(data.y, labels):
            sns.lineplot(x=data.x, y=y, ax=ax, label=lab or None, **kwargs)

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title if title is None else title)

        if data.legend_title is not None and any(labels):
            ax.legend(title=data.legend_title)

        ax.margins(x=0)

        return ax

    @staticmethod
    def plot_surface(
            data: _SurfaceData,
            surface: bool = False,
            ax: 'plt.Axes' = None,
            title: str = None,
            file: str = None,
            show: bool = True
    ) -> 'plt.Axes':
        """
        Draw a bivariate function on its grid as a 3D surface or as a 2D heatmap with colorbar.

        :param data: The bivariate function on its grid.
        :param surface: Whether to draw a 3D surface. Otherwise a heatmap is drawn.
        :param ax: Axes to draw on (a 3D axes is created if needed for ``surface``).
        :param title: Title replacing the title of the data, ``None`` to keep it.
        :param file: File to save the plot to.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        zlim = {key: value for key, value in dict(vmin=data.vmin, vmax=data.vmax).items() if value is not None}
        z = np.asarray(data.z).T

        if surface:
            if ax is None:
                ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(*np.meshgrid(data.x, data.y), z, cmap='viridis', **zlim)
            ax.set_zlabel(data.zlabel)
            if data.vmax is not None:
                ax.set_zlim(data.vmin if data.vmin is not None else 0.0, data.vmax)
        else:
            if ax is None:
                ax = plt.gca()
            mesh = ax.pcolormesh(data.x, data.y, z, shading='auto', cmap='viridis', **zlim)
            ax.figure.colorbar(mesh, ax=ax)

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title if title is None else title)
        return Visualization.show_and_save(ax, file=file, show=show)

    @staticmethod
    @clear_show_save
    def plot_rates(
            ax: 'plt.Axes',
            data: _CurveData,
            file: str = None,
            show: bool = None,
            clear: bool = True,
            title: str = None,
            ylabel: str = None,
            kwargs: dict = None
    ) -> 'plt.Axes':
        """
        Draw rate trajectories as step functions.

        :param ax: Axes to plot on.
        :param data: The trajectories.
        :param file: File to save the plot to.
        :param show: Whether to show the plot.
        :param clear: Whether to clear the current figure.
        :param title: Title replacing the title of the data, ``None`` to keep it.
        :param ylabel: Label replacing the y-axis label of the data, ``None`` to keep it.
        :param kwargs: Keyword arguments passed to the plot function.
        :return: Axes.
        """
        if kwargs is None:
            kwargs = {}

        for label, y in zip(data.labels, data.y):
            ax.plot(data.x, y, drawstyle='steps-post', label=label, **kwargs)

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel if ylabel is None else ylabel)
        ax.set_title(data.title if title is None else title)

        # add legend if more than one rate
        if len(data.labels) > 1:
            ax.legend()

        ax.margins(x=0)

        return ax
