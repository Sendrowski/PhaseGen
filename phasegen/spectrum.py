"""
SFS and 2-SFS classes.
"""

import copy
import logging
from typing import Dict, Iterable, Iterator

import jsonpickle
import numpy as np
# noinspection PyUnresolvedReferences
from fastdfe import Spectrum, Spectra

logger = logging.getLogger('phasegen').getChild('spectrum')


class SFS(Spectrum):
    """
    A site-frequency spectrum.
    """
    pass


class SFS2(Iterable):
    """
    A 2-dimensional site-frequency spectrum.
    """

    def __init__(self, data: np.ndarray | list):
        """
        Construct from data matrix.
        
        :param data:
        """
        data = np.array(data).copy()

        if data.ndim != 2:
            raise ValueError('Data has to be 2-dimensional')

        if data.shape[0] != data.shape[1]:
            raise ValueError('Matrix has to be square.')

        self.n = data.shape[0]

        # width
        self.w = self.n // 2 + 1 if self.n % 2 == 1 else self.n // 2

        self.data = data

    def to_file(self, file):
        """
        Save to file (in JSON format).
        
        :param file: File path.
        """
        with open(file, 'w') as f:
            f.write(self.to_json())

    def to_json(self) -> str:
        """
        Convert data to JSON string.
        
        :return: JSON string
        """
        obj = copy.deepcopy(self)

        # convert numpy array to list
        obj.data = obj.data.tolist()

        return jsonpickle.encode(obj)

    @staticmethod
    def from_file(file: str) -> 'SFS2':
        """
        Load from file.

        :param file: File path.
        :return: SFS2
        """
        with open(file, 'r') as f:
            return SFS2.from_json(f.read())

    @staticmethod
    def from_json(json: str) -> 'SFS2':
        """
        Load from JSON string.

        :param json: JSON string.
        :return: SFS2
        """
        obj = jsonpickle.decode(json)

        # convert list to numpy array
        obj.data = np.array(obj.data)

        return obj

    def is_folded(self) -> bool:
        """
        Check if the 2-SFS is folded.

        :return: Whether the 2-SFS is folded.
        """
        return np.all(self.data == self.fold().data)

    def __add__(self, other) -> 'SFS2':
        """
        Add two 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self + other.data

        return SFS2(self.data + other)

    def __sub__(self, other) -> 'SFS2':
        """
        Subtract two 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self - other.data

        return SFS2(self.data - other)

    def __mul__(self, other) -> 'SFS2':
        """
        Multiply 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self * other.data

        return SFS2(self.data * other)

    def __floordiv__(self, other) -> 'SFS2':
        """
        Divide 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self // other.data

        return SFS2(self.data // other)

    def __truediv__(self, other) -> 'SFS2':
        """
        Divide 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self / other.data

        return SFS2(self.data / other)

    def __iter__(self) -> Iterator:
        """
        Iterate over entries.

        :return: Iterator
        """
        return self.data.__iter__()

    def __pow__(self, power) -> 'SFS2':
        """
        Power operator.

        :param power: exponent
        :return: Spectrum
        """
        return SFS2(self.data ** power)

    def fold(self) -> 'SFS2':
        """
        Fold 2-SFS by adding up ``i`` and ``n - i`` for both axes.
        Node that this only make sense for counts or frequencies.

        :return: Folded 2-SFS.
        """
        data = self.data.copy()

        for _ in range(2):
            # compute left and right half and merge them
            left = np.concatenate((data[:self.w], np.zeros((self.n - self.w, self.n))))
            right = np.concatenate((data[self.w:][::-1], np.zeros((self.w, self.n))))

            # add parts and rotate
            data = (left + right).T

        return SFS2(data)

    def copy(self) -> 'SFS2':
        """
        Create deep copy.

        :return: Deep copy.
        """
        return copy.deepcopy(self)

    def symmetrize(self) -> 'SFS2':
        """
        Symmetric SFS so that ``i, j`` and ``j, i`` are the same.

        :return: Symmetric 2-SFS.
        """
        return SFS2((self.data + self.data.T) / 2)

    def fill_monomorphic(self, fill_value=np.nan) -> 'SFS2':
        """
        Remote the diagonal entries of the given array.
        
        :param fill_value: Value to fill diagonal entries with.
        :return: 2-SFS
        """
        other = self.copy()

        other.data[:1, :] = fill_value
        other.data[-1:, :] = fill_value
        other.data[:, :1] = fill_value
        other.data[:, -1] = fill_value

        return other

    def plot(
            self,
            ax: 'plt.Axes' = None,
            title: str = None,
            max_abs: float = None,
            log_scale: bool = False,
            cbar_kws: Dict = None,
            fill_diagonal_entries: bool = False,
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot as a heatmap.

        :param title: Title of the plot.
        :param ax: Axes to plot on.
        :param max_abs: Maximum absolute value to plot.
        :param log_scale: Use log scale.
        :param cbar_kws: Keyword arguments for color bar.
        :param fill_diagonal_entries: Fill diagonal entries.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
        import seaborn as sns

        if self.n < 3:
            logger.warning('Nothing to plot.')
            return plt.gca()

        if cbar_kws is None:
            cbar_kws = dict(pad=0.05)

        data = self.data.copy()

        # remove monomorphic entries
        data = self._remove_monomorphic(data)

        # mask diagonal entries
        if fill_diagonal_entries:
            data = self._fill_diagonals(data)

        # truncate data if folded
        if self.is_folded():
            data = data[:self.w - 1, :self.w - 1]

        # determine colorbar bounds if not specified
        if max_abs is None:
            max_abs = self._get_max_abs_entry(data)

        # plot heatmap using a symmetric log norm
        ax = sns.heatmap(
            data,
            norm=SymLogNorm(
                linthresh=max_abs / 10,
                vmin=-max_abs,
                vmax=max_abs
            ),
            cmap='PuOr_r',
            cbar_kws=cbar_kws,
            ax=ax
        )

        # invert y-axis and remove ticks
        ax.invert_yaxis()
        ax.axis('square')

        if log_scale:
            ax.set_xscale('log', base=1.001)
            ax.set_yscale('log', base=1.001)

        ax.set_xticklabels(range(1, len(data) + 1))
        ax.set_yticklabels(range(1, len(data) + 1))

        # remove confusing color bar ticks
        ax.collections[0].colorbar.ax.tick_params(size=0)

        # add frame around plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor('grey')

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def plot_surface(
            self,
            ax: 'plt.Axes' = None,
            title: str = None,
            max_abs: float = None,
            vmin: float = None,
            vmax: float = None,
            fill_diagonal_entries: bool = False,
            fill_value: float = np.nan,
            log_scale: bool = False,
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot as a surface.
        
        :param title:
        :param ax: Axes to plot on.
        :param max_abs: Maximum absolute value to plot.
        :param vmin: Minimum value to plot.
        :param vmax: Maximum value to plot.
        :param fill_diagonal_entries: Fill diagonal entries.
        :param fill_value: Value to fill diagonal entries with.
        :param log_scale: Use log scale.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm

        if self.n < 3:
            logger.warning('Nothing to plot.')
            return plt.gca()

        data = self.data.copy()

        # remove monomorphic entries
        data = self._remove_monomorphic(data)

        # mask diagonal entries
        if fill_diagonal_entries:
            data = self._fill_diagonals(data, fill_value)

        # truncate data if folded
        if self.is_folded():
            data = data[:self.w - 1, :self.w - 1]

        # determine color bar bounds if not specified
        if max_abs is None:
            max_abs = self._get_max_abs_entry(data) or 1

        x = np.arange(1, data.shape[0] + 1)
        y = np.arange(1, data.shape[0] + 1)

        x_grid, y_grid = np.meshgrid(x, y)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # vmin and vmax don't seem to work here
        ax.plot_surface(
            x_grid,
            y_grid,
            data,
            cmap='PuOr_r',
            vmin=vmin,
            vmax=vmax,
            norm=SymLogNorm(
                linthresh=max_abs / 10,
                vmin=-max_abs,
                vmax=max_abs
            )
        )

        # if log_scale:
        # ax.yaxis.set_scale('log')
        # ax.set_yscale('log', base=1.001)

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    @staticmethod
    def _remove_monomorphic(data: np.ndarray) -> np.ndarray:
        """
        Remove monomorphic sites from given 2-SFS matrix.

        :return: The data without monomorphic sites.
        """
        return data[1:-1, 1:-1]

    @staticmethod
    def _fill_diagonals(data: np.ndarray, fill_value=np.nan) -> np.ndarray:
        """
        Remote the diagonal entries of the given array.

        :param data: The data to fill.
        :param fill_value: The value to fill the diagonal with.
        :return: The filled data.
        """
        if len(data) == 0:
            return data

        if len(data) == 1:
            return np.array([[fill_value]])

        np.fill_diagonal(data, fill_value)
        data = np.fliplr(data)

        np.fill_diagonal(data, fill_value)
        data = np.fliplr(data)

        return data

    @classmethod
    def _mask(cls, data: np.ndarray = None) -> np.ndarray:
        """
        Remove diagonal and monomorphic entries.

        :param data: The data to mask.
        :return: The masked data.
        """
        data = data.copy()

        data = cls._fill_diagonals(data)

        data = cls._remove_monomorphic(data)

        return data[~np.isnan(data) & ~np.isinf(data)]

    @classmethod
    def _get_max_abs_entry(cls, data: np.ndarray) -> float:
        """
        Get the maximum absolute entry of the given data.

        :param data: The data.
        :return: The maximum absolute entry.
        """
        entries = np.abs(cls._mask(data))

        return entries.max() if len(entries) > 0 else 1
