import copy
from typing import Dict, cast, Iterable

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastdfe import Spectrum
from matplotlib.colors import SymLogNorm
from numpy.random import default_rng


def _remove_monomorphic(data):
    """
    Remove monomorphic sites from given 2-SFS matrix.
    
    :return:
    """
    return data[1:-1, 1:-1]


def _fill_diagonals(data: np.ndarray, fill_value=np.nan):
    """
    Remote the diagonal entries of the given array.
    
    :param data:
    :param fill_value:
    :return:
    """
    np.fill_diagonal(data, fill_value)
    data = np.fliplr(data)

    np.fill_diagonal(data, fill_value)
    data = np.fliplr(data)

    return data


def _mask(data: np.ndarray = None) -> np.ndarray:
    """
    Remove diagonal and monomorphic entries.

    :param data: The data to mask.
    :return: The masked data.
    """
    data = data.copy()

    data = _fill_diagonals(data)

    data = _remove_monomorphic(data)

    return data[~np.isnan(data) & ~np.isinf(data)]


def _get_max_abs_entry(data: np.ndarray) -> float | None:
    """
    Get the maximum absolute entry of the given data.
    
    :param data: The data.
    :return: The maximum absolute entry.
    """
    entries = np.abs(_mask(data))

    return entries.max() if len(entries) > 0 else None


class SFS(Spectrum):
    """
    A site-frequency spectrum.
    """
    pass


class SFS2(Iterable):
    """
    A 2-dimensional site-frequency spectrum.
    """

    def __init__(self, data: np.ndarray | list, folded: bool = False):
        """
        Construct from data matrix.
        
        :param data:
        :param folded:
        """
        data = np.array(data).copy()

        if data.ndim != 2:
            raise AssertionError('Data has to be 2-dimensional')

        if data.shape[0] != data.shape[1]:
            raise AssertionError('Matrix has to be square.')

        self.n = data.shape[0]

        # width
        self.w = self.n // 2 + 1 if self.n % 2 == 1 else self.n // 2

        self.data = data
        self.folded = folded

    def to_file(self, file) -> None:
        """
        Save to file.
        
        :param file:
        :return:
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

    def __add__(self, other) -> 'SFS2':
        """
        Add two 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self + other.data

        return SFS2(self.data + other, folded=self.folded)

    def __sub__(self, other) -> 'SFS2':
        """
        Subtract two 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self - other.data

        return SFS2(self.data - other, folded=self.folded)

    def __mul__(self, other) -> 'SFS2':
        """
        Multiply 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self * other.data

        return SFS2(self.data * other, folded=self.folded)

    def __floordiv__(self, other) -> 'SFS2':
        """
        Divide 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self // other.data

        return SFS2(self.data // other, folded=self.folded)

    def __truediv__(self, other) -> 'SFS2':
        """
        Divide 2-SFS.
        
        :param other:
        :return:
        """
        if isinstance(other, SFS2):
            return self / other.data

        return SFS2(self.data / other, folded=self.folded)

    def __iter__(self):
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
        Fold 2-SFS.
        """
        if not self.folded:
            data = self.data.copy()

            for _ in range(2):
                # compute left and right half and merge them
                left = np.concatenate((data[:self.w], np.zeros((self.n - self.w, self.n))))
                right = np.concatenate((data[self.w:][::-1], np.zeros((self.w, self.n))))

                # add parts and rotate
                data = (left + right).T

            return SFS2(data, folded=True)

        return self

    def copy(self) -> 'SFS2':
        """
        Copy SFS.
        """
        return copy.deepcopy(self)

    def symmetrize(self) -> 'SFS2':
        """
        Symmetric SFS.
        """
        return SFS2((self.data + self.data.T) / 2, folded=self.folded)

    def resample_hypergeometric(self, multiplier: int = 100000, N: int = 100) -> 'SFS2':
        """
        Resample counts under the hyper-geometric distribution.

        :param multiplier: multiplier for counts to determine number
            of draws.
        :param N: population size
        :return:
        """
        rng = default_rng()

        counts = (self.data * multiplier).astype(int)
        sfs2_corr = SFS2(np.zeros_like(counts))
        n = self.n - 1

        # iterate over frequency counts
        for i in range(n + 1):
            for j in range(n + 1):
                count = counts[i, j]

                # select n out N samples assuming p = i / N
                f1 = rng.hypergeometric(i * N / n, (n - i) * N / n, n, count)
                f2 = rng.hypergeometric(j * N / n, (n - j) * N / n, n, count)

                # add resampled counts
                for k, l in zip(f1, f2):
                    sfs2_corr.data[k, l] += 1

        sfs2_corr.data = (sfs2_corr.data.astype(float) / multiplier)

        return cast(SFS2, sfs2_corr).symmetrize()

    def fill_diagonals(self, fill_value=np.nan) -> 'SFS2':
        """
        Remote the diagonal entries of the given array.
        
        :param fill_value:
        :return:
        """
        other = self.copy()

        other.data = _fill_diagonals(other.data, fill_value)

        return other

    def fill_monomorphic(self, fill_value=np.nan) -> 'SFS2':
        """
        Remote the diagonal entries of the given array.
        
        :param fill_value:
        :return:
        """
        other = self.copy()

        other.data[:1, :] = fill_value
        other.data[-1:, :] = fill_value
        other.data[:, :1] = fill_value
        other.data[:, -1] = fill_value

        return other

    def get_max_abs_entry(self) -> float:
        """
        Get maximum magnitude entry outside the diagonal.
        
        :return:
        """
        return _get_max_abs_entry(self.data)

    def mask(self) -> np.ndarray:
        """
        Return data without monomorphic or diagonal sites.
        
        :return: Masked data.
        """
        return _mask(self.data)

    def plot_heatmap(
            self,
            ax: plt.Axes = None,
            title: str = None,
            max_abs: float = None,
            log_scale: bool = False,
            cbar_kws: Dict = dict(pad=-0.05),
            fill_diagonal_entries: bool = False,
            show: bool = True,
    ) -> plt.Axes:
        """
        Plot as a heatmap.
        
        :param title:
        :param ax: Axes to plot on.
        :param max_abs: Maximum absolute value to plot.
        :param log_scale: Use log scale.
        :param cbar_kws: Keyword arguments for color bar.
        :param fill_diagonal_entries: Fill diagonal entries.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        data = self.data.copy()

        # remove monomorphic entries
        data = _remove_monomorphic(data)

        # mask diagonal entries
        if fill_diagonal_entries:
            data = _fill_diagonals(data)

        # truncate data if folded
        if self.folded:
            data = data[:self.w - 1, :self.w - 1]

        # determine colorbar bounds if not specified
        if max_abs is None:
            max_abs = _get_max_abs_entry(data)

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
        ax.tick_params(size=0)

        if log_scale:
            ax.set_xscale('log', base=1.001)
            ax.set_yscale('log', base=1.001)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
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

    def plot(
            self,
            ax: plt.Axes = None,
            title: str = None,
            max_abs: float = None,
            vmin: float = None,
            vmax: float = None,
            fill_diagonal_entries: bool = False,
            fill_value: float = np.nan,
            log_scale: bool = False,
            show: bool = True,
    ) -> plt.Axes:
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
        data = self.data.copy()

        # remove monomorphic entries
        data = _remove_monomorphic(data)

        # mask diagonal entries
        if fill_diagonal_entries:
            data = _fill_diagonals(data, fill_value)

        # truncate data if folded
        if self.folded:
            data = data[:self.w - 1, :self.w - 1]

        # determine color bar bounds if not specified
        if max_abs is None:
            max_abs = _get_max_abs_entry(data) or 1

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

        if log_scale:
            pass  # log scale currently doesn't work
            # ax.yaxis.set_scale('log')
            # ax.set_yscale('log', base=1.001)

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax
