"""
Classes for working with the site-frequency spectrum (SFS) and 2-SFS.
"""

import copy
import logging
from typing import Dict, Iterable, Iterator, Sequence, Tuple

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
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot as a heatmap.

        :param title: Title of the plot.
        :param ax: Axes to plot on.
        :param max_abs: Maximum absolute value to plot.
        :param log_scale: Use log scale.
        :param cbar_kws: Keyword arguments for color bar.
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

        if max_abs is None:
            max_abs = self.get_max_abs() or 1

        # remove monomorphic sites
        data = self.data[1:-1, 1:-1]

        # truncate data if folded
        if self.is_folded():
            data = data[:self.w - 1, :self.w - 1]

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

        ax.set_xticks(ax.get_yticks())
        ax.set_xticklabels([str(int(label + 1)) for label in ax.get_xticks()])
        ax.set_yticklabels([str(int(label + 1)) for label in ax.get_yticks()])


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
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot as a surface.
        
        :param title:
        :param ax: Axes to plot on.
        :param max_abs: Maximum absolute value to plot.
        :param vmin: Minimum value to plot.
        :param vmax: Maximum value to plot.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm

        if self.n < 3:
            logger.warning('Nothing to plot.')
            return plt.gca()

        if max_abs is None:
            max_abs = self.get_max_abs() or 1

        # remove monomorphic sites
        data = self.data[1:-1, 1:-1]

        # truncate data if folded
        if self.is_folded():
            data = data[:self.w - 1, :self.w - 1]

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

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def mask_diagonal(self, fill_value=np.nan) -> 'SFS2':
        """
        Mask both the primary and secondary diagonal entries of the 2-SFS matrix.

        The primary diagonal runs from the top-left to the bottom-right,
        and the secondary diagonal runs from the top-right to the bottom-left.

        :param fill_value: The value to fill the diagonal entries with.
        :return: A new SFS2 object with both diagonals masked.
        """
        data = self.data.copy()
        np.fill_diagonal(data, fill_value)

        data = np.fliplr(data)
        np.fill_diagonal(data, fill_value)
        data = np.fliplr(data)

        return SFS2(data)

    def get_max_abs(self) -> float:
        """
        Get the maximum absolute entry of the 2-SFS matrix.

        :return: The maximum absolute entry.
        """
        return np.nanmax(np.abs(self.data))

    def mask_upper(self, fill_value=np.nan) -> 'SFS2':
        """
        Mask the upper triangular entries of the 2-SFS matrix.

        :param fill_value: The value to fill the upper triangular entries with.
        :return: A new SFS2 object with upper triangular entries masked.
        """
        data = self.copy().data

        data[np.tril(np.ones_like(data, dtype=bool), k=-1)] = fill_value

        return SFS2(data)


class TwoLocusSFS(SFS2):
    """
    The two-locus site-frequency spectrum under recombination: a square matrix whose entry ``(i, j)`` is the
    expected product of the branch length subtending ``i`` samples at locus 0 and ``j`` samples at locus 1, for two
    loci separated by recombination rate ``r``. It interpolates between the within-tree cross-moment of the SFS at
    ``r = 0`` (fully linked, equal to ``Coalescent.sfs.cov`` plus the outer product of the marginal means) and the
    outer product of the marginal SFS as ``r → ∞`` (independent loci). It shares the container machinery of
    :class:`SFS2` (plotting, folding, arithmetic, serialization).
    """
    pass


class JointSFS(Iterable):
    """
    A joint (multi-population) site-frequency spectrum.

    The data is a ``P``-dimensional array of shape ``(n_0 + 1, ..., n_{P-1} + 1)`` where ``P`` is the number of
    populations and entry ``(k_0, ..., k_{P-1})`` corresponds to branches subtending ``k_p`` samples from population
    ``p``. For two populations this is a 2-dimensional array (analogous to but generally rectangular, unlike the
    square :class:`SFS2`); for three populations it is a 3-dimensional array, and so on.
    """

    def __init__(self, data: np.ndarray | list):
        """
        Construct from a data array.

        :param data: A ``P``-dimensional array.
        """
        data = np.asarray(data)

        if data.ndim < 1:
            raise ValueError('Data has to be at least 1-dimensional.')

        #: The joint SFS array.
        self.data: np.ndarray = data

    @property
    def n_pops(self) -> int:
        """
        Number of populations (dimensions of the joint SFS).
        """
        return self.data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the joint SFS array.
        """
        return self.data.shape

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface so the joint SFS can be used directly in numpy operations.

        :param dtype: Optional dtype.
        :return: The underlying array.
        """
        return self.data if dtype is None else self.data.astype(dtype)

    def __iter__(self) -> Iterator:
        """
        Iterate over the first axis of the joint SFS.

        :return: Iterator.
        """
        return self.data.__iter__()

    def __getitem__(self, item):
        """
        Index into the joint SFS array.

        :param item: Index.
        :return: Indexed value or sub-array.
        """
        return self.data[item]

    def __add__(self, other) -> 'JointSFS':
        return JointSFS(self.data + (other.data if isinstance(other, JointSFS) else other))

    def __sub__(self, other) -> 'JointSFS':
        return JointSFS(self.data - (other.data if isinstance(other, JointSFS) else other))

    def __mul__(self, other) -> 'JointSFS':
        return JointSFS(self.data * (other.data if isinstance(other, JointSFS) else other))

    def __truediv__(self, other) -> 'JointSFS':
        return JointSFS(self.data / (other.data if isinstance(other, JointSFS) else other))

    def __pow__(self, power) -> 'JointSFS':
        return JointSFS(self.data ** power)

    def copy(self) -> 'JointSFS':
        """
        Create a deep copy.

        :return: Deep copy.
        """
        return copy.deepcopy(self)

    def marginalize(self, pops: Sequence[int]) -> 'JointSFS':
        """
        Marginalize the joint SFS onto a subset of populations by summing over the other populations. This is useful
        for example to obtain a 2-dimensional view of a higher-dimensional joint SFS.

        :param pops: The population indices to keep, in the desired axis order.
        :return: A joint SFS over the specified populations.
        """
        keep = tuple(int(p) for p in pops)

        if any(p < 0 or p >= self.n_pops for p in keep):
            raise ValueError(f'Population indices must be in [0, {self.n_pops - 1}].')

        drop = tuple(i for i in range(self.n_pops) if i not in keep)

        data = self.data.sum(axis=drop) if drop else self.data

        # reorder the remaining axes (which are in ascending order) to match the requested order
        order = [sorted(keep).index(p) for p in keep]

        return JointSFS(np.transpose(data, order))

    def plot(
            self,
            pops: Tuple[int, int] = (0, 1),
            ax: 'plt.Axes' = None,
            title: str = None,
            log_scale: bool = False,
            mask_monomorphic: bool = True,
            cbar_kws: Dict = None,
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot the joint SFS as a 2-dimensional heatmap. For more than two populations, the joint SFS is first
        marginalized onto the two requested populations (summing over the others).

        :param pops: The two population indices to plot (y-axis, x-axis).
        :param ax: Axes to plot on.
        :param title: Title of the plot.
        :param log_scale: Whether to use a logarithmic color scale.
        :param mask_monomorphic: Whether to mask the monomorphic corners (all-zero and all-derived).
        :param cbar_kws: Keyword arguments for the color bar.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import seaborn as sns

        if len(pops) != 2:
            raise ValueError('Exactly two populations must be specified for a 2-dimensional plot.')

        # reduce to the two requested populations
        data = (self.marginalize(pops) if self.n_pops > 2 else self).data.astype(float).copy()

        if data.ndim != 2:
            raise ValueError('Plotting requires a 2-dimensional (marginalized) joint SFS.')

        if mask_monomorphic:
            data[0, 0] = np.nan
            data[-1, -1] = np.nan

        if cbar_kws is None:
            cbar_kws = dict(pad=0.05)

        # create a fresh 2-D axes if none is given (so we never draw onto a leftover 3-D axes from plot_surface)
        if ax is None:
            _, ax = plt.subplots()

        ax = sns.heatmap(
            data,
            norm=LogNorm() if log_scale else None,
            cmap='viridis',
            cbar_kws=cbar_kws,
            ax=ax
        )

        # put the origin at the bottom left
        ax.invert_yaxis()
        ax.set_xlabel(f'allele count pop {pops[1]}')
        ax.set_ylabel(f'allele count pop {pops[0]}')

        # square cells, a grey frame, and unobtrusive color bar ticks (as for the 2-SFS plot)
        ax.set_aspect('equal')
        ax.collections[0].colorbar.ax.tick_params(size=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('grey')

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def plot_surface(
            self,
            pops: Tuple[int, int] = (0, 1),
            ax: 'plt.Axes' = None,
            title: str = None,
            log_scale: bool = False,
            mask_monomorphic: bool = True,
            cmap: str = 'viridis',
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot the joint SFS as a 3-dimensional surface. For more than two populations, the joint SFS is first
        marginalized onto the two requested populations (summing over the others).

        :param pops: The two population indices to plot (y-axis, x-axis).
        :param ax: Axes to plot on.
        :param title: Title of the plot.
        :param log_scale: Whether to use a logarithmic color scale.
        :param mask_monomorphic: Whether to mask the monomorphic corners (all-zero and all-derived).
        :param cmap: The colormap.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if len(pops) != 2:
            raise ValueError('Exactly two populations must be specified for a surface plot.')

        # reduce to the two requested populations
        data = (self.marginalize(pops) if self.n_pops > 2 else self).data.astype(float).copy()

        if data.ndim != 2:
            raise ValueError('Plotting requires a 2-dimensional (marginalized) joint SFS.')

        if mask_monomorphic:
            data[0, 0] = np.nan
            data[-1, -1] = np.nan

        # allele-count grid (0..n_p) for each of the two populations
        x_grid, y_grid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        ax.plot_surface(x_grid, y_grid, data, cmap=cmap, norm=LogNorm() if log_scale else None)

        ax.set_xlabel(f'allele count pop {pops[1]}')
        ax.set_ylabel(f'allele count pop {pops[0]}')
        ax.set_zlabel('branch length')

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def to_file(self, file: str):
        """
        Save to file (in JSON format).

        :param file: File path.
        """
        with open(file, 'w') as f:
            f.write(self.to_json())

    def to_json(self) -> str:
        """
        Convert to a JSON string.

        :return: JSON string.
        """
        obj = copy.deepcopy(self)

        # convert numpy array to list
        obj.data = obj.data.tolist()

        return jsonpickle.encode(obj)

    @staticmethod
    def from_file(file: str) -> 'JointSFS':
        """
        Load from file.

        :param file: File path.
        :return: JointSFS
        """
        with open(file, 'r') as f:
            return JointSFS.from_json(f.read())

    @staticmethod
    def from_json(json: str) -> 'JointSFS':
        """
        Load from a JSON string.

        :param json: JSON string.
        :return: JointSFS
        """
        obj = jsonpickle.decode(json)

        # convert list back to numpy array
        obj.data = np.array(obj.data)

        return obj
