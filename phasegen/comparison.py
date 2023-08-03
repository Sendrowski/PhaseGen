from functools import cached_property
from typing import List, Iterable, Dict, Tuple

import numpy as np
from fastdfe import Spectra, Spectrum
from matplotlib import pyplot as plt

from .demography import PiecewiseTimeHomogeneousDemography, ExponentialDemography, TimeHomogeneousDemography, Demography
from .distributions import PiecewiseTimeHomogeneousCoalescent, TimeHomogeneousCoalescent, MsprimeCoalescent
from .serialization import Serializable
from .spectrum import SFS2


class Comparison(Serializable):
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """

    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List | Dict[str, List] = None,
            times: np.ndarray | List | Dict[str, List] = None,
            migration_matrix: np.ndarray | List | Dict[Tuple[str, str], float] = None,
            growth_rate: float = None,
            N0: float = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            alpha: np.ndarray | List = None,
            comparisons: dict = None
    ):
        """
        Initialize Comparison object.

        :param n: Number of lineages.
        :param pop_sizes: Population sizes at different times or different populations if a dictionary is given.
        :param times: Times of population size changes or different populations if a dictionary is given.
        :param migration_matrix: Migration matrix.
        :param growth_rate: Exponential growth rate so that at time ``t`` in the past we have
            ``N0 * exp(- growth_rate * t)``.
        :param N0: Initial population size (only used if growth_rate is specified).
        :param num_replicates: Number of replicates to use.
        :param n_threads: Number of threads to use.
        :param parallelize: Whether to parallelize the msprime simulations.
        :param alpha: Initial distribution of the phase-type coalescent.
        :param comparisons: Dictionary specifying which comparisons to make.
        """
        self.comparisons = comparisons
        self.n = n
        self.pop_sizes = pop_sizes
        self.times = times
        self.migration_matrix = migration_matrix
        self.growth_rate = growth_rate
        self.N0 = N0
        self.num_replicates = num_replicates
        self.n_threads = n_threads
        self.parallelize = parallelize
        self.alpha = alpha

    def get_demography(self) -> Demography:
        """
        Get the demography.
        """
        if self.growth_rate is not None:
            return ExponentialDemography(
                growth_rate=self.growth_rate,
                N0=self.N0,
                migration_matrix=self.migration_matrix
            )
        else:
            return PiecewiseTimeHomogeneousDemography(
                pop_sizes=self.pop_sizes,
                times=self.times,
                migration_matrix=self.migration_matrix
            )

    @cached_property
    def ph(self):
        """
        Get the phase-type coalescent.
        """
        return PiecewiseTimeHomogeneousCoalescent(
            n=self.n,
            demography=self.get_demography(),
            parallelize=self.parallelize
        )

    @cached_property
    def ph_const(self):
        """
        Get the phase-type coalescent with constant population size.
        """
        return TimeHomogeneousCoalescent(
            n=self.n,
            demography=TimeHomogeneousDemography(
                pop_size=self.pop_sizes[0],
                migration_matrix=self.migration_matrix
            ),
            parallelize=self.parallelize
        )

    @cached_property
    def ms(self):
        """
        Get the msprime coalescent.
        """
        return MsprimeCoalescent(
            n=self.n,
            demography=self.get_demography(),
            num_replicates=self.num_replicates,
            n_threads=self.n_threads,
            parallelize=self.parallelize
        )

    @staticmethod
    def rel_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the maximum relative difference between two arrays.

        :param a: The first array.
        :param b: The second array.
        :return: The mean relative difference.
        """
        a, b = np.array(a), np.array(b)

        # compute relative difference
        diff = np.abs(a - b) / ((a + b) / 2)

        # only consider finite values
        return diff[np.isfinite(diff)]

    def compare(self, title: str = None, do_assertion: bool = True):
        """
        Compare the distributions of the given statistic for the given types.

        :param title: Title of the plot.
        :param do_assertion: Whether to assert that the distributions are the same.
        """
        # iterate over types
        for t in self.comparisons['types']:

            # iterate over distributions
            for dist, value in self.comparisons['tolerance'].items():

                # iterate over statistics
                for stat, tol in value.items():

                    ph_stat = getattr(getattr(getattr(self, t), dist), stat)
                    ms_stat = getattr(getattr(self.ms, dist), stat)

                    if isinstance(ph_stat, float):

                        diff = self.rel_diff(ms_stat, ph_stat).max()

                    # assume we have an SFS
                    elif isinstance(ph_stat, Iterable):

                        ms_stat = np.array(list(ms_stat))
                        ph_stat = np.array(list(ph_stat))
                        diff = self.rel_diff(ms_stat, ph_stat).max()

                        if ph_stat.ndim == 1:

                            s = Spectra.from_spectra(dict(ms=Spectrum(ms_stat), ph=Spectrum(ph_stat)))

                            s.plot(title=title)

                        else:
                            fig, axs = plt.subplots(ncols=2, subplot_kw={"projection": "3d"}, figsize=(8, 4))

                            fig.suptitle(f"{stat}: {title}")

                            SFS2(ph_stat).plot(ax=axs[0], title='ph', show=False)
                            SFS2(ms_stat).plot(ax=axs[1], title='ms')

                    # assume we have a PDF or CDF
                    elif callable(ph_stat):

                        x = np.linspace(0, 2, 100)

                        y_ph = ph_stat(x)
                        y_ms = ms_stat(x)

                        plt.plot(x, y_ph, label=t)
                        plt.plot(x, y_ms, label='msprime')

                        plt.legend()
                        plt.title(f"{stat.upper()}: {title}")

                        plt.show()

                        diff = self.rel_diff(y_ms, y_ph).max()

                    else:
                        raise ValueError(f"Unknown type {type(ph_stat)}.")

                    if do_assertion:
                        assert diff < tol, f"Maximum relative difference {diff} exceeds threshold {tol}."

                    plt.clf()
