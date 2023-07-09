from typing import List

import numpy as np
from fastdfe import Spectra, Spectrum
from matplotlib import pyplot as plt

from .demography import PiecewiseConstantDemography, ExponentialDemography
from .spectrum import SFS2
from .distributions_deprecated import VariablePopSizeCoalescent as VariablePopSizeCoalescentLegacy
from .distributions import PiecewiseConstantPopSizeCoalescent, ConstantPopSizeCoalescent, MsprimeCoalescent
from .serialization import Serializable


class Comparison(Serializable):
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """

    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List = None,
            times: np.ndarray | List = None,
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
        :param pop_sizes: Population sizes.
        :param times: Times of population size changes.
        :param growth_rate: Exponential growth rate so that at time ``t`` in the past we have
            ``N0 * exp(- growth_rate * t)``.
        :param N0: Initial population size (only used if growth_rate is specified).
        :param num_replicates: Number of replicates to use.
        :param n_threads: Number of threads to use.
        :param parallelize: Whether to parallelize the msprime simulations.
        :param alpha: Initial distribution of the phase-type coalescent.
        :param comparisons: Dictionary specifying which comparisons to make.
        :param kwargs: Additional arguments.
        """
        self.comparisons = comparisons

        # msprime coalescent
        self.ms = MsprimeCoalescent(
            n=n,
            pop_sizes=pop_sizes,
            times=times,
            growth_rate=growth_rate,
            N0=N0,
            num_replicates=num_replicates,
            n_threads=n_threads,
            parallelize=parallelize
        )

        if growth_rate is not None:
            # phase-type coalescent
            self.ph = PiecewiseConstantPopSizeCoalescent(
                n=n,
                demography=ExponentialDemography(
                    growth_rate=growth_rate,
                    N0=N0
                ),
                alpha=alpha,
                parallelize=parallelize
            )
        else:
            # phase-type coalescent
            self.ph = PiecewiseConstantPopSizeCoalescent(
                n=n,
                demography=PiecewiseConstantDemography(
                    pop_sizes=pop_sizes,
                    times=times
                ),
                alpha=alpha,
                parallelize=parallelize
            )

        # legacy phase-type coalescent
        self.ph_legacy = VariablePopSizeCoalescentLegacy(
            n=n,
            demography=PiecewiseConstantDemography(
                pop_sizes=pop_sizes,
                times=times
            ),
            alpha=alpha[:-1] if alpha is not None else None
        )

        # phase-type coalescent with constant population size
        self.ph_const = ConstantPopSizeCoalescent(
            n=n,
            Ne=pop_sizes[0],
            alpha=alpha,
            parallelize=parallelize
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

    def compare(self):
        """
        Compare the distributions of the given statistic for the given types.
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
                    elif isinstance(ph_stat, np.ndarray):

                        diff = self.rel_diff(ms_stat, ph_stat).max()

                        if ph_stat.ndim == 1:

                            s = Spectra.from_spectra(dict(ms=Spectrum(ms_stat), ph=Spectrum(ph_stat)))

                            s.plot()

                        else:
                            _, axs = plt.subplots(ncols=2, subplot_kw={"projection": "3d"}, figsize=(8, 4))

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
                        props = dict((k, self.ms.__dict__[k]) for k in ['n', 'pop_sizes', 'times'])
                        plt.title(f"{stat.upper()} for {props}")

                        plt.show()

                        diff = self.rel_diff(y_ms, y_ph).max()

                    assert diff < tol, f"Maximum difference {diff} exceeds threshold {tol}."
