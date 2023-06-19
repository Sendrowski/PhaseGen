from typing import List

import numpy as np
from fastdfe import Spectra, Spectrum
from matplotlib import pyplot as plt

from . import PiecewiseConstantDemography, VariablePopSizeCoalescent, MsprimeCoalescent, ConstantPopSizeCoalescent
from .serialization import Serializable
from .distributions_deprecated import VariablePopSizeCoalescent as VariablePopSizeCoalescentLegacy


class Comparison(Serializable):
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """

    def __init__(
            self,
            n: int,
            pop_sizes: np.ndarray | List,
            times: np.ndarray | List,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            alpha: np.ndarray | List = None,
            comparisons: dict = None
    ):
        """
        Initialize Comparison object.

        :param n: Sample size.
        :param pop_sizes: Population sizes.
        :param times: Times of population size changes.
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
            num_replicates=num_replicates,
            n_threads=n_threads,
            parallelize=parallelize
        )

        # phase-type coalescent
        self.ph = VariablePopSizeCoalescent(
            n=n,
            demography=PiecewiseConstantDemography(
                pop_sizes=pop_sizes,
                times=times
            ),
            alpha=alpha
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
            alpha=alpha
        )

    @staticmethod
    def mean_relative_difference(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the mean relative difference between two arrays.

        :param a: The first array.
        :param b: The second array.
        :return: The mean relative difference.
        """
        a, b = np.array(a), np.array(b)
        diff = np.abs(a - b) / ((a + b) / 2)
        diff = diff[np.isfinite(diff)]

        return np.mean(diff)

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

                        diff_rel = self.mean_relative_difference(ms_stat, ph_stat)
                        assert diff_rel < tol, f"Difference {diff_rel} exceeds threshold {tol} for {stat} of {dist}."

                    # assume we have an SFS
                    elif isinstance(ph_stat, np.ndarray):

                        mean_diff_rel = self.mean_relative_difference(ms_stat, ph_stat)

                        s = Spectra.from_spectra(dict(ms=Spectrum(ms_stat), ph=Spectrum(ph_stat)))

                        s.plot()

                        assert mean_diff_rel < tol, f"Difference mean {mean_diff_rel} exceeds threshold {tol}."

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

                        mean_diff_rel = self.mean_relative_difference(y_ms, y_ph)
                        assert mean_diff_rel < tol, f"Difference mean {mean_diff_rel} exceeds threshold {tol}."


