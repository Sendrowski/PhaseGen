from functools import cached_property
from typing import List, Iterable, Dict, Literal

import numpy as np
from fastdfe import Spectra, Spectrum
from matplotlib import pyplot as plt

from .coalescent_models import CoalescentModel, StandardCoalescent, BetaCoalescent, DiracCoalescent
from .demography import PiecewiseConstantDemography, Demography
from .distributions import Coalescent, _MsprimeCoalescent
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
            pop_sizes: Dict[str, Dict[float, float]] | List[Dict[float, float]] | Dict[float, float],
            migration_rates: Dict[float, np.ndarray] | None = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            comparisons: dict = None,
            model: Literal['standard', 'beta'] = 'standard',
            alpha: float = 1.5,
            psi: float = 0.5,
            c: float = 1
    ):
        """
        Initialize Comparison object.

        :param n: Number of lineages.
        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name, or a list of dictionaries of the form ``{time1: size1, time2: size2}`` ordered
            by population index, or a single dictionary of the form ``{time1: size1, time2: size2}`` for a single
            population. Note that the first time must always be 0.
        :param migration_rates: Migration matrix. Use ``None`` for no migration.
            A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2: rate2}}`` where ``m_ij`` is the
            migration rate from population ``pop_i`` to population ``pop_j`` at time ``time1`` and ``time2`` etc.
            Alternatively, a dictionary of 2-dimensional numpy arrays where the rows correspond to the source
            population and the columns to the destination. Note that migration rates for which the source and
            destination population are the same are ignored and that the first time must always be 0.
        :param num_replicates: Number of replicates to use.
        :param n_threads: Number of threads to use.
        :param parallelize: Whether to parallelize the msprime simulations.
        :param alpha: Initial distribution of the phase-type coalescent.
        :param comparisons: Dictionary specifying which comparisons to make.
        :param model: Coalescent model to use.
        :param alpha: Alpha parameter of the beta coalescent.
        :param psi: Psi parameter of the Dirac coalescent.
        :param c: C parameter of the Dirac coalescent.
        """
        self.comparisons = comparisons
        self.n = n
        self.pop_sizes = pop_sizes
        self.migration_rates = migration_rates
        self.num_replicates = num_replicates
        self.n_threads = n_threads
        self.parallelize = parallelize
        self.alpha = alpha
        self.psi = psi
        self.c = c

        self.model = self.load_coalescent_model(model)

    def get_demography(self) -> Demography:
        """
        Get the demography.
        """
        return PiecewiseConstantDemography(
            pop_sizes=self.pop_sizes,
            migration_rates=self.migration_rates
        )

    def load_coalescent_model(
            self,
            name: Literal['standard', 'beta', 'dirac']
    ) -> CoalescentModel:
        """
        Load the coalescent model.

        :param name: Name of the coalescent model.
        :return: The coalescent model.
        :raises ValueError: if the name is unknown.
        """
        if name == 'standard':
            return StandardCoalescent()

        if name == 'beta':
            return BetaCoalescent(alpha=self.alpha)

        if name == 'dirac':
            return DiracCoalescent(psi=self.psi, c=self.c)

        raise ValueError(f"Unknown coalescent model {name}.")

    @cached_property
    def ph(self):
        """
        Get the phase-type coalescent.
        """
        return Coalescent(
            n=self.n,
            demography=self.get_demography(),
            parallelize=self.parallelize,
            model=self.model
        )

    @cached_property
    def ms(self):
        """
        Get the msprime coalescent.
        """
        return _MsprimeCoalescent(
            n=self.n,
            demography=self.get_demography(),
            num_replicates=self.num_replicates,
            n_threads=self.n_threads,
            parallelize=self.parallelize,
            model=self.model
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

    def compare(self, title: str = None, do_assertion: bool = True, plot: bool = True):
        """
        Compare the distributions of the given statistic for the given types.

        :param title: Title of the plot.
        :param do_assertion: Whether to assert that the distributions are the same.
        :param plot: Whether to plot the distributions.
        :raises AssertionError: If `do_assertion is True and the distributions differ by more than the given tolerance.
            ValueError: if the type is unknown.
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

                        if plot:
                            if ph_stat.ndim == 1:

                                s = Spectra.from_spectra(dict(ms=Spectrum(ms_stat), ph=Spectrum(ph_stat)))

                                s.plot(title=title)

                            # assume we have a 2-dimensional SFS
                            elif len(ph_stat) > 3:
                                fig, axs = plt.subplots(ncols=2, subplot_kw={"projection": "3d"}, figsize=(8, 4))

                                fig.suptitle(f"{stat}: {title}")

                                SFS2(ph_stat).plot(ax=axs[0], title='ph', show=False)
                                SFS2(ms_stat).plot(ax=axs[1], title='ms')

                    # assume we have a PDF or CDF
                    elif callable(ph_stat):

                        x = np.linspace(0, getattr(getattr(self, t), dist).quantile(0.99), 100)

                        y_ph = ph_stat(x)
                        y_ms = ms_stat(x)

                        if plot:
                            plt.plot(x, y_ph, label=t)
                            plt.plot(x, y_ms, label='msprime')

                            plt.legend()
                            plt.title(f"{stat.upper()}: {title}")

                            plt.show()
                        
                        diff = np.abs(y_ms - y_ph).mean() if stat == 'pdf' else self.rel_diff(y_ms, y_ph)[1:].max()

                    else:
                        raise ValueError(f"Unknown type {type(ph_stat)}.")

                    if do_assertion:
                        if not diff <= tol:
                            raise AssertionError(f"Maximum relative difference {diff} exceeds threshold {tol}.")

                    plt.clf()
