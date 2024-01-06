from functools import cached_property
from typing import Iterable, Dict, Literal, List

import numpy as np
from fastdfe import Spectra, Spectrum
import yaml
from matplotlib import pyplot as plt

from .locus import LocusConfig
from .demography import Demography, DiscreteRateChanges
from .distributions import Coalescent, MsprimeCoalescent, PhaseTypeDistribution
from .coalescent_models import CoalescentModel, StandardCoalescent, BetaCoalescent, DiracCoalescent
from .serialization import Serializable
from .spectrum import SFS2


class Comparison(Serializable):
    """
    Class for simulation population genetic scenarios
    using both phase-type theory and msprime, for comparison.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int],
            pop_sizes: Dict[str, Dict[float, float]],
            migration_rates: Dict[tuple[str, str], Dict[float, float]] = {},
            n_loci: int = 1,
            recombination_rate: float = 0,
            num_replicates: int = 10000,
            record_migration: bool = False,
            n_threads: int = 100,
            parallelize: bool = True,
            max_iter: int = 100,
            rtol: float = 1e-9,
            atol: float = 1e-16,
            comparisons: dict = None,
            model: Literal['standard', 'beta'] = 'standard',
            alpha: float = 1.5,
            psi: float = 0.5,
            c: float = 1
    ):
        """
        Initialize Comparison object.

        :param n: Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values.
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
        :param n_loci: Number of loci.
        :param recombination_rate: Recombination rate.
        :param num_replicates: Number of replicates to use.
        :param record_migration: Whether to record migrations.
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
        self.n_loci = n_loci
        self.recombination_rate = recombination_rate
        self.num_replicates = num_replicates
        self.record_migration = record_migration
        self.n_threads = n_threads
        self.parallelize = parallelize
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol
        self.alpha = alpha
        self.psi = psi
        self.c = c

        self.model = self.load_coalescent_model(model)

    @staticmethod
    def from_yaml(file: str) -> 'Comparison':
        """
        Load the comparison from a YAML file.

        :param file: Path to YAML file.
        """
        # load config from file
        with open(file, 'r') as f:
            config = yaml.full_load(f)

        return Comparison(**config)

    def get_demography(self) -> Demography:
        """
        Get the demography.
        """
        return Demography(events=[
            DiscreteRateChanges(pop_sizes=self.pop_sizes, migration_rates=self.migration_rates)
        ])

    def get_locus_config(self) -> LocusConfig:
        """
        Get the locus configuration.
        """
        return LocusConfig(
            n=self.n_loci,
            recombination_rate=self.recombination_rate
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
            loci=self.get_locus_config(),
            parallelize=self.parallelize,
            model=self.model,
            max_iter=self.max_iter,
            rtol=self.rtol,
            atol=self.atol
        )

    @cached_property
    def ms(self):
        """
        Get the msprime coalescent.
        """
        return MsprimeCoalescent(
            n=self.n,
            demography=self.get_demography(),
            loci=self.get_locus_config(),
            num_replicates=self.num_replicates,
            record_migration=self.record_migration,
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

    def compare_dist_(
            self,
            ph: PhaseTypeDistribution,
            ms: PhaseTypeDistribution,
            stat: Literal['pdf', 'cdf', 'mean', 'var', 'std', 'cov', 'corr'],
            tol: float,
            visualize: bool = True,
            title: str = 'stat',
            do_assertion: bool = True
    ):
        """
        Compare the given distributions and return their difference.

        :param ph: Phase-type distribution.
        :param ms: Phase-type distribution.
        :param stat: Statistic to compare.
        :param tol: Tolerance.
        :param visualize: Whether to plot the distributions what is being compared.
        :param title: Title of the plot.
        :param do_assertion: Whether to assert that the distributions are the same.
        """
        ph_stat = getattr(ph, stat)
        ms_stat = getattr(ms, stat)

        if isinstance(ph_stat, float):

            diff = self.rel_diff(ms_stat, ph_stat).max()

        # assume we have an SFS
        elif isinstance(ph_stat, Iterable):

            ms_stat = np.array(list(ms_stat))
            ph_stat = np.array(list(ph_stat))
            diff = self.rel_diff(ms_stat, ph_stat).max()

            if visualize:
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
        elif stat in ['pdf', 'cdf']:

            # use cached values if available
            if hasattr(ms, '_cache') and stat in ms._cache:
                t = ms._cache['t']
                y_ms = ms._cache[stat]
            else:
                t = np.linspace(0, self.ph.tree_height.quantile(0.99), 100)
                y_ms = ms_stat(t)

            y_ph = ph_stat(t)

            if visualize:
                plt.plot(t, y_ph, label='phasegen')
                plt.plot(t, y_ms, label='msprime')

                plt.legend()
                plt.title(f"{stat.upper()}: {title}")

                plt.show()

            diff = np.abs(y_ms - y_ph).mean() if stat == 'pdf' else self.rel_diff(y_ms, y_ph)[2:].max()

        else:
            raise ValueError(f"Unknown type {type(ph_stat)}.")

        if do_assertion:
            if not diff <= tol:
                raise AssertionError(f"Maximum relative difference {diff} exceeds threshold {tol}.")

        plt.clf()

    def _compare_stat_recursively(
            self,
            ph: PhaseTypeDistribution,
            ms: PhaseTypeDistribution,
            data: dict,
            do_assertion: bool = True,
            visualize: bool = True,
            title: str = 'stat'
    ):
        """
        Compare the given statistics recursively.

        :param ph: Phase-type distribution.
        :param ms: Phase-type distribution.
        :param data: Dictionary of statistics to compare, possibly nested.
        :param do_assertion: Whether to assert that the distributions are the same.
        :param visualize: Whether to plot what is being compared.
        :param title: Title of the plot.
        """

        # statistic, distribution or nested demes dictionary
        stat: Literal['pdf', 'cdf', 'mean', 'var', 'std', 'cov', 'corr', 'demes']

        # tolerance or dictionary of statistics
        sub: float | dict

        for stat, sub in data.items():

            if stat == 'demes':

                for pop in self.ph.demography.pop_names:
                    self._compare_stat_recursively(
                        ph=ph.demes[pop],
                        ms=ms.demes[pop],
                        data=sub,
                        visualize=visualize,
                        title=f"{title}: {pop}",
                    )
            else:

                self.compare_dist_(
                    ph=ph,
                    ms=ms,
                    stat=stat,
                    tol=sub,
                    visualize=visualize,
                    title=title,
                    do_assertion=do_assertion
                )

    def compare(self, title: str = None, do_assertion: bool = True, visualize: bool = True):
        """
        Compare the distributions of the given statistics.

        :param title: Title of the plot.
        :param do_assertion: Whether to assert that the distributions are the same.
        :param visualize: Whether to plot what is being compared.
        :raises AssertionError: If `do_assertion is True and the distributions differ by more than the given tolerance.
            ValueError: if the type is unknown.
        """
        for dist, data in self.comparisons['tolerance'].items():
            self._compare_stat_recursively(
                do_assertion=do_assertion,
                ph=getattr(self.ph, dist),
                ms=getattr(self.ms, dist),
                data=data,
                visualize=visualize,
                title=title
            )
