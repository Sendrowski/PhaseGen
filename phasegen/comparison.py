"""
Compare statistics between PhaseGen and Msprime.
"""
import ast
import copy
import itertools
import logging
import os
import time
from .caching import cached_property
from typing import Iterable, Dict, Literal, List

import numpy as np
import yaml
from fastdfe import Spectra
from matplotlib import pyplot as plt

from .coalescent_models import CoalescentModel, StandardCoalescent, BetaCoalescent, DiracCoalescent
from .demography import Demography, DiscreteRateChanges
from .distributions import Coalescent, MsprimeCoalescent, SampledCoalescent, PhaseTypeDistribution, \
    MarginalDistributions, MarginalLocusDistributions, MarginalDemeDistributions
from .locus import LocusConfig
from .serialization import Serializable
from .spectrum import SFS, JointSFS, TwoLocusSFS
from .utils import takewhile_inclusive

logger = logging.getLogger('phasegen')


class Comparison(Serializable):
    """
    Class for comparing statistics between PhaseGen and Msprime.
    """
    # DPI of the saved figure
    dpi = 300

    # Path to save the figure to
    figure_path: str = None

    # Whether to assert that the distributions are the same
    do_assertion: bool = True

    # Whether to visualize the distributions
    visualize: bool = True

    # Whether to show the title of the plot
    show_title: bool = True

    def __init__(
            self,
            n: int | Dict[str, int] | List[int],
            pop_sizes: Dict[str, Dict[float, float]],
            migration_rates: Dict[tuple[str, str], Dict[float, float]] = None,
            n_loci: int = 1,
            recombination_rate: float = 0,
            num_replicates: int = 10000,
            n_samples: int = None,
            mutation_rate: float = None,
            record_migration: bool = False,
            simulate_mutations: bool = False,
            mass_threshold: float = 0.9,
            end_time: float = None,
            n_threads: int = 100,
            parallelize: bool = True,
            seed: int = None,
            comparisons: dict = None,
            model: Literal['standard', 'beta'] = 'standard',
            alpha: float = 1.5,
            psi: float = 0.5,
            c: float = 1
    ) -> None:
        """
        Initialize Comparison object.

        :param n: Either a single integer if only one population, or a list of integers
            or a dictionary with population names as keys and number of lineages as values.
        :param pop_sizes: Population sizes. Either a dictionary of the form ``{pop_i: {time1: size1, time2: size2}}``,
            indexed by population name, or a list of dictionaries of the form ``{time1: size1, time2: size2}`` ordered
            by population index, or a single dictionary of the form ``{time1: size1, time2: size2}`` for a single
            population. Note that the first time must always be 0.
        :param migration_rates: Migration matrix. Use ```None``` for no migration.
            A dictionary of the form ``{(pop_i, pop_j): {time1: rate1, time2: rate2}}`` where ``m_ij`` is the
            migration rate from population ``pop_i`` to population ``pop_j`` at time ``time1`` and `time2` etc.
            Alternatively, a dictionary of 2-dimensional numpy arrays where the rows correspond to the source
            population and the columns to the destination. Note that migration rates for which the source and
            destination population are the same are ignored and that the first time must always be 0.
        :param n_loci: Number of loci.
        :param recombination_rate: Recombination rate.
        :param num_replicates: Number of replicates to use.
        :param n_samples: If set, the ``ms`` operand is PhaseGen's own trajectory sampler
            (:class:`~phasegen.distributions.SampledCoalescent`) drawing ``n_samples`` trajectories, instead of
            msprime. The comparison then validates PhaseGen's sampler against its exact analytic distributions.
        :param mutation_rate: Mutation rate. Only used if simulate_mutations is True.
        :param record_migration: Whether to record migrations.
        :param simulate_mutations: Whether to simulate mutations. This is used for comparing mutational configurations
            rather than branch lengths.
        :param mass_threshold: Probability threshold above which to stop generating mutational configurations.
        :param end_time: End time of the computation.
        :param n_threads: Number of threads to use.
        :param parallelize: Whether to parallelize the msprime simulations.
        :param seed: Seed for the random number generator.
        :param alpha: Initial distribution of the phase-type coalescent.
        :param comparisons: Dictionary specifying which comparisons to make.
        :param model: Coalescent model to use.
        :param alpha: Alpha parameter of the beta coalescent.
        :param psi: Psi parameter of the Dirac coalescent.
        :param c: C parameter of the Dirac coalescent.
        """
        if migration_rates is None:
            migration_rates = {}

        self.logger = logging.getLogger('phasegen').getChild(self.__class__.__name__)

        self.comparisons = comparisons
        self.n = n
        self.pop_sizes = pop_sizes
        self.migration_rates = migration_rates
        self.n_loci = n_loci
        self.recombination_rate = recombination_rate
        self.num_replicates = num_replicates
        self.n_samples = n_samples
        self.mutation_rate = mutation_rate
        self.record_migration = record_migration
        self.simulate_mutations = simulate_mutations
        self.mass_threshold = mass_threshold
        self.end_time = end_time
        self.n_threads = n_threads
        self.parallelize = parallelize
        self.seed = seed
        self.alpha = alpha
        self.psi = psi
        self.c = c

        self.model = self.load_coalescent_model(model)

        #: Number of assertions made
        self.n_assertions: int = 0

        #: Wall-clock runtime (seconds) of the phasegen side of each compared statistic, keyed by its title.
        self.runtimes: dict = {}

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

    def _make_coalescent(self) -> 'Coalescent':
        """
        Build a fresh analytic PhaseGen coalescent from the configuration.
        """
        return Coalescent(
            n=self.n,
            demography=self.get_demography(),
            loci=self.get_locus_config(),
            end_time=self.end_time,
            model=self.model
        )

    @cached_property
    def ph(self) -> 'Coalescent':
        """
        PhaseGen coalescent (the exact analytic reference operand).
        """
        return self._make_coalescent()

    @cached_property
    def ms(self) -> 'MsprimeCoalescent | SampledCoalescent':
        """
        The empirical (candidate) operand: PhaseGen's own trajectory sampler when ``n_samples`` is set (validated
        against the exact analytic :attr:`ph`), otherwise the msprime simulation (the independent ground truth).
        """
        if self.n_samples is not None:
            # a fresh analytic coalescent (not self.ph, which must stay out of the serialized fixture); it is
            # dropped before serialization
            return SampledCoalescent(
                coalescent=self._make_coalescent(),
                n_samples=self.n_samples,
                seed=self.seed
            )

        return MsprimeCoalescent(
            n=self.n,
            demography=self.get_demography(),
            loci=self.get_locus_config(),
            num_replicates=self.num_replicates,
            mutation_rate=self.mutation_rate,
            record_migration=self.record_migration,
            simulate_mutations=self.simulate_mutations,
            end_time=self.end_time,
            n_threads=self.n_threads,
            parallelize=self.parallelize,
            seed=self.seed,
            model=self.model
        )

    @classmethod
    def rel_diff(cls, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
        """
        Compute the relative difference between two arrays.

        :param a: The first array.
        :param b: The second array.
        :return: The relative difference.
        """
        # vectorize
        if not isinstance(a, Iterable) and not isinstance(b, Iterable):
            return cls.rel_diff([a], [b])[0]

        a, b = np.array(a), np.array(b)

        # compute relative difference; where both values are 0 the 0/0 is replaced by 0 below, so silence the
        # benign divide warning at the source
        with np.errstate(divide='ignore', invalid='ignore'):
            diff = np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2)

        # set relative difference to 0 if both values are 0
        diff[(a == 0) & (b == 0)] = 0

        return diff

    def _save_and_show(self, name: str, pad=2, extra_right: float = 0.0) -> None:
        """
        Save and show the figure if a figure path is set.

        :param name: File name for the saved figure.
        :param extra_right: Extra whitespace (inches) added to the right of the tight bounding box (e.g. for the 3D
            surface panels, whose rightmost axis labels otherwise sit flush against the edge).
        """
        plt.tight_layout(pad=pad)

        if self.figure_path is not None:
            if not os.path.exists(self.figure_path):
                os.makedirs(self.figure_path)

            path = self.figure_path + f'/{name}.png'
            # bbox_inches='tight' expands the saved bounding box to include every artist -- tight_layout alone does not
            # account for 3D z-axis labels, so the rightmost surface panel's axis label would otherwise be clipped
            bbox = 'tight'
            if extra_right:
                try:  # extend the tight bbox on the right only (a uniform pad_inches would pad all four sides)
                    from matplotlib.transforms import Bbox
                    fig = plt.gcf()
                    fig.canvas.draw()
                    tb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
                    bbox = Bbox.from_extents(tb.x0, tb.y0, tb.x1 + extra_right, tb.y1)
                except Exception:
                    bbox = 'tight'
            plt.savefig(path, dpi=self.dpi, bbox_inches=bbox)

        plt.show()
        # under headless Agg, free the figure right away; for a display backend (native window or
        # PyCharm SciView) leave it open so the plot actually lands -- test teardown closes it
        if plt.get_backend().lower() == 'agg':
            plt.close('all')

    def compare_stat(
            self,
            ph: PhaseTypeDistribution,
            ms: PhaseTypeDistribution,
            stat: Literal['pdf', 'cdf', 'pairwise_cdf', 'mean', 'var', 'std', 'cov', 'corr', 'demes', 'loci', 'm3', 'm4'],
            tol: float,
            title: str = 'stat',
            name: str = '',
            mode: str = None
    ) -> None:
        """
        Compare the given distributions and return their difference.

        :param ph: Phase-type distribution.
        :param ms: Phase-type distribution.
        :param stat: Statistic to compare.
        :param tol: Tolerance.
        :param title: Title of the plot.
        :param name: Name of the plot.
        """
        title = f"{title}: {stat}"
        name = f"{name}_{stat}"
        t0 = time.perf_counter()  # time the phasegen-side evaluation + diff of this statistic

        ph_stat, ms_stat = self._fetch_stat(ph, ms, stat)

        diff = 0.0
        plot = None  # deferred visualisation: invoked with the final result message, so the plot title == the log line

        if isinstance(ph_stat, float):
            diff = self.rel_diff(ms_stat, ph_stat).max()
        elif stat == 'mutation_configs':
            diff, plot = self._diff_and_plot_mutation_configs(ph_stat, ms_stat, name)
        elif isinstance(ph_stat, Iterable):  # a spectrum: SFS / jSFS / 2-SFS / covariance matrix
            diff, plot = self._diff_and_plot_spectrum(ph_stat, ms_stat, stat, name)
        elif stat in ['pdf', 'cdf', 'quantile']:
            diff, plot = self._diff_and_plot_curve(ph, ms, ph_stat, ms_stat, stat, mode, name)
        else:
            raise ValueError(f"Unknown type {type(ph_stat)}.")

        runtime = time.perf_counter() - t0
        self.runtimes = getattr(self, 'runtimes', {})  # robust to deserialized objects that bypass __init__
        self.runtimes[title] = runtime

        msg = self._result_message(title, diff, tol, self._diff_label(stat), runtime)
        if self.visualize and plot is not None:
            plot(msg)
        self._log_result(msg, diff, tol)

    def _fetch_stat(self, ph: PhaseTypeDistribution, ms: PhaseTypeDistribution, stat: str) -> tuple:
        """Fetch the ``(phasegen, msprime)`` statistic pair for ``stat``: the centered higher moments (``m3``/``m4``),
        the mutation-configuration probabilities (truncated at the mass threshold), or the named attribute otherwise."""
        if stat in ['m3', 'm4']:
            return ph.moment(int(stat[1]), center=False), getattr(ms, stat)

        if stat == 'mutation_configs':
            ph_it = ph.get_mutation_configs(theta=self.mutation_rate)
            ph_stat = list(takewhile_inclusive(lambda _: ph.generated_mass < self.mass_threshold, ph_it))
            # align the msprime probabilities to phasegen's configurations by key, so the comparison is independent
            # of the generation order (phasegen generates by descending probability, msprime by ascending count)
            ms_stat = [(config, ms.get_mutation_config(config)) for config, _ in ph_stat]
            return ph_stat, ms_stat

        return getattr(ph, stat), getattr(ms, stat)

    def _diff_and_plot_mutation_configs(self, ph_stat, ms_stat, name: str) -> tuple:
        """Total-variation distance between the mutation-configuration probability distributions, with a deferred line
        plot. The configs are a probability distribution (over descendant-count configurations), so the natural
        discrepancy is the total variation ``0.5 * sum|p_ph - p_ms|`` -- bounded, mass-weighted, and the fraction of
        probability mass misallocated -- rather than a mean per-config *relative* difference, which the rare,
        near-zero-probability configs (where the relative difference saturates) would dominate as sampling noise."""
        configs = [x[0] for x in ph_stat]
        ms_stat = np.array([x[1] for x in ms_stat])
        ph_stat = np.array([x[1] for x in ph_stat])
        diff = 0.5 * float(np.abs(ph_stat - ms_stat).sum())

        plot = None
        if self.visualize:
            def plot(msg, ph_stat=ph_stat, ms_stat=ms_stat, configs=configs) -> None:
                plt.plot(ph_stat, label='phasegen')
                plt.plot(ms_stat, label='msprime')
                plt.xticks(range(len(configs)), [str(config) for config in configs], rotation=90)
                plt.legend()
                if self.show_title: plt.title(msg, fontsize=self.title_fontsize)
                self._save_and_show(name)

        return diff, plot

    def _diff_and_plot_spectrum(self, ph_stat, ms_stat, stat: str, name: str) -> tuple:
        """Worst relative difference of a spectrum statistic (SFS / jSFS / 2-SFS / covariance matrix), with a deferred
        plot chosen by its shape: side-by-side heatmaps for a 2-D joint / two-locus SFS, a grouped bar + per-bin
        difference for a 1-D SFS, or phasegen / msprime / element-wise-difference surfaces for a square matrix."""
        # whether this is a joint (multi-population) SFS or a two-locus SFS -- both are 2-D spectra drawn as
        # side-by-side heatmaps (the joint SFS may be rectangular / higher-dimensional, the two-locus SFS square)
        is_joint = isinstance(ph_stat, JointSFS)
        heatmap_cls = JointSFS if is_joint else (TwoLocusSFS if isinstance(ph_stat, TwoLocusSFS) else None)

        ms_stat = np.array(list(ms_stat))
        ph_stat = np.array(list(ph_stat))
        diff = self.rel_diff(ms_stat, ph_stat).max()

        plot = None
        if self.visualize:
            if heatmap_cls is not None and ph_stat.ndim == 2:
                # plot the joint / two-locus SFS as side-by-side heatmaps, but only when it is 2-dimensional
                def plot(msg, ph_stat=ph_stat, ms_stat=ms_stat, cls=heatmap_cls) -> None:
                    plt.close('all')  # avoid empty plots
                    fig, axs = plt.subplots(ncols=2, figsize=(8, 5))
                    if self.show_title: plt.suptitle(msg, fontsize=self.suptitle_fontsize)
                    axs[0].set_title('phasegen', fontsize=self.title_fontsize)
                    axs[1].set_title('msprime', fontsize=self.title_fontsize)
                    cls(ph_stat).plot(ax=axs[0], show=False)
                    cls(ms_stat).plot(ax=axs[1], show=False)
                    self._save_and_show(name, pad=1.5)

            elif heatmap_cls is None and ph_stat.ndim == 1:
                def plot(msg, ph_stat=ph_stat, ms_stat=ms_stat) -> None:
                    self._plot_sfs_with_diff(ph_stat, ms_stat, msg if self.show_title else None, name,
                                             left_title=name.upper() if name else 'SFS')

            # a square 2-dimensional statistic (an SFS covariance / correlation matrix or a 2-SFS); n = 2 (a 3x3
            # matrix with a single polymorphic bin) is a legitimate two-locus SFS. Drawn as phasegen / msprime /
            # element-wise relative-difference surfaces.
            elif ph_stat.ndim == 2 and ph_stat.shape[0] == ph_stat.shape[1] and len(ph_stat) > 2:
                def plot(msg, ph_stat=ph_stat, ms_stat=ms_stat) -> None:
                    idx = np.arange(len(ph_stat))
                    self._plot_surface_triple(
                        idx, idx, ph_stat, ms_stat, self.rel_diff(ms_stat, ph_stat), zlabel=stat,
                        xlabel='frequency class i', ylabel='frequency class j',
                        title=msg if self.show_title else None, name=name)

        return diff, plot

    def _diff_and_plot_curve(self, ph, ms, ph_stat, ms_stat, stat: str, mode: str, name: str) -> tuple:
        """Difference of a pdf / cdf / quantile curve (per-point or per-bin), with a deferred two-panel curve +
        difference plot. The msprime curve uses cached grid values when available; the phasegen curve uses the fast
        ``mode``-dependent (de Hoog / cosine) curve where applicable, else the exact per-point callable."""
        # the quantile function lives on the probability axis q in (0, 1); the pdf/cdf on the value axis t
        grid_key = 'q' if stat == 'quantile' else 't'

        # use cached values if available
        if hasattr(ms, '_cache') and stat in ms._cache:
            t = ms._cache[grid_key]
            y_ms = np.asarray(ms._cache[stat])
        elif stat == 'quantile':
            t = np.linspace(0.05, 0.95, 50)
            y_ms = np.asarray(ms_stat(t))
        else:
            # grid the distribution being compared over its own support (identical to the tree height for
            # tree_height, but wider for e.g. total_branch_length, whose accumulated reward exceeds it). For an
            # SFS the quantile is per-bin, so take the widest bin's support.
            t = np.linspace(0, float(np.max(ph.quantile(0.99))), 100)
            y_ms = np.asarray(ms_stat(t))

        curve = 'cdf' if stat == 'cdf' else 'pdf'
        method = self._curve_method(mode)  # de_hoog/None -> 'dehoog', cosine -> 'cos' (passed per call)
        if stat == 'quantile':
            # invert the cached CDF curve rather than the per-point de Hoog bisection (~2 s per point for
            # accumulated rewards such as total_branch_length)
            y_ph = self._quantile_values(ph, t, n_bins=y_ms.shape[1] if y_ms.ndim == 2 else None, method=method)
        elif mode is not None and hasattr(ph, 'bin'):
            # a moded spectrum pdf/cdf uses each bin's de_hoog/cosine *curve* (the per-point callable is
            # mode-independent); the monomorphic edge bins are zero placeholders, dropped below
            nb = y_ms.shape[0] if (y_ms.ndim == 2 and y_ms.shape[1] == len(t)) else y_ms.shape[1]
            y_ph = np.array([np.zeros(len(t)) if b in (0, nb - 1)
                             else np.asarray(getattr(ph.bin(b), curve).curve(t, method=method), dtype=float)
                             for b in range(nb)])
        elif mode is not None and hasattr(ph, '_reward_distribution') \
                and hasattr(getattr(ph._reward_distribution, curve), 'curve'):
            # a moded scalar reward distribution (e.g. total_branch_length) uses its mode-dependent curve
            y_ph = np.asarray(getattr(ph._reward_distribution, curve).curve(t, method=method), dtype=float)
        else:
            y_ph = np.asarray(ph_stat(t))  # exact per-point (mode is None, e.g. the expm tree height)

        # per-bin distributions (the SFS) are 2-D; orient both as (n_bins, len(grid)) and keep only the
        # polymorphic bins (the monomorphic edges are a degenerate atom at 0)
        per_bin = y_ph.ndim == 2 or y_ms.ndim == 2
        if per_bin:
            if y_ph.ndim == 2 and y_ph.shape[-1] != len(t):
                y_ph = y_ph.T
            if y_ms.ndim == 2 and y_ms.shape[-1] != len(t):
                y_ms = y_ms.T
            y_ph, y_ms = y_ph[1:-1], y_ms[1:-1]

        # Metric: the CDF (bounded in [0,1]) uses the worst *absolute* difference (its first two points -- the
        # near-zero head / per-bin atom at 0 -- are discarded, where the difference is unstable); the pdf the
        # total-variation distance between the densities (:meth:`_pdf_diff`); the quantile the relative Wasserstein-1
        # distance (:meth:`_quantile_diff`, atom-robust without dropping points).
        if stat == 'pdf':
            ms_p = y_ms[:, 2:] if per_bin else y_ms
            ph_p = y_ph[:, 2:] if per_bin else y_ph
            diff = self._pdf_diff(ms_p, ph_p, t[2:] if per_bin else t)
        elif stat == 'cdf':
            d = (y_ms - y_ph)[:, 2:] if per_bin else (y_ms - y_ph)[2:]
            diff = float(np.abs(d).max())
        else:  # quantile
            diff = self._quantile_diff(y_ms, y_ph, t)

        plot = None
        if self.visualize:
            def plot(msg, t=t, y_ph=y_ph, y_ms=y_ms, per_bin=per_bin) -> None:
                xlabel = 'q' if stat == 'quantile' else 'time'
                if per_bin:
                    # drop the first grid point: an SFS bin's atom at 0 spikes the empirical pdf / jumps the cdf
                    tp, yph_p, yms_p = t[1:], y_ph[:, 1:], y_ms[:, 1:]
                    series = [(yph_p[k], yms_p[k], self._pointwise_diff(stat, yph_p[k], yms_p[k]), f'bin {k + 1}')
                              for k in range(yph_p.shape[0])]
                else:
                    tp = t
                    series = [(y_ph, y_ms, self._pointwise_diff(stat, y_ph, y_ms), '')]
                self._plot_curves_with_diff(tp, series, xlabel, msg if self.show_title else None, name)

        return diff, plot

    #: Standard thinned grid size per axis for the slow per-point **de Hoog 2D surface** comparison (≈ every other
    #: node of the 25-point empirical grid). Fixed, not config-exposed: the empirical stays cached on the full grid and
    #: is subsampled to the same nodes, so de Hoog surfaces cost ~(13/25)^2 of the full grid with no second cache.
    DE_HOOG_2D_GRID: int = 13

    def _de_hoog_thin(self, n_full: int) -> np.ndarray:
        """Indices of an evenly-spaced thinned subset of a length-``n_full`` axis for the de Hoog 2D surface (see
        :attr:`DE_HOOG_2D_GRID`); all indices if the standard size is not smaller."""
        n = self.DE_HOOG_2D_GRID
        if n >= n_full:
            return np.arange(n_full)
        return np.unique(np.linspace(0, n_full - 1, n).round().astype(int))

    @staticmethod
    def _diff_label(stat: str) -> str:
        """Human-readable name of the difference metric used for a statistic (shown in the comparison log): the CDF
        uses the worst *absolute* difference; the pdf and the mutation configurations use the *total-variation
        distance* between the two distributions (``0.5 * integral|f_ref - f|`` for a density, ``0.5 * sum|p - q|`` for
        the discrete configs); the quantile uses the *relative Wasserstein-1* distance (the mean-normalised area
        between the quantile curves); the remaining scalars (mean/var/cov/corr, ...) use a worst *relative* difference."""
        return {'cdf': 'max abs', 'pairwise_cdf': 'max abs', 'loci_pairwise_cdf': 'max abs',
                'pdf': 'total variation', 'pairwise_pdf': 'total variation', 'loci_pairwise_pdf': 'total variation',
                'mutation_configs': 'total variation', 'quantile': 'rel. Wasserstein'}.get(stat, 'max rel')

    @staticmethod
    def _pdf_diff(y_ref, y_ph, *axes) -> float:
        """Total-variation distance between two densities: ``0.5 * integral|f_ref - f|`` -- the proper distributional
        distance (the continuous analogue of the :meth:`_diff_and_plot_mutation_configs` TV; in ``[0, 1]`` for
        probability densities and support-width-independent, since a density integrates to its dimensionless mass).
        The integral is a trapezoidal rule over the coordinate ``axes``: one axis for a 1-D curve or a per-bin
        spectrum (a leading bin axis, one trailing value axis -- the *worst bin's* TV is returned, matching the CDF's
        worst-over-bins metric), two axes for a 2-D surface (both integrated)."""
        def integ(d: np.ndarray, x: np.ndarray) -> np.ndarray:
            """Trapezoidal integral of ``d`` over its last axis with coordinates ``x``."""
            x = np.asarray(x, dtype=float)
            return 0.5 * np.sum((d[..., 1:] + d[..., :-1]) * np.diff(x), axis=-1)

        d = np.abs(np.asarray(y_ref, dtype=float) - np.asarray(y_ph, dtype=float))
        if len(axes) == 2:
            xs, ys = axes
            return float(0.5 * integ(integ(d, ys), xs))  # integrate the trailing axis, then the leading one
        return float(0.5 * np.max(integ(d, axes[0])))  # 1-D: per bin if 2-D, then the worst bin

    @staticmethod
    def _quantile_diff(y_ms, y_ph, q) -> float:
        """Relative Wasserstein-1 (earth-mover) distance between an empirical and analytic quantile curve over the
        probability grid ``q``: ``integral|Q_ph - Q_ms| dq / integral Q_ms dq``. The L1 distance between the quantile
        functions is a proper distributional distance (it equals the area between the CDFs); normalising by the
        reference mean (``integral Q dq = E[L]``) makes it dimensionless and transferable across scenarios.

        It is naturally **atom-robust**: for an SFS bin with an atom ``P(L_i = 0) = p0`` the inverse CDF is exactly 0
        for every probability below ``p0``, so on that flat region both quantiles are 0 and the integrand contributes
        nothing -- there is no per-point relative blow-up of the tiny near-atom values that the old worst-relative
        metric suffered from. For a per-bin spectrum the worst bin's value is returned; a fully degenerate (``Q ~ 0``)
        bin is 0."""
        y_ms, y_ph, q = np.asarray(y_ms, dtype=float), np.asarray(y_ph, dtype=float), np.asarray(q, dtype=float)

        def integ(d: np.ndarray) -> np.ndarray:
            """Trapezoidal integral over the last (probability) axis."""
            return 0.5 * np.sum((d[..., 1:] + d[..., :-1]) * np.diff(q), axis=-1)

        num, den = integ(np.abs(y_ph - y_ms)), integ(np.abs(y_ms))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(den > 1e-300, num / den, 0.0)  # a fully degenerate (Q ~ 0) bin contributes nothing
        return float(np.max(rel))

    def _result_message(self, title: str, diff: float, tol: float, label: str, runtime: float) -> str:
        """Assign this comparison the next sequential index and format the one-line result message used *identically*
        as the log line and the plot title: ``#i <title>: <diff> <=|> <tol> (<metric>, <runtime>s)``."""
        self._comp_index = getattr(self, '_comp_index', 0) + 1
        op = '<=' if diff <= tol else '>'
        return f"#{self._comp_index} {title}: {diff:.5f} {op} {tol} ({label}, {runtime:.3f}s)"

    def _log_result(self, msg: str, diff: float, tol: float) -> None:
        """Log a comparison result (critical if it exceeds the tolerance, info otherwise); under ``do_assertion``
        raise on failure and count the assertion."""
        if not diff <= tol:
            self.logger.critical(msg)
            if self.do_assertion:
                raise AssertionError(msg)
        else:
            self.logger.info(msg)
        if self.do_assertion:
            self.n_assertions += 1

    @staticmethod
    def _method_2d(mode: str, default: str) -> str:
        """Map a comparison inversion ``mode`` to the joint (2D) ``method`` keyword (``'dehoog'`` / ``'cos'``) passed to
        ``jd.pdf`` / ``jd.cdf``: ``de_hoog`` -> ``'dehoog'`` (nested de Hoog), ``cosine`` -> ``'cos'`` (cosine
        expansion); ``None`` keeps the call site's ``default``."""
        return {'de_hoog': 'dehoog', 'cosine': 'cos'}.get(mode, default)

    @staticmethod
    def _curve_method(mode: str) -> str:
        """Map a comparison inversion mode to the 1D curve method passed per call to ``cdf_curve`` / ``pdf_curve`` /
        ``quantile``: ``cosine`` -> ``'cos'``; ``de_hoog`` / ``None`` -> ``'dehoog'`` (the accurate default)."""
        return 'cos' if mode == 'cosine' else 'dehoog'

    @staticmethod
    def _parse_collection_key(k: str) -> list | None:
        """Parse a quoted collection key (``"[...]"`` / ``"{...}"``) into its list of elements, or ``None`` if ``k`` is
        not a collection literal. Beyond the ``ast.literal_eval``-able forms (``"[1, 3, 9]"``, ``"[(1, 3), (2, 3)]"``)
        this also accepts **bare-identifier** elements (``"[cosine, de_hoog]"``, broadcasting a sub-spec over both
        inversion modes), which ``ast.literal_eval`` rejects -- those are split on top-level commas and kept as strings.
        """
        s = k.strip()
        if s[:1] not in ('[', '{') or s[-1:] not in (']', '}'):
            return None
        try:
            parsed = ast.literal_eval(s)
            return list(parsed) if isinstance(parsed, (list, set)) else None
        except (ValueError, SyntaxError):
            pass

        # bare-identifier collection (e.g. mode names): split the body on commas at bracket depth 0
        elems, depth, start = [], 0, 0
        body = s[1:-1]
        for idx, ch in enumerate(body):
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            elif ch == ',' and depth == 0:
                elems.append(body[start:idx])
                start = idx + 1
        elems.append(body[start:])

        out = []
        for part in elems:
            part = part.strip()
            if not part:
                return None
            try:
                out.append(ast.literal_eval(part))
            except (ValueError, SyntaxError):
                out.append(part)  # bare string element (e.g. an inversion-mode name)
        return out

    @staticmethod
    def _expand_keys(data: dict) -> dict:
        """
        Normalise a (possibly terse) comparison-tolerance subtree by expanding **collection keys** that broadcast their
        sub-spec over several elements -- e.g. (note: YAML cannot use a bare ``[...]``/``{...}`` as a key, so quote it)::

            "[1, 3, 9]": {pdf: 0.01}          ->  1: {pdf: 0.01}, 3: {pdf: 0.01}, 9: {pdf: 0.01}
            "[(1, 2), (1, 9)]": {cdf: 0.02}   ->  "(1, 2)": {cdf: 0.02}, "(1, 9)": {cdf: 0.02}

        A quoted key that ``ast.literal_eval``s to a **list or set** is expanded over its elements (an ``int`` becomes a
        bin key, a ``tuple`` becomes an ``"(i, j)"`` pair-string key); a bare ``tuple`` (``"(1, 2)"``) stays a single
        pair. Broadcasting is a deep copy, and an already-present target is merged into (later wins on conflicts).
        Applied recursively, leaving non-collection keys untouched.
        """
        out = {}

        def _put(key, value) -> None:
            value = Comparison._expand_keys(value) if isinstance(value, dict) else value
            if key in out and isinstance(out[key], dict) and isinstance(value, dict):
                out[key] = {**out[key], **value}
            else:
                out[key] = copy.deepcopy(value)

        for k, v in data.items():
            parsed = Comparison._parse_collection_key(k) if isinstance(k, str) else None
            if parsed is not None:
                for elem in parsed:
                    _put(f"({elem[0]}, {elem[1]})" if isinstance(elem, tuple) else elem, v)
            else:
                _put(k, v)

        return out

    def _compare_stat_recursively(
            self,
            ph: PhaseTypeDistribution | MarginalDistributions,
            ms: PhaseTypeDistribution | MarginalDistributions,
            data: dict,
            title: str = 'stat',
            name: str = '',
            mode: str = None
    ) -> None:
        """
        Compare the given statistics recursively.

        :param ph: Phase-type distribution.
        :param ms: Phase-type distribution.
        :param data: Dictionary of statistics to compare, possibly nested.
        :param title: Title prefix for the plot.
        :param name: Name prefix for the plot.
        """

        # statistic, distribution or nested demes dictionary
        stat: Literal['pdf', 'cdf', 'pairwise_cdf', 'mean', 'var', 'std', 'cov', 'corr', 'demes', 'loci', 'm3', 'm4']

        # tolerance or dictionary of statistics
        sub: float | dict

        for stat, sub in data.items():

            # an explicit inversion-mode wrapper: route the nested stats through de Hoog or the cosine expansion
            # (``de_hoog`` -> nested inversion / mode='dehoog'; ``cosine`` -> the fast cosine path / mode='cos'). When
            # absent (``mode is None``) the original per-statistic default is used, so existing configs are unchanged.
            if stat in ('de_hoog', 'cosine'):
                self._compare_stat_recursively(ph=ph, ms=ms, data=sub, title=f"{title}: {stat}",
                                               name=f"{name}_{stat}", mode=stat)

            # if the statistic is nested, recurse
            elif isinstance(ph, MarginalDistributions) and not hasattr(ph, stat):
                if isinstance(ph, MarginalDemeDistributions):
                    items = self.ph.demography.pop_names
                elif isinstance(ph, MarginalLocusDistributions):
                    items = range(self.n_loci)
                else:
                    raise ValueError(f"Unknown type {type(ph)} for marginal distributions.")

                # iterate over demes or loci
                for item in items:
                    self.compare_stat(
                        ph=ph[item],
                        ms=ms[item],
                        stat=stat,
                        tol=sub,
                        title=f"{title}: {item}",
                        name=f"{name}_{item}",
                        mode=mode
                    )

            elif stat in ['demes', 'loci']:

                # a cross-locus 'pairwise' joint group (loci only) compares the joint distribution across the two loci
                # using the *parent* distributions (which carry the cached joint and the per-locus joint builder), not
                # the per-locus marginal container; any remaining keys (mean/var/cov/corr) recurse as usual.
                rest = sub
                if stat == 'loci' and isinstance(sub, dict) and 'pairwise' in sub:
                    rest = {k: v for k, v in sub.items() if k != 'pairwise'}
                    self._compare_loci_pairwise(ph=ph, ms=ms, sub=sub['pairwise'],
                                                title=f"{title}: loci", name=f"{name}_loci", mode=mode)

                if rest:
                    self._compare_stat_recursively(
                        ph=getattr(ph, stat),
                        ms=getattr(ms, stat),
                        data=rest,
                        title=f"{title}: {stat}",
                        name=f"{name}_{stat}",
                        mode=mode
                    )

            elif stat == 'pairwise':

                # nested pairwise group. A pair key like '(1, 2)' carries {cdf, pdf} tolerances for the full-grid
                # surface comparison of that single bin pair (each optionally wrapped in a de_hoog/cosine mode).
                for key, subtol in sub.items():
                    if key in ('de_hoog', 'cosine'):
                        self._compare_stat_recursively(ph=ph, ms=ms, data={'pairwise': subtol},
                                                       title=f"{title}: {key}", name=f"{name}_{key}", mode=key)
                    else:
                        pair = ast.literal_eval(key) if isinstance(key, str) else tuple(key)
                        self._compare_pairwise_surface(ph=ph, ms=ms, pair=pair, tols=subtol, title=title, name=name,
                                                       mode=mode)

            elif isinstance(stat, int) or (isinstance(stat, str) and stat.lstrip('-').isdigit()):

                # per-bin SFS targeting: ``sfs: {i}: {stat}`` compares only spectrum bin ``i`` (its mean/var and its
                # 1D pdf/cdf/quantile), rather than the spectrum-wide statistic
                self._compare_sfs_bin(ph=ph, ms=ms, i=int(stat), tols=sub, title=title, name=name, mode=mode)

            else:

                self.compare_stat(
                    ph=ph,
                    ms=ms,
                    stat=stat,
                    tol=sub,
                    title=title,
                    name=name,
                    mode=mode
                )

    def _compare_loci_pairwise(self, ph, ms, sub: dict, title: str, name: str, mode: str = None) -> None:
        """
        Compare the cross-locus joint distribution (the per-locus tree height / total branch length at the two loci,
        separated by recombination) against the msprime ground truth, as a **full-grid surface** over the single locus
        pair ``(0, 1)`` -- the same machinery as the SFS/jSFS/two-locus surfaces (:meth:`_compare_pairwise_surface`),
        routed through ``ph.loci.joint_distribution`` and the cached ``ms._loci_joint_surface``. A ``de_hoog`` /
        ``cosine`` key routes the 2D inversion (``mode``); the ``cdf`` / ``pdf`` tolerances are asserted over the grid.
        """
        for key, subtol in sub.items():
            if key in ('de_hoog', 'cosine'):
                self._compare_loci_pairwise(ph=ph, ms=ms, sub=subtol, title=f"{title}: {key}",
                                            name=f"{name}_{key}", mode=key)
        tols = {k: v for k, v in sub.items() if k in ('cdf', 'pdf')}
        if tols:
            self._compare_pairwise_surface(ph=ph, ms=ms, pair=(0, 1), tols=tols, title=title, name=name, mode=mode,
                                           joint_fn=lambda a, b: ph.loci.joint_distribution(a, b),
                                           surface_attr='_loci_joint_surface', stat_label='loci_pairwise')

    def _compare_sfs_bin(self, ph, ms, i: int, tols: dict, title: str, name: str, mode: str = None) -> None:
        """
        Compare a single SFS bin's statistics (config ``sfs: {i}: {stat}``) against the msprime ground truth: the
        scalar ``mean`` / ``var`` of bin ``i``, and its 1D ``pdf`` / ``cdf`` / ``quantile`` (bin ``i``'s reward
        distribution vs the cached empirical per-bin curves). The per-statistic metric matches the spectrum-wide
        comparison: the CDF uses the worst absolute difference, the pdf the mean absolute difference, and the
        quantile / mean / var a relative difference (the near-zero head is dropped for the curves, as elsewhere).
        A ``de_hoog`` / ``cosine`` key under the bin routes its sub-stats through that inversion (``mode``).
        """
        for stat, tol in tols.items():
            # an inversion-mode wrapper (``sfs: {i}: {de_hoog|cosine}: {stat}``) routes the bin's curves accordingly
            if stat in ('de_hoog', 'cosine'):
                self._compare_sfs_bin(ph=ph, ms=ms, i=i, tols=tol, title=f"{title}: {stat}", name=f"{name}_{stat}",
                                      mode=stat)
                continue
            t0 = time.perf_counter()
            sub_title = f"{title}: {i}: {stat}"

            if stat in ('mean', 'var', 'std'):
                ph_arr = getattr(ph, stat)
                ph_val = float(np.asarray(ph_arr.data if hasattr(ph_arr, 'data') else list(ph_arr)).ravel()[i])
                ms_val = float(np.asarray(list(getattr(ms, stat))).ravel()[i])
                diff = float(self.rel_diff(np.array([ms_val]), np.array([ph_val])).max())

            elif stat in ('pdf', 'cdf', 'quantile'):
                # the empirical per-bin curves were cached over a grid by ``touch``; orient to (n_bins, len(grid))
                grid_key = 'q' if stat == 'quantile' else 't'
                t = np.asarray(ms._cache[grid_key], dtype=float)
                y_ms_all = np.asarray(ms._cache[stat], dtype=float)
                if y_ms_all.ndim == 2 and y_ms_all.shape[-1] != len(t):
                    y_ms_all = y_ms_all.T
                y_ms = y_ms_all[i]
                d = ph.bin(i)  # only this bin's distribution (the spectrum-wide quantile would compute every bin)
                method = self._curve_method(mode)  # de_hoog/None -> 'dehoog', cosine -> 'cos' (passed per call)
                if stat == 'quantile':
                    y_ph = np.array([float(d.quantile(float(q), method=method)) for q in t])
                    diff = self._quantile_diff(y_ms, y_ph, t)
                else:
                    y_ph = np.asarray(d.cdf.curve(t, method=method) if stat == 'cdf'
                                      else d.pdf.curve(t, method=method), dtype=float)
                    diff = (float(np.abs(y_ms - y_ph)[2:].max()) if stat == 'cdf'
                            else self._pdf_diff(y_ms[2:], y_ph[2:], t[2:]))

            else:
                raise ValueError(f"Unsupported per-bin SFS statistic '{stat}' for bin {i} "
                                 f"(use mean / var / pdf / cdf / quantile).")

            runtime = time.perf_counter() - t0
            self.runtimes = getattr(self, 'runtimes', {})  # robust to deserialized objects that bypass __init__
            self.runtimes[sub_title] = runtime
            msg = self._result_message(sub_title, diff, tol, self._diff_label(stat), runtime)

            if self.visualize and stat in ('pdf', 'cdf', 'quantile'):
                # drop the first point: an atom-bearing bin's empirical pdf spikes there (the P(L=0) mass binned into
                # one narrow cell), which otherwise squashes the whole curve; the cdf/quantile lose only the t=0 edge
                sl = slice(1, None)
                series = [(y_ph[sl], y_ms[sl], self._pointwise_diff(stat, y_ph[sl], y_ms[sl]), '')]
                self._plot_curves_with_diff(t[sl], series, 'q' if stat == 'quantile' else 'time',
                                            msg if self.show_title else None, f"{name}_{i}_{stat}")

            self._log_result(msg, diff, tol)

    @staticmethod
    def _eval_statistic(coal, stat: str, args: list) -> float:
        """Evaluate a coalescent-level scalar statistic, calling it with ``args`` if it is a method (e.g. ``f2``)."""
        value = getattr(coal, stat)

        return value(*args) if callable(value) else value

    def _compare_scalar(self, ph: float, ms: float, tol: float, title: str) -> None:
        """Compare two scalar statistics within a relative tolerance, mirroring :meth:`compare_stat`."""
        diff = self.rel_diff(ms, ph)

        if not diff <= tol:
            self.logger.critical(f"{title}: {diff:.5f} > {tol}")

            if self.do_assertion:
                raise AssertionError(f"Relative difference {diff:.5f} exceeds threshold {tol} for {title}.")
        else:
            self.logger.info(f"{title}: {diff:.5f} <= {tol}")

        if self.do_assertion:
            self.n_assertions += 1

    @staticmethod
    def _quantile_values(ph, q, n_bins: int = None, method: str = 'dehoog') -> np.ndarray:
        """
        Quantile values of ``ph`` at probabilities ``q`` via its own quantile (which bisects the cached CDF curve --
        the de Hoog spline by default). An earlier version interpolated the inverse on a uniform grid over
        ``[0, mean + 12 std]``; for a heavily skewed reward (a time-inhomogeneous demography spanning 0 to many tens)
        that grid is far too coarse near the origin, giving large errors at small ``q`` -- so the quantile is now
        evaluated directly (the cached spline makes the bisection cheap). Returns a 1-D array for a scalar
        distribution, or ``(len(q), n_bins)`` for a spectrum (one column per bin; the monomorphic edge bins are held
        at 0).

        ``method='cos'`` (the comparison's cosine mode) instead inverts each bin's / the scalar reward's cosine CDF
        curve directly; ``'dehoog'`` (default) uses the distribution's own quantile (exact for the expm tree height).

        :param ph: The phase-type distribution (scalar, or a spectrum exposing :meth:`bin`).
        :param q: Probabilities at which to evaluate the quantile.
        :param n_bins: Number of spectrum bins (incl. the monomorphic edges); ``None`` for a scalar distribution.
        :param method: ``'dehoog'`` (default, the dist's own quantile) or ``'cos'`` (invert the cosine curve per bin).
        """
        q = np.asarray(q, dtype=float)

        if method != 'cos':
            # default (de Hoog) path: the distribution's own quantile. The spectrum's is vectorised over probabilities
            # and returns one column per bin (monomorphic edges held at 0); the scalar's is a per-probability callable
            if n_bins is not None:
                return np.asarray(ph.quantile(q), dtype=float)
            return np.array([float(ph.quantile(float(qq))) for qq in q])

        # cosine path: invert each bin's / the scalar reward's cosine CDF curve via the leaf RewardDistribution
        if n_bins is not None:
            cols = [np.zeros(len(q)) if b in (0, n_bins - 1)
                    else np.array([float(ph.bin(b).quantile(float(qq), method='cos')) for qq in q])
                    for b in range(n_bins)]
            return np.stack(cols, axis=1)
        rd = getattr(ph, '_reward_distribution', ph)
        return np.array([float(rd.quantile(float(qq), method='cos')) for qq in q])

    def _compare_pairwise_surface(self, ph, ms, pair: tuple, tols: dict, title: str, name: str, mode: str = None,
                                  joint_fn=None, surface_attr: str = '_joint_surface', stat_label: str = None) -> None:
        """
        Full-grid comparison of the within-tree joint distribution of one bin pair ``(i, j)``: the analytic
        ``joint_distribution(i, j)`` versus the cached empirical joint CDF / density over a 2D grid. For each of
        ``cdf`` and ``pdf`` requested in ``tols`` it asserts the worst element-wise difference over the grid and (when
        visualizing) draws three surfaces side by side -- phasegen, msprime and their element-wise difference. A
        ``de_hoog`` / ``cosine`` key under the pair routes the surfaces through that inversion (``mode='dehoog'/'cos'``).
        """
        # per-pair inversion-mode wrappers, e.g. '(1, 2): {de_hoog: {cdf, pdf}, cosine: {cdf, pdf}}'
        if any(k in ('de_hoog', 'cosine') for k in tols):
            for k, sub in tols.items():
                wrapped = k in ('de_hoog', 'cosine')
                self._compare_pairwise_surface(ph=ph, ms=ms, pair=pair, tols=sub if wrapped else {k: sub},
                                               title=f"{title}: {k}" if wrapped else title,
                                               name=f"{name}_{k}" if wrapped else name,
                                               mode=k if wrapped else mode, joint_fn=joint_fn,
                                               surface_attr=surface_attr, stat_label=stat_label)
            return

        i, j = pair
        entry = next((e for e in getattr(ms, surface_attr, []) if (e[0], e[1]) == (i, j)), None)
        if entry is None:
            raise ValueError(f"No cached empirical surface for pair {pair}; regenerate the comparison fixture.")
        _i, _j, xs, ys, cdf_ms, pdf_ms = entry
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        jd = joint_fn(i, j) if joint_fn is not None else ph.joint_distribution(i, j)

        # a degenerate bin -- one with (almost) no off-zero mass, e.g. a high-frequency class under an extreme
        # multiple-merger (a star-like genealogy) -- has a zero-width empirical support, so its CDF/density grid is
        # constant/non-finite and there is no continuous surface to compare; skip the pair (nothing to assert)
        if xs[-1] <= xs[0] or ys[-1] <= ys[0] or not np.isfinite(np.asarray(cdf_ms, dtype=float)).all():
            self.logger.info(f"{title}: pairwise {pair}: skipped (degenerate empirical surface)")
            return

        # skip the first two grid points on each axis: there the joint law has its atom edge (P=0 head for the cdf,
        # the empirical pdf's one-sided boundary difference), where phasegen and msprime disagree spuriously
        sx = sy = slice(2, None)

        for kind in ('cdf', 'pdf'):
            if kind not in tols:
                continue
            t0 = time.perf_counter()

            # the joint cdf/pdf on the whole grid; default (no mode) uses the fast cosine inversion -- a per-point
            # de Hoog grid is far slower -- and the atom-edge head where the cosine box is biased is dropped below
            # (the first two points per axis). A de_hoog/cosine wrapper overrides the inversion.
            m2d = self._method_2d(mode, 'cos')
            dehoog = m2d == 'dehoog'
            # the per-point de Hoog surface is slow (one nested inversion per grid node); evaluate it on a thinned
            # subset of the standard grid (the empirical is subsampled to the same nodes). The fast cosine path stays
            # on the full grid.
            gx, gy = (self._de_hoog_thin(len(xs)), self._de_hoog_thin(len(ys))) if dehoog \
                else (np.arange(len(xs)), np.arange(len(ys)))
            xs_d, ys_d = xs[gx], ys[gy]
            ms_grid = (cdf_ms if kind == 'cdf' else pdf_ms)
            grid_ms = np.asarray(ms_grid, dtype=float)[np.ix_(gx, gy)]
            grid_ph = np.asarray(jd.cdf(xs_d, ys_d, method=m2d) if kind == 'cdf'
                                 else jd.pdf(xs_d, ys_d, method=m2d), dtype=float)

            xs_p, ys_p = xs_d[sx], ys_d[sy]
            grid_ph, grid_ms = grid_ph[sx, sy], grid_ms[sx, sy]

            # the CDF (bounded in [0, 1]) uses the worst absolute element-wise difference; the density uses the
            # total-variation distance 0.5*integral|f_ref - f| over the 2-D grid (a proper, support-width-independent
            # distributional distance -- see ``_pdf_diff``)
            diff = (float(np.abs(grid_ph - grid_ms).max()) if kind == 'cdf'
                    else self._pdf_diff(grid_ms, grid_ph, xs_p, ys_p))
            # the loci (single-pair) surface logs as ``{stat_label}_{kind}`` (e.g. ``loci_pairwise_cdf``) so its config
            # tolerance leaf matches; the per-pair SFS/jSFS/2-locus surfaces carry the pair in the title
            label_key = f"{stat_label}_{kind}" if stat_label else f"pairwise_{kind}"
            sub_title = f"{title}: {label_key}" if stat_label else f"{title}: pairwise {pair} {kind}"
            runtime = time.perf_counter() - t0
            self.runtimes = getattr(self, 'runtimes', {})  # robust to deserialized objects that bypass __init__
            self.runtimes[sub_title] = runtime
            msg = self._result_message(sub_title, diff, tols[kind], self._diff_label(label_key), runtime)

            if self.visualize:
                # the difference surface is coloured blue at 0 up to red at the saturation level. For the CDF (bounded
                # in [0,1]) it shows the *absolute* difference (a probability-mass error, matching the assertion); the
                # density shows the per-point absolute difference normalised by the peak (mode) of the reference
                # density, which averages to the scalar metric asserted on (see ``_pdf_diff`` / ``_plot_surface_triple``).
                if kind == 'cdf':
                    diff_grid, dlabel, dzlabel = np.abs(grid_ms - grid_ph), 'absolute difference', 'abs. diff'
                else:
                    den = max(float(np.abs(grid_ms).max()), 1e-300)
                    diff_grid = np.abs(grid_ms - grid_ph) / den
                    dlabel, dzlabel = 'normalized abs. difference', 'norm. abs'
                self._plot_surface_triple(xs_p, ys_p, grid_ph, grid_ms, diff_grid, zlabel=kind.upper(),
                                          title=msg if self.show_title else None,
                                          name=f"{name}_pairwise_{i}_{j}_{kind}", diff_label=dlabel,
                                          diff_zlabel=dzlabel)

            self._log_result(msg, diff, tols[kind])

    #: Relative-difference level at which the difference surface's colormap saturates to red (blue at 0).
    surface_diff_saturation: float = 0.1

    #: Minimum vertical (z-axis) span of a difference-surface panel, so a tiny diff is not auto-zoomed into noise.
    min_diff_axis_height: float = 0.01

    #: Font sizes for comparison-plot subplot titles and figure suptitles (slightly above the matplotlib defaults).
    title_fontsize: int = 13
    suptitle_fontsize: int = 15

    def _pointwise_diff(self, stat: str, y_ph: np.ndarray, y_ms: np.ndarray) -> np.ndarray:
        """Per-point discrepancy curve for the difference panel of a plot (a dimensionless, plotting-only curve whose
        integral relates to the asserted metric): absolute for the CDF; for the pdf and the quantile the per-point
        absolute difference normalised by the mean reference (the density mean, resp. the reference mean ``E[L]``), so
        it has no per-point relative blow-up near the atom."""
        y_ph, y_ms = np.asarray(y_ph, float), np.asarray(y_ms, float)
        if stat == 'cdf':
            return np.abs(y_ph - y_ms)
        return np.abs(y_ph - y_ms) / max(float(np.abs(y_ms).mean()), 1e-300)

    def _plot_curves_with_diff(self, t, series, xlabel: str, title: str, name: str) -> None:
        """Two panels side by side: left overlays phasegen (solid) vs msprime (dashed) for each ``series`` entry
        ``(y_ph, y_ms, diff, label)``; right shows each per-point ``diff`` as a line **coloured by its magnitude**
        (the same ``coolwarm`` scale saturating at :attr:`surface_diff_saturation` as the surface diff plots, with a
        shared colorbar), the diff axis floored to that saturation so a tiny diff is not zoomed into noise."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        t = np.asarray(t, float)
        sat = self.surface_diff_saturation
        fig, (axc, axd) = plt.subplots(ncols=2, figsize=(13, 5))
        norm = plt.Normalize(0.0, sat)
        ymax, lc = sat, None
        for y_ph, y_ms, diff, label in series:
            line, = axc.plot(t, y_ph, linewidth=1.5, alpha=0.8, label=f'{label} (phasegen)' if label else 'phasegen')
            axc.plot(t, y_ms, '--', color=line.get_color(), linewidth=1.2, alpha=0.8,
                     label=f'{label} (msprime)' if label else 'msprime')
            d = np.asarray(diff, float)
            pts = np.array([t, d]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='coolwarm', norm=norm)
            lc.set_array(0.5 * (d[:-1] + d[1:]))  # colour each segment by its difference height
            lc.set_linewidth(1.6)
            axd.add_collection(lc)
            ymax = max(ymax, float(np.nanmax(d)))

        axc.set_xlabel(xlabel)
        axc.legend(fontsize=7 if len(series) > 1 else 10)
        axd.set_xlim(float(t.min()), float(t.max()))
        axd.set_ylim(0.0, ymax)  # floored to the saturation level (sat) unless the diff exceeds it
        axd.set_xlabel(xlabel)
        axd.set_ylabel('difference')
        axd.set_title('difference', fontsize=self.title_fontsize)
        if lc is not None:
            fig.colorbar(lc, ax=axd)
        if title and self.show_title:
            fig.suptitle(title, fontsize=self.suptitle_fontsize)
        self._save_and_show(name)

    def _plot_sfs_with_diff(self, ph_stat, ms_stat, title: str, name: str, left_title: str = 'SFS') -> None:
        """Two panels side by side: left the grouped SFS bar comparison (phasegen vs msprime via the ``Spectra``
        plotter), right the per-bin **relative difference** (the asserted ``max rel`` metric) as bars coloured by
        magnitude -- the same ``coolwarm`` scale saturating at :attr:`surface_diff_saturation` as the curve/surface
        diff panels, the axis floored to that saturation so a tiny diff is not zoomed into noise. Pure-zero bins (the
        monomorphic SFS edges) are dropped so the difference bars line up with the polymorphic bars on the left."""
        plt.close('all')  # avoid empty plots
        fig, (axs, axd) = plt.subplots(ncols=2, figsize=(13, 5))

        Spectra.from_spectra(dict(msprime=SFS(ms_stat), phasegen=SFS(ph_stat))).plot(ax=axs, show=False)
        axs.legend(fontsize=10)
        axs.set_title(left_title, fontsize=self.title_fontsize)

        ms_arr, ph_arr = np.asarray(ms_stat, float), np.asarray(ph_stat, float)
        diff = np.asarray(self.rel_diff(ms_arr, ph_arr), float)
        classes = np.arange(len(diff))
        poly = (np.abs(ms_arr) + np.abs(ph_arr)) > 0  # drop monomorphic edges (both spectra ~0 there)
        classes, diff = classes[poly], diff[poly]

        sat = self.surface_diff_saturation
        norm = plt.Normalize(0.0, sat)
        axd.bar(classes, diff, color=plt.cm.coolwarm(norm(diff)))
        if classes.size:
            axd.set_xticks(classes)
        axd.set_ylim(0.0, max(sat, float(np.nanmax(diff)) if diff.size else sat))
        axd.set_xlabel('frequency class')
        axd.set_ylabel('relative difference')
        axd.set_title('relative difference', fontsize=self.title_fontsize)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axd)

        if title and self.show_title:
            fig.suptitle(title, fontsize=self.suptitle_fontsize)
        self._save_and_show(name)

    def _plot_surface_triple(self, xs, ys, grid_ph, grid_ms, diff_grid, zlabel: str, title: str, name: str,
                             xlabel: str = 'L_i', ylabel: str = 'L_j', diff_label: str = 'relative difference',
                             diff_zlabel: str = 'rel. diff') -> None:
        """Draw phasegen / msprime / difference surfaces side by side over the ``xs x ys`` grid. The two distributions
        use a sequential colormap; the third is the element-wise difference (``diff_grid``: relative by default, or
        absolute for a CDF), coloured blue at 0 up to red at :attr:`surface_diff_saturation` (so it reads red wherever
        phasegen and msprime disagree by that much or more)."""
        plt.close('all')  # avoid empty plots
        # a taller figure: the 3D axes fill more of it, shrinking the whitespace margins between the three panels
        fig, axs = plt.subplots(ncols=3, subplot_kw={'projection': '3d'}, figsize=(13, 5.5))
        X, Y = np.meshgrid(xs, ys)
        sat = self.surface_diff_saturation

        for ax, grid, sub, cmap, zlab, lim in zip(
                axs, (grid_ph, grid_ms, diff_grid), ('phasegen', 'msprime', diff_label),
                ('viridis', 'viridis', 'coolwarm'), (zlabel, zlabel, diff_zlabel), (None, None, (0.0, sat))
        ):
            kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
            ax.plot_surface(X, Y, np.asarray(grid).T, cmap=cmap, **kw)
            ax.set_title(sub, fontsize=self.title_fontsize)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlab)

        # share one vertical scale across the phasegen and msprime panels (the max of the two) so they are directly
        # comparable rather than each auto-scaled to its own height
        zmin = min(float(np.nanmin(grid_ph)), float(np.nanmin(grid_ms)))
        zmax = max(float(np.nanmax(grid_ph)), float(np.nanmax(grid_ms)))
        if zmax > zmin:
            axs[0].set_zlim(zmin, zmax)
            axs[1].set_zlim(zmin, zmax)

        # floor the difference panel's vertical span to ``min_diff_axis_height`` so a tiny diff is not auto-zoomed up
        # into what looks like a large disagreement (the colour scale already saturates at ``surface_diff_saturation``)
        axs[2].set_zlim(0.0, max(self.min_diff_axis_height, float(np.nanmax(diff_grid))))

        if title:
            plt.suptitle(title, fontsize=self.suptitle_fontsize)

        self._save_and_show(name, pad=2.8, extra_right=1.2)

    def _pairwise_surface_pairs(self) -> dict:
        """The per-distribution bin pairs that request a full-grid pairwise surface comparison (the non-``cdf``/``pdf``
        keys under a ``pairwise`` group), parsed from the comparison config -- used to cache their empirical grids."""
        out = {}
        for dist, data in self._expand_keys(self.comparisons.get('tolerance', {})).items():
            pairwise = data.get('pairwise') if isinstance(data, dict) else None
            if not isinstance(pairwise, dict):
                continue

            # pair keys are everything that is not an aggregate stat ('cdf'/'pdf'); they may sit directly under
            # ``pairwise`` or be nested under a de_hoog/cosine mode wrapper, so descend into those
            pairs = []

            def _collect(d) -> None:
                for k, v in d.items():
                    if k in ('cdf', 'pdf'):
                        continue
                    if k in ('de_hoog', 'cosine') and isinstance(v, dict):
                        _collect(v)
                    else:
                        pairs.append(ast.literal_eval(k) if isinstance(k, str) else tuple(k))

            _collect(pairwise)
            if pairs:
                out[dist] = list(dict.fromkeys(pairs))  # de-dupe, preserve order
        return out

    def cache_ground_truth(self) -> None:
        """Cache the msprime ground truth needed by the configured comparisons: the standard per-statistic caches
        (:meth:`MsprimeCoalescent.touch`) plus any full-grid pairwise surface grids the config requests. Call before
        :meth:`MsprimeCoalescent.drop` so the grids are serialized with the comparison."""
        self.ms.touch()
        for dist, pairs in self._pairwise_surface_pairs().items():
            getattr(self.ms, dist).cache_joint_surface(pairs)

    def compare(self, title: str = '') -> None:
        """
        Compare the distributions of the given statistics.

        :param title: Title prefix for the plots.
        :raises AssertionError: If `do_assertion is True and the distributions differ by more than the given tolerance.
            ValueError: if the type is unknown.
        """
        # enlarge titles globally so plots delegated to the distribution/spectrum ``.plot()`` methods (which set their
        # own titles at the matplotlib default) match the comparison's own explicitly-sized titles/suptitles
        plt.rcParams['axes.titlesize'] = self.title_fontsize
        plt.rcParams['figure.titlesize'] = self.suptitle_fontsize
        self._comp_index = 0  # sequential comparison counter, prepended as '#i' to each result message / plot title

        for dist, data in self._expand_keys(self.comparisons['tolerance']).items():
            self._compare_stat_recursively(
                ph=getattr(self.ph, dist),
                ms=getattr(self.ms, dist),
                data=data,
                title=f"{title}: {dist}",
                name=dist
            )

        # coalescent-level scalar statistics (optionally parameterized with population arguments), e.g. F_ST and
        # the Patterson f-statistics f2/f3/f4. Each entry is either ``<stat>: <tol>`` or
        # ``<stat>: {args: [...], tol: <tol>}``.
        for stat, spec in self.comparisons.get('statistics', {}).items():
            args = spec.get('args', []) if isinstance(spec, dict) else []
            tol = spec['tol'] if isinstance(spec, dict) else spec
            label = f"{title}: {stat}" + (f"({', '.join(map(str, args))})" if args else "")

            self._compare_scalar(
                ph=self._eval_statistic(self.ph, stat, args),
                ms=self._eval_statistic(self.ms, stat, args),
                tol=tol,
                title=label
            )

        self.logger.info(f"Number of assertions: {self.n_assertions}")
