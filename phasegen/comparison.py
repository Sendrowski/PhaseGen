"""
Compare statistics between PhaseGen and Msprime.
"""
import ast
import copy
import logging
import os
import time
from .caching import cached_property
from typing import Iterable, Dict, Literal, List

import numpy as np
import yaml
from sfsutils import Spectra
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

    # Number of sampler trajectories for the ``ms`` operand; None falls back to msprime. Declared at class level
    # so fixtures serialized before the trajectory sampler was added deserialize without this attribute set.
    n_samples: int = None

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

        #: Ground truth of the configured coalescent-level scalar statistics, keyed by ``(name, args)``
        #: (:meth:`cache_ground_truth`), so that it survives the drop of the simulated data it is computed from.
        self._ms_statistics: dict = {}

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
    def empirical(self) -> 'SampledCoalescent':
        """
        The self-consistency candidate operand: PhaseGen's own trajectory sampler (``n_samples`` draws), validated
        against the exact analytic :attr:`ph` rather than an external tool. Drives the nested ``tolerance.empirical``
        sub-spec, a different kind of check than :attr:`ms`.
        """
        # a fresh analytic coalescent (not self.ph, which must stay out of the serialized fixture); it is dropped
        # before serialization
        return SampledCoalescent(
            coalescent=self._make_coalescent(),
            n_samples=self.n_samples,
            seed=self.seed
        )

    @cached_property
    def ms(self) -> 'MsprimeCoalescent':
        """
        The external ground-truth candidate operand: an independent msprime simulation. Drives the top-level
        ``tolerance`` stats -- a falsification test of the exact analytic :attr:`ph` against a separate tool.
        """
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
                plt.close('all')  # avoid empty plots
                fig, (axs, axd) = plt.subplots(ncols=2, figsize=(13, 5))
                classes = np.arange(len(configs))
                labels = [str(config) for config in configs]

                # left: the two probability distributions over configurations
                axs.plot(ph_stat, label='phasegen')
                axs.plot(ms_stat, label='msprime')
                axs.set_xticks(classes)
                axs.set_xticklabels(labels, rotation=90)
                axs.legend(fontsize=10)
                axs.set_title('mutation configs', fontsize=self.title_fontsize)

                # right: per-config absolute difference (the total-variation summand; heights are probabilities, so
                # the scale is honest -- coloured by magnitude for emphasis)
                adiff = np.abs(ph_stat - ms_stat)
                norm = plt.Normalize(0.0, float(adiff.max()) or 1.0)
                axd.bar(classes, adiff, color=plt.cm.coolwarm(norm(adiff)))
                axd.set_xticks(classes)
                axd.set_xticklabels(labels, rotation=90)
                axd.set_ylabel('absolute difference')
                axd.set_title('absolute difference', fontsize=self.title_fontsize)
                sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=axd)

                if self.show_title:
                    fig.suptitle(msg, fontsize=self.suptitle_fontsize)
                self._save_and_show(name, pad=1.5)

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
                # phasegen / msprime / relative-difference surfaces for a 2-D joint or two-locus SFS (the joint SFS
                # may be rectangular, so index each axis by its own extent)
                def plot(msg, ph_stat=ph_stat, ms_stat=ms_stat) -> None:
                    # _plot_surface_triple transposes the grid, so index x by the first axis and y by the second
                    # (this keeps a rectangular joint SFS, where the two axes differ in length, from mismatching)
                    xs = np.arange(ph_stat.shape[0])
                    ys = np.arange(ph_stat.shape[1])
                    xlabel, ylabel = ('allele count pop_0', 'allele count pop_1') if is_joint else ('L_i', 'L_j')
                    self._plot_surface_triple(
                        xs, ys, ph_stat, ms_stat, self.rel_diff(ms_stat, ph_stat), zlabel=stat,
                        xlabel=xlabel, ylabel=ylabel, title=msg if self.show_title else None, name=name)

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

        # the cdf is read pointwise; the pdf is averaged over each cell of the grid, because that is the functional
        # the empirical density estimates (see :meth:`_cell_average`)
        evaluate = ((lambda f: self._cell_average(f, t)) if stat == 'pdf'
                    else (lambda f: np.asarray(f(t), dtype=float)))

        if stat == 'quantile':
            y_ph = self._quantile_values(ph, t, n_bins=y_ms.shape[1] if y_ms.ndim == 2 else None, mode=mode)
        elif mode is not None and hasattr(ph, 'bin'):
            # a moded spectrum pdf/cdf compares each bin's *inverted* curve; the monomorphic edge bins are zero
            # placeholders, dropped below
            nb = y_ms.shape[0] if (y_ms.ndim == 2 and y_ms.shape[1] == len(t)) else y_ms.shape[1]
            y_ph = np.array([np.zeros(len(t)) if b in (0, nb - 1) else evaluate(getattr(ph.bin(b), curve))
                             for b in range(nb)])
        elif mode is not None and hasattr(ph, '_reward_distribution'):
            # a moded scalar reward distribution (e.g. total_branch_length) compares its inverted curve
            y_ph = evaluate(getattr(ph._reward_distribution, curve))
        else:
            y_ph = evaluate(ph_stat)  # exact (mode is None, e.g. the expm tree height)

        # per-bin distributions (the SFS) are 2-D; orient both as (n_bins, len(grid)) and keep only the
        # polymorphic bins (the monomorphic edges are a degenerate atom at 0)
        per_bin = y_ph.ndim == 2 or y_ms.ndim == 2
        if per_bin:
            if y_ph.ndim == 2 and y_ph.shape[-1] != len(t):
                y_ph = y_ph.T
            if y_ms.ndim == 2 and y_ms.shape[-1] != len(t):
                y_ms = y_ms.T
            y_ph, y_ms = y_ph[1:-1], y_ms[1:-1]

        # Metric: the CDF (bounded in [0,1]) uses the worst *absolute* difference over the *whole* grid, including
        # the point at 0 -- so the atom ``P(R = 0)`` of an SFS bin is asserted rather than skipped. It is well defined
        # on both sides (analytically ``phi(inf)``, empirically the fraction of zero replicates) and the cosine
        # inversion splits it off instead of trying to resolve the jump. The pdf spans the whole grid too, the head
        # included: both sides are cell averages of the *continuous* sub-density (the atom excluded), so the cell at
        # the origin is finite and well posed on both -- unlike a pointwise empirical density there, which is a delta
        # spike and had to be dropped. The quantile uses the relative Wasserstein-1 distance (:meth:`_quantile_diff`,
        # atom-robust without dropping points).
        if stat == 'pdf':
            diff = self._pdf_diff(y_ms, y_ph, t)
        elif stat == 'cdf':
            diff = float(np.abs(y_ms - y_ph).max())
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

    #: Gauss-Legendre nodes per cell used to integrate an exact density over a comparison cell. Enough for a smooth
    #: density, and evaluated for every cell in a single vectorised call.
    _CELL_QUAD_NODES = 8

    #: Nodes beyond which a cell's integral is accepted as it stands. A near-atom (an epoch boundary that collapses the
    #: population size, say) puts a spike orders of magnitude narrower than a cell inside it, and no fixed-order rule
    #: integrates that: the estimate has to be refined until it stops moving, or the comparison reports a density error
    #: that is the quadrature's, not phasegen's.
    _CELL_QUAD_MAX_NODES = 512

    #: Change in a cell's *mass* (its density times its width) between successive refinements below which the cell is
    #: converged. Tied to the mass rather than the density because the mass is what the total-variation metric
    #: integrates: a cell out in the tail may hold a wildly uncertain relative density and still be irrelevant to it.
    _CELL_QUAD_TOL = 1e-4

    @classmethod
    def _quadrature(cls, f, lo: np.ndarray, hi: np.ndarray, n_nodes: int) -> np.ndarray:
        """Gauss-Legendre average of ``f`` over each cell ``[lo, hi)``, all cells in one vectorised call.

        :param f: The density, a vectorised callable (1-D, or per-bin returning ``(n_bins, len(x))``).
        :param lo: Lower cell edges.
        :param hi: Upper cell edges.
        :param n_nodes: Nodes per cell.
        :return: The cell averages, shaped like ``f``'s output.
        """
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        nodes = 0.5 * (hi - lo)[:, None] * (x[None, :] + 1.0) + lo[:, None]

        y = np.asarray(f(nodes.ravel()), dtype=float)
        y = y.reshape(*y.shape[:-1], len(lo), n_nodes)

        # the 0.5 * (hi - lo) Jacobian of the quadrature cancels the 1 / (hi - lo) of the average
        return 0.5 * (y * w).sum(axis=-1)

    @classmethod
    def _cell_average(cls, f, t: np.ndarray) -> np.ndarray:
        """
        The exact density ``f`` averaged over each cell of the grid ``t`` -- the same functional the empirical density
        estimates (:class:`~phasegen.distributions.empirical._EmpiricalDensityFunction`), so that a pdf comparison
        pits like against like.

        Comparing a sample's cell average against a *pointwise* exact density instead imposes an ``O(h f')``
        discrepancy that is no part of phasegen's error and that more replicates do not remove. It dominates wherever
        the density turns sharply within a cell, which for an SFS bin is exactly the origin.

        Cells whose integral is still moving are refined until it settles, and only those: a spike much narrower than
        its cell defeats a fixed-order rule, and the resulting error lands in the comparison as if it were phasegen's.

        :param f: The exact density, a vectorised callable (1-D, or per-bin returning ``(n_bins, len(x))``).
        :param t: The grid whose cells to average over; the last cell is extended by the final spacing.
        :return: The cell averages, shaped like ``f``'s output.
        """
        edges = np.append(t, 2 * t[-1] - t[-2])
        lo, hi = edges[:-1], edges[1:]
        widths = hi - lo

        avg = cls._quadrature(f, lo, hi, cls._CELL_QUAD_NODES)

        # half the nodes is not the answer but the error estimate: where the two rules agree the density is resolved,
        # and where they do not the cell holds a feature the rule cannot see, which only refinement settles
        probe = cls._quadrature(f, lo, hi, cls._CELL_QUAD_NODES // 2)
        cells = cls._unconverged(np.abs(avg - probe) * widths > cls._CELL_QUAD_TOL, np.arange(len(lo)))

        n_nodes = cls._CELL_QUAD_NODES
        while cells.size and n_nodes < cls._CELL_QUAD_MAX_NODES:
            n_nodes *= 4

            refined = cls._quadrature(f, lo[cells], hi[cells], n_nodes)
            moved = np.abs(refined - avg[..., cells]) * widths[cells] > cls._CELL_QUAD_TOL
            avg[..., cells] = refined

            cells = cls._unconverged(moved, cells)

        return avg

    @staticmethod
    def _unconverged(moved: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """The cells still to refine: those whose integral moved, in any bin of a per-bin density.

        :param moved: Whether the cell's integral moved, with the cell axis trailing.
        :param cells: The cells ``moved`` refers to.
        :return: The subset of ``cells`` to refine.
        """
        return cells[moved.any(axis=tuple(range(moved.ndim - 1))) if moved.ndim > 1 else moved]

    @staticmethod
    def _diff_label(stat: str) -> str:
        """Human-readable name of the difference metric used for a statistic (shown in the comparison log): the CDF
        uses the worst *absolute* difference; the pdf and the mutation configurations use the *total-variation
        distance* between the two distributions (``0.5 * integral|f_ref - f|`` for a density, ``0.5 * sum|p - q|`` for
        the discrete configs); the quantile uses the *relative Wasserstein-1* distance (the mean-normalised area
        between the quantile curves); the remaining scalars (mean/var/cov/corr, ...) use a worst *relative* difference."""
        return {'cdf': 'max abs', 'pairwise_cdf': 'max abs', 'loci_pairwise_cdf': 'max abs',
                'pdf': 'total variation', 'pairwise_pdf': 'total variation', 'loci_pairwise_pdf': 'total variation',
                'mutation_configs': 'total variation', 'quantile': 'rel. Wasserstein',
                'conditional_total_probability': 'max abs'}.get(stat, 'max rel')

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
                                                title=f"{title}: loci", name=f"{name}_loci")

                if rest:
                    self._compare_stat_recursively(
                        ph=getattr(ph, stat),
                        ms=getattr(ms, stat),
                        data=rest,
                        title=f"{title}: {stat}",
                        name=f"{name}_{stat}",
                        mode=mode
                    )

            elif stat == 'conditional':

                # nested conditional group: the self-consistency checks of the conditional path, on freely chosen bin
                # pairs. These are identities the analytic joint must satisfy, so they need no msprime operand and a
                # pair can be added without regenerating the fixture. The one exception is the ``atom`` sub-block,
                # which *is* compared against msprime (see :meth:`_compare_atom_conditional`) and does need the
                # cached ground truth.
                for key, subtol in sub.items():
                    pair = ast.literal_eval(key) if isinstance(key, str) else tuple(key)
                    self._compare_conditional(ph.joint_distribution(*pair), pair, subtol, title, name, ms=ms)

            elif stat == 'pairwise':

                # nested pairwise group. A pair key like '(1, 2)' carries {cdf, pdf} tolerances for the full-grid
                # surface comparison of that single bin pair.
                for key, subtol in sub.items():
                    pair = ast.literal_eval(key) if isinstance(key, str) else tuple(key)
                    self._compare_pairwise_surface(ph=ph, ms=ms, pair=pair, tols=subtol, title=title, name=name)

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

    def _compare_loci_pairwise(self, ph, ms, sub: dict, title: str, name: str) -> None:
        """
        Compare the cross-locus joint distribution (the per-locus tree height / total branch length at the two loci,
        separated by recombination) against the msprime ground truth, as a **full-grid surface** over the single locus
        pair ``(0, 1)`` -- the same machinery as the SFS/jSFS/two-locus surfaces (:meth:`_compare_pairwise_surface`),
        routed through ``ph.loci.joint_distribution`` and the cached ``ms._loci_joint_surface``. The ``cdf`` / ``pdf``
        tolerances are asserted over the grid.
        """
        tols = {k: v for k, v in sub.items() if k in ('cdf', 'pdf')}
        if tols:
            self._compare_pairwise_surface(ph=ph, ms=ms, pair=(0, 1), tols=tols, title=title, name=name,
                                           joint_fn=lambda a, b: ph.loci.joint_distribution(a, b),
                                           surface_attr='_loci_joint_surface', stat_label='loci_pairwise')

    def _compare_sfs_bin(self, ph, ms, i: int, tols: dict, title: str, name: str, mode: str = None) -> None:
        """
        Compare a single SFS bin's statistics (config ``sfs: {i}: {stat}``) against the msprime ground truth: the
        scalar ``mean`` / ``var`` of bin ``i``, and its 1D ``pdf`` / ``cdf`` / ``quantile`` (bin ``i``'s reward
        distribution vs the cached empirical per-bin curves). The per-statistic metric matches the spectrum-wide
        comparison: the CDF uses the worst absolute difference, the pdf the total variation between the cell-averaged
        densities, and the quantile / mean / var a relative difference.
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
                if stat == 'quantile':
                    y_ph = np.asarray(d.quantile(t), dtype=float)
                    diff = self._quantile_diff(y_ms, y_ph, t)
                else:
                    # the pdf is averaged over each cell, as the spectrum-wide comparison does: the empirical density
                    # is a cell average, and a pointwise exact density is a different functional (see _cell_average)
                    y_ph = (np.asarray(d.cdf(t), dtype=float) if stat == 'cdf'
                            else self._cell_average(d.pdf, t))
                    diff = (float(np.abs(y_ms - y_ph).max()) if stat == 'cdf'
                            else self._pdf_diff(y_ms, y_ph, t))

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
    def _quantile_values(ph, q, n_bins: int = None, mode: str = None) -> np.ndarray:
        """
        Quantile values of ``ph`` at probabilities ``q``, via its own (vectorised) quantile function. Returns a 1-D
        array for a scalar distribution, or ``(len(q), n_bins)`` for a spectrum (one column per bin; the monomorphic
        edge bins are held at 0).

        An earlier version interpolated the inverse on a uniform grid over ``[0, mean + 12 std]``; for a heavily
        skewed reward (a time-inhomogeneous demography spanning 0 to many tens) that grid is far too coarse near the
        origin, giving large errors at small ``q``.

        :param ph: The phase-type distribution (scalar, or a spectrum exposing :meth:`bin`).
        :param q: Probabilities at which to evaluate the quantile.
        :param n_bins: Number of spectrum bins (incl. the monomorphic edges); ``None`` for a scalar distribution.
        :param mode: ``None`` compares the distribution's own quantile (the matrix exponential, for the tree height);
            an inversion mode (``'cosine'``) compares the *inverted* accumulated-reward quantile instead.
        """
        q = np.asarray(q, dtype=float)

        if n_bins is not None:
            if mode is None:
                return np.asarray(ph.quantile(q), dtype=float)
            cols = [np.zeros(len(q)) if b in (0, n_bins - 1)
                    else np.asarray(ph.bin(b).quantile(q), dtype=float) for b in range(n_bins)]
            return np.stack(cols, axis=1)

        target = ph if mode is None else getattr(ph, '_reward_distribution', ph)
        return np.asarray(target.quantile(q), dtype=float)

    #: The conditional self-consistency checks a ``pairwise: {pair}: conditional:`` block may request, mapped to the
    #: :class:`~phasegen.distributions.reward.JointRewardDistribution` method that runs each. All are exact identities
    #: the conditional must satisfy, so each is asserted against the analytic joint alone -- no msprime, no fixture.
    _CONDITIONAL_CHECKS = {
        # pointwise: the conditional mean at each conditioning value, against the exact derivative identity. The two
        # tower checks below integrate the conditional back out over the conditioning axis, so errors at different
        # values can cancel; this one cannot be fooled that way.
        'moments': 'check_conditional_moments',
        # the same identity, but against the moments of the conditional's cosine cdf/pdf GRID rather than of its
        # transform -- the only check that reaches that layer, and several times dearer (a grid per conditioning point)
        'grid_moments': 'check_conditional_grid_moments',
        'total_expectation': 'check_total_expectation',
        'total_probability': 'check_total_probability',
    }

    #: Reserved keys of a ``conditional:`` block that configure a check rather than declare a tolerance, mapped to the
    #: checks they apply to. ``quantiles`` targets specific conditioning values (of the conditioning marginal) instead
    #: of the default span; ``curves`` additionally draws that many conditional densities per axis (~1 s each, nothing
    #: asserted on them). Neither reaches the tower checks, which integrate over the whole conditioning axis and so
    #: choose their own nodes.
    _CONDITIONAL_OPTS = {'quantiles': ('moments', 'grid_moments'), 'curves': ('moments',)}

    def _compare_atom_conditional(self, jd, ms, pair: tuple, tols: dict, title: str, name: str = '') -> None:
        """
        Compare the exact **atom conditional** ``R_other | R_on = 0`` against the msprime ground truth, for each
        conditioning axis of one bin pair.

        The only conditional with a ground truth worth the name: ``{R_on = 0}`` has positive probability, so the
        replicates whose conditioning bin is empty *are* the conditioning set, with no window and no bandwidth bias
        (unlike ``R_other | R_on = v``, which a sample can only estimate over a window). It is also the only check
        that reaches ``value = 0`` at all: every other conditional check places its conditioning values at
        ``quantile(p0 + (1 - p0) u)``, strictly above the atom, so none of them exercise the closed-form atom
        transform.

        Asserts the atom's ``mass``, and hands every other requested statistic to :meth:`compare_stat`, so both the
        conditional's moments (``mean`` / ``var``) and its ``cdf`` / ``pdf`` / ``quantile`` grids are validated against
        the sample by the same machinery as any other distribution's. On an axis whose conditioning bin is never empty
        there is no atom, and only the mass is asserted (it must be zero on both sides).

        :param jd: The analytic joint distribution of the pair.
        :param ms: The msprime operand, carrying the cached ground truth.
        :param pair: The bin pair ``(i, j)``.
        :param tols: ``{stat: tolerance}``, over ``mass`` and any statistic :meth:`compare_stat` accepts.
        :param title: Title prefix for the log line.
        :param name: Name prefix for the plot file.
        :raises ValueError: If the ground truth was not cached for this pair (the fixture predates it).
        """
        cached = {(i, j, on): rest for i, j, on, *rest in getattr(ms, '_atom_conditional', [])}

        # a fixture predating this cache has no entry at all; that must fail loudly rather than pass by checking
        # nothing. Both axes are always cached (an axis with no atom carries a zero mass), so both must be there
        if not all((pair[0], pair[1], on) in cached for on in ('a', 'b')):
            raise ValueError(
                f"No cached atom-conditional ground truth for pair {pair}. Regenerate the fixture "
                f"(create_comparison) after adding an 'atom' block, so the msprime side is cached with it."
            )

        for on in ('a', 'b'):
            mass, emp = cached[(pair[0], pair[1], on)]
            sub_title = f"{title}: conditional {pair} atom on {on}"
            sub_name = f"{name}_conditional_{pair[0]}_{pair[1]}_atom_{on}"

            if 'mass' in tols:
                t0 = time.perf_counter()
                diff = abs(float(jd._atoms['a0' if on == 'a' else 'b0']) - mass)
                runtime = time.perf_counter() - t0
                self.runtimes = getattr(self, 'runtimes', {})
                self.runtimes[f"{sub_title}: mass"] = runtime
                self._log_result(self._result_message(f"{sub_title}: mass", diff, tols['mass'], 'max abs', runtime),
                                 diff, tols['mass'])

            if emp is None:
                continue  # this bin is never empty, so there is no atom to condition on and nothing else to compare

            cond = jd.conditional(on, 0.0)
            for stat, tol in tols.items():
                if stat != 'mass':
                    self.compare_stat(ph=cond, ms=emp, stat=stat, tol=tol, title=sub_title, name=sub_name)

    def _compare_windowed_conditional(self, jd, ms, pair: tuple, tols: dict, title: str) -> None:
        """
        Compare the **nested conditional** ``R_other | R_on = v`` against the msprime ground truth, over the
        conditioning windows cached for this pair.

        The only external check the nested conditional has (away from the atom): every other conditional check is an
        identity the analytic joint must satisfy, so a systematic error shared by the transform and the identity would
        pass them all. Here msprime decides.

        Both sides are averaged over the *same* window -- the sample by construction, phasegen by
        :meth:`~phasegen.distributions.reward.JointRewardDistribution.window_average` -- so the ``O(h)`` window bias
        cancels rather than being corrected for, and the residual is the sample's standard error alone.

        The ``mean`` is reported in **standard errors of the sample**, and its tolerance is a number of sigmas. Once
        the window bias is gone, the mean has no floor other than the sampling noise of the window, so a sigma is the
        only scale that means the same thing across demographies and replicate counts: a relative tolerance would have
        to be loosened for a noisy scenario, and would silently stop biting as a scenario's replicate count rose. It
        costs one conditional cumulant per quadrature node.

        The ``cdf`` is reported as a plain absolute difference. A sigma is the wrong unit for it: the empirical CDF's
        binomial error collapses in the tails, where a z-score explodes on an absolute agreement that is in fact
        excellent. Its tolerance is bounded by *msprime's* replicate count rather than by phasegen -- against a large
        sample the nested conditional's CDF resolves to a few 1e-4 -- so it is a tripwire with an order of magnitude
        of headroom, not a precision bound.

        The cdf is also the dear one: every quadrature node is a whole cosine grid, where the mean needs only a
        cumulant. It therefore runs on the axes named by ``cdf_axes`` (default both), so a config can pay for it on
        one conditioning axis while the mean, which is nearly free, still covers both.

        :param jd: The analytic joint distribution of the pair.
        :param ms: The msprime operand, carrying the cached ground truth.
        :param pair: The bin pair ``(i, j)``.
        :param tols: ``{'mean': sigmas}`` and/or ``{'cdf': max_abs}``, plus the ``quantiles`` / ``window`` / ``nodes``
            / ``cdf_axes`` options.
        :param title: Title prefix for the log line.
        :raises ValueError: If the ground truth was not cached for this pair (the fixture predates it), or a requested
            stat is not one of ``mean`` / ``cdf``.
        """
        cached = [c for c in getattr(ms, '_windowed_conditional', []) if (c[0], c[1]) == tuple(pair)]

        if not cached:
            raise ValueError(
                f"No cached windowed-conditional ground truth for pair {pair}. Regenerate the fixture "
                f"(create_comparison) after adding a 'windowed' block, so the msprime side is cached with it."
            )

        stats = [s for s in tols if s not in self._WINDOWED_OPTS]
        for stat in stats:
            if stat not in ('mean', 'cdf'):
                raise ValueError(f"Unknown windowed-conditional stat '{stat}'; expected 'mean' or 'cdf'.")

        nodes = tols.get('nodes')
        cdf_axes = tols.get('cdf_axes', ('a', 'b'))
        worst = {s: 0.0 for s in stats}
        t0 = time.perf_counter()

        for _, _, on, v, h, n_win, mean, mean_se, ys, cdf in cached:
            if 'mean' in worst:
                got = float(jd.window_average(lambda c: c.mean, on, v, h, n_nodes=nodes)[0])
                worst['mean'] = max(worst['mean'], abs(got - mean) / max(mean_se, 1e-300))
            if 'cdf' in worst and on in cdf_axes:
                got = np.asarray(jd.window_average(lambda c: c.cdf(ys), on, v, h, n_nodes=nodes), dtype=float)
                worst['cdf'] = max(worst['cdf'], float(np.abs(got - cdf).max()))

        runtime = time.perf_counter() - t0
        for stat in stats:
            sub_title = f"{title}: conditional {pair} windowed: {stat}"
            self.runtimes = getattr(self, 'runtimes', {})
            self.runtimes[sub_title] = runtime
            label = 'sigma' if stat == 'mean' else 'max abs'
            self._log_result(self._result_message(sub_title, worst[stat], tols[stat], label, runtime),
                             worst[stat], tols[stat])

    def _compare_conditional(self, jd, pair: tuple, tols: dict, title: str, name: str = '', ms=None) -> None:
        """
        Run the requested conditional self-consistency checks for one bin pair and assert each against its tolerance.

        :param jd: The analytic joint distribution of the pair.
        :param pair: The bin pair ``(i, j)``.
        :param tols: ``{check_name: tolerance}``, keyed by :attr:`_CONDITIONAL_CHECKS`.
        :param title: Title prefix for the log line.
        :param name: Name prefix for the plot file.
        :param ms: The msprime operand, needed only by the ``atom`` sub-block.
        :raises ValueError: If a requested check is not one of :attr:`_CONDITIONAL_CHECKS`.
        """
        for key, tol in tols.items():
            if key in self._CONDITIONAL_OPTS:
                continue
            if key == 'atom':
                # the one conditional check with an msprime ground truth (see :meth:`_compare_atom_conditional`)
                self._compare_atom_conditional(jd, ms, pair, tol, title, name)
                continue
            if key == 'windowed':
                # the nested conditional against msprime, both sides averaged over the same conditioning window
                self._compare_windowed_conditional(jd, ms, pair, tol, title)
                continue
            if key not in self._CONDITIONAL_CHECKS:
                raise ValueError(f"Unknown conditional check '{key}' for pair {pair}; expected one of "
                                 f"{list(self._CONDITIONAL_CHECKS)} (or an option: {list(self._CONDITIONAL_OPTS)}).")
            opts = {o: tols[o] for o, checks in self._CONDITIONAL_OPTS.items() if o in tols and key in checks}
            t0 = time.perf_counter()
            res = getattr(jd, self._CONDITIONAL_CHECKS[key])(tol=tol, **opts)
            diff = max(res.values()) if res else 0.0  # worst over the two conditioning axes
            runtime = time.perf_counter() - t0

            sub_title = f"{title}: conditional {pair} {key}"
            self.runtimes = getattr(self, 'runtimes', {})  # robust to deserialized objects that bypass __init__
            self.runtimes[sub_title] = runtime
            msg = self._result_message(sub_title, diff, tol, self._diff_label(f"conditional_{key}"), runtime)

            # plot the conditional mean against the conditioning quantile, phasegen (nested inversion) vs the exact
            # derivative identity, plus the conditional densities themselves when ``curves`` asked for them. Only
            # ``moments`` yields a curve; the tower checks collapse to a single scalar.
            if self.visualize and key == 'moments':
                curves = getattr(jd, 'conditional_moment_curves', {})
                if curves:
                    self._plot_conditional_moments(curves, getattr(jd, 'conditional_densities', {}),
                                                   msg if self.show_title else None,
                                                   f"{name}_conditional_{pair[0]}_{pair[1]}_{key}")
            self._log_result(msg, diff, tol)

    def _compare_pairwise_surface(self, ph, ms, pair: tuple, tols: dict, title: str, name: str,
                                  joint_fn=None, surface_attr: str = '_joint_surface', stat_label: str = None) -> None:
        """
        Full-grid comparison of the within-tree joint distribution of one bin pair ``(i, j)``: the analytic
        ``joint_distribution(i, j)`` versus the cached empirical joint CDF / density over a 2D grid. For each of
        ``cdf`` and ``pdf`` requested in ``tols`` it asserts the worst element-wise difference over the grid and (when
        visualizing) draws three surfaces side by side -- phasegen, msprime and their element-wise difference.
        """
        i, j = pair
        jd = joint_fn(i, j) if joint_fn is not None else ph.joint_distribution(i, j)

        entry = next((e for e in getattr(ms, surface_attr, []) if (e[0], e[1]) == (i, j)), None)
        if entry is None:
            raise ValueError(f"No cached empirical surface for pair {pair}; regenerate the comparison fixture.")
        _i, _j, xs, ys, cdf_ms, pdf_ms = entry
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

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

            # the joint cdf/pdf on the whole grid (2D cosine inversion). The atom-edge head, where the cosine box is
            # biased, is dropped below (the first two points per axis).
            xs_d, ys_d = xs, ys
            ms_grid = (cdf_ms if kind == 'cdf' else pdf_ms)
            grid_ms = np.asarray(ms_grid, dtype=float)
            grid_ph = np.asarray(jd.cdf(xs_d, ys_d) if kind == 'cdf' else jd.pdf(xs_d, ys_d), dtype=float)

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

    def _plot_conditional_moments(self, curves: dict, densities: dict, title: str, name: str) -> None:
        """
        The conditional-mean check, in two or three panels: the conditional mean at each conditioning quantile as
        **grouped bars** (nested inversion beside the exact derivative identity, one group per axis); the **scaled
        error the check asserts on** as bars coloured by magnitude (the same ``coolwarm`` scale saturating at
        :attr:`surface_diff_saturation` as the other diff panels) -- the same metric the result line reports, and not a
        raw absolute difference, which the large-mean bars would dominate while saying nothing where the conditional
        mean is ~0; and, when ``densities`` were computed, the conditional densities the means summarise.

        Bars, not a curve, for the means: the check evaluates a handful of *discrete* conditioning quantiles, and a
        line between them would draw an interpolation that was never computed. The densities *are* curves, being the
        distribution the check does not otherwise look at (the mean comes from the transform, not from this grid).

        :param curves: ``{axis: (quantiles, exact, nested, errors)}``, as stashed by
            :meth:`~phasegen.distributions.reward.JointRewardDistribution.check_conditional_moments`.
        :param densities: ``{axis: [(quantile, value, ys, density)]}`` from the same call, possibly empty.
        :param title: Plot title (the comparison log line), or ``None``.
        :param name: Plot file name.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        sat = self.surface_diff_saturation
        norm = colors.Normalize(vmin=0.0, vmax=sat)
        n_panels = 3 if densities else 2
        fig, axes = plt.subplots(ncols=n_panels, figsize=(6.5 * n_panels, 5))
        axm, axd = axes[0], axes[1]

        # lay every (axis, quantile) pair out on one categorical axis, grouped by conditioning axis
        labels, exact, nested, errs = [], [], [], []
        for on, (us, ex, ne, er) in curves.items():
            for u, e, n, r in zip(us, ex, ne, er):
                labels.append(f"R_{on}\n{u:.2f}")
                exact.append(e)
                nested.append(n)
                errs.append(r)
        if not labels:
            plt.close(fig)
            return

        exact, nested = np.asarray(exact, float), np.asarray(nested, float)
        x = np.arange(len(labels))
        w = 0.38

        axm.bar(x - w / 2, nested, w, label='nested inversion', alpha=0.9)
        axm.bar(x + w / 2, exact, w, label='exact identity', alpha=0.9)
        axm.set_xticks(x)
        axm.set_xticklabels(labels, fontsize=8)
        axm.set_xlabel('conditioning axis and quantile')
        axm.set_ylabel('conditional mean')
        axm.legend()

        diff = np.asarray(errs, float)
        axd.bar(x, diff, 0.6, color=cm.coolwarm(norm(diff)))
        axd.set_xticks(x)
        axd.set_xticklabels(labels, fontsize=8)
        axd.set_xlabel('conditioning axis and quantile')
        axd.set_ylabel('relative difference')
        axd.set_ylim(0.0, max(sat, float(diff.max()) * 1.1 if diff.size else sat))
        axd.set_title('difference', fontsize=self.title_fontsize)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=axd)

        if densities:
            # log reward axis: conditioning at u = 0.01 and at u = 0.99 puts the two conditionals orders of magnitude
            # apart in scale, and on a linear axis every curve but the widest collapses onto the origin
            axc = axes[2]
            for on, series in densities.items():
                other = 'b' if on == 'a' else 'a'
                for u, v, ys, pdf in series:
                    axc.plot(ys[1:], pdf[1:], ls='-' if on == 'a' else '--',
                             label=f"$R_{other} \\mid R_{on} = {v:.3g}$ ($u = {u:.2f}$)")
            axc.set_xscale('log')
            axc.set_xlabel('time')
            axc.set_ylabel('conditional density')
            axc.set_title('conditional distributions', fontsize=self.title_fontsize)
            axc.legend(fontsize=7)

        if title and self.show_title:
            fig.suptitle(title, fontsize=self.suptitle_fontsize)
        self._save_and_show(name)

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

    def _pairwise_surface_pairs(self, spec: dict = None) -> dict:
        """The per-distribution bin pairs in ``spec`` that request a full-grid pairwise surface comparison (the
        non-``cdf``/``pdf`` keys under a ``pairwise`` group) -- used to cache their empirical grids. ``spec`` is a
        tolerance subtree (the top-level msprime stats, or the nested ``empirical`` sub-spec); it defaults to the
        whole ``tolerance`` tree (an ``empirical`` sub-block carries no top-level ``pairwise`` key, so it is skipped)."""
        out = {}
        for dist, data in self._expand_keys(spec if spec is not None else self.comparisons.get('tolerance', {})).items():
            pairwise = data.get('pairwise') if isinstance(data, dict) else None
            if not isinstance(pairwise, dict):
                continue

            # pair keys are everything that is not an aggregate stat ('cdf'/'pdf')
            pairs = [ast.literal_eval(k) if isinstance(k, str) else tuple(k)
                     for k in pairwise if k not in ('cdf', 'pdf')]
            if pairs:
                out[dist] = list(dict.fromkeys(pairs))  # de-dupe, preserve order
        return out

    #: Defaults of a ``conditional: {pair}: windowed:`` block. ``quantiles`` places the window centres in quantile
    #: space of the conditioning marginal (so they mean the same across demographies); ``window`` is the half-width as
    #: a *fraction* of the centre, which keeps it below the centre and so clear of the conditioning atom at 0. Wider
    #: is not worse here -- both sides average over the same window, so a wide one buys replicates rather than bias.
    _WINDOWED_DEFAULTS = {'quantiles': (0.25, 0.5, 0.75), 'window': 0.2}

    #: Keys of a ``windowed:`` block that configure the check rather than declare a tolerance. ``nodes`` sets the
    #: quadrature nodes per window (2 suffices; 1 is not a quadrature at all but the conditional at the window centre,
    #: which reinstates the very window bias the check exists to cancel), and ``cdf_axes`` restricts the dear ``cdf``
    #: to some conditioning axes while the cheap ``mean`` still runs on all of them.
    _WINDOWED_OPTS = ('quantiles', 'window', 'nodes', 'cdf_axes')

    def _windowed_conditional_specs(self, spec: dict) -> dict:
        """The ``(i, j, on, value, half_width)`` conditioning windows requested by any ``conditional: {pair}:
        windowed:`` block, per distribution. The centres come from the **exact** marginal's quantiles, so they are
        deterministic and the msprime side can be cached against them."""
        out = {}
        for dist, data in self._expand_keys(spec).items():
            conditional = data.get('conditional') if isinstance(data, dict) else None
            if not isinstance(conditional, dict):
                continue

            specs = []
            for key, sub in conditional.items():
                if not isinstance(sub, dict) or 'windowed' not in sub:
                    continue
                pair = ast.literal_eval(key) if isinstance(key, str) else tuple(key)
                specs += self._windows_of(getattr(self.ph, dist).joint_distribution(*pair), pair, sub['windowed'])

            if specs:
                out[dist] = specs
        return out

    def _windows_of(self, jd, pair: tuple, tols: dict) -> list:
        """The conditioning windows of one pair: each axis, each requested quantile of that axis's exact marginal."""
        qs = tols.get('quantiles', self._WINDOWED_DEFAULTS['quantiles'])
        rel = float(tols.get('window', self._WINDOWED_DEFAULTS['window']))

        specs = []
        for on in ('a', 'b'):
            marg = jd.marginal(on)
            p0 = float(jd._atoms['a0' if on == 'a' else 'b0'])
            for q in qs:
                # place the centre above the atom, in quantile space of the *continuous* part
                v = float(marg.quantile(p0 + (1.0 - p0) * float(q)))
                specs.append((pair[0], pair[1], on, v, rel * v))
        return specs

    def _atom_conditional_pairs(self, spec: dict) -> dict:
        """The per-distribution bin pairs whose ``conditional:`` block requests an ``atom`` check, so their
        (msprime) atom-conditional ground truth is cached. Unlike the other conditional checks, which are analytic
        identities, this one needs a sample and so needs the fixture regenerated when a pair is added."""
        out = {}
        for dist, data in self._expand_keys(spec).items():
            conditional = data.get('conditional') if isinstance(data, dict) else None
            if not isinstance(conditional, dict):
                continue

            pairs = [ast.literal_eval(k) if isinstance(k, str) else tuple(k)
                     for k, sub in conditional.items() if isinstance(sub, dict) and 'atom' in sub]
            if pairs:
                out[dist] = list(dict.fromkeys(pairs))
        return out

    def cache_ground_truth(self) -> None:
        """Cache the ground truth needed by the configured comparisons -- the standard per-statistic caches
        (:meth:`MsprimeCoalescent.touch` / :meth:`SampledCoalescent.touch`), any full-grid pairwise surface grids, and
        the atom-conditional ground truth. The msprime operand is touched for the top-level ``tolerance`` stats, the
        sampler for the nested ``empirical`` sub-spec; each only if its stats are present, so a config validates
        against msprime, the sampler, or both. Call before :meth:`drop` so the grids are serialized with the
        comparison."""
        tol = self._expand_keys(self.comparisons.get('tolerance', {}))
        empirical_spec = tol.get('empirical')
        msprime_spec = {k: v for k, v in tol.items() if k != 'empirical'}

        if msprime_spec or self.comparisons.get('statistics'):
            self.ms.touch()

            # the coalescent-level scalar statistics (F_ST, the Patterson f-statistics) are evaluated straight off the
            # simulated data and the demography, both of which :meth:`MsprimeCoalescent.drop` discards, so their values
            # have to be cached here rather than recomputed at comparison time
            for stat, spec in self.comparisons.get('statistics', {}).items():
                args = spec.get('args', []) if isinstance(spec, dict) else []
                self._ms_statistics[(stat, tuple(args))] = self._eval_statistic(self.ms, stat, args)

            for dist, pairs in self._pairwise_surface_pairs(msprime_spec).items():
                getattr(self.ms, dist).cache_joint_surface(pairs)
            for dist, pairs in self._atom_conditional_pairs(msprime_spec).items():
                getattr(self.ms, dist).cache_atom_conditional(pairs)
            for dist, specs in self._windowed_conditional_specs(msprime_spec).items():
                getattr(self.ms, dist).cache_windowed_conditional(specs)

        if empirical_spec:
            if self.n_samples is None:
                raise ValueError("A 'tolerance.empirical' block requires 'n_samples' to be set in the config.")
            self.empirical.touch()
            for dist, pairs in self._pairwise_surface_pairs(empirical_spec).items():
                getattr(self.empirical, dist).cache_joint_surface(pairs)

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

        tol = self._expand_keys(self.comparisons['tolerance'])
        empirical_spec = tol.pop('empirical', None)  # the nested self-consistency sub-spec (vs the sampler)

        for dist, data in tol.items():
            self._compare_stat_recursively(
                ph=getattr(self.ph, dist),
                ms=getattr(self.ms, dist),
                data=data,
                title=f"{title}: {dist}",
                name=dist
            )

        # nested ``empirical`` sub-spec: the same stats, but the candidate operand is PhaseGen's own sampler
        # (:attr:`empirical`), not msprime -- a self-consistency check. The ``empirical`` marker rides in the title
        # (``...: empirical: ...``) so a downstream reader can tell the two kinds of comparison apart.
        for dist, data in (empirical_spec or {}).items():
            self._compare_stat_recursively(
                ph=getattr(self.ph, dist),
                ms=getattr(self.empirical, dist),
                data=data,
                title=f"{title}: empirical: {dist}",
                name=f"empirical_{dist}"
            )

        # coalescent-level scalar statistics (optionally parameterized with population arguments), e.g. F_ST and
        # the Patterson f-statistics f2/f3/f4. Each entry is either ``<stat>: <tol>`` or
        # ``<stat>: {args: [...], tol: <tol>}``.
        for stat, spec in self.comparisons.get('statistics', {}).items():
            args = spec.get('args', []) if isinstance(spec, dict) else []
            tol = spec['tol'] if isinstance(spec, dict) else spec
            label = f"{title}: {stat}" + (f"({', '.join(map(str, args))})" if args else "")

            if (stat, tuple(args)) not in self._ms_statistics:
                raise KeyError(f"The ground truth of '{stat}' is not cached in this comparison. It is computed from "
                               f"the simulated data, which the serialized comparison drops, so a newly configured "
                               f"statistic needs its fixture regenerated (see the 'regenerate_fixtures' rule).")

            self._compare_scalar(
                ph=self._eval_statistic(self.ph, stat, args),
                ms=self._ms_statistics[(stat, tuple(args))],
                tol=tol,
                title=label
            )

        self.logger.info(f"Number of assertions: {self.n_assertions}")
