"""Empirical (msprime-backed) distributions and the MsprimeCoalescent simulator."""

import itertools
import logging
from collections import defaultdict
from ..caching import cached_property, cache
from typing import Generator, List, Callable, Tuple, Dict, Iterator, Optional, Sequence, Type, TYPE_CHECKING
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ..coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from ..demography import Demography
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..spectrum import SFS, SFS2, JointSFS, TwoLocusSFS
from ..utils import parallelize

from .base import DensityAwareDistribution
from .spectra import FoldedSFSDistribution, SFSDistribution, TajimaSFSMixin, UnfoldedSFSDistribution
from .coalescent import AbstractCoalescent, Coalescent

if TYPE_CHECKING:
    import msprime
    import tskit

expm = Backend.expm
logger = logging.getLogger('phasegen')


class EmpiricalJointSFSDistribution:  # pragma: no cover
    """
    Empirical (msprime-based) joint site-frequency spectrum, exposing the same ``mean``/``var``/``m2``/``m3``
    interface as :class:`JointSFSDistribution` so that the two can be compared by
    :class:`~phasegen.comparison.Comparison`. The moments are pre-computed arrays (so the object can be serialized
    as cached ground truth).
    """

    def __init__(self, moments: np.ndarray, samples: np.ndarray = None):
        """
        Initialize the distribution.

        :param moments: Per-configuration (non-central) moments of orders ``1, 2, ...``, stacked along the first
            axis, i.e. an array of shape ``(max_order, n_0 + 1, ..., n_{P-1} + 1)``.
        :param samples: Optional per-replicate joint SFS branch lengths, shape ``(n_replicates, n_0 + 1, ...)``, used
            to pre-compute the within-tree joint ground truth (:meth:`cache_joint`); dropped before serialization.
        """
        #: Non-central moments per descendant configuration, indexed by order minus one.
        self._moments: np.ndarray = np.asarray(moments)

        #: Per-replicate joint SFS branch lengths (samples-free after :meth:`cache_joint`).
        self.samples: np.ndarray | None = None if samples is None else np.asarray(samples)

        #: Cached within-tree joint ground truth: ``[(config_a, config_b, cross, [(x, y, cdf), ...]), ...]``.
        self._joint: list = []

    def cross_moment(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...]) -> float:
        """Empirical within-tree cross-moment ``E[L_{config_a} · L_{config_b}]`` from the per-replicate samples."""
        a = self.samples[(slice(None),) + tuple(config_a)]
        b = self.samples[(slice(None),) + tuple(config_b)]
        return float((a * b).mean())

    def joint_cdf(self, config_a: Tuple[int, ...], config_b: Tuple[int, ...], x: float, y: float) -> float:
        """Empirical within-tree joint CDF ``P(L_{config_a} <= x, L_{config_b} <= y)`` from the per-replicate samples."""
        a = self.samples[(slice(None),) + tuple(config_a)]
        b = self.samples[(slice(None),) + tuple(config_b)]
        return float(((a <= x) & (b <= y)).mean())

    def cache_joint(self, pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], quantiles: List[Tuple[float, float]]):
        """Pre-compute, for each config pair, the empirical cross-moment and the joint CDF at marginal-quantile
        points, so the joint ground truth is serialized with the comparison and survives :meth:`drop`. Mirrors
        :meth:`EmpiricalPhaseTypeSFSDistribution.cache_joint` but indexed by descendant configuration."""
        self._joint = []
        for ca, cb in pairs:
            a = self.samples[(slice(None),) + tuple(ca)]
            b = self.samples[(slice(None),) + tuple(cb)]
            points = [(float(np.quantile(a, qa)), float(np.quantile(b, qb)),
                       self.joint_cdf(ca, cb, np.quantile(a, qa), np.quantile(b, qb))) for qa, qb in quantiles]
            self._joint.append((tuple(ca), tuple(cb), float((a * b).mean()), points))

    def drop(self):
        """Drop the (large) per-replicate samples once the joint ground truth has been cached."""
        self.samples = None

    @property
    def mean(self) -> JointSFS:
        """
        Mean of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[0])

    @property
    def m2(self) -> JointSFS:
        """
        Second (non-central) moment of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[1])

    @property
    def m3(self) -> JointSFS:
        """
        Third (non-central) moment of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[2])

    @property
    def var(self) -> JointSFS:
        """
        Variance of the joint site-frequency spectrum.
        """
        return JointSFS(self._moments[1] - self._moments[0] ** 2)

    @property
    def data(self) -> np.ndarray:
        """
        The mean joint site-frequency spectrum array.
        """
        return self._moments[0]


class EmpiricalDistribution(DensityAwareDistribution):  # pragma: no cover
    """
    Probability distribution based on realisations.
    """

    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples: 1-D array of samples.
        """
        super().__init__()

        self._cache = None

        #: Samples
        self.samples = np.array(samples, dtype=float)

    def touch(self, t: np.ndarray):
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super().touch()

        self._cache = dict(
            t=t,
            cdf=self.cdf(t),
            pdf=self.pdf(t)
        )

    def drop(self):
        """
        Drop simulated samples.
        """
        self.samples = None

    @cached_property
    def mean(self) -> float | np.ndarray:
        """
        First moment / mean.
        """
        return np.mean(self.samples, axis=0)

    @cached_property
    def var(self) -> float | np.ndarray:
        """
        Second central moment / variance.
        """
        return np.var(self.samples, axis=0)

    @cached_property
    def m2(self) -> float | np.ndarray:
        """
        Second non-central moment.
        """
        return np.mean(self.samples ** 2, axis=0)

    @cached_property
    def m3(self) -> float | np.ndarray:
        """
        Third non-central moment.
        """
        return np.mean(self.samples ** 3, axis=0)

    @cached_property
    def m4(self) -> float | np.ndarray:
        """
        Fourth non-central moment.
        """
        return np.mean(self.samples ** 4, axis=0)

    @cached_property
    def cov(self) -> float | np.ndarray:
        """
        Covariance matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.cov(self.samples, rowvar=False))

    @cached_property
    def corr(self) -> float | np.ndarray:
        """
        Correlation matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.corrcoef(self.samples, rowvar=False))

    def moment(self, k: int) -> float | np.ndarray:
        """
        Get the kth moment.

        :param k: The order of the moment
        :return: The kth moment
        """
        return np.mean(self.samples ** k, axis=0)

    def _cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Time.
        :return: Cumulative probability.
        """
        x = np.sort(self.samples)
        y = np.arange(1, len(self.samples) + 1) / len(self.samples)

        if x.ndim == 1:
            return np.interp(t, x, y)

        if x.ndim == 2:
            return np.array([np.interp(t, x_, y) for x_ in x.T])

        raise ValueError("Samples must be 1 or 2 dimensional.")

    def _quantile(self, q: float) -> float:
        """
        Get the qth quantile.

        :param q: Quantile.
        :return: Quantile.
        """
        return np.quantile(self.samples, q=q)

    def _pdf(
            self,
            t: float | np.ndarray,
            n_bins: int = 10000,
            sigma: float = None,
            samples: np.ndarray = None,
            **kwargs: dict
    ) -> float | np.ndarray:
        """
        Density function.

        :param sigma: Sigma for Gaussian filter.
        :param n_bins: Number of bins.
        :param t: Time.
        :param samples: Samples.
        :return: Density.
        """
        samples = self.samples if samples is None else samples

        if samples.ndim == 2:
            return np.array([self._pdf(t, n_bins=n_bins, sigma=sigma, samples=s) for s in samples.T])

        hist, bin_edges = np.histogram(samples, range=(0, max(samples)), bins=n_bins, density=True)

        # determine bins for u
        bins = np.minimum(np.sum(bin_edges <= t[:, None], axis=1) - 1, np.full_like(t, n_bins - 1, dtype=int))

        # use proper bins for y values
        y = hist[bins]

        # smooth using gaussian filter
        if sigma is not None:
            y = gaussian_filter1d(y, sigma=sigma)

        return y


class EmpiricalSFSDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    SFS probability distribution based on realisations.
    """

    def __init__(self, samples: np.ndarray | list):
        """
        Create object.

        :param samples: 2-D array of samples.
        """
        super().__init__(samples)

    @cached_property
    def mean(self) -> SFS:
        """
        First moment / mean.
        """
        return SFS(super().mean)

    @cached_property
    def var(self) -> SFS:
        """
        Second central moment / variance.
        """
        return SFS(super().var)

    @cached_property
    def m2(self) -> SFS:
        """
        Second non-central moment.
        """
        return SFS(super().m2)

    @cached_property
    def cov(self) -> SFS2:
        """
        Covariance matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return SFS2(np.nan_to_num(np.cov(self.samples, rowvar=False)))

    @cached_property
    def corr(self) -> SFS2:
        """
        Correlation matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return SFS2(np.nan_to_num(np.corrcoef(self.samples, rowvar=False)))


class DictContainer(dict):  # pragma: no cover
    """
    Dictionary container.
    """
    pass


class EmpiricalPhaseTypeDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    Phase-type distribution based on realisations.
    """

    def __init__(
            self,
            samples: np.ndarray | list,
            pops: List[str],
            locus_agg: Callable = lambda x: x.sum(axis=0)
    ):
        """
        Create object.

        :param samples: 3-D array of samples.
        :param pops: List of population names.
        :param locus_agg: Aggregation function for loci.
        """
        over_loci = locus_agg(samples).astype(float)
        over_demes = samples.sum(axis=1).astype(float)

        super().__init__(over_loci.sum(axis=0))

        #: Population names
        self.pops = pops

        #: Samples by deme and locus
        self._samples = samples

        # zero-variance demes/loci make corrcoef divide by zero; the resulting NaNs are expected here, so
        # silence the benign warning
        with np.errstate(divide='ignore', invalid='ignore'):
            #: Covariance matrix for the demes
            self.pops_cov: np.ndarray = np.cov(over_loci)

            #: Correlation matrix for the demes
            self.pops_corr: np.ndarray = np.corrcoef(over_loci)

            #: Covariance matrix for the loci
            self.loci_corr: np.ndarray = np.corrcoef(over_demes)

            #: Correlation matrix for the loci
            self.loci_cov: np.ndarray = np.cov(over_demes)

    def touch(self, t: np.ndarray):
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super().touch(t)

        [d.touch(t) for d in self.demes.values()]
        [l.touch(t) for l in self.loci.values()]

    def drop(self):
        """
        Drop simulated samples.
        """
        super().drop()

        self._samples = None

        [d.drop() for d in self.demes.values()]
        [l.drop() for l in self.loci.values()]

    @cached_property
    def demes(self) -> Dict[str, EmpiricalDistribution]:
        """
        Get the distribution for each deme.

        :return: Dictionary of distributions.
        """
        demes = DictContainer(
            {pop: EmpiricalDistribution(self._samples.sum(axis=0)[i]) for i, pop in enumerate(self.pops)}
        )

        # TODO this is the covariance in the tree height but phasegen
        #  provides the covariance in the number of lineages per deme
        demes.cov = self.pops_cov
        demes.corr = self.pops_corr

        return demes

    @cached_property
    def loci(self) -> Dict[int, EmpiricalDistribution]:
        """
        Get the distribution for each locus.

        :return: Dictionary of distributions.
        """
        loci = DictContainer(
            {i: EmpiricalDistribution(self._samples[i].sum(axis=0)) for i in range(self._samples.shape[0])}
        )

        loci.cov = self.loci_cov
        loci.corr = self.loci_corr

        return loci


class EmpiricalPhaseTypeSFSDistribution(EmpiricalPhaseTypeDistribution, TajimaSFSMixin):  # pragma: no cover
    """
    SFS phase-type distribution based on realisations.
    """

    def _tajima_n(self) -> int:
        # derive n from the (serialized) mean vector so this works on fixtures restored without ``n``
        return len(np.asarray(self.mean)) - 1

    def _tajima_mean(self) -> np.ndarray:
        n = self._tajima_n()
        return np.asarray(self.mean)[1:n]

    def _tajima_cov(self) -> np.ndarray:
        n = self._tajima_n()
        return np.asarray(self.cov)[1:n, 1:n]

    def __init__(
            self,
            branch_lengths: np.ndarray,
            mutations: np.ndarray,
            pops: List[str],
            sfs_dist: Type[SFSDistribution],
            locus_agg: Callable = lambda x: x.sum(axis=0),
    ):
        """
        Create object.

        :param branch_lengths: 4-D array of branch length samples.
        :param mutations: 4-D array of mutation counts.
        :param pops: List of population names.
        :param sfs_dist: SFS distribution class.
        :param locus_agg: Aggregation function for loci.
        """
        over_loci = locus_agg(branch_lengths).astype(float)

        EmpiricalDistribution.__init__(self, over_loci.sum(axis=0))

        #: Population names
        self.pops = pops

        # : Number of lineages
        self.n = branch_lengths.shape[-1] - 1

        #: SFS distribution class
        self._sfs_dist = sfs_dist

        #: Branch length samples by deme and locus
        self._samples = branch_lengths

        #: Mutation counts by deme and locus
        self._mutations = mutations

        #: Correlation matrix for the loci
        self.pops_corr = self._get_stat_pops(over_loci, np.corrcoef)

        #: Covariance matrix for the demes
        self.pops_cov: np.ndarray = self._get_stat_pops(over_loci, np.cov)

        #: Correlation matrix for the loci
        self.loci_corr: np.ndarray = None

        #: Covariance matrix for the loci
        self.loci_cov: np.ndarray = None

        #: Generated probability mass by iterator returned from :meth:`get_mutation_configs`.
        self.generated_mass = 0

    def drop(self):
        """
        Drop simulated samples.
        """
        super().drop()

        self._mutations = None

    def cross_moment(self, i: int, j: int) -> float:
        """
        Empirical cross-moment ``E[L_i L_j]`` of the branch lengths subtending ``i`` and ``j`` samples, from the
        per-replicate SFS branch-length samples — the simulated counterpart of
        :meth:`JointRewardDistribution.moment` ``(1, 1)``.

        :param i: First frequency class.
        :param j: Second frequency class.
        :return: The empirical cross-moment.
        """
        return float((self.samples[:, i] * self.samples[:, j]).mean())

    def joint_cdf(self, i: int, j: int, x: float, y: float) -> float:
        """
        Empirical joint CDF ``P(L_i <= x, L_j <= y)`` of two SFS bins, from the per-replicate samples — the
        simulated counterpart of :meth:`JointRewardDistribution.cdf`.

        :param i: First frequency class.
        :param j: Second frequency class.
        :param x: Threshold for ``L_i``.
        :param y: Threshold for ``L_j``.
        :return: The empirical joint probability.
        """
        return float(((self.samples[:, i] <= x) & (self.samples[:, j] <= y)).mean())

    def cache_joint(self, pairs: List[Tuple[int, int]], quantiles: List[Tuple[float, float]]):
        """
        Pre-compute, for each bin pair, the empirical cross-moment and the joint CDF at marginal-quantile points, so
        the (1M-replicate) joint ground truth is serialized with the comparison and survives :meth:`drop`. Stored as
        ``self._joint = [(i, j, cross, [(x, y, cdf), ...]), ...]`` (small, samples-free).
        """
        s = self.samples
        self._joint = [
            (i, j, float((s[:, i] * s[:, j]).mean()),
             [(float(np.quantile(s[:, i], qa)), float(np.quantile(s[:, j], qb)),
               self.joint_cdf(i, j, np.quantile(s[:, i], qa), np.quantile(s[:, j], qb))) for qa, qb in quantiles])
            for i, j in pairs
        ]

    @staticmethod
    def _get_stat_pops(samples: np.ndarray, callback: Callable) -> np.ndarray:
        """
        Get the covariance matrix for the demes.

        :param callback: Callback function to apply to the samples.
        :return: Covariance matrix.
        """
        stats = np.zeros((samples.shape[0], samples.shape[0], samples.shape[2], samples.shape[2]))

        # bins with no variance (e.g. always-zero monomorphic counts) make np.corrcoef divide by a zero standard
        # deviation; the resulting NaNs are expected here, so silence the benign warning rather than emit it.
        with np.errstate(divide='ignore', invalid='ignore'):
            for i, j in itertools.product(range(1, samples.shape[2] - 1), range(1, samples.shape[2] - 1)):
                stats[:, :, i, j] = callback(samples[:, :, i])

        return stats

    @cached_property
    def demes(self) -> Dict[str, EmpiricalDistribution]:
        """
        Get the distribution for each deme.

        :return: Dictionary of distributions.
        """
        return {pop: EmpiricalSFSDistribution(self._samples.sum(axis=0)[i]) for i, pop in enumerate(self.pops)}

    @cached_property
    def mutation_configs(self) -> Dict[Tuple[float, ...], float]:
        """
        Get a dictionary of all mutation configurations and their probabilities.

        :return: Dictionary of distributions.
        """
        configs = defaultdict(lambda: 0)

        for config in self._mutations[0, 0]:
            configs[tuple(config)] += 1 / self._mutations.shape[2]

        return configs

    def get_mutation_config(self, config: Sequence[int]) -> float:
        """
        Get the probability of observing the given mutational configuration.

        :param config: The mutational configuration.
        :return: The probability of observing the given mutational configuration.
        """
        return self.mutation_configs[tuple(config)]

    def get_mutation_configs(self) -> Iterator[Tuple[Tuple[float, ...], float]]:
        """
        An iterator over the probabilities of observing mutational configurations according to the infinite sites model.
        The order of the mutational configurations generated ascends in the number of mutations observed.

        :return: An iterator over the probabilities of observing mutational configurations.
        """
        # reset generated mass
        self.generated_mass = 0

        # iterate over number of mutations
        i = 0
        while True:
            # iterate over configurations
            for config in self._sfs_dist._get_configs(self.n, i):
                p = self.get_mutation_config(config=config)
                self.generated_mass += p
                yield config, p

            # increase counter for number of mutations
            i += 1


class EmpiricalTwoLocusSFSDistribution:  # pragma: no cover
    """
    Empirical (msprime-based) two-locus SFS, exposing the same ``mean`` interface as
    :class:`TwoLocusSFSDistribution` (a :class:`~phasegen.spectrum.TwoLocusSFS`) so the two can be compared by
    :class:`~phasegen.comparison.Comparison`.
    """

    def __init__(self, mean: np.ndarray, left: np.ndarray = None, right: np.ndarray = None):
        """
        :param mean: The simulated mean two-locus SFS array.
        :param left: Optional per-replicate locus-0 SFS branch lengths ``(num_replicates, n + 1)`` (for the joint
            distribution / cross-moment tracking). Dropped on :meth:`drop` and absent from a serialized comparison.
        :param right: Optional per-replicate locus-1 SFS branch lengths.
        """
        self._mean = np.asarray(mean)
        self._left = None if left is None else np.asarray(left)
        self._right = None if right is None else np.asarray(right)

    @property
    def mean(self) -> TwoLocusSFS:
        """Mean two-locus SFS."""
        return TwoLocusSFS(self._mean)

    def drop(self):
        """Drop the per-replicate samples (the mean is retained)."""
        self._left = None
        self._right = None

    def cross_moment(self, i: int, j: int) -> float:
        """
        Empirical cross-moment ``E[L^0_i L^1_j]`` (the two-locus SFS entry) from the per-replicate locus branch
        lengths — the simulated counterpart of :meth:`JointRewardDistribution.moment` ``(1, 1)``.

        :param i: Locus-0 frequency class.
        :param j: Locus-1 frequency class.
        :return: The empirical cross-moment.
        """
        return float((self._left[:, i] * self._right[:, j]).mean())

    def joint_cdf(self, i: int, j: int, x: float, y: float) -> float:
        """
        Empirical joint CDF ``P(L^0_i <= x, L^1_j <= y)`` from the per-replicate locus branch lengths — the
        simulated counterpart of :meth:`JointRewardDistribution.cdf`.

        :param i: Locus-0 frequency class.
        :param j: Locus-1 frequency class.
        :param x: Threshold for ``L^0_i``.
        :param y: Threshold for ``L^1_j``.
        :return: The empirical joint probability.
        """
        return float(((self._left[:, i] <= x) & (self._right[:, j] <= y)).mean())

    def cache_joint(self, pairs: List[Tuple[int, int]], quantiles: List[Tuple[float, float]]):
        """Pre-compute the two-locus joint ground truth (cross-moment and joint CDF at marginal-quantile points)
        for serialization; same structure as :meth:`EmpiricalPhaseTypeSFSDistribution.cache_joint`."""
        self._joint = [
            (i, j, self.cross_moment(i, j),
             [(float(np.quantile(self._left[:, i], qa)), float(np.quantile(self._right[:, j], qb)),
               self.joint_cdf(i, j, np.quantile(self._left[:, i], qa), np.quantile(self._right[:, j], qb)))
              for qa, qb in quantiles])
            for i, j in pairs
        ]


class MsprimeCoalescent(AbstractCoalescent):
    """
    Empirical coalescent distribution based on `msprime` simulations.
    This is used for testing purposes. Note that the results are stochastic.
    """

    def __init__(
            self,
            n: int | Dict[str, int] | List[int] | LineageConfig,
            demography: Demography = None,
            model: CoalescentModel = StandardCoalescent(),
            loci: int | LocusConfig = 1,
            recombination_rate: float = None,
            mutation_rate: float = None,
            end_time: float = None,
            num_replicates: int = 10000,
            n_threads: int = 100,
            parallelize: bool = True,
            record_migration: bool = False,
            simulate_mutations: bool = False,
            seed: int = None
    ):
        """
        Simulate data using msprime.

        :param n: Number of Lineages.
        :param demography: Demography.
        :param model: Coalescent model.
        :param loci: Number of loci or locus configuration.
        :param recombination_rate: Recombination rate.
        :param mutation_rate: Mutation rate.
        :param end_time: Time when to end the simulation.
        :param num_replicates: Number of replicates.
        :param n_threads: Number of threads.
        :param parallelize: Whether to parallelize.
        :param record_migration: Whether to record migrations which is necessary to calculate statistics per deme.
        :param simulate_mutations: Whether to simulate mutations.
        :param seed: Random seed.
        """
        super().__init__(
            n=n,
            model=model,
            loci=loci,
            recombination_rate=recombination_rate,
            demography=demography,
            end_time=end_time
        )

        if mutation_rate is not None and not simulate_mutations:
            self._logger.warning("Mutation rate is set but mutations are not simulated.")

        #: Site frequency spectrum counts per locus, deme and replicate.
        self.sfs_lengths: np.ndarray | None = None

        #: Total branch lengths per locus, deme and replicate.
        self.total_branch_lengths: np.ndarray | None = None

        #: Tree heights per locus, deme and replicate.
        self.heights: np.ndarray | None = None

        #: Mutations per locus, deme and replicate.
        self.mutations: np.ndarray | None = None

        #: Joint SFS (non-central) moments per descendant configuration, of orders 1, ..., ``_jsfs_max_order``.
        self.jsfs_moments: np.ndarray | None = None

        #: Per-replicate joint SFS branch lengths (capped subset) for the within-tree joint ground truth.
        self.jsfs_samples: np.ndarray | None = None

        #: Number of replicates.
        self.num_replicates: int = num_replicates

        #: Mutation rate.
        self.mutation_rate: float = mutation_rate

        #: Number of threads.
        self.n_threads: int = n_threads

        #: Whether to parallelize computations.
        self.parallelize: bool = parallelize

        #: Whether to record migrations.
        self.record_migration: bool = record_migration

        #: Whether to simulate mutations.
        self.simulate_mutations: bool = simulate_mutations

        #: Random seed.
        self.seed: int = seed

    def get_coalescent_model(self) -> 'msprime.AncestryModel':
        """
        Get the coalescent model.

        :return: msprime coalescent model.
        """
        import msprime as ms

        if isinstance(self.model, StandardCoalescent):
            return ms.StandardCoalescent()

        if isinstance(self.model, BetaCoalescent):
            return ms.BetaCoalescent(alpha=self.model.alpha)

        if isinstance(self.model, DiracCoalescent):
            return ms.DiracCoalescent(psi=self.model.psi, c=self.model.c)

    @cache
    def simulate(self):
        """
        Simulate data using msprime.
        """
        # number of replicates for one thread
        num_replicates = self.num_replicates // self.n_threads
        samples = self.lineage_config.lineage_dict
        demography = self.demography.to_msprime()
        model = self.get_coalescent_model()
        end_time = self.end_time
        n_pops = self.demography.n_pops
        sample_size = self.lineage_config.n

        # joint SFS is accumulated from the same trees, but only for multi-population, single-locus scenarios where
        # it is meaningful (the descendant configuration is by deme of origin)
        compute_jsfs = self.lineage_config.n_pops > 1 and self.locus_config.n == 1
        jsfs_max_order = self._jsfs_max_order
        jsfs_shape = tuple(int(s) + 1 for s in self.lineage_config.lineages)
        name_to_index = {name: i for i, name in enumerate(self.demography.pop_names)}
        n_total = num_replicates * self.n_threads
        # retain a capped subset of per-replicate joint SFS branch lengths (the moments use all replicates; the
        # within-tree joint CDF / cross-moment ground truth needs only enough samples for a ~0.02 tolerance)
        jsfs_sample_cap = self._jsfs_sample_cap // self.n_threads

        def simulate_batch(seed: Optional[int]) -> dict:
            """
            Simulate statistics.

            :param seed: Random seed.
            :return: Statistics.
            """
            import msprime as ms
            import tskit

            # simulate trees
            g: Generator = ms.sim_ancestry(
                sequence_length=self.locus_config.n,
                recombination_rate=self.locus_config.recombination_rate,
                samples=samples,
                num_replicates=num_replicates,
                record_migrations=self.record_migration,
                demography=demography,
                model=model,
                ploidy=1,
                end_time=end_time,
                random_seed=seed
            )

            # initialize variables
            heights = np.zeros((self.locus_config.n, n_pops, num_replicates), dtype=float)
            total_branch_lengths = np.zeros((self.locus_config.n, n_pops, num_replicates), dtype=float)
            sfs = np.zeros((self.locus_config.n, n_pops, num_replicates, sample_size + 1), dtype=float)
            mutations = np.zeros((self.locus_config.n, n_pops, num_replicates, sample_size + 1), dtype=int)

            # joint SFS moment accumulator (non-central moments of orders 1, ..., jsfs_max_order)
            jsfs_acc = np.zeros((jsfs_max_order,) + jsfs_shape)

            # capped per-replicate joint SFS branch lengths for the within-tree joint ground truth
            jsfs_samples = np.zeros((min(num_replicates, jsfs_sample_cap),) + jsfs_shape) if compute_jsfs else None

            # iterate over trees and compute statistics
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):

                # map each sample to the index of its sampling population (deme of origin) for the joint SFS
                if compute_jsfs:
                    pop_of_leaf = {
                        u: name_to_index[ts.population(ts.node(u).population).metadata['name']]
                        for u in ts.samples()
                    }

                tree: tskit.Tree
                for j, tree in enumerate(self._expand_trees(ts)):

                    # TODO record_migration only appears to work for relatively simple scenarios
                    if self.record_migration:

                        lineages = np.array(list(samples.values()))
                        t_coal = ts.tables.nodes.time[sample_size:]
                        node = sample_size - 1
                        t_migration = ts.migrations_time
                        i_migration = 0
                        time = 0

                        # population state per leave
                        pop_states = {n: tree.population(n) for n in range(sample_size)}

                        # iterate over coalescence events
                        for coal_time in t_coal:

                            # iterate over migration events within this coalescence event
                            while i_migration < len(t_migration) and time < t_migration[i_migration] <= coal_time:
                                delta = t_migration[i_migration] - time

                                # update statistics
                                heights[j, :, i] += delta * lineages / sum(lineages)
                                total_branch_lengths[j, :, i] += delta * lineages

                                for n, pop in pop_states.items():
                                    sfs[j, pop, i, tree.get_num_leaves(n)] += delta

                                # update lineages with migrations
                                lineages[ts.migrations_source[i_migration]] -= 1
                                lineages[ts.migrations_dest[i_migration]] += 1
                                pop_states[ts.migrations_node[i_migration]] = ts.migrations_dest[i_migration]

                                i_migration += 1
                                time += delta

                            # remaining time to next coalescence event
                            delta = coal_time - time

                            # update statistics
                            heights[j, :, i] += delta * lineages / sum(lineages)
                            total_branch_lengths[j, :, i] += delta * lineages

                            for n, pop in pop_states.items():
                                sfs[j, pop, i, tree.get_num_leaves(n)] += delta

                            # reduce by number of coalesced lineages
                            lineages[tree.population(node + 1)] -= len(tree.get_children(node + 1)) - 1

                            # delete children from pop_states
                            [pop_states.__delitem__(n) for n in tree.get_children(node + 1)]

                            # add parent to pop_states
                            pop_states[node + 1] = tree.population(node + 1)

                            time += delta
                            node += 1

                    else:

                        heights[j, 0, i] = tree.time(tree.roots[0])
                        total_branch_lengths[j, 0, i] = tree.total_branch_length

                        for node in tree.nodes():
                            t = tree.get_branch_length(node)
                            n = tree.get_num_leaves(node)

                            sfs[j, 0, i, n] += t

                    # accumulate the joint SFS from the same tree (single locus only)
                    if compute_jsfs and j == 0:
                        jsfs_rep = np.zeros(jsfs_shape)

                        for node in tree.nodes():

                            # the root subtends all samples (monomorphic) and is skipped
                            if tree.parent(node) == -1:
                                continue

                            # count descendant samples by population (deme of origin)
                            vec = [0] * len(jsfs_shape)
                            for leaf in tree.leaves(node):
                                vec[pop_of_leaf[leaf]] += 1

                            if sum(vec) > 0:
                                jsfs_rep[tuple(vec)] += tree.get_branch_length(node)

                        for order in range(jsfs_max_order):
                            jsfs_acc[order] += jsfs_rep ** (order + 1)

                        if i < jsfs_sample_cap:
                            jsfs_samples[i] = jsfs_rep

                # simulate mutations if specified
                if self.simulate_mutations:

                    mts = ms.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)
                    tree = next(mts.trees())

                    for node in mts.mutations_node:
                        mutations[0, 0, i, tree.get_num_leaves(node)] += 1

            return dict(
                main=np.concatenate([[heights.T], [total_branch_lengths.T], sfs.T, mutations.T]),
                jsfs=jsfs_acc,
                jsfs_samples=jsfs_samples
            )

        # parallelize over threads
        batches = parallelize(
            func=simulate_batch,
            data=[self.seed + i if self.seed is not None else None for i in range(self.n_threads)],
            parallelize=self.parallelize,
            batch_size=num_replicates,
            desc="Simulating trees",
            dtype=object
        )

        # combine the per-replicate statistics across threads
        res = np.hstack([b['main'] for b in batches])

        # store results
        self.heights = res[0].T
        self.total_branch_lengths = res[1].T
        self.sfs_lengths = res[2:sample_size + 3].T
        self.mutations = res[sample_size + 3:].T.astype(int)

        # combine the joint SFS moments (summed over replicates) across threads and normalize to moments
        self.jsfs_moments = np.sum([b['jsfs'] for b in batches], axis=0) / n_total if compute_jsfs else None

        # combine the (capped) per-replicate joint SFS branch lengths across threads for the joint ground truth
        self.jsfs_samples = np.concatenate([b['jsfs_samples'] for b in batches]) if compute_jsfs else None

    @staticmethod
    def _expand_trees(ts: 'tskit.TreeSequence') -> Iterator['tskit.Tree']:
        """
        Expand tree sequence to `n` trees where `n` is the number of loci.

        :param ts: Tree sequence.
        :return: List of trees.
        """
        for tree in ts.trees():
            for _ in range(int(tree.length)):
                yield tree

    def _get_cached_times(self) -> np.ndarray:
        """
        Get cached times.

        """
        t_max = self.heights.sum(axis=1).max()

        return np.linspace(0, t_max, 100)

    def touch(self, **kwargs: dict):
        """
        Touch cached properties.

        :param kwargs: Additional keyword arguments.
        """
        self.simulate()

        t = self._get_cached_times()

        self.tree_height.touch(t)
        self.total_tree_height.touch(t)
        self.total_branch_length.touch(t)
        self.sfs.touch(t)
        self.fsfs.touch(t)

        # cache the within-tree joint distribution ground truth (cross-moment + joint CDF at marginal-quantile
        # points) so the full-replicate joint is serialized with the comparison and survives the subsequent drop().
        # The two-locus joint is cached by the dedicated two-locus fixture script (which does not call touch()).
        # all bin pairs (i <= j), including the self-pair (i, i): there R_a = R_b a.s., so the joint law is singular
        # on the diagonal but its CDF reduces exactly to the marginal at min(x, y) (handled in JointRewardDistribution)
        if self.lineage_config.n_pops == 1 and self.locus_config.n == 1:
            n = self.lineage_config.n
            self.sfs.cache_joint([(i, j) for i in range(1, n) for j in range(i, n)],
                                 [(0.4, 0.6), (0.6, 0.4), (0.7, 0.7)])

        # cache the joint SFS distribution (its moments were already accumulated by simulate() above) for
        # multi-population, single-locus scenarios, so it is serialized along with the comparison, together with the
        # within-tree joint distribution ground truth
        if self.lineage_config.n_pops > 1 and self.locus_config.n == 1:
            self.cache_jsfs_joint()

    def drop(self):
        """
        Drop simulated data.
        """
        self.heights = None
        self.total_branch_lengths = None
        self.sfs_lengths = None
        self.mutations = None

        # the moments are retained by the cached jsfs distribution (referenced before drop), so this only removes
        # the duplicate reference held on the coalescent
        self.jsfs_moments = None
        self.jsfs_samples = None

        self.tree_height.drop()
        self.total_tree_height.drop()
        self.total_branch_length.drop()
        self.sfs.drop()
        self.fsfs.drop()

        # caused problems when serializing
        self.demography = None

    @cached_property
    def tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """
        Tree height distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(
            self.heights,
            pops=self.demography.pop_names,
            locus_agg=lambda x: x.max(axis=0)
        )

    @cached_property
    def total_tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """
        Total tree height distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(self.heights, pops=self.demography.pop_names)

    @cached_property
    def total_branch_length(self) -> EmpiricalPhaseTypeDistribution:
        """
        Total branch length distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeDistribution(self.total_branch_lengths, pops=self.demography.pop_names)

    @cached_property
    def sfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Unfolded site-frequency spectrum distribution.
        """
        self.simulate()

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=self.sfs_lengths,
            mutations=self.mutations.T[1:-1].T,
            pops=self.demography.pop_names,
            sfs_dist=UnfoldedSFSDistribution
        )

    @cached_property
    def fsfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """
        Folded site-frequency spectrum distribution.
        """
        self.simulate()

        mid = (self.lineage_config.n + 1) // 2

        # fold SFS branch lengths
        lengths = self.sfs_lengths.copy().T
        lengths[:mid] += lengths[-mid:][::-1]
        lengths[-mid:] = 0

        # fold SFS mutations
        mutations = self.mutations.copy().T
        mutations[:mid] += mutations[-mid:][::-1]
        mutations = mutations[1:self.lineage_config.n // 2 + 1]

        return EmpiricalPhaseTypeSFSDistribution(
            branch_lengths=lengths.T,
            mutations=mutations.T,
            pops=self.demography.pop_names,
            sfs_dist=FoldedSFSDistribution
        )

    #: Highest moment order computed for the empirical joint SFS ground truth.
    _jsfs_max_order: int = 3

    #: Max number of per-replicate joint SFS branch lengths retained for the within-tree joint ground truth
    _jsfs_sample_cap: int = 200000

    @cached_property
    def jsfs(self) -> 'EmpiricalJointSFSDistribution':
        """
        Joint (multi-population) site-frequency spectrum ground truth, accumulated from the same simulated trees as
        the other statistics (see :meth:`simulate`). Returns an :class:`EmpiricalJointSFSDistribution` exposing
        ``mean``, ``m2``, ``m3`` and ``var`` as arrays of shape ``(n_0 + 1, ..., n_{P-1} + 1)``, matching
        :class:`JointSFSDistribution`. The descendant configuration of a branch is the number of its sample
        descendants from each population (its deme of origin). Only available for multi-population, single-locus
        scenarios.
        """
        self.simulate()

        if self.jsfs_moments is None:
            raise NotImplementedError(
                "The joint SFS is only available for multi-population, single-locus scenarios."
            )

        return EmpiricalJointSFSDistribution(moments=self.jsfs_moments, samples=self.jsfs_samples)

    def cache_jsfs_joint(self):
        """
        Cache the within-tree joint distribution ground truth on the joint SFS, then drop the per-replicate samples.
        The config pairs are the few descendant configurations carrying the most branch length (so the
        marginal-quantile evaluation points sit well away from any atom at 0), paired all-against-all (including the
        self-pair). The cross-moment and the joint CDF at marginal-quantile points are then serialized with the
        comparison and validated at test time via :meth:`Comparison.compare_stat` (the ``joint`` stat).
        """
        jsfs = self.jsfs
        mean = self.jsfs_moments[0]
        full = tuple(int(s) for s in self.lineage_config.lineages)
        configs = [c for c in np.ndindex(mean.shape) if 0 < sum(c) < sum(full) and c != full]
        top = sorted(configs, key=lambda c: mean[c], reverse=True)[:4]
        pairs = [(top[a], top[b]) for a in range(len(top)) for b in range(a, len(top))]
        jsfs.cache_joint(pairs, [(0.4, 0.6), (0.6, 0.4), (0.7, 0.7)])
        jsfs.drop()

    @cached_property
    def sfs2(self) -> 'EmpiricalTwoLocusSFSDistribution':
        """
        Two-locus SFS ground truth, simulated with msprime: two sites at recombination distance ``r`` (the two loci),
        the per-bin branch-length cross product averaged over replicates. Only available for two-locus, single-locus-
        sample scenarios. Returns an :class:`EmpiricalTwoLocusSFSDistribution` exposing ``mean`` as a
        :class:`~phasegen.spectrum.TwoLocusSFS`, matching :class:`TwoLocusSFSDistribution`.
        """
        import msprime as ms

        if self.locus_config.n != 2:
            raise NotImplementedError("The two-locus SFS is only available for two-locus scenarios.")

        n = self.lineage_config.n
        demography = self.demography.to_msprime()
        model = self.get_coalescent_model()

        out = np.zeros((n + 1, n + 1))
        lefts = np.zeros((self.num_replicates, n + 1))   # per-replicate locus-0 / locus-1 SFS branch lengths,
        rights = np.zeros((self.num_replicates, n + 1))  # retained for the joint distribution / cross-moments
        for rep, ts in enumerate(ms.sim_ancestry(
                samples=self.lineage_config.lineage_dict,
                sequence_length=2,
                recombination_rate=self.locus_config.recombination_rate,
                demography=demography,
                model=model,
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self.seed,
        )):
            t0, t1 = ts.at(0.5), ts.at(1.5)
            left = np.zeros(n + 1)
            right = np.zeros(n + 1)
            for nd in t0.nodes():
                if t0.parent(nd) != -1:
                    left[t0.num_samples(nd)] += t0.branch_length(nd)
            for nd in t1.nodes():
                if t1.parent(nd) != -1:
                    right[t1.num_samples(nd)] += t1.branch_length(nd)
            out += np.outer(left, right)
            lefts[rep] = left
            rights[rep] = right

        return EmpiricalTwoLocusSFSDistribution(out / self.num_replicates, left=lefts, right=rights)

    @cached_property
    def fst(self) -> float:
        r"""
        Hudson's :math:`F_{ST}` ground truth, simulated with msprime: ``1 - mean within-population branch diversity /
        mean between-population branch divergence``, averaged over replicate trees. Requires at least two populations,
        each with at least two sampled lineages. Matches :meth:`Coalescent.fst`.
        """
        import msprime as ms

        pops = self.demography.pop_names

        if len(pops) < 2:
            raise ValueError(f"F_ST requires at least two populations (got {len(pops)}).")

        within = np.zeros(self.num_replicates)
        between = np.zeros(self.num_replicates)

        for k, ts in enumerate(ms.sim_ancestry(
                samples=self.lineage_config.lineage_dict,
                sequence_length=1,
                demography=self.demography.to_msprime(),
                model=self.get_coalescent_model(),
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self.seed,
        )):
            sample_sets = [ts.samples(population=i) for i in range(len(pops))]

            # within-population diversity (only populations with at least two samples are informative)
            w = [ts.diversity(s, mode='branch') for s in sample_sets if len(s) >= 2]
            # between-population divergence over distinct population pairs
            b = [ts.divergence([sample_sets[i], sample_sets[j]], mode='branch')
                 for i in range(len(pops)) for j in range(i + 1, len(pops))
                 if len(sample_sets[i]) and len(sample_sets[j])]

            within[k] = np.mean(w)
            between[k] = np.mean(b)

        return float(1 - within.mean() / between.mean())

    def _branch_f_statistic(self, kind: str, pops: List[str]) -> float:
        """
        msprime branch-mode Patterson f-statistic ground truth (``f2``/``f3``/``f4``) over the given populations,
        averaged over replicate trees (tskit branch mode uses the same 2x pairwise-coalescence convention as the
        analytical :class:`Coalescent` f-statistics).
        """
        import msprime as ms

        names = self.demography.pop_names
        for pop in pops:
            if pop not in names:
                raise ValueError(f"Unknown population '{pop}'. Available populations: {names}.")
        idx = [names.index(pop) for pop in pops]

        values = np.zeros(self.num_replicates)

        for k, ts in enumerate(ms.sim_ancestry(
                samples=self.lineage_config.lineage_dict,
                sequence_length=1,
                demography=self.demography.to_msprime(),
                model=self.get_coalescent_model(),
                ploidy=1,
                num_replicates=self.num_replicates,
                random_seed=self.seed,
        )):
            sample_sets = [ts.samples(population=i) for i in idx]
            values[k] = getattr(ts, kind)(sample_sets, mode='branch')

        return float(values.mean())

    def f2(self, pop_0: str, pop_1: str) -> float:
        """msprime branch-mode ``f2`` ground truth. Matches :meth:`Coalescent.f2`."""
        return self._branch_f_statistic('f2', [pop_0, pop_1])

    def f3(self, pop_target: str, pop_0: str, pop_1: str) -> float:
        """msprime branch-mode ``f3`` ground truth. Matches :meth:`Coalescent.f3`."""
        return self._branch_f_statistic('f3', [pop_target, pop_0, pop_1])

    def f4(self, pop_0: str, pop_1: str, pop_2: str, pop_3: str) -> float:
        """msprime branch-mode ``f4`` ground truth. Matches :meth:`Coalescent.f4`."""
        return self._branch_f_statistic('f4', [pop_0, pop_1, pop_2, pop_3])

    def to_phasegen(self) -> Coalescent:
        """
        Convert to native phasegen coalescent.

        :return: phasegen coalescent.
        """
        return Coalescent(
            n=self.lineage_config,
            model=self.model,
            demography=self.demography,
            loci=self.locus_config,
            recombination_rate=self.locus_config.recombination_rate,
            end_time=self.end_time
        )

