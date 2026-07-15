"""
Empirical distributions estimated from simulated genealogies -- via msprime (:class:`MsprimeCoalescent`) or
PhaseGen's own trajectory sampler (:class:`SampledCoalescent`) -- together with the containers that compute
statistics from the sampled realisations.
"""

import itertools
import logging
from collections import defaultdict
from ..caching import cached_property, cache
from typing import Generator, List, Callable, Tuple, Dict, Iterator, Optional, Sequence, Type, TYPE_CHECKING
import numpy as np
from ..coalescent_models import StandardCoalescent, CoalescentModel, BetaCoalescent, DiracCoalescent
from ..demography import Demography
from ..expm import Backend
from ..lineage import LineageConfig
from ..locus import LocusConfig
from ..settings import Settings
from ..spectrum import SFS, SFS2, JointSFS, TwoLocusSFS
from ..utils import parallelize

from .base import DensityAwareDistribution, CumulativeDistributionFunction, DensityFunction, QuantileFunction
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

    def __init__(self, moments: np.ndarray, samples: np.ndarray = None) -> None:
        """
        Initialize the distribution.

        :param moments: Per-configuration (non-central) moments of orders ``1, 2, ...``, stacked along the first
            axis, i.e. an array of shape ``(max_order, n_0 + 1, ..., n_{P-1} + 1)``.
        :param samples: Optional per-replicate joint SFS branch lengths, shape ``(n_replicates, n_0 + 1, ...)``, used
            to pre-compute the within-tree joint surface ground truth (:meth:`cache_joint_surface`); dropped before
            serialization.
        """
        #: Non-central moments per descendant configuration, indexed by order minus one.
        self._moments: np.ndarray = np.asarray(moments)

        #: Per-replicate joint SFS branch lengths (samples-free after :meth:`cache_joint_surface`).
        self.samples: np.ndarray | None = None if samples is None else np.asarray(samples)

        #: Number of samples (retained after :meth:`drop`, so it is recorded in a serialized comparison).
        self.n_samples: Optional[int] = None if samples is None else np.asarray(samples).shape[0]

        #: Cached full-grid joint surface ground truth: ``[(config_a, config_b, xs, ys, cdf_grid, pdf_grid), ...]``.
        self._joint_surface: list = []

    def cache_joint_surface(self, pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], n_grid: int = 25,
                            q_max: float = 0.95) -> None:
        """Pre-compute, for each config pair, the empirical joint CDF and density over a 2D grid (the full-grid
        surface comparison ground truth). Mirrors :meth:`EmpiricalPhaseTypeSFSDistribution.cache_joint_surface` but
        indexed by descendant configuration."""
        s = self.samples
        n = s.shape[0]
        self._joint_surface = []
        for ca, cb in pairs:
            li, lj = s[(slice(None),) + tuple(ca)], s[(slice(None),) + tuple(cb)]
            xs = np.linspace(0.0, float(np.quantile(li, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(lj, q_max)), n_grid)
            cdf = ((li[:, None] <= xs[None, :]).astype(float).T @ (lj[:, None] <= ys[None, :]).astype(float)) / n
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._joint_surface.append((tuple(ca), tuple(cb), xs, ys, cdf, pdf))

    def drop(self) -> None:
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


class _EmpiricalCumulativeDistributionFunction(CumulativeDistributionFunction):  # pragma: no cover
    """The empirical CDF (interpolated step function of the sorted samples), read from the distribution's samples.
    Handles both a 1-D sample vector (a scalar distribution) and a 2-D per-bin matrix (a spectrum)."""

    def __call__(self, t) -> 'np.ndarray':
        # sort along the replicate axis (axis 0); for 2-D (per-bin) samples this must not be the default last axis,
        # which would sort across bins within a replicate and produce a meaningless ECDF
        samples = self._distribution.samples
        x = np.sort(samples, axis=0)
        y = np.arange(1, len(samples) + 1) / len(samples)

        if x.ndim == 1:
            return np.interp(t, x, y)

        if x.ndim == 2:
            return np.array([np.interp(t, x_, y) for x_ in x.T])

        raise ValueError("Samples must be 1 or 2 dimensional.")


class _EmpiricalQuantileFunction(QuantileFunction):  # pragma: no cover
    """The empirical quantile (sample quantile over the replicate axis; one column per bin for a spectrum)."""

    def __call__(self, q) -> 'np.ndarray':
        # over the replicate axis (axis 0); for 2-D (per-bin) samples this gives one quantile per bin (shape
        # ``(len(q), n_bins)`` for an array ``q``), as the default flattening would mix bins together
        return np.quantile(self._distribution.samples, q=q, axis=0)


class _EmpiricalDensityFunction(DensityFunction):  # pragma: no cover
    """
    The empirical density over a grid: the **cell-average** density of each cell of ``t``, that is, the fraction of
    replicates falling in the cell divided by the cell's width. The atom at 0 is excluded, so this estimates the
    continuous sub-density ``f(t), t > 0``, which integrates to ``P(R > 0)`` and so matches the analytic pdf (also
    atom-excluded) rather than spiking at the origin.

    A cell average, not a point estimate, because that is the only density functional a sample determines without a
    bandwidth. The comparison integrates the exact density over the *same* cells
    (:meth:`~phasegen.comparison.Comparison._cell_average`), so both sides are the same functional: the estimate
    carries no smoothing bias, and the discrepancy is Monte-Carlo noise alone, falling as ``1 / sqrt(n)``.

    That property is the point of it. Any pointwise estimate -- a histogram read at ``t``, a kernel, or the derivative
    of an interpolated ECDF -- compares a *smoothed* density against an unsmoothed one, and its bandwidth sets an
    ``O(h f')`` bias floor that more replicates do not lower. Measured against the exact pdf of an SFS bin, such an
    estimate is an order of magnitude further off and stops improving with the replicate count entirely.

    The cells are ``[t_i, t_i+1)``, the last one extended by the final spacing. Handles a 1-D sample vector (a scalar
    distribution) or a 2-D per-bin matrix (a spectrum).
    """

    def __call__(self, t, **kwargs) -> 'np.ndarray':
        samples = self._distribution.samples
        t = np.atleast_1d(np.asarray(t, dtype=float))

        if t.size < 2:
            raise ValueError("The empirical density is a cell average, so it needs a grid of at least two points.")

        edges = np.append(t, 2 * t[-1] - t[-2])
        widths = np.diff(edges)

        if samples.ndim == 1:
            return self._cell_density(samples, edges, widths)

        if samples.ndim == 2:
            return np.array([self._cell_density(s, edges, widths) for s in samples.T])

        raise ValueError("Samples must be 1 or 2 dimensional.")

    @staticmethod
    def _cell_density(samples: np.ndarray, edges: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """The cell-average density of one sample vector. Normalised by the *total* replicate count, not by the
        positive one, so the atom at 0 lowers the sub-density instead of being redistributed over the cells."""
        counts, _ = np.histogram(samples[samples > 0], bins=edges)

        return counts / samples.size / widths


class EmpiricalDistribution(DensityAwareDistribution):  # pragma: no cover
    """
    Probability distribution based on realisations.
    """
    # the cdf / pdf / quantile evaluation lives on these sample-based function objects; the distribution supplies the
    # ``samples`` they read (the per-bin spectrum case is handled by the same objects, on 2-D samples)
    _cdf_function = _EmpiricalCumulativeDistributionFunction
    _pdf_function = _EmpiricalDensityFunction
    _quantile_function = _EmpiricalQuantileFunction

    def __init__(self, samples: np.ndarray | list) -> None:
        """
        Create object.

        :param samples: 1-D array of samples.
        """
        super().__init__()

        self._cache = None

        #: Samples
        self.samples = np.array(samples, dtype=float)

        #: Number of samples (retained after :meth:`drop`, so it is recorded in a serialized comparison).
        self.n_samples: int = self.samples.shape[0]

        #: Standard error of each moment statistic (:meth:`cache_standard_errors`), retained after :meth:`drop`.
        self.standard_errors: dict = {}

    def touch(self, t: np.ndarray) -> None:
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super().touch()

        # probability grid for the quantile function (kept off the extreme tails, where the empirical quantile is
        # noisy and -- for SFS bins with an atom at 0 -- flat at 0 below the atom mass)
        q = np.linspace(0.05, 0.95, 50)

        self._cache = dict(
            t=t,
            cdf=self.cdf(t),
            pdf=self.pdf(t),
            q=q,
            quantile=self.quantile(q)
        )

        self.cache_standard_errors()

    #: Statistics :meth:`cache_standard_errors` estimates a standard error for.
    _STANDARD_ERROR_STATISTICS = ('mean', 'var', 'm2', 'm3', 'm4', 'cov', 'corr')

    def cache_standard_errors(self, n_blocks: int = 100) -> None:
        """
        Estimate and cache the standard error of each moment statistic, so that it survives :meth:`drop` and a
        consumer of the (samples-free) distribution can tell how much of a discrepancy against it is the distribution's
        own Monte-Carlo noise.

        The estimator splits the samples into ``n_blocks`` disjoint blocks and evaluates the statistic on each. The
        spread across blocks is the standard error at the *block* sample size, so the standard error at the full sample
        size is that spread divided by ``sqrt(n_blocks)``. This holds for any statistic, however nonlinear (a
        correlation, a fourth moment), which the closed forms do not: ``SE[var]`` needs the fourth central moment,
        ``SE[m4]`` the eighth, and the coalescent's rewards are heavy-tailed enough that assuming normality to dodge
        them is not an option.

        :param n_blocks: Number of blocks. The spread itself is estimated from ``n_blocks`` numbers, so its own
            relative error is about ``1 / sqrt(2 * n_blocks)``. Reduced for a sample too small to fill that many
            blocks (a conditional sub-sample, say); below two blocks no spread is defined and none is cached.
        """
        n_blocks = min(n_blocks, self.samples.shape[0] // 2)

        if n_blocks < 2:
            return

        blocks = self.samples[:self.samples.shape[0] // n_blocks * n_blocks]
        blocks = blocks.reshape(n_blocks, -1, *self.samples.shape[1:])

        # the base class' statistics are plain numpy; the subclasses only wrap the identical numerics in an SFS type
        stats = [EmpiricalDistribution(block) for block in blocks]

        self.standard_errors = {}
        for name in self._STANDARD_ERROR_STATISTICS:
            if name in ('cov', 'corr') and self.samples.ndim == 1:
                continue  # a 1-D sample has no covariance/correlation: corrcoef is the constant 1, SE a bogus 0
            values = np.array([np.asarray(getattr(s, name), dtype=float) for s in stats])
            self.standard_errors[name] = np.std(values, axis=0) / np.sqrt(n_blocks)

    def drop(self) -> None:
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


class EmpiricalSFSDistribution(EmpiricalDistribution):  # pragma: no cover
    """
    SFS probability distribution based on realisations.
    """

    def __init__(self, samples: np.ndarray | list) -> None:
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
    ) -> None:
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

        #: Cross-locus full-grid joint surface ground truth: ``[(l1, l2, xs, ys, cdf_grid, pdf_grid), ...]``.
        self._loci_joint_surface: list = []

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

    def touch(self, t: np.ndarray) -> None:
        """
        Touch all cached properties.

        :param t: Times to cache properties for.
        """
        super().touch(t)

        [d.touch(t) for d in self.demes.values()]
        [l.touch(t) for l in self.loci.values()]

    def drop(self) -> None:
        """
        Drop simulated samples.
        """
        super().drop()

        self._samples = None

        [d.drop() for d in self.demes.values()]
        [l.drop() for l in self.loci.values()]

    def cache_standard_errors(self, n_blocks: int = 100) -> None:
        """
        In addition to the scalar-total standard errors, block-estimate the standard error of the deme-deme and
        locus-locus covariance / correlation matrices that :attr:`demes` and :attr:`loci` expose (``demes.cov`` etc.),
        so the tolerance tuner has a real noise floor for those leaves rather than the scalar total's variance error.
        The matrices are keyed ``"demes.cov"`` / ``"demes.corr"`` / ``"loci.cov"`` / ``"loci.corr"``.
        """
        super().cache_standard_errors(n_blocks)

        if self._samples is None:
            return

        for key, data, fn in (
            ('demes.cov', self._samples.sum(axis=0), np.cov),    # (n_demes, n_rep), summed over loci
            ('demes.corr', self._samples.sum(axis=0), np.corrcoef),
            ('loci.cov', self._samples.sum(axis=1), np.cov),     # (n_loci, n_rep), summed over demes
            ('loci.corr', self._samples.sum(axis=1), np.corrcoef),
        ):
            se = self._matrix_block_standard_error(data, fn, n_blocks)
            if se is not None:
                self.standard_errors[key] = se

    @staticmethod
    def _matrix_block_standard_error(data: np.ndarray, fn, n_blocks: int) -> Optional[np.ndarray]:
        """Standard error of a matrix statistic (``np.cov`` / ``np.corrcoef``) of ``data`` (shape ``(series, reps)``),
        by the same block subsampling :meth:`EmpiricalDistribution.cache_standard_errors` uses: the spread of the
        per-block matrix divided by ``sqrt(n_blocks)``. Returns ``None`` when there is nothing to correlate (fewer
        than two series, e.g. a single deme or single locus) or too few replicates to block."""
        if data.ndim != 2 or data.shape[0] < 2:
            return None

        n_reps = data.shape[1]
        n_blocks = min(n_blocks, n_reps // 2)
        if n_blocks < 2:
            return None

        blocks = data[:, :n_reps // n_blocks * n_blocks].reshape(data.shape[0], n_blocks, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mats = np.array([fn(blocks[:, b, :]) for b in range(n_blocks)])

        return np.std(mats, axis=0) / np.sqrt(n_blocks)

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

    def _locus_samples(self, locus: int) -> np.ndarray:
        """Per-replicate accumulated reward at a single locus (summed over demes), matching :attr:`loci`."""
        return self._samples[locus].sum(axis=0)

    def cache_loci_joint_surface(self, pairs: List[Tuple[int, int]], n_grid: int = 25, q_max: float = 0.95) -> None:
        """Pre-compute, for each locus pair, the empirical cross-locus joint CDF and density over a 2D grid (the
        full-grid surface comparison ground truth). Mirrors :meth:`EmpiricalPhaseTypeSFSDistribution.cache_joint_surface`
        but indexed by locus."""
        self._loci_joint_surface = []
        for l1, l2 in pairs:
            a, b = self._locus_samples(l1), self._locus_samples(l2)
            n = len(a)
            xs = np.linspace(0.0, float(np.quantile(a, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(b, q_max)), n_grid)
            cdf = ((a[:, None] <= xs[None, :]).astype(float).T @ (b[:, None] <= ys[None, :]).astype(float)) / n
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._loci_joint_surface.append((int(l1), int(l2), xs, ys, cdf, pdf))


class _WindowedConditional(EmpiricalDistribution):  # pragma: no cover
    """
    The replicates a windowed conditional selected (see :meth:`EmpiricalJointRewardDistribution.conditional`), with a
    **local-linear** :attr:`mean`. Everything else -- the variance, the cdf, the quantile -- is the plain estimate over
    the window.

    Weighting the selected replicates equally would make the mean a Nadaraya-Watson estimator, which is ``O(h)``-biased
    wherever ``E[R_other | R_on = v]`` has slope in ``v``: the conditioning values are not symmetric inside the window,
    so the slope leaks in. A local-linear fit cancels that term.

    :param samples: The other reward over the selected replicates.
    :param offsets: The selected replicates' conditioning values, *centered* on the conditioning value.
    :param window: Half-width of the window, the scale the weights are taken on.
    """

    def __init__(self, samples: np.ndarray, offsets: np.ndarray, window: float) -> None:
        super().__init__(samples)

        #: Conditioning offsets ``R_on - value`` of the selected replicates.
        self._offsets = np.asarray(offsets, dtype=float)

        #: Half-width of the window.
        self._window = float(window)

    @cached_property
    def mean(self) -> float:
        """The local-linear estimate of ``E[R_other | R_on = value]``: the intercept, at the conditioning value, of a
        tricube-weighted least-squares line through the selected replicates. Falls back to the plain window mean for a
        degenerate fit (a zero-width window, or one whose conditioning values do not vary)."""
        x, y, h = self._offsets, self.samples, self._window

        if h <= 0 or x.size < 3:
            return float(np.mean(y))

        w = (1.0 - np.minimum(np.abs(x / h), 1.0) ** 3) ** 3
        sw, swx, swx2 = w.sum(), (w * x).sum(), (w * x * x).sum()
        det = sw * swx2 - swx ** 2

        if not np.isfinite(det) or abs(det) < 1e-300:
            return float(np.mean(y))

        return float((swx2 * (w * y).sum() - swx * (w * x * y).sum()) / det)


class EmpiricalJointRewardDistribution:  # pragma: no cover
    """
    Empirical counterpart of :class:`~phasegen.distributions.reward.JointRewardDistribution`: the sampled joint
    distribution of two accumulated rewards, built from the per-replicate samples and sliced into the 1D
    :meth:`marginal` and :meth:`conditional` distributions.

    .. warning::
        :meth:`marginal` is an ordinary sample estimate, but :meth:`conditional` is not. No replicate lands exactly
        on the conditioning value, so it keeps those in a *window* around it: an estimate of the conditional
        *averaged over the window*, not at the value. Widening it smears the conditional wherever it varies with the
        conditioning value, narrowing it leaves few replicates behind the estimate. A 2D sample therefore says far
        less about a conditional than it does about the marginals, and this is a rough check on the exact
        conditional, not a ground truth for it.
    """

    def __init__(self, samples_a: np.ndarray, samples_b: np.ndarray, label: str = None) -> None:
        """
        :param samples_a: Per-replicate realisations of the first reward.
        :param samples_b: Per-replicate realisations of the second reward.
        :param label: Optional human-readable label used in plot titles.
        """
        #: Per-replicate realisations of the two rewards.
        self._a = np.asarray(samples_a, dtype=float)
        self._b = np.asarray(samples_b, dtype=float)

        #: Optional human-readable label (e.g. ``"SFS bins (1, 2)"``).
        self.label = label

    def marginal(self, which: str = 'a') -> EmpiricalDistribution:
        """
        The empirical marginal distribution of the first reward (``which='a'``) or the second (``which='b'``).

        :param which: Which reward's marginal, ``'a'`` or ``'b'``.
        :return: The empirical marginal distribution.
        :raises ValueError: If ``which`` is not ``'a'`` or ``'b'``.
        """
        if which not in ('a', 'b'):
            raise ValueError("`which` must be 'a' or 'b'.")
        return EmpiricalDistribution(self._a if which == 'a' else self._b)

    def conditional(self, on: str = 'a', value: float = 0.0, window: float = None) -> '_WindowedConditional':
        """
        The empirical conditional distribution of the *other* reward given ``R_{on}`` close to ``value``, estimated
        from the replicates whose conditioning reward falls in a window around ``value``. The sampled counterpart of
        :meth:`~phasegen.distributions.reward.JointRewardDistribution.conditional`.

        :param on: Which reward to condition on, ``'a'`` or ``'b'``.
        :param value: The conditioning value.
        :param window: Half-width of the symmetric window, in the conditioning reward's units. Defaults to the
            smallest window around ``value`` holding at least ``max(200, n / 50)`` replicates, a data-adaptive
            bandwidth that keeps the estimate stable wherever ``value`` sits (narrower reduces bias, wider reduces
            noise).
        :return: The empirical distribution of the other reward over the selected replicates.
        :raises ValueError: If ``on`` is not ``'a'`` / ``'b'`` or no replicate falls in the window.
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")
        cond, other = (self._a, self._b) if on == 'a' else (self._b, self._a)
        distance = np.abs(cond - value)
        if window is None:
            k = min(max(200, cond.size // 50), cond.size - 1)
            window = float(np.partition(distance, k)[k])
        mask = distance <= window
        if not mask.any():
            raise ValueError(f"No samples within window {window:g} of {value:g}.")

        return _WindowedConditional(other[mask], cond[mask] - value, float(window))

    def conditional_on_atom(self, on: str = 'a') -> Tuple[float, EmpiricalDistribution]:
        """
        The empirical conditional distribution of the *other* reward given the **atom event** ``{R_{on} = 0}``, with
        the atom's own mass.

        Unlike :meth:`conditional`, this is not a window estimate and carries no bandwidth: the atom event has
        positive probability, so the replicates in which the conditioning reward is exactly zero *are* the
        conditioning set. It is an ordinary sample estimate of an ordinary conditional law, and so the one place a
        sample is a genuine ground truth for a conditional -- which is what makes it worth comparing the exact
        (``_AtomConditional``) path against.

        :param on: Which reward to condition on, ``'a'`` or ``'b'``.
        :return: The atom's mass ``P(R_{on} = 0)`` and the distribution of the other reward over those replicates.
        :raises ValueError: If ``on`` is not ``'a'`` / ``'b'``, or no replicate has the conditioning reward at zero.
        """
        if on not in ('a', 'b'):
            raise ValueError("`on` must be 'a' or 'b'.")

        cond, other = (self._a, self._b) if on == 'a' else (self._b, self._a)
        empty = cond == 0.0

        if not empty.any():
            raise ValueError(f"No replicate has R_{on} = 0, so the atom conditional cannot be estimated.")

        return float(empty.mean()), EmpiricalDistribution(other[empty])

    def cdf(self, x: float, y: float) -> float:
        """The empirical joint CDF ``P(R_a <= x, R_b <= y)``."""
        return float(((self._a <= x) & (self._b <= y)).mean())

    @property
    def mean(self) -> np.ndarray:
        """The pair of marginal means ``(E[R_a], E[R_b])``."""
        return np.array([self._a.mean(), self._b.mean()])

    def cov(self) -> float:
        """The empirical covariance of the two rewards."""
        return float(np.cov(self._a, self._b)[0, 1])

    def corr(self) -> float:
        """The empirical Pearson correlation of the two rewards."""
        return float(np.corrcoef(self._a, self._b)[0, 1])


class EmpiricalPhaseTypeSFSDistribution(EmpiricalPhaseTypeDistribution, TajimaSFSMixin):  # pragma: no cover
    """
    SFS phase-type distribution based on realisations.

    The per-bin (2-D samples) cdf / pdf / quantile evaluation is handled by the inherited ``_Empirical*`` function
    objects; the per-bin plotting is the bin-aware :meth:`_plot_per_bin`.
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
    ) -> None:
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

        #: Number of lineages
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

        #: Atom-conditional ground truth: ``[(i, j, on, mass, mean, xs, cdf), ...]``, see
        #: :meth:`cache_atom_conditional`. Survives :meth:`drop` and is serialized with the comparison.
        self._atom_conditional: list = []

        #: Cached windowed-conditional ground truth, see :meth:`cache_windowed_conditional`.
        self._windowed_conditional: list = []

    def _plot_per_bin(self, kind: str, ax, grid, n_points, show, file, clear, title, bins) -> 'plt.Axes':
        """
        Plot the per-bin empirical pdf / cdf / quantile (one curve per polymorphic SFS bin), the empirical
        counterpart of :meth:`SFSDistribution._plot_cdf` etc. The inherited (1D) plotters cannot be used because the
        SFS samples are per-bin (2D), so we draw each bin's column separately.
        """
        from ..visualization import Visualization
        import matplotlib.pyplot as plt

        samples = np.asarray(self.samples)
        if bins is None:
            bins = range(1, samples.shape[1] - 1)  # polymorphic bins (drop the monomorphic edges)
        per = [(i, EmpiricalDistribution(samples[:, i])) for i in bins]

        if ax is None:
            ax = plt.gca()
            if clear:
                ax.clear()

        if grid is None:
            qe = Settings.plot_endpoint_quantile
            grid = np.linspace(1.0 - qe, qe, n_points) if kind == 'quantile' \
                else np.linspace(0, max(d.quantile(qe) for _, d in per), n_points)

        # the empirical density is a cell average, so the grid *is* the binning: a plotting grid as fine as the
        # comparison's would leave a handful of replicates per cell and come out as noise. Coarsen it with the sample
        # size, and plot the cell averages at the cell centres, which is where they are unbiased
        grid_pdf = np.linspace(grid[0], grid[-1], int(np.clip(np.sqrt(samples.shape[0]), 20, 100)))
        centres = grid_pdf + 0.5 * (grid_pdf[1] - grid_pdf[0])

        ylabel = {'cdf': 'F(x)', 'pdf': 'f(x)', 'quantile': 'quantile'}[kind]
        xlabel = 'q' if kind == 'quantile' else 't'
        for k, (i, d) in enumerate(per):
            x = grid
            if kind == 'cdf':
                y = d.cdf(grid)
            elif kind == 'pdf':
                x, y = centres, d.pdf(grid_pdf)
            else:
                y = np.array([d.quantile(float(q)) for q in grid])
            Visualization.plot(ax=ax, x=x, y=y, xlabel=xlabel, ylabel=ylabel, label=str(i), file=file,
                               show=(k == len(per) - 1 and show), clear=clear, title=title)
        return ax

    def _plot_cdf(self, ax=None, t=None, bins=None, n_points=200, show=True, file=None, clear=True,
                  title='SFS bin CDFs') -> 'plt.Axes':
        """Plot the empirical CDF of every (polymorphic) SFS bin at once."""
        return self._plot_per_bin('cdf', ax, t, n_points, show, file, clear, title, bins)

    def _plot_pdf(self, ax=None, t=None, bins=None, n_points=200, show=True, file=None, clear=True,
                  title='SFS bin PDFs', **kwargs) -> 'plt.Axes':
        """Plot the empirical PDF of every (polymorphic) SFS bin at once."""
        return self._plot_per_bin('pdf', ax, t, n_points, show, file, clear, title, bins)

    def _plot_quantile(self, ax=None, q=None, bins=None, n_points=99, show=True, file=None, clear=True,
                       title='SFS bin quantile functions') -> 'plt.Axes':
        """Plot the empirical quantile function of every (polymorphic) SFS bin at once."""
        return self._plot_per_bin('quantile', ax, q, n_points, show, file, clear, title, bins)

    def drop(self) -> None:
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

    def cache_joint_surface(self, pairs: List[Tuple[int, int]], n_grid: int = 25, q_max: float = 0.95) -> None:
        """
        Pre-compute, for each requested bin pair, the empirical joint CDF and density over a 2D grid (spanning each
        bin's support up to its ``q_max`` quantile), for the full-grid surface comparison. The density is the mixed
        second difference of the CDF grid (grid spacing = bandwidth). Stored as
        ``self._joint_surface = [(i, j, xs, ys, cdf_grid, pdf_grid), ...]`` and serialized with the comparison.
        """
        s = self.samples
        n = s.shape[0]
        self._joint_surface = []
        for i, j in pairs:
            li, lj = s[:, i], s[:, j]
            xs = np.linspace(0.0, float(np.quantile(li, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(lj, q_max)), n_grid)
            # empirical joint CDF on the grid: P(L_i <= x_a, L_j <= y_b) = (1/N) sum_r 1{li_r<=x_a} 1{lj_r<=y_b}
            a = (li[:, None] <= xs[None, :]).astype(float)  # (N, X)
            b = (lj[:, None] <= ys[None, :]).astype(float)  # (N, Y)
            cdf = (a.T @ b) / n                             # (X, Y)
            # density via the mixed second difference of the CDF surface (no separate bandwidth needed)
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._joint_surface.append((int(i), int(j), xs, ys, cdf, pdf))

    def cache_atom_conditional(self, pairs: List[Tuple[int, int]], n_grid: int = 100) -> None:
        """
        Pre-compute, for each requested bin pair and each conditioning axis, the empirical **atom conditional**: the
        mass of ``{L_on = 0}`` and the distribution of the other bin's length over exactly those replicates.

        This is the one conditional a sample pins exactly -- the atom event has positive probability, so the
        conditioning set needs no window and carries no bandwidth bias (see
        :meth:`EmpiricalJointRewardDistribution.conditional_on_atom`). Nothing else validates the ``value = 0``
        branch: every conditional check places its conditioning values at ``quantile(p0 + (1 - p0) u)``, strictly
        *above* the atom.

        Each conditional is cached as an ordinary :class:`EmpiricalDistribution`, touched on its own support and then
        dropped, so its moments and its cdf / pdf / quantile grids are compared by the same machinery as every other
        distribution rather than by a bespoke path.

        Stored as ``self._atom_conditional = [(i, j, on, mass, dist), ...]`` and serialized with the comparison.

        An axis whose conditioning bin is never empty has no atom to condition on, and is recorded with a zero mass
        and no distribution rather than omitted. Omitting it would make an unwired axis indistinguishable from a
        fixture predating this cache, and the zero mass is itself worth asserting: a bin that cannot be empty (for
        ``n = 4`` every tree has a doubleton branch) must carry no atom in the exact joint either.

        :param pairs: Bin pairs to cache.
        :param n_grid: Points of the cdf / pdf grid each conditional is cached on.
        """
        s = np.asarray(self.samples)
        self._atom_conditional = []

        for i, j in pairs:
            jd = EmpiricalJointRewardDistribution(s[:, i], s[:, j])
            for on in ('a', 'b'):
                try:
                    mass, dist = jd.conditional_on_atom(on)
                except ValueError:
                    self._atom_conditional.append((int(i), int(j), on, 0.0, None))
                    continue

                dist.touch(np.linspace(0.0, float(np.max(dist.samples)), n_grid))
                dist.drop()
                self._atom_conditional.append((int(i), int(j), on, mass, dist))

    def cache_windowed_conditional(self, specs: List[tuple], n_grid: int = 500, q_max: float = 0.999) -> None:
        """
        Pre-compute, for each requested conditioning window, the empirical conditional of the other bin's length over
        exactly the replicates whose conditioning bin falls in that window: the mean (with its standard error) and the
        CDF over a grid.

        The window is *the* thing being cached, and it is deliberately not corrected for. What a sample measures is
        the conditional averaged over the window, and the comparison averages the exact conditional over the same
        window (:meth:`~phasegen.distributions.reward.JointRewardDistribution.window_average`) rather than evaluating
        it at the centre, so the two sides are the same functional. That is what makes this the only ground truth the
        nested conditional has away from the atom, and it is why no bandwidth correction is applied here.

        The mean's standard error is cached because the mean has no other floor: both sides measure the same
        functional, so once the window bias is gone what separates them is the sampling noise of the window alone.
        The comparison reports the mean's deviation in standard errors, which keeps that check independent of the
        replicate count instead of silently loosening with it. The CDF is compared as a plain absolute difference
        instead: its per-point binomial error collapses in the tails, where a sigma would explode on an agreement that
        is in fact excellent.

        The CDF grid is dense and runs to ``q_max`` of the window's own samples, because on the exact side it is free:
        the conditional's cosine grid is built regardless, and reading it at 500 points rather than 50 costs an
        interpolation. A coarse grid stopping at the 99th percentile would simply discard the tail, and with it any
        chance of the check seeing a discrepancy there.

        Stored as ``self._windowed_conditional = [(i, j, on, v, h, n_win, mean, mean_se, ys, cdf), ...]``.

        :param specs: ``(i, j, on, value, half_width)`` windows to cache, the values fixed by the exact marginal.
        :param n_grid: Points of the CDF grid.
        :param q_max: Quantile of the windowed samples the grid runs to.
        :raises ValueError: If a window holds no replicates at all.
        """
        s = np.asarray(self.samples)
        self._windowed_conditional = []

        for i, j, on, v, h in specs:
            cond, other = (s[:, i], s[:, j]) if on == 'a' else (s[:, j], s[:, i])
            sel = other[np.abs(cond - v) <= h]

            if sel.size == 0:
                raise ValueError(
                    f"No replicate of bins ({i}, {j}) falls in the conditioning window R_{on} = {v:g} +- {h:g}, so "
                    f"the windowed conditional cannot be estimated there."
                )

            ys = np.linspace(0.0, float(np.quantile(sel, q_max)), n_grid)

            # from the sorted sample, not an (n_win x n_grid) boolean matrix, which at this resolution would be
            # hundreds of millions of entries
            cdf = np.searchsorted(np.sort(sel), ys, side='right') / sel.size

            self._windowed_conditional.append(
                (int(i), int(j), on, float(v), float(h), int(sel.size), float(sel.mean()),
                 float(sel.std() / np.sqrt(sel.size)), ys, cdf)
            )

    def joint_distribution(self, i: int, j: int) -> 'EmpiricalJointRewardDistribution':
        """
        The empirical joint distribution of the branch lengths of bins ``i`` and ``j``, from the per-replicate
        samples — the sampled counterpart of
        :meth:`~phasegen.distributions.spectra.SFSDistribution.joint_distribution`, exposing the same
        :meth:`~EmpiricalJointRewardDistribution.marginal` and :meth:`~EmpiricalJointRewardDistribution.conditional`
        slices for a sanity check against the exact joint.

        :param i: First frequency class.
        :param j: Second frequency class.
        :return: The empirical joint reward distribution of ``(L_i, L_j)``.
        :raises ValueError: If the per-replicate samples have been dropped.
        """
        if self.samples is None:
            raise ValueError("The per-replicate samples have been dropped; joint_distribution needs them.")
        s = np.asarray(self.samples)
        return EmpiricalJointRewardDistribution(s[:, i], s[:, j], label=f"SFS bins ({i}, {j})")

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

    def __init__(self, mean: np.ndarray, left: np.ndarray = None, right: np.ndarray = None) -> None:
        """
        :param mean: The simulated mean two-locus SFS array.
        :param left: Optional per-replicate locus-0 SFS branch lengths ``(num_replicates, n + 1)`` (for the joint
            distribution / cross-moment tracking). Dropped on :meth:`drop` and absent from a serialized comparison.
        :param right: Optional per-replicate locus-1 SFS branch lengths.
        """
        self._mean = np.asarray(mean)
        self._left = None if left is None else np.asarray(left)
        self._right = None if right is None else np.asarray(right)

        #: Number of samples (retained after :meth:`drop`, so it is recorded in a serialized comparison).
        self.n_samples: Optional[int] = None if left is None else np.asarray(left).shape[0]

    @property
    def mean(self) -> TwoLocusSFS:
        """Mean two-locus SFS."""
        return TwoLocusSFS(self._mean)

    def drop(self) -> None:
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

    def cache_joint_surface(self, pairs: List[Tuple[int, int]], n_grid: int = 25, q_max: float = 0.95) -> None:
        """Pre-compute, for each cross-locus bin pair ``(i, j)`` (locus-0 class i, locus-1 class j), the empirical
        joint CDF and density over a 2D grid (the full-grid surface comparison ground truth). Same structure as
        :meth:`EmpiricalPhaseTypeSFSDistribution.cache_joint_surface`, indexed by the two loci's frequency classes."""
        n = self._left.shape[0]
        self._joint_surface = []
        for i, j in pairs:
            li, rj = self._left[:, i], self._right[:, j]
            xs = np.linspace(0.0, float(np.quantile(li, q_max)), n_grid)
            ys = np.linspace(0.0, float(np.quantile(rj, q_max)), n_grid)
            cdf = ((li[:, None] <= xs[None, :]).astype(float).T @ (rj[:, None] <= ys[None, :]).astype(float)) / n
            pdf = np.gradient(np.gradient(cdf, xs, axis=0), ys, axis=1)
            self._joint_surface.append((int(i), int(j), xs, ys, cdf, pdf))


class _ReplicateStatistic:  # pragma: no cover
    """
    A per-replicate statistic accumulated in a **single pass** over the simulated tree sequences (the observer
    pattern: :meth:`MsprimeCoalescent.simulate` iterates the trees once and feeds every registered statistic, so
    each statistic is a self-contained component and adding one does not touch the simulation loop).

    A statistic allocates its own per-replicate storage and updates from each tree (:meth:`process_tree`, called for
    every locus ``j`` of replicate ``i``) and/or each whole replicate (:meth:`process_replicate`, tree-sequence
    level). Both default to no-ops, so a statistic implements only the hook it needs.
    """

    def process_tree(self, i: int, j: int, tree, ts, ctx: dict) -> None:
        """Update from the locus-``j`` tree of replicate ``i`` (``ctx`` carries shared per-replicate data)."""

    def process_replicate(self, i: int, ts, ctx: dict, seed) -> None:
        """Update from the whole replicate ``i`` (the tree sequence ``ts``)."""


class _TreeStatistics(_ReplicateStatistic):  # pragma: no cover
    """Tree height, total branch length and SFS read directly from each tree: the root time, the tree's total branch
    length, and per-node branch length binned by descendant count (population index 0; no migration recording)."""

    def __init__(self, n_loci: int, n_pops: int, num_replicates: int, sample_size: int) -> None:
        self.heights = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.total_branch_lengths = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.sfs = np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=float)

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        self.heights[j, 0, i] = tree.time(tree.roots[0])
        self.total_branch_lengths[j, 0, i] = tree.total_branch_length

        for node in tree.nodes():
            t = tree.get_branch_length(node)
            n = tree.get_num_leaves(node)

            self.sfs[j, 0, i, n] += t


class _MigrationTreeStatistics(_ReplicateStatistic):  # pragma: no cover
    """Tree height, total branch length and **per-deme** SFS reconstructed from the recorded migration history, so
    each quantity is attributed to the deme a lineage occupies through time (walking the coalescence and migration
    events). Only validated for relatively simple scenarios."""

    def __init__(self, n_loci: int, n_pops: int, num_replicates: int, sample_size: int, samples: dict) -> None:
        self.heights = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.total_branch_lengths = np.zeros((n_loci, n_pops, num_replicates), dtype=float)
        self.sfs = np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=float)
        self._samples = samples
        self._sample_size = sample_size

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        samples, sample_size = self._samples, self._sample_size

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
                self.heights[j, :, i] += delta * lineages / sum(lineages)
                self.total_branch_lengths[j, :, i] += delta * lineages

                for n, pop in pop_states.items():
                    self.sfs[j, pop, i, tree.get_num_leaves(n)] += delta

                # update lineages with migrations
                lineages[ts.migrations_source[i_migration]] -= 1
                lineages[ts.migrations_dest[i_migration]] += 1
                pop_states[ts.migrations_node[i_migration]] = ts.migrations_dest[i_migration]

                i_migration += 1
                time += delta

            # remaining time to next coalescence event
            delta = coal_time - time

            # update statistics
            self.heights[j, :, i] += delta * lineages / sum(lineages)
            self.total_branch_lengths[j, :, i] += delta * lineages

            for n, pop in pop_states.items():
                self.sfs[j, pop, i, tree.get_num_leaves(n)] += delta

            # reduce by number of coalesced lineages
            lineages[tree.population(node + 1)] -= len(tree.get_children(node + 1)) - 1

            # delete children from pop_states
            [pop_states.__delitem__(n) for n in tree.get_children(node + 1)]

            # add parent to pop_states
            pop_states[node + 1] = tree.population(node + 1)

            time += delta
            node += 1


class _JointSFSStatistics(_ReplicateStatistic):  # pragma: no cover
    """The joint (per-deme-of-origin) SFS branch lengths from the first locus' tree: the non-central moments (orders
    1..``max_order``) over all replicates plus a capped subset of per-replicate values for the within-tree joint
    ground truth. Single-locus only (accumulated from the ``j == 0`` tree)."""

    def __init__(self, num_replicates: int, max_order: int, shape: tuple, sample_cap: int) -> None:
        self.jsfs_acc = np.zeros((max_order,) + shape)
        self.jsfs_samples = np.zeros((min(num_replicates, sample_cap),) + shape)
        self._max_order = max_order
        self._shape = shape
        self._cap = sample_cap

    def process_tree(self, i, j, tree, ts, ctx) -> None:
        if j != 0:  # accumulated from the first locus' tree only
            return

        pop_of_leaf = ctx['pop_of_leaf']
        jsfs_rep = np.zeros(self._shape)

        for node in tree.nodes():

            # the root subtends all samples (monomorphic) and is skipped
            if tree.parent(node) == -1:
                continue

            # count descendant samples by population (deme of origin)
            vec = [0] * len(self._shape)
            for leaf in tree.leaves(node):
                vec[pop_of_leaf[leaf]] += 1

            if sum(vec) > 0:
                jsfs_rep[tuple(vec)] += tree.get_branch_length(node)

        for order in range(self._max_order):
            self.jsfs_acc[order] += jsfs_rep ** (order + 1)

        if i < self._cap:
            self.jsfs_samples[i] = jsfs_rep


class _MutationStatistics(_ReplicateStatistic):  # pragma: no cover
    """The mutation-count SFS: drop mutations on the replicate's tree sequence at the configured rate and bin them by
    the number of leaves the carrying node subtends (population index 0, single locus)."""

    def __init__(self, n_loci: int, n_pops: int, num_replicates: int, sample_size: int, mutation_rate: float) -> None:
        self.mutations = np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=int)
        self._rate = mutation_rate

    def process_replicate(self, i, ts, ctx, seed) -> None:
        import msprime as ms

        # draw mutations with a per-replicate seed: reusing the single batch seed across all replicates freezes the
        # mutation randomness (the per-replicate count is then a deterministic function of the tree rather than a
        # fresh Poisson draw), which biases the mutational-configuration distribution. The batch seeds are spaced by
        # one (``self.seed + thread``), so a large prime offset keeps the per-replicate seeds collision-free and
        # reproducible.
        rep_seed = None if seed is None else int((seed * 1_000_003 + i) % (2 ** 31 - 1)) + 1

        mts = ms.sim_mutations(ts, rate=self._rate, random_seed=rep_seed)
        tree = next(mts.trees())

        for node in mts.mutations_node:
            self.mutations[0, 0, i, tree.get_num_leaves(node)] += 1


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
    ) -> None:
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
    def simulate(self) -> None:
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
            Simulate one batch of replicates, accumulating every requested statistic in a **single pass** over the
            tree sequences via self-contained :class:`_ReplicateStatistic` components.

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

            # the per-statistic accumulators this scenario needs; the tree-height / total-branch-length / SFS triple
            # is recorded either directly from each tree or, with migration recording, from the migration history
            n_loci = self.locus_config.n
            tree_stats = (_MigrationTreeStatistics(n_loci, n_pops, num_replicates, sample_size, samples)
                          if self.record_migration
                          else _TreeStatistics(n_loci, n_pops, num_replicates, sample_size))
            jsfs_stats = (_JointSFSStatistics(num_replicates, jsfs_max_order, jsfs_shape, jsfs_sample_cap)
                          if compute_jsfs else None)
            mutation_stats = (_MutationStatistics(n_loci, n_pops, num_replicates, sample_size, self.mutation_rate)
                              if self.simulate_mutations else None)
            stats = [s for s in (tree_stats, jsfs_stats, mutation_stats) if s is not None]

            # iterate over the tree sequences once, feeding every statistic
            ts: tskit.TreeSequence
            for i, ts in enumerate(g):

                # map each sample to the index of its sampling population (deme of origin) for the joint SFS
                ctx = {}
                if compute_jsfs:
                    ctx['pop_of_leaf'] = {
                        u: name_to_index[ts.population(ts.node(u).population).metadata['name']]
                        for u in ts.samples()
                    }

                tree: tskit.Tree
                for j, tree in enumerate(self._expand_trees(ts)):
                    for stat in stats:
                        stat.process_tree(i, j, tree, ts, ctx)

                for stat in stats:
                    stat.process_replicate(i, ts, ctx, seed)

            # the mutations / jSFS arrays default to zeros / None when not requested (kept in the return layout for
            # the cross-thread aggregation in ``simulate``)
            mutations = (mutation_stats.mutations if mutation_stats is not None
                         else np.zeros((n_loci, n_pops, num_replicates, sample_size + 1), dtype=int))
            jsfs_acc = jsfs_stats.jsfs_acc if jsfs_stats is not None else np.zeros((jsfs_max_order,) + jsfs_shape)
            jsfs_samples = jsfs_stats.jsfs_samples if jsfs_stats is not None else None

            return dict(
                main=np.concatenate([[tree_stats.heights.T], [tree_stats.total_branch_lengths.T],
                                     tree_stats.sfs.T, mutations.T]),
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

    @staticmethod
    def _get_cached_times(dist: 'EmpiricalPhaseTypeDistribution') -> np.ndarray:
        """
        The grid a distribution's curves are cached on: **its own** support, from 0 up to the largest value it
        sampled. Taken from the distribution's reduced (per-replicate) samples, not from the raw arrays, because the
        reduction differs per distribution -- the tree height takes the *maximum* over loci, the total branch length
        the *sum*.

        Each distribution must get its own grid rather than share the tree height's. They live on different scales
        (the total branch length exceeds the tree height by roughly ``2 H_{n-1}``), so a shared grid runs one of them
        off its own support, where both the sampled and the exact curve are zero and the comparison passes while
        asserting nothing.

        :param dist: The distribution whose curves are to be cached.
        :return: The grid.
        """
        return np.linspace(0, float(np.max(dist.samples)), 100)

    def touch(self, **kwargs: dict) -> None:
        """
        Touch cached properties.

        :param kwargs: Additional keyword arguments.
        """
        self.simulate()

        t = self._get_cached_times(self.tree_height)

        self.tree_height.touch(t)
        self.total_tree_height.touch(self._get_cached_times(self.total_tree_height))
        self.total_branch_length.touch(self._get_cached_times(self.total_branch_length))
        self.sfs.touch(t)
        self.fsfs.touch(t)

        # cache the cross-locus joint surface ground truth (per-locus tree height / total branch length at the two
        # loci, separated by recombination) for two-locus scenarios, so it is serialized with the comparison and
        # survives the subsequent drop(). The single pair (0, 1) over a full grid. The within-tree (single-locus and
        # multi-population) joint surfaces are cached separately by :meth:`Comparison.cache_ground_truth` from the
        # configured pairwise surface pairs.
        if self.locus_config.n == 2:
            for dist in (self.tree_height, self.total_branch_length):
                dist.cache_loci_joint_surface([(0, 1)])  # full-grid cross-locus surface ground truth

    def drop(self) -> None:
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


class SampledCoalescent(AbstractCoalescent):  # pragma: no cover
    """
    PhaseGen-sampled empirical coalescent: the same per-statistic distributions as :class:`MsprimeCoalescent`, but
    estimated from PhaseGen's own trajectory sampler (:meth:`PhaseTypeDistribution._sample`) rather than msprime.
    Used by :class:`~phasegen.comparison.Comparison` to validate the sampler against the exact analytic
    :class:`Coalescent`. The sampled realization is frozen into the comparison fixture at creation time; the
    per-statistic seeds make it reproducible and independent of access order.

    .. warning::
        Each statistic is sampled in its **own** simulation run, so different statistics come from **different
        genealogies**. Within one statistic the samples are coherent (the SFS bins of a single ``sfs`` draw share
        their trajectories, so their joints, covariances and correlations are valid), but pairing the raw ``.samples``
        of *two* statistics is not meaningful -- ``corrcoef(sc.tree_height.samples, sc.total_branch_length.samples)``
        is ~0 where the truth is ~1.
    """

    #: Per-statistic seed offsets so each distribution is sampled reproducibly and independently of access order.
    _seed_offsets = dict(tree_height=0, total_branch_length=1, sfs=2, fsfs=3, jsfs=4, sfs2=5)

    def __init__(self, coalescent: Coalescent, n_samples: int = 10000, seed: int = None) -> None:
        """
        :param coalescent: The analytic coalescent to sample from.
        :param n_samples: Number of trajectories to simulate per statistic.
        :param seed: Random seed.
        """
        # adopt the wrapped coalescent's configuration: this satisfies the AbstractCoalescent contract and retains
        # the config after the analytic coalescent is dropped (Comparison / serialization need it)
        super().__init__(
            n=coalescent.lineage_config,
            model=coalescent.model,
            demography=coalescent.demography,
            loci=coalescent.locus_config,
            end_time=coalescent.end_time
        )

        self._coalescent: Optional[Coalescent] = coalescent

        #: Number of trajectories sampled per statistic.
        self.n_samples: int = n_samples

        #: Random seed.
        self.seed: Optional[int] = seed

    def _to_empirical(self, name: str):
        """Sample the named analytic distribution into its empirical counterpart, seeded reproducibly."""
        if self.seed is not None:
            np.random.seed(self.seed + self._seed_offsets[name])

        return getattr(self._coalescent, name).to_empirical(self.n_samples)

    @cached_property
    def tree_height(self) -> EmpiricalPhaseTypeDistribution:
        """Sampled tree height distribution."""
        return self._to_empirical('tree_height')

    @cached_property
    def total_branch_length(self) -> EmpiricalPhaseTypeDistribution:
        """Sampled total branch length distribution."""
        return self._to_empirical('total_branch_length')

    @cached_property
    def sfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """Sampled unfolded site-frequency spectrum distribution."""
        return self._to_empirical('sfs')

    @cached_property
    def fsfs(self) -> EmpiricalPhaseTypeSFSDistribution:
        """Sampled folded site-frequency spectrum distribution."""
        return self._to_empirical('fsfs')

    @cached_property
    def jsfs(self) -> EmpiricalJointSFSDistribution:
        """Sampled joint (multi-population) site-frequency spectrum distribution."""
        return self._to_empirical('jsfs')

    @cached_property
    def sfs2(self) -> EmpiricalTwoLocusSFSDistribution:
        """Sampled two-locus site-frequency spectrum distribution."""
        return self._to_empirical('sfs2')

    def _get_cached_times(self) -> np.ndarray:
        """Grid for caching the empirical curve (cdf/pdf/quantile) ground truth, as in :class:`MsprimeCoalescent`."""
        t_max = float(np.max(self.tree_height.samples))

        return np.linspace(0, t_max, 100)

    def touch(self, **kwargs: dict) -> None:
        """Build and cache the empirical distributions (so the cached stats/surfaces survive :meth:`drop` and are
        serialized with the comparison)."""
        t = self._get_cached_times()

        self.tree_height.touch(t)
        self.total_branch_length.touch(t)

        # the single-locus site-frequency spectra (undefined for multiple loci, where ``sfs2`` is used instead)
        if self.locus_config.n == 1:
            self.sfs.touch(t)
            self.fsfs.touch(t)

            # multi-population: the joint SFS
            if len(self.lineage_config.pop_names) > 1:
                _ = self.jsfs

        # two loci: the cross-locus joint surface ground truth, and the two-locus SFS (single-population only, as in
        # the analytic two-locus block-counting state space)
        if self.locus_config.n == 2:
            for dist in (self.tree_height, self.total_branch_length):
                dist.cache_loci_joint_surface([(0, 1)])
            if len(self.lineage_config.pop_names) == 1:
                _ = self.sfs2

    def drop(self) -> None:
        """Drop the per-sample data and the analytic coalescent; the cached stats and surfaces are retained."""
        for name in ('tree_height', 'total_branch_length', 'sfs', 'fsfs', 'jsfs', 'sfs2'):
            if name in self.__dict__:
                self.__dict__[name].drop()

        self._coalescent = None
        self.demography = None

