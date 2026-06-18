"""Distribution base classes and marginal (per-deme / per-locus) views."""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from ..caching import cached_property
from typing import Callable, Iterator, Sequence, TYPE_CHECKING
import numpy as np
from ..expm import Backend
from ..rewards import DemeReward, LocusReward, CombinedReward
from ..settings import Settings

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .phase_type import PhaseTypeDistribution

expm = Backend.expm
logger = logging.getLogger('phasegen')


def adaptive_grid(f, a: float, b: float, n_init: int = 9, tol: float = None, max_points: int = None):
    """
    Adaptively sample a scalar function ``f`` on ``[a, b]``, concentrating evaluations where the curve bends.

    Starts from a coarse uniform grid and repeatedly bisects any interval whose midpoint value deviates from the
    straight chord between its endpoints by more than ``tol`` times the function's range. For an expensive ``f`` (the
    per-point de Hoog inversion) this hits a given visual accuracy with far fewer evaluations than a uniform grid --
    e.g. it resolves the near-zero atom spike of an SFS bin density that a uniform grid would miss.

    :param f: Scalar function to sample (called as ``f(x)`` for a float ``x``).
    :param a: Left endpoint.
    :param b: Right endpoint.
    :param n_init: Number of initial uniform points (>= 2).
    :param tol: Relative deviation tolerance; defaults to :attr:`Settings.plot_adaptive_tol`.
    :param max_points: Maximum number of evaluations; defaults to :attr:`Settings.plot_n_grid`.
    :return: Sorted ``(x, y)`` arrays.
    """
    from collections import deque

    tol = Settings.plot_adaptive_tol if tol is None else tol
    max_points = Settings.plot_n_grid if max_points is None else max_points

    xs = list(np.linspace(a, b, n_init))
    ys = [float(f(x)) for x in xs]
    thr = tol * max(max(ys) - min(ys), 1e-300)

    # work queue of intervals; bisect those whose midpoint departs from the chord by more than the threshold
    stack = deque((xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(xs) - 1))
    ex, ey = [], []
    while stack and len(xs) + len(ex) < max_points:
        xl, yl, xr, yr = stack.popleft()
        xm = 0.5 * (xl + xr)
        ym = float(f(xm))
        ex.append(xm)
        ey.append(ym)
        if abs(ym - 0.5 * (yl + yr)) > thr:
            stack.append((xl, yl, xm, ym))
            stack.append((xm, ym, xr, yr))

    x = np.array(xs + ex)
    y = np.array(ys + ey)
    order = np.argsort(x)
    return x[order], y[order]


class DistributionFunction:
    """
    A distribution function -- callable (evaluate) and plottable -- returned by a distribution's ``pdf`` / ``cdf`` /
    ``quantile`` property.

    Calling it evaluates the function (e.g. ``coal.sfs.pdf(t)`` returns the per-bin densities at ``t``), while
    :meth:`plot` draws it (e.g. ``coal.sfs.pdf.plot()`` overlays every bin's density curve).

    This base class is rarely used directly: each property returns one of the thin typed subclasses
    (:class:`PDF` / :class:`CDF` / :class:`QuantileFunction`, in plain,
    ``Marginal...``, ``Joint...`` and ``Conditional...`` flavours). They behave identically but carry distinct
    docstrings describing *what* the function is and *how* it is computed, and -- being real classes with real
    :meth:`plot` / :meth:`__call__` methods -- they let IDEs resolve ``.plot`` to a definition and surface those
    docstrings (unlike a dynamically bound attribute). Supersedes the former ``plot_pdf`` / ``plot_cdf`` methods (now
    deprecated aliases).

    A function object holds its owning ``distribution`` and dispatches by :attr:`kind` to the distribution's
    ``_<kind>`` (evaluate) and ``_plot_<kind>`` (plot) -- so it is fully defined by the distribution, not patched
    together from loose callbacks.

    :param distribution: The distribution this function belongs to.
    """
    #: Short kind label (``'pdf'`` / ``'cdf'`` / ``'quantile'``), set by the kind subclasses; selects the
    #: distribution's ``_<kind>`` / ``_plot_<kind>`` methods and is used in ``repr``.
    kind: str = ''

    def __init__(self, distribution: 'CallableDistributionFunctions'):
        self._distribution = distribution

    def __call__(self, *args, **kwargs):
        """Evaluate the distribution function at the given point(s) (the distribution's ``_<kind>``)."""
        return getattr(self._distribution, '_' + self.kind)(*args, **kwargs)

    def plot(self, *args, **kwargs) -> 'plt.Axes':
        """
        Plot the distribution function (the distribution's ``_plot_<kind>``). Accepted arguments depend on the
        distribution; common ones are ``exact`` (use the slower per-point de Hoog inversion instead of the fast COS
        curve), ``bins`` / ``configs`` (select which spectrum bins to draw), ``n_points`` (grid resolution),
        ``ax`` / ``show`` / ``file`` / ``title``.
        """
        return getattr(self._distribution, '_plot_' + self.kind)(*args, **kwargs)

    def __repr__(self):
        return f"<{type(self).__name__}: call to evaluate, .plot() to draw>"


class _SurfacePlottable:
    """Mixin adding :meth:`plot_surface` for bivariate (joint) distribution functions (a 3D surface in addition to the
    2D heatmap drawn by :meth:`plot`). Univariate function classes deliberately lack it."""

    def plot_surface(self, *args, **kwargs) -> 'plt.Axes':
        """Plot the joint distribution function as a 3D surface (the distribution's ``_plot_<kind>_surface``)."""
        return getattr(self._distribution, '_plot_' + self.kind + '_surface')(*args, **kwargs)


# --- function kinds -------------------------------------------------------------------------------------------------

class PDF(DistributionFunction):
    """Probability density function.

    - **Callable** ``pdf(x)``: the exact pointwise density by per-point de Hoog Laplace inversion (the exact
      matrix-exponential density for the tree height; a histogram for empirical samples).
    - **Plot** ``pdf.plot()``: the fast cosine curve (the derivative of the cosine CDF), or the exact per-point
      density with ``exact=True``.
    """
    kind = 'pdf'


class CDF(DistributionFunction):
    """Cumulative distribution function -- the probability of being at most ``x``.

    - **Callable** ``cdf(x)``: the exact value by per-point de Hoog inversion (the exact matrix exponential for the
      tree height; the empirical CDF for samples).
    - **Plot** ``cdf.plot()``: the fast cosine curve, or the exact per-point CDF with ``exact=True``.
    """
    kind = 'cdf'


class QuantileFunction(DistributionFunction):
    """Quantile function -- the inverse CDF.

    - **Callable** ``quantile(q)``: the value at which the CDF reaches ``q``, by bisection on the de Hoog CDF (or the
      sample quantile for empirical data).
    - **Plot** ``quantile.plot()``: inverts the fast cosine CDF curve (or the per-point bisection with ``exact=True``).
    """
    kind = 'quantile'


# --- marginal (per-bin spectrum) flavours ---------------------------------------------------------------------------

class MarginalPDF(PDF):
    """Per-bin marginal densities of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``pdf(x)``: every bin's density at ``x`` by per-point de Hoog inversion.
    - **Plot** ``pdf.plot()``: overlays every bin's fast cosine density curve (or per-point de Hoog with ``exact=True``).
    """


class MarginalCDF(CDF):
    """Per-bin marginal CDFs of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``cdf(x)``: every bin's probability of being at most ``x`` by per-point de Hoog inversion.
    - **Plot** ``cdf.plot()``: overlays every bin's fast cosine CDF curve (or per-point de Hoog with ``exact=True``).
    """


class MarginalQuantileFunction(QuantileFunction):
    """Per-bin marginal quantile functions of a spectrum (one per SFS / jSFS bin).

    - **Callable** ``quantile(q)``: each bin's quantile by bisection on its de Hoog CDF.
    - **Plot** ``quantile.plot()``: overlays every bin's quantile, inverting the fast cosine CDF curve.
    """


# --- joint (bivariate) flavours -------------------------------------------------------------------------------------

class JointPDF(_SurfacePlottable, PDF):
    """Joint density of two rewards / bins (the within-tree pair of branch lengths).

    - **Callable** ``pdf(x, y)``: the continuous part of the joint law -- the accurate nested de Hoog inversion by
      default, or the fast 2D cosine expansion with ``method='cos'``. Accepts scalars or arrays.
    - **Plot** ``pdf.plot()`` / ``pdf.plot_surface()``: heatmap / 3D surface of the fast cosine density, or the
      nested de Hoog with ``method='dehoog'``.
    """


class JointCDF(_SurfacePlottable, CDF):
    """Joint CDF of two rewards / bins -- the probability both are at most their thresholds.

    - **Callable** ``cdf(x, y)``: the axis atoms (where a reward is zero) plus the continuous part -- the accurate
      nested de Hoog box by default (no near-origin bias for skewed multi-epoch rewards), or the fast 2D cosine box
      with ``method='cos'``. Accepts scalars or arrays.
    - **Plot** ``cdf.plot()`` / ``cdf.plot_surface()``: heatmap / 3D surface of the fast cosine box, or the nested de
      Hoog box with ``method='dehoog'``.
    """


# (a bivariate joint has no quantile flavour: a 2D quantile is not well-defined -- use a marginal or conditional)


# --- conditional flavours -------------------------------------------------------------------------------------------

class ConditionalPDF(PDF):
    """Density of one reward conditional on another being held at a value (e.g. one bin's length given another's).

    - **Callable** ``pdf(y)``: the exact pointwise conditional density by per-point de Hoog of the nested-inversion
      conditional transform (an inner inversion along the conditioned axis, then the outer one).
    - **Plot** ``pdf.plot()``: the de Hoog spline density curve; pass ``exact=True`` for the per-point density.
    """


class ConditionalCDF(CDF):
    """CDF of one reward conditional on another being held at a value.

    - **Callable** ``cdf(y)``: the exact pointwise conditional CDF by per-point de Hoog of the nested-inversion
      conditional transform.
    - **Plot** ``cdf.plot()``: the de Hoog spline CDF curve; pass ``exact=True`` for the per-point CDF.
    """


class ConditionalQuantileFunction(QuantileFunction):
    """Quantile function of one reward conditional on another being held at a value.

    - **Callable** ``quantile(q)``: bisection on the conditional de Hoog CDF.
    - **Plot** ``quantile.plot()``: the conditional quantile over a probability grid.
    """


class CallableDistributionFunctions:
    """
    Mixin exposing ``pdf`` / ``cdf`` / ``quantile`` as callable-and-plottable distribution-function properties. Each
    concrete distribution supplies the evaluators ``_pdf`` / ``_cdf`` / ``_quantile`` and the plotters ``_plot_pdf`` /
    ``_plot_cdf`` / ``_plot_quantile``; this mixin wires them together. Subclasses pick the *flavour* of the returned
    function objects by overriding :attr:`_pdf_function` / :attr:`_cdf_function` / :attr:`_quantile_function` (e.g. a
    spectrum returns the ``Marginal...`` flavours, a conditional the ``Conditional...`` flavours), which is what gives
    IDEs the right type and docstring. Also keeps the former ``plot_pdf`` / ``plot_cdf`` methods as deprecated aliases.
    """
    #: The distribution-function classes returned by the properties; overridden by subclasses to select the flavour.
    #: ``_quantile_function = None`` marks a distribution without a quantile (e.g. a bivariate joint).
    _pdf_function = PDF
    _cdf_function = CDF
    _quantile_function = QuantileFunction

    @property
    def cdf(self) -> CDF:
        """Cumulative distribution function: callable (``cdf(t)``) and plottable (``cdf.plot()`` / -- joint --
        ``cdf.plot_surface()``)."""
        return self._cdf_function(self)

    @property
    def pdf(self) -> PDF:
        """Probability density function: callable (``pdf(t)``) and plottable (``pdf.plot()`` / -- joint --
        ``pdf.plot_surface()``)."""
        return self._pdf_function(self)

    @property
    def quantile(self) -> QuantileFunction:
        """Quantile function: callable (``quantile(q)``) and plottable (``quantile.plot()``)."""
        if self._quantile_function is None:
            raise NotImplementedError(f"{type(self).__name__} has no quantile function "
                                      "(a bivariate joint quantile is not well-defined; use a marginal/conditional).")
        return self._quantile_function(self)

    def plot_cdf(self, *args, **kwargs):
        """Deprecated: use :attr:`cdf`.plot() instead."""
        warnings.warn("plot_cdf() is deprecated; use .cdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self._plot_cdf(*args, **kwargs)

    def plot_pdf(self, *args, **kwargs):
        """Deprecated: use :attr:`pdf`.plot() instead."""
        warnings.warn("plot_pdf() is deprecated; use .pdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self._plot_pdf(*args, **kwargs)

    def _warn_if_negative(self, values: np.ndarray, label: str, rtol: float = 1e-3) -> np.ndarray:
        """Warn (via this distribution's logger) if ``values`` has a substantial negative entry relative to its scale,
        then return it unchanged (the caller clips). A density / probability must be non-negative, so a real negative
        -- beyond the ``rtol`` numerical-noise band -- signals inversion ringing (Gibbs) worth surfacing rather than
        silently clipping. Gated by :attr:`Settings.check_inversions`."""
        arr = np.asarray(values, dtype=float)
        if Settings.check_inversions and arr.size:
            scale = max(float(np.abs(arr).max()), 1e-300)
            mn = float(np.nanmin(arr))
            if mn < -rtol * scale:
                self._logger.warning(f"{label}: substantial negative value ({mn:.2e} vs scale {scale:.2e}); clipping "
                                     f"to 0 -- the numerical inversion may be imprecise here")
        return values

    def _warn_if_nonmonotone(self, cdf: np.ndarray, label: str, rtol: float = 1e-3) -> np.ndarray:
        """Warn (via this distribution's logger) if ``cdf`` has a substantial downward step relative to its range, then
        return it unchanged (the caller enforces monotonicity). A CDF must be non-decreasing, so a real drop -- beyond
        the ``rtol`` numerical-noise band -- signals inversion ringing (a wiggle). Gated by
        :attr:`Settings.check_inversions`."""
        arr = np.asarray(cdf, dtype=float)
        if Settings.check_inversions and arr.size > 1:
            rng = max(float(np.nanmax(arr) - np.nanmin(arr)), 1e-300)
            drop = -float(np.nanmin(np.diff(arr)))
            if drop > rtol * rng:
                self._logger.warning(f"{label}: non-monotone CDF (downward step {drop:.2e} vs range {rng:.2e}); "
                                     f"enforcing monotonicity -- the numerical inversion may be imprecise here")
        return cdf


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    def __init__(self):
        """
        Create object.
        """
        #: Logger
        self._logger = logger.getChild(self.__class__.__name__)

    def touch(self, **kwargs: dict):
        """
        Touch all cached properties.

        :param kwargs: Additional keyword arguments.
        """
        for cls in self.__class__.__mro__:
            for attr, value in cls.__dict__.items():
                if isinstance(value, cached_property):
                    getattr(self, attr)


class MomentAwareDistribution(ProbabilityDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments can be calculated.
    """

    @abstractmethod
    @cached_property
    def mean(self) -> float:
        """
        First moment / mean.
        """
        pass

    @abstractmethod
    @cached_property
    def var(self) -> float:
        """
        Second central moment / variance.
        """
        pass

    @abstractmethod
    @cached_property
    def m2(self) -> float:
        """
        Second (non-central) moment.
        """
        pass


class MarginalDistributions(Mapping, ABC):
    """
    Base class for marginal distributions.
    """

    @abstractmethod
    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix.
        """
        pass

    @abstractmethod
    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix.
        """
        pass

    @abstractmethod
    def get_cov(self, d1, d2) -> float:
        """
        Get the covariance between two marginal distributions.

        :param d1: The index of the first marginal distribution.
        :param d2: The index of the second marginal distribution.
        :return: covariance
        """
        pass

    @abstractmethod
    def get_corr(self, d1, d2) -> float:
        """
        Get the correlation coefficient between two marginal distributions.

        :param d1: The index of the first marginal distribution.
        :param d2: The index of the second marginal distribution.
        :return: correlation coefficient
        """
        pass


class MarginalLocusDistributions(MarginalDistributions):
    """
    Marginal locus distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution'):
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item):
        """
        Get the distribution for the given locus.

        :param item: Deme name.
        :return: Distribution.
        """
        return self.loci[item]

    def __iter__(self) -> Iterator:
        """
        Iterate over distributions.

        :return: Iterator.
        """
        return iter(self.loci)

    def __len__(self) -> int:
        """
        Get the number of distributions.

        :return: Number of distributions.
        """
        return len(self.loci)

    @cached_property
    def loci(self) -> 'MarginalLocusDistributions':
        """
        Distributions marginalized over loci.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        from .phase_type import PhaseTypeDistribution, TreeHeightDistribution
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        loci = {}
        for locus in range(self.dist.locus_config.n):
            loci[locus] = cls(
                state_space=self.dist.state_space,
                tree_height=self.dist.tree_height,
                demography=self.dist.demography,
                reward=CombinedReward([self.dist.reward, LocusReward(locus)])
            )

        return loci

    def get_cov(self, locus1: int, locus2: int) -> float:
        """
        Get the covariance between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The covariance.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        return self.dist.moment(
            k=2,
            rewards=(
                CombinedReward([self.dist.reward, LocusReward(locus1)]),
                CombinedReward([self.dist.reward, LocusReward(locus2)])
            ),
            center=True
        )

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_cov(i, j) for i in range(n_loci)] for j in range(n_loci)])

    def get_corr(self, locus1: int, locus2: int) -> float:
        """
        Get the correlation coefficient between two loci.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The correlation coefficient.
        """
        locus1 = int(locus1)
        locus2 = int(locus2)

        return self.get_cov(locus1, locus2) / (self.loci[locus1].std * self.loci[locus2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across loci.
        """
        n_loci = self.dist.locus_config.n

        return np.array([[self.get_corr(i, j) for i in range(n_loci)] for j in range(n_loci)])

    def joint_distribution(self, locus1: int, locus2: int) -> 'JointRewardDistribution':
        """
        Joint distribution of the accumulated reward at ``locus1`` and at ``locus2`` — e.g. the per-locus tree height
        or total branch length at two loci separated by recombination. This is the distributional object behind the
        cross-locus covariance :meth:`get_cov`/:meth:`get_corr`: it is built from the host distribution's own reward
        restricted to each locus (``CombinedReward([reward, LocusReward(locus)])``, exactly the cross-moment reward
        pair used there), so its marginals are the per-locus distributions :attr:`loci` and it shares the joint LST /
        2D inversion machinery with the within-tree SFS pairwise joint.

        :param locus1: The first locus.
        :param locus2: The second locus.
        :return: The joint accumulated-reward distribution across the two loci.
        """
        locus1, locus2 = int(locus1), int(locus2)

        if locus1 not in range(self.dist.locus_config.n) or locus2 not in range(self.dist.locus_config.n):
            raise ValueError(f"Locus {locus1} or {locus2} does not exist.")

        return self.dist.joint_distribution(
            CombinedReward([self.dist.reward, LocusReward(locus1)]),
            CombinedReward([self.dist.reward, LocusReward(locus2)])
        )


class MarginalDemeDistributions(MarginalDistributions):
    """
    Marginal deme distributions.
    """

    def __init__(self, dist: 'PhaseTypeDistribution'):
        """
        Initialize the distributions.

        :param dist: The distribution.
        """
        self.dist = dist

    def __getitem__(self, item):
        """
        Get the distribution for the given deme.

        :param item: Deme name.
        :return: Distribution.
        """
        return self.demes[item]

    def __iter__(self) -> Iterator:
        """
        Iterate over distributions.

        :return: Iterator.
        """
        return iter(self.demes)

    def __len__(self) -> int:
        """
        Get the number of distributions.

        :return: Number of distributions.
        """
        return len(self.demes)

    @cached_property
    def demes(self) -> 'MarginalDemeDistributions':
        """
        Distributions marginalized over demes.
        """
        # get class of distribution but use PhaseTypeDistribution
        # if this is a TreeHeightDistribution as TreeHeightDistribution
        # only works with default rewards
        from .phase_type import PhaseTypeDistribution, TreeHeightDistribution
        cls = self.dist.__class__ if not isinstance(self.dist, TreeHeightDistribution) else PhaseTypeDistribution

        demes = {}
        for pop in self.dist.lineage_config.pop_names:
            demes[pop] = cls(
                state_space=self.dist.state_space,
                tree_height=self.dist.tree_height,
                demography=self.dist.demography,
                reward=CombinedReward([self.dist.reward, DemeReward(pop)])
            )

        return demes

    def get_cov(self, pop1: str, pop2: str) -> float:
        """
        Get the covariance between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The covariance.
        """
        if pop1 not in self.dist.lineage_config.pop_names or pop2 not in self.dist.lineage_config.pop_names:
            raise ValueError(f"Population {pop1} or {pop2} does not exist.")

        return self.dist.moment(
            k=2,
            rewards=(
                CombinedReward([self.dist.reward, DemeReward(pop1)]),
                CombinedReward([self.dist.reward, DemeReward(pop2)])
            ),
            center=True
        )

    @cached_property
    def cov(self) -> np.ndarray:
        """
        Covariance matrix across demes.
        """
        pops = self.dist.lineage_config.pop_names

        return np.array([[self.get_cov(p1, p2) for p1 in pops] for p2 in pops])

    def get_corr(self, pop1: str, pop2: str) -> float:
        """
        Get the correlation coefficient between two demes.

        :param pop1: The first deme.
        :param pop2: The second deme.
        :return: The correlation coefficient.
        """
        return self.get_cov(pop1, pop2) / (self.demes[pop1].std * self.demes[pop2].std)

    @cached_property
    def corr(self) -> np.ndarray:
        """
        Correlation matrix across demes.
        """
        pops = self.dist.lineage_config.pop_names

        return np.array([[self.get_corr(p1, p2) for p1 in pops] for p2 in pops])


class DensityAwareDistribution(CallableDistributionFunctions, MomentAwareDistribution, ABC):
    """
    Abstract base class for probability distributions for which moments and densities can be calculated. The
    ``cdf`` / ``pdf`` / ``quantile`` are exposed as callable-and-plottable :class:`DistributionFunction`s (see
    :class:`CallableDistributionFunctions`); subclasses implement the ``_cdf`` / ``_pdf`` / ``_quantile`` evaluators.
    """

    @abstractmethod
    def _cdf(self, t: float | Sequence[float]) -> float | np.ndarray:
        """
        Cumulative distribution function.

        :param t: Value or values to evaluate the CDF at.
        :return: CDF.
        """
        pass

    @abstractmethod
    def _quantile(self, q: float) -> float:
        """
        Get the qth quantile.
        """
        pass

    @abstractmethod
    def _pdf(self, t: float | Sequence[float], **kwargs: dict) -> float | np.ndarray:
        """
        Density function.

        :param t: Value or values to evaluate the density function at.
        :param kwargs: Additional keyword arguments.
        :return: Density.
        """
        pass

    def _plot_quantile(
            self,
            ax: 'plt.Axes' = None,
            q: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Quantile function'
    ) -> 'plt.Axes':
        """
        Plot the quantile function (value versus probability ``q``).

        :param ax: Axes to plot on.
        :param q: Probabilities to evaluate the quantile at. By default, 99 evenly spaced values in ``(0, 1)``.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if q is None:
            q = np.linspace(1.0 - Settings.plot_endpoint_quantile, Settings.plot_endpoint_quantile, Settings.plot_n_grid)

        return Visualization.plot(
            ax=ax,
            x=q,
            y=np.array([self._quantile(float(p)) for p in q]),
            xlabel='q',
            ylabel='quantile',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

    def _plot_cdf(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height CDF'
    ) -> 'plt.Axes':
        """
        Plot cumulative distribution function.

        :param ax: Axes to plot on.
        :param t: Values to evaluate the CDF at. By default, 200 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :return: Axes.
        """
        from ..visualization import Visualization

        if t is None:
            t = np.linspace(0, self._quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

        ax = Visualization.plot(
            ax=ax,
            x=t,
            y=self._cdf(t),
            xlabel='t',
            ylabel='F(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )
        ax.set_ylim(0.0, 1.02)  # a CDF spans [0, 1]
        return ax

    def _plot_pdf(
            self,
            ax: 'plt.Axes' = None,
            t: np.ndarray = None,
            show: bool = True,
            file: str = None,
            clear: bool = True,
            label: str = None,
            title: str = 'Tree height PDF',
            dx: float = None
    ) -> 'plt.Axes':
        """
        Plot density function.

        :param ax: The axes to plot on.
        :param t: Values to evaluate the density function at.
            By default, 200 evenly spaced values between 0 and the 99th percentile.
        :param show: Whether to show the plot.
        :param file: File to save the plot to.
        :param clear: Whether to clear the plot before plotting.
        :param label: Label for the plot.
        :param title: Title of the plot.
        :param dx: Step size for numerical differentiation. By default, the 99th percentile divided by 1e10.
        :return: Axes.
        """
        from ..visualization import Visualization

        if dx is None:
            dx = self._quantile(Settings.plot_endpoint_quantile) / 1e10

        if t is None:
            t = np.linspace(0, self._quantile(Settings.plot_endpoint_quantile), Settings.plot_n_grid)

        return Visualization.plot(
            ax=ax,
            x=t,
            y=self._pdf(t, dx=dx),
            xlabel='t',
            ylabel='f(t)',
            label=label,
            file=file,
            show=show,
            clear=clear,
            title=title
        )

