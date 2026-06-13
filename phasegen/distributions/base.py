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

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from .phase_type import PhaseTypeDistribution

expm = Backend.expm
logger = logging.getLogger('phasegen')


class DistributionFunction:
    """
    A distribution function -- a ``pdf``, ``cdf`` or ``quantile`` -- that is both **callable** and **plottable**.

    Calling it evaluates the function exactly as the former method did (e.g. ``coal.sfs.pdf(t)`` returns the per-bin
    densities at ``t``), while :meth:`plot` draws it (e.g. ``coal.sfs.pdf.plot()`` draws every bin's density curve at
    once, and for a bivariate distribution ``coal.sfs2.joint_distribution(i, j).pdf.plot()`` draws the 2D heatmap).

    Returned by the ``pdf`` / ``cdf`` / ``quantile`` properties of the distributions; it supersedes the former
    ``plot_pdf`` / ``plot_cdf`` methods (now deprecated aliases for ``.pdf.plot()`` / ``.cdf.plot()``).
    """

    def __init__(self, evaluate: Callable, plot: Callable, kind: str = ''):
        """
        :param evaluate: The evaluation callback (the former ``cdf`` / ``pdf`` / ``quantile`` method body).
        :param plot: The plotting callback (the former ``plot_cdf`` / ``plot_pdf`` body, or a quantile-function plot).
        :param kind: One of ``'cdf'``, ``'pdf'``, ``'quantile'`` (for ``repr``).
        """
        self._evaluate = evaluate
        self._plot = plot
        #: The kind of function: ``'cdf'``, ``'pdf'`` or ``'quantile'``.
        self.kind = kind

    def __call__(self, *args, **kwargs):
        """Evaluate the distribution function (delegates to the wrapped evaluator)."""
        return self._evaluate(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """Plot the distribution function (delegates to the wrapped plotter)."""
        return self._plot(*args, **kwargs)

    def __repr__(self):
        return f"<{self.kind or 'distribution'} function: call to evaluate, .plot() to draw>"


class CallableDistributionFunctions:
    """
    Mixin exposing ``pdf`` / ``cdf`` / ``quantile`` as callable-and-plottable :class:`DistributionFunction`
    properties. Each concrete distribution supplies the evaluators ``_pdf`` / ``_cdf`` / ``_quantile`` and the
    plotters ``_plot_pdf`` / ``_plot_cdf`` / ``_plot_quantile``; this mixin wires them together and keeps the former
    ``plot_pdf`` / ``plot_cdf`` methods working as (deprecated) aliases.
    """

    @property
    def cdf(self) -> DistributionFunction:
        """Cumulative distribution function: callable (``cdf(t)``) and plottable (``cdf.plot()``)."""
        return DistributionFunction(self._cdf, self._plot_cdf, 'cdf')

    @property
    def pdf(self) -> DistributionFunction:
        """Probability density function: callable (``pdf(t)``) and plottable (``pdf.plot()``)."""
        return DistributionFunction(self._pdf, self._plot_pdf, 'pdf')

    @property
    def quantile(self) -> DistributionFunction:
        """Quantile function: callable (``quantile(q)``) and plottable (``quantile.plot()``)."""
        return DistributionFunction(self._quantile, self._plot_quantile, 'quantile')

    def plot_cdf(self, *args, **kwargs):
        """Deprecated: use :attr:`cdf`.plot() instead."""
        warnings.warn("plot_cdf() is deprecated; use .cdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self._plot_cdf(*args, **kwargs)

    def plot_pdf(self, *args, **kwargs):
        """Deprecated: use :attr:`pdf`.plot() instead."""
        warnings.warn("plot_pdf() is deprecated; use .pdf.plot() instead.", DeprecationWarning, stacklevel=2)
        return self._plot_pdf(*args, **kwargs)


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
            q = np.linspace(0.01, 0.99, 99)

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
            t = np.linspace(0, self._quantile(0.99), 200)

        return Visualization.plot(
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
            dx = self._quantile(0.99) / 1e10

        if t is None:
            t = np.linspace(0, self._quantile(0.99), 200)

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

