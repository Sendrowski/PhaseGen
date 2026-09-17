"""
Norms and likelihood functions for comparing observed and modeled values with
:class:`~phasegen.inference.Inference`.
"""

from abc import ABC
from typing import Any, Iterable

import numpy as np
from ._likelihood import Likelihood as _Likelihood


class Norm(ABC):
    """
    Abstract class for norms.
    """

    def compute(self, a: Any, b: Any) -> float | int:
        """
        Compare two values.

        :param a: A value.
        :param b: Another value.
        :return: A numerical value representing the difference between the two values.
        """
        pass


class LNorm(Norm):
    """
    Class for :math:`L^p`-norms of the element-wise difference,

    .. math::

        \\|\\mathbf{a} - \\mathbf{b}\\|_p = \\left( \\sum_i |a_i - b_i|^p \\right)^{1/p},

    with the inputs flattened first, so a multi-dimensional input (e.g. a joint SFS matrix) yields the element-wise
    vector distance and not an induced matrix norm.
    """

    def __init__(self, p: int) -> None:
        """
        Initialize the class with the provided parameters.

        :param p: The order of the norm. see :func:`numpy.linalg.norm` for details.
        """
        #: The order of the norm.
        self.p: int = np.inf if np.isinf(p) else int(p)

    def compute(self, a: float | np.ndarray, b: float | np.ndarray) -> float | int:
        """
        Compare two values.

        :param a: A value.
        :param b: Another value.
        :return: A numerical value representing the difference between the two values.
        """
        # flatten so a multi-dimensional input (e.g. a joint SFS matrix) yields the element-wise vector distance
        # rather than a matrix norm
        return np.linalg.norm(np.ravel(a - b), ord=self.p)


class L2Norm(LNorm):
    """
    Class for the :math:`L^2`-norm (Euclidean distance),
    :math:`\\|\\mathbf{a} - \\mathbf{b}\\|_2 = \\sqrt{\\sum_i (a_i - b_i)^2}`.
    """

    def __init__(self) -> None:
        """
        Initialize the class.
        """
        super().__init__(p=2)


class L1Norm(LNorm):
    """
    Class for the :math:`L^1`-norm (Manhattan distance),
    :math:`\\|\\mathbf{a} - \\mathbf{b}\\|_1 = \\sum_i |a_i - b_i|`.
    """

    def __init__(self) -> None:
        """
        Initialize the class.
        """
        super().__init__(p=1)


class LInfNorm(LNorm):
    """
    Class for the :math:`L^\\infty`-norm (Chebyshev distance),
    :math:`\\|\\mathbf{a} - \\mathbf{b}\\|_\\infty = \\max_i |a_i - b_i|`.
    """

    def __init__(self) -> None:
        """
        Initialize the class.
        """
        super().__init__(p=np.inf)


class Likelihood(Norm, ABC):
    """
    Abstract class for likelihoods.
    """
    pass


class PoissonLikelihood(Likelihood):
    """
    Class for Poisson likelihoods. Site frequency spectra are often assumed to be
    independent Poisson random variables.

    For observed counts :math:`k_i` and modelled means :math:`\\mu_i`, the additive inverse of the log-likelihood

    .. math::

        L = -\\sum_i \\left( k_i \\log \\mu_i - \\mu_i - \\log k_i! \\right)

    is returned, a positive value to be minimized.
    """

    def compute(self, observed: Iterable | float, modelled: Iterable | float) -> float | int:
        """
        Return additive inverse of Poisson log-likelihood assuming independent entries.
        Note that the returned likelihood is a positive value which ought to be minimized.

        :param observed: Observed value or values.
        :param modelled: Modelled value or values.
        :return: A numerical value representing the difference between the two values.
        """
        # special case: single value
        if not isinstance(observed, Iterable) or not isinstance(modelled, Iterable):
            return self.compute(observed=[observed], modelled=[modelled])

        return - _Likelihood.log_poisson(
            mu=np.array(list(modelled)),
            k=np.array(list(observed))
        ).sum()


class MultinomialLikelihood(Likelihood):
    """
    Class for Multinomial likelihoods. Used when modeling observed counts distributed
    across categories, given expected probabilities.

    The modelled values :math:`m_i` are normalized to form a valid probability distribution,
    :math:`p_i = m_i / \\sum_j m_j`, and the additive inverse of the log-likelihood

    .. math::

        L = -\\sum_i k_i \\log p_i

    is returned, a positive value to be minimized (the multinomial coefficient, constant in the parameters, is
    dropped).
    """

    def compute(self, observed: Iterable, modelled: Iterable) -> float:
        """
        Return the additive inverse of the Multinomial log-likelihood.
        The result is a positive value that should be minimized.

        :param observed: Observed counts per category.
        :param modelled: Modelled values (will be normalized to probabilities).
        :return: Negative log-likelihood as a float.
        """
        observed = np.array(list(observed))
        modelled = np.array(list(modelled))
        modelled = modelled / max(modelled.sum(), np.finfo(float).tiny)

        # floor the probabilities before the log so a category the model assigns zero probability yields a large
        # finite penalty rather than an infinite objective, matching the epsilon convention of the Poisson likelihood
        mask = observed > 0
        return -np.sum(observed[mask] * np.log(np.maximum(modelled[mask], 1e-50)))
