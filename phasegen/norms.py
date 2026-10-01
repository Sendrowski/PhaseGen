"""
Norms and likelihood functions for comparing observed and modeled values with
:class:`~phasegen.inference.Inference`.
"""

from abc import ABC
from typing import Any, Iterable

import numpy as np
from ._likelihood import Likelihood as _Likelihood
from .errors import ModelError

#: Magnitude, relative to the largest modelled magnitude, up to which a negative modelled value is taken as round-off
#: and clamped to zero
_NEGATIVE_RTOL = 1e-10


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

    @staticmethod
    def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
        """
        Check that two compared arrays have the same shape.

        :param a: An array.
        :param b: Another array.
        :raises ValueError: If the shapes differ.
        """
        if np.shape(a) != np.shape(b):
            raise ValueError(f'Compared values must have the same shape, got {np.shape(a)} and {np.shape(b)}.')


class LNorm(Norm):
    """
    Class for :math:`L^p`-norms of the element-wise difference,

    .. math::

        \\|\\mathbf{a} - \\mathbf{b}\\|_p = \\left( \\sum_i |a_i - b_i|^p \\right)^{1/p},

    with the inputs flattened first, so a multi-dimensional input (e.g. a joint SFS matrix) yields the element-wise
    vector distance and not an induced matrix norm.
    """

    def __init__(self, p: float) -> None:
        """
        Initialize the class with the provided parameters.

        :param p: The order of the norm, any real number or :math:`\\pm\\infty`, as for vectors in
            :func:`numpy.linalg.norm`.
        """
        #: The order of the norm.
        self.p: float = float(p)

    def compute(self, a: float | np.ndarray, b: float | np.ndarray) -> float | int:
        """
        Compare two values.

        :param a: A value.
        :param b: Another value.
        :return: A numerical value representing the difference between the two values.
        :raises ValueError: If the two values differ in shape.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        self._check_shapes(a, b)

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

    @staticmethod
    def _check_signs(observed: np.ndarray, modelled: np.ndarray) -> np.ndarray:
        """
        Check that the observed counts are non-negative and the modelled values are non-negative up to round-off.

        :param observed: Observed counts.
        :param modelled: Modelled values.
        :return: The modelled values, those negative within round-off of zero set to zero.
        :raises ValueError: If an observed count is negative.
        :raises ModelError: If a modelled value is negative beyond round-off, relative to the largest modelled
            magnitude.
        """
        if np.any(observed < 0):
            raise ValueError(f'Observed counts must be non-negative, got minimum {np.min(observed)}.')

        if np.any(modelled < 0):
            if np.min(modelled) < -_NEGATIVE_RTOL * np.max(np.abs(modelled)):
                raise ModelError(f'Modelled values must be non-negative, got minimum {np.min(modelled)}.')

            return np.maximum(modelled, 0)

        return modelled


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
        :raises ValueError: If the observed and modelled values differ in shape, or an observed count is negative.
        :raises ModelError: If a modelled value is negative beyond round-off.
        """
        # special case: single value
        if not isinstance(observed, Iterable) or not isinstance(modelled, Iterable):
            return self.compute(observed=[observed], modelled=[modelled])

        observed = np.array(list(observed))
        modelled = np.array(list(modelled))
        self._check_shapes(observed, modelled)
        modelled = self._check_signs(observed, modelled)

        return - _Likelihood.log_poisson(mu=modelled, k=observed).sum()


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
        :raises ValueError: If the observed and modelled values differ in shape, or an observed count is negative.
        :raises ModelError: If a modelled value is negative beyond round-off.
        """
        observed = np.array(list(observed))
        modelled = np.array(list(modelled))
        self._check_shapes(observed, modelled)
        modelled = self._check_signs(observed, modelled)

        modelled = modelled / max(modelled.sum(), np.finfo(float).tiny)

        # floor the probabilities before the log so a category the model assigns zero probability yields a large
        # finite penalty rather than an infinite objective, matching the epsilon convention of the Poisson likelihood
        mask = observed > 0
        return -np.sum(observed[mask] * np.log(np.maximum(modelled[mask], 1e-50)))
