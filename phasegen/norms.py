"""
Norms and likelihoods for comparing two values.
"""

from abc import ABC
from typing import Any, Iterable

import numpy as np
from fastdfe.likelihood import Likelihood as PoissonLikelihoodFastDFE


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
    Class for L-norms.
    """

    def __init__(self, p: int):
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
        return np.linalg.norm(a - b, ord=self.p)


class L2Norm(LNorm):
    """
    Class for L2-norm (Euclidean distance).
    """

    def __init__(self):
        """
        Initialize the class.
        """
        super().__init__(p=2)


class L1Norm(LNorm):
    """
    Class for L1-norm (Manhattan distance).
    """

    def __init__(self):
        """
        Initialize the class.
        """
        super().__init__(p=1)


class LInfNorm(LNorm):
    """
    Class for L-infinity norm (Chebyshev distance).
    """

    def __init__(self):
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
    """

    def compute(self, observed: Iterable, modelled: Iterable) -> float | int:
        """
        Return additive inverse of Poisson log-likelihood assuming independent entries.
        Note that the returned likelihood is a positive value which ought to be minimized.

        :param observed: Observed values.
        :param modelled: Modelled values.
        :return: A numerical value representing the difference between the two values.
        """
        return - PoissonLikelihoodFastDFE.log_poisson(
            mu=np.array(list(modelled)),
            k=np.array(list(observed))
        ).sum()
