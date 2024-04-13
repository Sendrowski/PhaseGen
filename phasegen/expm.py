"""
Matrix exponential computation.
"""
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import scipy


class ExpmBackend(ABC):
    """
    Base class for matrix exponentiation.

    :meta private:
    """

    @abstractmethod
    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential.
        """
        pass


class TensorFlowExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using TensorFlow. Tends to be faster than scipy.
    Note that tensorflow is an optional dependency and thus needs to be installed separately.

    .. note::
        Recommended backend for fast and reliable matrix exponentiation.
    """

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using TensorFlow.

        :param m: Matrix.
        :return: Matrix exponential
        """
        import tensorflow as tf

        return tf.linalg.expm(tf.convert_to_tensor(m, dtype=tf.float64)).numpy()


class SciPyExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using SciPy.

    .. note::
        This is the default backend. Use :class:`TensorFlowExpmBackend` if performance is an issue.
    """
    def __init__(self, precision: Literal['np.float32', 'np.float64'] = np.float64):
        """
        Initialize the backend.

        :param precision: Precision of the matrix exponential, defaults to double precision. A lower precision is
            faster but much more prone to numerical issues, so please use with caution.
        """
        #: Precision of the matrix exponential
        self.precision = precision

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using SciPy.

        :param m: Matrix
        :return: Matrix exponential
        """
        return scipy.linalg.expm(m.astype(self.precision))


class JaxExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using Jax. This is faster than the other backends but tends to be less precise, so
    please use with caution. Note that jax is an optional dependency and thus needs to be installed separately.
    """
    def __init__(self, max_squarings: int = 2 ** 10):
        """
        Initialize the backend.

        :param max_squarings: Maximum number of squarings (see jax.scipy.linalg.expm).
        """
        #: Maximum number of squarings
        self.max_squarings = max_squarings

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using Jax.

        :param m: Matrix
        :return: Matrix exponential
        """
        import jax

        return jax.scipy.linalg.expm(m, max_squarings=self.max_squarings)


class Backend(ABC):
    """
    Configure the backend for matrix exponentiation.
    """
    #: Backend for matrix exponentiation
    backend: ExpmBackend = SciPyExpmBackend()

    @classmethod
    @abstractmethod
    def expm(cls, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential.
        """
        return cls.backend.compute(m)

    @classmethod
    def register(cls, backend: ExpmBackend):
        """
        Register a backend.
        """
        cls.backend = backend
