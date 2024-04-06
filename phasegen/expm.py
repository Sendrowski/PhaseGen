"""
Matrix exponential computation.
"""
from abc import ABC, abstractmethod

import numpy as np
import scipy


class MatrixExponentiation(ABC):
    """
    Base class for matrix exponentiation.
    """

    @staticmethod
    @abstractmethod
    def compute(m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential.
        """
        pass


class TensorFlowExpm(MatrixExponentiation):
    """
    Compute the matrix exponential using TensorFlow.
    """

    @staticmethod
    def compute(m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using TensorFlow.

        :param m: Matrix.
        :return: Matrix exponential
        """
        import tensorflow as tf

        return tf.linalg.expm(tf.convert_to_tensor(m, dtype=tf.float64)).numpy()


class SciPyExpm(MatrixExponentiation):
    """
    Compute the matrix exponential using SciPy.
    """

    @staticmethod
    def compute(m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using SciPy.

        :param m: Matrix
        :return: Matrix exponential
        """
        return scipy.linalg.expm(m)


class Backend(ABC):
    """
    Base class for backends.
    """
    backend: MatrixExponentiation = SciPyExpm

    @classmethod
    @abstractmethod
    def expm(cls, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential.
        """
        return cls.backend.compute(m)

    @classmethod
    def register(cls, backend: MatrixExponentiation):
        """
        Register a backend.
        """
        cls.backend = backend
