"""
Matrix exponentiation backends.
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

    def compute_action(self, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action of the matrix exponential on a vector (or thin matrix), ``exp(a) @ b``.

        The default implementation densifies ``a`` and forms the dense exponential via :meth:`compute`, so the action
        uses the backend's own exponentiation. :class:`SciPyExpmBackend` overrides this with scipy's sparse
        Krylov/Taylor ``expm_multiply``, which exploits the sparsity of ``a`` without forming the dense exponential;
        other backends may likewise override it (e.g. with a GPU Krylov method).

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :return: ``exp(a) @ b``.
        """
        a_dense = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)

        return self.compute(a_dense) @ b


class TensorFlowExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using TensorFlow. Tends to be faster than scipy.
    Note that tensorflow is an optional dependency and thus needs to be installed separately.
    GPU acceleration may be available depending on the underlying hardware.
    Tends to be faster than :class:`SciPyExpmBackend` for large matrices and highly parallelized computations.
    """

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using TensorFlow.

        :param m: Matrix.
        :return: Matrix exponential
        """
        # noinspection PyUnresolvedReferences
        import tensorflow as tf

        return tf.linalg.expm(tf.convert_to_tensor(m, dtype=tf.float64)).numpy()


class SciPyExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using SciPy.

    .. note::
        This is the default backend. Recommended for smaller matrices. Consider switching to other backends for larger
        matrices, such as :class:`JaxExpmBackend`, which is both efficient and lightweight to install.
    """

    def __init__(self, precision: Literal['np.float32', 'np.float64'] = np.float64) -> None:
        """
        Initialize the backend.

        :param precision: Precision of the matrix exponential, defaults to double precision. A lower precision may be
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

    def compute_action(self, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action ``exp(a) @ b`` using scipy's sparse Krylov/Taylor ``expm_multiply``, which exploits the
        sparsity of ``a`` without forming the dense exponential.

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :return: ``exp(a) @ b``.
        """
        from scipy.sparse.linalg import expm_multiply

        return expm_multiply(a, b)


class JaxExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using Jax.
    Note that jax is an optional dependency and thus needs to be installed separately.
    GPU acceleration may be available depending on the underlying hardware.
    Tends to be faster than :class:`SciPyExpmBackend` for larger matrices and highly parallelized computations.
    """

    def __init__(self, max_squarings: int = 2 ** 10) -> None:
        """
        Initialize the backend.

        :param max_squarings: Maximum number of squarings (see jax.scipy.linalg.expm).
        """
        import jax

        # enable double precision
        jax.config.update("jax_enable_x64", True)

        #: Maximum number of squarings
        self.max_squarings = max_squarings

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using Jax.

        :param m: Matrix
        :return: Matrix exponential
        """
        import jax

        # casting explicitly to np.float64 to avoid problems with object type
        return jax.scipy.linalg.expm(m.astype(np.float64), max_squarings=self.max_squarings)


class PyTorchExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using PyTorch.
    Note that PyTorch is an optional dependency and thus needs to be installed separately.
    GPU acceleration may be available depending on the underlying hardware.
    """

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using PyTorch.

        :param m: Matrix
        :return: Matrix exponential
        """
        # noinspection PyUnresolvedReferences
        import torch

        # casting explicitly to np.float64 to avoid problems with object type
        return torch.matrix_exp(torch.tensor(m.astype(np.float64), dtype=torch.float64)).numpy()


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
    def expm_multiply(cls, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action of the matrix exponential, ``exp(a) @ b``, via the active backend without forming the
        dense exponential.
        """
        return cls.backend.compute_action(a, b)

    @classmethod
    def register(cls, backend: ExpmBackend) -> None:
        """
        Register a backend.
        """
        cls.backend = backend
