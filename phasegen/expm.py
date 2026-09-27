"""
Matrix exponentiation backends.

.. deprecated::
    The backend registry is deprecated and will be removed. The coalescent statistics issue many small matrix
    exponentials, for which the default SciPy backend is faster than the TensorFlow, Jax and PyTorch ones, whose
    per-call overhead dominates at these sizes.

Two operations are exposed: the dense matrix exponential :math:`\\exp(\\mathbf{A})`
(:meth:`ExpmBackend.compute() <phasegen.expm.ExpmBackend.compute>`) and its action
:math:`\\exp(\\mathbf{A})\\mathbf{v}` on a vector or thin matrix
(:meth:`ExpmBackend.compute_action() <phasegen.expm.ExpmBackend.compute_action>`).

A registered backend reaches the Van Loan evaluation of moments, the tree-height distribution functions and the
mutational configurations. The Laplace transform of an accumulated reward always calls SciPy. The occupation times of
spectra call SciPy above :attr:`Settings.closed_form_sparse_min_states
<phasegen.settings.Settings.closed_form_sparse_min_states>` and :attr:`Settings.expm_action_min_dim
<phasegen.settings.Settings.expm_action_min_dim>`, and the registered backend below them.
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import scipy

logger = logging.getLogger('phasegen')


class ExpmBackend(ABC):
    """
    Base class for matrix exponentiation backends. A custom backend implements :meth:`compute` and is activated with
    :meth:`Backend.register() <phasegen.expm.Backend.register>`.
    """

    @abstractmethod
    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential :math:`\\exp(\\mathbf{A})`.

        :param m: Square matrix.
        :return: The matrix exponential of ``m``.
        """
        pass

    def compute_action(self, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action of the matrix exponential on a vector (or thin matrix),
        :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``).

        The default implementation densifies ``a`` and forms the dense exponential via :meth:`compute`, so the action
        uses the backend's own exponentiation. :class:`SciPyExpmBackend` overrides it with the truncated Taylor
        algorithm of Al-Mohy and Higham (2011) in :func:`scipy.sparse.linalg.expm_multiply`, which exploits the
        sparsity of :math:`\\mathbf{A}` without forming the dense exponential. Other backends may override it likewise.

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :return: ``exp(a) @ b``.
        """
        a_dense = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)

        return self.compute(a_dense) @ b


class TensorFlowExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using TensorFlow, an optional dependency with the installation, GPU and
    performance notes of :class:`JaxExpmBackend`.
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

    def __init__(self, precision: type | str | np.dtype = np.float64) -> None:
        """
        Initialize the backend.

        :param precision: Floating-point precision of the matrix exponential and its action, as a NumPy floating type
            such as ``np.float32`` or ``np.float64``, or its name such as ``'float32'``. Defaults to double precision.
            A lower precision may be faster but is much more prone to numerical issues.
        :raises TypeError: If ``precision`` is not a NumPy floating-point type.
        """
        try:
            dtype = np.dtype(precision)
        except TypeError:
            dtype = None

        if dtype is None or dtype.kind != 'f':
            raise TypeError(f"Precision must be a NumPy floating-point type such as np.float64, got {precision!r}.")

        #: Precision of the matrix exponential and its action
        self.precision: np.dtype = dtype

    def compute(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential using SciPy.

        :param m: Matrix
        :return: Matrix exponential
        """
        return scipy.linalg.expm(m.astype(self.precision))

    def compute_action(self, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``) with the truncated Taylor algorithm
        of Al-Mohy and Higham (2011) in :func:`scipy.sparse.linalg.expm_multiply`, which exploits the sparsity of
        :math:`\\mathbf{A}` without forming the dense exponential.

        :param a: Matrix (typically a sparse matrix).
        :param b: Vector or thin matrix.
        :return: ``exp(a) @ b``.
        """
        from scipy.sparse.linalg import expm_multiply

        return expm_multiply(a.astype(self.precision), np.asarray(b, dtype=self.precision))


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
        Compute the matrix exponential :math:`\\exp(\\mathbf{A})`, as a writable NumPy array whatever array type the
        backend returns.
        """
        return cls._writable(cls.backend.compute(m))

    @classmethod
    def expm_multiply(cls, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action of the matrix exponential, :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``), via the
        active backend without forming the dense exponential, as a writable NumPy array.
        """
        return cls._writable(cls.backend.compute_action(a, b))

    @staticmethod
    def _writable(x) -> np.ndarray:
        """
        The array as a writable NumPy array, copied only if it is not one.

        :param x: Array of any array type.
        :return: Writable NumPy array.
        """
        return x if isinstance(x, np.ndarray) and x.flags.writeable else np.array(x)

    @classmethod
    def register(cls, backend: ExpmBackend) -> None:
        """
        Register a backend.

        .. deprecated::
            The backend registry is deprecated and will be removed; see :mod:`phasegen.expm`.
        """
        logger.warning(
            "Backend.register is deprecated and will be removed; phasegen will call SciPy directly."
        )
        cls.backend = backend
