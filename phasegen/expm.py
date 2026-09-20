"""
Matrix exponentiation backends.

.. deprecated::
    The backend registry is deprecated and will be removed. The coalescent statistics issue many small matrix
    exponentials, for which SciPy is the fastest option, and no other backend is used. Call sites will move to SciPy
    directly.

Two operations are exposed: the dense matrix exponential :math:`\\exp(\\mathbf{A})`
(:meth:`ExpmBackend.compute() <phasegen.expm.ExpmBackend.compute>`) and its action
:math:`\\exp(\\mathbf{A})\\mathbf{v}` on a vector or thin matrix
(:meth:`ExpmBackend.compute_action() <phasegen.expm.ExpmBackend.compute_action>`).

A registered backend reaches the Van Loan evaluation of moments, the tree-height distribution functions and the
mutational configurations. The Laplace transform of an accumulated reward always calls SciPy; the occupation times of
spectra call SciPy above :attr:`Settings.closed_form_sparse_min_states
<phasegen.settings.Settings.closed_form_sparse_min_states>` and :attr:`Settings.expm_action_min_dim
<phasegen.settings.Settings.expm_action_min_dim>` and the registered backend below them.
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


class SciPyExpmBackend(ExpmBackend):
    """
    Compute the matrix exponential using SciPy.

    .. note::
        This is the default backend, and the one every call site uses.
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
        Compute the matrix exponential :math:`\\exp(\\mathbf{A})`.
        """
        return cls.backend.compute(m)

    @classmethod
    def expm_multiply(cls, a, b: np.ndarray) -> np.ndarray:
        """
        Compute the action of the matrix exponential, :math:`\\exp(\\mathbf{A})\\mathbf{v}` (``exp(a) @ b``), via the
        active backend without forming the dense exponential.
        """
        return cls.backend.compute_action(a, b)

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
